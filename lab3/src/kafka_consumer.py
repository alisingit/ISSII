"""Kafka consumer: читает запросы из топика, обращается к BentoML REST,
пишет ответы в выходной топик и публикует prometheus-метрики."""

import json
import os
import sys
import time
from pathlib import Path

import requests
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable
from prometheus_client import Counter, Histogram, start_http_server

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import KAFKA_BOOTSTRAP_SERVERS, INPUT_TOPIC, OUTPUT_TOPIC  # noqa: E402

BENTOML_URL = os.getenv("BENTOML_URL", "http://localhost:3000/predict")
METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))
CONSUMER_GROUP_ID = os.getenv("KAFKA_CONSUMER_GROUP", "olist-inference")

REQUESTS_TOTAL = Counter(
    "kafka_consumer_requests_total",
    "Общее число обработанных Kafka-сообщений",
    labelnames=("status",),
)
INFERENCE_LATENCY = Histogram(
    "kafka_consumer_inference_latency_seconds",
    "Время от получения сообщения из Kafka до отправки ответа в BentoML",
)


def _wait_for_brokers(timeout: int = 120, delay: float = 2.0) -> None:
    """Дождаться доступности Kafka."""
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            consumer = KafkaConsumer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                request_timeout_ms=2000,
                api_version_auto_timeout_ms=2000,
            )
            consumer.close()
            return
        except NoBrokersAvailable as exc:
            last_error = exc
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(delay)
    raise RuntimeError(
        f"Kafka {KAFKA_BOOTSTRAP_SERVERS} недоступна за {timeout}s: {last_error}"
    )


def _wait_for_bentoml(timeout: int = 120, delay: float = 2.0) -> None:
    """Дождаться готовности BentoML по эндпоинту /healthz."""
    healthz = BENTOML_URL.rsplit("/", 1)[0] + "/healthz"
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(healthz, timeout=2)
            if response.ok:
                return
            last_error = RuntimeError(f"HTTP {response.status_code}")
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(delay)
    raise RuntimeError(f"BentoML {healthz} недоступен за {timeout}s: {last_error}")


def _build_kafka_clients() -> tuple[KafkaConsumer, KafkaProducer]:
    consumer = KafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
        group_id=CONSUMER_GROUP_ID,
        enable_auto_commit=True,
    )
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    return consumer, producer


def main() -> None:
    start_http_server(METRICS_PORT)
    print(f"Метрики Prometheus: http://0.0.0.0:{METRICS_PORT}/metrics")

    _wait_for_brokers()
    _wait_for_bentoml()

    consumer, producer = _build_kafka_clients()
    print(
        f"Подписан на {INPUT_TOPIC} (group={CONSUMER_GROUP_ID}), "
        f"REST endpoint: {BENTOML_URL}"
    )

    for msg in consumer:
        payload = msg.value or {}
        request_id = payload.get("id", "unknown")
        features = payload.get("features")

        start = time.perf_counter()
        try:
            response = requests.post(BENTOML_URL, json=features, timeout=10)
            response.raise_for_status()
            prediction = response.json()
            producer.send(
                OUTPUT_TOPIC,
                value={"id": request_id, "prediction": prediction},
            )
            REQUESTS_TOTAL.labels(status="ok").inc()
            print(f"Обработан {request_id}: {prediction}")
        except Exception as exc:  # noqa: BLE001
            REQUESTS_TOTAL.labels(status="error").inc()
            print(f"Ошибка обработки {request_id}: {exc}")
        finally:
            INFERENCE_LATENCY.observe(time.perf_counter() - start)


if __name__ == "__main__":
    main()
