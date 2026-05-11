"""CLI-скрипт для отправки одного или нескольких тестовых запросов в Kafka.

Использование:
    python src/kafka_producer.py            # 1 сообщение
    python src/kafka_producer.py 100        # 100 сообщений
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import KAFKA_BOOTSTRAP_SERVERS, INPUT_TOPIC  # noqa: E402

DEFAULT_N_FEATURES = 71


def _create_producer(retries: int = 30, delay: float = 2.0) -> KafkaProducer:
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            return KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
        except NoBrokersAvailable as exc:
            last_error = exc
            time.sleep(delay)
    raise RuntimeError(
        f"Не удалось подключиться к Kafka {KAFKA_BOOTSTRAP_SERVERS}: {last_error}"
    )


def main(n_messages: int = 1, n_features: int = DEFAULT_N_FEATURES) -> None:
    producer = _create_producer()
    try:
        for i in range(n_messages):
            payload = {
                "id": f"test_{i:06d}",
                "features": np.random.rand(1, n_features).astype(np.float32).tolist(),
            }
            future = producer.send(INPUT_TOPIC, value=payload)
            future.get(timeout=10)
        producer.flush()
        print(f"Отправлено {n_messages} сообщений в топик {INPUT_TOPIC}")
    finally:
        producer.close()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(n)
