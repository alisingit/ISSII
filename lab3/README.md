# Лабораторная 3 — Развёртывание ONNX-модели через BentoML и Kafka

## 1. Краткое описание

Реализован сервинг бинарной классификации удовлетворённости клиентов Olist
(`StandardScaler + XGBClassifier`, обученной в ЛР2 и зафиксированной в MLflow).

Полный цикл:

1. Лучшая модель из MLflow экспортируется в **ONNX** (вместе с препроцессингом).
2. Модель сервируется через **BentoML** (REST `/predict` + автоматический `/metrics`).
3. Параллельно работает Kafka-pipeline: producer пишет запросы в `inference_requests`,
   consumer читает их, ходит REST-ом в BentoML и публикует ответы в `inference_responses`.
4. Сервис подключён к **Prometheus + Grafana** (auto-provisioning датасорса и дашборда).
5. Нагрузочное тестирование через **Locust**.

## 2. Анализ форматов экспорта

| Формат | Что включает | Плюсы | Минусы | Применимость |
|--------|--------------|-------|--------|--------------|
| Pickle (sklearn native) | Бинарный Python-объект | Минимум кода, нативно | Привязка к версиям Python/библиотек, не межплатформенно, без оптимизации, риски при загрузке чужих файлов | Прототип, MLflow tracking |
| MLflow PyFunc / sklearn flavor | Артефакт + conda окружение | Удобно с MLflow, без ручной конвертации, MLflow Serving из коробки | Требует Python + sklearn + xgboost в проде, тяжёлый Docker | Альтернатива при ML-инфре на MLflow Serving |
| **ONNX** *(выбран)* | Граф вычислений: препроцессинг + модель в одном файле | Кросс-платформенно, без зависимости от python-обучения, быстрый рантайм (`onnxruntime`), нативная поддержка в BentoML (`bentoml.onnx`) | Нужны конвертеры (`skl2onnx`, `onnxmltools`), не все операторы покрыты | Идеально подходит — лёгкий артефакт + единый рантайм |
| TensorRT | Скомпилированный engine под NVIDIA GPU | Максимальный throughput на GPU | Только NVIDIA GPU, тяжёлый toolkit, не нужен для CPU-инференса бустингов | Не применимо, инференс CPU |
| PMML | XML-описание модели | Кросс-платформенно | Устаревший стандарт, ограничены операторы, плохо для XGBoost | Не применимо |
| Treelite / m2cgen | Скомпилированные деревья | Самый быстрый CPU-инференс | Только деревья, без препроцессинга, неудобно обновлять | Резерв при оптимизации |

**Выбран ONNX**, потому что:
1. Конвертация всего sklearn-pipeline (`StandardScaler` + `XGBClassifier`) в один граф через
   `skl2onnx` + регистрацию `onnxmltools.convert.xgboost.convert_xgboost`.
2. Лёгкий рантайм `onnxruntime` без необходимости тащить в продакшен `xgboost`/`sklearn`.
3. BentoML поддерживает ONNX «из коробки» (`bentoml.onnx.save_model`/`to_runner`).

## 3. Архитектура

```
                              /metrics
                  ┌──────────────────────────────┐
                  │                              │
locust ──HTTP──▶ BentoML Service (ONNX runner) ──┴─▶ Prometheus ──▶ Grafana
                  ▲                                  ▲
                  │ HTTP                             │ /metrics
producer ──▶ Kafka(in) ──▶ Kafka consumer (REST) ────┘
                              │
                              └──▶ Kafka(out)
```

- **BentoML** — компонент сервинга (не самописное решение). Принимает 2D-массив признаков,
  возвращает вероятность положительного класса.
- **Kafka** — источник сообщений, демонстрирует асинхронный сценарий.
- **Kafka consumer** не дублирует инференс — ходит REST-ом в BentoML, выполняя роль интеграции
  с очередью. Это сохраняет «единое место правды» для модели и упрощает обновление.
- **Prometheus + Grafana** — мониторинг p50/p95/p99 latency, RPS, ошибок и собственных метрик
  consumer.

## 4. Структура проекта

```
lab3/
├── README.md
├── requirements.txt              # полный набор для локальной разработки
├── requirements-service.txt      # минимальный набор для BentoML Docker
├── requirements-consumer.txt     # минимальный набор для consumer Docker
├── Dockerfile                    # BentoML сервис
├── Dockerfile.consumer           # Kafka consumer
├── docker-compose.yml
├── bentofile.yaml
├── config.py                     # переменные окружения с дефолтами
├── locustfile.py                 # сценарий нагрузки
├── model.onnx                    # артефакт (генерируется export_model.py)
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       ├── dashboards/lab3.json
│       └── provisioning/
│           ├── dashboards/dashboards.yml
│           └── datasources/prometheus.yml
└── src/
    ├── export_model.py           # MLflow → ONNX (полный pipeline)
    ├── service.py                # BentoML сервис
    ├── kafka_producer.py         # CLI для отправки тестовых запросов
    └── kafka_consumer.py         # consumer + prometheus-метрики
```

## 5. Запуск

### 5.1 Поднять весь стек одним докером

```bash
cd lab3
docker compose up --build
```

После старта будут доступны:

| Сервис | URL |
|--------|-----|
| BentoML REST | http://localhost:3000 |
| BentoML metrics | http://localhost:3000/metrics |
| BentoML health | http://localhost:3000/healthz |
| Kafka (для клиентов с хоста) | localhost:9092 |
| Kafka consumer metrics | http://localhost:8000/metrics |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001 (admin / admin, есть anonymous) |

В Grafana автоматически провижится дашборд **«Lab3 — BentoML & Kafka»** в папке `Lab3`.

### 5.2 (Пере)экспорт ONNX-модели из MLflow

```bash
pip install -r requirements.txt

# .env или переменные окружения:
export MLFLOW_TRACKING_URI=http://localhost:5000
export BEST_RUN_ID=<run_id из MLflow>

python src/export_model.py
```

Скрипт сохранит `model.onnx` в корне `lab3/`. После этого нужно пересобрать образ
BentoML: `docker compose up --build bentoml`.

### 5.3 Проверка REST вручную

```bash
curl -X POST http://localhost:3000/predict \
  -H 'Content-Type: application/json' \
  -d "$(python -c 'import json,numpy; print(json.dumps(numpy.random.rand(1,62).tolist()))')"
```

Ответ — JSON-массив с вероятностями положительного класса.

### 5.4 Проверка Kafka-пайплайна

```bash
# из корня lab3
python src/kafka_producer.py 100  # отправить 100 тестовых сообщений
```

Логи `kafka_consumer` (в выводе `docker compose`) покажут, что сообщения обработаны.
Ответы попадают в топик `inference_responses` (можно прочитать `kafka-console-consumer`).

## 6. Нагрузочное тестирование (Locust)

```bash
mkdir -p reports
locust -f locustfile.py --host http://localhost:3000 \
       --headless -u 50 -r 5 --run-time 1m \
       --csv reports/load_test --html reports/load_test.html
```

Параметры:

- `-u` — одновременно работающих пользователей;
- `-r` — нарастание (users/sec);
- `--run-time` — длительность.

Сценарий имитирует реальный онлайн-трафик: батчи 1/4/8 строк, перерыв 0.1–0.5 c.
Параллельно метрики отдаёт BentoML, что позволяет сверить локально измеренный latency с
тем, что реально отвечает сервис.

### Шаблон отчёта по нагрузочному тестированию

| Сценарий | Users | RPS (avg) | p50 (мс) | p95 (мс) | p99 (мс) | Errors |
|----------|-------|-----------|----------|----------|----------|--------|
| 10 users | 10    |           |          |          |          |        |
| 50 users | 50    |           |          |          |          |        |
| 100 users| 100   |           |          |          |          |        |

Заполняется по результатам Locust + Prometheus.

## 7. Мониторинг

В Prometheus собираются (см. `monitoring/prometheus.yml`):

- BentoML — встроенные метрики:
  - `bentoml_api_server_request_total{endpoint, http_response_code}`
  - `bentoml_api_server_request_duration_seconds_bucket`
  - `bentoml_api_server_request_in_progress`
- Kafka consumer — собственные метрики:
  - `kafka_consumer_requests_total{status="ok|error"}`
  - `kafka_consumer_inference_latency_seconds_bucket`

Grafana-дашборд `lab3.json` показывает RPS, latency квантили (p50/p95/p99) и Kafka-rate.

## 8. Анализ результатов (что описать в финальном отчёте)

1. **Производительность ONNX vs sklearn-pipeline.** Сравнить latency сырого `predict_proba`
   sklearn (например, в jupyter) и ONNX-рантайма для одинаковых входов.
2. **Поведение под нагрузкой.** Графики p50/p95/p99 при 10/50/100 users; точка насыщения,
   где растёт хвост latency и появляются 5xx-ошибки.
3. **Сравнение каналов доступа.** Latency прямого REST-вызова vs end-to-end через Kafka
   (метрики `kafka_consumer_inference_latency_seconds`).
4. **Стоимость экспорта.** Размер ONNX-файла vs MLflow-артефакта, объём итогового Docker-образа.
5. **Выводы.** Достаточен ли BentoML + ONNX для онлайнового сценария, какие узкие места
   обнаружены, что бы делал по-другому в проде (батчинг BentoML, асинхронные REST, gRPC).

## 9. Что соответствует пунктам задания

| Пункт задания | Реализация |
|---------------|------------|
| 1a) Анализ форматов экспорта | Раздел 2 этого README |
| 1b) Экспорт модели | `src/export_model.py` (sklearn pipeline → ONNX через skl2onnx + onnxmltools) |
| 2) Импорт модели в готовый компонент + REST | `src/service.py` (BentoML ONNX runner, `/predict`) |
| 3) Нагрузочное тестирование | `locustfile.py`, инструкции в разделе 6, шаблон отчёта |
| 4) Брокер сообщений как источник данных | `src/kafka_producer.py`, `src/kafka_consumer.py`, Kafka в `docker-compose.yml` |
| 5) Мониторинг | Prometheus + Grafana с auto-provisioning датасорса и дашборда |
| 6) Анализ результатов | Раздел 8 этого README (заполняется после тестов) |
