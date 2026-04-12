# Лабораторная работа 1 — Предобработка данных

## Постановка задачи

Предсказание удовлетворённости клиента (бинарная классификация) на основе данных
бразильского маркетплейса Olist: время доставки, категория товара, тональность отзыва.

**Целевая переменная**: `is_satisfied` (1 = оценка ≥ 4, 0 = оценка < 4)

**Метрики оценки модели**: F1-score (основная), ROC-AUC, Precision, Recall.
Классы несбалансированы (~57% довольных), поэтому accuracy не показательна.

## Архитектура

```
Kaggle CSV
    │
    ▼
data/raw/  ──► MinIO (raw/)
                   │
          ┌────────┴────────┐
          ▼                 ▼
   transactions_preprocess   reviews_preprocess
   (транзакции,               (отзывы + геоданные,
    товары, клиенты)          pandas + sklearn)
          │                 │
          └────────┬────────┘
                   ▼
            join_features
                   │
                   ▼
         MinIO (processed/final_dataset.parquet)
                   │
          AirFlow (ежедневный DAG)
          + инкрементальная загрузка
```

## Выбор хранилища — MinIO (локальный S3)

**Обоснование**:
- S3-совместимый API — промышленный стандарт для data lake
- Легко разворачивается в Docker без облачных расходов
- Доступ через `boto3` / `s3fs` из любого Python-скрипта
- Слоистая структура хранения: `raw/` → `staging/` → `processed/`
- Бесшовная замена на облачный S3/Yandex Object Storage в будущем

## Быстрый старт

### 1. Требования
- Docker Desktop ≥ 20.x
- **Kaggle CLI** в виртуальном окружении: `source ../.venv/bin/activate` (из `lab1/`) и `pip install kaggle`, либо один раз `../.venv/bin/pip install kaggle`.
- Токен API: файл `kaggle.json` в **`~/.kaggle/`** или в **`../.kaggle/`** от корня папки курса — во втором случае перед командами kaggle задайте `export KAGGLE_CONFIG_DIR="$(cd .. && pwd)/.kaggle"` из `lab1/` (путь подправьте под своё расположение репозитория).

### 2. Запуск инфраструктуры

```bash
cd lab1/
docker compose up -d
```

Дождитесь, пока все контейнеры запустятся (~2 минуты при первом запуске).

### 3. Скачивание датасета

Команда `kaggle` появляется в `PATH` только после **`source` на `.venv`** (или вызывайте полный путь к интерпретатору, см. ниже).

Из корня папки **ИССИИ** (где лежат `.venv` и при необходимости `.kaggle`):

```bash
source .venv/bin/activate
export KAGGLE_CONFIG_DIR="$PWD/.kaggle"   # опционально, если токен не в ~/.kaggle
cd lab1/data/raw
kaggle datasets download -d olistbr/brazilian-ecommerce --unzip
```

Без активации venv:

```bash
"/путь/к/ИССИИ/.venv/bin/kaggle" datasets download -d olistbr/brazilian-ecommerce --unzip
```

### 4. Загрузка данных в MinIO

Рекомендуется активировать виртуальное окружение курса (или создать `lab1/.venv`).

```bash
source ../.venv/bin/activate   # пример: venv в корне папки ИССИИ
pip install -r requirements.txt
python scripts/upload_raw_data.py
```

Переменные по умолчанию указывают на MinIO на `localhost:9000` (после `docker compose up`).

### 5. Запуск пайплайна

Открыть AirFlow UI: **http://localhost:8081** (внешний порт 8081, внутри контейнера по-прежнему 8080 — так проще избежать конфликта с чужим nginx на 8080).  
Логин: `admin` / Пароль: `admin`

Найти DAG `ecommerce_preprocessing_pipeline` → нажать ▶ (Trigger DAG).

### 6. Просмотр данных в MinIO

Открыть MinIO UI: http://localhost:9001  
Логин: `minioadmin` / Пароль: `minioadmin`

## Структура проекта

```
lab1/
├── docker-compose.yml         # AirFlow + MinIO + PostgreSQL
├── Dockerfile                 # AirFlow + зависимости
├── requirements.txt
├── dags/
│   └── ecommerce_pipeline.py  # Основной DAG
├── scripts/
│   ├── minio_utils.py         # Утилиты для работы с MinIO
│   ├── upload_raw_data.py     # Загрузка CSV в MinIO (разово)
│   ├── transactions_preprocess.py  # транзакционные данные
│   ├── reviews_preprocess.py  # Студент 2: отзывы + геоданные (pandas + sklearn)
│   ├── join_features.py       # Объединение фичей
│   └── load_increment.py      # Инкрементальная загрузка
├── notebooks/
│   ├── eda_transactions.ipynb # EDA транзакционной части
│   └── eda_reviews.ipynb      # EDA отзывов и геоданных
└── data/
    ├── raw/                   # Исходные CSV с Kaggle
    └── increment/             # Файлы для инкрементальной загрузки
```

## DAG: ecommerce_preprocessing_pipeline

| Задача | Описание |
|---|---|
| `check_raw_data` | Проверяет наличие CSV в MinIO `raw/` |
| `transactions_preprocess` | Join таблиц, обработка пропусков, feature engineering |
| `reviews_preprocess` | Токенизация отзывов, TF-IDF (sklearn), расстояние покупатель-продавец |
| `validate_staging` | Проверка схемы и заполненности промежуточных файлов |
| `join_features` | Объединение результатов обеих веток предобработки |
| `check_increment` | Ветвление: есть ли новые файлы в `increment/`? |
| `load_increment` | Загрузка и мерж инкрементальных данных |

Расписание: `@daily` (каждый день в полночь).

### Почему `load_increment` = skipped

Это **нормально**. Задача `check_increment` — ветвление (Branch): если в MinIO в префиксе `increment/` **нет** новых `.csv`/`.parquet`, запускается только `pipeline_complete`, а `load_increment` **намеренно пропускается** (Airflow помечает downstream как `skipped`). Чтобы `load_increment` выполнился, положите файл в бакет `data-lake` под ключ `increment/...` (см. раздел ниже).

### Логи в UI: «Could not read served logs…»

После `docker compose up -d` в `docker-compose.yml` заданы `AIRFLOW__WEBSERVER__BASE_URL` и `hostname` у scheduler/webserver — это устраняет типичную ошибку с пустым URL лог-сервера в Docker. Если сообщение осталось, пересоздайте контейнеры: `docker compose down && docker compose up -d`. У **skipped** задач логов может почти не быть — смотрите логи у `check_increment` или у соседних успешных задач.

## Инкрементальная загрузка

### Быстрый демо-инкремент (из части processed-датасета)

После успешного полного прогона пайплайна в MinIO есть `processed/final_dataset.parquet`. Скрипт берёт случайные строки, присваивает им **новые** `order_id` и кладёт parquet в `data/increment/` и в MinIO `increment/`:

```bash
cd lab1/
source ../.venv/bin/activate   # при необходимости
python scripts/build_increment_sample.py -n 400
```

Флаг `--no-upload` — только локальный файл. Затем в Airflow запустите DAG: выполнится **`load_increment`** (не skipped).

### Вручную

Положить новый CSV/parquet в `data/increment/` и загрузить в MinIO:

```bash
cp новые_данные.csv data/increment/
python -c "
from scripts.minio_utils import upload_csv
upload_csv('data/increment/новые_данные.csv', 'increment/новые_данные.csv')
"
```

DAG при следующем запуске обработает файл и дополнит `processed/final_dataset.parquet`.
