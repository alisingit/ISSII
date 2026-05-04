"""
Конфигурация лабораторной работы 2.

Настройки по умолчанию читаются из переменных окружения.
При запуске из lab1 значения MINIO_ENDPOINT уже прописаны в docker-compose.yml.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "olist_satisfaction_v9")

# MinIO
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "data-lake")

# Данные
FEATURES_DIR = os.getenv("FEATURES_DIR", "processed/final_dataset")

# Обучение
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.25  # доля от train_val (80% данных) -> 0.25*0.8 = 0.2 от полного