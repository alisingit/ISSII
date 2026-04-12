"""
Загружает все CSV-файлы из data/raw/ в MinIO в папку raw/.
Запускается один раз вручную после скачивания датасета с Kaggle.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from minio_utils import upload_csv

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def upload_all_csvs():
    csv_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        print("Нет CSV-файлов в data/raw/. Сначала скачайте датасет с Kaggle.")
        return

    for filename in csv_files:
        local_path = os.path.join(RAW_DATA_DIR, filename)
        s3_key = f"raw/{filename}"
        upload_csv(local_path, s3_key)

    print(f"\nЗагружено файлов: {len(csv_files)}")


if __name__ == "__main__":
    upload_all_csvs()
