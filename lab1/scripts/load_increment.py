"""
Инкрементальная загрузка: обрабатывает новые файлы из increment/ и
дописывает результат к processed/final_dataset.parquet.

Логика:
1. Проверяет наличие файлов в increment/
2. Если файл — сырой CSV (содержит order_purchase_timestamp),
   прогоняет его через те же этапы pandas-предобработки;
   иначе (parquet или уже обработанный CSV) добавляет напрямую.
3. Мержит с текущим processed/final_dataset.parquet (дедупликация по order_id).
4. Сохраняет обновлённый датасет обратно.
"""

import io
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from minio_utils import (
    download_df, upload_df, upload_df_partition,
    download_ids_index, update_ids_index, get_s3_client,
    list_keys,
)

BUCKET = os.getenv("MINIO_BUCKET", "data-lake")

# Колонка, однозначно указывающая на сырые данные заказов
_RAW_SIGNAL_COL = "order_purchase_timestamp"


def _preprocess_raw_csv(raw_bytes: bytes) -> pd.DataFrame:
    """
    Прогоняет сырой CSV заказов через те же этапы предобработки, что и
    основной пайплайн (без загрузки в MinIO).
    """
    import transactions_preprocess

    df = pd.read_csv(io.BytesIO(raw_bytes))
    df = transactions_preprocess.handle_datetime(df)
    df = transactions_preprocess.handle_missing(df)
    df = transactions_preprocess.feature_engineering(df)
    df = transactions_preprocess.encode_categoricals(df)
    df = transactions_preprocess.select_features(df)
    return df


def move_processed_increment(s3_key: str) -> None:
    """Перемещает обработанный файл из increment/ в increment/done/."""
    client = get_s3_client()
    new_key = s3_key.replace("increment/", "increment/done/", 1)
    client.copy_object(Bucket=BUCKET, CopySource={"Bucket": BUCKET, "Key": s3_key}, Key=new_key)
    client.delete_object(Bucket=BUCKET, Key=s3_key)
    print(f"Moved {s3_key} -> {new_key}")


def run() -> dict:
    """
    Возвращает {"new_rows": int} — сколько строк добавлено.
    Возвращает {"new_rows": 0} если инкремента нет или все записи уже существуют.
    """
    print("=== Load increment: start ===")

    increment_keys = [k for k in list_keys("increment/") if k.endswith(".csv") or k.endswith(".parquet")]

    if not increment_keys:
        print("Инкремента нет, пропускаем.")
        return {"new_rows": 0}

    print(f"Найдено файлов инкремента: {len(increment_keys)}")

    # 1. Загружаем только индексный файл
    existing_ids = download_ids_index()

    # 2. Обрабатываем новые файлы
    new_parts = []
    for key in increment_keys:
        print(f"Обрабатываем: {key}")
        client = get_s3_client()
        response = client.get_object(Bucket=BUCKET, Key=key)
        raw_bytes = response["Body"].read()

        if key.endswith(".csv"):
            probe = pd.read_csv(io.BytesIO(raw_bytes), nrows=1)
            if _RAW_SIGNAL_COL in probe.columns:
                print(f"  → сырой CSV, запускаем предобработку")
                chunk = _preprocess_raw_csv(raw_bytes)
            else:
                print(f"  → обработанный CSV, добавляем напрямую")
                chunk = pd.read_csv(io.BytesIO(raw_bytes))
        else:
            chunk = pd.read_parquet(io.BytesIO(raw_bytes))

        new_parts.append(chunk)
        move_processed_increment(key)

    if not new_parts:
        return {"new_rows": 0}

    new_data = pd.concat(new_parts, ignore_index=True)

    # 3. Дедупликация по имеющемуся индексу
    if "order_id" in new_data.columns:
        new_data = new_data[~new_data["order_id"].isin(existing_ids)]
        if new_data.empty:
            print("Все новые записи уже существуют.")
            return {"new_rows": 0}

    # 4. Сохраняем новый файл-партицию (не трогая старые)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d%H%M%S")
    partition_key = f"processed/final_dataset/increment_{ts}.parquet"
    upload_df_partition(new_data, partition_key)

    # 5. Обновляем индексный файл
    update_ids_index(new_data["order_id"].tolist())

    added = len(new_data)
    print(f"=== Done. Добавлено строк: {added} ===")
    return {"new_rows": added}


if __name__ == "__main__":
    result = run()
    print(result)
