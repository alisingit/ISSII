"""
Строит файл инкремента из случайной части уже обработанного датасета.

Строки берутся из processed/final_dataset.parquet (те же фичи),
но order_id заменяются на новые уникальные значения, тогда load_increment
добавит их к основному parquet (дедупликация по старым order_id их не отсечёт).

Результат:
  - lab1/data/increment/orders_increment.parquet (локально)
  - s3://data-lake/increment/orders_increment.parquet (MinIO)
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from minio_utils import download_df, upload_df, get_s3_client

DEFAULT_N = 400
BUCKET = os.getenv("MINIO_BUCKET", "data-lake")

def build_increment(n_rows: int, seed: int, upload: bool) -> str:
    # Читаем все parquet-файлы из директории processed/final_dataset/
    # (кроме индексного _order_ids.parquet)
    client = get_s3_client()
    prefix = "processed/final_dataset/"
    response = client.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    keys = [obj["Key"] for obj in response.get("Contents", []) 
            if obj["Key"].endswith(".parquet") and not obj["Key"].endswith("_order_ids.parquet")]
    
    if not keys:
        raise RuntimeError("Нет parquet-файлов в processed/final_dataset/ (исключая _order_ids.parquet)")
    
    # Читаем и объединяем все найденные партиции
    parts = []
    for key in keys:
        parts.append(download_df(key))
    processed = pd.concat(parts, ignore_index=True)
    
    n = min(n_rows, len(processed))
    if n < 1:
        raise RuntimeError("processed/final_dataset/*.parquet пустой")

    sample = processed.sample(n=n, random_state=seed).reset_index(drop=True).copy()
    prefix = pd.Timestamp.utcnow().strftime("inc-%Y%m%d-")
    sample["order_id"] = [f"{prefix}{i:06d}" for i in range(len(sample))]

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "increment")
    os.makedirs(out_dir, exist_ok=True)
    local_path = os.path.join(out_dir, "orders_increment.parquet")
    sample.to_parquet(local_path, index=False, engine="pyarrow")
    print(f"Локально: {local_path} ({len(sample)} строк, {len(sample.columns)} колонок)")

    if upload:
        # Загружаем инкремент в MinIO в папку increment/
        upload_df(sample, "increment/orders_increment.parquet")
        print("MinIO: s3://data-lake/increment/orders_increment.parquet")

    return local_path


def main():
    p = argparse.ArgumentParser(description="Собрать инкремент из части processed-датасета")
    p.add_argument("-n", type=int, default=DEFAULT_N, help="Число строк (по умолчанию 400)")
    p.add_argument("--seed", type=int, default=42, help="random_state для sample")
    p.add_argument("--no-upload", action="store_true", help="только локальный parquet, без MinIO")
    args = p.parse_args()
    build_increment(args.n, args.seed, upload=not args.no_upload)


if __name__ == "__main__":
    main()
