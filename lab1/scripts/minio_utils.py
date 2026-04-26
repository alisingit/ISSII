import io
import os

import boto3
import pandas as pd
from botocore.client import Config
from botocore.exceptions import ClientError

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET = os.getenv("MINIO_BUCKET", "data-lake")
PREVIEW_PREFIX = os.getenv("MINIO_PREVIEW_PREFIX", "preview")
PREVIEW_ROWS = int(os.getenv("MINIO_PREVIEW_ROWS", "50"))


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def _build_preview_key(s3_key: str) -> str:
    base, ext = os.path.splitext(s3_key)
    if ext.lower() == ".parquet":
        return f"{PREVIEW_PREFIX}/{base}.sample.csv"
    return f"{PREVIEW_PREFIX}/{s3_key}.sample.csv"


def upload_df(df: pd.DataFrame, s3_key: str) -> None:
    """Загружает DataFrame в MinIO как parquet-файл и CSV-preview для UI."""
    client = get_s3_client()
    parquet_buffer = io.BytesIO()
    df.to_parquet(parquet_buffer, index=False, engine="pyarrow")
    parquet_buffer.seek(0)
    client.put_object(
        Bucket=BUCKET,
        Key=s3_key,
        Body=parquet_buffer.getvalue(),
        ContentType="application/octet-stream",
    )
    print(f"Uploaded {len(df)} rows -> s3://{BUCKET}/{s3_key}")

    # Preview кладём в отдельный префикс, чтобы UI мог открыть CSV
    preview_key = _build_preview_key(s3_key)
    preview_buffer = io.StringIO()
    df.head(PREVIEW_ROWS).to_csv(preview_buffer, index=False)
    client.put_object(
        Bucket=BUCKET,
        Key=preview_key,
        Body=preview_buffer.getvalue().encode("utf-8"),
        ContentType="text/csv; charset=utf-8",
    )
    print(f"Uploaded preview ({min(len(df), PREVIEW_ROWS)} rows) -> s3://{BUCKET}/{preview_key}")


def upload_csv(local_path: str, s3_key: str) -> None:
    """Загружает локальный CSV-файл в MinIO как есть."""
    client = get_s3_client()
    with open(local_path, "rb") as f:
        client.put_object(
            Bucket=BUCKET,
            Key=s3_key,
            Body=f.read(),
            ContentType="text/csv; charset=utf-8",
        )
    print(f"Uploaded {local_path} -> s3://{BUCKET}/{s3_key}")


def download_df(s3_key: str) -> pd.DataFrame:
    """Скачивает parquet из MinIO в DataFrame."""
    client = get_s3_client()
    response = client.get_object(Bucket=BUCKET, Key=s3_key)
    return pd.read_parquet(io.BytesIO(response["Body"].read()), engine="pyarrow")


def download_csv(s3_key: str) -> pd.DataFrame:
    """Скачивает CSV из MinIO в DataFrame."""
    client = get_s3_client()
    response = client.get_object(Bucket=BUCKET, Key=s3_key)
    return pd.read_csv(io.BytesIO(response["Body"].read()))


def list_keys(prefix: str) -> list[str]:
    """Возвращает список ключей в MinIO по префиксу."""
    client = get_s3_client()
    response = client.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    return [obj["Key"] for obj in response.get("Contents", [])]


def key_exists(s3_key: str) -> bool:
    """Проверяет, существует ли объект в MinIO."""
    client = get_s3_client()
    try:
        client.head_object(Bucket=BUCKET, Key=s3_key)
        return True
    except ClientError:
        return False

def upload_df_partition(df: pd.DataFrame, key: str) -> None:
    """
    Сохраняет DataFrame как отдельный parquet-файл в MinIO.
    Используется для добавления новой партиции в директорию processed/final_dataset/.
    """
    client = get_s3_client()
    parquet_buffer = io.BytesIO()
    df.to_parquet(parquet_buffer, index=False, engine="pyarrow")
    parquet_buffer.seek(0)
    client.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=parquet_buffer.getvalue(),
        ContentType="application/octet-stream",
    )
    print(f"Uploaded partition ({len(df)} rows) -> s3://{BUCKET}/{key}")


def download_ids_index() -> set:
    """
    Загружает индексный файл processed/final_dataset/_order_ids.parquet
    и возвращает множество уже существующих order_id.
    Если файла нет, возвращает пустой set.
    """
    try:
        df = download_df("processed/final_dataset/_order_ids.parquet")
        return set(df["order_id"].tolist())
    except Exception:
        return set()


def update_ids_index(new_ids: list[str]) -> None:
    """
    Дополняет индексный файл новыми order_id.
    Загружает текущий индекс, объединяет с new_ids и перезаписывает файл.
    """
    current = download_ids_index()
    current.update(new_ids)
    idx_df = pd.DataFrame({"order_id": list(current)})
    upload_df(idx_df, "processed/final_dataset/_order_ids.parquet")