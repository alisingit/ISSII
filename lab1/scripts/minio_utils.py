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


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def upload_df(df: pd.DataFrame, s3_key: str) -> None:
    """Загружает DataFrame в MinIO как parquet-файл."""
    client = get_s3_client()
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow")
    buffer.seek(0)
    client.put_object(Bucket=BUCKET, Key=s3_key, Body=buffer.getvalue())
    print(f"Uploaded {len(df)} rows -> s3://{BUCKET}/{s3_key}")


def upload_csv(local_path: str, s3_key: str) -> None:
    """Загружает локальный CSV-файл в MinIO как есть."""
    client = get_s3_client()
    with open(local_path, "rb") as f:
        client.put_object(Bucket=BUCKET, Key=s3_key, Body=f.read())
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
