"""
Загрузка итогового датасета из MinIO.

Читает директорию processed/final_dataset/ (партиционированное хранение),
объединяет все parquet-файлы, исключая индексный _order_ids.parquet.
"""

from pathlib import Path
import pandas as pd
import boto3
from botocore.client import Config
import io

from .config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_BUCKET,
    FEATURES_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
)


def _get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def load_dataset() -> pd.DataFrame:
    """
    Загружает итоговый датасет из MinIO (директория processed/final_dataset/).

    Returns
    -------
    pd.DataFrame
        Полный датасет со всеми признаками и целевой переменной is_satisfied.
    """
    client = _get_s3_client()
    prefix = f"{FEATURES_DIR}/"
    
    keys = []
    continuation_token = None
    while True:
        list_kwargs = dict(Bucket=MINIO_BUCKET, Prefix=prefix)
        if continuation_token:
            list_kwargs["ContinuationToken"] = continuation_token
        response = client.list_objects_v2(**list_kwargs)
        
        for obj in response.get("Contents", []):
            if obj["Key"].endswith(".parquet") and not obj["Key"].endswith("_order_ids.parquet"):
                keys.append(obj["Key"])
        
        if not response.get("IsTruncated"):
            break
        continuation_token = response.get("NextContinuationToken")
    
    if not keys:
        raise FileNotFoundError(f"Нет parquet-файлов в s3://{MINIO_BUCKET}/{prefix}")
    
    parts = []
    for key in keys:
        resp = client.get_object(Bucket=MINIO_BUCKET, Key=key)
        parts.append(pd.read_parquet(io.BytesIO(resp["Body"].read())))
    
    df = pd.concat(parts, ignore_index=True)

    print(f"Загружен датасет: {df.shape}")
    print(f"Баланс классов:\n{df['is_satisfied'].value_counts(normalize=True)}")

    return df


def prepare_train_test_val(df: pd.DataFrame):
    """
    Разделяет датасет на обучающую, валидационную и тестовую выборки.

    Параметры разбиения задаются в config.py.
    При TEST_SIZE=0.2 и VALIDATION_SIZE=0.25 (доля от train_val) получаем:
        - test:  20% от полного датасета
        - val:   25% от оставшихся 80% = 20% от полного датасета
        - train: 75% от оставшихся 80% = 60% от полного датасета

    Returns
    -------
    tuple[DataFrame, DataFrame, DataFrame, Series, Series, Series, list]
        X_train, X_val, X_test, y_train, y_val, y_test, feature_cols
    """
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import VarianceThreshold

    target = "is_satisfied"
    exclude = [
        target,
        "order_id",
        "customer_id",
        "product_id",
        "review_score",
    ]
    feature_cols = [c for c in df.columns if c not in exclude and c in df.columns]

    X = df[feature_cols].copy()
    y = df[target].copy()

    # Приводим все колонки к числовому типу, если есть object
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # Заполняем пропуски медианой
    X = X.fillna(X.median())

    selector = VarianceThreshold(threshold=0.0)
    X = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])
    feature_cols = list(X.columns)

    # 1. Отделяем тестовую выборку (20% от всех данных)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 2. Из оставшихся 80% выделяем валидацию (25% от train_val, т.е. 20% от полного)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=VALIDATION_SIZE,  # теперь это доля от train_val
        random_state=RANDOM_STATE, stratify=y_train_val
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols