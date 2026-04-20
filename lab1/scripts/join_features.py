"""
Объединяет результаты двух веток предобработки в финальный датасет.

Входные данные (из MinIO staging/):
  - transactions_features.parquet  (от transactions_preprocess.py)
  - reviews_features.parquet        (от reviews_preprocess.py)

Выходные данные (в MinIO processed/):
  - final_dataset.parquet
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from minio_utils import download_df, upload_df


def validate_df(df: pd.DataFrame, name: str) -> None:
    """Базовая валидация: проверяем наличие данных и ключевых колонок."""
    assert len(df) > 0, f"{name}: датафрейм пустой"
    assert "order_id" in df.columns, f"{name}: нет колонки order_id"
    null_ratio = df.isnull().mean()
    high_null = null_ratio[null_ratio > 0.5]
    if len(high_null) > 0:
        print(f"[WARN] {name}: колонки с >50% пропусков: {high_null.index.tolist()}")
    print(f"[OK] {name}: {df.shape}, null_total={df.isnull().sum().sum()}")


def run():
    print("=== Join features: start ===")

    transactions = download_df("staging/transactions_features.parquet")
    reviews = download_df("staging/reviews_features.parquet")

    validate_df(transactions, "transactions")
    validate_df(reviews, "reviews")

    # Агрегируем транзакции до уровня заказа (order_id уникален в reviews)
    # items_per_order - сколько позиций в заказе
    agg_transactions = transactions.groupby("order_id").agg(
        items_count=("product_id", "count"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        avg_freight_ratio=("freight_ratio", "mean"),
        delivery_days=("delivery_days", "first"),
        estimated_days=("estimated_days", "first"),
        delivery_delay_days=("delivery_delay_days", "first"),
        is_late_delivery=("is_late_delivery", "first"),
        purchase_dayofweek=("purchase_dayofweek", "first"),
        purchase_month=("purchase_month", "first"),
        purchase_hour=("purchase_hour", "first"),
    ).reset_index()
    agg_transactions["freight_to_price_ratio"] = (
        agg_transactions["total_freight"] / (agg_transactions["total_price"] + 1e-6)
    )

    # Добавляем OHE-колонки (берём первую запись по order_id)
    ohe_cols = [c for c in transactions.columns if c.startswith(("order_status_", "customer_state_", "price_bin_"))]
    if ohe_cols:
        ohe_part = transactions.groupby("order_id")[ohe_cols].first().reset_index()
        agg_transactions = agg_transactions.merge(ohe_part, on="order_id", how="left")

    final = agg_transactions.merge(reviews, on="order_id", how="inner")
    print(f"После join: {final.shape}")

    # Финальная валидация
    validate_df(final, "final_dataset")
    assert "is_satisfied" in final.columns, "Нет целевой переменной is_satisfied"

    upload_df(final, "processed/final_dataset.parquet")
    print(f"=== Done. Shape: {final.shape} ===")
    print(f"Целевая переменная — баланс классов:\n{final['is_satisfied'].value_counts(normalize=True)}")


if __name__ == "__main__":
    run()
