"""
Предобработка транзакционных данных (pandas): заказы, позиции, товары, клиенты.

Входные данные (из MinIO raw/):
  - olist_orders_dataset.csv
  - olist_order_items_dataset.csv
  - olist_products_dataset.csv
  - product_category_name_translation.csv (имя из Kaggle)
  - olist_customers_dataset.csv
  - olist_order_payments_dataset.csv

Выходные данные (в MinIO staging/):
  - transactions_features.parquet
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from minio_utils import download_csv, upload_df


def load_tables() -> dict[str, pd.DataFrame]:
    tables = {
        "orders": download_csv("raw/olist_orders_dataset.csv"),
        "items": download_csv("raw/olist_order_items_dataset.csv"),
        "products": download_csv("raw/olist_products_dataset.csv"),
        "categories": download_csv("raw/product_category_name_translation.csv"),
        "customers": download_csv("raw/olist_customers_dataset.csv"),
        "payments": download_csv("raw/olist_order_payments_dataset.csv"),
    }
    print("Загружены таблицы:", {k: v.shape for k, v in tables.items()})
    return tables


def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует платежи до уровня order_id.

    Один заказ может оплачиваться несколькими способами, поэтому:
      - total_payment_value: сумма всех транзакций
      - max_installments: максимальное число платежей в рассрочку
      - payment_type: доминирующий способ оплаты (с наибольшей суммой)
    """
    dominant_type = (
        payments.sort_values("payment_value", ascending=False)
        .drop_duplicates("order_id", keep="first")[["order_id", "payment_type"]]
    )
    agg = payments.groupby("order_id").agg(
        total_payment_value=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
    ).reset_index()
    return agg.merge(dominant_type, on="order_id", how="left")


def join_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    payments_agg = aggregate_payments(tables["payments"])
    df = (
        tables["orders"]
        .merge(tables["customers"], on="customer_id", how="left")
        .merge(tables["items"], on="order_id", how="left")
        .merge(tables["products"], on="product_id", how="left")
        .merge(tables["categories"], on="product_category_name", how="left")
        .merge(payments_agg, on="order_id", how="left")
    )
    print(f"После join: {df.shape}")
    return df


def handle_datetime(df: pd.DataFrame) -> pd.DataFrame:
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df["delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days

    df["estimated_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.days

    # Насколько доставка отличалась от прогноза (отрицательное = раньше)
    df["delivery_delay_days"] = df["delivery_days"] - df["estimated_days"]

    df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour

    return df


DELIVERY_COLS_KEEP_NAN = [
    "delivery_days",
    "estimated_days",
    "delivery_delay_days",
]

TOP_CUSTOMER_STATES = 15


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    print("Пропуски до обработки:\n", df.isnull().sum()[df.isnull().sum() > 0])

    # Колонки, связанные с доставкой, оставляем как есть: глобальная медиана
    # искажала бы распределение (в ~3% случаев дата доставки отсутствует полностью).
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(DELIVERY_COLS_KEEP_NAN)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("unknown")

    print("Пропуски после обработки:", df.isnull().sum()[df.isnull().sum() > 0].to_dict())
    return df


def cap_customer_state(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляем топ-N штатов по числу заказов, остальные сворачиваем в 'other'"""
    if "customer_state" not in df.columns:
        return df
    top = df["customer_state"].value_counts().head(TOP_CUSTOMER_STATES).index
    df["customer_state"] = df["customer_state"].where(df["customer_state"].isin(top), "other")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Ценовая категория товара
    df["price_bin"] = pd.cut(
        df["price"],
        bins=[0, 50, 150, 500, np.inf],
        labels=["low", "medium", "high", "premium"],
    )

    # Отношение стоимости фрахта к цене товара
    df["freight_ratio"] = df["freight_value"] / (df["price"] + 1e-6)
    df["is_late_delivery"] = (df["delivery_delay_days"] > 0).astype(int)

    # Бинарная целевая переменная будет добавлена после join с отзывами
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    low_card_cols = ["order_status", "customer_state", "price_bin", "payment_type"]
    for col in low_card_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "order_id",
        "customer_id",
        "product_id",
        "delivery_days",
        "estimated_days",
        "delivery_delay_days",
        "is_late_delivery",
        "purchase_dayofweek",
        "purchase_month",
        "purchase_hour",
        "price",
        "freight_value",
        "freight_ratio",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
        "product_photos_qty",
        "product_name_lenght",
        "product_description_lenght",
        "total_payment_value",
        "max_installments",
    ]
    ohe_cols = [
        c for c in df.columns
        if c.startswith(("order_status_", "customer_state_", "price_bin_", "payment_type_"))
    ]
    keep_cols = [c for c in keep_cols if c in df.columns] + ohe_cols
    return df[keep_cols]


def run():
    print("=== transactions preprocessing: start ===")
    tables = load_tables()
    df = join_tables(tables)
    df = handle_datetime(df)
    df = handle_missing(df)
    df = cap_customer_state(df)
    df = feature_engineering(df)
    df = encode_categoricals(df)
    df = select_features(df)
    upload_df(df, "staging/transactions_features.parquet")
    print(f"=== Done. Shape: {df.shape} ===")


if __name__ == "__main__":
    run()
