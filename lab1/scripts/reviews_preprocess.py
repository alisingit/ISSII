"""
Предобработка отзывов и геоданных (pandas + sklearn).

Содержание:
  - токенизация, удаление стоп-слов (EN + PT), TF-IDF через sklearn
  - L2-норма TF-IDF вектора как числовой признак (tfidf_norm)
  - флаги наличия текста и время ответа на отзыв
  - расстояние покупатель–продавец по формуле Хаверсина

Входные данные (из MinIO raw/):
  - olist_order_reviews_dataset.csv
  - olist_geolocation_dataset.csv
  - olist_sellers_dataset.csv
  - olist_orders_dataset.csv
  - olist_order_items_dataset.csv
  - olist_customers_dataset.csv

Выходные данные (в MinIO staging/):
  - reviews_features.parquet
"""

import math
import os
import sys
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

sys.path.insert(0, os.path.dirname(__file__))

from minio_utils import download_csv, upload_df


_PT_STOPWORDS = {
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para",
    "com", "uma", "os", "no", "se", "na", "por", "mais", "as",
    "dos", "como", "mas", "ao", "ele", "das", "seu", "sua",
    "ou", "quando", "muito", "nos", "ja", "eu", "tambem", "so",
    "pelo", "pela", "ate", "isso", "ela", "entre", "depois",
    "sem", "mesmo", "aos", "seus", "quem", "nas", "me", "esse",
    "eles", "voce", "essa", "num", "nem", "suas", "meu",
    "minha", "numa", "pelos", "elas", "havia", "seja", "qual",
    "sera", "tenho", "lhe", "deles", "essas", "esses",
    "pelas", "este", "dele", "tu", "te", "voces", "vos", "lhes",
    "meus", "minhas", "teu", "tua", "teus", "tuas", "nosso",
    "nossa", "nossos", "nossas", "dela", "delas", "esta", "estes",
    "estas", "aquele", "aquela", "aqueles", "aquelas", "isto",
    "aquilo", "estou", "estamos", "estao", "estive", "nao",
    "sao", "ha", "foi", "era", "ser", "ter", "bem", "muito",
}

# Объединённый список стоп-слов для TF-IDF: корпус преимущественно PT,
# но встречаются английские слова в названиях товаров и комментариях.
# Фильтруем по длине, чтобы список был согласован с token_pattern (`\w{2,}`).
_TFIDF_STOPWORDS = sorted({w for w in (set(ENGLISH_STOP_WORDS) | _PT_STOPWORDS) if len(w) >= 2})


def _strip_accents(text: str) -> str:
    """Удаляет диакритику, чтобы `não/esta/está` совпадали со стоп-словами."""
    if not isinstance(text, str):
        return ""
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )

# NLP

def process_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Токенизация, TF-IDF, длина текста, целевая переменная."""
    reviews = reviews.copy()
    reviews["review_comment_message"] = reviews["review_comment_message"].fillna("")
    reviews["review_comment_title"] = reviews["review_comment_title"].fillna("")
    reviews["review_creation_date"] = pd.to_datetime(reviews["review_creation_date"], errors="coerce")
    reviews["review_answer_timestamp"] = pd.to_datetime(reviews["review_answer_timestamp"], errors="coerce")

    full_text_raw = (
        reviews["review_comment_title"].fillna("") + " " + reviews["review_comment_message"].fillna("")
    ).str.lower()
    reviews["full_text"] = full_text_raw.map(_strip_accents)
    reviews["has_title"] = (reviews["review_comment_title"].str.strip() != "").astype(int)
    reviews["has_message"] = (reviews["review_comment_message"].str.strip() != "").astype(int)

    def _token_count(text: str) -> int:
        tokens = text.split()
        return sum(1 for t in tokens if t not in _PT_STOPWORDS and len(t) > 1)

    reviews["review_text_length"] = reviews["full_text"].apply(_token_count)
    reviews["response_hours"] = (
        reviews["review_answer_timestamp"] - reviews["review_creation_date"]
    ) / pd.Timedelta(hours=1)

    # TF-IDF на нормализованном тексте, объединённые стоп-слова EN+PT
    vectorizer = TfidfVectorizer(
        min_df=5,
        max_features=512,
        stop_words=_TFIDF_STOPWORDS,
        token_pattern=r"(?u)\b\w{2,}\b",
    )
    tfidf_matrix = vectorizer.fit_transform(reviews["full_text"])

    # L2-норма строки ≡ норма TF-IDF вектора для каждого отзыва
    reviews["tfidf_norm"] = np.sqrt(np.asarray(tfidf_matrix.power(2).sum(axis=1)).ravel())

    # Целевая переменная
    reviews["is_satisfied"] = (reviews["review_score"].astype(int) >= 4).astype(int)

    return reviews[
        [
            "order_id",
            "review_score",
            "is_satisfied",
            "has_title",
            "has_message",
            "review_text_length",
            "response_hours",
            "tfidf_norm",
        ]
    ]


# Геодистанция

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расстояние между двумя точками на сфере (км)"""
    if any(v != v for v in (lat1, lon1, lat2, lon2)):   # NaN check
        return float("nan")
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def process_geodistance() -> pd.DataFrame:
    """Среднее расстояние покупатель–продавец по order_id."""
    geo = download_csv("raw/olist_geolocation_dataset.csv")
    sellers = download_csv("raw/olist_sellers_dataset.csv")
    orders = download_csv("raw/olist_orders_dataset.csv")
    items = download_csv("raw/olist_order_items_dataset.csv")
    customers = download_csv("raw/olist_customers_dataset.csv")

    # Средние координаты по ZIP-коду
    geo_avg = (
        geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .mean()
        .rename(columns={"geolocation_lat": "lat", "geolocation_lng": "lng"})
        .reset_index()
    )

    sellers_geo = sellers.merge(
        geo_avg,
        left_on="seller_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    ).rename(columns={"lat": "seller_lat", "lng": "seller_lng"})[
        ["seller_id", "seller_lat", "seller_lng"]
    ]

    customers_geo = customers.merge(
        geo_avg,
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    ).rename(columns={"lat": "customer_lat", "lng": "customer_lng"})[
        ["customer_id", "customer_lat", "customer_lng"]
    ]

    geo_df = (
        items[["order_id", "seller_id"]]
        .merge(orders[["order_id", "customer_id"]], on="order_id", how="left")
        .merge(sellers_geo, on="seller_id", how="left")
        .merge(customers_geo, on="customer_id", how="left")
    )

    geo_df["distance_km"] = geo_df.apply(
        lambda r: _haversine(r["seller_lat"], r["seller_lng"],
                             r["customer_lat"], r["customer_lng"]),
        axis=1,
    )

    return geo_df.groupby("order_id")["distance_km"].mean().rename("avg_seller_distance_km").reset_index()


def run():
    print("=== pandas preprocessing (reviews + geo): start ===")

    reviews_raw = download_csv("raw/olist_order_reviews_dataset.csv")
    reviews_df = process_reviews(reviews_raw)
    print(f"Reviews: {reviews_df.shape}")

    geo_df = process_geodistance()
    print(f"Geo: {geo_df.shape}")

    result = reviews_df.merge(geo_df, on="order_id", how="left")
    upload_df(result, "staging/reviews_features.parquet")
    print(f"=== Done. Shape: {result.shape} ===")


if __name__ == "__main__":
    run()
