"""
Студент 2 — PySpark-предобработка отзывов и геоданных.

Входные данные (из MinIO raw/):
  - olist_order_reviews_dataset.csv
  - olist_geolocation_dataset.csv
  - olist_sellers_dataset.csv
  - olist_customers_dataset.csv  (для координат покупателей)

Выходные данные (в MinIO staging/):
  - reviews_features.parquet
"""

import os
import sys

from pyspark.ml.feature import (
    HashingTF,
    IDF,
    StopWordsRemover,
    Tokenizer,
)
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType

sys.path.insert(0, os.path.dirname(__file__))

from minio_utils import (
    BUCKET,
    MINIO_ACCESS_KEY,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    download_csv,
    upload_df,
)

# Геоданные: расстояние по формуле Хаверсина
HAVERSINE_UDF = F.udf(
    lambda lat1, lon1, lat2, lon2: _haversine(lat1, lon1, lat2, lon2),
    FloatType(),
)


def _haversine(lat1, lon1, lat2, lon2):
    """Расстояние между двумя точками на сфере (км)."""
    import math

    if None in (lat1, lon1, lat2, lon2):
        return None
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return float(2 * R * math.asin(math.sqrt(a)))


def build_spark_session() -> SparkSession:
    """Создаёт SparkSession с коннектором к MinIO через S3A."""
    endpoint = MINIO_ENDPOINT.replace("http://", "").replace("https://", "")
    spark = (
        SparkSession.builder.appName("ecommerce-spark-preprocess")
        .master("local[*]")
        .config("spark.hadoop.fs.s3a.endpoint", f"http://{endpoint}")
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_csv_to_spark(spark: SparkSession, s3_key: str):
    """Загружает CSV из MinIO в Spark DataFrame через pandas (без hadoop-aws зависимости)."""
    pdf = download_csv(s3_key)
    return spark.createDataFrame(pdf)


def process_reviews(spark: SparkSession):
    """Токенизация, удаление стоп-слов, TF-IDF для текстов отзывов."""
    reviews = load_csv_to_spark(spark, "raw/olist_order_reviews_dataset.csv")

    # Заполняем пустые тексты
    reviews = reviews.fillna({"review_comment_message": "", "review_comment_title": ""})

    # Объединяем заголовок и текст отзыва
    reviews = reviews.withColumn(
        "full_text",
        F.concat_ws(" ", F.col("review_comment_title"), F.col("review_comment_message")),
    )

    # Нижний регистр
    reviews = reviews.withColumn("full_text", F.lower(F.col("full_text")))

    # Токенизация
    tokenizer = Tokenizer(inputCol="full_text", outputCol="words_raw")
    reviews = tokenizer.transform(reviews)

    # Удаление стоп-слов (английские + португальские)
    pt_stopwords = [
        "de", "a", "o", "que", "e", "do", "da", "em", "um", "para",
        "com", "uma", "os", "no", "se", "na", "por", "mais", "as",
        "dos", "como", "mas", "ao", "ele", "das", "à", "seu", "sua",
        "ou", "quando", "muito", "nos", "já", "eu", "também", "só",
        "pelo", "pela", "até", "isso", "ela", "entre", "depois",
        "sem", "mesmo", "aos", "seus", "quem", "nas", "me", "esse",
        "eles", "você", "essa", "num", "nem", "suas", "meu", "às",
        "minha", "numa", "pelos", "elas", "havia", "seja", "qual",
        "será", "nós", "tenho", "lhe", "deles", "essas", "esses",
        "pelas", "este", "dele", "tu", "te", "vocês", "vos", "lhes",
        "meus", "minhas", "teu", "tua", "teus", "tuas", "nosso",
        "nossa", "nossos", "nossas", "dela", "delas", "esta", "estes",
        "estas", "aquele", "aquela", "aqueles", "aquelas", "isto",
        "aquilo", "estou", "está", "estamos", "estão", "estive",
    ]
    remover = StopWordsRemover(
        inputCol="words_raw",
        outputCol="words",
        stopWords=StopWordsRemover.loadDefaultStopWords("english") + pt_stopwords,
    )
    reviews = remover.transform(reviews)

    # TF-IDF (hashingTF — не требует fit на всём корпусе)
    hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=512)
    reviews = hashing_tf.transform(reviews)

    idf = IDF(inputCol="raw_features", outputCol="tfidf_features", minDocFreq=5)
    idf_model = idf.fit(reviews)
    reviews = idf_model.transform(reviews)

    # Длина отзыва как числовой признак
    reviews = reviews.withColumn("review_text_length", F.size(F.col("words")))

    # Бинарная целевая переменная: оценка >= 4 => доволен (1)
    reviews = reviews.withColumn(
        "is_satisfied",
        (F.col("review_score").cast("int") >= 4).cast("int"),
    )

    result_cols = [
        "order_id",
        "review_score",
        "is_satisfied",
        "review_text_length",
        "tfidf_features",
    ]
    return reviews.select(result_cols)


def process_geodistance(spark: SparkSession):
    """Вычисляет расстояние между покупателем и продавцом."""
    geo = load_csv_to_spark(spark, "raw/olist_geolocation_dataset.csv")
    sellers = load_csv_to_spark(spark, "raw/olist_sellers_dataset.csv")
    orders = load_csv_to_spark(spark, "raw/olist_orders_dataset.csv")
    items = load_csv_to_spark(spark, "raw/olist_order_items_dataset.csv")

    # Средние координаты по zip-коду
    geo_avg = geo.groupBy("geolocation_zip_code_prefix").agg(
        F.avg("geolocation_lat").alias("lat"),
        F.avg("geolocation_lng").alias("lng"),
    )

    # Координаты продавцов
    sellers_geo = sellers.join(
        geo_avg,
        sellers["seller_zip_code_prefix"] == geo_avg["geolocation_zip_code_prefix"],
        "left",
    ).select(
        F.col("seller_id"),
        F.col("lat").alias("seller_lat"),
        F.col("lng").alias("seller_lng"),
    )

    # Координаты покупателей по customer_id (уникальный customer_unique_id не нужен)
    customers = load_csv_to_spark(spark, "raw/olist_customers_dataset.csv")
    customers_geo = customers.join(
        geo_avg,
        customers["customer_zip_code_prefix"] == geo_avg["geolocation_zip_code_prefix"],
        "left",
    ).select(
        F.col("customer_id"),
        F.col("lat").alias("customer_lat"),
        F.col("lng").alias("customer_lng"),
    )

    # order_items + заказ → customer_id, затем координаты покупателя и продавца
    items_with_customer = items.join(
        orders.select("order_id", "customer_id"),
        on="order_id",
        how="left",
    )

    geo_df = (
        items_with_customer.join(sellers_geo, on="seller_id", how="left")
        .join(customers_geo, on="customer_id", how="left")
    )

    geo_df = geo_df.withColumn(
        "distance_km",
        HAVERSINE_UDF(
            F.col("seller_lat"),
            F.col("seller_lng"),
            F.col("customer_lat"),
            F.col("customer_lng"),
        ),
    )

    return geo_df.groupBy("order_id").agg(
        F.avg("distance_km").alias("avg_seller_distance_km")
    )


def run():
    print("=== Spark preprocessing: start ===")
    spark = build_spark_session()

    reviews_df = process_reviews(spark)
    geo_df = process_geodistance(spark)

    # Объединяем отзывы и геоданные по order_id
    result = reviews_df.join(geo_df, on="order_id", how="left")

    # Spark -> pandas -> MinIO
    # tfidf_features — вектор, сохраняем как отдельные столбцы нормой
    result_pd = result.select(
        "order_id", "review_score", "is_satisfied",
        "review_text_length", "avg_seller_distance_km",
    ).toPandas()

    upload_df(result_pd, "staging/reviews_features.parquet")
    print(f"=== Done. Shape: {result_pd.shape} ===")
    spark.stop()


if __name__ == "__main__":
    run()
