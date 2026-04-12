"""
DAG: ecommerce_preprocessing_pipeline

Граф задач:
  check_raw_data
        |
  load_raw_to_minio  (если данные не загружены)
        |
  ┌─────┴─────┐
pandas_preprocess  spark_preprocess   <- параллельно
  └─────┬─────┘
  validate_staging
        |
  join_features
        |
  check_increment
        |
  load_increment  (если есть новые файлы в increment/)
        |
  pipeline_complete
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator

sys.path.insert(0, "/opt/airflow/scripts")

DEFAULT_ARGS = {
    "owner": "lab1-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def _check_raw_data(**ctx) -> str:
    """Проверяет, загружены ли уже сырые данные в MinIO."""
    import minio_utils
    keys = minio_utils.list_keys("raw/")
    csv_keys = [k for k in keys if k.endswith(".csv")]
    if len(csv_keys) >= 8:
        print(f"Raw data OK: {len(csv_keys)} файлов")
        return "pandas_preprocess"
    else:
        print(f"Raw data не найдены ({len(csv_keys)} CSV). Нужна ручная загрузка.")
        # В реальном проекте здесь был бы оператор загрузки с Kaggle API.
        # Пока бросаем исключение с инструкцией.
        raise FileNotFoundError(
            "Загрузите CSV-файлы датасета Olist в data/raw/ и запустите: "
            "python scripts/upload_raw_data.py"
        )


def _validate_staging(**ctx) -> None:
    """Проверяет, что оба staging-файла существуют и не пустые."""
    import minio_utils
    import pandas as pd

    for key in ["staging/transactions_features.parquet", "staging/reviews_features.parquet"]:
        df = minio_utils.download_df(key)
        assert len(df) > 0, f"{key} пустой!"
        assert "order_id" in df.columns, f"{key}: нет order_id"
        print(f"[OK] {key}: {df.shape}")


def _check_increment(**ctx) -> str:
    """Ветвление: есть ли инкремент?"""
    import minio_utils
    keys = minio_utils.list_keys("increment/")
    new_files = [k for k in keys if (k.endswith(".csv") or k.endswith(".parquet")) and "done/" not in k]
    if new_files:
        print(f"Найден инкремент: {new_files}")
        return "load_increment"
    else:
        print("Инкремента нет.")
        return "pipeline_complete"


def _pipeline_complete(**ctx) -> None:
    print("Pipeline завершён успешно.")


with DAG(
    dag_id="ecommerce_preprocessing_pipeline",
    default_args=DEFAULT_ARGS,
    description="Предобработка данных Olist: pandas + Spark ветки, MinIO, инкремент",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["lab1", "preprocessing", "ecommerce"],
) as dag:

    check_raw = PythonOperator(
        task_id="check_raw_data",
        python_callable=_check_raw_data,
    )

    def _run_pandas(**ctx):
        import pandas_preprocess
        pandas_preprocess.run()

    def _run_spark(**ctx):
        import spark_preprocess
        spark_preprocess.run()

    def _run_join(**ctx):
        import join_features
        join_features.run()

    def _run_increment(**ctx):
        import load_increment
        return load_increment.run()

    task_pandas = PythonOperator(
        task_id="pandas_preprocess",
        python_callable=_run_pandas,
    )

    task_spark = PythonOperator(
        task_id="spark_preprocess",
        python_callable=_run_spark,
    )

    validate_staging = PythonOperator(
        task_id="validate_staging",
        python_callable=_validate_staging,
    )

    task_join = PythonOperator(
        task_id="join_features",
        python_callable=_run_join,
    )

    check_incr = BranchPythonOperator(
        task_id="check_increment",
        python_callable=_check_increment,
    )

    task_increment = PythonOperator(
        task_id="load_increment",
        python_callable=_run_increment,
    )

    task_done = PythonOperator(
        task_id="pipeline_complete",
        python_callable=_pipeline_complete,
        trigger_rule="none_failed_min_one_success",
    )

    # Граф зависимостей
    check_raw >> [task_pandas, task_spark]
    [task_pandas, task_spark] >> validate_staging
    validate_staging >> task_join
    task_join >> check_incr
    check_incr >> [task_increment, task_done]
    task_increment >> task_done
