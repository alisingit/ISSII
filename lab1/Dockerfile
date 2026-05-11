FROM apache/airflow:2.9.1-python3.11

USER airflow

ENV PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=10

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
