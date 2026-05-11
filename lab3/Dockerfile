FROM python:3.12-slim

WORKDIR /app

ENV PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=10 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BENTOML_HOME=/app/bentoml

COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

COPY src/ src/
COPY model.onnx .
COPY bentofile.yaml .

# Регистрируем ONNX-модель в BentoML model store, чтобы service.py мог её получить
# при импорте модуля во время `bentoml serve`.
RUN python -c "import onnx, bentoml; bentoml.onnx.save_model('olist_classifier', onnx.load('model.onnx'))"

EXPOSE 3000

CMD ["bentoml", "serve", "src.service:svc", "--host", "0.0.0.0", "--port", "3000"]
