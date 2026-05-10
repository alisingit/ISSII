FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY model.onnx .
COPY bentofile.yaml .
RUN python -c "import onnx, bentoml; model = onnx.load('model.onnx'); bentoml.onnx.save_model('olist_classifier', model)" && bentoml build
CMD ["bentoml", "serve", "olist_satisfaction:latest", "--host", "0.0.0.0", "--port", "3000"]