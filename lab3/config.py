import os
from dotenv import load_dotenv

load_dotenv()

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
BEST_RUN_ID = os.getenv("BEST_RUN_ID", "2cfa345feae245d8bf9c7076c88dc5de")  # XGBoost run

# Пути
ONNX_MODEL_PATH = "model.onnx"
BENTO_MODEL_TAG = "olist_classifier:latest"

# Kafka
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
INPUT_TOPIC = "inference_requests"
OUTPUT_TOPIC = "inference_responses"