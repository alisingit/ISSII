import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
# Лучший run по test_f1 = 0.9043 (XGBoost n_estimators=300, learning_rate=0.05).
BEST_RUN_ID = os.getenv("BEST_RUN_ID", "e86b0c215eea4ad98086a186bed7cdf3")
# Прямой путь к локальной директории артефакта (MLmodel + model.pkl). Если задан и существует —
# используется напрямую, без обращения к MLflow tracking server.
_default_local_path = (
    Path(__file__).resolve().parent.parent
    / "lab2"
    / "mlruns"
    / "1"
    / BEST_RUN_ID
    / "artifacts"
    / "model"
)
BEST_RUN_PATH = os.getenv("BEST_RUN_PATH", str(_default_local_path))

# Пути
ONNX_MODEL_PATH = "model.onnx"
BENTO_MODEL_TAG = "olist_classifier:latest"

# Kafka
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
INPUT_TOPIC = "inference_requests"
OUTPUT_TOPIC = "inference_responses"