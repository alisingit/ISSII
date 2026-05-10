import mlflow
import onnx
from onnxmltools.convert.xgboost import convert as convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from config import MLFLOW_TRACKING_URI, BEST_RUN_ID, ONNX_MODEL_PATH

def export_xgboost():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Загрузка sklearn-пайплайна, внутри которого шаг 'model' – XGBoost
    model = mlflow.sklearn.load_model(f"runs:/{BEST_RUN_ID}/model")
    booster = model.named_steps['model'].get_booster()
    n_features = booster.num_features()
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(model.named_steps['model'], initial_types=initial_type)
    onnx.save_model(onnx_model, ONNX_MODEL_PATH)
    print(f"ONNX модель сохранена: {ONNX_MODEL_PATH}")

if __name__ == "__main__":
    export_xgboost()