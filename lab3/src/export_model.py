"""Экспорт лучшей XGBoost-модели из MLflow в ONNX.

В отличие от прежней версии экспортируется ВЕСЬ sklearn-pipeline
(StandardScaler + XGBClassifier), чтобы препроцессинг включался в граф ONNX
и сервис принимал «сырые» признаки, не зависящие от версии sklearn в проде.
"""

import sys
from pathlib import Path

import mlflow
import onnx
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
)
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import BEST_RUN_ID, MLFLOW_TRACKING_URI, ONNX_MODEL_PATH  # noqa: E402


def _register_xgboost_converter() -> None:
    """Регистрирует конвертер XGBClassifier в skl2onnx (его нет в стандартной поставке)."""
    update_registered_converter(
        XGBClassifier,
        "XGBoostXGBClassifier",
        calculate_linear_classifier_output_shapes,
        convert_xgboost,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )


def export_pipeline() -> Path:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    pipeline = mlflow.sklearn.load_model(f"runs:/{BEST_RUN_ID}/model")
    booster = pipeline.named_steps["model"].get_booster()
    n_features = booster.num_features()

    _register_xgboost_converter()

    onnx_model = convert_sklearn(
        pipeline,
        initial_types=[("float_input", FloatTensorType([None, n_features]))],
        target_opset={"": 14, "ai.onnx.ml": 2},
        # zipmap=False -> вероятности возвращаются как 2D-массив, а не как список dict
        options={id(pipeline): {"zipmap": False}},
    )

    target = Path(ONNX_MODEL_PATH).resolve()
    onnx.save_model(onnx_model, str(target))
    print(f"ONNX модель сохранена: {target} (входов признаков: {n_features})")
    return target


if __name__ == "__main__":
    export_pipeline()
