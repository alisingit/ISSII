"""Экспорт лучшей XGBoost-модели из MLflow в ONNX.

Экспортируется весь sklearn-pipeline (StandardScaler + XGBClassifier), чтобы
препроцессинг включался в граф ONNX и сервис принимал «сырые» признаки,
не зависящие от версии sklearn в продакшене.

Источник модели:
1) Если задан и существует локальный путь BEST_RUN_PATH — модель загружается напрямую
   через `mlflow.sklearn.load_model(<path>)` (без MLflow tracking server).
2) Иначе используется fallback на `runs:/{BEST_RUN_ID}/model` против
   `MLFLOW_TRACKING_URI`.
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

from config import (  # noqa: E402
    BEST_RUN_ID,
    BEST_RUN_PATH,
    MLFLOW_TRACKING_URI,
    ONNX_MODEL_PATH,
)


def _register_xgboost_converter() -> None:
    """Регистрирует конвертер XGBClassifier в skl2onnx (его нет в стандартной поставке)."""
    update_registered_converter(
        XGBClassifier,
        "XGBoostXGBClassifier",
        calculate_linear_classifier_output_shapes,
        convert_xgboost,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )


def _load_pipeline():
    local = Path(BEST_RUN_PATH) if BEST_RUN_PATH else None
    if local and local.exists() and local.is_dir():
        print(f"Загрузка модели из локального пути: {local}")
        return mlflow.sklearn.load_model(str(local))
    print(
        f"Локальный путь не найден ({local}). "
        f"Загрузка через MLflow tracking URI {MLFLOW_TRACKING_URI}"
    )
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(f"runs:/{BEST_RUN_ID}/model")


def export_pipeline() -> Path:
    pipeline = _load_pipeline()
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
