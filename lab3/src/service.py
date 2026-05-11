"""BentoML-сервис, экспонирующий ONNX-модель Olist через REST."""

import os
from pathlib import Path

import numpy as np
import onnx
import bentoml
from bentoml.exceptions import NotFound
from bentoml.io import NumpyNdarray

MODEL_NAME = os.getenv("BENTO_MODEL_NAME", "olist_classifier")
MODEL_TAG = os.getenv("BENTO_MODEL_TAG", f"{MODEL_NAME}:latest")
ONNX_MODEL_PATH = os.getenv(
    "ONNX_MODEL_PATH",
    str(Path(__file__).resolve().parent.parent / "model.onnx"),
)


def _ensure_model_in_store() -> bentoml.Model:
    """Регистрирует ONNX-модель в BentoML model store, если её там ещё нет.

    Это позволяет одинаково работать и в Docker (где модель сохраняется на этапе
    сборки образа), и при локальном запуске `bentoml serve src.service:svc`.
    """
    try:
        return bentoml.onnx.get(MODEL_TAG)
    except NotFound:
        return bentoml.onnx.save_model(MODEL_NAME, onnx.load(ONNX_MODEL_PATH))


_runner = _ensure_model_in_store().to_runner()
svc = bentoml.Service("olist_satisfaction", runners=[_runner])


def _extract_positive_proba(outputs) -> np.ndarray:
    """Достаёт вероятность класса 1 из выходов ONNX-классификатора.

    Принимает либо кортеж `(labels, probability_matrix)`, либо одиночный массив:
    из 2D-матрицы вероятностей берётся колонка положительного класса.
    """
    if isinstance(outputs, (list, tuple)):
        for out in outputs:
            arr = np.asarray(out)
            if arr.ndim == 2 and arr.shape[-1] >= 2:
                return arr[:, 1].astype(np.float32)
        return np.asarray(outputs[0]).astype(np.float32)
    arr = np.asarray(outputs)
    if arr.ndim == 2 and arr.shape[-1] >= 2:
        return arr[:, 1].astype(np.float32)
    return arr.astype(np.float32)


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
async def predict(input_arr: np.ndarray) -> np.ndarray:
    """Возвращает вероятность положительного класса для каждого ряда признаков."""
    if input_arr.ndim == 1:
        input_arr = input_arr.reshape(1, -1)
    input_arr = input_arr.astype(np.float32)

    outputs = await _runner.async_run(input_arr)
    return _extract_positive_proba(outputs)
