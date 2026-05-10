import numpy as np
import bentoml
from bentoml.io import NumpyNdarray
import onnxruntime as ort

session = ort.InferenceSession("/app/model.onnx")

svc = bentoml.Service("olist_satisfaction")

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(input_arr: np.ndarray) -> np.ndarray:
    ort_inputs = {"float_input": input_arr.astype(np.float32)}
    ort_outs = session.run(None, ort_inputs)
    return ort_outs[0]   # Одномерный массив вероятностей