import json
import numpy as np
import onnxruntime as ort
from kafka import KafkaConsumer, KafkaProducer
from config import KAFKA_BOOTSTRAP_SERVERS, INPUT_TOPIC, OUTPUT_TOPIC, ONNX_MODEL_PATH

session = ort.InferenceSession(ONNX_MODEL_PATH)
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)
consumer = KafkaConsumer(
    INPUT_TOPIC,
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

for msg in consumer:
    data = msg.value
    arr = np.array(data['features'], dtype=np.float32)
    outputs = session.run(None, {"float_input": arr})
    prob = float(outputs[0][0])          # одномерный массив, берём первый (и единственный) элемент
    producer.send(OUTPUT_TOPIC, value={'id': data['id'], 'probability': prob})
    print(f"Обработан {data['id']}")