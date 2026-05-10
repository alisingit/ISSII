import json
import numpy as np
from kafka import KafkaProducer
from config import KAFKA_BOOTSTRAP_SERVERS, INPUT_TOPIC

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)
data = {
    "id": "test_001",
    "features": np.random.rand(1, 62).tolist()
}
producer.send(INPUT_TOPIC, value=data)
print("Тестовый запрос отправлен")