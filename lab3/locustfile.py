"""Сценарий нагрузочного тестирования BentoML REST endpoint.

Запуск:
    locust -f locustfile.py --host http://localhost:3000 \\
        --headless -u 50 -r 5 --run-time 1m \\
        --csv reports/load_test --html reports/load_test.html
"""

import os
import random

import numpy as np
from locust import HttpUser, between, task

N_FEATURES = int(os.getenv("LOCUST_N_FEATURES", "71"))
BATCH_SIZES = [1, 1, 1, 1, 4, 8]


class InferenceUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict(self) -> None:
        batch_size = random.choice(BATCH_SIZES)
        payload = np.random.rand(batch_size, N_FEATURES).astype(np.float32).tolist()
        # name= группирует все запросы в общий счётчик независимо от размера батча
        self.client.post("/predict", json=payload, name="POST /predict")
