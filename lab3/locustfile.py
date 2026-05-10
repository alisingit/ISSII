from locust import HttpUser, task, between
import numpy as np

class InferenceUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict(self):
        data = np.random.rand(1, 62).astype(np.float32).tolist()
        self.client.post("/predict", json=data)