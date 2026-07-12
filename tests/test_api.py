"""
Unit tests for src/app.py (FastAPI serving layer).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402
from fastapi.testclient import TestClient

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.joblib")

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="Trained model artifact not found; run `python src/train.py` first.",
)

from app import app 

client = TestClient(app)
client.__enter__()

VALID_PAYLOAD = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0.0,
    "thal": 1.0,
}


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "service" in response.json()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict_endpoint_valid_input():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["label"] in ("Disease Present", "No Disease")


def test_predict_endpoint_missing_field_returns_422():
    bad_payload = VALID_PAYLOAD.copy()
    del bad_payload["age"]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_stats_endpoint_tracks_requests():
    before = client.get("/stats").json()["total_requests"]
    client.get("/health")
    after = client.get("/stats").json()["total_requests"]
    assert after > before


def test_prometheus_metrics_endpoint_exposed():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text or "python_info" in response.text
