"""
app.py
------
FastAPI serving application for the Heart Disease risk classifier.

Endpoints:
    GET  /            -> health/info
    GET  /health       -> liveness probe (used by Docker/K8s)
    POST /predict       -> returns prediction + confidence for a patient record

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import logging
import time
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

# --------------------------------------------------------------------------
# Logging setup (Task 8: Monitoring & Logging - API request logging)
# --------------------------------------------------------------------------
_log_handlers = [logging.StreamHandler()]
try:
    _log_path = os.path.join(os.path.dirname(__file__), "..", "api_requests.log")
    _log_handlers.append(logging.FileHandler(_log_path))
except (PermissionError, OSError) as e:
    # Falls back to console-only logging if the filesystem is read-only or
    # not writable by the container's user (e.g. a locked-down K8s
    # securityContext). Never let logging setup crash the whole app.
    print(f"WARNING: could not open log file for writing ({e}); logging to console only.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=_log_handlers,
)
logger = logging.getLogger("heart-disease-api")

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.joblib")

model = None
metadata = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, metadata
    logger.info("Loading model from %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    logger.info("Model loaded: %s", metadata.get("model_name"))
    yield
    logger.info("Shutting down API")


app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Predicts risk of heart disease from patient health data (UCI Heart Disease dataset).",
    version="1.0.0",
    lifespan=lifespan,
)

# In-memory request counters for the simple human-readable stats endpoint
REQUEST_STATS = {"total_requests": 0, "predict_requests": 0, "errors": 0}

# Prometheus instrumentation: exposes GET /metrics in Prometheus exposition
# format (request counts, latency histograms, status codes, etc.) so
# Prometheus can scrape it directly and Grafana can visualize it (Task 8).
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


class PatientRecord(BaseModel):
    age: int = Field(..., json_schema_extra={"example": 63}, description="Age in years")
    sex: int = Field(..., json_schema_extra={"example": 1}, description="1 = male, 0 = female")
    cp: int = Field(..., json_schema_extra={"example": 3}, description="Chest pain type (0-3)")
    trestbps: int = Field(..., json_schema_extra={"example": 145}, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., json_schema_extra={"example": 233}, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(
        ..., json_schema_extra={"example": 1}, description="Fasting blood sugar > 120 mg/dl (1=true, 0=false)"
    )
    restecg: int = Field(..., json_schema_extra={"example": 0}, description="Resting ECG results (0-2)")
    thalach: int = Field(..., json_schema_extra={"example": 150}, description="Maximum heart rate achieved")
    exang: int = Field(..., json_schema_extra={"example": 0}, description="Exercise induced angina (1=yes, 0=no)")
    oldpeak: float = Field(..., json_schema_extra={"example": 2.3}, description="ST depression induced by exercise")
    slope: int = Field(..., json_schema_extra={"example": 0}, description="Slope of the peak exercise ST segment (0-2)")
    ca: float = Field(
        ..., json_schema_extra={"example": 0.0}, description="Number of major vessels colored by fluoroscopy (0-3)"
    )
    thal: float = Field(
        ...,
        json_schema_extra={"example": 1.0},
        description="Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)",
    )


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    model_name: str


@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()
    REQUEST_STATS["total_requests"] += 1
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(
        "%s %s -> %s (%.2f ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/")
def root():
    return {
        "service": "Heart Disease Risk Prediction API",
        "status": "running",
        "model": metadata.get("model_name") if metadata else "not loaded",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/stats")
def stats():
    """Simple human-readable request counters (separate from Prometheus /metrics)."""
    return REQUEST_STATS


@app.post("/predict", response_model=PredictionResponse)
def predict(record: PatientRecord):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        REQUEST_STATS["predict_requests"] += 1
        input_df = pd.DataFrame([record.model_dump()])
        proba = model.predict_proba(input_df)[0]
        pred = int(proba[1] >= 0.5)
        confidence = float(proba[pred])

        logger.info("Prediction made: input=%s -> prediction=%s confidence=%.3f", record.model_dump(), pred, confidence)

        return PredictionResponse(
            prediction=pred,
            label="Disease Present" if pred == 1 else "No Disease",
            confidence=round(confidence, 4),
            model_name=metadata.get("model_name", "unknown"),
        )
    except Exception as e:
        REQUEST_STATS["errors"] += 1
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
