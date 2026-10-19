import logging
import time
import joblib
import pandas as pd
from fastapi import FastAPI, Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from src.config import MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("heart-api")
REQUESTS = Counter("prediction_requests_total", "Prediction requests")
LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")
app = FastAPI(title="Heart Disease Risk API", version="1.0.0")
model = None


class Patient(BaseModel):
    age: float = Field(ge=0, le=120)
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: float


@app.on_event("startup")
def load_model():
    global model
    model = joblib.load(MODEL)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(patient: Patient):
    start = time.perf_counter()
    REQUESTS.inc()
    probability = float(model.predict_proba(pd.DataFrame([patient.model_dump()]))[0, 1])
    prediction = int(probability >= 0.5)
    LATENCY.observe(time.perf_counter() - start)
    log.info(
        "prediction=%s confidence=%.4f", prediction, max(probability, 1 - probability)
    )
    return {
        "prediction": prediction,
        "heart_disease_risk": probability,
        "confidence": max(probability, 1 - probability),
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
