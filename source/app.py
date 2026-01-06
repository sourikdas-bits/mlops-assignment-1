from fastapi import FastAPI, Request
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import logging
import os
import time

from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Heart Disease Prediction API")
Instrumentator().instrument(app).expose(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_TARGET_URI=os.getenv("MLFLOW_TARGET_URI")
mlflow.set_tracking_uri(MLFLOW_TARGET_URI)
model = mlflow.sklearn.load_model("models:/HeartDiseaseModel/latest")

class Patient(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: Patient):
    df = pd.DataFrame([data.dict()])
    prob = model.predict_proba(df)[0][1]
    pred = int(prob > 0.5)

    logger.info(f"Prediction={pred}, Probability={prob:.4f}")

    return {
        "prediction": pred,
        "probability": prob
    }

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.2f}ms"
    )
    return response
