#!/usr/bin/env python
"""
Simple prediction service (FastAPI).
Can be run locally or inside Docker / Codespaces.
"""

from pathlib import Path
from typing import List
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from src.features import prepare_features
from src.data import load_raw_data

MODELS_DIR = Path(__file__).parent / "models"

app = FastAPI(
    title="Berlin EMS Demand Predictor",
    description="Predict next-day Emergency Medical Service mission volume in Berlin",
    version="1.0.0",
)

# Load artifacts at startup
model = None
feature_names = None


@app.on_event("startup")
def load_model():
    global model, feature_names
    model = joblib.load(MODELS_DIR / "model.joblib")
    feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")
    print("Model loaded successfully.")


class PredictionRequest(BaseModel):
    # Optional: allow user to supply recent history; otherwise we use the latest data
    use_latest: bool = True


class PredictionResponse(BaseModel):
    predicted_ems_missions: float
    model_name: str
    features_used: List[str]


@app.get("/")
def root():
    return {
        "message": "Berlin EMS Demand Predictor API",
        "docs": "/docs",
        "predict": "/predict",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    """
    Predict the number of EMS missions for the next day.
    Uses the most recent available data to build lag features.
    """
    df = load_raw_data()
    # We need enough history for lags → take last 40 days
    recent = df.tail(40).copy()
    X, _, _ = prepare_features(recent, dropna=False)

    # Take the last complete row
    X_last = X.dropna().tail(1)
    if X_last.empty:
        raise ValueError("Not enough history to build features.")

    # Align columns
    X_last = X_last.reindex(columns=feature_names, fill_value=0)
    pred = float(model.predict(X_last)[0])

    return PredictionResponse(
        predicted_ems_missions=round(pred, 1),
        model_name=str(type(model).__name__),
        features_used=feature_names,
    )


if __name__ == "__main__":
    uvicorn.run("predict:app", host="0.0.0.0", port=8000, reload=True)
