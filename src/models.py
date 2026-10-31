"""Model training and evaluation helpers."""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
from pathlib import Path


def get_models() -> dict[str, Any]:
    """Return a dictionary of candidate models."""
    return {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
    }


def evaluate_model(model, X_train, y_train, X_test, y_test) -> dict[str, float]:
    """Train and return common regression metrics on the test set."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }


def time_series_cv_score(model, X, y, n_splits: int = 5) -> float:
    """Return mean RMSE from TimeSeriesSplit cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(
        model, X, y,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    return float(-scores.mean())


def save_artifacts(model, feature_names: list[str], metrics: dict, path: Path):
    """Save model, feature list and metrics."""
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / "model.joblib")
    joblib.dump(feature_names, path / "feature_names.joblib")
    joblib.dump(metrics, path / "metrics.joblib")
    print(f"Artifacts saved to {path}")


def load_artifacts(path: Path):
    """Load previously saved artifacts."""
    model = joblib.load(path / "model.joblib")
    feature_names = joblib.load(path / "feature_names.joblib")
    metrics = joblib.load(path / "metrics.joblib")
    return model, feature_names, metrics
