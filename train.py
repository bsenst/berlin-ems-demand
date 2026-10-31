#!/usr/bin/env python
"""
Train the final model for Berlin EMS Demand prediction.
Saves artifacts to models/.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import load_raw_data
from src.features import prepare_features
from src.models import get_models, evaluate_model, time_series_cv_score, save_artifacts

MODELS_DIR = Path(__file__).parent / "models"
TARGET = "mission_count_ems"


def main():
    print("=== Berlin EMS Demand – Training ===")

    # 1. Load data
    df = load_raw_data()
    print(f"Raw data shape: {df.shape}")

    # 2. Feature engineering
    X, y, df_enriched = prepare_features(df, target=TARGET)
    print(f"Feature matrix: {X.shape}")

    # 3. Time-based split (last 20% as hold-out)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"Train: {X_train.shape[0]} days | Test: {X_test.shape[0]} days")

    # 4. Train & evaluate candidates
    models = get_models()
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        cv_rmse = time_series_cv_score(model, X_train, y_train)
        metrics["cv_rmse"] = cv_rmse
        results[name] = metrics
        print(f"  Hold-out RMSE: {metrics['rmse']:.1f} | MAE: {metrics['mae']:.1f} | R²: {metrics['r2']:.3f}")
        print(f"  CV RMSE:       {cv_rmse:.1f}")

    # 5. Select best model (lowest hold-out RMSE)
    best_name = min(results, key=lambda k: results[k]["rmse"])
    best_model = models[best_name]
    print(f"\n>>> Best model: {best_name}")

    # Re-fit on full training data (already done inside evaluate, but re-fit for clarity)
    best_model.fit(X_train, y_train)

    # 6. Save artifacts
    save_artifacts(
        model=best_model,
        feature_names=list(X.columns),
        metrics={"best_model": best_name, **results[best_name], "all_results": results},
        path=MODELS_DIR,
    )

    # Also save a small sample of recent data for the dashboard
    df_enriched.tail(90).to_csv(MODELS_DIR / "recent_data.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
