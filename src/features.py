"""Feature engineering for Berlin EMS Demand prediction."""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features from the date column."""
    df = df.copy()
    df["year"] = df["mission_created_date"].dt.year
    df["month"] = df["mission_created_date"].dt.month
    df["day"] = df["mission_created_date"].dt.day
    df["dayofweek"] = df["mission_created_date"].dt.dayofweek  # 0=Mon
    df["dayofyear"] = df["mission_created_date"].dt.dayofyear
    df["weekofyear"] = df["mission_created_date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = df["mission_created_date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["mission_created_date"].dt.is_month_end.astype(int)

    # Cyclical encoding for seasonality
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def add_lag_features(df: pd.DataFrame, target: str = "mission_count_ems", lags: list[int] = None) -> pd.DataFrame:
    """Add lag and rolling-window features of the target."""
    if lags is None:
        lags = [1, 2, 3, 7, 14, 28]
    df = df.copy()
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)

    # Rolling statistics (shifted so they only use past information)
    df[f"{target}_roll7_mean"] = df[target].shift(1).rolling(7).mean()
    df[f"{target}_roll7_std"] = df[target].shift(1).rolling(7).std()
    df[f"{target}_roll14_mean"] = df[target].shift(1).rolling(14).mean()
    df[f"{target}_roll28_mean"] = df[target].shift(1).rolling(28).mean()
    return df


def prepare_features(
    df: pd.DataFrame,
    target: str = "mission_count_ems",
    dropna: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Full feature pipeline.
    Returns X (features) and y (target).
    """
    df = add_calendar_features(df)
    df = add_lag_features(df, target=target)

    # Select feature columns
    feature_cols = [
        "year", "month", "day", "dayofweek", "dayofyear", "weekofyear",
        "is_weekend", "is_month_start", "is_month_end",
        "month_sin", "month_cos", "dow_sin", "dow_cos",
        f"{target}_lag_1", f"{target}_lag_2", f"{target}_lag_3",
        f"{target}_lag_7", f"{target}_lag_14", f"{target}_lag_28",
        f"{target}_roll7_mean", f"{target}_roll7_std",
        f"{target}_roll14_mean", f"{target}_roll28_mean",
        # Additional useful raw columns
        "mission_count_ems_critical",
        "mission_count_fire",
        "mission_count_technical_rescue",
        "response_time_ems_critical_mean",
        "response_time_ems_critical_median",
    ]

    # Only keep columns that actually exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    if dropna:
        df = df.dropna(subset=feature_cols + [target]).reset_index(drop=True)

    X = df[feature_cols]
    y = df[target]
    return X, y, df  # also return the enriched df for later use


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer (useful for pipelines)."""

    def __init__(self, target: str = "mission_count_ems"):
        self.target = target

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Expects a DataFrame with the original columns
        X_feat, _, _ = prepare_features(X, target=self.target, dropna=False)
        return X_feat
