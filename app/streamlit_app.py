"""
Berlin EMS Demand – Interactive Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import sys

# Make sure we can import from src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_raw_data
from src.features import prepare_features, add_calendar_features

st.set_page_config(
    page_title="Berlin EMS Demand Predictor",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = ROOT / "models"


@st.cache_data(ttl=3600)
def load_data():
    return load_raw_data()


@st.cache_resource
def load_model_artifacts():
    model = joblib.load(MODELS_DIR / "model.joblib")
    feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")
    metrics = joblib.load(MODELS_DIR / "metrics.joblib")
    return model, feature_names, metrics


def main():
    st.title("🚑 Berlin EMS Demand Predictor")
    st.markdown(
        """
        **ML Zoomcamp 2026 Midterm Project**  
        Predict daily Emergency Medical Service (EMS) mission volume for the Berlin Fire Brigade  
        using open data from [Berliner Feuerwehr Open Data](https://github.com/Berliner-Feuerwehr/BF-Open-Data).
        """
    )

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Overview & EDA", "Model Performance", "Live Prediction", "About"],
    )

    df = load_data()

    if page == "Overview & EDA":
        show_overview(df)
    elif page == "Model Performance":
        show_model_performance()
    elif page == "Live Prediction":
        show_prediction(df)
    else:
        show_about()


def show_overview(df: pd.DataFrame):
    st.header("📊 Data Overview & Exploratory Analysis")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Days of data", f"{len(df):,}")
    col2.metric("First date", df["mission_created_date"].min().strftime("%Y-%m-%d"))
    col3.metric("Last date", df["mission_created_date"].max().strftime("%Y-%m-%d"))
    col4.metric("Avg. EMS missions / day", f"{df['mission_count_ems'].mean():.0f}")

    st.subheader("Daily EMS Missions over Time")
    fig = px.line(
        df,
        x="mission_created_date",
        y="mission_count_ems",
        title="Daily number of EMS missions",
        labels={"mission_count_ems": "EMS missions", "mission_created_date": "Date"},
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Seasonality
    st.subheader("Seasonality & Calendar Patterns")
    df_cal = add_calendar_features(df)

    c1, c2 = st.columns(2)
    with c1:
        by_dow = df_cal.groupby("dayofweek")["mission_count_ems"].mean().reset_index()
        by_dow["day_name"] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        fig = px.bar(by_dow, x="day_name", y="mission_count_ems", title="Average EMS by Day of Week")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        by_month = df_cal.groupby("month")["mission_count_ems"].mean().reset_index()
        fig = px.bar(by_month, x="month", y="mission_count_ems", title="Average EMS by Month")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Target Distribution")
    fig = px.histogram(df, x="mission_count_ems", nbins=50, title="Distribution of daily EMS missions")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation with other mission types")
    corr_cols = [
        "mission_count_ems",
        "mission_count_ems_critical",
        "mission_count_fire",
        "mission_count_technical_rescue",
        "response_time_ems_critical_mean",
    ]
    corr = df[corr_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation matrix")
    st.plotly_chart(fig, use_container_width=True)


def show_model_performance():
    st.header("🏆 Model Performance")

    try:
        model, feature_names, metrics = load_model_artifacts()
    except Exception as e:
        st.error(
            "Model artifacts not found. Please run `python train.py` first.\n\n"
            f"Error: {e}"
        )
        return

    st.success(f"Best model: **{metrics.get('best_model', 'unknown')}**")

    # Metrics table
    all_results = metrics.get("all_results", {})
    if all_results:
        rows = []
        for name, m in all_results.items():
            rows.append({
                "Model": name,
                "Hold-out RMSE": round(m["rmse"], 1),
                "Hold-out MAE": round(m["mae"], 1),
                "R²": round(m["r2"], 3),
                "CV RMSE": round(m.get("cv_rmse", np.nan), 1),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Feature Importance (if available)")
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        fig = px.bar(imp.head(15), x="importance", y="feature", orientation="h",
                     title="Top 15 Features")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importances not available for this model type.")


def show_prediction(df: pd.DataFrame):
    st.header("🔮 Live Prediction")

    try:
        model, feature_names, metrics = load_model_artifacts()
    except Exception:
        st.error("Model not found. Run `python train.py` first.")
        return

    st.markdown(
        "The model uses the most recent days of data to construct lag and rolling features, "
        "then predicts the **next day's** EMS mission volume."
    )

    if st.button("Predict next day", type="primary"):
        with st.spinner("Building features and predicting..."):
            recent = df.tail(40).copy()
            X, _, _ = prepare_features(recent, dropna=False)
            X_last = X.dropna().tail(1)

            if X_last.empty:
                st.error("Not enough history to build features.")
                return

            X_last = X_last.reindex(columns=feature_names, fill_value=0)
            pred = float(model.predict(X_last)[0])

            st.metric(
                label="Predicted EMS missions for the next day",
                value=f"{pred:,.0f}",
                delta=f"{pred - df['mission_count_ems'].iloc[-1]:,.0f} vs last observed day",
            )

            st.subheader("Recent context")
            st.line_chart(
                df.tail(30).set_index("mission_created_date")["mission_count_ems"],
                height=300,
            )

    st.subheader("Last 14 observed days")
    st.dataframe(
        df.tail(14)[["mission_created_date", "mission_count_ems", "mission_count_ems_critical", "mission_count_fire"]],
        use_container_width=True,
    )


def show_about():
    st.header("ℹ️ About this project")
    st.markdown(
        """
        ### Context
        This is the midterm project for the **DataTalksClub Machine Learning Zoomcamp 2026**.

        The goal is to apply the full ML workflow (data preparation → EDA → model selection → 
        deployment) to a real public-health / emergency-services problem in Berlin.

        ### Data Source
        - [Berliner Feuerwehr Open Data](https://github.com/Berliner-Feuerwehr/BF-Open-Data)
        - License: CC BY 4.0

        ### Author
        Benjamin Senst  
        Medical Expert & Data Scientist  
        Linked to DSSG Berlin activities

        ### Tech Stack
        - Python, pandas, scikit-learn, XGBoost
        - Streamlit for the interactive dashboard
        - FastAPI for optional REST API
        - Designed for free hosting (Streamlit Cloud / Hugging Face Spaces) and free compute (Kaggle / GitHub Codespaces)
        """
    )


if __name__ == "__main__":
    main()
