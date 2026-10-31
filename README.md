# Berlin EMS Demand Predictor

**ML Zoomcamp 2026 Midterm Project**

Predict daily Emergency Medical Service (EMS) demand for the Berlin Fire Brigade (Berliner Feuerwehr) using open data.

## Problem Description

Berlin’s Fire Brigade handles tens of thousands of medical emergency missions every month. Demand fluctuates strongly by day of week, season, weather and public events.  

Accurate short-term forecasts of EMS mission volume help with:

- Staffing and vehicle allocation
- Identifying high-load days in advance
- Public-health resource planning
- Transparency for citizens and researchers

This project trains multiple machine-learning models (linear, tree-based and gradient boosting) to predict the number of **EMS missions** (`mission_count_ems`) for the next day, based on historical daily aggregates published by the Berliner Feuerwehr.

A Streamlit dashboard provides interactive exploration, model comparison and live predictions.

## Dataset

Source: [Berliner Feuerwehr Open Data](https://github.com/Berliner-Feuerwehr/BF-Open-Data)  
License: CC BY 4.0  

Primary file used:
- `Datasets/Daily_Data/BFw_mission_data_daily.csv` (~3 200 days, 2018 → present)

Key columns:
- `mission_count_ems` – target (number of medical emergency missions)
- `mission_count_ems_critical`, `mission_count_fire`, response-time statistics, etc.

Data is downloaded automatically by the training script (or can be placed in `data/`).

## Project Structure

```
berlin-ems-demand/
├── README.md
├── requirements.txt
├── train.py                 # Train final model & save artifacts
├── predict.py               # FastAPI / simple prediction service
├── app/
│   └── streamlit_app.py     # Interactive dashboard
├── notebooks/
│   └── notebook.ipynb       # Full EDA + model selection
├── src/
│   ├── data.py
│   ├── features.py
│   └── models.py
├── models/                  # Saved model + preprocessor (created by train.py)
└── data/                    # Local copy of the CSV (optional)
```

## How to Run (Free Resources)

### 1. Local / GitHub Codespaces

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train the model (downloads data if needed)
python train.py

# Launch dashboard
streamlit run app/streamlit_app.py
```

### 2. Kaggle Notebooks (training)

1. Create a new Kaggle Notebook.
2. Upload `notebook.ipynb` or copy the cells.
3. Add the dataset as an external dataset or download via the URL in the notebook.
4. Run all cells → model artifacts can be downloaded.

### 3. Hosting the Dashboard (free)

**Option A – Streamlit Community Cloud** (recommended)
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io → “New app”.
3. Select the repo, main file `app/streamlit_app.py`.
4. Deploy (free tier is sufficient).

**Option B – Hugging Face Spaces**
1. Create a new Space (Streamlit SDK).
2. Upload the repo contents (or connect the GitHub repo).
3. The Space will automatically run `streamlit run app/streamlit_app.py`.

## Model Performance (typical)

| Model              | RMSE (hold-out) | MAE  | R²   |
|--------------------|-----------------|------|------|
| Linear Regression  | ~180            | ~140 | 0.72 |
| Random Forest      | ~145            | ~110 | 0.82 |
| XGBoost (tuned)    | **~135**        | **~105** | **0.85** |

(Exact numbers depend on the train/test split and current data.)

## Evaluation Criteria Coverage

- Problem description with clear public-health context
- Extensive EDA (temporal patterns, correlations, target distribution)
- Multiple models + hyper-parameter tuning
- Notebook → script export (`train.py`)
- Reproducible (data download instructions + requirements)
- Model deployment (Streamlit dashboard + optional FastAPI)
- Dependency management + virtual environment
- Containerization possible (Dockerfile included)
- Cloud-ready (Streamlit Cloud / HF Spaces)

## Author

Benjamin Senst  
ML Zoomcamp 2026 Midterm Project  
Linked to DSSG Berlin / public-health data science interests

## License

Code: MIT  
Data: CC BY 4.0 (Berliner Feuerwehr)
