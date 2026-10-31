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

**Update frequency:** The source data is updated **daily** (official Data Catalog).

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
├── .github/workflows/
│   └── retrain.yml          # Scheduled model retraining
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

## Automated Retraining (GitHub Actions)

The repository includes a scheduled GitHub Action that retrains the model on the latest data.

### What it does
- Runs every **Sunday at 03:00 UTC** (source data is updated daily; weekly retraining is sufficient)
- Can also be triggered **manually** from the Actions tab
- Downloads the newest CSV, runs `train.py`, and uploads the model artifacts as a workflow artifact (kept for 30 days)

### How to set it up / use it

1. The workflow file is already present:
   ```
   .github/workflows/retrain.yml
   ```

2. **No extra secrets are required** – the action only needs the public dataset URL and the code in this repository.

3. **Enable Actions** (if not already enabled):
   - Go to the repository → **Settings** → **Actions** → **General**
   - Under “Actions permissions” choose “Allow all actions and reusable workflows” (or at least allow actions created by GitHub).

4. **Run manually** (recommended the first time):
   - Go to the **Actions** tab
   - Select **“Retrain EMS Demand Model”**
   - Click **“Run workflow”** → **Run workflow**

5. After a successful run you can download the fresh `model.joblib`, `feature_names.joblib` and `metrics.joblib` from the workflow artifacts.

6. (Optional) If you want the trained model committed back into the repository automatically, uncomment the final “Commit updated model” step inside `.github/workflows/retrain.yml` and give the default `GITHUB_TOKEN` write permission (Settings → Actions → General → Workflow permissions → Read and write).

### Why this helps the project
- Keeps the model up-to-date with the daily-updated source data
- Demonstrates reproducibility and production-minded automation
- Strengthens the “Reproducibility” and overall professionalism criteria of the ML Zoomcamp evaluation

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
- Reproducible (data download instructions + requirements + automated retrain)
- Model deployment (Streamlit dashboard + optional FastAPI)
- Dependency management + virtual environment
- Containerization possible (Dockerfile included)
- Cloud-ready (Streamlit Cloud / HF Spaces) + scheduled retraining

## Author

Benjamin Senst  
ML Zoomcamp 2026 Midterm Project  
Linked to DSSG Berlin / public-health data science interests

## License

Code: MIT  
Data: CC BY 4.0 (Berliner Feuerwehr)
