"""Data loading utilities for Berlin EMS Demand project."""

from pathlib import Path
import pandas as pd
import requests

DATA_URL = (
    "https://raw.githubusercontent.com/Berliner-Feuerwehr/BF-Open-Data/"
    "main/Datasets/Daily_Data/BFw_mission_data_daily.csv"
)

LOCAL_PATH = Path(__file__).resolve().parents[1] / "data" / "BFw_mission_data_daily.csv"


def download_data(force: bool = False) -> Path:
    """Download the daily mission CSV if it does not exist locally."""
    LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOCAL_PATH.exists() and not force:
        return LOCAL_PATH

    print(f"Downloading data from {DATA_URL} ...")
    resp = requests.get(DATA_URL, timeout=60)
    resp.raise_for_status()
    LOCAL_PATH.write_bytes(resp.content)
    print(f"Saved to {LOCAL_PATH}")
    return LOCAL_PATH


def load_raw_data(force_download: bool = False) -> pd.DataFrame:
    """Load the raw daily mission data as a pandas DataFrame."""
    path = download_data(force=force_download)
    df = pd.read_csv(path, parse_dates=["mission_created_date"])
    df = df.sort_values("mission_created_date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(df.shape)
    print(df.head())
    print(df.dtypes)
