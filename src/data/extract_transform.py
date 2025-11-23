import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from src.config.settings import Settings
from src.data.fetch_db import fetch_exchange_rates as fetch_exchange_rate_table

# Load global settings instance
settings = Settings()


# LOGGING CONFIG
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# PATHS
RAW_PATH = settings.RAW_DATA_DIR / "ngn_us_exchange_rates_raw.csv"
PROCESSED_PATH = settings.PROCESSED_DATA_DIR / "ngn_us_exchange_rates_cleaned.csv"
FEATURES_PATH = settings.FEATURE_DATA_DIR / "ngn_us_exchange_rates_features.csv"

# Ensure directories exist
os.makedirs(settings.RAW_DATA_DIR, exist_ok=True)
os.makedirs(settings.PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(settings.FEATURE_DATA_DIR, exist_ok=True)


# CLEAN DATA
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert Date to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop rows with invalid date or rate
    df = df.dropna(subset=["date", "rate"])

    # Sort
    df = df.sort_values("date").reset_index(drop=True)

    logger.info("Cleaned dataset:")
    logger.info(df.head())

    return df


# FEATURE ENGINEERING
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.set_index("date")

    # --- Weekly aggregation ---
    df_weekly = df["rate"].resample("W-FRI").last()
    df_weekly = df_weekly.to_frame()

    # --- Differenced series ---
    df_feat = df_weekly.copy()
    df_feat["diff"] = df_feat["rate"].diff()

    # --- Lags ---
    for lag in [1, 2, 6, 8, 11, 13, 15, 16, 25, 32, 34]:
        df_feat[f"lag_{lag}"] = df_feat["diff"].shift(lag)

    # --- Rolling stats ---
    for window in [4, 8, 12]:
        df_feat[f"rmean_{window}"] = (
            df_feat["diff"].rolling(window).mean().shift(1)
        )
        df_feat[f"rstd_{window}"] = (
            df_feat["diff"].rolling(window).std().shift(1)
        )

    # --- Returns ---
    for diff in [3, 7, 14, 21, 30, 60, 90]:
        df_feat[f"ret_{diff}"] = df_feat["diff"].pct_change(diff)

    # Reset index
    df_feat = df_feat.reset_index().rename(columns={"index": "Date"})

    # Calendar features
    df_feat["month"] = df_feat["date"].dt.month
    df_feat["day"] = df_feat["date"].dt.day
    df_feat["year"] = df_feat["date"].dt.year

    # Handle infinities
    df_feat.replace([np.inf, -np.inf], 0, inplace=True)

    # Drop rows with NaN values
    df_feat = df_feat.dropna().reset_index(drop=True)

    logger.info("Engineered features sample:")
    logger.info(df_feat.head())

    return df_feat


# MAIN RUNNER
def run_pipeline():
    logger.info("Starting data ingestion + preprocessing + feature engineering pipeline...")

    # Step 1 - Fetch
    df_raw = fetch_exchange_rate_table()
    df_raw.to_csv(RAW_PATH, index=False)
    logger.info(f"Saved raw data → {RAW_PATH}")
    logger.info(f"Total records fetched: {len(df_raw)}")

    # Step 2 - Clean
    df_clean = clean_data(df_raw)
    df_clean.to_csv(PROCESSED_PATH, index=False)
    logger.info(f"Saved cleaned data → {PROCESSED_PATH}")

    # Step 3 - Features
    df_features = engineer_features(df_clean)
    df_features.to_csv(FEATURES_PATH, index=False)
    logger.info(f"Saved engineered features → {FEATURES_PATH}")
    logger.info(f"Total records after feature engineering: {len(df_features)}")

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
