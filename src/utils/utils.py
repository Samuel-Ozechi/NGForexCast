# src/utils/utils.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
from typing import Dict
import os, json, joblib, shutil
from src.config.settings import Settings
from src.monitoring.evidently_profile import save_reference_profile
from sklearn.base import BaseEstimator


settings = Settings()
PROD_PATH = settings.PROD_PATH
MODEL_PATH = settings.MODEL_PATH
META_PATH = settings.META_PATH
os.makedirs(PROD_PATH, exist_ok=True)


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    # avoid division by zero for MAPE
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-6, y_true))) * 100
    # R2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape), "R2": float(r2)}

def plot_predictions(dates, y_true, y_pred, title: str, path: str):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, y_true, label="Actual", linestyle="--", color="black")
    plt.plot(dates, y_pred, label="Predicted")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def promote_to_production(inference_pipeline, metadata):
    tmp = PROD_PATH / "_tmp.joblib"
    joblib.dump(inference_pipeline, tmp)
    os.replace(tmp, MODEL_PATH)          # atomic swap
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

def build_reference_drift_profile(df: pd.DataFrame, model: BaseEstimator) -> str:
    """
    Generates in-sample predictions using the trained inference pipeline,
    ensuring lengths match by using the transformer's output.
    """
    # 1. Weekly Resampling (matching what the transformer expects internally)
    df["date"] = pd.to_datetime(df["date"])
    df_resampled = df.set_index("date").sort_index()  
    df_resampled = df_resampled["rate"].resample("W-FRI").last().to_frame().reset_index()

    # 2. Get the transformed data (to know which rows were kept after dropna)
    # The first step of your model pipeline is the TimeSeriesFeatureEngineer
    transformer = model.named_steps["feat_engineer"]
    transformed_df = transformer.transform(df_resampled)
    
    # 3. Generate predictions
    # The model.predict(df_resampled) internally runs transform() then predict()
    preds = model.predict(df_resampled)

    # 4. ALIGNMENT: Create the reference dataframe using ONLY the rows 
    # that survived the feature engineering (the 'transformed_df' rows)
    reference_df = transformed_df[["date", "rate"]].copy()
    reference_df["prediction"] = preds

    # 5. Get proxy drift baseline
    save_reference_profile(reference_df)

    return str(settings.REFERENCE_PROFILE_PATH)