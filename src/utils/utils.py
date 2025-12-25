# src/utils/utils.py
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
from typing import Dict
import os, json, joblib, shutil
from src.config.settings import Settings

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
    tmp = PROD_PATH + "/_tmp.joblib"
    joblib.dump(inference_pipeline, tmp)
    os.replace(tmp, MODEL_PATH)          # atomic swap
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)