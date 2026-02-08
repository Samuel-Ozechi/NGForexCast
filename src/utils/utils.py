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
import mlflow
from datetime import datetime, timezone
import logging
from src.data.ingest import fetch_exchange_rates
import dagshub
from mlflow.tracking import MlflowClient
from pathlib import Path

settings = Settings()
PROD_PATH = settings.PROD_PATH
MODEL_PATH = settings.MODEL_PATH
META_PATH = settings.META_PATH
RAW_DATA_DIR = settings.RAW_DATA_DIR
os.makedirs(PROD_PATH, exist_ok=True) 
logger = logging.getLogger(__name__)


# MLflow setup
def _setup_mlflow():
    token = os.environ.get("DAGSHUB_USER_TOKEN")
    dagshub.auth.add_app_token(token)

    dagshub.init(
        repo_owner=os.getenv("DAGSHUB_USER"),
        repo_name="NGForexCast",
        mlflow=True,
    )

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    logger.info("MLflow tracking configured")

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

# Model version helpers
def _get_staged_model_version() -> str:
    """Return MLflow version currently pointed to by 'staging' alias."""
    client = MlflowClient()
    mv = client.get_model_version_by_alias(
        name=settings.MODEL_NAME,
        alias="staging",
    )
    return mv.version

def _get_local_model_version() -> str:
    meta_path = settings.META_PATH
    if not meta_path.exists():
        return None

    with open(meta_path, "r") as f:
        meta = json.load(f)

    return meta.get("mlflow_version")

def _save_model_meta(model_version):

    meta = {
        "model_name": settings.MODEL_NAME,
        "mlflow_version": model_version,
    }

    with open(settings.META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved model metadata: {meta}")

def _load_mlflow_model():
    model_uri = f"models:/{settings.MODEL_NAME}@staging"
    pipeline = mlflow.sklearn.load_model(model_uri)
    return pipeline

def get_predictions(df: pd.DataFrame, pipeline: BaseEstimator) -> pd.DataFrame:
    """
    Transforms input data using the feature engineering step of the model pipeline.
    """
    # 1. Weekly Resampling (matching what the transformer expects internally)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()  
    df_resampled = df["rate"].resample("W-FRI").last().to_frame().reset_index()

    # 2. Apply feature engineering transformation
    transformer = pipeline.named_steps["feat_engineer"]
    transformed_df = transformer.transform(df_resampled)

    # 3. Generate predictions
    # The model.predict(df_resampled) internally runs transform() then predict()
    preds = pipeline.predict(df_resampled)

    # 4. ALIGNMENT: Create the reference dataframe using ONLY the rows 
    # that survived the feature engineering (the 'transformed_df' rows)
    predicted_df = transformed_df[["date", "rate"]].copy()
    predicted_df["prediction"] = preds
    return predicted_df

def build_reference_drift_profile(df: pd.DataFrame, pipeline: BaseEstimator) -> str:
    """
    Generates in-sample predictions using the trained inference pipeline,
    ensuring lengths match by using the transformer's output.
    """
    # Get in-sample predictions
    reference_df = get_predictions(df, pipeline)

    # Save the actual data so drift.py can read it later
    ref_data_path = settings.REFERENCE_DATA_PATH
    ref_data_path .parent.mkdir(parents=True, exist_ok=True)
    reference_df.to_csv(ref_data_path, index=False)
    
    # Get proxy drift baseline
    save_reference_profile(reference_df)

    # Extract key metrics for MLflow Table
    with open(settings.REFERENCE_PROFILE_PATH, "r") as f:
        report_data = json.load(f)

    summary_metrics = []
    
    for metric in report_data.get("metrics", []):
        name = metric.get("metric")
        result = metric.get("result", {})
        
        # Extract Drift Summary
        if name == "DatasetDriftMetric":
            summary_metrics.append({
                "category": "Data Drift",
                "metric_name": "Dataset Drift Detected",
                "value": str(result.get("dataset_drift")),
                "details": f"Drifted: {result.get('number_of_drifted_columns')}/{result.get('number_of_columns')}"
            })
            
        # Extract Column Specific Drift (e.g., 'rate')
        elif name == "DataDriftTable":
            drift_cols = result.get("drift_by_columns", {})
            for col, data in drift_cols.items():
                summary_metrics.append({
                    "category": "Data Drift",
                    "metric_name": f"Column Drift: {col}",
                    "value": str(data.get("drift_detected")),
                    "details": f"Score: {data.get('drift_score'):.4f} ({data.get('stattest_name')})"
                })

        # Extract Regression Quality
        elif name == "RegressionQualityMetric":
            curr = result.get("current", {})
            for m_name in ["mae", "rmse", "r2_score"]:
                val = curr.get(m_name) if m_name != "mae" else curr.get("mean_abs_error")
                summary_metrics.append({
                    "category": "Regression Quality",
                    "metric_name": m_name.upper(),
                    "value": f"{val:.4f}" if val is not None else "N/A",
                    "details": "Reference Baseline"
                })

    # Convert to DataFrame for MLflow
    summary_df = pd.DataFrame(summary_metrics)
    
    # Log the summary table to the active MLflow run
    mlflow.log_table(data=summary_df, artifact_file="monitoring/reference_profile.json")

    return str(settings.REFERENCE_PROFILE_PATH)

def save_data_scope(df: pd.DataFrame) -> dict:
    """
    Calculates and persists the temporal scope of the training data.
    Used to distinguish between historical training data and new prediction data.
    """
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    scope = {
        "start_date": df["date"].min().strftime("%Y-%m-%d"),
        "end_date": df["date"].max().strftime("%Y-%m-%d"),
        "total_weeks": len(df),
        "last_updated": datetime.now(timezone.utc).isoformat()
    }
    
    scope_path = RAW_DATA_DIR / "data_scope.json"
    with open(scope_path, "w") as f:
        json.dump(scope, f, indent=4)
        
    logger.info(f"Data scope saved: {scope['start_date']} to {scope['end_date']}")
    return scope

def load_data_scope() -> dict:
    """
    Retrieve data_scope.json for the currently staged model
    without requiring a run_id.
    """
    _setup_mlflow()
    client = MlflowClient()

    # get staged model version and run id
    model_version = client.get_model_version_by_alias(
        name=settings.MODEL_NAME,
        alias="staging",
    )
    run_id = model_version.run_id

    logger.info(
        f"Fetching data_scope.json from run_id={run_id} "
        f"(model v{model_version.version})"
    )

    data_scope_dir = settings.RAW_DATA_DIR / "data_scope.json"
    data_scope_dir.parent.mkdir(parents=True, exist_ok=True) # ensure local directory exists

    # download artifact
    client.download_artifacts(
        run_id=run_id,
        path="data_scope/data_scope.json",
        dst_path=str(settings.RAW_DATA_DIR),
    )

    # load and return data scope
    with open(data_scope_dir) as f:
        return json.load(f)

if __name__ == "__main__":
    print(load_data_scope())