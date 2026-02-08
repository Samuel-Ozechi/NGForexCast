# src/monitoring/drift.py

import joblib
import pandas as pd
import json
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently import ColumnMapping
from src.config.settings import Settings
from src.data.ingest import fetch_exchange_rates
from src.utils.utils import  get_predictions, load_data_scope
from src.model.predict import _load_mlflow_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = Settings()   

def run_drift_check():
    logger.info("Starting drift check...")
    scope = load_data_scope()
    train_end_date = pd.to_datetime(scope["end_date"])
    pipeline = _load_mlflow_model()

    # get reference data (same period as training data)
    logger.info("Loading reference data...")
    train_data = fetch_exchange_rates(end_date=train_end_date.strftime("%Y-%m-%d"))
    reference_data = get_predictions(train_data, pipeline)
    print(f"length of reference data {len(train_data)}")

    # get current data (new data since training period)
    logger.info("Loading current data...")
    current_start_date = train_end_date + pd.Timedelta(days=1)
    current_data = fetch_exchange_rates(start_date=current_start_date.strftime("%Y-%m-%d"))
    current_data = get_predictions(current_data, pipeline)
    print(f"length of current data {len(current_data)}")

    # Prepare data for evidently
    column_mapping = ColumnMapping(
        target="rate",
        prediction="prediction",
        numerical_features=["rate"],
        categorical_features=[]
    )

    # Create and run evidently report
    report = Report(metrics=[
        DataDriftPreset(),
        RegressionPreset()
    ])

    try:
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
    except Exception as e:
        logger.error(f"Evidently failed to compute drift: {e}")
        return {
            "status": "failed",
            "reason": "evidently_runtime_error",
            "error": str(e),
        }
        
    # Extract drift results
    result = report.as_dict()
    metrics = {m["metric"]: m["result"] for m in result["metrics"]}

    return {
        "dataset_drift": metrics["DatasetDriftMetric"]["dataset_drift"],
        "feature_drift": any(c["drift_detected"] for c in metrics["DataDriftTable"]["drift_by_columns"].values()),
        "prediction_drift": metrics.get("PredictionDriftMetric", {}).get("drift_detected", False),
        "target_drift": metrics.get("TargetDriftMetric", {}).get("drift_detected", False),
        "concept_drift": metrics.get("ResidualsDriftMetric", {}).get("drift_detected", False),
        "raw_report": result
    }


if __name__ == "__main__":
    # Run drift check
    drift_report = run_drift_check()
    # print(json.dumps(drift_report, indent=4))

