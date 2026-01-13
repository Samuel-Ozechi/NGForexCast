# src/monitoring/drift.py

import joblib
import pandas as pd
import json
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently import ColumnMapping
from src.config.settings import Settings
from src.utils.utils import get_prediction_data, get_predictions
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = Settings()   

def run_drift_check():
    logger.info("Starting drift check...")

    logger.info("Loading reference data profile...")
    ref_path = settings.MONITORING_DIR / "reference_data.csv"
    if not ref_path.exists():
        raise FileNotFoundError("Reference profile not found")
    
    logger.info("Loading prediction data...")
    live_df = get_prediction_data()  
    logger.info("Generating live predictions...")

    pipeline = joblib.load(settings.MODEL_PATH)
    live_df = get_predictions(live_df, pipeline)
    logger.info("Live predictions generated.")

    column_mapping = ColumnMapping(
        target="rate",
        prediction="prediction",
        numerical_features=["rate"],
        categorical_features=[]
    )

    report = Report(metrics=[
        DataDriftPreset(),
        RegressionPreset()
    ])

    report.run(
        reference_data=pd.read_csv(ref_path),
        current_data=live_df,
        column_mapping=column_mapping   
    )

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
    print(json.dumps(drift_report, indent=4))

