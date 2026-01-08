# src/monitoring/drift.py

import pandas as pd
import json
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently import ColumnMapping
from src.config.settings import Settings

settings = Settings()

def run_drift_check(live_df: pd.DataFrame):
    ref_path = settings.MONITORING_DIR / "reference_profile.json"

    if not ref_path.exists():
        raise FileNotFoundError("Reference profile not found")

    report = Report(metrics=[
        DataDriftPreset(),
        RegressionPreset()
    ])

    report.run(
        reference_data=pd.read_json(ref_path),
        current_data=live_df
    )

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]

    return drift_detected, result
