# src/monitoring/evidently_profile.py

import pandas as pd
import joblib
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from src.config.settings import Settings

settings = Settings()

def save_reference_profile(train_df: pd.DataFrame):
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset()
    ])
    report.run(reference_data=train_df, current_data=train_df)

    profile_path = settings.MONITORING_DIR / "reference_profile.json"
    report.save_json(profile_path)

    return profile_path
