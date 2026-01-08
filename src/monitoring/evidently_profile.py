# src/monitoring/evidently_profile.py

import pandas as pd
import joblib

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently import ColumnMapping

from src.config.settings import Settings

settings = Settings()

def save_reference_profile(train_df: pd.DataFrame):
    column_mapping = ColumnMapping(
        target="rate",
        prediction="rate",
        numerical_features=["rate"],
        categorical_features=[]
    )

    report = Report(metrics=[
        DataDriftPreset(),
        RegressionPreset()
    ])

    report.run(
        reference_data=train_df,
        current_data=train_df,
        column_mapping=column_mapping
    )

    profile_path = settings.REFERENCE_PROFILE_PATH
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_json(profile_path)

    return profile_path


if __name__ == "__main__":
    # Example usage
    train_data_path = settings.RAW_DATA_DIR / "train_data.csv"
    train_df = pd.read_csv(train_data_path)

    profile_path = save_reference_profile(train_df)
    print(f"Reference profile saved at: {profile_path}")