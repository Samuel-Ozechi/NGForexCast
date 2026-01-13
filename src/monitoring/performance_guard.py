# src/monitoring/performance_guard.py

from src.monitoring.drift import run_drift_check


DRIFT_THRESHOLD = 0.4

def should_retrain(live_df):
    drift = run_drift_check(live_df)

    if drift["concept_drift"]:
        return True, "concept_drift", drift

    if drift["prediction_drift"] and drift["target_drift"]:
        return True, "label_behavior_shift", drift

    if drift["dataset_drift"] and drift["feature_drift"]:
        return True, "population_shift", drift

    return False, "healthy", drift


if __name__ == "__main__":
    import pandas as pd
    import joblib
    from src.utils.utils import get_prediction_data, get_predictions
    from src.utils.utils import get_prediction_data
    from src.config.settings import Settings

    settings = Settings()

    # Fetch recent data for drift check
    recent_data = get_prediction_data()  
    print(f"Fetched {recent_data.shape[0]} records for drift check.")

    # Load inference pipeline
    print("Loading inference pipeline...")
    pipeline = joblib.load(settings.MODEL_PATH)

    print("Generating live predictions...")
    # Get live predictions
    live_df = get_predictions(recent_data, pipeline)
    print("Live predictions generated:", live_df.shape)

    retrain, reason, drift_report = should_retrain(live_df)

    if retrain:
        print(f"Retraining triggered due to: {reason}")
    else:
        print("No retraining needed. Model is healthy.")

    # print("Drift Report:", drift_report)