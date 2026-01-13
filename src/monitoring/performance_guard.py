# src/monitoring/performance_guard.py

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from src.monitoring.drift import run_drift_check


def should_retrain():
    drift = run_drift_check()

    if drift["concept_drift"]:
        return True, "concept_drift", drift

    if drift["prediction_drift"] and drift["target_drift"]:
        return True, "label_behavior_shift", drift

    if drift["dataset_drift"] and drift["feature_drift"]:
        return True, "population_shift", drift

    return False, "healthy", drift


if __name__ == "__main__":

    retrain, reason, drift_report = should_retrain()

    if retrain:
        print(f"Retraining triggered due to: {reason}")
    else:
        print("No retraining needed. Model is healthy.")

    # print("Drift Report:", drift_report)