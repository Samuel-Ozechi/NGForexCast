# src/monitoring/performance_guard.py

from src.monitoring.drift import run_drift_check

DRIFT_THRESHOLD = 0.4

def should_retrain(live_df):
    drifted, report = run_drift_check(live_df)

    if drifted:
        return True, report
    return False, report
