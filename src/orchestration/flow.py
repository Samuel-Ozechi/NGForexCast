# src/orchestration/flows.py

from prefect import flow
from src.orchestration.tasks import (
     run_forecast, run_monitor, retrain
)

@flow(name="daily_forecast_flow")
def daily_forecast_flow(days: int = 1):
    forecasts = run_forecast(days)
    return forecasts

@flow(name="monitor_flow")
def monitor_flow():
    retrain_flag, reason, drift = run_monitor()
    if retrain_flag:
        retrain_flow(reason)

@flow(name="retrain_flow")
def retrain_flow(reason: str):
    print(f"⚠️ Retraining triggered: {reason}")
    retrain()

@flow(name="autonomous_forex_ai")
def autonomous_forex_ai(days: int = 1):
    # # Generate daily forecasts
    # daily_forecast_flow(days)

    # Check health and retrain if necessary
    monitor_flow()
