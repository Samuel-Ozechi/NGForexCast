# src/orchestration/flows.py

from prefect import flow
from src.orchestration.tasks import (
     run_monitor, retrain
)


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
def autonomous_forex_ai():
    # Check health and retrain if necessary
    monitor_flow()
