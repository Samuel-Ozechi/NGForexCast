# src/orchestration/tasks.py
from prefect import task
from src.monitoring.performance_guard import should_retrain
from src.model.train import run_train
from src.api.app import forecast_days

@task
def run_forecast(days: int):
    return forecast_days(days)

@task
def run_monitor():
    return should_retrain()

@task
def retrain():
    return run_train()