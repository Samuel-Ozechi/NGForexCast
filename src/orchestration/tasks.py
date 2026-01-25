# src/orchestration/tasks.py
from prefect import task
from src.monitoring.performance_guard import should_retrain
from src.model.train import run_train

@task
def run_monitor():
    return should_retrain()

@task
def retrain():
    return run_train()