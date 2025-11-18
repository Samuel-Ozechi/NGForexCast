# src/config/settings.py

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field
from dotenv import load_dotenv

# load variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """
    Centralized configuration for the ML system.
    Loads values from .env and environment variables.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore" 
    )

    # --- Project paths ---
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "01_raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "02_processed"
    FEATURE_DATA_DIR: Path = DATA_DIR / "03_features"
    MODELS_DIR: Path = BASE_DIR / "models"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # --- Database & APIs ---
    SUPABASE_DB_URL: str = Field(...)
    EXCHANGE_RATE_API: str = Field(...)
    
    # Pydantic computed_field property to build the exchange rate API URL dynamically
    @computed_field
    @property
    def EXCHANGE_RATE_API_URL(self) -> str:
        """Public exchange rate API endpoint constructed with the loaded key."""
        return f"https://v6.exchangerate-api.com/v6/{self.EXCHANGE_RATE_API}/latest/USD"

    # --- Data settings ---
    FETCH_INTERVAL_HOURS: int = 24
    TARGET_COLUMN: str = "Rate"
    DATE_COLUMN: str = "Date"

    # --- Model training ---
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    SCORING_METRIC: str = "MAPE"
    CROSS_VALIDATION_FOLDS: int = 5
    ENABLE_HYPERPARAM_TUNING: bool = True

    # --- Drift detection ---
    DRIFT_THRESHOLD: float = 0.1   # Threshold for triggering retraining
    DRIFT_METHOD: str = "evidently"  # or "ks_test", "psi", "adwin"

    # --- MLflow / Experiment Tracking ---
    USE_MLFLOW: bool = True
    MLFLOW_TRACKING_URI: str = Field(
        default="http://localhost:5000", 
        description="MLflow server URL or local path"
    )
    MLFLOW_EXPERIMENT_NAME: str = "currency_rate_prediction"

    # --- Model serving / API ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    MODEL_VERSION: str = "v1.0.0"

    # --- Monitoring ---
    ENABLE_LOGGING: bool = True
    LOG_LEVEL: str = "INFO"
    ENABLE_DRIFT_MONITORING: bool = True

if __name__ == "__main__":
    # Global instance to import anywhere
    settings = Settings()
    # print(settings.EXCHANGE_RATE_API_URL)