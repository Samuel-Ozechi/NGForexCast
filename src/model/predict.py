# src/api/predict.py

import os
import json
import logging
from datetime import timedelta, datetime, timezone
from pathlib import Path

import pandas as pd
import mlflow
import dagshub
import joblib
from mlflow.tracking import MlflowClient

from src.config.settings import Settings
from src.data.ingest import fetch_exchange_rates
from src.utils.utils import _setup_mlflow


# logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class ForexInference:
    def __init__(self):
        self.settings = Settings()
        self.model_name = self.settings.MODEL_NAME

        self.artifact_dir = self.settings.PROD_PATH
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.local_model_path = self.settings.MODEL_PATH
        self.meta_path = self.settings.META_PATH

        _setup_mlflow()
        self.client = MlflowClient()


    # # MLflow setup
    # def _setup_mlflow(self):
    #     token = os.environ.get("DAGSHUB_USER_TOKEN")
    #     dagshub.auth.add_app_token(token)

    #     dagshub.init(
    #         repo_owner=os.getenv("DAGSHUB_USER"),
    #         repo_name="NGForexCast",
    #         mlflow=True,
    #     )

    #     mlflow.set_tracking_uri(self.settings.MLFLOW_TRACKING_URI)
    #     logger.info("MLflow tracking configured")


    # Model version helpers
    def _get_staged_model_version(self) -> str:
        """Return MLflow version currently pointed to by 'staging' alias."""
        mv = self.client.get_model_version_by_alias(
            name=self.model_name,
            alias="staging",
        )
        return mv.version

    def _get_local_model_version(self) -> str:
        if not self.meta_path.exists():
            return None

        with open(self.meta_path, "r") as f:
            meta = json.load(f)

        return meta.get("mlflow_version")

    def _save_model_meta(self, model_version):
    

        meta = {
            "model_name": self.model_name,
            "mlflow_version": model_version,
        }

        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved model metadata: {meta}")

    
    # version-aware model loading
    def _load_pipeline(self):
        staged_version = self._get_staged_model_version()
        local_version = self._get_local_model_version()

        logger.info(f"Staged model version: {staged_version}")
        logger.info(f"Local model version: {local_version}")

        # Case 1: Local model exists and is up-to-date
        if (
            self.local_model_path.exists()
            and local_version == staged_version
        ):
            logger.info("Using cached local inference pipeline")
            return joblib.load(self.local_model_path)

        # Case 2: Version mismatch or missing local model → download
        logger.info("Local model missing or outdated — downloading from MLflow")

        model_uri = f"models:/{self.model_name}@staging"
        pipeline = mlflow.sklearn.load_model(model_uri)

        joblib.dump(pipeline, self.local_model_path)
        self._save_model_meta(staged_version)

        logger.info(
            f"Downloaded and cached model version {staged_version}"
        )

        return pipeline


    # Prediction API
    def recursive_forecast(self, steps: int = 12):
        """
        Recursive autoregressive multi-step forecast.
        Uses version-aware cached inference pipeline.
        """
        pipeline = self._load_pipeline()

        logger.info("Loading raw data from database")
        history = fetch_exchange_rates().copy()
        history["date"] = pd.to_datetime(history["date"])
        history = history.sort_values("date")

        forecasts = []

        logger.info(f"Starting recursive forecasting for {steps} steps")

        for i in range(steps):
            y_next = pipeline.predict(history)[-1]
            next_date = history["date"].max() + timedelta(days=7)

            history = pd.concat(
                [
                    history,
                    pd.DataFrame(
                        {"date": [next_date], "rate": [y_next]}
                    ),
                ],
                ignore_index=True,
            )

            forecasts.append(
                {
                    "date": next_date.strftime("%Y-%m-%d"),
                    "prediction": float(y_next),
                }
            )

            logger.info(
                f"Step {i + 1}/{steps}: {y_next:.6f} @ {next_date.date()}"
            )

        logger.info("Recursive forecasting completed")
        return forecasts


if __name__ == "__main__":
    inferencer = ForexInference()
    print(inferencer.recursive_forecast(steps=4))


### Prediction code without MLflow but with local model (for reference, not used in final version)   
# import joblib
# import pandas as pd
# from datetime import timedelta
# import logging
# from src.config.settings import Settings
# from src.data.ingest import fetch_exchange_rates

# logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
# logger = logging.getLogger(__name__)
# settings = Settings()


# class ForexInference:

#     def __init__(self):
#         self.pipeline = joblib.load(settings.MODEL_PATH)

#     def recursive_forecast(self, steps: int = 12):
#         """
#         Fully recursive autoregressive multi-step forecast.
#         Returns list of {"date":..., "prediction":...}
#         """

#         # Load full historical series
#         logger.info(f"Loading raw data from database")
#         history = fetch_exchange_rates().copy()
#         history["date"] = pd.to_datetime(history["date"])
#         history = history.sort_values("date")

#         forecasts = []

#         logger.info(f"Starting recursive forecasting for {steps} steps")
#         for i in range(steps):

#             # Predict next step
#             y_next = self.pipeline.predict(history)[-1]

#             # Compute next timestamp
#             next_date = history["date"].max() + timedelta(days=7)

#             # Append back into history (model becomes its own future)
#             history = pd.concat(
#                 [
#                     history,
#                     pd.DataFrame({"date": [next_date], "rate": [y_next]})
#                 ],
#                 ignore_index=True
#             )

#             forecasts.append({
#                 "date": next_date.strftime("%Y-%m-%d"),
#                 "prediction": float(y_next)
#             })
#             logger.info(f"Step {i+1}/{steps}: Predicted {y_next} for date {next_date.strftime('%Y-%m-%d')}")
#         logger.info("Recursive forecasting completed")
#         return forecasts
    

# if __name__ == "__main__":
#     inferencer = ForexInference()
#     results = inferencer.recursive_forecast(steps=12)
#     print(results)