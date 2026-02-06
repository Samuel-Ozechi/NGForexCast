# src/api/predict.py

import os
import logging
from datetime import timedelta

import pandas as pd
import mlflow
import dagshub
from mlflow.tracking import MlflowClient

from src.config.settings import Settings
from src.data.ingest import fetch_exchange_rates

# logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class ForexInference:
    def __init__(self):
        self.settings = Settings()
        self.model_name = "ngn_us_exchange_model"
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Configure connection to DagsHub MLflow Remote"""

        # Initialize DagsHub MLflow integration
        token = os.environ.get("DAGSHUB_USER_TOKEN")
        dagshub.auth.add_app_token(token)

        dagshub.init(
        repo_owner=os.getenv("DAGSHUB_USER"),
        repo_name="NGForexCast",
        mlflow=True,
    )
        
        # MLflow experiment setup
        settings = Settings()
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        
        print("Tracking URI:", mlflow.get_tracking_uri())
        print("Experiments:", mlflow.search_experiments())

        logger.info("MLflow tracking configured")

    def _load_latest_staging_model(self):
        """
        Load the latest model pointed to by the 'staging' alias.
        This guarantees we always use the current promoted model.
        """
        model_uri = f"models:/{self.model_name}@staging"
        logger.info(f"Loading model from MLflow Registry: {model_uri}")

        try:
            return mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            logger.exception("Failed to load staging model")
            raise RuntimeError(
                f"Could not load model '{self.model_name}' because of error: {str(e)}"
            ) from e


    def recursive_forecast(self, steps: int = 12):
        """
        Fully recursive autoregressive multi-step forecast.
        Always uses the latest staged model at call time.
        """
        # Load model at call time (not at init)
        pipeline = self._load_latest_staging_model()

        logger.info("Loading raw data from database")
        history = fetch_exchange_rates().copy()
        history["date"] = pd.to_datetime(history["date"])
        history = history.sort_values("date")

        forecasts = []

        logger.info(f"Starting recursive forecasting for {steps} steps")
        for i in range(steps):
            # Predict next step
            y_next = pipeline.predict(history)[-1]

            next_date = history["date"].max() + timedelta(days=7)

            history = pd.concat(
                [
                    history,
                    pd.DataFrame({"date": [next_date], "rate": [y_next]})
                ],
                ignore_index=True
            )

            forecasts.append({
                "date": next_date.strftime("%Y-%m-%d"),
                "prediction": float(y_next)
            })

            logger.info(
                f"Step {i + 1}/{steps}: Predicted {y_next:.6f} for {next_date.date()}"
            )

        logger.info("Recursive forecasting completed")
        return forecasts


if __name__ == "__main__":
    inferencer = ForexInference()
    results = inferencer.recursive_forecast(steps=12)
    print(results)



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