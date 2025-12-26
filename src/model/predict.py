# src/api/predict.py

import joblib
import pandas as pd
from datetime import timedelta
import logging
from src.config.settings import Settings
from src.data.ingest import fetch_exchange_rates

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
settings = Settings()


class ForexInference:

    def __init__(self):
        self.pipeline = joblib.load(settings.MODEL_PATH)

    def recursive_forecast(self, steps: int = 12):
        """
        Fully recursive autoregressive multi-step forecast.
        Returns list of {"date":..., "prediction":...}
        """

        # Load full historical series
        logger.info(f"Loading raw data from database")
        history = fetch_exchange_rates().copy()
        history["date"] = pd.to_datetime(history["date"])
        history = history.sort_values("date")

        forecasts = []

        logger.info(f"Starting recursive forecasting for {steps} steps")
        for i in range(steps):

            # Predict next step
            y_next = self.pipeline.predict(history)[-1]

            # Compute next timestamp
            next_date = history["date"].max() + timedelta(days=7)

            # Append back into history (model becomes its own future)
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
            logger.info(f"Step {i+1}/{steps}: Predicted {y_next} for date {next_date.strftime('%Y-%m-%d')}")
        logger.info("Recursive forecasting completed")
        return forecasts
    

if __name__ == "__main__":
    inferencer = ForexInference()
    results = inferencer.recursive_forecast(steps=12)
    print(results)