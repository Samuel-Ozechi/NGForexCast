# src/features/transform.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
import logging

logger = logging.getLogger(__name__)


class TimeSeriesFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible transformer that performs:
    - Weekly resampling (W-FRI)
    - Differencing
    - Lag features
    - Rolling statistics
    - Returns over multiple periods
    - Date-based calendar features

    Output matches the engineered dataset created by ETL.
    """

    def __init__(
        self, 
        lag_list: List[int] = None,
        rolling_windows: List[int] = None,
        return_diffs: List[int] = None,
    ):
        self.lag_list = lag_list or [1, 2, 6, 8, 11, 13, 15, 16, 25, 32, 34]
        self.rolling_windows = rolling_windows or [4, 8, 12]
        self.return_diffs = return_diffs or [3, 7, 14, 21, 30, 60, 90]

    def fit(self, X: pd.DataFrame, y=None):
        """
        Nothing to fit — transformation is deterministic.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a DataFrame containing at least:
        - "date"
        - "rate"

        Returns a fully-feature engineered DataFrame.
        """

        df = X.copy()

        if "date" not in df.columns or "rate" not in df.columns:
            raise ValueError("Input DataFrame must contain 'date' and 'rate' columns.")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df = df.set_index("date")

        # --- Weekly Resampling ---
        df_weekly = df["rate"].resample("W-FRI").last().to_frame()

        # --- Differenced series ---
        df_feat = df_weekly.copy()
        df_feat["diff"] = df_feat["rate"].diff()

        # --- Lag Features ---
        for lag in self.lag_list:
            df_feat[f"lag_{lag}"] = df_feat["diff"].shift(lag)

        # --- Rolling Means & STD ---
        for window in self.rolling_windows:
            df_feat[f"rmean_{window}"] = (
                df_feat["diff"].rolling(window).mean().shift(1)
            )
            df_feat[f"rstd_{window}"] = (
                df_feat["diff"].rolling(window).std().shift(1)
            )

        # --- Returns / Percent Change ---
        for diff in self.return_diffs:
            df_feat[f"ret_{diff}"] = df_feat["diff"].ffill().pct_change(periods=diff, fill_method=None)

        # Reset index for calendar features
        df_feat = df_feat.reset_index()

        # --- Calendar Features ---
        df_feat["month"] = df_feat["date"].dt.month
        df_feat["day"] = df_feat["date"].dt.day
        df_feat["year"] = df_feat["date"].dt.year

        # Replace infinities
        df_feat.replace([np.inf, -np.inf], 0, inplace=True)

        # Drop rows with NaNs introduced by lag/rolling
        df_feat = df_feat.dropna().reset_index(drop=True)

        logger.info("Feature engineering complete. Sample:")
        logger.info(df_feat.tail(2))

        return df_feat