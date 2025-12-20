# src/model/train.py

"""
Training pipeline.

- Loads data using src/data/ingest.py..
- Builds a preprocessing pipeline (ColumnTransformer + StandardScaler) using src/features/transform.py.
- Runs model/hyperparameter search using TimeSeriesSplit CV.
- Logs runs, params, metrics, artifacts to MLflow.
- Registers best model in MLflow Model Registry and saves an inference pipeline artifact.
"""

import os
import joblib
import json
import tempfile
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import mlflow
import mlflow.sklearn

# local imports
from src.config.settings import settings
from src.data.ingest import fetch_exchange_rates
from src.features.transform import TimeSeriesFeatureEngineer
from src.utils.utils import evaluate_metrics, plot_predictions


# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_train():
    experiment_name = settings.MLFLOW_EXPERIMENT_NAME
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow tracking uri: {settings.MLFLOW_TRACKING_URI}. Experiment: {experiment_name}")

    # 1) Load input data from DB
    logger.info(f"Loading raw data from database")
    df = fetch_exchange_rates()
    # Expect data to have 'date' and 'rate' columns
    if "date" not in df.columns:
        raise ValueError("input data must include 'date' column")
    if target_col not in df.columns:
        raise ValueError(f"input data must include target column '{target_col}'")

    # 2) Build feature engineering transformer (sklearn transformer)
    feature_engineer = TimeSeriesFeatureEngineer()

    # 3) Create dataframe of engineered features (offline)
    logger.info("Generating engineered features (offline)")
    df_feat = feature_engineer.transform(df)  # df_feat contains 'date' and 'rate' and engineered features
    # normalize column names to lower
    df_feat.columns = [c.lower() for c in df_feat.columns]

    # 4) Time-series train/test split (index by date)
    df_feat = df_feat.sort_values("date").reset_index(drop=True)
    df_feat["date"] = pd.to_datetime(df_feat["date"])
    df_feat = df_feat.set_index("date")
    total = len(df_feat)
    test_fraction = settings.TEST_SIZE
    test_size = int(total * test_fraction)
    if test_size < 1:
        test_size = 1
    train_df = df_feat.iloc[:-test_size].copy()
    test_df = df_feat.iloc[-test_size:].copy()

    logger.info(f"Train range: {train_df.index.min().date()} to {train_df.index.max().date()}")
    logger.info(f"Test range: {test_df.index.min().date()} to {test_df.index.max().date()}")
    logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # 5) Prepare X, y  
    feature_cols = [c for c in df_feat.columns if c not in [target_col.lower()]]
    target_col= settings.TARGET_COLUMN.lower()
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col.lower()].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col.lower()].copy()

    # 6) Build preprocessing ColumnTransformer
    # All feature columns are numeric in our scenario. Use StandardScaler on all numeric features.
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="passthrough"
    )

    # 7) Model candidates and param grids (small, extendable)
    random_state = settings.RANDOM_STATE   
    models_and_grids = [
        ("LinearRegression", LinearRegression(), {}),
        ("Ridge", Ridge(), {"alpha": [0.1, 1.0, 10.0, 50.0]}),
        ("RandomForest", RandomForestRegressor(random_state=random_state), {
            "n_estimators": [100, 200], "max_depth": [5, 10], "min_samples_split": [2, 5]
        }),
        ("GradientBoosting", GradientBoostingRegressor(random_state=random_state), {
            "n_estimators": [100, 200], "learning_rate": [0.01, 0.05], "max_depth": [3, 5]
        }),
        ("ExtraTrees", ExtraTreesRegressor(random_state=random_state), {
            "n_estimators": [100, 200], "max_depth": [5, 10], "min_samples_split": [2, 5]
        }),
        ("XGBoost", XGBRegressor(random_state=random_state, objective="reg:squarederror"), {
            "n_estimators": [100, 200], "learning_rate": [0.01, 0.05], "max_depth": [3, 5]
        }),
        ("LightGBM", LGBMRegressor(random_state=random_state), {
            "n_estimators": [100, 200], "learning_rate": [0.01, 0.05], "max_depth": [3, 5]
        }),
        ("CatBoost", CatBoostRegressor(random_state=random_state, verbose=0), {
            "iterations": [100, 200], "learning_rate": [0.01, 0.05], "depth": [3, 5]
        }),
    ]

    # 8) CV strategy
    cv = TimeSeriesSplit(n_splits=5)

    # 9) Iterate models, grid search, log to MLflow
    model_results = []
    best_overall = None
    best_metric = float("inf")  # lower is better for MAE

    for name, model_obj, param_grid in models_and_grids:
        logger.info(f"Starting training candidate: {name}")
        # pipeline: preprocessing + model
        pipeline = Pipeline([
            ("preproc", preproc),
            ("model", model_obj)
        ])

        # If no params to search, wrap with a trivial grid of empty dict
        if param_grid:
            # convert param grid to pipeline param names: model__param
            grid = {f"model__{k}": v for k, v in param_grid.items()}
        else:
            grid = {}

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid or [{}],
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
            verbose=0,
            refit=True
        )

        search.fit(X_train, y_train)
        logger.info(f"{name} best params: {search.best_params_}")

        # Evaluate on test set
        y_pred = search.predict(X_test)
        metrics = evaluate_metrics(y_test.values, y_pred)
        logger.info(f"{name} test metrics: {metrics}")

        # MLflow logging per candidate
        with mlflow.start_run(run_name=f"{name}_{datetime.utcnow().isoformat()}"):
            mlflow.log_param("model_name", name)
            # Log best params (transform keys to remove model__)
            param_log = {k.replace("model__", ""): v for k, v in (search.best_params_ or {}).items()}
            for k, v in param_log.items():
                mlflow.log_param(k, v)
            # log metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            # log model artifact (pipeline with fitted preproc + model)
            # Save locally
            tmp_model_path = os.path.join(tempfile.gettempdir(), f"{name}_pipeline.joblib")
            joblib.dump(search.best_estimator_, tmp_model_path)
            mlflow.log_artifact(tmp_model_path, artifact_path="models")
            # also log a small JSON of metrics
            mlflow.log_dict(metrics, "metrics.json")

        model_results.append({
            "name": name,
            "best_estimator": search.best_estimator_,
            "metrics": metrics,
            "best_params": search.best_params_,
        })

        # Update best overall by MAE
        if metrics["MAE"] < best_metric:
            best_metric = metrics["MAE"]
            best_overall = {
                "name": name,
                "estimator": search.best_estimator_,
                "metrics": metrics,
                "params": search.best_params_,
            }

    # 10) Register best model in MLflow Model Registry and save final inference pipeline
    if best_overall is None:
        raise RuntimeError("No model was trained successfully.")

    best_name = best_overall["name"]
    best_estimator = best_overall["estimator"]
    best_metrics = best_overall["metrics"]
    logger.info(f"Best overall model: {best_name} with MAE={best_metrics['MAE']:.6f}")

    # Save final inference pipeline (we'll include the feature-engineer + preproc + model)
    # Compose full inference pipeline that accepts raw cleaned dataframe (date, rate)
    full_inference_pipeline = Pipeline([
        ("feat_engineer", TimeSeriesFeatureEngineer()),
        ("preproc", preproc),
        ("model", best_estimator.named_steps["model"])  # extracted model
    ])

    # Fit the full inference pipeline on the full training+test data for production
    X_full = df_feat.reset_index(drop=False)  # date column included; feat transformer expects date & rate
    # We will fit the preproc+model portion with engineered X (pipeline already has fit components for preproc and model)
    # To keep things consistent, transform X_full via feat_engineer, then fit preproc+model
    engineered = full_inference_pipeline.named_steps["feat_engineer"].transform(X_full)
    engineered_cols = [c.lower() for c in engineered.columns]
    X_engineered = engineered.drop(columns=[target_col.lower()]) if target_col.lower() in engineered.columns else engineered
    y_engineered = engineered[target_col.lower()] if target_col.lower() in engineered.columns else None

    # fit preprocessing and model on engineered (we need to fit preproc and model)
    # We will create a final pipeline without feat_engineer (because we store feat_engineer separately in MLflow model signature)
    final_preproc_and_model = Pipeline([
        ("preproc", preproc),
        ("model", best_estimator.named_steps["model"])
    ])
    if y_engineered is None:
        raise RuntimeError("Target column missing after engineering.")
    final_preproc_and_model.fit(X_engineered, y_engineered)

    # Combine feat_engineer + preproc+model for a reusable inference object
    inference_pipeline = Pipeline([
        ("feat_engineer", TimeSeriesFeatureEngineer()),
        ("preproc_and_model", final_preproc_and_model)
    ])

    # Save artifact locally and log to MLflow as a registered model
    artifact_path = os.path.join(tempfile.gettempdir(), f"inference_pipeline_{datetime.utcnow().isoformat()}.joblib")
    joblib.dump(inference_pipeline, artifact_path)

    # Start a new MLflow run to register final model
    with mlflow.start_run(run_name=f"best_model_register_{datetime.utcnow().isoformat()}") as run:
        mlflow.log_param("best_model_name", best_name)
        for k, v in best_metrics.items():
            mlflow.log_metric(k, v)
        # log pipeline artifact
        mlflow.log_artifact(artifact_path, artifact_path="inference_pipeline")
        # log metadata
        mlflow.log_dict({"model_name": best_name, "params": best_overall["params"]}, "model_meta.json")

        # Register model in Model Registry
        # mlflow.sklearn.log_model expects an sklearn model/pipeline object; we register the preproc+model as sklearn model
        # We will log the final_preproc_and_model and register that (feat_engineer is separate; but we include it by saving the full joblib)
        mlflow.sklearn.log_model(final_preproc_and_model, artifact_path="sklearn_model",
                                 registered_model_name="ngn_us_exchange_model")

        # Optional: promote to 'Staging'
        client = mlflow.tracking.MlflowClient()
        # get latest version of registered model
        versions = client.get_latest_versions("ngn_us_exchange_model")
        if versions:
            latest = versions[-1]
            client.transition_model_version_stage(
                name="ngn_us_exchange_model",
                version=latest.version,
                stage="Staging",
                archive_existing_versions=False
            )
        logger.info("Registered model in MLflow Model Registry as 'ngn_us_exchange_model'")

    # 11) Plot test predictions vs actual for best model and log
    y_best_pred = best_overall["estimator"].predict(X_test)
    plot_path = os.path.join(tempfile.gettempdir(), f"pred_vs_actual_{datetime.utcnow().isoformat()}.png")
    plot_predictions(X_test.index, y_test.values, y_best_pred, f"{best_name} Predictions", plot_path)
    # Log plot to MLflow (attach to the registration run)
    mlflow.log_artifact(plot_path, artifact_path="plots")

    # 12) Save summary results locally too
    results_summary = {
        "best_model": best_name,
        "best_metrics": best_metrics,
        "timestamp": datetime.utcnow().isoformat()
    }
    summary_path = os.path.join(settings.MODELS_DIR, "latest_training_summary.json")
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"Saved results summary to {summary_path}")

    return best_overall


if __name__ == "__main__":
    # entrypoint: change the path to your processed CSV
    processed_csv_path = settings.PROCESSED_DATA_DIR / "ngn_us_exchange_rates_cleaned.csv"
    run_train(str(processed_csv_path))
