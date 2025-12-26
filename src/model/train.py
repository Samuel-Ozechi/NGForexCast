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
import shutil
import joblib
import json
import tempfile
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import dagshub
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# local imports
from src.config.settings import Settings
from src.data.ingest import fetch_exchange_rates
from src.features.transform import TimeSeriesFeatureEngineer
from src.utils.utils import evaluate_metrics, plot_predictions, promote_to_production

sklearn.set_config(transform_output="pandas")

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Initialize DagsHub MLflow integration
dagshub.init(repo_owner='Chiebukar', repo_name='NGForexCast', mlflow=True)

def run_train():
    settings = Settings()
    experiment_name = settings.MLFLOW_EXPERIMENT_NAME
    mlflow.set_experiment(experiment_name)
    # mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking uri: {settings.MLFLOW_TRACKING_URI}. Experiment: {experiment_name}")

    # 1) Load input data from DB
    logger.info(f"Loading raw data from database")
    df = fetch_exchange_rates()
    target_col= settings.TARGET_COLUMN.lower()
    # Expect data to have 'date' and 'rate' columns
    if "date" not in df.columns:
        raise ValueError("input data must include 'date' column")
    if target_col not in df.columns:
        raise ValueError(f"input data must include target column '{target_col}'")
    
    # 2) Feature Engineering
    logger.info("Generating engineered features (offline)")
    feature_engineer = TimeSeriesFeatureEngineer()
    df_feat = feature_engineer.transform(df)
    df_feat.columns = [c.lower() for c in df_feat.columns]

    # 4) Train/Test Split
    logger.info("Preparing train and test sets")
    df_feat = df_feat.sort_values("date").reset_index(drop=True)
    df_feat["date"] = pd.to_datetime(df_feat["date"])
    df_feat = df_feat.set_index("date")
    
    test_size = int(len(df_feat) * settings.TEST_SIZE) 
    train_df = df_feat.iloc[:-test_size].copy()
    test_df = df_feat.iloc[-test_size:].copy()
    logger.info(f"Train range: {train_df.index.min().date()} to {train_df.index.max().date()}")
    logger.info(f"Test range: {test_df.index.min().date()} to {test_df.index.max().date()}")
    logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # 5) Prepare X, y
    feature_cols = [c for c in df_feat.columns if c != target_col]
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    # 6) Preprocessing
    logger.info("Data preprocessing")
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    preproc = ColumnTransformer([("num", StandardScaler(), numeric_features)], remainder="drop")

    # 7) Model candidates
    logger.info("Model Training and Hyperparameter Tuning")
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
        ("LightGBM", LGBMRegressor(random_state=random_state, verbosity=-1), {
            "n_estimators": [100, 200], "learning_rate": [0.01, 0.05], "max_depth": [3, 5]
        }),
        ("CatBoost", CatBoostRegressor(random_state=random_state, verbose=0, allow_writing_files=False), {
            "iterations": [100, 200], "learning_rate": [0.01, 0.05], "depth": [3, 5]
        }),
    ]

    # 8) CV strategy
    cv = TimeSeriesSplit(n_splits=5)
    parent_run_name = f"Retraining_Session_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M')}"
    
    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        logger.info(f"Started Parent MLflow Run: {parent_run_name}")

        # 9) Iterate models, grid search, log to MLflow
        model_results = []
        best_overall = None
        best_metric = float("inf") 

        for name, model_obj, param_grid in models_and_grids:
            logger.info(f"Starting training candidate: {name}")
            # pipeline: preprocessing + model
            pipeline = Pipeline([("preproc", preproc), ("model", model_obj)])
            grid = {f"model__{k}": v for k, v in param_grid.items()}

            search = GridSearchCV(pipeline, grid or [{}], scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1)
            search.fit(X_train, y_train)

            y_pred = search.predict(X_test)
            metrics = evaluate_metrics(y_test.values, y_pred)
            logger.info(f"{name} test metrics: {metrics}")

            with mlflow.start_run(run_name=f"{name}_{datetime.now(timezone.utc).isoformat()}", nested=True):
                mlflow.log_param("model_name", name)
                mlflow.log_metrics(metrics)
                # Plot and log plot for this specific candidate model
                temp_plot_path = os.path.join(tempfile.gettempdir(), f"{name}_pred.png")
                plot_predictions(test_df.index, y_test.values, y_pred, f"{name} Test Results", temp_plot_path)
                mlflow.log_artifact(temp_plot_path, artifact_path="plots")

                # # Log model locally then to mlflow
                # tmp_path = os.path.join(tempfile.gettempdir(), f"{name}.joblib")
                # joblib.dump(search.best_estimator_, tmp_path)
                # mlflow.log_artifact(tmp_path, artifact_path="models")

            if metrics["MAE"] < best_metric:
                best_metric = metrics["MAE"]
                best_overall = {
                    "name": name,
                    "estimator": search.best_estimator_,
                    "metrics": metrics,
                    "params": search.best_params_,
                }
        # --- 10) Register best model in MLflow Model Registry and save final inference pipeline ---
        if best_overall:
            best_name = best_overall["name"]
            best_estimator = best_overall["estimator"]
            best_metrics = best_overall["metrics"]
            logger.info(f"Best overall model: {best_name} with MAE={best_metrics['MAE']:.6f}")


            # Fit final model on full dataset
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
        # Normalize columns to lowercase
        engineered.columns = [c.lower() for c in engineered.columns]

        # Drop BOTH the target and the date column before fitting
        cols_to_drop = [target_col.lower(), "date"]
        X_engineered = engineered.drop(columns=[c for c in cols_to_drop if c in engineered.columns])
        y_engineered = engineered[target_col.lower()]

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
        temp_artifact_path = os.path.join(tempfile.gettempdir(), f"inference_pipeline_{datetime.now(timezone.utc).isoformat()}.joblib")
        joblib.dump(inference_pipeline, temp_artifact_path)

        # Generate and save training plot
        y_best_pred = best_estimator.predict(X_test)
        temp_plot_path = os.path.join(tempfile.gettempdir(), "inference_results.png")
        plot_predictions(test_df.index, y_test.values, y_best_pred, f"{best_name} Test Results", temp_plot_path)

        with mlflow.start_run(run_name=f"best_model_register_{datetime.now(timezone.utc).isoformat()}", nested=True) as run:
            logger.info(f"Logging best model and artifacts to MLflow")
            mlflow.log_param("best_model_name", best_name)
            for k, v in best_metrics.items():
                mlflow.log_metric(k, v)
            # log pipeline artifact
            mlflow.log_artifact(temp_artifact_path, artifact_path="inference_pipeline")
            # log metadata
            mlflow.log_dict({"model_name": best_name, "params": best_overall["params"]}, "model_meta.json")
            mlflow.log_artifact(temp_plot_path, artifact_path="plots")

            # Define an input example (just a small slice of our training data)
            # We use X_engineered because final_preproc_and_model is fitted on it
            # Convert integer columns to float for MLflow compatibility
            input_example = X_engineered.head(1).copy()
            for col in input_example.select_dtypes(include=['int64', 'int32']).columns:
                input_example[col] = input_example[col].astype('float64')

            # Register model in Model Registry
            # mlflow.sklearn.log_model expects an sklearn model/pipeline object; we register the preproc+model as sklearn model
            # We will log the final_preproc_and_model and register that (feat_engineer is separate; but we include it by saving the full joblib)
            # Log the model with the 'name' and 'input_example'
            model_info = mlflow.sklearn.log_model(
                sk_model=final_preproc_and_model, 
                name="sklearn_model",  
                input_example=input_example,
                registered_model_name="ngn_us_exchange_model"
            )
            
            model_name = "ngn_us_exchange_model"
            new_version = model_info.registered_model_version
            new_mae = best_overall["metrics"]["MAE"]
            client = MlflowClient()

            try:
                # Retrieve the current "Staging" champion
                champion_version = client.get_model_version_by_alias(model_name, "staging")
                champion_run = client.get_run(champion_version.run_id)
                champion_mae = champion_run.data.metrics.get("MAE", float("inf"))
                
                logger.info(f"Challenger (v{new_version}) MAE: {new_mae:.4f} | Champion (v{champion_version.version}) MAE: {champion_mae:.4f}")

                if new_mae < champion_mae:
                    logger.info("Challenger is better! Promoting to Staging.")
                    client.set_registered_model_alias(model_name, "staging", new_version)
                    promote_to_production(
                        inference_pipeline,
                        {
                            "model_name": best_name,
                            "mlflow_version": new_version,
                            "MAE": new_mae,
                            "promoted_at": datetime.now(timezone.utc).isoformat()
                        }
                    )
                    shutil.copy(temp_plot_path, settings.TRAINING_PLOT_PATH)

                else:
                    logger.info("Champion remains. Challenger not promoted.")
                    
            except Exception:
                # If no staging alias exists, promote this first one
                logger.info("No current staging model found. Promoting as first champion.")
                client.set_registered_model_alias(model_name, "staging", new_version)
                client.set_registered_model_alias(model_name, "staging", new_version)

                promote_to_production(
                    inference_pipeline,
                    {
                        "model_name": best_name,
                        "mlflow_version": new_version,
                        "MAE": new_mae,
                        "promoted_at": datetime.now(timezone.utc).isoformat()
                    }
                )
                shutil.copy(temp_plot_path, settings.TRAINING_PLOT_PATH)

    return best_overall

if __name__ == "__main__":
    run_train()