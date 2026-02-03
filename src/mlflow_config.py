"""MLflow configuration helpers for the project."""

from __future__ import annotations

from typing import Mapping, Optional

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_EXPERIMENT_NAME = "OC_P6_Credit_Scoring"


def configure_mlflow(
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    *,
    autolog: bool = True,
    log_models: bool = False,
    extra_tags: Optional[Mapping[str, str]] = None,
) -> mlflow:
    """Configure MLflow tracking for this project.

    Returns the mlflow module to allow `mlflow = configure_mlflow()` usage.
    """
    if autolog:
        mlflow.autolog(log_models=log_models)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if extra_tags:
        for key, value in extra_tags.items():
            mlflow.set_tag(key, value)

    return mlflow
