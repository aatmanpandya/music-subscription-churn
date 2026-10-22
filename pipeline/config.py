"""Paths and registry constants shared by every stage.

Everything can be overridden with environment variables so tests and Docker
can point at temporary stores without code changes.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

TRAIN_FEATURES = DATA / "features.parquet"          # Feb cohort (training reference)
MARCH_FEATURES = DATA / "features_oot.parquet"      # Mar cohort (out-of-time scoring batch)

MODEL_NAME = os.environ.get("CHURN_MODEL_NAME", "kkbox-churn")
EXPERIMENT_NAME = os.environ.get("CHURN_EXPERIMENT", "kkbox-churn")
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"
PREVIOUS_ALIAS = "previous"

SEED = 42


def tracking_uri() -> str:
    """SQLite backend (the registry needs a database) plus a local artifact folder."""
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")


def artifact_root() -> str:
    return os.environ.get("MLFLOW_ARTIFACT_ROOT", str(ROOT / "mlartifacts"))
