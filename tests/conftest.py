"""Shared fixtures: a small synthetic cohort generator (the real parquets are gitignored,
so tests must not depend on them) and a throwaway MLflow store."""

import os

import numpy as np
import pandas as pd
import pytest

from pipeline.schema import CAT, NUM  # noqa: F401


def make_frame(n: int = 20000, seed: int = 0, shift: float = 0.0, with_label: bool = True) -> pd.DataFrame:
    """Schema-valid synthetic cohort. `shift` adds to the true log-odds of churn
    (a pure base-rate change, the situation prior-shift correction assumes)."""
    r = np.random.default_rng(seed)
    auto = r.binomial(1, 0.85, n)
    cancel = r.binomial(1, 0.03, n)
    plan = r.choice([7, 30, 30, 30, 90, 180, 365], n)
    active30 = r.integers(0, 31, n)
    active_prev = np.clip(active30 + r.integers(-5, 6, n), 0, 30)
    hours30 = active30 * r.uniform(0, 5, n)
    hours_prev = active_prev * r.uniform(0, 5, n)
    tenure = r.integers(0, 700, n)
    price = plan * r.choice([3.0, 4.0, 5.0], n)
    paid = np.where(r.random(n) < 0.1, price * 0.8, price)
    nlog = r.random(n) < 0.12
    df = pd.DataFrame({
        "msno": [f"u{seed}_{i}" for i in range(n)],
        "age": np.where(r.random(n) < 0.6, np.nan, r.integers(15, 60, n)),
        "account_age_days": r.integers(0, 4000, n).astype(float),
        "n_transactions": r.integers(1, 40, n).astype(float),
        "n_cancels": r.integers(0, 4, n).astype(float),
        "subscriber_tenure_days": tenure.astype(float),
        "share_auto_renew": np.round(r.random(n), 3),
        "plan_days": plan.astype(float),
        "auto_renew": auto.astype(float),
        "last_tx_cancel": cancel.astype(float),
        "list_price": price,
        "amount_paid": paid,
        "discount": price - paid,
        "days_to_expiry": r.integers(-5, 40, n).astype(float),
        "days_since_last_listen": np.where(nlog, np.nan, r.integers(0, 60, n)),
        "active_days_30": active30.astype(float),
        "active_days_prev30": active_prev.astype(float),
        "hours_30": hours30,
        "hours_prev30": hours_prev,
        "hours_change": hours30 - hours_prev,
        "avg_unique_songs_30": np.where(r.random(n) < 0.2, np.nan, r.uniform(1, 80, n)),
        "completion_rate_30": np.where(r.random(n) < 0.2, np.nan, r.random(n)),
        "has_logs": (~nlog).astype(float),
        "has_profile": r.binomial(1, 0.88, n).astype(float),
        "city": np.where(r.random(n) < 0.1, np.nan, r.choice([1, 5, 13, 22], n, p=[.5, .2, .2, .1])),
        "registered_via": np.where(r.random(n) < 0.1, np.nan, r.choice([3, 4, 7, 9, 13], n)),
        "payment_method": r.choice([20, 28, 38, 41], n).astype(float),
        "gender": r.choice(["male", "female", None], n, p=[.2, .2, .6]),
    })
    if with_label:
        z = (-2.7 + 3.0 * (1 - auto) + 2.5 * cancel - 0.10 * active30 + 0.6 * (plan == 7)
             - 0.004 * tenure + shift + r.normal(0, 0.3, n))
        df["is_churn"] = (r.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df


@pytest.fixture(scope="session")
def train_frame():
    return make_frame(30000, seed=1)


@pytest.fixture(scope="session")
def shifted_frame():
    """Same population, higher base rate: a miniature of the March incident."""
    return make_frame(40000, seed=2, shift=0.7)


@pytest.fixture(scope="session")
def mlflow_env(tmp_path_factory):
    d = tmp_path_factory.mktemp("mlflow")
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{d / 'mlflow.db'}"
    os.environ["MLFLOW_ARTIFACT_ROOT"] = str(d / "artifacts")
    yield d
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    os.environ.pop("MLFLOW_ARTIFACT_ROOT", None)


@pytest.fixture(scope="session")
def trained(train_frame, tmp_path_factory, mlflow_env):
    """Train a tiny model through the real pipeline and register it in the temp store."""
    from pipeline import train as T

    p = tmp_path_factory.mktemp("data") / "features.parquet"
    train_frame.to_parquet(p, index=False)
    args = T.parse(["--features", str(p), "--oot-features", "", "--depths", "3", "4",
                    "--max-trees", "300", "--learning-rate", "0.2", "--n-jobs", "2", "--run-name", "pytest"])
    result = T.train(args)
    result["train_path"] = p
    return result
