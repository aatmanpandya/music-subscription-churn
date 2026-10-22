"""Ranking and calibration metrics shared by training, monitoring and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def reliability_table(y, p, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    """Bin predictions and compare mean predicted risk with the observed churn rate.

    strategy="quantile": equal-count bins. Scores are very skewed (most subscribers
        sit near 2%), so equal-width bins would put nearly everyone in the first bin.
    strategy="uniform": equal-width bins on [0, 1], the textbook reliability diagram.
    """
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    elif strategy == "uniform":
        idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    else:
        raise ValueError(strategy)
    df = pd.DataFrame({"bin": idx, "y": y, "p": p})
    g = df.groupby("bin")
    t = pd.DataFrame({"n": g.size(), "mean_pred": g.p.mean(), "actual": g.y.mean()})
    t["gap"] = t.mean_pred - t.actual
    return t.reset_index()


def ece(y, p, n_bins: int = 10, strategy: str = "quantile") -> float:
    """Expected calibration error: bin-weighted mean |predicted - actual|."""
    t = reliability_table(y, p, n_bins, strategy)
    return float((t.n * t.gap.abs()).sum() / t.n.sum())


def brier(y, p) -> float:
    y = np.asarray(y, dtype="float64")
    return float(np.mean((np.asarray(p, dtype="float64") - y) ** 2))


def log_loss(y, p, eps: float = 1e-7) -> float:
    y = np.asarray(y, dtype="float64")
    p = np.clip(np.asarray(p, dtype="float64"), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def summarize(y, p) -> dict[str, float]:
    """One dict with every number the reports use."""
    y = np.asarray(y)
    p = np.asarray(p, dtype="float64")
    actual = float(y.mean())
    return {
        "n": int(len(y)),
        "actual_rate": actual,
        "mean_pred": float(p.mean()),
        "mean_pred_over_actual": float(p.mean() / actual) if actual > 0 else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": brier(y, p),
        "log_loss": log_loss(y, p),
        "ece_quantile10": ece(y, p, 10, "quantile"),
        "ece_uniform10": ece(y, p, 10, "uniform"),
    }
