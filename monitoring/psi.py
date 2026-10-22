"""Population Stability Index (PSI) with frozen reference bins.

PSI compares how a variable is spread across bins in a reference sample (expected, e)
and a new sample (actual, a):

    PSI = sum over bins of (a_i - e_i) * ln(a_i / e_i)

It is symmetric, 0 for identical distributions, and grows with the shift. Common
rules of thumb: < 0.1 stable, 0.1-0.25 moderate shift (warn), > 0.25 major (alert).

Design choices (see DECISIONS_PRODUCTION.md):
  * Bin edges are frozen from the reference and reused on every new batch, so PSI
    values are comparable over time.
  * Numeric features: up to 10 reference-quantile bins, plus a separate bin for
    missing values so a change in null rate is itself visible. Features with <= 12
    distinct values (flags, day counts) get one bin per value.
  * Categorical features: one bin per training category (incl. 'missing') plus an
    'unseen' bin for codes the model never saw.
  * Empty bins are floored at 1e-4 so ln() is defined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.features import canonical_strings

N_BINS = 10
MAX_DISCRETE = 12
FLOOR = 1e-4
WARN = 0.10
ALERT = 0.25


# ---------------------------------------------------------------- bin specs
def numeric_spec(ref: pd.Series, n_bins: int = N_BINS) -> dict:
    x = ref.astype("float64").dropna().to_numpy()
    uniq = np.unique(x)
    if len(uniq) <= MAX_DISCRETE:
        spec = {"kind": "discrete", "values": uniq.tolist()}
    else:
        edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))
        spec = {"kind": "quantile", "edges": edges.tolist()}
    spec["proportions"] = _props(spec, ref)
    return spec


def categorical_spec(ref: pd.Series, name: str, categories: list[str]) -> dict:
    spec = {"kind": "categorical", "categories": list(categories)}
    spec["proportions"] = _props(spec, ref, name)
    return spec


def score_spec(ref_scores, n_bins: int = 20) -> dict:
    spec = numeric_spec(pd.Series(np.asarray(ref_scores, dtype="float64")), n_bins)
    return spec


# ---------------------------------------------------------------- binning
def bin_counts(spec: dict, cur: pd.Series, name: str | None = None) -> np.ndarray:
    kind = spec["kind"]
    if kind == "categorical":
        vals = canonical_strings(cur, name)
        cats = spec["categories"]
        codes = pd.Categorical(vals, categories=cats).codes  # -1 = unseen
        counts = np.bincount(np.where(codes < 0, len(cats), codes), minlength=len(cats) + 1)
        return counts
    x = cur.astype("float64").to_numpy()
    null = np.isnan(x)
    if kind == "discrete":
        vals = np.asarray(spec["values"])
        pos = np.searchsorted(vals, x)
        pos_c = np.clip(pos, 0, len(vals) - 1)
        hit = (vals[pos_c] == x) & ~null
        idx = np.where(hit, pos_c, len(vals))          # len(vals) = 'other' (unseen value)
        idx = np.where(null, len(vals) + 1, idx)       # +1 = missing
        return np.bincount(idx, minlength=len(vals) + 2)
    edges = np.asarray(spec["edges"])
    idx = np.searchsorted(edges, x, side="left")        # 0..len(edges)
    idx = np.where(null, len(edges) + 1, idx)
    return np.bincount(idx, minlength=len(edges) + 2)


def _props(spec: dict, s: pd.Series, name: str | None = None) -> list[float]:
    c = bin_counts(spec, s, name)
    return (c / max(c.sum(), 1)).tolist()


# ---------------------------------------------------------------- PSI
def psi_from_props(expected, actual) -> float:
    e = np.maximum(np.asarray(expected, dtype="float64"), FLOOR)
    a = np.maximum(np.asarray(actual, dtype="float64"), FLOOR)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_against_spec(spec: dict, cur: pd.Series, name: str | None = None) -> float:
    c = bin_counts(spec, cur, name)
    return psi_from_props(spec["proportions"], c / max(c.sum(), 1))


def psi_two_samples(reference, current, n_bins: int = N_BINS) -> float:
    """PSI between two numeric samples, binning on the reference. Convenience for tests."""
    ref = pd.Series(np.asarray(reference, dtype="float64"))
    spec = numeric_spec(ref, n_bins)
    return psi_against_spec(spec, pd.Series(np.asarray(current, dtype="float64")))


def status(value: float, warn: float = WARN, alert: float = ALERT) -> str:
    return "ALERT" if value >= alert else "WARN" if value >= warn else "OK"
