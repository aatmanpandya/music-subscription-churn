"""Post-hoc probability calibrators.

All three are monotone increasing maps p -> p', so they cannot change the order of
subscribers: ROC-AUC and the top-k lists are unchanged (isotonic can create ties,
see DECISIONS_PRODUCTION.md). Applying a calibrator is numpy-only so the API image
needs no scikit-learn; *fitting* imports scikit-learn lazily.

Serialised as plain JSON so a calibrator travels inside the model bundle.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype="float64")))


class Calibrator:
    kind = "base"

    def transform(self, p):  # pragma: no cover - interface
        raise NotImplementedError

    def to_dict(self) -> dict:  # pragma: no cover - interface
        raise NotImplementedError

    @staticmethod
    def from_dict(d: dict) -> "Calibrator":
        kinds = {c.kind: c for c in (Identity, PriorShift, Platt, Isotonic)}
        return kinds[d["kind"]]._from_dict(d)


class Identity(Calibrator):
    kind = "identity"

    def transform(self, p):
        return np.asarray(p, dtype="float64")

    def to_dict(self):
        return {"kind": self.kind}

    @classmethod
    def _from_dict(cls, d):
        return cls()


class PriorShift(Calibrator):
    """Adjust log-odds for a change in base rate, keeping P(x | y) fixed.

    If the model was fitted at base rate pi_old and the new population has base rate
    pi_new, Bayes' rule gives  logit(p') = logit(p) + logit(pi_new) - logit(pi_old).
    One parameter, no fitting beyond knowing the new base rate.
    """
    kind = "prior_shift"

    def __init__(self, old_rate: float, new_rate: float):
        self.old_rate, self.new_rate = float(old_rate), float(new_rate)
        self.offset = float(logit(new_rate) - logit(old_rate))

    def transform(self, p):
        return expit(logit(p) + self.offset)

    def to_dict(self):
        return {"kind": self.kind, "old_rate": self.old_rate, "new_rate": self.new_rate}

    @classmethod
    def _from_dict(cls, d):
        return cls(d["old_rate"], d["new_rate"])


class Platt(Calibrator):
    """Logistic regression on the model's log-odds: p' = sigmoid(a * logit(p) + b)."""
    kind = "platt"

    def __init__(self, a: float, b: float):
        self.a, self.b = float(a), float(b)

    @classmethod
    def fit(cls, p, y) -> "Platt":
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(logit(p).reshape(-1, 1), np.asarray(y))
        return cls(lr.coef_[0, 0], lr.intercept_[0])

    def transform(self, p):
        return expit(self.a * logit(p) + self.b)

    def to_dict(self):
        return {"kind": self.kind, "a": self.a, "b": self.b}

    @classmethod
    def _from_dict(cls, d):
        return cls(d["a"], d["b"])


class Isotonic(Calibrator):
    """Non-parametric monotone step function, stored as knots and applied with np.interp.

    A plain isotonic fit is only weakly monotone: it flattens runs of scores into one
    value (on March, 800k distinct raw scores collapsed into 188 values), which ties
    users that the raw model could rank and costs PR-AUC / top-k quality. So the output is
    blended with a vanishing share of the raw score, (1 - 1e-6) * isotonic + 1e-6 * p.
    That makes the map strictly increasing, keeps the order of every pair the raw model
    ordered, and moves probabilities by less than 1e-6.
    """
    kind = "isotonic"
    TIE_EPS = 1e-6

    def __init__(self, x, y):
        self.x = np.asarray(x, dtype="float64")
        self.y = np.asarray(y, dtype="float64")

    @classmethod
    def fit(cls, p, y) -> "Isotonic":
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(np.asarray(p, dtype="float64"), np.asarray(y))
        return cls(iso.X_thresholds_, iso.y_thresholds_)

    def transform(self, p):
        p = np.asarray(p, dtype="float64")
        return (1 - self.TIE_EPS) * np.interp(p, self.x, self.y) + self.TIE_EPS * p

    def to_dict(self):
        return {"kind": self.kind, "x": self.x.tolist(), "y": self.y.tolist()}

    @classmethod
    def _from_dict(cls, d):
        return cls(d["x"], d["y"])


def estimate_prior_em(p, old_rate: float, iters: int = 200, tol: float = 1e-8) -> float:
    """Label-free estimate of the new base rate (Saerens, Latinne & Decaestecker 2002).

    Iterate: adjust scores for the current guess of the new rate, average them to get a
    new guess. Valid ONLY if the shift is pure label shift (P(x|y) unchanged) and the
    model was calibrated on the old data. DECISIONS_PRODUCTION.md shows it fails on March.
    """
    p = np.asarray(p, dtype="float64")
    new = float(old_rate)
    for _ in range(iters):
        nxt = float(PriorShift(old_rate, new).transform(p).mean())
        if abs(nxt - new) < tol:
            return nxt
        new = nxt
    return new
