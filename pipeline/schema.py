"""Input schema for the model: columns, dtypes, value ranges and null rates.

validate_features() raises SchemaValidationError listing *every* violation, so a
bad upstream batch is rejected loudly instead of silently producing scores.

Two kinds of problem are deliberately kept apart:
  * Schema violations (this file): the data is structurally wrong or physically
    impossible (missing column, text in a numeric field, hours_30 = 900, a flag
    equal to 7). The batch must not be scored.
  * Drift (monitoring/): the data is valid but looks different from training.
    The batch is scored and the monitor raises a flag.

Ranges are *sanity bounds*, wider than anything seen in training, because a
range check that fires on a legitimate extreme customer is worse than none.
Null limits are set at roughly 2x the training null rate (or 0 for columns the
feature builder always fills), so a broken join upstream is caught.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class SchemaValidationError(ValueError):
    """Raised when an input frame violates the feature schema."""

    def __init__(self, violations: list[str]):
        self.violations = violations
        super().__init__(
            f"{len(violations)} schema violation(s):\n  - " + "\n  - ".join(violations)
        )


@dataclass(frozen=True)
class NumericSpec:
    lo: float
    hi: float
    max_null: float          # maximum allowed share of nulls in a batch
    integer: bool = False    # values must be whole numbers
    binary: bool = False     # values must be exactly 0 or 1


# name -> spec. Order here is the model's numeric feature order (matches the notebook).
NUMERIC: dict[str, NumericSpec] = {
    "age": NumericSpec(10, 80, 0.75),
    "account_age_days": NumericSpec(-400, 20000, 0.25, integer=True),
    "n_transactions": NumericSpec(0, 5000, 0.01, integer=True),
    "n_cancels": NumericSpec(0, 5000, 0.01, integer=True),
    "subscriber_tenure_days": NumericSpec(0, 5000, 0.01, integer=True),
    "share_auto_renew": NumericSpec(0, 1, 0.01),
    "plan_days": NumericSpec(0, 1000, 0.01, integer=True),
    "auto_renew": NumericSpec(0, 1, 0.01, binary=True),
    "last_tx_cancel": NumericSpec(0, 1, 0.01, binary=True),
    "list_price": NumericSpec(0, 10000, 0.01),
    "amount_paid": NumericSpec(0, 10000, 0.01),
    "discount": NumericSpec(-10000, 10000, 0.01),
    "days_to_expiry": NumericSpec(-20000, 4000, 0.01, integer=True),
    "days_since_last_listen": NumericSpec(0, 5000, 0.30, integer=True),
    "active_days_30": NumericSpec(0, 30, 0.0, integer=True),
    "active_days_prev30": NumericSpec(0, 30, 0.0, integer=True),
    "hours_30": NumericSpec(0, 720, 0.0),
    "hours_prev30": NumericSpec(0, 720, 0.0),
    "hours_change": NumericSpec(-720, 720, 0.0),
    "avg_unique_songs_30": NumericSpec(0, 5000, 0.45),
    "completion_rate_30": NumericSpec(0, 1, 0.45),
    "has_logs": NumericSpec(0, 1, 0.0, binary=True),
    "has_profile": NumericSpec(0, 1, 0.0, binary=True),
}

# Categorical features arrive as numeric codes (city, registered_via, payment_method)
# or strings (gender). Values never seen in training are *not* schema errors: the
# preprocessor maps them to missing and the monitor reports the unseen share.
CATEGORICAL: dict[str, NumericSpec | None] = {
    "city": NumericSpec(0, 100, 0.25, integer=True),
    "registered_via": NumericSpec(0, 100, 0.25, integer=True),
    "payment_method": NumericSpec(0, 200, 0.01, integer=True),
    "gender": None,  # string column, checked separately
}

NUM = list(NUMERIC)
CAT = list(CATEGORICAL)
FEATURES = NUM + CAT          # model input order
ID_COL = "msno"
LABEL_COL = "is_churn"
ALLOWED_GENDER = {"male", "female", ""}

# Null-rate checks are statistical, so they only make sense on a real batch.
MIN_ROWS_FOR_NULL_CHECK = 1000


def validate_features(
    df: pd.DataFrame,
    *,
    check_null_rates: bool = True,
    require_label: bool = False,
    require_unique_ids: bool = False,
) -> None:
    """Raise SchemaValidationError if `df` is not valid model input."""
    v: list[str] = []

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        v.append(f"missing required columns: {missing}")
    if require_label and LABEL_COL not in df.columns:
        v.append(f"missing label column '{LABEL_COL}'")
    if len(df) == 0:
        v.append("frame has zero rows")
    if v:  # structural failure; value checks would only add noise
        raise SchemaValidationError(v)

    for name, spec in {**NUMERIC, **{k: s for k, s in CATEGORICAL.items() if s}}.items():
        col = df[name]
        if not (pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col)):
            v.append(f"{name}: expected numeric dtype, got {col.dtype}")
            continue
        x = col.astype("float64")
        if np.isinf(x.to_numpy()).any():
            v.append(f"{name}: contains infinite values")
        nulls = float(x.isna().mean())
        if spec.max_null == 0.0 and nulls > 0:
            # Columns the feature builder always fills: any null is an upstream bug,
            # and this holds for a 1-row API request as much as for a full batch.
            v.append(f"{name}: must not be null ({int(x.isna().sum())} null value(s))")
        elif check_null_rates and len(df) >= MIN_ROWS_FOR_NULL_CHECK and nulls > spec.max_null:
            v.append(f"{name}: null rate {nulls:.2%} exceeds limit {spec.max_null:.2%}")
        xv = x.dropna()
        xv = xv[np.isfinite(xv)]
        n_out = int(((xv < spec.lo) | (xv > spec.hi)).sum())
        if n_out:
            v.append(f"{name}: {n_out} value(s) outside [{spec.lo}, {spec.hi}] "
                     f"(observed min {xv.min():g}, max {xv.max():g})")
        if spec.binary:
            n_bad = int((~xv.isin([0, 1])).sum())
            if n_bad:
                v.append(f"{name}: {n_bad} value(s) not in {{0, 1}}")
        elif spec.integer:
            n_frac = int((xv != np.round(xv)).sum())
            if n_frac:
                v.append(f"{name}: {n_frac} non-integer value(s)")

    g = df["gender"]
    if not (pd.api.types.is_string_dtype(g) or g.dtype == object or g.isna().all()):
        v.append(f"gender: expected string dtype, got {g.dtype}")
    else:
        bad = set(g.dropna().astype(str).unique()) - ALLOWED_GENDER
        if bad:
            v.append(f"gender: unexpected values {sorted(bad)[:5]}")
        if check_null_rates and len(df) >= MIN_ROWS_FOR_NULL_CHECK and g.isna().mean() > 0.80:
            v.append(f"gender: null rate {g.isna().mean():.2%} exceeds limit 80.00%")

    if require_label and LABEL_COL in df.columns:
        lab = df[LABEL_COL]
        if lab.isna().any() or not lab.isin([0, 1]).all():
            v.append(f"{LABEL_COL}: must be 0/1 with no nulls")
    if require_unique_ids and ID_COL in df.columns and df[ID_COL].duplicated().any():
        v.append(f"{ID_COL}: {int(df[ID_COL].duplicated().sum())} duplicate id(s)")

    if v:
        raise SchemaValidationError(v)
