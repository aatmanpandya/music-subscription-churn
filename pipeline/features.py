"""Feature preprocessing: the ONE code path used at training time and scoring time.

The notebook built the model matrix inline and rebuilt it with a slightly different
function for the March cohort. Here a single Preprocessor does both. It pins the
categorical vocabularies at fit time and stores them as JSON next to the model, so a
scoring run can never invent its own category codes (XGBoost splits on the integer
codes, so a different vocabulary would silently change predictions).

Rules (identical to the notebook):
  * numeric columns -> float64, missing stays NaN (XGBoost routes NaN natively)
  * categoricals -> string, null -> the explicit category "missing"
  * a category value never seen at fit time -> NaN (treated as missing by XGBoost),
    and counted, so the monitor can report the unseen share.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.schema import CAT, FEATURES, NUM

MISSING = "missing"


def canonical_strings(col: pd.Series, name: str) -> pd.Series:
    """Categorical column -> clean strings with nulls as 'missing'.

    Code-valued columns (city, registered_via, payment_method) may arrive as int,
    nullable Int64 or float64 depending on the parquet writer, so integers are
    rendered without a trailing '.0'. The sort order of these strings is the same
    as the notebook's ('1.0', '10.0', ...) so category codes are unchanged.
    """
    if name == "gender":
        s = col.astype("string").str.strip()
        s = s.mask(s == "")
    else:
        s = col.astype("float64").round().astype("Int64").astype("string")
    return s.fillna(MISSING)


class Preprocessor:
    def __init__(self, categories: dict[str, list[str]]):
        if sorted(categories) != sorted(CAT):
            raise ValueError(f"categories must cover exactly {CAT}, got {sorted(categories)}")
        self.categories = {c: list(categories[c]) for c in CAT}

    # -- fit / persistence -------------------------------------------------------
    @classmethod
    def fit(cls, df: pd.DataFrame) -> "Preprocessor":
        cats = {}
        for c in CAT:
            cats[c] = sorted(canonical_strings(df[c], c).unique().tolist())
        return cls(cats)

    def to_dict(self) -> dict:
        return {"version": 1, "numeric": NUM, "categorical": CAT, "categories": self.categories}

    @classmethod
    def from_dict(cls, d: dict) -> "Preprocessor":
        if d["numeric"] + d["categorical"] != FEATURES:
            raise ValueError("saved feature list does not match pipeline.schema.FEATURES; "
                             "refusing to score with a mismatched preprocessor")
        return cls(d["categories"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessor":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # -- transform ---------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the model matrix (NUM + CAT order) with pinned categories."""
        out = df[NUM].astype("float64").copy()
        for c in CAT:
            cats = self.categories[c]
            vals = canonical_strings(df[c], c)
            out[c] = pd.Categorical(vals.where(vals.isin(cats)), categories=cats)
        return out[FEATURES]

    def unseen_share(self, df: pd.DataFrame) -> dict[str, float]:
        """Share of rows per categorical whose value was not in the training vocabulary."""
        res = {}
        for c in CAT:
            vals = canonical_strings(df[c], c)
            res[c] = float((~vals.isin(self.categories[c])).mean())
        return res


def load_features(path: str | Path) -> pd.DataFrame:
    """Read a features parquet. DuckDB-written parquets use nullable Int64; leave
    dtypes alone here: the schema validator and Preprocessor handle them."""
    return pd.read_parquet(path)
