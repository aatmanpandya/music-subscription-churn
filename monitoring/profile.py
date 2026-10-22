"""Reference profile: a compact, frozen description of the training data.

Saved inside every model bundle so the monitor never needs the training set:
  * per-feature bin proportions (for PSI), null rate and mean
  * the raw-score distribution on held-out reference data (prediction drift)
  * reference performance and calibration (to compare against once labels arrive)
  * XGBoost gain importance, so the report can say which drifting features matter
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from monitoring import psi
from pipeline.features import Preprocessor
from pipeline.schema import CAT, NUM


def build_profile(
    features: pd.DataFrame,
    prep: Preprocessor,
    ref_scores,
    ref_labels,
    importance: dict[str, float],
    reference_metrics: dict,
) -> dict:
    feats = {}
    for c in NUM:
        s = features[c]
        spec = psi.numeric_spec(s)
        x = s.astype("float64")
        spec.update(null_rate=float(x.isna().mean()), mean=float(x.mean()), median=float(x.median()))
        feats[c] = spec
    for c in CAT:
        spec = psi.categorical_spec(features[c], c, prep.categories[c])
        spec.update(null_rate=float(features[c].isna().mean()))
        feats[c] = spec
    score = psi.score_spec(ref_scores)
    score.update(mean=float(np.mean(ref_scores)), actual_rate=float(np.mean(ref_labels)))
    tot = sum(importance.values()) or 1.0
    return {
        "n_reference_rows": int(len(features)),
        "features": feats,
        "score": score,
        "importance_share": {k: v / tot for k, v in importance.items()},
        "reference_metrics": reference_metrics,
    }
