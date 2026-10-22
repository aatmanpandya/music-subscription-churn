"""Compute the monitoring result: feature drift, prediction drift, calibration, performance.

Pure functions over (features, scores, labels, frozen reference profile) so they are
testable without MLflow. Status vocabulary: OK < WARN < ALERT; sections that need
labels report NOT_EVALUATED until labels exist and do not affect the overall status.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from monitoring import psi
from pipeline.features import Preprocessor
from pipeline.metrics import reliability_table, summarize
from pipeline.schema import CAT, NUM

# A feature counts as "material" for escalation if it carries >= 1% of the model's gain.
# PSI on a feature the model barely uses cannot move predictions much, so an ALERT-level PSI
# there is capped at WARN in the component status (the raw PSI status stays in the table).
MATERIAL_IMPORTANCE = 0.01

ORDER = {"NOT_EVALUATED": -1, "OK": 0, "WARN": 1, "ALERT": 2}

# Thresholds. Chosen a priori from convention / business tolerance, not tuned on March.
THRESHOLDS = {
    "psi": {"warn": psi.WARN, "alert": psi.ALERT},
    # |mean predicted - actual| / actual. 10% off on a 6% base rate is 0.6pp.
    "calibration_in_the_large_rel": {"warn": 0.10, "alert": 0.20},
    # ECE (10 equal-count bins) divided by actual churn rate.
    "ece_rel": {"warn": 0.10, "alert": 0.20},
    # ROC-AUC lost versus the held-out reference.
    "auc_drop": {"warn": 0.01, "alert": 0.03},
}


def _level(value: float, t: dict) -> str:
    return "ALERT" if value >= t["alert"] else "WARN" if value >= t["warn"] else "OK"


def worst(statuses) -> str:
    return max(statuses, key=lambda s: ORDER[s], default="OK")


def feature_drift(features: pd.DataFrame, profile: dict, prep: Preprocessor) -> pd.DataFrame:
    rows = []
    imp = profile["importance_share"]
    for name, spec in profile["features"].items():
        cur = features[name]
        value = psi.psi_against_spec(spec, cur, name if name in CAT else None)
        row = {
            "feature": name,
            "type": "categorical" if name in CAT else "numeric",
            "psi": value,
            "status": psi.status(value),
            "importance_share": imp.get(name, 0.0),
            "ref_null_rate": spec["null_rate"],
            "cur_null_rate": float(cur.isna().mean()),
            "ref_mean": spec.get("mean"),
            "cur_mean": float(cur.astype("float64").mean()) if name in NUM else None,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    unseen = prep.unseen_share(features)
    df.loc[df.type == "categorical", "unseen_share"] = df.loc[df.type == "categorical", "feature"].map(unseen)
    return df.sort_values("psi", ascending=False).reset_index(drop=True)


def prediction_drift(raw_scores, profile: dict) -> dict:
    spec = profile["score"]
    value = psi.psi_against_spec(spec, pd.Series(np.asarray(raw_scores, dtype="float64")))
    cur_mean = float(np.mean(raw_scores))
    return {
        "psi": value,
        "status": psi.status(value),
        "ref_mean_score": spec["mean"],
        "cur_mean_score": cur_mean,
        "mean_score_change_rel": cur_mean / spec["mean"] - 1,
    }


def performance(y, p_final, p_raw, profile: dict) -> dict:
    ref = profile["reference_metrics"]
    m = summarize(y, p_final)
    cal_gap = abs(m["mean_pred"] - m["actual_rate"]) / m["actual_rate"]
    ece_rel = m["ece_quantile10"] / m["actual_rate"]
    auc_drop = ref["roc_auc"] - m["roc_auc"]
    t = THRESHOLDS
    statuses = {
        "calibration_in_the_large": _level(cal_gap, t["calibration_in_the_large_rel"]),
        "ece": _level(ece_rel, t["ece_rel"]),
        "auc": _level(auc_drop, t["auc_drop"]),
    }
    return {
        "metrics": m,
        "reference_metrics": ref,
        "derived": {"calibration_gap_rel": cal_gap, "ece_rel": ece_rel, "auc_drop": auc_drop,
                    "base_rate_change": m["actual_rate"] - ref["actual_rate"]},
        "statuses": statuses,
        "status": worst(statuses.values()),
        "reliability": reliability_table(y, p_final, 10, "quantile").to_dict(orient="records"),
        "reliability_uniform": reliability_table(y, p_final, 10, "uniform").to_dict(orient="records"),
    }


def run(features: pd.DataFrame, raw_scores, final_scores, profile: dict, prep: Preprocessor,
        labels=None, top_n: int = 10) -> dict:
    fd = feature_drift(features, profile, prep)
    pdrift = prediction_drift(raw_scores, profile)
    fd["material"] = fd.importance_share >= MATERIAL_IMPORTANCE
    capped = fd.status.where(~((fd.status == "ALERT") & ~fd.material), "WARN")
    feature_status = worst(capped)
    raw_feature_status = worst(fd.status)
    components = {"feature_drift": feature_status, "prediction_drift": pdrift["status"]}
    perf = None
    if labels is not None:
        perf = performance(np.asarray(labels), np.asarray(final_scores), np.asarray(raw_scores), profile)
        components["calibration"] = worst([perf["statuses"]["calibration_in_the_large"], perf["statuses"]["ece"]])
        components["ranking_performance"] = perf["statuses"]["auc"]
    else:
        components["calibration"] = "NOT_EVALUATED"
        components["ranking_performance"] = "NOT_EVALUATED"
    return {
        "overall_status": worst(components.values()),
        "components": components,
        "thresholds": THRESHOLDS,
        "n_rows": int(len(features)),
        "feature_drift": {
            "raw_max_psi_status": raw_feature_status,
            "material_importance_threshold": MATERIAL_IMPORTANCE,
            "n_warn": int((fd.status == "WARN").sum()),
            "n_alert": int((fd.status == "ALERT").sum()),
            "top": fd.head(top_n).to_dict(orient="records"),
            "all": fd.to_dict(orient="records"),
        },
        "prediction_drift": pdrift,
        "performance": perf,
    }
