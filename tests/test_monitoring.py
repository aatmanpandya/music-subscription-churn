"""The monitor must stay quiet on stable data and flag the miniature March incident."""
import numpy as np

from monitoring import drift
from monitoring.run_monitoring import monitor
from tests.conftest import make_frame


def test_stable_batch_is_ok(trained):
    stable = make_frame(80000, seed=11)                        # same generating process
    res = monitor(stable, trained["scorer"], trained["profile"], labels=stable["is_churn"].to_numpy())
    assert res["overall_status"] == "OK", res["components"]
    assert res["feature_drift"]["n_alert"] == 0


def test_label_shift_flags_calibration_but_not_features(trained, shifted_frame):
    res = monitor(shifted_frame, trained["scorer"], trained["profile"],
                  labels=shifted_frame["is_churn"].to_numpy())
    assert res["components"]["calibration"] == "ALERT"
    assert res["components"]["feature_drift"] == "OK"          # inputs did not move, the labels did
    assert res["overall_status"] == "ALERT"
    assert res["performance"]["metrics"]["mean_pred_over_actual"] < 0.85


def test_feature_drift_is_ranked_and_flagged(trained):
    drifted = make_frame(30000, seed=12)
    drifted["active_days_30"] = np.clip(drifted["active_days_30"] * 0.3, 0, 30).round()
    drifted["auto_renew"] = 0.0
    res = monitor(drifted, trained["scorer"], trained["profile"])
    top = [r["feature"] for r in res["feature_drift"]["top"][:3]]
    assert "auto_renew" in top and "active_days_30" in top
    assert res["components"]["feature_drift"] == "ALERT"
    assert res["components"]["calibration"] == "NOT_EVALUATED"  # no labels supplied
    ranked = [r["psi"] for r in res["feature_drift"]["all"]]
    assert ranked == sorted(ranked, reverse=True)


def test_overall_status_is_worst_component():
    assert drift.worst(["OK", "WARN", "NOT_EVALUATED"]) == "WARN"
    assert drift.worst(["OK", "ALERT", "WARN"]) == "ALERT"
    assert drift.worst(["NOT_EVALUATED"]) == "NOT_EVALUATED"


def test_alert_on_an_unimportant_feature_is_capped_at_warn(trained):
    """Drift in an input the model barely uses must not page anyone on its own."""
    d = make_frame(30000, seed=13)
    d["completion_rate_30"] = np.clip(d["completion_rate_30"] * 0.2, 0, 1)       # big shift
    res = monitor(d, trained["scorer"], trained["profile"])
    row = next(r for r in res["feature_drift"]["all"] if r["feature"] == "completion_rate_30")
    assert row["status"] == "ALERT" and row["importance_share"] < drift.MATERIAL_IMPORTANCE
    assert res["feature_drift"]["raw_max_psi_status"] == "ALERT"
    assert res["components"]["feature_drift"] == "WARN"
