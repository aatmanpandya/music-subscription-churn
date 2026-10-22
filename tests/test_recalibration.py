"""Recalibration: fit on one random half, judge on the other; ranking must not change."""
import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from monitoring import recalibrate as R
from pipeline.calibration import Calibrator, Identity, Isotonic, Platt, PriorShift, estimate_prior_em
from pipeline.metrics import ece, summarize


@pytest.fixture(scope="module")
def shifted(trained, shifted_frame):
    """Raw scores of the trained model on a batch whose churn base rate moved up."""
    raw = trained["scorer"].predict(shifted_frame, validate=False)["raw_probability"].to_numpy()
    y = shifted_frame["is_churn"].to_numpy()
    ia, ib = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y, random_state=0)
    return raw, y, ia, ib, trained["metadata"]["training_base_rate"]


def test_the_incident_is_reproduced_in_miniature(shifted):
    raw, y, _, _, _ = shifted
    assert raw.mean() < 0.85 * y.mean()          # under-predicts the new base rate


@pytest.mark.parametrize("name", ["prior_shift", "platt", "isotonic"])
def test_every_method_improves_ece_on_heldout_half(shifted, name):
    raw, y, ia, ib, old = shifted
    cal = R.fit_candidates(raw[ia], y[ia], old)[name]
    before = ece(y[ib], raw[ib])
    after = ece(y[ib], cal.transform(raw[ib]))
    # Prior shift has one parameter: it fixes the average but not the shape, so it must
    # improve; the two fitted calibrators should do much better.
    assert after < (before if name == "prior_shift" else 0.5 * before), (name, before, after)
    gap_before = abs(raw[ib].mean() - y[ib].mean())
    gap_after = abs(cal.transform(raw[ib]).mean() - y[ib].mean())
    assert gap_after < 0.5 * gap_before


@pytest.mark.parametrize("name,tol", [("prior_shift", 1e-9), ("platt", 1e-9), ("isotonic", 1e-6)])
def test_ranking_is_unchanged(shifted, name, tol):
    raw, y, ia, ib, old = shifted
    cal = R.fit_candidates(raw[ia], y[ia], old)[name]
    auc0 = summarize(y[ib], raw[ib])["roc_auc"]
    auc1 = summarize(y[ib], cal.transform(raw[ib]))["roc_auc"]
    assert abs(auc1 - auc0) <= tol


def test_isotonic_keeps_the_raw_models_ordering_and_does_not_create_ties(shifted):
    """Plain isotonic collapses scores into flat steps; ours must not (top-k lists, PR-AUC)."""
    raw, y, ia, ib, old = shifted
    iso = R.fit_candidates(raw[ia], y[ia], old)["isotonic"]
    out = iso.transform(raw[ib])
    assert len(np.unique(out)) >= 0.95 * len(np.unique(raw[ib]))
    assert summarize(y[ib], out)["pr_auc"] == pytest.approx(summarize(y[ib], raw[ib])["pr_auc"], abs=1e-6)
    order = np.argsort(raw[ib], kind="stable")
    assert (np.diff(out[order]) >= 0).all()


def test_calibrators_are_monotone(shifted):
    raw, y, ia, _, old = shifted
    grid = np.linspace(0.001, 0.99, 400)
    for cal in R.fit_candidates(raw[ia], y[ia], old).values():
        assert (np.diff(cal.transform(grid)) >= -1e-12).all()


def test_halves_are_disjoint(shifted):
    _, _, ia, ib, _ = shifted
    assert set(ia).isdisjoint(ib) and len(ia) + len(ib) == len(set(ia) | set(ib))


def test_in_sample_ece_is_optimistic_for_isotonic_which_is_why_we_hold_out(shifted):
    raw, y, ia, ib, old = shifted
    iso = Isotonic.fit(raw[ia], y[ia])
    assert ece(y[ia], iso.transform(raw[ia])) <= ece(y[ib], iso.transform(raw[ib])) + 1e-4


def test_prior_shift_math():
    ps = PriorShift(0.06, 0.09)
    # For a subscriber the model gave exactly the old base rate, the new rate comes back.
    assert abs(float(ps.transform(0.06)) - 0.09) < 1e-9
    assert float(ps.transform(0.5)) > 0.5
    assert PriorShift(0.06, 0.06).transform(0.2) == pytest.approx(0.2)


def test_em_recovers_a_pure_label_shift_from_scores_alone():
    """Class-conditionals fixed, only the base rate moves: EM finds the new rate with no labels."""
    r = np.random.default_rng(0)
    old, new = 0.06, 0.12

    def sample(rate, n=300000):
        y = r.random(n) < rate
        return np.where(y, r.normal(1.5, 1, n), r.normal(0, 1, n))

    def old_posterior(x):          # exact posterior under the OLD prior
        z = np.log(old / (1 - old)) + 1.5 * x - 1.125
        return 1 / (1 + np.exp(-z))

    est = estimate_prior_em(old_posterior(sample(new)), old)
    assert abs(est - new) < 0.01
    # and with no shift it stays put
    assert abs(estimate_prior_em(old_posterior(sample(old)), old) - old) < 0.01


@pytest.mark.parametrize("cal", [Identity(), PriorShift(0.06, 0.09), Platt(0.9, 0.3),
                                 Isotonic([0, 0.5, 1], [0, 0.4, 1])])
def test_calibrator_json_roundtrip(cal):
    again = Calibrator.from_dict(cal.to_dict())
    grid = np.linspace(0.01, 0.99, 50)
    np.testing.assert_allclose(cal.transform(grid), again.transform(grid))


def test_promotion_rule_blocks_a_calibrator_that_changes_ranking():
    base = {"mean_pred": 0.065, "actual_rate": 0.09, "ece_quantile10": 0.025, "roc_auc": 0.873, "brier": 0.054}
    good = {"mean_pred": 0.0905, "actual_rate": 0.09, "ece_quantile10": 0.002, "roc_auc": 0.8731, "brier": 0.050}
    bad_auc = {**good, "roc_auc": 0.85}
    rep_ok = {"none (raw model)": [0.025] * 10, "x": [0.002] * 10}
    assert R.promotion_decision({"none (raw model)": base, "x": good}, "x", 100000, 9000, rep_ok)["promote"]
    d = R.promotion_decision({"none (raw model)": base, "x": bad_auc}, "x", 100000, 9000, rep_ok)
    assert not d["promote"] and not d["checks"]["auc_unchanged_within_0.001"]
    small = R.promotion_decision({"none (raw model)": base, "x": good}, "x", 1000, 90, rep_ok)
    assert not small["promote"]
