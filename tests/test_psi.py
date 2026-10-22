import numpy as np
import pandas as pd

from monitoring import psi


def test_identical_distributions_give_zero():
    x = np.random.default_rng(0).lognormal(size=50000)
    assert psi.psi_two_samples(x, x) < 1e-9


def test_same_distribution_different_sample_is_tiny():
    r = np.random.default_rng(1)
    a, b = r.normal(size=100000), r.normal(size=100000)
    assert psi.psi_two_samples(a, b) < 0.01


def test_shifted_distribution_is_large():
    r = np.random.default_rng(2)
    a = r.normal(0, 1, 50000)
    assert psi.psi_two_samples(a, r.normal(1.0, 1, 50000)) > 0.25
    assert 0.1 < psi.psi_two_samples(a, r.normal(0.4, 1, 50000)) < 0.25   # moderate shift: warn band


def test_psi_is_symmetric_for_given_bins():
    e = np.array([0.5, 0.3, 0.2])
    a = np.array([0.2, 0.3, 0.5])
    assert abs(psi.psi_from_props(e, a) - psi.psi_from_props(a, e)) < 1e-12


def test_known_value():
    # hand computed: (0.4-0.5)ln(0.4/0.5) + (0.6-0.5)ln(0.6/0.5)
    expected = (0.4 - 0.5) * np.log(0.4 / 0.5) + (0.6 - 0.5) * np.log(0.6 / 0.5)
    assert abs(psi.psi_from_props([0.5, 0.5], [0.4, 0.6]) - expected) < 1e-12


def test_null_rate_shift_is_detected():
    r = np.random.default_rng(3)
    ref = pd.Series(r.normal(size=40000))
    cur = ref.copy()
    cur[r.random(40000) < 0.3] = np.nan
    spec = psi.numeric_spec(ref)
    assert psi.psi_against_spec(spec, cur) > 0.25


def test_discrete_feature_with_ties_and_unseen_value():
    ref = pd.Series(np.random.default_rng(4).integers(0, 2, 20000).astype(float))   # flag
    spec = psi.numeric_spec(ref)
    assert spec["kind"] == "discrete"
    assert psi.psi_against_spec(spec, ref) < 1e-9
    assert psi.psi_against_spec(spec, pd.Series(np.ones(20000))) > 0.25
    assert psi.psi_against_spec(spec, pd.Series(np.full(20000, 5.0))) > 0.25        # never-seen value


def test_heavily_tied_numeric_does_not_break_bins():
    x = np.concatenate([np.zeros(80000), np.random.default_rng(5).exponential(size=20000)])
    assert psi.psi_two_samples(x, x) < 1e-9
    assert psi.psi_two_samples(x, np.zeros(100000)) > 0.25


def test_categorical_psi_counts_unseen_codes():
    cats = ["1", "5", "missing"]
    ref = pd.Series([1.0] * 500 + [5.0] * 400 + [np.nan] * 100)
    spec = psi.categorical_spec(ref, "city", cats)
    assert psi.psi_against_spec(spec, ref, "city") < 1e-9
    moved = pd.Series([1.0] * 100 + [5.0] * 100 + [13.0] * 800)            # mostly an unseen city
    assert psi.psi_against_spec(spec, moved, "city") > 0.25


def test_status_thresholds():
    assert psi.status(0.05) == "OK"
    assert psi.status(0.10) == "WARN"
    assert psi.status(0.2499) == "WARN"
    assert psi.status(0.25) == "ALERT"
