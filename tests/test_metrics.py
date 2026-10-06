import numpy as np
import pandas as pd

from fepp_cd.metrics import (
    bootstrap_ci, calibrated_operating_point, community_sensitivity, compute_metrics,
    expected_calibration_error, mcnemar, rationale_prf, tmr_far,
)


def test_tmr_far():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
    tmr, far = tmr_far(y, p)
    assert tmr == 0.25 and abs(far - 2 / 6) < 1e-12


def test_compute_metrics_counts():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, np.array([0, 1, 1, 0]), np.array([0.1, 0.6, 0.9, 0.4]))
    assert (m.TN, m.FP, m.FN, m.TP) == (1, 1, 1, 1) and m.accuracy == 0.5


def test_calibrated_operating_point_meets_target_and_is_max():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    s = np.clip(0.5 * y + rng.normal(0.25, 0.2, 2000), 0, 1)
    for t in (0.02, 0.05, 0.1):
        op = calibrated_operating_point(y, s, t)
        assert op["TMR"] <= t
        higher = np.unique(s[s > op["tau"]])
        if len(higher):
            assert ((s[y == 1] < higher.min()).mean()) > t  # any larger threshold violates


def test_ece_perfect_and_bad():
    y = np.array([0, 1] * 500)
    assert expected_calibration_error(y, np.full(1000, 0.5)) < 1e-9
    assert expected_calibration_error(y, np.where(y == 1, 0.0, 1.0)) > 0.99


def test_mcnemar():
    y = np.zeros(10, dtype=int)
    a = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
    b = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1])
    r = mcnemar(y, a, b)
    assert (r["b"], r["c"]) == (5, 2)
    assert abs(r["chi2"] - 4 / 7) < 1e-12


def test_bootstrap_is_deterministic():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300); s = rng.random(300); p = (s > 0.5).astype(int)
    assert bootstrap_ci(y, p, s, B=50) == bootstrap_ci(y, p, s, B=50)


def test_community_and_rationale():
    df = pd.DataFrame({"targets": ["['A']", "['A', 'B']", "[]"], "y_true": [1, 0, 1], "y_pred": [1, 1, 0]})
    cs = community_sensitivity(df, top_k=None).set_index("community")
    assert cs.loc["A", "n"] == 2 and cs.loc["unspecified", "n"] == 1
    assert rationale_prf([1, 2], [2, 3]) == (0.5, 0.5, 0.5)
