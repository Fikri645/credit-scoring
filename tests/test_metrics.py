"""Unit tests for the Gini-stability metric and ECE."""
import numpy as np
import pandas as pd

from src.metrics import gini_stability_metric, weekly_gini, expected_calibration_error


def _make_df(n_weeks=10, n_per=500, sep=2.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_weeks):
        y = rng.integers(0, 2, n_per)
        score = rng.normal(y * sep, 1.0)  # separable-ish
        rows.append(pd.DataFrame({"WEEK_NUM": w, "target": y, "score": score}))
    return pd.concat(rows, ignore_index=True)


def test_weekly_gini_range():
    df = _make_df()
    g = weekly_gini(df, "WEEK_NUM", "target", "score")
    assert (g >= -1).all() and (g <= 1).all()
    assert len(g) == 10


def test_stability_components_present():
    df = _make_df()
    out = gini_stability_metric(df)
    for k in ("stability", "mean_gini", "slope", "res_std"):
        assert k in out
    # a separable signal should have positive mean Gini
    assert out["mean_gini"] > 0.2


def test_stability_penalises_downtrend():
    # Build a model whose Gini falls over time -> stability < mean_gini.
    rng = np.random.default_rng(1)
    rows = []
    for w in range(10):
        y = rng.integers(0, 2, 500)
        sep = 3.0 - 0.25 * w          # signal degrades each week
        score = rng.normal(y * sep, 1.0)
        rows.append(pd.DataFrame({"WEEK_NUM": w, "target": y, "score": score}))
    df = pd.concat(rows, ignore_index=True)
    out = gini_stability_metric(df)
    assert out["slope"] < 0
    assert out["stability"] < out["mean_gini"]


def test_ece_perfect_calibration_low():
    # Predictions equal to true bin frequencies -> low ECE.
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.05
