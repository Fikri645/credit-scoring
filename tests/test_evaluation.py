"""Unit tests for business-cost thresholding and calibration utilities."""
import numpy as np

from src.evaluation import (
    optimal_threshold, cost_at_threshold, calibration_report, reliability_curve,
)


def _signal(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = np.clip(rng.normal(0.3 + 0.4 * y, 0.15), 0, 1)
    return y, p


def test_optimal_threshold_beats_half_on_cost():
    y, p = _signal()
    thr, cost_opt, cost_half = optimal_threshold(y, p, c_fn=5.0, c_fp=1.0)
    assert 0 < thr < 1
    assert cost_opt <= cost_half  # optimisation can never be worse than 0.5


def test_cost_monotonic_extremes():
    y, p = _signal()
    # Threshold 0 -> everyone flagged: zero FN, many FP.
    c0 = cost_at_threshold(y, p, 0.0, c_fn=1.0, c_fp=1.0)
    c1 = cost_at_threshold(y, p, 1.0, c_fn=1.0, c_fp=1.0)
    assert c0 > 0 and c1 > 0


def test_calibration_report_has_three_methods():
    y, p = _signal()
    half = len(y) // 2
    rep = calibration_report(y[half:], p[half:], y[:half], p[:half])
    assert set(rep["method"]) == {"raw", "platt", "isotonic"}
    assert (rep["ece"] >= 0).all()


def test_reliability_curve_shapes():
    y, p = _signal()
    mp, fp, counts = reliability_curve(y, p, n_bins=10)
    assert len(mp) == len(fp) == len(counts)
    assert counts.sum() == len(y)
