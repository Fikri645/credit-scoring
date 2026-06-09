"""Business-cost threshold optimisation and probability calibration.

Two ideas that separate a production credit model from a Kaggle submission:

1. **Cost-optimal threshold.** A credit model is a *cost minimiser*, not an
   accuracy maximiser. A missed default loses the exposure; a wrongly rejected
   good customer loses the loan margin. The optimal cut-off is the one that
   minimises expected cost, not the default 0.5.

2. **Calibration.** Pricing needs *probabilities*, not just a ranking
   (Expected Loss = PD x LGD x EAD). We compare Platt scaling (sigmoid) with
   isotonic regression and report the Expected Calibration Error — isotonic
   tends to win on large calibration sets (arXiv 2601.19944, 2026).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src import config
from src.metrics import expected_calibration_error


# --------------------------------------------------------------------------- #
# Business-cost threshold
# --------------------------------------------------------------------------- #
def cost_at_threshold(y_true, y_prob, threshold, c_fn=None, c_fp=None) -> float:
    c_fn = config.COST_FALSE_NEGATIVE if c_fn is None else c_fn
    c_fp = config.COST_FALSE_POSITIVE if c_fp is None else c_fp
    y_true = np.asarray(y_true)
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    fn = int(((pred == 0) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    return c_fn * fn + c_fp * fp


def optimal_threshold(y_true, y_prob, c_fn=None, c_fp=None, n_steps: int = 200):
    """Grid-search the threshold minimising expected business cost.

    Returns ``(best_threshold, best_cost, cost_at_half)``.
    """
    thresholds = np.linspace(0.001, 0.999, n_steps)
    costs = [cost_at_threshold(y_true, y_prob, t, c_fn, c_fp) for t in thresholds]
    best_idx = int(np.argmin(costs))
    return (
        float(thresholds[best_idx]),
        float(costs[best_idx]),
        cost_at_threshold(y_true, y_prob, 0.5, c_fn, c_fp),
    )


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def fit_platt(y_valid, p_valid):
    """Platt scaling: a 1-D logistic regression on the raw scores."""
    lr = LogisticRegression(C=1e10, solver="lbfgs")
    lr.fit(np.asarray(p_valid).reshape(-1, 1), np.asarray(y_valid))
    return lr


def apply_platt(lr, p):
    return lr.predict_proba(np.asarray(p).reshape(-1, 1))[:, 1]


def fit_isotonic(y_valid, p_valid):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.asarray(p_valid), np.asarray(y_valid))
    return iso


def apply_isotonic(iso, p):
    return iso.predict(np.asarray(p))


def calibration_report(y_true, p_raw, y_calib, p_calib) -> pd.DataFrame:
    """Compare raw vs Platt vs isotonic on Brier and ECE.

    ``(y_calib, p_calib)`` is the calibration fit set; ``(y_true, p_raw)`` is
    the held-out test set on which all three variants are scored.
    """
    platt = fit_platt(y_calib, p_calib)
    iso = fit_isotonic(y_calib, p_calib)

    variants = {
        "raw": np.asarray(p_raw),
        "platt": apply_platt(platt, p_raw),
        "isotonic": apply_isotonic(iso, p_raw),
    }
    rows = []
    for name, p in variants.items():
        rows.append({
            "method": name,
            "brier": brier_score_loss(y_true, p),
            "ece": expected_calibration_error(y_true, p),
        })
    return pd.DataFrame(rows)


def reliability_curve(y_true, y_prob, n_bins: int = 10):
    """Return ``(mean_predicted, fraction_positive, bin_counts)`` for plotting."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    mean_pred, frac_pos, counts = [], [], []
    for b in range(n_bins):
        mask = ids == b
        if not mask.any():
            continue
        mean_pred.append(y_prob[mask].mean())
        frac_pos.append(y_true[mask].mean())
        counts.append(int(mask.sum()))
    return np.array(mean_pred), np.array(frac_pos), np.array(counts)
