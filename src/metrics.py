"""Evaluation metrics for credit scoring.

The headline metric is the **Gini stability metric** introduced by the
Home Credit *Credit Risk Model Stability* (2024) competition. Unlike plain
AUC, it rewards a model whose discriminative power stays *stable* over time:
it averages the weekly Gini, then penalises any downward trend and any
week-to-week volatility. This directly encodes the regulator concern that a
model must not silently decay after deployment.

    stability = mean(gini_w)  +  88.0 * min(0, slope)  -  0.5 * std(residuals)

where ``gini_w = 2 * AUC_w - 1`` is computed per ``WEEK_NUM`` bucket, and
``slope``/``residuals`` come from an OLS fit of ``gini_w`` against week index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def weekly_gini(df: pd.DataFrame, week_col: str, y_col: str, score_col: str) -> pd.Series:
    """Return the Gini (2*AUC-1) for each week bucket, ordered by week."""
    def _gini(group: pd.DataFrame) -> float:
        # A week with a single class is undefined for AUC — skip it.
        if group[y_col].nunique() < 2:
            return np.nan
        return 2.0 * roc_auc_score(group[y_col], group[score_col]) - 1.0

    out = (
        df[[week_col, y_col, score_col]]
        .sort_values(week_col)
        .groupby(week_col)
        .apply(_gini, include_groups=False)
    )
    return out.dropna()


def gini_stability_metric(
    df: pd.DataFrame,
    week_col: str = "WEEK_NUM",
    y_col: str = "target",
    score_col: str = "score",
    w_falling_rate: float = 88.0,
    w_res_std: float = -0.5,
) -> dict:
    """Compute the competition's Gini stability metric and its components.

    Returns a dict with ``stability``, ``mean_gini``, ``slope`` (trend in
    Gini per week — negative is bad) and ``res_std`` (volatility).
    """
    gini_in_time = weekly_gini(df, week_col, y_col, score_col).to_numpy()
    if len(gini_in_time) < 2:
        # Not enough time buckets to fit a trend — fall back to mean Gini.
        mean_gini = float(np.mean(gini_in_time)) if len(gini_in_time) else 0.0
        return {"stability": mean_gini, "mean_gini": mean_gini, "slope": 0.0, "res_std": 0.0}

    x = np.arange(len(gini_in_time))
    slope, intercept = np.polyfit(x, gini_in_time, 1)
    y_hat = slope * x + intercept
    residuals = gini_in_time - y_hat
    res_std = float(np.std(residuals))
    mean_gini = float(np.mean(gini_in_time))

    stability = mean_gini + w_falling_rate * min(0.0, float(slope)) + w_res_std * res_std
    return {
        "stability": float(stability),
        "mean_gini": mean_gini,
        "slope": float(slope),
        "res_std": res_std,
    }


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) with equal-width bins.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|.  A perfectly calibrated
    model has ECE = 0. Reported alongside the Brier score because Brier
    conflates calibration and refinement, while ECE isolates calibration.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)
