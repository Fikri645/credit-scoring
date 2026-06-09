"""Fairness audit for the credit model (Fairlearn).

Credit scoring is a textbook setting for algorithmic fairness: lending
decisions are regulated (EU AI Act Annex III lists credit scoring as a
*high-risk* system; Indonesia's OJK increasingly expects explainable,
non-discriminatory lending). We audit on **gender** as the protected
attribute and report:

* **Demographic parity difference** — gap in approval (positive-prediction)
  rate across groups.
* **Equalized odds difference** — gap in TPR/FPR across groups.

A key, honest point we surface: these criteria are **mathematically
incompatible** when base rates differ (Kleinberg et al., 2016) — you cannot
satisfy both at once, so the choice is a policy decision, not a technical one.
``Fairlearn``'s ``ThresholdOptimizer`` then shows post-hoc per-group
threshold adjustment that equalises a chosen criterion *without retraining*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fairness_metrics(y_true, y_pred, sensitive) -> dict:
    """Compute demographic-parity and equalized-odds differences."""
    from fairlearn.metrics import (
        demographic_parity_difference,
        equalized_odds_difference,
        MetricFrame,
        selection_rate,
        true_positive_rate,
        false_positive_rate,
    )

    mf = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "tpr": true_positive_rate,
            "fpr": false_positive_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )
    return {
        "demographic_parity_diff": float(
            demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)
        ),
        "equalized_odds_diff": float(
            equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive)
        ),
        "by_group": mf.by_group.reset_index().to_dict(orient="records"),
    }


def mitigate_with_threshold_optimizer(
    estimator, X, y, sensitive, constraint: str = "equalized_odds", random_state: int = 42
):
    """Fit a Fairlearn ``ThresholdOptimizer`` wrapping a trained estimator.

    ``constraint`` is one of ``"demographic_parity"`` or ``"equalized_odds"``.
    The optimizer picks per-group thresholds that satisfy the constraint while
    maximising accuracy — a post-processing fix needing no model retraining.
    """
    from fairlearn.postprocessing import ThresholdOptimizer

    opt = ThresholdOptimizer(
        estimator=estimator,
        constraints=constraint,
        objective="accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    opt.fit(X, y, sensitive_features=sensitive)
    return opt


def audit_table(y_true, y_prob, sensitive, threshold: float) -> pd.DataFrame:
    """Per-group selection / TPR / FPR table at a given decision threshold."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    df = pd.DataFrame({"y": np.asarray(y_true), "pred": y_pred, "g": np.asarray(sensitive)})
    rows = []
    for g, sub in df.groupby("g"):
        pos = sub["y"] == 1
        neg = sub["y"] == 0
        rows.append({
            "group": g,
            "n": len(sub),
            "selection_rate": sub["pred"].mean(),
            "tpr": sub.loc[pos, "pred"].mean() if pos.any() else np.nan,
            "fpr": sub.loc[neg, "pred"].mean() if neg.any() else np.nan,
            "base_rate": sub["y"].mean(),
        })
    return pd.DataFrame(rows)
