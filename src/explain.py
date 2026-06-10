"""Explainability: SHAP (global + local) and DiCE counterfactuals.

Under the EU AI Act, credit scoring is a high-risk system requiring technical
transparency, and GDPR Art. 22 grants a "right to explanation" for automated
decisions. We provide two complementary layers:

* **SHAP TreeExplainer** — *exact* Shapley values for tree ensembles
  (O(TLD^2)), giving global feature importance (beeswarm), per-applicant
  attributions (waterfall), and pairwise interaction values.
* **DiCE counterfactuals** — *actionable* recourse: the minimal change to an
  applicant's features that would flip a rejection to an approval
  ("if income were 15% higher you'd be approved"). This is what a rejected
  customer is legally entitled to know.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #
def shap_explainer(booster):
    """Return a SHAP TreeExplainer for a LightGBM booster / sklearn wrapper."""
    import shap

    return shap.TreeExplainer(booster)


def global_importance(explainer, X: pd.DataFrame, max_display: int = 20) -> pd.DataFrame:
    """Mean |SHAP| per feature — the global importance ranking."""
    sv = explainer.shap_values(X)
    if isinstance(sv, list):           # older SHAP returns [neg, pos]
        sv = sv[1]
    imp = np.abs(sv).mean(axis=0)
    return (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": imp})
        .sort_values("mean_abs_shap", ascending=False)
        .head(max_display)
        .reset_index(drop=True)
    )


def local_explanation(explainer, x_row: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """Signed SHAP contributions for a single applicant (waterfall data)."""
    sv = explainer.shap_values(x_row)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv).reshape(-1)
    out = pd.DataFrame({
        "feature": x_row.columns,
        "value": x_row.iloc[0].values,
        "shap": sv,
    })
    out["abs"] = out["shap"].abs()
    return (
        out.sort_values("abs", ascending=False)
        .head(top_k)
        .drop(columns="abs")
        .reset_index(drop=True)
    )


def top_interactions(model, X: pd.DataFrame) -> pd.DataFrame:
    """Rank pairwise SHAP interactions over a compact numeric driver block.

    Full SHAP interaction values are O(T·L·D²·F²) and the production booster
    requires *all* 730 features at once (you cannot hand SHAP a column subset),
    so exact interactions across the full space are intractable. We instead fit
    a small numeric **surrogate** on the dominant drivers (same pattern the DiCE
    recourse uses) and compute the exact interaction tensor on it. ``X`` is the
    already-subset numeric frame; ``model`` is the surrogate fitted on it.

    Returns the off-diagonal pairs ranked by mean |interaction| — the
    pairwise-interaction documentation the EU AI Act expects for a high-risk
    system.
    """
    import shap

    cols = list(X.columns)
    explainer = shap.TreeExplainer(model)
    inter = explainer.shap_interaction_values(X)
    if isinstance(inter, list):
        inter = inter[1]
    mean_abs = np.abs(np.asarray(inter)).mean(axis=0)   # (k, k)

    rows = []
    for a in range(len(cols)):
        for b in range(a + 1, len(cols)):               # off-diagonal pairs only
            rows.append({
                "feature_a": cols[a],
                "feature_b": cols[b],
                "mean_abs_interaction": float(mean_abs[a, b]),
            })
    return (
        pd.DataFrame(rows)
        .sort_values("mean_abs_interaction", ascending=False)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# DiCE counterfactuals
# --------------------------------------------------------------------------- #
def build_dice(model, X_train: pd.DataFrame, y_train, continuous_features: list[str],
               target_name: str = "target"):
    """Construct a DiCE explainer around a fitted sklearn-style model.

    ``model`` must expose ``predict_proba``. Returns a ``dice_ml.Dice`` object.
    """
    import dice_ml

    train_df = X_train.copy()
    train_df[target_name] = np.asarray(y_train)
    data = dice_ml.Data(
        dataframe=train_df,
        continuous_features=[c for c in continuous_features if c in X_train.columns],
        outcome_name=target_name,
    )
    dice_model = dice_ml.Model(model=model, backend="sklearn")
    return dice_ml.Dice(data, dice_model, method="random")


def generate_counterfactuals(dice, x_row: pd.DataFrame, total_cfs: int = 3,
                             desired_class: int = 0):
    """Find ``total_cfs`` actionable changes flipping the prediction.

    ``desired_class=0`` = "what would make this rejected applicant approved".
    Returns a DataFrame of counterfactual examples (or ``None`` if DiCE fails
    to find any within its search budget).
    """
    try:
        cf = dice.generate_counterfactuals(
            x_row, total_CFs=total_cfs, desired_class=desired_class
        )
        return cf.cf_examples_list[0].final_cfs_df
    except Exception as exc:  # DiCE can fail to find CFs for some rows
        print(f"[dice] no counterfactual found: {exc}")
        return None
