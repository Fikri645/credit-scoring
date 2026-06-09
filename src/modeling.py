"""Modelling utilities: time-aware splitting, preprocessing, and a
PyTorch-autograd focal-loss objective for LightGBM.

Why autograd for focal loss?  The focal-loss gradient/Hessian are tedious and
error-prone to derive by hand. Because the loss is separable across samples,
we let ``torch.autograd`` compute the exact per-sample gradient and the
diagonal Hessian — correct by construction, and a clean demonstration of
custom objectives. (Reference: Lin et al., *Focal Loss for Dense Object
Detection*, 2017; focal-aware cost-sensitive boosting for credit, 2022/2024.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


# --------------------------------------------------------------------------- #
# Time-aware split — critical for the Gini *stability* metric
# --------------------------------------------------------------------------- #
def time_split(df: pd.DataFrame, valid_frac: float = 0.2, test_frac: float = 0.2):
    """Split by ``WEEK_NUM`` so validation/test are strictly *later* than train.

    A random split would leak future information and inflate every metric;
    the stability metric in particular only means something on a forward
    time hold-out. Returns ``(train_df, valid_df, test_df)``.
    """
    weeks = np.sort(df[config.WEEK_NUM].unique())
    n = len(weeks)
    n_test = max(1, int(n * test_frac))
    n_valid = max(1, int(n * valid_frac))
    test_weeks = set(weeks[-n_test:])
    valid_weeks = set(weeks[-(n_test + n_valid):-n_test])

    test_df = df[df[config.WEEK_NUM].isin(test_weeks)].copy()
    valid_df = df[df[config.WEEK_NUM].isin(valid_weeks)].copy()
    train_df = df[~df[config.WEEK_NUM].isin(test_weeks | valid_weeks)].copy()
    return train_df, valid_df, test_df


# --------------------------------------------------------------------------- #
# Feature / categorical handling
# --------------------------------------------------------------------------- #
NON_FEATURE = {config.TARGET, config.CASE_ID, config.WEEK_NUM, config.DATE_DECISION,
               "MONTH"}


def split_feature_columns(df: pd.DataFrame):
    """Return ``(numeric_cols, categorical_cols)`` excluding ids/target."""
    feats = [c for c in df.columns if c not in NON_FEATURE]
    cat = [c for c in feats if df[c].dtype == "object" or str(df[c].dtype) == "category"]
    num = [c for c in feats if c not in cat]
    return num, cat


def prepare_categoricals(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Cast categorical columns to pandas ``category`` for native LightGBM use."""
    df = df.copy()
    for c in cat_cols:
        df[c] = df[c].astype("category")
    return df


# --------------------------------------------------------------------------- #
# Focal loss (PyTorch autograd) → LightGBM custom objective
# --------------------------------------------------------------------------- #
def make_focal_loss_objective(alpha: float = 0.25, gamma: float = 2.0):
    """Build a LightGBM ``fobj`` computing focal-loss grad/Hessian via autograd.

    ``alpha`` rebalances the rare positive (default) class; ``gamma`` focuses
    learning on hard, borderline applicants by down-weighting easy negatives.
    """
    import torch

    def _objective(y_pred: np.ndarray, dataset) -> tuple[np.ndarray, np.ndarray]:
        y_true = torch.as_tensor(dataset.get_label(), dtype=torch.float64)
        z = torch.as_tensor(y_pred, dtype=torch.float64).requires_grad_(True)

        p = torch.sigmoid(z)
        p_t = y_true * p + (1.0 - y_true) * (1.0 - p)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        loss = -alpha_t * (1.0 - p_t).pow(gamma) * torch.log(p_t.clamp_min(1e-9))

        (grad,) = torch.autograd.grad(loss.sum(), z, create_graph=True)
        # Loss is separable across samples ⇒ d(grad.sum())/dz is the diagonal Hessian.
        (hess,) = torch.autograd.grad(grad.sum(), z)
        return grad.detach().numpy(), hess.detach().numpy()

    return _objective


def focal_loss_eval(alpha: float = 0.25, gamma: float = 2.0):
    """Matching ``feval`` so LightGBM can early-stop on focal loss itself."""
    def _eval(y_pred: np.ndarray, dataset):
        y_true = dataset.get_label()
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p_t = np.where(y_true == 1, p, 1.0 - p)
        alpha_t = np.where(y_true == 1, alpha, 1.0 - alpha)
        loss = -alpha_t * np.power(1.0 - p_t, gamma) * np.log(np.clip(p_t, 1e-9, 1.0))
        return "focal_loss", float(loss.mean()), False  # lower is better

    return _eval


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Map raw LightGBM margins to probabilities (needed when using ``fobj``)."""
    return 1.0 / (1.0 + np.exp(-x))
