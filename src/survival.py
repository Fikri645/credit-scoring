"""Survival analysis: time-to-default with a Cox proportional-hazards model.

A binary "will they default?" classifier cannot answer *when*. Banks need the
timing for IFRS 9 Expected Credit Loss and Basel IRB lifetime-PD curves
(MDPI Risks, 2025). We fit a regularised Cox PH model on a compact feature set
to estimate each applicant's hazard over time and produce survival curves.

For the Home Credit data we construct a survival target from the available
signal: ``event = target`` (default observed) and a synthetic ``duration``
proxy derived from the application timeline (``WEEK_NUM``), since the
competition does not ship an explicit time-to-event. The module is written so
that swapping in a real ``duration`` column requires no code change.
"""
from __future__ import annotations

import pandas as pd


def make_survival_frame(df: pd.DataFrame, feature_cols: list[str],
                        duration_col: str, event_col: str) -> pd.DataFrame:
    """Assemble a clean ``[features..., duration, event]`` frame for lifelines.

    Drops rows with non-positive duration and median-imputes feature NaNs
    (Cox PH cannot consume missing values).
    """
    cols = list(feature_cols) + [duration_col, event_col]
    out = df[cols].copy()
    out = out[out[duration_col] > 0]
    for c in feature_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].fillna(out[c].median())
    out[event_col] = out[event_col].astype(int)
    return out


def fit_coxph(surv_df: pd.DataFrame, duration_col: str, event_col: str,
              penalizer: float = 0.1):
    """Fit a penalised Cox PH model; returns the fitted ``CoxPHFitter``."""
    from lifelines import CoxPHFitter

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(surv_df, duration_col=duration_col, event_col=event_col,
            show_progress=False)
    return cph


def concordance(cph) -> float:
    """Harrell's C-index of the fitted model (0.5 = random, 1.0 = perfect)."""
    return float(cph.concordance_index_)


def predict_survival_curve(cph, x_row: pd.DataFrame):
    """Return the predicted survival function S(t) for one applicant."""
    return cph.predict_survival_function(x_row)
