"""Gradio demo for the credit-scoring model.

Because the engineered feature space has hundreds of columns, the demo lets a
reviewer pick a *real held-out applicant* from the test split, then shows:

* the predicted probability of default and the business-cost decision,
* the top SHAP factors driving that decision (signed contributions),
* the applicant's true outcome (for transparency).

Run:  ``python app.py``  (serves on http://localhost:7860)
"""
from __future__ import annotations

import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config

_MODEL = lgb.Booster(model_file=str(config.MODELS_DIR / "model_lgbm.txt"))
_SCHEMA = joblib.load(config.MODELS_DIR / "feature_schema.joblib")
_FEAT = [c for c in _SCHEMA["feat_cols"]]
_THRESHOLD = 0.5
_thr_path = config.MODELS_DIR / "analysis_threshold.json"
if _thr_path.exists():
    _THRESHOLD = json.loads(_thr_path.read_text())["threshold"]

_DEMO = pd.read_parquet(config.ARTIFACTS_DIR / "demo_applicants.parquet").reset_index(drop=True)
for _c in _SCHEMA["cat_cols"]:
    if _c in _DEMO.columns:
        _DEMO[_c] = _DEMO[_c].astype("category")

try:
    import shap
    _EXPL = shap.TreeExplainer(_MODEL)
except Exception:
    _EXPL = None


def score_applicant(idx: int):
    idx = int(idx) % len(_DEMO)
    row = _DEMO.iloc[[idx]]
    X = row[[c for c in _FEAT if c in row.columns]]
    prob = float(_MODEL.predict(X)[0])
    decision = "🔴 DECLINE" if prob >= _THRESHOLD else "🟢 APPROVE"
    true = int(row[config.TARGET].iloc[0]) if config.TARGET in row else -1

    summary = (
        f"## {decision}\n"
        f"- **Probability of default:** {prob:.2%}\n"
        f"- **Decision threshold (cost-optimal):** {_THRESHOLD:.3f}\n"
        f"- **True outcome (held-out):** {'defaulted' if true == 1 else 'repaid'}\n"
    )

    factors = pd.DataFrame(columns=["feature", "value", "shap_contribution"])
    if _EXPL is not None:
        sv = _EXPL.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv).reshape(-1)
        order = np.argsort(np.abs(sv))[::-1][:10]
        factors = pd.DataFrame({
            "feature": [X.columns[i] for i in order],
            "value": [_native(X.iloc[0, i]) for i in order],
            "shap_contribution": [round(float(sv[i]), 4) for i in order],
        })
    return summary, factors


def _native(v):
    if isinstance(v, np.generic):
        return v.item()
    if pd.isna(v):
        return None
    return v


def build_ui():
    import gradio as gr

    with gr.Blocks(title="Credit Scoring — Home Credit 2024") as demo:
        gr.Markdown(
            "# 🏦 Credit Default Scoring\n"
            "LightGBM on the Home Credit 2024 *Credit Risk Model Stability* "
            "dataset, scored under the Gini-stability metric, with a "
            "cost-optimal decision threshold and SHAP explanations. "
            "Pick a held-out applicant to score."
        )
        with gr.Row():
            idx = gr.Slider(0, len(_DEMO) - 1, value=0, step=1, label="Applicant #")
        btn = gr.Button("Score applicant", variant="primary")
        out_md = gr.Markdown()
        out_tbl = gr.Dataframe(label="Top SHAP factors", wrap=True)

        btn.click(score_applicant, inputs=idx, outputs=[out_md, out_tbl])
        idx.release(score_applicant, inputs=idx, outputs=[out_md, out_tbl])
    return demo


if __name__ == "__main__":
    build_ui().launch()
