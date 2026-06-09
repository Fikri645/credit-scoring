"""FastAPI scoring service.

Exposes ``POST /score`` which accepts a (partial) applicant feature mapping,
fills missing features with the training schema's defaults, returns the
probability of default, the business-cost decision, and the top SHAP factors
driving the decision.

Run:  ``uvicorn api.main:app --reload``
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src import config

app = FastAPI(title="Credit Scoring API", version="1.0.0")

_MODEL: lgb.Booster | None = None
_SCHEMA: dict | None = None
_THRESHOLD: float = 0.5
_EXPLAINER = None


def _load_artifacts():
    global _MODEL, _SCHEMA, _THRESHOLD, _EXPLAINER
    _MODEL = lgb.Booster(model_file=str(config.MODELS_DIR / "model_lgbm.txt"))
    _SCHEMA = joblib.load(config.MODELS_DIR / "feature_schema.joblib")
    thr_path = config.MODELS_DIR / "analysis_threshold.json"
    if thr_path.exists():
        _THRESHOLD = json.loads(thr_path.read_text())["threshold"]
    try:
        import shap
        _EXPLAINER = shap.TreeExplainer(_MODEL)
    except Exception:
        _EXPLAINER = None


@app.on_event("startup")
def _startup():
    if (config.MODELS_DIR / "model_lgbm.txt").exists():
        _load_artifacts()


class ScoreRequest(BaseModel):
    features: dict[str, Any]


class ScoreResponse(BaseModel):
    probability_of_default: float
    decision: str
    threshold: float
    top_factors: list[dict[str, Any]]


def _build_row(features: dict[str, Any]) -> pd.DataFrame:
    cols = _SCHEMA["feat_cols"]
    row = {c: features.get(c, np.nan) for c in cols}
    df = pd.DataFrame([row])
    for c in _SCHEMA["cat_cols"]:
        df[c] = df[c].astype("category")
    return df[cols]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _MODEL is not None}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    if _MODEL is None:
        _load_artifacts()
    df = _build_row(req.features)
    prob = float(_MODEL.predict(df)[0])
    decision = "decline" if prob >= _THRESHOLD else "approve"

    factors: list[dict[str, Any]] = []
    if _EXPLAINER is not None:
        sv = _EXPLAINER.shap_values(df)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv).reshape(-1)
        order = np.argsort(np.abs(sv))[::-1][:8]
        for i in order:
            factors.append({"feature": _SCHEMA["feat_cols"][i],
                            "shap": float(sv[i]),
                            "value": _to_native(df.iloc[0, i])})

    return ScoreResponse(
        probability_of_default=prob,
        decision=decision,
        threshold=_THRESHOLD,
        top_factors=factors,
    )


def _to_native(v):
    if isinstance(v, (np.generic,)):
        return v.item()
    if pd.isna(v):
        return None
    return v


# Allow ``python -m api.main`` for a quick local run.
if __name__ == "__main__":
    import uvicorn
    if not Path(config.MODELS_DIR / "model_lgbm.txt").exists():
        raise SystemExit("Model not trained yet — run `python -m src.train` first.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
