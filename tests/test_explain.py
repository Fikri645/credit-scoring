"""Tests for the explainability helpers that don't need the trained booster."""
import numpy as np
import pandas as pd
import pytest


def test_top_interactions_ranks_pairs():
    """top_interactions returns ranked off-diagonal pairs over a small model."""
    lgb = pytest.importorskip("lightgbm")
    pytest.importorskip("shap")
    from src.explain import top_interactions

    rng = np.random.default_rng(0)
    n = 400
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # y depends on an x0*x1 interaction, so that pair should rank highly
    logit = 2.0 * x0 * x1 + 0.3 * x2
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2})

    model = lgb.LGBMClassifier(n_estimators=60, num_leaves=15, verbose=-1)
    model.fit(X, y)

    out = top_interactions(model, X)

    # 3 features -> 3 unique off-diagonal pairs, ranked descending
    assert list(out.columns) == ["feature_a", "feature_b", "mean_abs_interaction"]
    assert len(out) == 3
    assert out["mean_abs_interaction"].is_monotonic_decreasing
    assert (out["mean_abs_interaction"] >= 0).all()
    # the engineered x0 x1 interaction should be the strongest pair
    top = {out.iloc[0]["feature_a"], out.iloc[0]["feature_b"]}
    assert top == {"x0", "x1"}
