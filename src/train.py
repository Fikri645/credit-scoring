"""End-to-end training orchestrator.

Loads the engineered feature table, makes a forward-in-time split, then trains
and compares three gradient-boosting configurations under the Gini *stability*
metric:

* **LightGBM (class_weight)**  — cost-sensitive baseline.
* **LightGBM (focal loss)**    — custom autograd objective, no SMOTE.
* **CatBoost (auto_class_weights)** — categorical-native comparison.

Everything is logged to MLflow; the winning booster, the fitted feature
schema, and a metrics summary are saved to ``models/`` for the serving layer
and the analysis notebooks (explain / fairness / calibration / survival).

Run:  ``python -m src.train``
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd

from src import config
from src.metrics import gini_stability_metric
from src.modeling import (
    time_split, split_feature_columns, prepare_categoricals,
    make_focal_loss_objective, focal_loss_eval, sigmoid,
)


def _score_frame(df, y, prob):
    """Build the [WEEK_NUM, target, score] frame the stability metric needs."""
    return pd.DataFrame({
        config.WEEK_NUM: df[config.WEEK_NUM].to_numpy(),
        "target": np.asarray(y),
        "score": np.asarray(prob),
    })


def train_lightgbm_weighted(X_tr, y_tr, X_va, y_va, cat_cols):
    import lightgbm as lgb

    params = dict(
        objective="binary", metric="auc", learning_rate=0.03,
        num_leaves=64, feature_fraction=0.7, bagging_fraction=0.7,
        bagging_freq=1, min_data_in_leaf=200, lambda_l2=2.0,
        is_unbalance=True, verbosity=-1, seed=config.RANDOM_STATE,
    )
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    model = lgb.train(
        params, dtr, num_boost_round=2000, valid_sets=[dva],
        callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)],
    )
    return model


def train_lightgbm_focal(X_tr, y_tr, X_va, y_va, cat_cols, alpha=0.25, gamma=2.0):
    import lightgbm as lgb

    # LightGBM >= 4.0 removed the ``fobj=`` argument: a custom objective is
    # passed through ``params["objective"]`` as a callable instead.
    params = dict(
        objective=make_focal_loss_objective(alpha, gamma),
        learning_rate=0.03, num_leaves=64, feature_fraction=0.7,
        bagging_fraction=0.7, bagging_freq=1, min_data_in_leaf=200,
        lambda_l2=2.0, verbosity=-1, seed=config.RANDOM_STATE,
    )
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    model = lgb.train(
        params, dtr, num_boost_round=2000, valid_sets=[dva],
        feval=focal_loss_eval(alpha, gamma),
        callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)],
    )
    return model


def train_catboost(X_tr, y_tr, X_va, y_va, cat_cols):
    from catboost import CatBoostClassifier, Pool

    # CatBoost needs categoricals as strings with no NaN.
    def _prep(X):
        X = X.copy()
        for c in cat_cols:
            X[c] = X[c].astype("object").where(X[c].notna(), "NA").astype(str)
        return X

    Xtr, Xva = _prep(X_tr), _prep(X_va)
    ptr = Pool(Xtr, y_tr, cat_features=cat_cols)
    pva = Pool(Xva, y_va, cat_features=cat_cols)
    model = CatBoostClassifier(
        iterations=2000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
        eval_metric="AUC", auto_class_weights="Balanced",
        random_seed=config.RANDOM_STATE, verbose=0, early_stopping_rounds=120,
    )
    model.fit(ptr, eval_set=pva)
    return model, _prep


def main():
    import mlflow

    t0 = time.time()
    path = config.PROCESSED_DIR / "train_features.parquet"
    print(f"[train] loading {path}")
    df = pd.read_parquet(path)

    if config.SAMPLE_FRAC < 1.0:
        df = df.sample(frac=config.SAMPLE_FRAC, random_state=config.RANDOM_STATE)
        print(f"[train] sampled to {df.shape}")

    num_cols, cat_cols = split_feature_columns(df)
    df = prepare_categoricals(df, cat_cols)
    feat_cols = num_cols + cat_cols

    tr, va, te = time_split(df)
    print(f"[train] split train={tr.shape} valid={va.shape} test={te.shape}")
    y_tr, y_va, y_te = tr[config.TARGET], va[config.TARGET], te[config.TARGET]

    mlflow.set_experiment("credit-scoring")
    results = {}
    models = {}

    # --- 1. LightGBM weighted ------------------------------------------------
    with mlflow.start_run(run_name="lgbm_weighted"):
        m = train_lightgbm_weighted(tr[feat_cols], y_tr, va[feat_cols], y_va, cat_cols)
        prob = m.predict(te[feat_cols])
        s = gini_stability_metric(_score_frame(te, y_te, prob))
        mlflow.log_metrics(s)
        results["lgbm_weighted"] = s
        models["lgbm_weighted"] = m
        print(f"[train] lgbm_weighted  stability={s['stability']:.4f} "
              f"mean_gini={s['mean_gini']:.4f}")

    # --- 2. LightGBM focal loss ---------------------------------------------
    with mlflow.start_run(run_name="lgbm_focal"):
        m = train_lightgbm_focal(tr[feat_cols], y_tr, va[feat_cols], y_va, cat_cols)
        prob = sigmoid(m.predict(te[feat_cols]))  # fobj outputs raw margins
        s = gini_stability_metric(_score_frame(te, y_te, prob))
        mlflow.log_params({"alpha": 0.25, "gamma": 2.0})
        mlflow.log_metrics(s)
        results["lgbm_focal"] = s
        models["lgbm_focal"] = m
        print(f"[train] lgbm_focal     stability={s['stability']:.4f} "
              f"mean_gini={s['mean_gini']:.4f}")

    # --- 3. CatBoost ---------------------------------------------------------
    with mlflow.start_run(run_name="catboost"):
        m, prep = train_catboost(tr[feat_cols], y_tr, va[feat_cols], y_va, cat_cols)
        prob = m.predict_proba(prep(te[feat_cols]))[:, 1]
        s = gini_stability_metric(_score_frame(te, y_te, prob))
        mlflow.log_metrics(s)
        results["catboost"] = s
        models["catboost"] = m
        print(f"[train] catboost       stability={s['stability']:.4f} "
              f"mean_gini={s['mean_gini']:.4f}")

    # --- 4. FT-Transformer (tabular DL comparison) --------------------------
    # Honest baseline: modern tabular DL vs gradient boosting. Guarded so a GPU
    # OOM or missing CUDA never breaks the canonical GBT bake-off.
    try:
        from src.ft_transformer import train_ft_transformer
        with mlflow.start_run(run_name="ft_transformer"):
            predict_fn, hist = train_ft_transformer(
                tr[feat_cols], y_tr, va[feat_cols], y_va, num_cols, epochs=8)
            prob = predict_fn(te[feat_cols])
            s = gini_stability_metric(_score_frame(te, y_te, prob))
            mlflow.log_metrics(s)
            results["ft_transformer"] = s
            print(f"[train] ft_transformer  stability={s['stability']:.4f} "
                  f"mean_gini={s['mean_gini']:.4f}")
    except Exception as e:  # noqa: BLE001
        print(f"[train] ft_transformer skipped: {e}")

    # --- pick & persist the winner ------------------------------------------
    best = max(results, key=lambda k: results[k]["stability"])
    print(f"[train] BEST = {best}  ({results[best]['stability']:.4f})")

    if best == "lgbm_weighted":
        models[best].save_model(str(config.MODELS_DIR / "model_lgbm.txt"))
    elif best == "lgbm_focal":
        models[best].save_model(str(config.MODELS_DIR / "model_lgbm_focal.txt"))
    else:
        models[best].save_model(str(config.MODELS_DIR / "model_catboost.cbm"))

    # Always persist the weighted LightGBM for the SHAP/serving layer (native
    # probability output + TreeExplainer support).
    models["lgbm_weighted"].save_model(str(config.MODELS_DIR / "model_lgbm.txt"))

    joblib.dump({"num_cols": num_cols, "cat_cols": cat_cols, "feat_cols": feat_cols},
                config.MODELS_DIR / "feature_schema.joblib")
    with open(config.MODELS_DIR / "metrics.json", "w") as f:
        json.dump({"results": results, "best": best}, f, indent=2)

    # Persist the forward-in-time TEST split so the analysis layer
    # (explain / fairness / calibration / survival) and the Gradio demo all
    # evaluate on the exact same held-out rows the metrics were computed on.
    keep = feat_cols + [config.TARGET, config.WEEK_NUM, config.CASE_ID]
    keep = [c for c in keep if c in te.columns]
    te[keep].to_parquet(config.ARTIFACTS_DIR / "test_split.parquet")
    tr[keep].sample(min(len(tr), 50_000), random_state=config.RANDOM_STATE).to_parquet(
        config.ARTIFACTS_DIR / "train_sample.parquet")
    te[keep].sample(min(len(te), 300), random_state=config.RANDOM_STATE).to_parquet(
        config.ARTIFACTS_DIR / "demo_applicants.parquet")

    print(f"[train] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
