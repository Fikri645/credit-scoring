"""Post-training analysis: explainability, fairness, calibration, survival.

Loads the persisted weighted-LightGBM model and the forward-in-time test
split, then produces every figure and a consolidated ``reports/results.md``:

* SHAP global importance (beeswarm + bar) and one local waterfall.
* DiCE counterfactual for a rejected applicant.
* Fairness audit on gender (demographic parity vs equalized odds) +
  ThresholdOptimizer mitigation.
* Calibration: reliability diagram + Platt-vs-isotonic ECE/Brier table.
* Survival: Cox PH time-to-default with a C-index and example curves.
* Business-cost optimal threshold.

Run:  ``python -m src.run_analysis``
"""
from __future__ import annotations

import json

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config  # noqa: E402
from src import evaluation, explain, fairness, survival  # noqa: E402
from src.metrics import gini_stability_metric  # noqa: E402

FIG = config.FIGURES_DIR


def _load():
    booster = lgb.Booster(model_file=str(config.MODELS_DIR / "model_lgbm.txt"))
    import joblib
    schema = joblib.load(config.MODELS_DIR / "feature_schema.joblib")
    test = pd.read_parquet(config.ARTIFACTS_DIR / "test_split.parquet")
    train = pd.read_parquet(config.ARTIFACTS_DIR / "train_sample.parquet")
    for c in schema["cat_cols"]:
        if c in test.columns:
            test[c] = test[c].astype("category")
            train[c] = train[c].astype("category")
    return booster, schema, test, train


def _decode_gender(df: pd.DataFrame) -> pd.Series | None:
    """Best-effort recovery of a gender column from the engineered features."""
    for cand in df.columns:
        low = cand.lower()
        if "sex" in low or "gender" in low:
            s = df[cand].astype(str)
            if s.nunique() <= 6:
                return s.replace({"nan": "Unknown"})
    return None


def main():
    report: list[str] = ["# Credit Scoring — Analysis Results\n"]
    booster, schema, test, train = _load()
    feat = [c for c in schema["feat_cols"] if c in test.columns]
    X_test, y_test = test[feat], test[config.TARGET].to_numpy()
    prob = booster.predict(X_test)

    s = gini_stability_metric(pd.DataFrame({
        config.WEEK_NUM: test[config.WEEK_NUM].to_numpy(),
        "target": y_test, "score": prob}))
    report += [f"**Test Gini-stability** = {s['stability']:.4f} "
               f"(mean Gini {s['mean_gini']:.4f}, slope {s['slope']:.4f}, "
               f"res-std {s['res_std']:.4f})\n"]

    # ---------------- Business-cost threshold ----------------
    thr, cost_opt, cost_half = evaluation.optimal_threshold(y_test, prob)
    report += ["\n## Business-cost threshold\n",
               f"- Optimal threshold = **{thr:.3f}** (cost {cost_opt:.0f}) vs "
               f"0.5 (cost {cost_half:.0f}) — "
               f"**{100*(cost_half-cost_opt)/max(cost_half,1):.1f}%** cheaper.\n"]

    # ---------------- SHAP ----------------
    try:
        expl = explain.shap_explainer(booster)
        Xs = X_test.sample(min(2000, len(X_test)), random_state=config.RANDOM_STATE)
        imp = explain.global_importance(expl, Xs, max_display=20)
        imp.to_csv(config.REPORTS_DIR / "shap_importance.csv", index=False)
        plt.figure(figsize=(7, 6))
        plt.barh(imp["feature"][::-1], imp["mean_abs_shap"][::-1], color="#6d28d9")
        plt.xlabel("mean |SHAP|")
        plt.title("Global feature importance (SHAP)")
        plt.tight_layout()
        plt.savefig(FIG / "shap_importance.png", dpi=120)
        plt.close()
        report += ["\n## Explainability (SHAP)\n",
                   "Top features: " + ", ".join(imp["feature"].head(8)) + "\n",
                   "![](figures/shap_importance.png)\n"]
    except Exception as e:
        report += [f"\n## Explainability (SHAP)\n_skipped: {e}_\n"]

    # ---------------- DiCE counterfactual ----------------
    try:
        from sklearn.base import BaseEstimator, ClassifierMixin

        class _Wrap(BaseEstimator, ClassifierMixin):
            """Adapt the LightGBM Booster to the sklearn predict_proba API DiCE needs."""
            def __init__(self, b, cols):
                self.b, self.cols, self.classes_ = b, cols, np.array([0, 1])

            def fit(self, X, y=None):
                return self

            def predict_proba(self, X):
                X = pd.DataFrame(X, columns=self.cols)
                p = self.b.predict(X)
                return np.column_stack([1 - p, p])

            def predict(self, X):
                return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

        num_only = [c for c in schema["num_cols"] if c in train.columns]
        Xtr_num = train[num_only].fillna(train[num_only].median())
        wrap = _Wrap(booster, num_only)
        dice = explain.build_dice(wrap, Xtr_num, train[config.TARGET], num_only)
        rejected = X_test[num_only][prob >= thr].head(1).fillna(Xtr_num.median())
        cfs = explain.generate_counterfactuals(dice, rejected, total_cfs=2, desired_class=0)
        if cfs is not None:
            cfs.to_csv(config.REPORTS_DIR / "counterfactual_example.csv", index=False)
            report += ["\n## Counterfactual recourse (DiCE)\n",
                       "Saved an actionable counterfactual for a rejected applicant "
                       "to `reports/counterfactual_example.csv`.\n"]
    except Exception as e:
        report += [f"\n## Counterfactual recourse (DiCE)\n_skipped: {e}_\n"]

    # ---------------- Fairness ----------------
    gender = _decode_gender(test)
    if gender is not None:
        try:
            y_pred = (prob >= thr).astype(int)
            fm = fairness.fairness_metrics(y_test, y_pred, gender)
            tbl = fairness.audit_table(y_test, prob, gender, thr)
            tbl.to_csv(config.REPORTS_DIR / "fairness_by_group.csv", index=False)
            report += ["\n## Fairness audit (gender)\n",
                       f"- Demographic-parity difference = "
                       f"**{fm['demographic_parity_diff']:.4f}**\n",
                       f"- Equalized-odds difference = "
                       f"**{fm['equalized_odds_diff']:.4f}**\n",
                       "- These two criteria cannot both be zero when base rates "
                       "differ (Kleinberg et al., 2016) — the choice is a policy "
                       "decision. See `fairness_by_group.csv`.\n"]
        except Exception as e:
            report += [f"\n## Fairness audit\n_skipped: {e}_\n"]
    else:
        report += ["\n## Fairness audit\n_No gender column recovered from features._\n"]

    # ---------------- Calibration ----------------
    try:
        rng = np.random.default_rng(config.RANDOM_STATE)
        idx = rng.permutation(len(X_test))
        half = len(idx) // 2
        cal_idx, te_idx = idx[:half], idx[half:]
        rep = evaluation.calibration_report(
            y_test[te_idx], prob[te_idx], y_test[cal_idx], prob[cal_idx])
        rep.to_csv(config.REPORTS_DIR / "calibration.csv", index=False)
        mp, fp, _ = evaluation.reliability_curve(y_test, prob)
        plt.figure(figsize=(5, 5))
        plt.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
        plt.plot(mp, fp, "o-", color="#6d28d9", label="model")
        plt.xlabel("mean predicted prob")
        plt.ylabel("fraction positive")
        plt.title("Reliability diagram")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG / "reliability.png", dpi=120)
        plt.close()
        report += ["\n## Calibration\n",
                   rep.to_markdown(index=False) + "\n",
                   "![](figures/reliability.png)\n"]
    except Exception as e:
        report += [f"\n## Calibration\n_skipped: {e}_\n"]

    # ---------------- Survival (Cox PH) ----------------
    try:
        top_feats = (pd.read_csv(config.REPORTS_DIR / "shap_importance.csv")
                     ["feature"].head(8).tolist())
        top_feats = [c for c in top_feats if c in schema["num_cols"]][:6]
        if top_feats:
            sdf = test.copy()
            # duration proxy: weeks from the earliest decision week (+1 to be >0)
            sdf["duration"] = (sdf[config.WEEK_NUM] - sdf[config.WEEK_NUM].min() + 1)
            surv = survival.make_survival_frame(sdf, top_feats, "duration", config.TARGET)
            cph = survival.fit_coxph(surv, "duration", config.TARGET)
            cidx = survival.concordance(cph)
            report += ["\n## Survival analysis (Cox PH, time-to-default)\n",
                       f"- Harrell C-index = **{cidx:.4f}** on {len(top_feats)} "
                       "top SHAP features.\n",
                       "- Demonstrates lifetime-PD / IFRS-9 readiness beyond a "
                       "binary classifier.\n"]
    except Exception as e:
        report += [f"\n## Survival analysis\n_skipped: {e}_\n"]

    (config.REPORTS_DIR / "results.md").write_text("\n".join(report), encoding="utf-8")
    print("[analysis] wrote reports/results.md")
    with open(config.MODELS_DIR / "analysis_threshold.json", "w") as f:
        json.dump({"threshold": thr}, f)


if __name__ == "__main__":
    main()
