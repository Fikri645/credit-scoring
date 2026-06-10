"""Exploratory data analysis -> ``reports/eda.md`` + figures.

This runs against the persisted, feature-engineered artifacts
(``artifacts/train_sample.parquet`` + ``artifacts/test_split.parquet``) rather
than the 27 GB raw Kaggle download, so it is fully reproducible on any machine
that has cloned the repo and trained once. It documents the four things a
credit-risk reviewer looks for first:

* **target imbalance** — how rare default is (drives the focal-loss choice);
* **feature inventory** — numeric vs categorical width;
* **missingness** — the depth-based relational schema leaves many sparse
  columns, which motivates the null-pruning in ``features.py``;
* **temporal drift** — the weekly default rate across ``WEEK_NUM``, which is
  exactly what the competition's Gini-*stability* metric penalises.

Run:  ``python -m src.eda``
"""
from __future__ import annotations

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config  # noqa: E402

FIG = config.FIGURES_DIR


def _load():
    schema = joblib.load(config.MODELS_DIR / "feature_schema.joblib")
    train = pd.read_parquet(config.ARTIFACTS_DIR / "train_sample.parquet")
    test = pd.read_parquet(config.ARTIFACTS_DIR / "test_split.parquet")
    return schema, train, test


def main():
    schema, train, test = _load()
    full = pd.concat([train, test], ignore_index=True)
    y = full[config.TARGET].to_numpy()
    out: list[str] = ["# Credit Scoring — Exploratory Data Analysis\n"]

    # ---------------- Dataset shape + target imbalance ----------------
    rate = float(y.mean())
    out += [
        "## Dataset\n",
        f"- Engineered analysis frame: **{len(full):,} applications** "
        f"× **{full.shape[1]:,} columns** "
        f"(train sample {len(train):,} + held-out test {len(test):,}).\n",
        f"- Numeric features: **{len(schema['num_cols'])}**, "
        f"categorical: **{len(schema['cat_cols'])}** "
        f"(total modelled features {len(schema['feat_cols'])}).\n",
        "\n## Target imbalance\n",
        f"- Default rate = **{rate:.2%}** "
        f"({int(y.sum()):,} defaults / {len(y):,}) — a "
        f"**1 : {(1 - rate) / max(rate, 1e-9):.0f}** positive-to-negative ratio.\n",
        "- This severe imbalance is why the project uses a **focal-loss "
        "objective** + cost-sensitive weighting rather than SMOTE, and scores "
        "on **Gini-stability** rather than raw accuracy.\n",
    ]
    plt.figure(figsize=(4, 4))
    counts = pd.Series(y).value_counts().sort_index()
    plt.bar(["repaid (0)", "default (1)"], counts.values, color=["#6d28d9", "#dc2626"])
    for i, v in enumerate(counts.values):
        plt.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("applications")
    plt.title(f"Target distribution (default = {rate:.2%})")
    plt.tight_layout()
    plt.savefig(FIG / "eda_target.png", dpi=120)
    plt.close()
    out += ["\n![](figures/eda_target.png)\n"]

    # ---------------- Missingness ----------------
    miss = (full[schema["feat_cols"]].isna().mean()
            .sort_values(ascending=False))
    n_high = int((miss > 0.5).sum())
    top_miss = miss.head(15)
    out += [
        "\n## Missingness\n",
        f"- **{n_high}** of {len(schema['feat_cols'])} features are >50% null — "
        "an artefact of the depth-0/1/2 relational tables, where most "
        "applicants have no bureau / previous-application history.\n",
        f"- Mean missingness across all features = **{miss.mean():.1%}**.\n",
        "- `features.py` drops all-null columns per source table before the "
        "join; LightGBM/CatBoost then consume the remaining NaNs natively.\n",
    ]
    plt.figure(figsize=(7, 5))
    plt.barh(top_miss.index[::-1], top_miss.values[::-1], color="#9333ea")
    plt.xlabel("fraction missing")
    plt.title("Top-15 most-missing features")
    plt.tight_layout()
    plt.savefig(FIG / "eda_missingness.png", dpi=120)
    plt.close()
    out += ["\n![](figures/eda_missingness.png)\n"]

    # ---------------- Temporal drift (the stability story) ----------------
    wk = (pd.DataFrame({"w": full[config.WEEK_NUM].to_numpy(), "y": y})
          .groupby("w")["y"].agg(["mean", "count"]).reset_index())
    out += [
        "\n## Temporal drift (why stability matters)\n",
        f"- Decisions span **WEEK_NUM {int(wk['w'].min())}–{int(wk['w'].max())}** "
        f"({len(wk)} weekly buckets).\n",
        f"- Weekly default rate ranges **{wk['mean'].min():.2%}–"
        f"{wk['mean'].max():.2%}** (std {wk['mean'].std():.3%}). A model whose "
        "discrimination decays as this base rate moves is penalised by the "
        "Gini-stability metric — the central modelling concern of this dataset.\n",
        "- The train/test split is **forward-in-time** on WEEK_NUM (no future "
        "leakage), mirroring how a deployed scorecard actually ages.\n",
    ]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(wk["w"], wk["mean"], "o-", color="#dc2626", label="default rate")
    ax1.set_xlabel("WEEK_NUM")
    ax1.set_ylabel("weekly default rate", color="#dc2626")
    ax1.tick_params(axis="y", labelcolor="#dc2626")
    ax2 = ax1.twinx()
    ax2.bar(wk["w"], wk["count"], alpha=0.15, color="#6d28d9")
    ax2.set_ylabel("applications / week", color="#6d28d9")
    plt.title("Default rate drift over time")
    fig.tight_layout()
    plt.savefig(FIG / "eda_drift.png", dpi=120)
    plt.close()
    out += ["\n![](figures/eda_drift.png)\n"]

    (config.REPORTS_DIR / "eda.md").write_text("\n".join(out), encoding="utf-8")
    print("[eda] wrote reports/eda.md (+ 3 figures)")


if __name__ == "__main__":
    main()
