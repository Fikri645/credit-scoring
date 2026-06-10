# credit-scoring

[![CI](https://github.com/Fikri645/credit-scoring/actions/workflows/ci.yml/badge.svg)](https://github.com/Fikri645/credit-scoring/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)](https://lightgbm.readthedocs.io/)

**Production-shaped credit default scoring on the Home Credit 2024 _Credit Risk
Model Stability_ dataset** — built around the 2024–2026 state of the art rather
than the textbook recipe. It optimises the competition's **Gini-stability
metric** (not just AUC), handles class imbalance with a **PyTorch-autograd focal
loss** instead of SMOTE, and ships the regulatory layer modern lending teams
actually need: **SHAP + DiCE counterfactuals** (EU AI Act / GDPR Art. 22),
a **Fairlearn** fairness audit, **probability calibration** (Platt vs isotonic),
and a **Cox PH survival** model for lifetime-PD (IFRS 9 / Basel IRB).

> Why these choices? See [`docs/METHOD_NOTES.md`](docs/METHOD_NOTES.md) for the
> 2024–2026 literature behind every design decision (focal loss over SMOTE,
> stability metric, counterfactual recourse, calibration, survival analysis).

---

## What makes this *current* (2026), not a 2019 credit-scoring tutorial

| Concern | Common tutorial | This project (2024–2026 SOTA) |
|:--|:--|:--|
| Target metric | ROC-AUC only | **Gini-stability** — penalises models that decay over time |
| Class imbalance | SMOTE oversampling | **Focal loss** (autograd) + cost-sensitive weights; *no SMOTE* |
| Model | one XGBoost | **LightGBM (weighted) vs LightGBM (focal) vs CatBoost vs FT-Transformer** |
| Explainability | feature_importances_ | **SHAP** (global + local + interactions) + **DiCE counterfactual recourse** |
| Fairness | — | **Fairlearn** demographic-parity vs equalized-odds + ThresholdOptimizer |
| Probabilities | raw model output | **Platt vs isotonic calibration** + ECE / reliability diagram |
| Decision rule | threshold 0.5 | **business-cost optimal threshold** (expected-loss minimiser) |
| Time-to-event | — | **Cox PH survival** model (lifetime PD, IFRS 9 / Basel IRB) |

---

## Architecture

```
Home Credit 2024 (depth-based relational parquet: base + depth-0/1/2 tables)
        │   src/features.py  (Polars: suffix-driven aggregation → 1 row / case_id,
        │                     date→relative-days leakage guard, column pruning)
        ▼
  data/processed/train_features.parquet
        │   src/train.py  (forward-in-time WEEK_NUM split)
        ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Model bake-off  →  Gini-stability metric  →  MLflow         │
  │   • LightGBM (is_unbalance)                                  │
  │   • LightGBM (focal loss, autograd grad/Hessian)            │
  │   • CatBoost (auto_class_weights)                           │
  │   • FT-Transformer (optional DL comparison)                 │
  └─────────────────────────────────────────────────────────────┘
        │   src/run_analysis.py
        ▼
  SHAP · DiCE counterfactuals · Fairlearn audit · calibration ·
  business-cost threshold · Cox PH survival   →  reports/results.md + figures/
        │
        ▼
  Serving:  FastAPI  /score   (api/main.py)      Gradio demo  (app.py)
```

---

## Dataset

[**Home Credit — Credit Risk Model Stability** (Kaggle, 2024)](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability).
~1.5M loan applications with a **depth-based relational schema** (a static
`base` table plus depth-0/1/2 one-to-many tables for bureau records, previous
applications, person data, tax registry, deposits…). Column names encode their
transform via a trailing letter (`P` DPD, `A` amount, `D` date, `M` masked
category). The competition's headline metric rewards **stability over time**,
which is what makes it a more honest benchmark than the 2018 Home Credit set.

> Requires accepting the competition rules, then `python -m src.download_data`.

---

## Quick start

```bash
# 1. Environment (Python 3.11; RTX-class GPU optional, used by FT-Transformer)
py -3.11 -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on Linux
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Data  (needs ~/.kaggle/kaggle.json + accepted competition rules)
python -m src.download_data

# 3. Feature engineering  →  data/processed/train_features.parquet
python -m src.features

# 4. Train + compare models (logs to MLflow, saves winner to models/)
python -m src.train

# 5. EDA report (imbalance, missingness, temporal drift) → reports/eda.md
python -m src.eda

# 6. Full analysis: SHAP + interactions, DiCE, fairness + mitigation,
#    calibration, survival → reports/results.md
python -m src.run_analysis

# 7a. Serve the REST API
uvicorn api.main:app --reload          #  POST /score

# 7b. …or the interactive Gradio demo
python app.py                          #  http://localhost:7860
```

For a fast laptop run, `SAMPLE_FRAC` in `src/config.py` subsamples the base
table (default `0.30`); set it to `1.0` for the full competition run.

---

## Modules

| File | Responsibility |
|:--|:--|
| `src/config.py` | Paths, competition slug, modelling + cost constants |
| `src/download_data.py` | Kaggle API download + unzip |
| `src/eda.py` | Reproducible EDA → `reports/eda.md` (imbalance, missingness, drift) |
| `src/features.py` | Polars feature engineering (aggregation, date transform, pruning) |
| `src/metrics.py` | Gini-stability metric, Expected Calibration Error |
| `src/modeling.py` | Time split, categorical prep, **focal-loss autograd objective** |
| `src/train.py` | Model bake-off, MLflow logging, artifact persistence |
| `src/ft_transformer.py` | FT-Transformer tabular DL comparison |
| `src/evaluation.py` | Business-cost threshold, Platt/isotonic calibration |
| `src/explain.py` | SHAP (global/local/interaction) + DiCE counterfactuals |
| `src/fairness.py` | Fairlearn metrics + ThresholdOptimizer mitigation |
| `src/survival.py` | Cox PH time-to-default |
| `src/run_analysis.py` | Orchestrates the analysis layer → `reports/results.md` |
| `api/main.py` | FastAPI `/score` service |
| `app.py` | Gradio demo (score held-out applicants + SHAP factors) |

---

## Results

Full report regenerated on every run into [`reports/results.md`](reports/results.md).
Representative numbers from a `SAMPLE_FRAC = 0.30` run (≈458k applications,
forward-in-time test split):

### Model bake-off — Gini stability metric

| Model | Gini-stability | Mean Gini |
|:--|:--:|:--:|
| **CatBoost** 🏆 | **0.694** | 0.716 |
| **LightGBM (focal loss, autograd)** | **0.676** | 0.696 |
| LightGBM (`is_unbalance`) | 0.642 | 0.664 |
| FT-Transformer (tabular DL) | 0.131 | 0.170 |

Two findings, both matching the 2024-2026 literature:
- The custom **focal-loss objective beats naive class weighting by +0.034
  stability** — better imbalance handling than SMOTE/`is_unbalance`.
- **Gradient boosting decisively beats the tabular transformer** (0.69 vs
  0.13): FT-Transformer underperforms on this wide, sparse, heavily-categorical
  credit data — the honest, expected outcome (cf. Booking.com, arXiv 2405.13692).

### Business-cost decision threshold

Cost-optimal threshold **0.813** vs the naive 0.5 → **53.9 % lower expected
cost** on the test set.

### Calibration — isotonic wins (as the 2026 literature predicts)

| Method | Brier | ECE |
|:--|:--:|:--:|
| raw model | 0.126 | 0.254 |
| Platt scaling | 0.020 | 0.0016 |
| **Isotonic regression** | **0.019** | **0.0009** |

### Fairness (gender)

Demographic-parity gap **0.009**, equalized-odds gap **0.061**. Fairlearn's
**`ThresholdOptimizer`** (per-group thresholds, no retraining) closes the
equalized-odds gap to **≈0.000** in-sample — a demonstration of the
post-processing remedy, not a held-out generalisation claim.

> [!warning] Honest finding
> The single largest SHAP feature is **`sex_738L` (gender)** — the model leans
> on a *protected attribute*. In production this would be illegal in most
> jurisdictions; the fairness audit exists precisely to surface this, and the
> `ThresholdOptimizer` / drop-protected-attribute paths are the remedies.

### Survival (Cox PH, time-to-default)

Harrell **C-index 0.639** — lifetime-PD signal for IFRS 9 / Basel IRB beyond
the binary classifier.

### Recourse (DiCE)

Generates the minimal feature changes that flip a rejected applicant to
approved (`reports/counterfactual_example.csv`) — GDPR Art. 22 in practice.

### Feature interactions (SHAP interaction values)

Exact pairwise SHAP interaction values over the dominant numeric drivers
(`reports/shap_interactions.csv`). The strongest interactions are all between
delinquency signals — days-past-due metrics × overdue-installment counts —
which is the credit-intuitive result. (Computed on a compact numeric surrogate,
since the full booster requires all 730 features at once.)

> EDA report — target imbalance, missingness, and the **weekly default-rate
> drift** that motivates the stability metric — is regenerated into
> [`reports/eda.md`](reports/eda.md) via `python -m src.eda`.

---

## Deploy the demo to HuggingFace Spaces

A lean, deploy-ready setup lives in `deploy/`:

```bash
pip install huggingface_hub
set HF_TOKEN=hf_xxx                 # a write token from huggingface.co/settings/tokens
python deploy/deploy_hf_space.py fikri0o0/credit-scoring
```

This creates a Gradio Space (free CPU) and uploads only what the demo needs
(LightGBM model, feature schema, demo applicants, `app.py`, minimal `src/`, a
lean `requirements.txt`, and the Space `README.md`). The full requirements set
is not needed at serving time — the demo only loads LightGBM + SHAP.

## Testing

```bash
pytest tests/ -v --cov=src        # pure-function tests, no dataset needed
flake8 src api tests --max-line-length=100
```

CI (GitHub Actions) installs CPU PyTorch, lints, and runs the test suite on
every push — the tests cover the stability metric, the focal-loss gradient,
the time-aware split, calibration, and the Polars feature primitives.

---

## License

MIT
