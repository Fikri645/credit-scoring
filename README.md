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

# 5. Full analysis: SHAP, DiCE, fairness, calibration, survival → reports/
python -m src.run_analysis

# 6a. Serve the REST API
uvicorn api.main:app --reload          #  POST /score

# 6b. …or the interactive Gradio demo
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

Generated into [`reports/results.md`](reports/results.md) by `src/run_analysis.py`
(numbers depend on `SAMPLE_FRAC` and are reproduced on each run). The report
includes the test Gini-stability and its components, the cost-optimal threshold
saving vs 0.5, SHAP global importance, a calibration table (raw vs Platt vs
isotonic ECE/Brier), the fairness gaps, and the survival C-index.

---

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
