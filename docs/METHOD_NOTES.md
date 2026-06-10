# Method Notes — why these 2024–2026 choices

This project deliberately avoids the 2019-era credit-scoring tutorial recipe.
Each decision below is tied to the current (2024–2026) literature.

## 1. Target metric — Gini *stability*, not plain AUC
The Home Credit 2024 *Credit Risk Model Stability* competition scores
`mean(weekly_gini) + 88·min(0, slope) − 0.5·std(residuals)`. It rewards a model
whose discriminative power is **stable across time**, encoding the regulator
concern that a model must not silently decay post-deployment. We compute it in
`src/metrics.py` and select models on it.

## 2. Model family — gradient boosting still wins, but we prove it
A 2025 multi-benchmark study confirms GBTs match/beat deep nets on tabular
credit data. We still train an **FT-Transformer** (Gorishniy et al., 2021) as an
honest comparison — and expect LightGBM/CatBoost to win unless data volume is
very large (Booking.com, arXiv 2405.13692).

## 3. Imbalance — focal loss, not SMOTE
SMOTE is now treated as a weak baseline: it synthesises minority points without
respecting the data manifold. Current practice is cost-sensitive learning and
**focal loss** (Lin et al., 2017; focal-aware cost-sensitive boosting for
credit, 2022/2024). We implement focal loss as a **custom LightGBM objective
whose gradient and diagonal Hessian are computed by `torch.autograd`** — exact
by construction (`src/modeling.py`).

## 4. Explainability — SHAP + counterfactual recourse
EU AI Act (Reg. 2024/1689) classifies credit scoring as **high-risk**; GDPR
Art. 22 grants a "right to explanation". We provide exact **SHAP TreeExplainer**
attributions (global beeswarm, local waterfall, interaction values) *and*
**DiCE** counterfactuals — the actionable "what would flip this rejection"
recourse a declined applicant is entitled to (`src/explain.py`).

## 5. Fairness — and its impossibility
We audit demographic parity vs equalized odds with **Fairlearn**, and apply
`ThresholdOptimizer` for post-hoc mitigation. We surface the honest result that
these criteria are mathematically incompatible under unequal base rates
(Kleinberg et al., 2016) — fairness choice is policy, not a tuning knob.

## 6. Calibration — pricing needs probabilities
Expected Loss = PD·LGD·EAD requires calibrated probabilities, not just a
ranking. We compare **Platt scaling** vs **isotonic regression** and report
**ECE** + reliability diagrams; isotonic tends to win on large calibration sets
(arXiv 2601.19944, 2026) (`src/evaluation.py`).

## 7. Decision rule — business-cost threshold
A credit model is a cost minimiser. We grid-search the threshold that minimises
expected cost (missed default vs wrongly rejected good customer) instead of
using 0.5.

## 8. Survival analysis — lifetime PD
Binary classifiers cannot answer *when* a default happens, which IFRS 9 ECL and
Basel IRB lifetime-PD curves require (MDPI Risks, 2025). We fit a penalised
**Cox PH** model (`src/survival.py`) and report Harrell's C-index.

## 9. Considered but out of scope
Methods that were on the 2024–2026 research short-list and deliberately *not*
shipped, with the reason — included to show the design space was surveyed, not
that these were missed:

| Method | What it adds | Why out of scope here |
|:--|:--|:--|
| **TabPFN-2** | Zero-shot prior-fitted transformer; strong on *small* tabular data | Capped at ~10k rows / ~100 features — this dataset is 1.5M × 730, its worst case |
| **LDAM loss** | Label-distribution-aware margins for imbalance | Focal loss already covers the imbalance axis; LDAM's margin schedule adds tuning surface for marginal gain |
| **NICE counterfactuals** | Faster nearest-instance recourse | DiCE already delivers diverse, actionable recourse; NICE would be a speed swap, not a capability gain |
| **Venn-Abers predictors** | Calibration with validity guarantees | Isotonic + ECE/reliability is sufficient for the pricing story; Venn-Abers is a heavier dependency for a portfolio demo |
| **DeepHit** | Discrete-time neural survival, competing risks | Cox PH is the interpretable, regulator-legible baseline; DeepHit needs far more data/tuning to beat it here |
| **AutoGluon ensemble** | Stacked multi-layer AutoML | Obscures the explicit model-vs-model comparison this project is built to *show* |

## Key references
- Home Credit — Credit Risk Model Stability (Kaggle, 2024)
- Lin et al., *Focal Loss for Dense Object Detection* (2017)
- Gorishniy et al., *Revisiting Deep Learning Models for Tabular Data* (2021)
- Kleinberg, Mullainathan, Raghavan, *Inherent Trade-Offs in Fair Determination* (2016)
- EU AI Act, Regulation (EU) 2024/1689
- Survival Analysis for Credit Risk: Basel IRB Compliance (MDPI Risks, 2025)
- Classifier Calibration at Scale (arXiv 2601.19944, 2026)
