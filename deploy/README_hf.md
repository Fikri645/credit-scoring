---
title: Credit Default Scoring
emoji: 🏦
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
short_description: Credit default scoring with SHAP explanations
---

# Credit Default Scoring — Home Credit 2024

LightGBM credit default model trained on the Home Credit 2024 *Credit Risk
Model Stability* dataset, scored under the Gini-stability metric with a
**cost-optimal decision threshold** and **SHAP explanations**. Pick a held-out
applicant to score and see the probability of default, the approve/decline
decision, and the top SHAP factors driving it.

Full project (training, focal-loss imbalance handling, DiCE counterfactuals,
Fairlearn fairness audit, isotonic calibration, Cox PH survival, FastAPI
service): [github.com/Fikri645/credit-scoring](https://github.com/Fikri645/credit-scoring).

> This Space is the demo surface only. See the GitHub repo for the full
> 2024-2026 SOTA pipeline and the analysis report.
