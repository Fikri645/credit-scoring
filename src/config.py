"""Central configuration for the credit-scoring pipeline.

All paths, the competition slug, modelling constants, and the protected
attributes used by the fairness audit live here so notebooks and ``src``
modules share a single source of truth.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # unzipped Kaggle parquet/csv files
PROCESSED_DIR = DATA_DIR / "processed"  # feature-engineered tables
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ARTIFACTS_DIR = ROOT / "artifacts"

for _d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Kaggle competition
# --------------------------------------------------------------------------- #
COMPETITION = "home-credit-credit-risk-model-stability"

# Parquet base tables live under parquet_files/{train,test}/
TRAIN_DIR = RAW_DIR / "parquet_files" / "train"
TEST_DIR = RAW_DIR / "parquet_files" / "test"

# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
TARGET = "target"
CASE_ID = "case_id"
DATE_DECISION = "date_decision"
WEEK_NUM = "WEEK_NUM"          # time bucket used by the Gini-stability metric

RANDOM_STATE = 42
N_FOLDS = 5

# Down-sampling for a laptop-friendly run. The full base table is ~1.5M rows;
# set SAMPLE_FRAC = 1.0 for the full competition run.
SAMPLE_FRAC = 0.30

# --------------------------------------------------------------------------- #
# Fairness — protected attributes available in the Home Credit data
# --------------------------------------------------------------------------- #
# These come from the static person table after feature engineering.
# Gender / age are the standard protected attributes for credit fairness.
PROTECTED_GENDER = "sex_decoded"      # M / F, decoded from sex_738L
AGE_FEATURE = "age_years"            # derived from birth_259D

# --------------------------------------------------------------------------- #
# Business cost model (used for threshold optimisation)
# --------------------------------------------------------------------------- #
# A missed default (false negative) loses the exposure; a wrongly rejected
# good applicant (false positive) loses the expected margin on the loan.
COST_FALSE_NEGATIVE = 1.0     # relative loss per missed default (exposure)
COST_FALSE_POSITIVE = 0.20    # relative loss per rejected good customer (margin)
