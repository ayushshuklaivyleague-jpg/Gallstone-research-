"""
configs/default.py — Project configurations, random seeds, and feature sets.
"""

from pathlib import Path

# Base Paths (Relative to repository root)
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
PLOTS_DIR = ROOT_DIR / "plots"

# Reproducibility
RANDOM_SEED = 42

# Train / Validation / Test Partitioning Ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Bootstrap Evaluation Parameters
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_ALPHA = 0.05  # 95% Confidence Intervals

# Primary 15 Harmonized Non-Imaging Features (Ultrasound Ground-Truth Benchmark)
HARMONIZED_15_FEATURES = [
    "Age",
    "Gender",
    "Height",
    "Weight",
    "BMI",
    "Glucose",
    "Total Cholesterol",
    "LDL",
    "HDL",
    "Triglyceride",
    "AST",
    "ALT",
    "ALP",
    "Creatinine",
    "CRP",
]

# Parsimonious Baseline Subsets
CORE_6_FEATURES = [
    "Age",
    "Gender",
    "BMI",
    "Glucose",
    "Total Cholesterol",
    "Triglyceride",
]

DEMO_3_FEATURES = [
    "Age",
    "Gender",
    "BMI",
]

# GallstoneNet Model Hyperparameters
GALLSTONENET_CONFIG = {
    "hidden_dims": [128, 64, 32],
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 120,
    "patience": 15,
    "batch_size": 32,
}

# XGBoost Hyperparameters
XGBOOST_CONFIG = {
    "n_estimators": 150,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
}

# LightGBM Hyperparameters
LIGHTGBM_CONFIG = {
    "n_estimators": 150,
    "max_depth": 5,
    "learning_rate": 0.05,
    "random_state": 42,
    "verbose": -1,
}

# Random Forest Hyperparameters
RANDOM_FOREST_CONFIG = {
    "n_estimators": 1000,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}
