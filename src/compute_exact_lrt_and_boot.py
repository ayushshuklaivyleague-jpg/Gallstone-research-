"""
src/compute_exact_lrt_and_boot.py

Computes:
1. Likelihood Ratio Test (LRT) between nested logistic models:
   - Deviance = 2 * (LL_full - LL_reduced), degrees of freedom, p-value (Chi-square)
2. Paired Non-Parametric Bootstrap for AUROC difference (1,000 resamples):
   - Mean ΔAUROC, 95% Bootstrap CI, two-sided empirical p-value for AUROC difference
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.evaluation import compute_paired_bootstrap_auroc_test

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
df_nh3 = pd.read_csv(os.path.join(DATA_DIR, "nhanes3_ultrasound.csv"))
sub = df_nh3[df_nh3["target_cholecystectomy_us"] == 0].copy()

features_15 = [
    "Age", "Gender", "Height", "Weight", "BMI",
    "Glucose", "Total Cholesterol", "LDL", "HDL", "Triglyceride",
    "AST", "ALT", "ALP", "Creatinine", "CRP"
]

X_raw = sub[features_15].values
y_raw = sub["target_active_us"].values.astype(int)

idx = np.arange(len(sub))
idx_train, idx_temp, y_train, y_temp = train_test_split(idx, y_raw, test_size=0.30, random_state=42, stratify=y_raw)
idx_val, idx_test, y_val, y_test = train_test_split(idx_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

imputer = SimpleImputer(strategy="median")
X_tr = imputer.fit_transform(X_raw[idx_train])
X_te = imputer.transform(X_raw[idx_test])

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# 1. Demographics Only (Age, Sex, BMI)
f_demo = [features_15.index("Age"), features_15.index("Gender"), features_15.index("BMI")]
m_demo = LogisticRegression(random_state=42, max_iter=1000)
m_demo.fit(X_tr_s[:, f_demo], y_train)
p_demo = m_demo.predict_proba(X_te_s[:, f_demo])[:, 1]
ll_demo = -log_loss(y_test, p_demo, normalize=False)

# 2. Core-6 (Age, Sex, BMI, Glucose, Total Cholesterol, Triglycerides)
f_core = [features_15.index(f) for f in ["Age", "Gender", "BMI", "Glucose", "Total Cholesterol", "Triglyceride"]]
m_core = LogisticRegression(random_state=42, max_iter=1000)
m_core.fit(X_tr_s[:, f_core], y_train)
p_core = m_core.predict_proba(X_te_s[:, f_core])[:, 1]
ll_core = -log_loss(y_test, p_core, normalize=False)

# 3. Full 15-Feature LR
m_full_lr = LogisticRegression(random_state=42, max_iter=1000)
m_full_lr.fit(X_tr_s, y_train)
p_full_lr = m_full_lr.predict_proba(X_te_s)[:, 1]
ll_full_lr = -log_loss(y_test, p_full_lr, normalize=False)

# 4. Full XGBoost
m_xgb = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss")
m_xgb.fit(X_tr_s, y_train)
p_xgb = m_xgb.predict_proba(X_te_s)[:, 1]
ll_xgb = -log_loss(y_test, p_xgb, normalize=False)

# Classical nested LRT: fit both models by maximum likelihood on the same
# development sample (train + validation). The held-out test set remains
# untouched and is used only for out-of-sample AUROC/bootstrap comparisons.
idx_dev = np.concatenate([idx_train, idx_val])
y_dev = y_raw[idx_dev]

dev_imputer = SimpleImputer(strategy="median")
X_dev_imp = dev_imputer.fit_transform(X_raw[idx_dev])
dev_scaler = StandardScaler()
X_dev_s = dev_scaler.fit_transform(X_dev_imp)

def fit_unpenalized_lr(X, y):
    return LogisticRegression(C=1e9, solver="lbfgs", max_iter=5000)

def fitted_loglik(model, X, y):
    p = model.predict_proba(X)[:, 1]
    return -log_loss(y, p, normalize=False, labels=[0, 1])

# Re-fit nested logistic models as true maximum-likelihood models.
lrt_demo = fit_unpenalized_lr(X_dev_s[:, f_demo], y_dev)
lrt_core = fit_unpenalized_lr(X_dev_s[:, f_core], y_dev)
lrt_full = fit_unpenalized_lr(X_dev_s, y_dev)

lrt_demo.fit(X_dev_s[:, f_demo], y_dev)
lrt_core.fit(X_dev_s[:, f_core], y_dev)
lrt_full.fit(X_dev_s, y_dev)

ll_demo_dev = fitted_loglik(lrt_demo, X_dev_s[:, f_demo], y_dev)
ll_core_dev = fitted_loglik(lrt_core, X_dev_s[:, f_core], y_dev)
ll_full_dev = fitted_loglik(lrt_full, X_dev_s, y_dev)

dev_core_demo = 2 * (ll_core_dev - ll_demo_dev)
p_lrt_core_demo = stats.chi2.sf(dev_core_demo, df=3)

dev_full_core = 2 * (ll_full_dev - ll_core_dev)
p_lrt_full_core = stats.chi2.sf(dev_full_core, df=9)

print("=== LIKELIHOOD RATIO TESTS (DEVELOPMENT SAMPLE) ===")
print(f"Development sample: N={len(y_dev)} (train + validation); held-out test N={len(y_test)}")
print(f"Core-6 vs Demo (df=3):     Delta LL = +{ll_core_dev - ll_demo_dev:.2f}, Chi2 Deviance = {dev_core_demo:.2f}, p(LRT) = {p_lrt_core_demo:.4e}")
print(f"Full LR vs Core-6 (df=9):  Delta LL = +{ll_full_dev - ll_core_dev:.2f}, Chi2 Deviance = {dev_full_core:.2f}, p(LRT) = {p_lrt_full_core:.4e}")

# Paired Bootstrap
boot_core_demo = compute_paired_bootstrap_auroc_test(y_test, p_core, p_demo, n_bootstraps=1000, seed=42)
boot_full_core = compute_paired_bootstrap_auroc_test(y_test, p_full_lr, p_core, n_bootstraps=1000, seed=42)
boot_xgb_full = compute_paired_bootstrap_auroc_test(y_test, p_xgb, p_full_lr, n_bootstraps=1000, seed=42)

print("\n=== PAIRED BOOTSTRAP FOR AUROC ===")
print(f"Core-6 vs Demo:    Delta AUROC = +{boot_core_demo['delta_mean']:.4f} [{boot_core_demo['delta_ci_95'][0]:.4f} to {boot_core_demo['delta_ci_95'][1]:.4f}], p(AUROC) = {boot_core_demo['p_value']:.4f}")
print(f"Full LR vs Core-6: Delta AUROC = +{boot_full_core['delta_mean']:.4f} [{boot_full_core['delta_ci_95'][0]:.4f} to {boot_full_core['delta_ci_95'][1]:.4f}], p(AUROC) = {boot_full_core['p_value']:.4f}")
print(f"Full XGB vs LR:    Delta AUROC = +{boot_xgb_full['delta_mean']:.4f} [{boot_xgb_full['delta_ci_95'][0]:.4f} to {boot_xgb_full['delta_ci_95'][1]:.4f}], p(AUROC) = {boot_xgb_full['p_value']:.4f}")

