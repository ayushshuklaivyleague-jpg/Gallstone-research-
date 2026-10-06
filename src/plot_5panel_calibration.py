"""
src/plot_5panel_calibration.py

Generates Figure 5: Five-Panel Comprehensive Calibration Curves:
  Panel A: NHANES III In-Domain
  Panel B: UCI Hospital In-Domain
  Panel C: Cross-Domain: UCI -> NHANES III (Zero-shot outward transfer)
  Panel D: Cross-Domain: NHANES III -> UCI (Zero-shot inward transfer)
  Panel E: Joint UCI + NHANES 2017–2020 Model (20 harmonized features)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.harmonized_dataset import load_harmonized_data

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

df_nh3 = pd.read_csv(os.path.join(DATA_DIR, "nhanes3_ultrasound.csv"))
df_uci = pd.read_csv(os.path.join(DATA_DIR, "gallstone_.csv"))
df_nh = pd.read_csv(os.path.join(DATA_DIR, "nhanes_gallstone.csv"))

features_15 = [
    "Age", "Gender", "Height", "Weight", "BMI",
    "Glucose", "Total Cholesterol", "LDL", "HDL", "Triglyceride",
    "AST", "ALT", "ALP", "Creatinine", "CRP"
]

uci_15 = [
    "Age", "Gender", "Height", "Weight", "Body Mass Index (BMI)",
    "Glucose", "Total Cholesterol (TC)", "Low Density Lipoprotein (LDL)",
    "High Density Lipoprotein (HDL)", "Triglyceride",
    "Aspartat Aminotransferaz (AST)", "Alanin Aminotransferaz (ALT)",
    "Alkaline Phosphatase (ALP)", "Creatinine", "C-Reactive Protein (CRP)"
]

# NHANES III data
sub_nh3 = df_nh3[df_nh3["target_cholecystectomy_us"] == 0].copy()
X_nh3 = sub_nh3[features_15].values
y_nh3 = sub_nh3["target_active_us"].values.astype(int)

# UCI data
X_uci = df_uci[uci_15].values
y_uci = df_uci["Gallstone Status"].values.astype(int)

# Splits
idx_n = np.arange(len(sub_nh3))
tr_n, te_n, y_tr_n, y_te_n = train_test_split(idx_n, y_nh3, test_size=0.15, random_state=42, stratify=y_nh3)

imp_n = SimpleImputer(strategy="median")
X_tr_n_imp = imp_n.fit_transform(X_nh3[tr_n])
X_te_n_imp = imp_n.transform(X_nh3[te_n])

scl_n = StandardScaler()
X_tr_n_s = scl_n.fit_transform(X_tr_n_imp)
X_te_n_s = scl_n.transform(X_te_n_imp)

# Fit NH3 model (Random Forest / Logistic / XGBoost)
clf_nh3 = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
clf_nh3.fit(X_tr_n_s, y_tr_n)
prob_nh3_in = clf_nh3.predict_proba(X_te_n_s)[:, 1]

# Fit UCI model
idx_u = np.arange(len(df_uci))
tr_u, te_u, y_tr_u, y_te_u = train_test_split(idx_u, y_uci, test_size=0.15, random_state=42, stratify=y_uci)

imp_u = SimpleImputer(strategy="median")
X_tr_u_imp = imp_u.fit_transform(X_uci[tr_u])
X_te_u_imp = imp_u.transform(X_uci[te_u])

scl_u = StandardScaler()
X_tr_u_s = scl_u.fit_transform(X_tr_u_imp)
X_te_u_s = scl_u.transform(X_te_u_imp)

clf_uci = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
clf_uci.fit(X_tr_u_s, y_tr_u)
prob_uci_in = clf_uci.predict_proba(X_te_u_s)[:, 1]

# Transfer Exp 4A: UCI model -> NH3 test
X_nh3_under_uci = scl_u.transform(imp_u.transform(X_nh3[te_n]))
prob_uci_to_nh3 = clf_uci.predict_proba(X_nh3_under_uci)[:, 1]

# Transfer Exp 4B: NH3 model -> UCI test
X_uci_under_nh3 = scl_n.transform(imp_n.transform(X_uci[te_u]))
prob_nh3_to_uci = clf_nh3.predict_proba(X_uci_under_nh3)[:, 1]

# Joint Model (Experiment 5 / joint 20-feature UCI + NHANES cohort)
# Use the same pooled cohort and 70/15/15 split defined by the harmonized
# benchmark loader (N=9,529 total: UCI + NHANES 2017–2020).
joint_bundle = load_harmonized_data(mode="joint")
X_joint_s = joint_bundle["raw_arrays"]["X_train"]
y_joint = joint_bundle["raw_arrays"]["y_train"]
X_te_joint_s = joint_bundle["raw_arrays"]["X_test"]
y_te_joint = joint_bundle["raw_arrays"]["y_test"]

clf_joint = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
clf_joint.fit(X_joint_s, y_joint)
prob_joint = clf_joint.predict_proba(X_te_joint_s)[:, 1]

# Plot 5 Panels
fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), dpi=300)

configs = [
    ("A. NHANES III In-Domain", y_te_n, prob_nh3_in, "#1f77b4"),
    ("B. UCI Clinic In-Domain", y_te_u, prob_uci_in, "#2ca02c"),
    ("C. Transfer: UCI → NHANES III", y_te_n, prob_uci_to_nh3, "#d62728"),
    ("D. Transfer: NHANES III → UCI", y_te_u, prob_nh3_to_uci, "#9467bd"),
    ("E. Joint UCI + NHANES Model (20 features)", y_te_joint, prob_joint, "#ff7f0e")
]

for ax, (title, y_true, y_p, col) in zip(axes, configs):
    # compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_p, n_bins=8, strategy="quantile")
    ax.plot(prob_pred, prob_true, marker="o", lw=2, color=col, label="Model")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect Calibration")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax.set_ylabel("Observed Fraction Positive", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "calibration_curves_5panel.png"))
plt.close()
print("Saved plots/calibration_curves_5panel.png successfully!")
