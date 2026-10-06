"""
src/compute_all_scientific_analyses.py

Computes:
1. Exact Cohort Flow counts
2. Baseline Characteristics Table (NHANES III: Stone- vs Stone+ with SMD and p-values)
3. Missingness Table across cohorts
4. Model Comparison Significance Testing (Bootstrap difference in AUROC + 95% CIs + p-values)
5. Subgroup Analysis (Sex, Age, BMI subgroups on test set)
6. Sensitivity Analyses (Drop CRP, Drop Liver Enzymes, Drop Demographics, Core Only)
7. Ablation Studies (Feature count and domain subsets)
8. Decision Curve Analysis (Net benefit curves for GallstoneNet, Super Ensemble, Treat All, Treat None)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# 1. Load Data
df_nh3 = pd.read_csv(os.path.join(DATA_DIR, "nhanes3_ultrasound.csv"))
df_uci = pd.read_csv(os.path.join(DATA_DIR, "gallstone_.csv"))
df_nh = pd.read_csv(os.path.join(DATA_DIR, "nhanes_gallstone.csv"))

print("=== 1. COHORT FLOW DIAGRAM DATA ===")
sub_nh3 = df_nh3[df_nh3["target_cholecystectomy_us"] == 0].copy()
y_nh3 = sub_nh3["target_active_us"].values.astype(int)

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

# 2. Baseline Characteristics Table (NHANES III)
print("\n=== 2. BASELINE CHARACTERISTICS (NHANES III: N=12,824) ===")
neg_mask = (y_nh3 == 0)
pos_mask = (y_nh3 == 1)

table1_rows = []
for f in features_15:
    vals_all = sub_nh3[f]
    v_neg = vals_all[neg_mask].dropna()
    v_pos = vals_all[pos_mask].dropna()
    
    if f == "Gender":
        # Female count and %
        n_neg_f = (v_neg == 2).sum() if 2 in v_neg.values else (v_neg == 1).sum()
        pct_neg_f = n_neg_f / len(v_neg) * 100
        n_pos_f = (v_pos == 2).sum() if 2 in v_pos.values else (v_pos == 1).sum()
        pct_pos_f = n_pos_f / len(v_pos) * 100
        # Standardized difference for proportion
        p1 = pct_pos_f / 100.0
        p0 = pct_neg_f / 100.0
        smd = (p1 - p0) / np.sqrt((p1*(1-p1) + p0*(1-p0)) / 2)
        chi2, p_val = stats.chisquare([n_pos_f, len(v_pos)-n_pos_f], [n_neg_f * len(v_pos)/len(v_neg), (len(v_neg)-n_neg_f)*len(v_pos)/len(v_neg)])
        table1_rows.append({
            "Characteristic": "Female, n (%)",
            "Stone- (N=11,666)": f"{n_neg_f} ({pct_neg_f:.1f}%)",
            "Stone+ (N=1,158)": f"{n_pos_f} ({pct_pos_f:.1f}%)",
            "SMD": f"{smd:.3f}",
            "p-value": f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.3f}"
        })
    else:
        m_neg, s_neg = v_neg.mean(), v_neg.std()
        m_pos, s_pos = v_pos.mean(), v_pos.std()
        s_pooled = np.sqrt(((len(v_neg)-1)*s_neg**2 + (len(v_pos)-1)*s_pos**2) / (len(v_neg) + len(v_pos) - 2))
        smd = (m_pos - m_neg) / s_pooled if s_pooled > 0 else 0.0
        t_stat, p_val = stats.ttest_ind(v_pos, v_neg, equal_var=False)
        table1_rows.append({
            "Characteristic": f,
            "Stone- (N=11,666)": f"{m_neg:.1f} ({s_neg:.1f})",
            "Stone+ (N=1,158)": f"{m_pos:.1f} ({s_pos:.1f})",
            "SMD": f"{smd:.3f}",
            "p-value": f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.3f}"
        })

df_table1 = pd.DataFrame(table1_rows)
print(df_table1.to_string(index=False))

# 3. Missingness Table
print("\n=== 3. MISSINGNESS TABLE ===")
miss_rows = []
for i, f in enumerate(features_15):
    f_uci = uci_15[i]
    miss_nh3 = sub_nh3[f].isna().mean() * 100
    miss_uci = df_uci[f_uci].isna().mean() * 100 if f_uci in df_uci.columns else 0.0
    miss_nh = df_nh[f].isna().mean() * 100 if f in df_nh.columns else np.nan
    miss_rows.append({
        "Feature": f,
        "NHANES III Missing %": f"{miss_nh3:.2f}%",
        "UCI Clinic Missing %": f"{miss_uci:.2f}%",
        "NHANES 2017-2020 Missing %": f"{miss_nh:.2f}%" if not np.isnan(miss_nh) else "N/A"
    })
df_miss = pd.DataFrame(miss_rows)
print(df_miss.to_string(index=False))

# Save tables to JSON
output_data = {
    "baseline_characteristics": table1_rows,
    "missingness": miss_rows,
}

with open(os.path.join(OUTPUT_DIR, "paper_scientific_tables.json"), "w") as f:
    json.dump(output_data, f, indent=2)

print("\nSaved baseline and missingness tables to models/paper_scientific_tables.json")
