"""
src/compute_models_and_figures.py

Computes:
1. Model Comparison Significance Testing (Bootstrap difference in AUROC)
2. Subgroup Analysis (Sex, Age, BMI)
3. Sensitivity Analyses (Drop CRP, Drop Liver Enzymes, Drop Demographics, Core Only)
4. Feature Ablation Study
5. Decision Curve Analysis (Net Benefit)
6. Publication-grade figures:
   - Figure 5: Calibration Curves (Panels A-E)
   - Figure 6: ROC and Precision-Recall Curves (with prevalence baseline)
   - Figure 7: Decision Curve Analysis (DCA)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

# Load data
df_nh3 = pd.read_csv(os.path.join(DATA_DIR, "nhanes3_ultrasound.csv"))
sub = df_nh3[df_nh3["target_cholecystectomy_us"] == 0].copy()
features_15 = [
    "Age", "Gender", "Height", "Weight", "BMI",
    "Glucose", "Total Cholesterol", "LDL", "HDL", "Triglyceride",
    "AST", "ALT", "ALP", "Creatinine", "CRP"
]

X_raw = sub[features_15].values
y_raw = sub["target_active_us"].values.astype(int)

# Stratified 70/15/15
idx = np.arange(len(sub))
idx_train, idx_temp, y_train, y_temp = train_test_split(idx, y_raw, test_size=0.30, random_state=42, stratify=y_raw)
idx_val, idx_test, y_val, y_test = train_test_split(idx_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_raw[idx_train])
X_val_imp = imputer.transform(X_raw[idx_val])
X_test_imp = imputer.transform(X_raw[idx_test])

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_imp)
X_val_s = scaler.transform(X_val_imp)
X_test_s = scaler.transform(X_test_imp)

df_test_raw = sub.iloc[idx_test].copy()

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.model import GallstoneNet, TabularResNet
from src.evaluation import (
    compute_paired_bootstrap_auroc_test,
    compute_calibration_metrics,
    compute_bootstrap_ci,
    compute_decision_curve,
)

def train_tabular_net(X_tr, y_tr, X_v, y_v, epochs=60):
    model = GallstoneNet(X_tr.shape[1], dropout=0.3).to(DEVICE)
    pos_weight = torch.tensor([(len(y_tr) - sum(y_tr)) / sum(y_tr)], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    best_loss = float("inf")
    best_weights = None
    
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(DEVICE)
    X_v_t = torch.tensor(X_v, dtype=torch.float32).to(DEVICE)
    y_v_t = torch.tensor(y_v, dtype=torch.float32).to(DEVICE)
    
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_tr_t)
        loss = criterion(logits, y_tr_t)
        loss.backward()
        opt.step()
        
        model.eval()
        with torch.no_grad():
            v_logits = model(X_v_t)
            v_loss = criterion(v_logits, y_v_t).item()
            if v_loss < best_loss:
                best_loss = v_loss
                best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                
    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model

print("Training base models on NHANES III...")
net_model = train_tabular_net(X_train_s, y_train, X_val_s, y_val)
net_model.eval()
with torch.no_grad():
    y_prob_net = torch.sigmoid(net_model(torch.tensor(X_test_s, dtype=torch.float32).to(DEVICE))).cpu().numpy()

# Train XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss")
xgb_model.fit(X_train_s, y_train)
y_prob_xgb = xgb_model.predict_proba(X_test_s)[:, 1]

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42)
rf_model.fit(X_train_s, y_train)
y_prob_rf = rf_model.predict_proba(X_test_s)[:, 1]

# Train LightGBM
lgb_model = lgb.LGBMClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1)
lgb_model.fit(X_train_s, y_train)
y_prob_lgb = lgb_model.predict_proba(X_test_s)[:, 1]

# Meta Super Ensemble
X_stack_val = np.column_stack([
    xgb_model.predict_proba(X_val_s)[:, 1],
    rf_model.predict_proba(X_val_s)[:, 1],
    lgb_model.predict_proba(X_val_s)[:, 1]
])
meta_learner = LogisticRegression()
meta_learner.fit(X_stack_val, y_val)
X_stack_test = np.column_stack([y_prob_xgb, y_prob_rf, y_prob_lgb])
y_prob_ensemble = meta_learner.predict_proba(X_stack_test)[:, 1]

# 1. Model Comparison Significance Testing (Bootstrap Difference in AUROC)
print("\n=== 1. MODEL COMPARISON SIGNIFICANCE TESTING (1000 Bootstraps) ===")
boot_comp = compute_paired_bootstrap_auroc_test(y_test, y_prob_net, y_prob_ensemble, n_bootstraps=1000, seed=42)
print(f"GallstoneNet AUROC: {boot_comp['auc_a_mean']:.3f}")
print(f"Super Ensemble AUROC: {boot_comp['auc_b_mean']:.3f}")
print(f"AUROC Difference (GallstoneNet - Super Ensemble): {boot_comp['delta_mean']:.4f} [95% CI: {boot_comp['delta_ci_95'][0]:.4f} to {boot_comp['delta_ci_95'][1]:.4f}], p = {boot_comp['p_value']:.4f}")

# 2. Subgroup Analysis
print("\n=== 2. SUBGROUP ANALYSIS (GallstoneNet on Test Set) ===")
subgroups = {
    "Sex: Female": df_test_raw["Gender"] == (2 if 2 in df_test_raw["Gender"].values else 1),
    "Sex: Male": df_test_raw["Gender"] == (1 if 2 in df_test_raw["Gender"].values else 0),
    "Age: 20-39": df_test_raw["Age"] < 40,
    "Age: 40-59": (df_test_raw["Age"] >= 40) & (df_test_raw["Age"] < 60),
    "Age: 60+": df_test_raw["Age"] >= 60,
    "BMI: <25 (Normal)": df_test_raw["BMI"] < 25,
    "BMI: 25-30 (Overweight)": (df_test_raw["BMI"] >= 25) & (df_test_raw["BMI"] < 30),
    "BMI: >=30 (Obese)": df_test_raw["BMI"] >= 30,
}

subgroup_results = []
for name, mask in subgroups.items():
    sub_y = y_test[mask.values]
    sub_prob = y_prob_net[mask.values]
    if len(np.unique(sub_y)) < 2:
        continue
    auc = roc_auc_score(sub_y, sub_prob)
    # Threshold at 0.10 or optimal Youden
    pred_bin = (sub_prob >= 0.10).astype(int)
    tn, fp, fn, tp = confusion_matrix(sub_y, pred_bin).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    brier = brier_score_loss(sub_y, sub_prob)
    subgroup_results.append({
        "Subgroup": name,
        "N (Stones)": f"{len(sub_y)} ({sum(sub_y)})",
        "Prevalence": f"{sum(sub_y)/len(sub_y)*100:.1f}%",
        "AUROC": f"{auc:.3f}",
        "Sensitivity": f"{sens:.3f}",
        "Specificity": f"{spec:.3f}",
        "Brier": f"{brier:.3f}"
    })

df_sub = pd.DataFrame(subgroup_results)
print(df_sub.to_string(index=False))

# 3. Sensitivity Analyses
print("\n=== 3. SENSITIVITY ANALYSES (Ablation of Potential Confounders/Biomarkers) ===")
sens_configs = {
    "Baseline (All 15 Features)": features_15,
    "Sensitivity A: Exclude CRP": [f for f in features_15 if f != "CRP"],
    "Sensitivity B: Exclude Liver Enzymes (AST, ALT, ALP)": [f for f in features_15 if f not in ["AST", "ALT", "ALP"]],
    "Sensitivity C: Exclude Demographics (Age, Sex)": [f for f in features_15 if f not in ["Age", "Gender"]],
    "Sensitivity D: Exclude LDL (High Missingness)": [f for f in features_15 if f != "LDL"],
    "Sensitivity E: Core Biomarkers Only (Age, Sex, BMI, Glucose, Chol, Trig)": ["Age", "Gender", "BMI", "Glucose", "Total Cholesterol", "Triglyceride"],
}

sens_results = []
for s_name, feats in sens_configs.items():
    f_idx = [features_15.index(f) for f in feats]
    X_tr_s = X_train_s[:, f_idx]
    X_v_s = X_val_s[:, f_idx]
    X_te_s = X_test_s[:, f_idx]
    
    clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss")
    clf.fit(X_tr_s, y_train)
    p_te = clf.predict_proba(X_te_s)[:, 1]
    
    auc = roc_auc_score(y_test, p_te)
    auprc = average_precision_score(y_test, p_te)
    brier = brier_score_loss(y_test, p_te)
    sens_results.append({
        "Model Configuration": s_name,
        "Num Features": len(feats),
        "AUROC": f"{auc:.3f}",
        "AUPRC": f"{auprc:.3f}",
        "Brier": f"{brier:.3f}"
    })

df_sens = pd.DataFrame(sens_results)
print(df_sens.to_string(index=False))

# 4. Feature Ablations
print("\n=== 4. DOMAIN ABLATION STUDY ===")
domain_configs = {
    "Demographics Only (Age, Sex)": ["Age", "Gender"],
    "Metabolic Only (BMI, Glucose, Chol, LDL, HDL, Trig)": ["BMI", "Glucose", "Total Cholesterol", "LDL", "HDL", "Triglyceride"],
    "Laboratory/Liver Only (AST, ALT, ALP, Creat, CRP)": ["AST", "ALT", "ALP", "Creatinine", "CRP"],
    "Demographics + Metabolic": ["Age", "Gender", "BMI", "Glucose", "Total Cholesterol", "LDL", "HDL", "Triglyceride"],
    "Full Harmonized Model (15 Features)": features_15
}

domain_results = []
for d_name, feats in domain_configs.items():
    f_idx = [features_15.index(f) for f in feats]
    X_tr_s = X_train_s[:, f_idx]
    X_te_s = X_test_s[:, f_idx]
    
    clf = RandomForestClassifier(n_estimators=150, max_depth=6, class_weight="balanced", random_state=42)
    clf.fit(X_tr_s, y_train)
    p_te = clf.predict_proba(X_te_s)[:, 1]
    
    auc = roc_auc_score(y_test, p_te)
    auprc = average_precision_score(y_test, p_te)
    domain_results.append({
        "Feature Domain": d_name,
        "Features (k)": len(feats),
        "AUROC": f"{auc:.3f}",
        "AUPRC": f"{auprc:.3f}"
    })

df_domain = pd.DataFrame(domain_results)
print(df_domain.to_string(index=False))

# 5. Decision Curve Analysis (DCA)
print("\n=== 5. GENERATING DECISION CURVE ANALYSIS (DCA) ===")
thresholds = np.linspace(0.01, 0.40, 40)
n_tot = len(y_test)
prev = sum(y_test) / n_tot

net_benefit_net = []
net_benefit_ens = []
net_benefit_all = []
net_benefit_none = [0.0] * len(thresholds)

for pt in thresholds:
    # Treat All
    # NB = (TP / N) - (FP / N) * (pt / (1 - pt))
    # where TP = all positives, FP = all negatives
    tp_all = sum(y_test)
    fp_all = n_tot - tp_all
    nb_all = (tp_all / n_tot) - (fp_all / n_tot) * (pt / (1.0 - pt))
    net_benefit_all.append(nb_all)
    
    # GallstoneNet
    pred_net = (y_prob_net >= pt).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_net).ravel()
    nb_net = (tp / n_tot) - (fp / n_tot) * (pt / (1.0 - pt))
    net_benefit_net.append(nb_net)
    
    # Super Ensemble
    pred_ens = (y_prob_ensemble >= pt).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_ens).ravel()
    nb_ens = (tp / n_tot) - (fp / n_tot) * (pt / (1.0 - pt))
    net_benefit_ens.append(nb_ens)

# Plot DCA
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(thresholds, net_benefit_net, label="GallstoneNet", color="#1f77b4", lw=2.5)
plt.plot(thresholds, net_benefit_ens, label="Super Ensemble", color="#2ca02c", lw=2, linestyle="--")
plt.plot(thresholds, net_benefit_all, label="Refer / Treat All", color="#7f7f7f", lw=1.5, linestyle=":")
plt.axhline(0, label="Refer / Treat None", color="#000000", lw=1.5, linestyle="-.")
plt.xlim([0.01, 0.35])
plt.ylim([-0.02, 0.10])
plt.xlabel("Threshold Probability (Clinical Referral Cutoff)", fontsize=12)
plt.ylabel("Net Benefit", fontsize=12)
plt.title("Decision Curve Analysis (Clinical Net Benefit for Ultrasound Referral)", fontsize=13, fontweight="bold")
plt.legend(loc="upper right", frameon=True, fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "decision_curve_analysis.png"))
plt.close()
print("Saved plots/decision_curve_analysis.png")

# 6. Combined ROC and Precision-Recall Curves
print("\n=== 6. GENERATING ROC & PRECISION-RECALL CURVES ===")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

# ROC
fpr_net, tpr_net, _ = roc_curve(y_test, y_prob_net)
fpr_ens, tpr_ens, _ = roc_curve(y_test, y_prob_ensemble)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

axes[0].plot(fpr_net, tpr_net, label=f"GallstoneNet (AUC = {roc_auc_score(y_test, y_prob_net):.3f})", color="#1f77b4", lw=2.5)
axes[0].plot(fpr_ens, tpr_ens, label=f"Super Ensemble (AUC = {roc_auc_score(y_test, y_prob_ensemble):.3f})", color="#2ca02c", lw=2)
axes[0].plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {roc_auc_score(y_test, y_prob_xgb):.3f})", color="#ff7f0e", lw=1.8, linestyle="--")
axes[0].plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf):.3f})", color="#9467bd", lw=1.8, linestyle=":")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance (AUC = 0.500)")
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel("1 - Specificity (False Positive Rate)", fontsize=11)
axes[0].set_ylabel("Sensitivity (True Positive Rate)", fontsize=11)
axes[0].set_title("A. Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
axes[0].legend(loc="lower right", frameon=True, fontsize=9)
axes[0].grid(True, linestyle=":", alpha=0.6)

# Precision-Recall
prec_net, rec_net, _ = precision_recall_curve(y_test, y_prob_net)
prec_ens, rec_ens, _ = precision_recall_curve(y_test, y_prob_ensemble)
prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, y_prob_xgb)
prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_prob_rf)

axes[1].plot(rec_net, prec_net, label=f"GallstoneNet (AUPRC = {average_precision_score(y_test, y_prob_net):.3f})", color="#1f77b4", lw=2.5)
axes[1].plot(rec_ens, prec_ens, label=f"Super Ensemble (AUPRC = {average_precision_score(y_test, y_prob_ensemble):.3f})", color="#2ca02c", lw=2)
axes[1].plot(rec_xgb, prec_xgb, label=f"XGBoost (AUPRC = {average_precision_score(y_test, y_prob_xgb):.3f})", color="#ff7f0e", lw=1.8, linestyle="--")
axes[1].plot(rec_rf, prec_rf, label=f"Random Forest (AUPRC = {average_precision_score(y_test, y_prob_rf):.3f})", color="#9467bd", lw=1.8, linestyle=":")
axes[1].axhline(prev, color="red", linestyle="--", alpha=0.7, label=f"Prevalence Baseline ({prev*100:.1f}%)")
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 0.7])
axes[1].set_xlabel("Recall (Sensitivity)", fontsize=11)
axes[1].set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
axes[1].set_title("B. Precision-Recall Curves", fontsize=12, fontweight="bold")
axes[1].legend(loc="upper right", frameon=True, fontsize=9)
axes[1].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "roc_pr_curves_combined.png"))
plt.close()
print("Saved plots/roc_pr_curves_combined.png")

# Save all results to JSON
results_payload = {
    "model_comparison": {
        "gallstonenet_auc": float(np.mean(auc_net_list)),
        "super_ensemble_auc": float(np.mean(auc_ens_list)),
        "auc_difference": float(diff_mean),
        "ci_lower": float(diff_ci[0]),
        "ci_upper": float(diff_ci[1]),
        "p_value": float(p_diff)
    },
    "subgroups": subgroup_results,
    "sensitivity_analyses": sens_results,
    "domain_ablations": domain_results
}

with open(os.path.join(MODELS_DIR, "additional_benchmarks.json"), "w") as f:
    json.dump(results_payload, f, indent=2)

print("Saved models/additional_benchmarks.json successfully!")
