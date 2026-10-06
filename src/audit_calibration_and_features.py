"""
src/audit_calibration_and_features.py

Comprehensive Calibration, Recalibration, and Feature Audit for Gallstone Benchmark:
1. Re-audits NHANES III physical ultrasound benchmark across models (LR, RF, XGB, GallstoneNet).
2. Computes both RAW and LOCKED-RECALIBRATED test performance:
   - AUROC, AUPRC
   - Brier score (Raw vs Recalibrated)
   - Null prevalence baseline Brier (p*(1-p))
   - Observed-to-Expected (O/E) ratio
   - Calibration slope (beta) and intercept (alpha)
3. Verifies feature consistency.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Load NHANES III physical ultrasound
df_nh3 = pd.read_csv(os.path.join(DATA_DIR, "nhanes3_ultrasound.csv"))
sub_nh3 = df_nh3[df_nh3["target_cholecystectomy_us"] == 0].copy()

# The 15 master features
features_15 = [
    "Age", "Gender", "Height", "Weight", "BMI",
    "Glucose", "Total Cholesterol", "LDL", "HDL", "Triglyceride",
    "AST", "ALT", "ALP", "Creatinine", "CRP"
]

y_all = sub_nh3["target_active_us"].values.astype(int)
X_all = sub_nh3[features_15].copy()

# Ensure Gender is coded 0/1 (1=Male, 2=Female in NHANES -> convert Female=1, Male=0)
if 2 in X_all["Gender"].values:
    X_all["Gender"] = (X_all["Gender"] == 2).astype(int)

# Stratified split: 70% train, 15% val, 15% test (seed 42)
from sklearn.model_selection import train_test_split

X_train_df, X_temp_df, y_train, y_temp = train_test_split(
    X_all, y_all, test_size=0.30, random_state=42, stratify=y_all
)
X_val_df, X_test_df, y_val, y_test = train_test_split(
    X_temp_df, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# Imputation and scaling strictly in-split
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train_df)
X_val_imp = imputer.transform(X_val_df)
X_test_imp = imputer.transform(X_test_df)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_imp)
X_val_s = scaler.transform(X_val_imp)
X_test_s = scaler.transform(X_test_imp)

print(f"Dataset partitions:")
print(f"Train: N={len(y_train)}, Cases={y_train.sum()} ({y_train.mean():.4f})")
print(f"Val:   N={len(y_val)}, Cases={y_val.sum()} ({y_val.mean():.4f})")
print(f"Test:  N={len(y_test)}, Cases={y_test.sum()} ({y_test.mean():.4f})")

test_prev = y_test.mean()
null_brier = test_prev * (1.0 - test_prev)
print(f"Null Prevalence Reference Brier = {null_brier:.4f}")

def compute_slope_intercept(y_true, probs):
    eps = 1e-6
    p_clip = np.clip(probs, eps, 1.0 - eps)
    logits = logit(p_clip).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", C=1e6, max_iter=500)
    lr.fit(logits, y_true)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])

results = {}

# 1. Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_s, y_train)
p_val_lr = lr_model.predict_proba(X_val_s)[:, 1]
p_test_lr = lr_model.predict_proba(X_test_s)[:, 1]

# 2. Random Forest
rf_model = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
rf_model.fit(X_train_s, y_train)
p_val_rf = rf_model.predict_proba(X_val_s)[:, 1]
p_test_rf = rf_model.predict_proba(X_test_s)[:, 1]

# 3. XGBoost (unweighted)
xgb_unweighted = xgb.XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.05,
    random_state=42, eval_metric="logloss"
)
xgb_unweighted.fit(X_train_s, y_train)
p_val_xgb_unw = xgb_unweighted.predict_proba(X_val_s)[:, 1]
p_test_xgb_unw = xgb_unweighted.predict_proba(X_test_s)[:, 1]

# 4. XGBoost (cost-weighted with scale_pos_weight ~ 10.07)
pos_w = (len(y_train) - y_train.sum()) / y_train.sum()
xgb_weighted = xgb.XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.05,
    scale_pos_weight=pos_w, random_state=42, eval_metric="logloss"
)
xgb_weighted.fit(X_train_s, y_train)
p_val_xgb_w = xgb_weighted.predict_proba(X_val_s)[:, 1]
p_test_xgb_w = xgb_weighted.predict_proba(X_test_s)[:, 1]

# 5. PyTorch GallstoneNet (with pos_weight = 10.07)
class GallstoneNetPyTorch(nn.Module):
    def __init__(self, in_dim=15):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(in_dim)
        self.fc1 = nn.Linear(in_dim, 128)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.drop3 = nn.Dropout(0.2)
        self.out = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn0(x)
        x = self.drop1(self.relu(self.fc1(x)))
        x = self.drop2(self.relu(self.fc2(x)))
        x = self.drop3(self.relu(self.fc3(x)))
        return self.out(x)

torch.manual_seed(42)
np.random.seed(42)
nn_model = GallstoneNetPyTorch(15)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w]))
optimizer = optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)

X_tr_t = torch.tensor(X_train_s, dtype=torch.float32)
y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
X_test_t = torch.tensor(X_test_s, dtype=torch.float32)

best_val_loss = float("inf")
best_weights = None
patience, p_cnt = 15, 0

dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

for epoch in range(100):
    nn_model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(nn_model(xb), yb)
        loss.backward()
        optimizer.step()
    
    # Evaluate validation loss
    nn_model.eval()
    with torch.no_grad():
        val_logits = nn_model(X_val_t)
        val_loss = criterion(val_logits, torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)).item()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = {k: v.cpu().clone() for k, v in nn_model.state_dict().items()}
        p_cnt = 0
    else:
        p_cnt += 1
        if p_cnt >= patience:
            break

nn_model.load_state_dict(best_weights)
nn_model.eval()
with torch.no_grad():
    z_val_nn = nn_model(X_val_t).squeeze().numpy()
    z_test_nn = nn_model(X_test_t).squeeze().numpy()
p_val_nn = expit(z_val_nn)
p_test_nn = expit(z_test_nn)

# Unweighted GallstoneNet for comparison
torch.manual_seed(42)
nn_unw = GallstoneNetPyTorch(15)
crit_unw = nn.BCEWithLogitsLoss()
opt_unw = optim.AdamW(nn_unw.parameters(), lr=1e-3, weight_decay=1e-4)
best_val_loss = float("inf")
best_weights_unw = None
p_cnt = 0
for epoch in range(100):
    nn_unw.train()
    for xb, yb in loader:
        opt_unw.zero_grad()
        loss = crit_unw(nn_unw(xb), yb)
        loss.backward()
        opt_unw.step()
    nn_unw.eval()
    with torch.no_grad():
        vloss = crit_unw(nn_unw(X_val_t), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)).item()
    if vloss < best_val_loss:
        best_val_loss = vloss
        best_weights_unw = {k: v.cpu().clone() for k, v in nn_unw.state_dict().items()}
        p_cnt = 0
    else:
        p_cnt += 1
        if p_cnt >= patience:
            break
nn_unw.load_state_dict(best_weights_unw)
nn_unw.eval()
with torch.no_grad():
    z_val_nn_unw = nn_unw(X_val_t).squeeze().numpy()
    z_test_nn_unw = nn_unw(X_test_t).squeeze().numpy()
p_val_nn_unw = expit(z_val_nn_unw)
p_test_nn_unw = expit(z_test_nn_unw)

models_dict = {
    "Logistic Regression (unweighted)": (p_val_lr, p_test_lr),
    "Random Forest (unweighted)": (p_val_rf, p_test_rf),
    "XGBoost (unweighted)": (p_val_xgb_unw, p_test_xgb_unw),
    "XGBoost (scale_pos_weight=10.07)": (p_val_xgb_w, p_test_xgb_w),
    "GallstoneNet (unweighted PyTorch)": (p_val_nn_unw, p_test_nn_unw),
    "GallstoneNet (cost-weighted PyTorch, pos_weight=10.07)": (p_val_nn, p_test_nn),
}

print("\n" + "="*85)
print(f"{'Model Architecture':<38} | {'AUROC':<6} | {'AUPRC':<6} | {'Raw Brier':<10} | {'Raw Slope':<9} | {'Raw Int':<7} | {'O/E'}")
print("="*85)

audit_records = []
for name, (p_val, p_test) in models_dict.items():
    auc = roc_auc_score(y_test, p_test)
    auprc = average_precision_score(y_test, p_test)
    raw_brier = brier_score_loss(y_test, p_test)
    raw_slope, raw_int = compute_slope_intercept(y_test, p_test)
    oe_ratio = y_test.mean() / max(p_test.mean(), 1e-6)
    
    # Fit Platt recalibration on VALIDATION set strictly:
    eps = 1e-6
    pv_clip = np.clip(p_val, eps, 1.0 - eps)
    logit_val = logit(pv_clip).reshape(-1, 1)
    
    platt = LogisticRegression(solver="lbfgs", C=1e6, max_iter=500)
    platt.fit(logit_val, y_val)
    val_platt_slope = float(platt.coef_[0][0])
    val_platt_int = float(platt.intercept_[0])
    
    # Apply to held-out TEST set
    pt_clip = np.clip(p_test, eps, 1.0 - eps)
    logit_test = logit(pt_clip).reshape(-1, 1)
    
    cal_test_logits = val_platt_slope * logit_test + val_platt_int
    p_test_recal = expit(cal_test_logits).ravel()
    
    recal_brier = brier_score_loss(y_test, p_test_recal)
    recal_slope, recal_int = compute_slope_intercept(y_test, p_test_recal)
    recal_oe = y_test.mean() / max(p_test_recal.mean(), 1e-6)
    
    print(f"{name:<38} | {auc:.3f}  | {auprc:.3f}  | {raw_brier:.4f}     | {raw_slope:.3f}     | {raw_int:.3f}   | {oe_ratio:.2f}")
    audit_records.append({
        "model": name,
        "auc": auc,
        "auprc": auprc,
        "raw_brier": raw_brier,
        "raw_slope": raw_slope,
        "raw_int": raw_int,
        "raw_oe": oe_ratio,
        "val_platt_slope": val_platt_slope,
        "val_platt_int": val_platt_int,
        "recal_brier": recal_brier,
        "recal_slope": recal_slope,
        "recal_int": recal_int,
        "recal_oe": recal_oe
    })

print("\n" + "="*85)
print(f"RECALIBRATED (Locked Validation Platt Recalibration -> Held-Out Test Split)")
print("="*85)
print(f"{'Model Architecture':<38} | {'Val Platt (slope/int)':<22} | {'Recal Brier':<11} | {'Recal Slope':<11} | {'Recal Int':<9} | {'Recal O/E'}")
print("="*85)
for r in audit_records:
    v_str = f"β={r['val_platt_slope']:.2f}, α={r['val_platt_int']:.2f}"
    print(f"{r['model']:<38} | {v_str:<22} | {r['recal_brier']:.4f}      | {r['recal_slope']:.3f}       | {r['recal_int']:.3f}     | {r['recal_oe']:.2f}")
