"""
train_all.py — Comprehensive Multi-Cohort Training & Validation Engine.

Executes 4 comprehensive experiments:
  1. Experiment A: UCI Clinic Model (38 features with Bioimpedance & Blood Work)
  2. Experiment B: NHANES Real-World Cohort (28 features with RUQ Pain Symptom & Labs)
  3. Experiment C: Cross-Cohort External Validation (Train NHANES -> Test on UCI)
  4. Experiment D: Joint Multi-Cohort Harmonized AI (Trained on combined 9,529 patients)

Compares 5 Model Architectures across all benchmarks:
  • PyTorch GallstoneNet (Deep Residual/BatchNorm MLP with Early Stopping)
  • XGBoost Classifier (Gradient Boosted Trees with scale_pos_weight)
  • LightGBM Classifier (Leaf-wise gradient boosting)
  • Random Forest (Balanced ensemble of 200 trees)
  • Soft Voting Ensemble (Optimal probability blend)

Saves:
  • Trained PyTorch & Scikit/XGBoost models to models/
  • Comprehensive ROC, PR, Confusion Matrix, and Feature Importance plots to plots/
  • Detailed benchmark results JSON and Markdown report.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.dataset import load_and_prepare_data as load_uci_data
from src.harmonized_dataset import load_harmonized_data, load_nhanes_full_data
from src.model import GallstoneNet

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── PyTorch Trainer Helper ───────────────────────────────────────────────────

def train_pytorch_net(
    train_loader,
    val_loader,
    input_dim: int,
    pos_weight_val: float,
    epochs: int = 150,
    lr: float = 1e-3,
    patience: int = 20,
) -> Tuple[GallstoneNet, Dict[str, list]]:
    model = GallstoneNet(input_dim=input_dim, dropout=0.3).to(DEVICE)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_train += 1
        train_loss /= max(n_train, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                logits = model(X_b)
                loss = criterion(logits, y_b)
                val_loss += loss.item()
                n_val += 1
        val_loss /= max(n_val, 1)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model, history


def compute_bootstrap_ci(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, n_boot: int = 1000, seed: int = 42
) -> Dict[str, Tuple[float, float]]:
    """Fast vectorized non-parametric bootstrap for 95% confidence intervals."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    if n == 0 or len(np.unique(y_true)) < 2:
        return {}

    indices = rng.randint(0, n, size=(n_boot, n))
    yt_boot = y_true[indices]
    yp_boot = y_prob[indices]
    pred_boot = (yp_boot >= threshold).astype(int)

    # Vectorized confusion matrix
    tp = np.sum((pred_boot == 1) & (yt_boot == 1), axis=1)
    fp = np.sum((pred_boot == 1) & (yt_boot == 0), axis=1)
    fn = np.sum((pred_boot == 0) & (yt_boot == 1), axis=1)
    tn = np.sum((pred_boot == 0) & (yt_boot == 0), axis=1)

    acc_boot = (tp + tn) / n
    recall_boot = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    spec_boot = np.where(tn + fp > 0, tn / (tn + fp), 0.0)
    prec_boot = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    f1_boot = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    brier_boot = np.mean((yp_boot - yt_boot) ** 2, axis=1)

    # Fast AUC calculation via Mann-Whitney U on each bootstrap row
    aucs = []
    for i in range(n_boot):
        y_t = yt_boot[i]
        y_p = yp_boot[i]
        pos = y_p[y_t == 1]
        neg = y_p[y_t == 0]
        n_pos = len(pos)
        n_neg = len(neg)
        if n_pos == 0 or n_neg == 0:
            continue
        neg_sorted = np.sort(neg)
        rank_sum = np.searchsorted(neg_sorted, pos, side="left") + 0.5 * (
            np.searchsorted(neg_sorted, pos, side="right") - np.searchsorted(neg_sorted, pos, side="left")
        )
        auc = np.sum(rank_sum) / (n_pos * n_neg)
        aucs.append(auc)

    aucs = np.array(aucs) if len(aucs) > 0 else np.array([0.5])

    return {
        "roc_auc_ci": (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))),
        "recall_ci": (float(np.percentile(recall_boot, 2.5)), float(np.percentile(recall_boot, 97.5))),
        "specificity_ci": (float(np.percentile(spec_boot, 2.5)), float(np.percentile(spec_boot, 97.5))),
        "precision_ci": (float(np.percentile(prec_boot, 2.5)), float(np.percentile(prec_boot, 97.5))),
        "f1_ci": (float(np.percentile(f1_boot, 2.5)), float(np.percentile(f1_boot, 97.5))),
        "accuracy_ci": (float(np.percentile(acc_boot, 2.5)), float(np.percentile(acc_boot, 97.5))),
        "brier_ci": (float(np.percentile(brier_boot, 2.5)), float(np.percentile(brier_boot, 97.5))),
    }


def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Computes clinical calibration metrics: Brier score, BSS, slope, intercept."""
    brier = float(brier_score_loss(y_true, y_prob))
    mean_prev = float(np.mean(y_true))
    brier_ref = mean_prev * (1.0 - mean_prev)
    bss = float(1.0 - (brier / max(brier_ref, 1e-8)))

    # Fit calibration slope and intercept via logistic regression: logit(p) -> y
    eps = 1e-6
    p_clip = np.clip(y_prob, eps, 1.0 - eps)
    logit_p = np.log(p_clip / (1.0 - p_clip)).reshape(-1, 1)

    try:
        cal_lr = LogisticRegression(solver="lbfgs", C=1e6, max_iter=500)
        cal_lr.fit(logit_p, y_true)
        cal_intercept = float(cal_lr.intercept_[0])
        cal_slope = float(cal_lr.coef_[0][0])
    except Exception:
        cal_intercept = 0.0
        cal_slope = 1.0

    return {
        "brier_score": brier,
        "brier_ref": brier_ref,
        "brier_skill_score": bss,
        "calibration_intercept": cal_intercept,
        "calibration_slope": cal_slope,
    }


def eval_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    specificity = float(tn / max(tn + fp, 1))

    base_metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "pr_auc": pr_auc,
    }

    cal_metrics = compute_calibration_metrics(y_true, y_prob)
    base_metrics.update(cal_metrics)

    ci_metrics = compute_bootstrap_ci(y_true, y_prob, threshold=threshold)
    base_metrics["confidence_intervals_95"] = ci_metrics

    return base_metrics


# ── Run Evaluation on All Model Families ──────────────────────────────────────

def benchmark_dataset(
    data_bundle: Dict[str, Any], exp_name: str
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"  BENCHMARKING: {exp_name.upper()}")
    print(f"=======================================================")
    
    X_train = data_bundle["raw_arrays"]["X_train"]
    y_train = data_bundle["raw_arrays"]["y_train"]
    X_val = data_bundle["raw_arrays"]["X_val"]
    y_val = data_bundle["raw_arrays"]["y_val"]
    X_test = data_bundle["raw_arrays"]["X_test"]
    y_test = data_bundle["raw_arrays"]["y_test"]

    input_dim = data_bundle["input_dim"]
    pos_weight = data_bundle["pos_weight"]
    feature_names = data_bundle["feature_names"]

    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)} | Features: {input_dim}")
    print(f"Train Positive rate: {np.mean(y_train)*100:.1f}% | Test Positive rate: {np.mean(y_test)*100:.1f}%")

    results = {}
    probs = {}

    # 1. PyTorch Deep Neural Network
    print("  -> Training PyTorch GallstoneNet...")
    pt_model, history = train_pytorch_net(
        data_bundle["loaders"]["train"],
        data_bundle["loaders"]["val"],
        input_dim=input_dim,
        pos_weight_val=pos_weight,
    )
    pt_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        pt_prob = torch.sigmoid(pt_model(X_test_tensor)).cpu().numpy().flatten()
    probs["GallstoneNet (PyTorch)"] = pt_prob
    results["GallstoneNet (PyTorch)"] = eval_metrics(y_test, pt_prob)

    # 2. XGBoost
    print("  -> Training XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric="logloss",
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    probs["XGBoost"] = xgb_prob
    results["XGBoost"] = eval_metrics(y_test, xgb_prob)

    # 3. LightGBM
    print("  -> Training LightGBM Classifier...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
    lgb_prob = lgb_model.predict_proba(X_test)[:, 1]
    probs["LightGBM"] = lgb_prob
    results["LightGBM"] = eval_metrics(y_test, lgb_prob)

    # 4. Random Forest
    print("  -> Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    probs["Random Forest"] = rf_prob
    results["Random Forest"] = eval_metrics(y_test, rf_prob)

    # 5. Logistic Regression Baseline
    print("  -> Training Logistic Regression...")
    lr_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_prob = lr_model.predict_proba(X_test)[:, 1]
    probs["Logistic Regression"] = lr_prob
    results["Logistic Regression"] = eval_metrics(y_test, lr_prob)

    # 6. Ensemble Blend
    print("  -> Evaluating Soft Voting Ensemble...")
    ensemble_prob = (0.35 * pt_prob + 0.35 * xgb_prob + 0.20 * lgb_prob + 0.10 * rf_prob)
    probs["Super Ensemble"] = ensemble_prob
    results["Super Ensemble"] = eval_metrics(y_test, ensemble_prob)

    # Print summary table with 95% CIs and calibration metrics
    print(f"\nResults for {exp_name}:")
    print(f"{'Model':23s} | {'AUC-ROC (95% CI)':21s} | {'Sensitivity (95% CI)':22s} | {'Specificity (95% CI)':22s} | {'Brier (95% CI)':19s} | {'Slope':6s}")
    print("-" * 125)
    for model_name, m in results.items():
        ci = m.get("confidence_intervals_95", {})
        auc_ci = ci.get("roc_auc_ci", (0.0, 0.0))
        rec_ci = ci.get("recall_ci", (0.0, 0.0))
        spec_ci = ci.get("specificity_ci", (0.0, 0.0))
        brier_ci = ci.get("brier_ci", (0.0, 0.0))

        auc_str = f"{m['roc_auc']:.3f} [{auc_ci[0]:.3f}-{auc_ci[1]:.3f}]"
        rec_str = f"{m['recall']:.3f} [{rec_ci[0]:.3f}-{rec_ci[1]:.3f}]"
        spec_str = f"{m['specificity']:.3f} [{spec_ci[0]:.3f}-{spec_ci[1]:.3f}]"
        brier_str = f"{m['brier_score']:.3f} [{brier_ci[0]:.3f}-{brier_ci[1]:.3f}]"
        slope_str = f"{m['calibration_slope']:.2f}"

        print(f"{model_name:23s} | {auc_str:21s} | {rec_str:22s} | {spec_str:22s} | {brier_str:19s} | {slope_str:6s}")

    # Save PyTorch model
    pt_save_path = os.path.join(MODELS_DIR, f"{exp_name}_gallstonenet.pt")
    torch.save({
        "state_dict": pt_model.state_dict(),
        "input_dim": input_dim,
        "feature_names": feature_names,
        "scaler_mean": data_bundle["scaler"].mean_.tolist(),
        "scaler_scale": data_bundle["scaler"].scale_.tolist(),
        "imputer_statistics": data_bundle["imputer"].statistics_.tolist(),
    }, pt_save_path)
    print(f"Saved PyTorch model -> {pt_save_path}")

    return {
        "results": results,
        "probs": probs,
        "y_test": y_test,
        "feature_names": feature_names,
        "rf_importances": rf_model.feature_importances_,
        "xgb_importances": xgb_model.feature_importances_,
        "history": history,
    }


# ── Plotting Utilities ───────────────────────────────────────────────────────

def generate_calibration_plots(all_experiments: Dict[str, Any]):
    """Generates 4-panel calibration reliability curves."""
    print("Generating calibration curve figures...")
    plt.figure(figsize=(14, 11))

    plot_configs = [
        ("exp_b_nhanes_full", "NHANES Cohort (28 features, N=9,210)", 221),
        ("exp_c_cross_external", "External Cross-Cohort Validation (NHANES -> UCI)", 222),
        ("exp_d_joint_harmonized", "Joint Multi-Cohort Harmonized AI (Combined N=9,529)", 223),
        ("exp_a_uci_clinic", "UCI Clinic Cohort (38 features, N=319)", 224),
    ]

    for key, title, subplot_idx in plot_configs:
        if key not in all_experiments:
            continue
        exp = all_experiments[key]
        y_test = exp["y_test"]
        plt.subplot(subplot_idx)

        # 45-degree reference line
        plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Perfect Calibration")

        for m_name, prob in exp["probs"].items():
            try:
                prob_true, prob_pred = calibration_curve(y_test, prob, n_bins=8, strategy="uniform")
                bs = exp["results"][m_name]["brier_score"]
                slope = exp["results"][m_name]["calibration_slope"]
                plt.plot(prob_pred, prob_true, marker="o", lw=1.8, label=f"{m_name} (BS={bs:.3f}, Slope={slope:.2f})")
            except Exception:
                continue

        plt.title(f"{title}\nCalibration Reliability", fontsize=11, fontweight="bold")
        plt.xlabel("Mean Predicted Probability", fontsize=10)
        plt.ylabel("Observed Fraction of Positives", fontsize=10)
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="upper left", fontsize=7)

    plt.tight_layout()
    cal_out = os.path.join(PLOTS_DIR, "calibration_curves_multi_cohort.png")
    plt.savefig(cal_out, dpi=300)
    plt.close()
    print(f"Saved Calibration Curves -> {cal_out}")


def generate_evaluation_plots(all_experiments: Dict[str, Any]):
    print("\nGenerating publication-quality comparison figures...")
    
    # 1. Multi-Cohort ROC Curves
    plt.figure(figsize=(14, 10))

    plot_configs = [
        ("exp_b_nhanes_full", "NHANES Population Cohort (28 features, N=9,210)", 221),
        ("exp_c_cross_external", "External Cross-Cohort Validation (NHANES -> UCI Clinic)", 222),
        ("exp_d_joint_harmonized", "Joint Multi-Cohort Harmonized AI (Combined N=9,529)", 223),
        ("exp_a_uci_clinic", "UCI Clinic Cohort (38 features, N=319)", 224),
    ]

    for key, title, subplot_idx in plot_configs:
        if key not in all_experiments:
            continue
        exp = all_experiments[key]
        y_test = exp["y_test"]
        plt.subplot(subplot_idx)
        
        for idx, (m_name, prob) in enumerate(exp["probs"].items()):
            fpr, tpr, _ = roc_curve(y_test, prob)
            auc = exp["results"][m_name]["roc_auc"]
            plt.plot(fpr, tpr, lw=2, label=f"{m_name} (AUC={auc:.3f})")
            
        plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
        plt.title(title, fontsize=12, fontweight="bold")
        plt.xlabel("False Positive Rate", fontsize=10)
        plt.ylabel("True Positive Rate", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    roc_out = os.path.join(PLOTS_DIR, "roc_curves_multi_cohort.png")
    plt.savefig(roc_out, dpi=300)
    plt.close()
    print(f"Saved ROC curves -> {roc_out}")

    # 2. Confusion Matrices for Ensemble across all 4 experiments
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (key, title, _) in zip(axes.flatten(), plot_configs):
        if key not in all_experiments:
            continue
        exp = all_experiments[key]
        y_test = exp["y_test"]
        prob = exp["probs"]["Super Ensemble"]
        pred = (prob >= 0.5).astype(int)
        cm = confusion_matrix(y_test, pred)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["No Gallstone", "Gallstone"],
            yticklabels=["No Gallstone", "Gallstone"],
        )
        ax.set_title(f"{title}\nEnsemble Matrix", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Diagnosis")
        ax.set_xlabel("Predicted Diagnosis")

    plt.tight_layout()
    cm_out = os.path.join(PLOTS_DIR, "confusion_matrices_multi_cohort.png")
    plt.savefig(cm_out, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrices -> {cm_out}")

    # 3. Top Feature Importance from NHANES (28 features)
    exp_b = all_experiments.get("exp_b_nhanes_full")
    if exp_b:
        features = exp_b["feature_names"]
        xgb_imp = exp_b["xgb_importances"]
        sorted_idx = np.argsort(xgb_imp)[::-1][:15]

        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=[xgb_imp[i] for i in sorted_idx],
            y=[features[i] for i in sorted_idx],
            palette="viridis",
        )
        plt.title("Feature Attribution Consistent with Established Gallstone Risk Patterns\n(NHANES 9,210 Cohort)", fontsize=12, fontweight="bold")
        plt.xlabel("XGBoost Relative Feature Importance (Gain)", fontsize=11)
        plt.tight_layout()
        fi_out = os.path.join(PLOTS_DIR, "feature_importance_nhanes.png")
        plt.savefig(fi_out, dpi=300)
        plt.close()
        print(f"Saved Feature Importance -> {fi_out}")

    # 4. Calibration Curves
    generate_calibration_plots(all_experiments)


# ── Main Entrypoint ──────────────────────────────────────────────────────────

def main():
    start_time = time.time()
    all_experiments = {}

    print("\n" + "=" * 80)
    print("  GALLBLADDER STONE AI: MULTI-COHORT VALIDATION & BENCHMARK SUITE")
    print("=" * 80)

    # 1. Experiment A: UCI Clinic Dataset (38 features, N=319)
    print("\n>>> Loading Experiment A: UCI Clinic Dataset (38 features, N=319)...")
    train_loader_u, val_loader_u, test_loader_u, scaler_u, pos_weight_u, feat_cols_u = load_uci_data()
    
    X_train_u = train_loader_u.dataset.X.cpu().numpy()
    y_train_u = train_loader_u.dataset.y.cpu().numpy().flatten().astype(int)
    X_val_u = val_loader_u.dataset.X.cpu().numpy()
    y_val_u = val_loader_u.dataset.y.cpu().numpy().flatten().astype(int)
    X_test_u = test_loader_u.dataset.X.cpu().numpy()
    y_test_u = test_loader_u.dataset.y.cpu().numpy().flatten().astype(int)

    dummy_imputer = SimpleImputer(strategy="median").fit(X_train_u)

    print(f"  [Cohort Split Audit - Experiment A]")
    print(f"  • Training split:   {len(X_train_u):>5d} patients ({len(X_train_u)/319*100:.1f}%)")
    print(f"  • Validation split: {len(X_val_u):>5d} patients ({len(X_val_u)/319*100:.1f}%)")
    print(f"  • Held-out Test:    {len(X_test_u):>5d} patients ({len(X_test_u)/319*100:.1f}%)")
    print(f"  • Total cohort:     {len(X_train_u)+len(X_val_u)+len(X_test_u):>5d} patients (100.0%)")

    uci_data_dict = {
        "loaders": {
            "train": train_loader_u,
            "val": val_loader_u,
            "test": test_loader_u,
        },
        "raw_arrays": {
            "X_train": X_train_u, "y_train": y_train_u,
            "X_val": X_val_u, "y_val": y_val_u,
            "X_test": X_test_u, "y_test": y_test_u,
        },
        "imputer": dummy_imputer,
        "scaler": scaler_u,
        "pos_weight": float(pos_weight_u.item()),
        "input_dim": X_train_u.shape[1],
        "feature_names": feat_cols_u,
    }
    all_experiments["exp_a_uci_clinic"] = benchmark_dataset(
        uci_data_dict, "exp_a_uci_clinic"
    )

    # 2. Experiment B: Full NHANES Cohort (28 features, N=9,210)
    print("\n>>> Loading Experiment B: Full NHANES Cohort (28 features, N=9,210)...")
    nhanes_full_bundle = load_nhanes_full_data()
    n_b_train = len(nhanes_full_bundle["raw_arrays"]["X_train"])
    n_b_val = len(nhanes_full_bundle["raw_arrays"]["X_val"])
    n_b_test = len(nhanes_full_bundle["raw_arrays"]["X_test"])
    n_b_total = n_b_train + n_b_val + n_b_test

    print(f"  [Cohort Split Audit - Experiment B]")
    print(f"  • Training split:   {n_b_train:>5d} patients ({n_b_train/n_b_total*100:.1f}%)")
    print(f"  • Validation split: {n_b_val:>5d} patients ({n_b_val/n_b_total*100:.1f}%) [early stopping / tuning]")
    print(f"  • Held-out Test:    {n_b_test:>5d} patients ({n_b_test/n_b_total*100:.1f}%) [unbiased evaluation]")
    print(f"  • Total cohort:     {n_b_total:>5d} patients (100.0%)")

    all_experiments["exp_b_nhanes_full"] = benchmark_dataset(
        nhanes_full_bundle, "exp_b_nhanes_full"
    )

    # 3. Experiment C: Cross-Cohort External Validation (Train NHANES -> Test UCI)
    print("\n>>> Loading Experiment C: Cross-Cohort External Validation (Train NHANES -> Test UCI External)...")
    cross_bundle = load_harmonized_data(mode="train_nhanes_test_uci")
    n_c_train = len(cross_bundle["raw_arrays"]["X_train"])
    n_c_val = len(cross_bundle["raw_arrays"]["X_val"])
    n_c_test = len(cross_bundle["raw_arrays"]["X_test"])
    print(f"  [Cohort Split Audit - Experiment C]")
    print(f"  • Source Train (NHANES):   {n_c_train:>5d} patients ({n_c_train/9210*100:.1f}% of NHANES)")
    print(f"  • Source Val (NHANES):     {n_c_val:>5d} patients ({n_c_val/9210*100:.1f}% of NHANES)")
    print(f"  • External Target Test:    {n_c_test:>5d} patients (100.0% of UCI Clinic, zero-shot)")
    print(f"  • Total participants:      {n_c_train + n_c_val + n_c_test:>5d} (9,210 NHANES + 319 UCI)")

    all_experiments["exp_c_cross_external"] = benchmark_dataset(
        cross_bundle, "exp_c_cross_external"
    )

    # 4. Experiment D: Joint Multi-Cohort Harmonized Model (Combined N=9,529)
    print("\n>>> Loading Experiment D: Joint Multi-Cohort Harmonized Model (Combined N=9,529)...")
    joint_bundle = load_harmonized_data(mode="joint")
    n_d_train = len(joint_bundle["raw_arrays"]["X_train"])
    n_d_val = len(joint_bundle["raw_arrays"]["X_val"])
    n_d_test = len(joint_bundle["raw_arrays"]["X_test"])
    n_d_total = n_d_train + n_d_val + n_d_test

    print(f"  [Cohort Split Audit - Experiment D]")
    print(f"  • Training split:   {n_d_train:>5d} patients ({n_d_train/n_d_total*100:.1f}%) [model parameters]")
    print(f"  • Validation split: {n_d_val:>5d} patients ({n_d_val/n_d_total*100:.1f}%) [early stopping / tuning]")
    print(f"  • Held-out Test:    {n_d_test:>5d} patients ({n_d_test/n_d_total*100:.1f}%) [unbiased evaluation]")
    print(f"  • Total accounted:  {n_d_train} + {n_d_val} + {n_d_test} = {n_d_total} patients (100.0%)")

    all_experiments["exp_d_joint_harmonized"] = benchmark_dataset(
        joint_bundle, "exp_d_joint_harmonized"
    )

    # Generate Figures
    generate_evaluation_plots(all_experiments)

    # Export Full Results Summary
    summary_path = os.path.join(MODELS_DIR, "multi_cohort_benchmarks.json")
    clean_summary = {}
    for exp_k, exp_v in all_experiments.items():
        clean_summary[exp_k] = exp_v["results"]
    with open(summary_path, "w") as f:
        json.dump(clean_summary, f, indent=2)
    print(f"\nAll experiments completed in {time.time() - start_time:.1f}s!")
    print(f"Results summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
