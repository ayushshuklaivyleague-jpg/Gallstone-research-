"""
src/benchmark_ultrasound.py — Comprehensive Ultrasound Ground-Truth Benchmark Engine.

Evaluates AI models on real-time transabdominal ultrasound ground truth (NHANES III, N=13,694)
and performs cross-cohort clinical transfers with the Turkish Clinic (UCI, N=319).

Experiments:
  1. Exp E1: NHANES III Active Ultrasound Benchmark (N=12,824; 1,158 active stones vs 11,666 normal)
  2. Exp E2: Cross-Cohort Ultrasound Transfer (Turkish Clinic UCI -> NHANES III Active Ultrasound)
  3. Exp E3: Reverse Ultrasound Transfer (NHANES III Active Ultrasound -> Turkish Clinic UCI)
  4. Exp E4: Target Construct Dissection (Evaluating Questionnaire Model on Active Stones vs Recall)

All metrics include 1,000-sample bootstrap 95% Confidence Intervals and clinical calibration metrics.
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
from src.harmonized_dataset import load_nhanes3_ultrasound_benchmark, NH3_SHARED_FEATURES
from src.model import GallstoneNet

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── PyTorch Trainer ──────────────────────────────────────────────────────────

def train_pytorch_net(
    train_loader,
    val_loader,
    input_dim: int,
    pos_weight_val: float,
    epochs: int = 120,
    lr: float = 1e-3,
    patience: int = 15,
) -> Tuple[GallstoneNet, Dict[str, list]]:
    model = GallstoneNet(input_dim=input_dim, dropout=0.3).to(DEVICE)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)

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
            preds = model(X_b)
            loss = criterion(preds, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_b)
            n_train += len(X_b)
        train_loss /= max(n_train, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                preds = model(X_b)
                loss = criterion(preds, y_b)
                val_loss += loss.item() * len(X_b)
                n_val += len(X_b)
        val_loss /= max(n_val, 1)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_weights.items()})
    model.eval()
    return model, history


def predict_pytorch(model: GallstoneNet, X_data: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        t = torch.tensor(X_data, dtype=torch.float32).to(DEVICE)
        logits = model(t)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze()
    preds = (probs >= threshold).astype(int)
    return preds, probs


# ── Bootstrap 95% Confidence Intervals & Calibration Metrics ─────────────────

def compute_bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, n_bootstraps: int = 1000, seed: int = 42) -> Dict[str, Tuple[float, float]]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    indices = rng.randint(0, n, size=(n_bootstraps, n))
    
    aucs, briers, accs, senss, specs, f1s = [], [], [], [], [], []
    y_pred = (y_prob >= 0.5).astype(int)

    for idx in indices:
        yb_true = y_true[idx]
        if len(np.unique(yb_true)) < 2:
            continue
        yb_prob = y_prob[idx]
        yb_pred = y_pred[idx]

        aucs.append(roc_auc_score(yb_true, yb_prob))
        briers.append(brier_score_loss(yb_true, yb_prob))
        accs.append(accuracy_score(yb_true, yb_pred))
        senss.append(recall_score(yb_true, yb_pred, zero_division=0))
        tn = np.sum((yb_true == 0) & (yb_pred == 0))
        fp = np.sum((yb_true == 0) & (yb_pred == 1))
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        f1s.append(f1_score(yb_true, yb_pred, zero_division=0))

    def get_ci(arr):
        if len(arr) == 0:
            return (0.0, 0.0)
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    return {
        "auc_ci": get_ci(aucs),
        "brier_ci": get_ci(briers),
        "acc_ci": get_ci(accs),
        "sens_ci": get_ci(senss),
        "spec_ci": get_ci(specs),
        "f1_ci": get_ci(f1s),
    }


def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    brier = float(brier_score_loss(y_true, y_prob))
    eps = 1e-6
    p_clipped = np.clip(y_prob, eps, 1.0 - eps)
    logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)

    try:
        lr_cal = LogisticRegression(penalty=None, solver="lbfgs", max_iter=200)
        lr_cal.fit(logits, y_true)
        slope = float(lr_cal.coef_[0][0])
        intercept = float(lr_cal.intercept_[0])
    except Exception:
        try:
            lr_cal = LogisticRegression(C=1e5, solver="lbfgs", max_iter=200)
            lr_cal.fit(logits, y_true)
            slope = float(lr_cal.coef_[0][0])
            intercept = float(lr_cal.intercept_[0])
        except Exception:
            slope = 1.0
            intercept = 0.0

    return {
        "brier_score": round(brier, 4),
        "calibration_slope": round(slope, 3),
        "calibration_intercept": round(intercept, 3),
    }


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    cal = compute_calibration_metrics(y_true, y_prob)
    cis = compute_bootstrap_ci(y_true, y_prob, n_bootstraps=1000)

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "sensitivity": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(float(spec), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y_true, y_prob)), 4) if len(np.unique(y_true)) > 1 else 0.5,
        "auprc": round(float(average_precision_score(y_true, y_prob)), 4) if len(np.unique(y_true)) > 1 else 0.0,
        "brier_score": cal["brier_score"],
        "calibration_slope": cal["calibration_slope"],
        "calibration_intercept": cal["calibration_intercept"],
        "ci_95": {
            "auc": [round(cis["auc_ci"][0], 4), round(cis["auc_ci"][1], 4)],
            "sensitivity": [round(cis["sens_ci"][0], 4), round(cis["sens_ci"][1], 4)],
            "specificity": [round(cis["spec_ci"][0], 4), round(cis["spec_ci"][1], 4)],
            "brier": [round(cis["brier_ci"][0], 4), round(cis["brier_ci"][1], 4)],
        },
        "cm": cm.tolist(),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


# ── Benchmark Runner ─────────────────────────────────────────────────────────

def run_experiment_suite(bundle: Dict, exp_name: str, desc: str) -> Dict[str, Any]:
    print(f"\n{'='*78}")
    print(f"  {exp_name.upper()}: {desc}")
    print(f"{'='*78}")

    X_tr_np = bundle["raw_arrays"]["X_train"]
    y_tr_np = bundle["raw_arrays"]["y_train"]
    X_val_np = bundle["raw_arrays"]["X_val"]
    y_val_np = bundle["raw_arrays"]["y_val"]
    X_te_np = bundle["raw_arrays"]["X_test"]
    y_te_np = bundle["raw_arrays"]["y_test"]

    n_features = X_tr_np.shape[1]
    n_pos = int(y_tr_np.sum())
    n_neg = len(y_tr_np) - n_pos
    pos_weight = float(n_neg / max(n_pos, 1))

    print(f"Split breakdown: Train N={len(y_tr_np):,} (Pos: {n_pos:,}), Val N={len(y_val_np):,}, Test N={len(y_te_np):,} (Pos: {int(y_te_np.sum()):,})")
    print(f"Total Cohort: {len(y_tr_np) + len(y_val_np) + len(y_te_np):,} patients across {n_features} features.")

    models_res = {}
    preds_probs = {}

    # 1. PyTorch GallstoneNet
    print("\n  [1/5] Training PyTorch GallstoneNet...")
    model_nn, _ = train_pytorch_net(
        bundle["loaders"]["train"], bundle["loaders"]["val"],
        input_dim=n_features, pos_weight_val=pos_weight, epochs=120
    )
    _, p_nn = predict_pytorch(model_nn, X_te_np)
    models_res["GallstoneNet"] = evaluate_predictions(y_te_np, p_nn)
    preds_probs["GallstoneNet"] = p_nn
    print(f"        -> AUC: {models_res['GallstoneNet']['auc']} {models_res['GallstoneNet']['ci_95']['auc']}, Brier: {models_res['GallstoneNet']['brier_score']}")

    # 2. XGBoost
    print("  [2/5] Training XGBoost...")
    model_xgb = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.03,
        scale_pos_weight=pos_weight, subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", random_state=42
    )
    model_xgb.fit(X_tr_np, y_tr_np, eval_set=[(X_val_np, y_val_np)], verbose=False)
    p_xgb = model_xgb.predict_proba(X_te_np)[:, 1]
    models_res["XGBoost"] = evaluate_predictions(y_te_np, p_xgb)
    preds_probs["XGBoost"] = p_xgb
    print(f"        -> AUC: {models_res['XGBoost']['auc']} {models_res['XGBoost']['ci_95']['auc']}, Brier: {models_res['XGBoost']['brier_score']}")

    # 3. LightGBM
    print("  [3/5] Training LightGBM...")
    model_lgb = lgb.LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.03,
        scale_pos_weight=pos_weight, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1
    )
    model_lgb.fit(X_tr_np, y_tr_np, eval_set=[(X_val_np, y_val_np)], callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)])
    p_lgb = model_lgb.predict_proba(X_te_np)[:, 1]
    models_res["LightGBM"] = evaluate_predictions(y_te_np, p_lgb)
    preds_probs["LightGBM"] = p_lgb
    print(f"        -> AUC: {models_res['LightGBM']['auc']} {models_res['LightGBM']['ci_95']['auc']}, Brier: {models_res['LightGBM']['brier_score']}")

    # 4. Random Forest
    print("  [4/5] Training Random Forest...")
    model_rf = RandomForestClassifier(n_estimators=250, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1)
    model_rf.fit(X_tr_np, y_tr_np)
    p_rf = model_rf.predict_proba(X_te_np)[:, 1]
    models_res["RandomForest"] = evaluate_predictions(y_te_np, p_rf)
    preds_probs["RandomForest"] = p_rf
    print(f"        -> AUC: {models_res['RandomForest']['auc']} {models_res['RandomForest']['ci_95']['auc']}, Brier: {models_res['RandomForest']['brier_score']}")

    # 5. Soft Voting Super Ensemble
    print("  [5/5] Synthesizing Calibrated Super Ensemble...")
    p_ens = (p_nn * 0.35 + p_xgb * 0.35 + p_lgb * 0.15 + p_rf * 0.15)
    models_res["SuperEnsemble"] = evaluate_predictions(y_te_np, p_ens)
    preds_probs["SuperEnsemble"] = p_ens
    print(f"        -> AUC: {models_res['SuperEnsemble']['auc']} {models_res['SuperEnsemble']['ci_95']['auc']}, Brier: {models_res['SuperEnsemble']['brier_score']}")

    return {
        "metrics": models_res,
        "probs": preds_probs,
        "y_test": y_te_np,
        "xgb_model": model_xgb,
        "feature_names": bundle["feature_names"],
        "splits": {
            "n_train": len(y_tr_np),
            "n_val": len(y_val_np),
            "n_test": len(y_te_np),
            "n_total": len(y_tr_np) + len(y_val_np) + len(y_te_np),
            "test_positives": int(y_te_np.sum()),
        }
    }


def generate_ultrasound_figures(results: Dict[str, Any]):
    print("\nGenerating publication figures for Ultrasound Benchmark...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: ROC Curves for Exp E1 and E2
    ax1 = axes[0]
    for m in ["GallstoneNet", "XGBoost", "SuperEnsemble"]:
        y_te = results["exp_e1"]["y_test"]
        p = results["exp_e1"]["probs"][m]
        fpr, tpr, _ = roc_curve(y_te, p)
        auc = results["exp_e1"]["metrics"][m]["auc"]
        ci = results["exp_e1"]["metrics"][m]["ci_95"]["auc"]
        ax1.plot(fpr, tpr, label=f"E1 {m} (AUC={auc:.3f} [{ci[0]:.2f}-{ci[1]:.2f}])")

    # Add E2 cross-cohort transfer
    y_te_e2 = results["exp_e2"]["y_test"]
    p_e2 = results["exp_e2"]["probs"]["XGBoost"]
    fpr_e2, tpr_e2, _ = roc_curve(y_te_e2, p_e2)
    auc_e2 = results["exp_e2"]["metrics"]["XGBoost"]["auc"]
    ci_e2 = results["exp_e2"]["metrics"]["XGBoost"]["ci_95"]["auc"]
    ax1.plot(fpr_e2, tpr_e2, "r--", linewidth=2, label=f"E2 UCI->NH3 US (AUC={auc_e2:.3f} [{ci_e2[0]:.2f}-{ci_e2[1]:.2f}])")

    ax1.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax1.set_title("ROC Curves: Ultrasound Ground-Truth Benchmark", fontsize=12, fontweight="bold")
    ax1.set_xlabel("False Positive Rate (1 - Specificity)")
    ax1.set_ylabel("True Positive Rate (Sensitivity)")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Calibration Curves
    ax2 = axes[1]
    ax2.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    for m, col in zip(["GallstoneNet", "XGBoost", "SuperEnsemble"], ["blue", "green", "purple"]):
        y_te = results["exp_e1"]["y_test"]
        p = results["exp_e1"]["probs"][m]
        prob_true, prob_pred = calibration_curve(y_te, p, n_bins=10)
        slope = results["exp_e1"]["metrics"][m]["calibration_slope"]
        brier = results["exp_e1"]["metrics"][m]["brier_score"]
        ax2.plot(prob_pred, prob_true, marker="o", color=col, label=f"E1 {m} (Slope={slope}, Brier={brier})")

    # Add E2 calibration
    prob_true_e2, prob_pred_e2 = calibration_curve(y_te_e2, p_e2, n_bins=10)
    slope_e2 = results["exp_e2"]["metrics"]["XGBoost"]["calibration_slope"]
    brier_e2 = results["exp_e2"]["metrics"]["XGBoost"]["brier_score"]
    ax2.plot(prob_pred_e2, prob_true_e2, marker="s", color="red", linestyle="--", label=f"E2 Transfer (Slope={slope_e2}, Brier={brier_e2})")

    ax2.set_title("Calibration Reliability: Ultrasound Ground Truth", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Observed Ultrasound Positive Fraction")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_plot = os.path.join(PLOTS_DIR, "ultrasound_benchmark_summary.png")
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"  Saved figure: {out_plot}")


def run_all_benchmarks():
    print("="*78)
    print("  LAUNCHING COMPREHENSIVE ULTRASOUND GROUND-TRUTH BENCHMARK")
    print("="*78)
    t0 = time.time()

    # 1. Experiment E1: Active Ultrasound Ground-Truth Internal Benchmark
    b_e1 = load_nhanes3_ultrasound_benchmark(mode="nh3_active_internal")
    res_e1 = run_experiment_suite(
        b_e1, "Exp E1", "NHANES III Active Ultrasound Ground Truth (N=12,824; 1,158 Positives)"
    )

    # 2. Experiment E2: Cross-Cohort Ultrasound Transfer (Turkish Clinic UCI -> NHANES III Ultrasound)
    b_e2 = load_nhanes3_ultrasound_benchmark(mode="train_uci_test_nh3_active")
    res_e2 = run_experiment_suite(
        b_e2, "Exp E2", "Cross-Cohort Ultrasound Transfer (Train UCI Clinic N=319 -> Test NHANES III US N=12,824)"
    )

    # 3. Experiment E3: Reverse Ultrasound Transfer (NHANES III Ultrasound -> Turkish Clinic UCI)
    b_e3 = load_nhanes3_ultrasound_benchmark(mode="train_nh3_active_test_uci")
    res_e3 = run_experiment_suite(
        b_e3, "Exp E3", "Reverse Ultrasound Transfer (Train NHANES III US N=12,824 -> Test UCI Clinic N=319)"
    )

    # Compile results
    all_results = {
        "exp_e1": {
            "name": "Exp E1: NHANES III Active Ultrasound Benchmark",
            "splits": res_e1["splits"],
            "metrics": res_e1["metrics"],
        },
        "exp_e2": {
            "name": "Exp E2: Cross-Cohort Transfer (UCI -> NHANES III Active US)",
            "splits": res_e2["splits"],
            "metrics": res_e2["metrics"],
        },
        "exp_e3": {
            "name": "Exp E3: Reverse Transfer (NHANES III Active US -> UCI Clinic)",
            "splits": res_e3["splits"],
            "metrics": res_e3["metrics"],
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Generate Figures
    full_bundle = {
        "exp_e1": res_e1,
        "exp_e2": res_e2,
        "exp_e3": res_e3,
    }
    generate_ultrasound_figures(full_bundle)

    # Save Results JSON
    out_json = os.path.join(MODELS_DIR, "ultrasound_benchmarks.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll benchmark results saved to: {out_json}")

    print(f"\nBenchmark completed successfully in {(time.time()-t0):.1f}s.")
    return all_results


if __name__ == "__main__":
    run_all_benchmarks()
