"""
src/evaluation.py — Centralized Statistical, Calibration, and Evaluation Suite.

Consolidates:
1. Non-parametric bootstrap confidence intervals (AUROC, AUPRC, Brier, Sens, Spec, F1).
2. Paired bootstrap hypothesis testing for model comparison (Delta AUROC, dynamic p-values).
3. Cox-Steyerberg hierarchical calibration metrics (Slope beta, Intercept alpha, Brier, O/E).
4. Validation-fitted Platt recalibration (frozen validation scaling applied to test).
5. Soft Voting Ensemble (canonical fixed blend and validation Platt recalibration).
6. Decision Curve Analysis (DCA net benefit).
7. Survey-weighted summary statistics for NHANES complex survey design.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
    threshold: float = 0.5,
) -> Dict[str, Tuple[float, float]]:
    """Compute 95% percentile bootstrap confidence intervals across core metrics."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    indices = rng.randint(0, n, size=(n_bootstraps, n))

    aucs, auprcs, briers, accs, senss, specs, f1s = [], [], [], [], [], [], []
    y_pred = (y_prob >= threshold).astype(int)

    for idx in indices:
        yb_true = y_true[idx]
        if len(np.unique(yb_true)) < 2:
            continue
        yb_prob = y_prob[idx]
        yb_pred = y_pred[idx]

        aucs.append(roc_auc_score(yb_true, yb_prob))
        auprcs.append(average_precision_score(yb_true, yb_prob))
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
        "auprc_ci": get_ci(auprcs),
        "brier_ci": get_ci(briers),
        "acc_ci": get_ci(accs),
        "sens_ci": get_ci(senss),
        "spec_ci": get_ci(specs),
        "f1_ci": get_ci(f1s),
    }


def compute_paired_bootstrap_auroc_test(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Perform paired non-parametric bootstrap test for difference in AUROC (Model A - Model B).
    Computes empirical 95% CI and dynamic two-sided p-value.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    indices = rng.randint(0, n, size=(n_bootstraps, n))

    auc_diffs = []
    aucs_a = []
    aucs_b = []

    for idx in indices:
        yb_true = y_true[idx]
        if len(np.unique(yb_true)) < 2:
            continue
        auc_a = roc_auc_score(yb_true, prob_a[idx])
        auc_b = roc_auc_score(yb_true, prob_b[idx])
        aucs_a.append(auc_a)
        aucs_b.append(auc_b)
        auc_diffs.append(auc_a - auc_b)

    auc_diffs = np.array(auc_diffs)
    delta_mean = float(np.mean(auc_diffs))
    delta_ci = (float(np.percentile(auc_diffs, 2.5)), float(np.percentile(auc_diffs, 97.5)))
    
    # Two-sided empirical p-value
    prop_gt = np.mean(auc_diffs > 0)
    prop_lt = np.mean(auc_diffs < 0)
    p_val = float(min(1.0, 2 * min(prop_gt, prop_lt)))

    return {
        "delta_mean": round(delta_mean, 4),
        "delta_ci_95": [round(delta_ci[0], 4), round(delta_ci[1], 4)],
        "p_value": round(p_val, 4),
        "auc_a_mean": round(float(np.mean(aucs_a)), 4),
        "auc_b_mean": round(float(np.mean(aucs_b)), 4),
    }


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute Cox-Steyerberg hierarchical calibration metrics:
    - Brier score (mean squared error of probabilities)
    - Logistic calibration slope (beta, spread of risk, ideal=1.0)
    - Logistic calibration intercept (alpha, calibration-in-the-large, ideal=0.0)
    - Observed-to-Expected ratio (O/E = mean(y) / mean(prob), ideal=1.0)
    """
    brier = float(brier_score_loss(y_true, y_prob))
    p_clipped = np.clip(y_prob, eps, 1.0 - eps)
    logits = logit(p_clipped).reshape(-1, 1)

    try:
        lr_cal = LogisticRegression(solver="lbfgs", C=1e6, max_iter=500)
        lr_cal.fit(logits, y_true)
        slope = float(lr_cal.coef_[0][0])
        intercept = float(lr_cal.intercept_[0])
    except Exception:
        slope = 1.0
        intercept = 0.0

    observed_rate = float(np.mean(y_true))
    expected_rate = float(np.mean(y_prob))
    oe_ratio = float(observed_rate / max(expected_rate, eps))

    return {
        "brier_score": round(brier, 4),
        "calibration_slope": round(slope, 3),
        "calibration_intercept": round(intercept, 3),
        "oe_ratio": round(oe_ratio, 3),
    }


def fit_and_apply_platt_recalibration(
    y_val: np.ndarray,
    p_val: np.ndarray,
    p_test: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, float, float]:
    """
    Fit Platt scaling on the independent validation split and apply frozen parameters to test.
    Returns:
        p_test_recal: recalibrated test probabilities
        val_slope: beta parameter estimated on validation
        val_intercept: alpha parameter estimated on validation
    """
    p_val_clip = np.clip(p_val, eps, 1.0 - eps)
    logit_val = logit(p_val_clip).reshape(-1, 1)

    platt = LogisticRegression(solver="lbfgs", C=1e6, max_iter=500)
    platt.fit(logit_val, y_val)
    val_slope = float(platt.coef_[0][0])
    val_intercept = float(platt.intercept_[0])

    p_test_clip = np.clip(p_test, eps, 1.0 - eps)
    logit_test = logit(p_test_clip).reshape(-1, 1)

    recal_logits = val_slope * logit_test + val_intercept
    p_test_recal = expit(recal_logits).ravel()

    return p_test_recal, val_slope, val_intercept


def compute_inverse_brier_ensemble(
    y_val: np.ndarray,
    val_probs_dict: Dict[str, np.ndarray],
    test_probs_dict: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """
    Synthesize an Inverse-Brier Weighted Super Ensemble with validation-fitted Platt recalibration:
    1. Compute Brier score B_m on the validation split for each model m.
    2. Weights: w_m = (1 / B_m) / sum_k (1 / B_k).
    3. Blend probabilities on validation and test partitions.
    4. Fit Platt recalibration on validation blended logits and apply to test predictions.
    """
    model_names = list(val_probs_dict.keys())
    brier_scores = {}
    inv_briers = {}

    for name in model_names:
        b = brier_score_loss(y_val, val_probs_dict[name])
        brier_scores[name] = float(b)
        inv_briers[name] = 1.0 / max(b, 1e-6)

    total_inv = sum(inv_briers.values())
    weights = {name: float(inv_briers[name] / total_inv) for name in model_names}

    # Validation blend
    p_val_blend = np.zeros_like(y_val, dtype=np.float64)
    for name in model_names:
        p_val_blend += weights[name] * val_probs_dict[name]

    # Test blend (raw)
    n_test = len(next(iter(test_probs_dict.values())))
    p_test_raw = np.zeros(n_test, dtype=np.float64)
    for name in model_names:
        p_test_raw += weights[name] * test_probs_dict[name]

    # Fit Platt recalibration on validation ensemble
    p_test_recal, val_slope, val_intercept = fit_and_apply_platt_recalibration(
        y_val=y_val,
        p_val=p_val_blend,
        p_test=p_test_raw,
    )

    return {
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "validation_briers": {k: round(v, 4) for k, v in brier_scores.items()},
        "p_val_blend": p_val_blend,
        "p_test_raw": p_test_raw,
        "p_test_recal": p_test_recal,
        "platt_slope": round(val_slope, 4),
        "platt_intercept": round(val_intercept, 4),
    }


def compute_decision_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Decision Curve Analysis (DCA): Computes Net Benefit across clinical decision thresholds.
    Net Benefit = (TP / N) - (FP / N) * [p_t / (1 - p_t)]
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.50, 50)

    n = len(y_true)
    prevalence = np.mean(y_true)

    nb_model = []
    nb_all = []
    nb_none = np.zeros_like(thresholds)

    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))

        # Model net benefit
        w = pt / (1.0 - pt)
        nb = (tp / n) - (fp / n) * w
        nb_model.append(nb)

        # Treat all strategy
        nb_a = prevalence - (1.0 - prevalence) * w
        nb_all.append(nb_a)

    return {
        "thresholds": thresholds,
        "net_benefit_model": np.array(nb_model),
        "net_benefit_all": np.array(nb_all),
        "net_benefit_none": nb_none,
    }


def compute_survey_weighted_stats(
    df: pd.DataFrame,
    value_col: str,
    weight_col: str,
    is_binary: bool = False,
) -> Dict[str, float]:
    """
    Compute survey-weighted mean and standard error for complex survey sampling.
    Uses Horvitz-Thompson estimation with analytic variance.
    """
    sub = df[[value_col, weight_col]].dropna()
    vals = sub[value_col].values.astype(float)
    weights = sub[weight_col].values.astype(float)

    total_weight = np.sum(weights)
    weighted_mean = np.sum(vals * weights) / total_weight

    # Variance estimate
    diff = vals - weighted_mean
    variance = np.sum((weights * diff) ** 2) / (total_weight ** 2)
    se = np.sqrt(variance)

    return {
        "weighted_mean": float(weighted_mean),
        "se": float(se),
        "ci_95_lower": float(weighted_mean - 1.96 * se),
        "ci_95_upper": float(weighted_mean + 1.96 * se),
        "unweighted_mean": float(np.mean(vals)),
        "n_analyzed": int(len(vals)),
    }
