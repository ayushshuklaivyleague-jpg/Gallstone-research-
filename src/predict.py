"""
predict.py — Multi-Cohort Gallstone Risk Prediction Engine.

Supports 3 Clinical Prediction Modes:
  1. Clinic Bioimpedance Profile (UCI Model, 38 features with Body Composition & Labs)
  2. Population Screening & Symptoms (NHANES Model, 28 features with RUQ Pain & Labs)
  3. Universal Harmonized Blood Work (Joint Model, 20 common clinical features)

Usage:
  # Run built-in clinical case studies:
  python -m src.predict --model uci
  python -m src.predict --model nhanes
  python -m src.predict --model joint

  # Run interactive patient triage:
  python -m src.predict --interactive --model nhanes
"""

import os
import sys
import argparse
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.model import GallstoneNet

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

MODEL_PRESETS = {
    "uci": os.path.join(MODELS_DIR, "exp_a_uci_clinic_gallstonenet.pt"),
    "nhanes": os.path.join(MODELS_DIR, "exp_b_nhanes_full_gallstonenet.pt"),
    "joint": os.path.join(MODELS_DIR, "exp_d_joint_harmonized_gallstonenet.pt"),
}


def load_model(preset: str = "uci", custom_path: str = None, device: str = "cpu"):
    """Loads a saved model checkpoint and reconstructs its scaler and imputer."""
    if custom_path:
        model_path = custom_path
    else:
        model_path = MODEL_PRESETS.get(preset, MODEL_PRESETS["uci"])
        if not os.path.exists(model_path):
            fallback = os.path.join(MODELS_DIR, "best_model.pt")
            if os.path.exists(fallback):
                model_path = fallback

    if not os.path.exists(model_path):
        print(f"❌ No model found at: {model_path}")
        print("   Please run 'python -m src.train_all' first!")
        sys.exit(1)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    input_dim = checkpoint["input_dim"]
    feature_names = checkpoint["feature_names"]

    model = GallstoneNet(input_dim=input_dim)
    state_key = "state_dict" if "state_dict" in checkpoint else "model_state_dict"
    model.load_state_dict(checkpoint[state_key])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = np.array(checkpoint["scaler_mean"])
    scaler.scale_ = np.array(checkpoint["scaler_scale"])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    imputer_stats = None
    if "imputer_statistics" in checkpoint:
        imputer_stats = np.array(checkpoint["imputer_statistics"], dtype=np.float32)

    return model, scaler, imputer_stats, feature_names, model_path


def predict_patient(model, scaler, imputer_stats, feature_names, patient_data: dict) -> dict:
    """Predict gallstone risk probability for a patient dictionary."""
    vec = np.full(len(feature_names), np.nan, dtype=np.float32)
    for i, name in enumerate(feature_names):
        if name in patient_data and patient_data[name] is not None:
            vec[i] = float(patient_data[name])

    # Impute missing values with dataset median statistics
    for i in range(len(vec)):
        if np.isnan(vec[i]):
            if imputer_stats is not None and not np.isnan(imputer_stats[i]):
                vec[i] = imputer_stats[i]
            else:
                vec[i] = 0.0

    # Scale features
    vec_scaled = vec.copy()
    if scaler is not None and hasattr(scaler, "n_features_in_"):
        if scaler.n_features_in_ == len(feature_names):
            vec_scaled = scaler.transform(vec.reshape(1, -1)).flatten()
        elif scaler.n_features_in_ < len(feature_names):
            diff = len(feature_names) - scaler.n_features_in_
            vec_scaled[diff:] = scaler.transform(vec[diff:].reshape(1, -1)).flatten()
    tensor_x = torch.tensor(vec_scaled.reshape(1, -1), dtype=torch.float32)

    with torch.no_grad():
        prob = model.predict_proba(tensor_x).item()

    if prob < 0.25:
        risk = "LOW RISK 🟢"
        action = "Routine check-up; gallstones unlikely."
    elif prob < 0.55:
        risk = "MODERATE RISK 🟡"
        action = "Evaluate biliary symptoms, lifestyle modifications, follow-up."
    elif prob < 0.75:
        risk = "HIGH RISK 🟠"
        action = "Abdominal ultrasound recommended to inspect gallbladder lumen."
    else:
        risk = "VERY HIGH RISK 🔴"
        action = "Urgent ultrasound & gastroenterology referral recommended."

    return {
        "probability": round(prob, 4),
        "prediction": "GALLSTONE DETECTED" if prob >= 0.5 else "NO GALLSTONE",
        "risk_level": risk,
        "action": action,
    }


def print_result(result):
    print()
    print("   ┌────────────────────────────────────────────────────────┐")
    print(f"   │  Prediction:    {result['prediction']:<38s} │")
    print(f"   │  Probability:   {result['probability']:.1%}                                   │")
    print(f"   │  Risk Tier:     {result['risk_level']:<38s} │")
    print(f"   │  Action:        {result['action']:<38s} │")
    print("   └────────────────────────────────────────────────────────┘")


NHANES_EXAMPLES = [
    {
        "name": "Patient 1: Symptomatic Female with RUQ Pain & Elevated BMI (High Suspicion)",
        "data": {
            "Gender": 1,  # Female
            "Age": 54,
            "Abdominal Pain RUQ": 1,  # Classic biliary colic symptom
            "Body Mass Index (BMI)": 34.2,
            "Waist Circumference": 102.0,
            "Glucose": 115,
            "Total Cholesterol (TC)": 245,
            "High Density Lipoprotein (HDL)": 38,
            "Low Density Lipoprotein (LDL)": 158,
            "Triglyceride": 210,
            "Aspartat Aminotransferaz (AST)": 38,
            "Alanin Aminotransferaz (ALT)": 44,
            "Alkaline Phosphatase (ALP)": 105,
            "Total Bilirubin": 1.4,
            "Hemoglobin (HGB)": 13.1,
            "Coronary Artery Disease (CAD)": 0,
            "Hypothyroidism": 1,
            "Hyperlipidemia": 1,
            "Diabetes Mellitus (DM)": 0,
        },
    },
    {
        "name": "Patient 2: Asymptomatic Healthy Young Adult (Low Suspicion)",
        "data": {
            "Gender": 0,  # Male
            "Age": 28,
            "Abdominal Pain RUQ": 0,
            "Body Mass Index (BMI)": 22.4,
            "Waist Circumference": 82.0,
            "Glucose": 88,
            "Total Cholesterol (TC)": 165,
            "High Density Lipoprotein (HDL)": 58,
            "Low Density Lipoprotein (LDL)": 92,
            "Triglyceride": 75,
            "Aspartat Aminotransferaz (AST)": 18,
            "Alanin Aminotransferaz (ALT)": 16,
            "Alkaline Phosphatase (ALP)": 62,
            "Total Bilirubin": 0.6,
            "Hemoglobin (HGB)": 15.6,
            "Coronary Artery Disease (CAD)": 0,
            "Hypothyroidism": 0,
            "Hyperlipidemia": 0,
            "Diabetes Mellitus (DM)": 0,
        },
    },
]


def run_interactive(model, scaler, imputer, feature_names):
    print("\n🩺 Enter Patient Data (press Enter to skip / impute missing):\n")
    data = {}
    for feat in feature_names:
        hint = ""
        if feat == "Gender":
            hint = " (0 = Male, 1 = Female)"
        elif feat in ["Abdominal Pain RUQ", "Comorbidity", "Coronary Artery Disease (CAD)", "Hypothyroidism", "Hyperlipidemia", "Diabetes Mellitus (DM)"]:
            hint = " (0 = No, 1 = Yes)"
        
        val_str = input(f"   {feat}{hint}: ").strip()
        if val_str:
            try:
                data[feat] = float(val_str)
            except ValueError:
                print(f"   ⚠️ Could not parse '{val_str}', skipping")
    result = predict_patient(model, scaler, imputer, feature_names, data)
    print_result(result)


def main():
    parser = argparse.ArgumentParser(description="Gallstone Risk Prediction Engine")
    parser.add_argument("--model", choices=["uci", "nhanes", "joint"], default="nhanes",
                        help="Model preset: 'uci' (clinic 38-feat), 'nhanes' (28-feat + RUQ pain), or 'joint' (harmonized 20-feat)")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive patient interview mode")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path")
    args = parser.parse_args()

    print("=" * 66)
    print(f"🩺 GALLSTONE AI PREDICTION SYSTEM [{args.model.upper()} PRESET]")
    print("=" * 66)

    model, scaler, imputer, feature_names, path = load_model(preset=args.model, custom_path=args.checkpoint)
    print(f"✅ Loaded: {os.path.basename(path)}")
    print(f"📊 Active Biomarkers: {len(feature_names)} features\n")

    if args.interactive:
        run_interactive(model, scaler, imputer, feature_names)
    else:
        print("── Clinical Case Demonstrations ──\n")
        for ex in NHANES_EXAMPLES:
            print(f"▶ {ex['name']}")
            res = predict_patient(model, scaler, imputer, feature_names, ex["data"])
            print_result(res)
            print()

        print("─" * 66)
        print("💡 To triage a custom patient interactively, run:")
        print(f"   python -m src.predict --interactive --model {args.model}")
        print("─" * 66)


if __name__ == "__main__":
    main()
