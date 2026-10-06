"""
harmonized_dataset.py — Multi-dataset harmonisation and loading pipeline.

Provides:
  1. `load_harmonized_data`:
     Harmonizes features present in BOTH UCI Gallstone and CDC NHANES:
       • Demographics: Age, Gender
       • Anthropometrics: Height, Weight, Body Mass Index (BMI)
       • Comorbidities: Coronary Artery Disease (CAD), Hypothyroidism,
                        Hyperlipidemia, Diabetes Mellitus (DM), Comorbidity
       • Clinical Labs: Glucose, Total Cholesterol (TC), HDL, LDL, Triglyceride,
                        AST, ALT, ALP, Creatinine, Hemoglobin (HGB)
     Modes supported:
       • 'joint': Pools UCI + NHANES, stratified train/val/test split.
       • 'train_nhanes_test_uci': External clinical validation (train on NHANES, test on UCI).
       • 'train_uci_test_nhanes': External population validation (train on UCI, test on NHANES).

  2. `load_nhanes_full_data`:
     Loads all 28 NHANES features (including Abdominal Pain RUQ symptom, Bilirubin, Albumin,
     Uric Acid, HbA1c, WBC, Platelets, Waist Circumference).
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
UCI_PATH = os.path.join(DATA_DIR, "gallstone_.csv")
NHANES_PATH = os.path.join(DATA_DIR, "nhanes_gallstone.csv")
NH3_PATH = os.path.join(DATA_DIR, "nhanes3_ultrasound.csv")

TARGET_COL = "Gallstone Status"

# ── Shared Harmonized Feature Set (20 features) ──────────────────────────────
SHARED_BINARY_COLS = [
    "Gender",
    "Comorbidity",
    "Coronary Artery Disease (CAD)",
    "Hypothyroidism",
    "Hyperlipidemia",
    "Diabetes Mellitus (DM)",
]

SHARED_CONTINUOUS_COLS = [
    "Age",
    "Height",
    "Weight",
    "Body Mass Index (BMI)",
    "Glucose",
    "Total Cholesterol (TC)",
    "High Density Lipoprotein (HDL)",
    "Low Density Lipoprotein (LDL)",
    "Triglyceride",
    "Aspartat Aminotransferaz (AST)",
    "Alanin Aminotransferaz (ALT)",
    "Alkaline Phosphatase (ALP)",
    "Creatinine",
    "Hemoglobin (HGB)",
]

SHARED_FEATURES = SHARED_BINARY_COLS + SHARED_CONTINUOUS_COLS


# ── Full NHANES Feature Set (28 features) ───────────────────────────────────
NHANES_SPECIFIC_BINARY = [
    "Abdominal Pain RUQ",
]

NHANES_SPECIFIC_CONTINUOUS = [
    "Waist Circumference",
    "Total Bilirubin",
    "Albumin",
    "Uric Acid",
    "HbA1c",
    "White Blood Cells (WBC)",
    "Platelets (PLT)",
]

NHANES_FULL_FEATURES = (
    SHARED_BINARY_COLS
    + NHANES_SPECIFIC_BINARY
    + SHARED_CONTINUOUS_COLS
    + NHANES_SPECIFIC_CONTINUOUS
)


class TabularDataset(Dataset):
    """Generic PyTorch Dataset for tabular medical features."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def preprocess_splits(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 64,
) -> Dict:
    """Imputes missing values and standardizes features (fit on train only)."""
    # 1. Median Imputer
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    # 2. Standard Scaler
    scaler = StandardScaler()
    X_train_scl = scaler.fit_transform(X_train_imp)
    X_val_scl = scaler.transform(X_val_imp)
    X_test_scl = scaler.transform(X_test_imp)

    # 3. Class weight (for imbalanced BCE loss)
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    pos_weight = float(n_neg / max(n_pos, 1))

    # 4. PyTorch DataLoaders
    train_ds = TabularDataset(X_train_scl, y_train)
    val_ds = TabularDataset(X_val_scl, y_val)
    test_ds = TabularDataset(X_test_scl, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        "loaders": {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        },
        "raw_arrays": {
            "X_train": X_train_scl,
            "y_train": y_train,
            "X_val": X_val_scl,
            "y_val": y_val,
            "X_test": X_test_scl,
            "y_test": y_test,
        },
        "imputer": imputer,
        "scaler": scaler,
        "pos_weight": pos_weight,
        "input_dim": X_train.shape[1],
    }


def load_harmonized_data(
    mode: str = "joint",
    test_size: float = 0.15,
    val_size: float = 0.15,
    batch_size: int = 64,
    random_state: int = 42,
) -> Dict:
    """
    Loads harmonized data across UCI and NHANES.
    Modes:
      'joint'                  : Combined UCI + NHANES, stratified split.
      'train_nhanes_test_uci'  : Train on NHANES, test on entire UCI dataset.
      'train_uci_test_nhanes'  : Train on UCI, test on entire NHANES dataset.
    """
    uci_df = pd.read_csv(UCI_PATH)
    nhanes_df = pd.read_csv(NHANES_PATH)

    uci_sub = uci_df[[TARGET_COL] + SHARED_FEATURES].copy()
    uci_sub["Cohort"] = "UCI"

    nhanes_sub = nhanes_df[[TARGET_COL] + SHARED_FEATURES].copy()
    nhanes_sub["Cohort"] = "NHANES"

    if mode == "joint":
        combined = pd.concat([uci_sub, nhanes_sub], ignore_index=True)
        X = combined[SHARED_FEATURES].values
        y = combined[TARGET_COL].values.astype(int)

        # Train / Temp split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), random_state=random_state, stratify=y
        )
        # Val / Test split
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1.0 - val_ratio), random_state=random_state, stratify=y_temp
        )

    elif mode == "train_nhanes_test_uci":
        X_nhanes = nhanes_sub[SHARED_FEATURES].values
        y_nhanes = nhanes_sub[TARGET_COL].values.astype(int)
        
        # Split NHANES into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_nhanes, y_nhanes, test_size=0.15, random_state=random_state, stratify=y_nhanes
        )
        # Test on UCI
        X_test = uci_sub[SHARED_FEATURES].values
        y_test = uci_sub[TARGET_COL].values.astype(int)

    elif mode == "train_uci_test_nhanes":
        X_uci = uci_sub[SHARED_FEATURES].values
        y_uci = uci_sub[TARGET_COL].values.astype(int)

        # Split UCI into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_uci, y_uci, test_size=0.15, random_state=random_state, stratify=y_uci
        )
        # Test on NHANES
        X_test = nhanes_sub[SHARED_FEATURES].values
        y_test = nhanes_sub[TARGET_COL].values.astype(int)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'joint', 'train_nhanes_test_uci', or 'train_uci_test_nhanes'.")

    data_bundle = preprocess_splits(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size
    )
    data_bundle["feature_names"] = SHARED_FEATURES
    data_bundle["mode"] = mode
    return data_bundle


def load_nhanes_full_data(
    test_size: float = 0.15,
    val_size: float = 0.15,
    batch_size: int = 64,
    random_state: int = 42,
) -> Dict:
    """Loads NHANES with all 28 features including abdominal pain symptom."""
    nhanes_df = pd.read_csv(NHANES_PATH)
    X = nhanes_df[NHANES_FULL_FEATURES].values
    y = nhanes_df[TARGET_COL].values.astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=random_state, stratify=y
    )
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1.0 - val_ratio), random_state=random_state, stratify=y_temp
    )

    data_bundle = preprocess_splits(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size
    )
    data_bundle["feature_names"] = NHANES_FULL_FEATURES
    data_bundle["mode"] = "nhanes_full"
    return data_bundle


# ── NHANES III Ultrasound & Shared Clinical Feature Set ─────────────────────────
NH3_SHARED_FEATURES = [
    "Age", "Gender", "Height", "Weight", "BMI",
    "Glucose", "Total Cholesterol", "LDL", "HDL", "Triglyceride",
    "AST", "ALT", "ALP", "Creatinine", "CRP"
]

UCI_15_MATCHING_COLS = [
    "Age", "Gender", "Height", "Weight", "Body Mass Index (BMI)",
    "Glucose", "Total Cholesterol (TC)", "Low Density Lipoprotein (LDL)",
    "High Density Lipoprotein (HDL)", "Triglyceride",
    "Aspartat Aminotransferaz (AST)", "Alanin Aminotransferaz (ALT)",
    "Alkaline Phosphatase (ALP)", "Creatinine", "C-Reactive Protein (CRP)"
]


def load_nhanes3_ultrasound_benchmark(
    mode: str = "nh3_active_internal",
    test_size: float = 0.15,
    val_size: float = 0.15,
    batch_size: int = 64,
    random_state: int = 42,
) -> Dict:
    """
    Loads the NHANES III Ultrasound Ground-Truth Cohort for benchmarking.
    
    Modes:
      • 'nh3_active_internal': Stratified 70/15/15 split on Active Ultrasound Gallstones (1,158 pos vs 11,666 neg; N=12,824).
      • 'nh3_total_internal': Stratified 70/15/15 split on Total Gallstone Disease (Active Stones OR Cholecystectomy: 2,028 pos vs 11,666 neg; N=13,694).
      • 'train_uci_test_nh3_active': Train on Turkish Clinic (UCI, N=319), test on NHANES III Active Ultrasound (N=12,824).
      • 'train_nh3_active_test_uci': Train on NHANES III Active Ultrasound (N=12,824), test on Turkish Clinic (UCI, N=319).
    """
    nh3_df = pd.read_csv(NH3_PATH)
    uci_df = pd.read_csv(UCI_PATH)

    if mode == "nh3_active_internal":
        sub_df = nh3_df[nh3_df["target_cholecystectomy_us"] == 0].copy()
        X = sub_df[NH3_SHARED_FEATURES].values
        y = sub_df["target_active_us"].values.astype(int)

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), random_state=random_state, stratify=y
        )
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1.0 - val_ratio), random_state=random_state, stratify=y_temp
        )

    elif mode == "nh3_total_internal":
        X = nh3_df[NH3_SHARED_FEATURES].values
        y = nh3_df["target_total_us"].values.astype(int)

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), random_state=random_state, stratify=y
        )
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1.0 - val_ratio), random_state=random_state, stratify=y_temp
        )

    elif mode == "train_uci_test_nh3_active":
        X_uci = uci_df[UCI_15_MATCHING_COLS].values
        y_uci = uci_df[TARGET_COL].values.astype(int)

        X_train, X_val, y_train, y_val = train_test_split(
            X_uci, y_uci, test_size=0.15, random_state=random_state, stratify=y_uci
        )
        sub_nh3 = nh3_df[nh3_df["target_cholecystectomy_us"] == 0].copy()
        X_test = sub_nh3[NH3_SHARED_FEATURES].values
        y_test = sub_nh3["target_active_us"].values.astype(int)

    elif mode == "train_nh3_active_test_uci":
        sub_nh3 = nh3_df[nh3_df["target_cholecystectomy_us"] == 0].copy()
        X_nh3 = sub_nh3[NH3_SHARED_FEATURES].values
        y_nh3 = sub_nh3["target_active_us"].values.astype(int)

        X_train, X_val, y_train, y_val = train_test_split(
            X_nh3, y_nh3, test_size=0.15, random_state=random_state, stratify=y_nh3
        )
        X_test = uci_df[UCI_15_MATCHING_COLS].values
        y_test = uci_df[TARGET_COL].values.astype(int)

    else:
        raise ValueError(f"Unknown mode '{mode}' for NHANES III benchmark.")

    data_bundle = preprocess_splits(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size
    )
    data_bundle["feature_names"] = NH3_SHARED_FEATURES
    data_bundle["mode"] = mode
    return data_bundle

