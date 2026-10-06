"""
dataset.py — PyTorch Dataset for the UCI Gallstone dataset.

Handles:
  • Loading the CSV
  • Encoding categorical features
  • Imputing missing values
  • Standardizing numerical features (fit on train only!)
  • Train/Val/Test splitting
  • Class-weight computation for imbalanced labels
"""

import os
import sys
import numpy as np
import pandas as pd

# Fix Windows console encoding for emoji/unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Column definitions ──────────────────────────────────────────────────────────

# Target
TARGET = "Gallstone Status"

# Binary / categorical columns (already 0/1 in the CSV)
BINARY_COLS = [
    "Gender",
    "Comorbidity",
    "Coronary Artery Disease (CAD)",
    "Hypothyroidism",
    "Hyperlipidemia",
    "Diabetes Mellitus (DM)",
]

# Continuous columns
CONTINUOUS_COLS = [
    "Age",
    "Height",
    "Weight",
    "Body Mass Index (BMI)",
    "Total Body Water (TBW)",
    "Extracellular Water (ECW)",
    "Intracellular Water (ICW)",
    "Extracellular Fluid/Total Body Water (ECF/TBW)",
    "Total Body Fat Ratio (TBFR) (%)",
    "Lean Mass (LM) (%)",
    "Body Protein Content (Protein) (%)",
    "Visceral Fat Rating (VFR)",
    "Bone Mass (BM)",
    "Muscle Mass (MM)",
    "Obesity (%)",
    "Total Fat Content (TFC)",
    "Visceral Fat Area (VFA)",
    "Visceral Muscle Area (VMA) (Kg)",
    "Hepatic Fat Accumulation (HFA)",
    "Glucose",
    "Total Cholesterol (TC)",
    "Low Density Lipoprotein (LDL)",
    "High Density Lipoprotein (HDL)",
    "Triglyceride",
    "Aspartat Aminotransferaz (AST)",
    "Alanin Aminotransferaz (ALT)",
    "Alkaline Phosphatase (ALP)",
    "Creatinine",
    "Glomerular Filtration Rate (GFR)",
    "C-Reactive Protein (CRP)",
    "Hemoglobin (HGB)",
    "Vitamin D",
]

FEATURE_COLS = BINARY_COLS + CONTINUOUS_COLS


# ── Dataset class ────────────────────────────────────────────────────────────────

class GallstoneDataset(Dataset):
    """A PyTorch Dataset that wraps the gallstone feature matrix + labels."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Data loading pipeline ────────────────────────────────────────────────────────

def load_and_prepare_data(
    csv_path: str = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    batch_size: int = 32,
):
    """
    Full data pipeline: load CSV → clean → split → scale → DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    scaler : StandardScaler  (fitted on train only)
    class_weights : torch.Tensor  (for BCEWithLogitsLoss pos_weight)
    feature_names : list[str]
    """
    if csv_path is None:
        # Default path relative to project root
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "gallstone_.csv"
        )

    # ── 1. Load ──────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    print(f"📂 Loaded {len(df)} patients, {df.shape[1]} columns")

    # ── 2. Basic cleaning ────────────────────────────────────────────────────
    # Drop rows with missing target
    before = len(df)
    df = df.dropna(subset=[TARGET])
    if len(df) < before:
        print(f"⚠️  Dropped {before - len(df)} rows with missing target")

    # Print class distribution
    counts = df[TARGET].value_counts()
    print(f"📊 Class distribution:  0 (no stone): {counts.get(0, 0)}  |  1 (stone): {counts.get(1, 0)}")

    # ── 3. Extract features & labels ─────────────────────────────────────────
    # Only keep columns that actually exist in the dataset
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    available_continuous = [c for c in CONTINUOUS_COLS if c in df.columns]

    X = df[available_features].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)

    # ── 4. Train / Val / Test split ──────────────────────────────────────────
    # First split off test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Then split train/val from the remainder
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val, random_state=random_state, stratify=y_temp
    )

    print(f"✂️  Split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

    # ── 5. Impute missing values (fit on train ONLY) ─────────────────────────
    # Structurally leak-free: fit imputer on X_train only, apply to val and test
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    # ── 6. Standardize continuous features (fit on train ONLY) ───────────────
    # Figure out which column indices correspond to continuous features
    cont_indices = [available_features.index(c) for c in available_continuous]

    scaler = StandardScaler()
    X_train[:, cont_indices] = scaler.fit_transform(X_train[:, cont_indices])
    X_val[:, cont_indices] = scaler.transform(X_val[:, cont_indices])
    X_test[:, cont_indices] = scaler.transform(X_test[:, cont_indices])

    # Persist continuous indices and imputer median statistics onto scaler
    scaler.continuous_indices_ = cont_indices
    scaler.imputer_statistics_ = imputer.statistics_.tolist()

    # ── 7. Compute class weights for imbalanced data ─────────────────────────
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    print(f"⚖️  Class weight (pos_weight): {pos_weight.item():.2f}")

    # ── 8. Create DataLoaders ────────────────────────────────────────────────
    train_ds = GallstoneDataset(X_train, y_train)
    val_ds = GallstoneDataset(X_val, y_val)
    test_ds = GallstoneDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler, pos_weight, available_features
