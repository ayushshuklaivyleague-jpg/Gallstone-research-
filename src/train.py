"""
train.py — Full training pipeline for the Gallstone Prediction AI.

Run with:
    python -m src.train

What it does:
  1. Loads and preprocesses data (train/val/test split)
  2. Trains a neural network with early stopping
  3. Evaluates on the test set (Accuracy, Precision, Recall, F1, AUC-ROC)
  4. Generates plots: loss curves, confusion matrix, ROC curve, feature importance
  5. Saves the trained model to models/
"""

import os
import sys
import time
import json
import numpy as np

# Fix Windows console encoding for emoji/unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.dataset import load_and_prepare_data
from src.model import GallstoneNet


# ── Config ───────────────────────────────────────────────────────────────────────

class Config:
    """All hyperparameters in one place — easy to tweak!"""

    # Data
    batch_size: int = 32
    test_size: float = 0.15
    val_size: float = 0.15

    # Model
    dropout: float = 0.3

    # Training
    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4          # L2 regularisation
    patience: int = 25                  # early stopping patience
    min_delta: float = 0.001            # minimum improvement to count

    # Paths
    model_dir: str = "models"
    plot_dir: str = "plots"


# ── Training loop ────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on a dataset. Returns loss, predictions, probabilities, labels."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_probs = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())

        total_loss += loss.item()
        n_batches += 1

    all_probs = np.concatenate(all_probs).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    all_preds = (all_probs >= 0.5).astype(int)

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, all_preds, all_probs, all_labels


# ── Plotting ─────────────────────────────────────────────────────────────────────

def plot_loss_curves(train_losses, val_losses, save_path):
    """Plot training & validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train Loss", linewidth=2, color="#2196F3")
    ax.plot(val_losses, label="Val Loss", linewidth=2, color="#FF5722")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (BCE)", fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"   📈 Loss curves saved → {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot a heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Stone", "Stone"],
        yticklabels=["No Stone", "Stone"],
        ax=ax,
        annot_kws={"size": 16},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"   🔲 Confusion matrix saved → {save_path}")


def plot_roc_curve(y_true, y_probs, save_path):
    """Plot the ROC curve with AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, linewidth=2.5, color="#4CAF50", label=f"Model (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC = 0.500)")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#4CAF50")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"   📉 ROC curve saved → {save_path}")


def plot_feature_importance(model, feature_names, save_path):
    """
    Approximate feature importance using the absolute weights
    of the first linear layer.
    """
    # Get the first Linear layer's weights
    first_linear = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            first_linear = module
            break

    if first_linear is None:
        return

    weights = first_linear.weight.detach().cpu().numpy()
    importance = np.abs(weights).mean(axis=0)  # average across output neurons

    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1][:20]  # top 20

    fig, ax = plt.subplots(figsize=(10, 8))
    names = [feature_names[i] for i in sorted_idx]
    values = importance[sorted_idx]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    bars = ax.barh(range(len(names)), values[::-1], color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=10)
    ax.set_xlabel("Mean |Weight|", fontsize=12)
    ax.set_title("Top 20 Feature Importance (First Layer Weights)", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"   🏆 Feature importance saved → {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    cfg = Config()

    print("=" * 60)
    print("🩺 GALLSTONE PREDICTION AI — Training Pipeline")
    print("=" * 60)
    print()

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ── Data ─────────────────────────────────────────────────────────────────
    print("── Loading & Preprocessing Data ──")
    train_loader, val_loader, test_loader, scaler, pos_weight, feature_names = (
        load_and_prepare_data(
            batch_size=cfg.batch_size,
            test_size=cfg.test_size,
            val_size=cfg.val_size,
        )
    )
    input_dim = len(feature_names)
    print(f"🔢 Input features: {input_dim}")
    print()

    # ── Model ────────────────────────────────────────────────────────────────
    model = GallstoneNet(input_dim=input_dim, dropout=cfg.dropout).to(device)
    print("── Model Architecture ──")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print()

    # ── Loss & Optimizer ─────────────────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # ── Training ─────────────────────────────────────────────────────────────
    print("── Training ──")
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.plot_dir, exist_ok=True)
    best_model_path = os.path.join(cfg.model_dir, "best_model.pt")

    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_preds, val_probs, val_labels = evaluate(
            model, val_loader, criterion, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Logging
        if epoch % 10 == 0 or epoch == 1:
            val_acc = accuracy_score(val_labels, val_preds)
            print(
                f"   Epoch {epoch:>3d}/{cfg.epochs} │ "
                f"Train Loss: {train_loss:.4f} │ "
                f"Val Loss: {val_loss:.4f} │ "
                f"Val Acc: {val_acc:.3f}"
            )

        # Early stopping
        if val_loss < best_val_loss - cfg.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                    "continuous_indices": getattr(scaler, "continuous_indices_", None),
                    "imputer_statistics": getattr(scaler, "imputer_statistics_", None),
                    "feature_names": feature_names,
                    "input_dim": input_dim,
                },
                best_model_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"\n   ⏹️  Early stopping at epoch {epoch} (best was epoch {best_epoch})")
                break

    elapsed = time.time() - start_time
    print(f"\n   ⏱️  Training completed in {elapsed:.1f}s")
    print(f"   💾 Best model saved → {best_model_path} (epoch {best_epoch}, val_loss={best_val_loss:.4f})")
    print()

    # ── Load best model for final evaluation ─────────────────────────────────
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    # ── Test evaluation ──────────────────────────────────────────────────────
    print("── Test Set Evaluation ──")
    test_loss, test_preds, test_probs, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    acc = accuracy_score(test_labels, test_preds)
    prec = precision_score(test_labels, test_preds, zero_division=0)
    rec = recall_score(test_labels, test_preds, zero_division=0)
    f1 = f1_score(test_labels, test_preds, zero_division=0)
    auc = roc_auc_score(test_labels, test_probs) if len(np.unique(test_labels)) > 1 else 0.0

    print(f"   Test Loss:      {test_loss:.4f}")
    print(f"   Accuracy:       {acc:.4f}  ({acc*100:.1f}%)")
    print(f"   Precision:      {prec:.4f}")
    print(f"   Recall:         {rec:.4f}")
    print(f"   F1 Score:       {f1:.4f}")
    print(f"   AUC-ROC:        {auc:.4f}")
    print()

    print("   Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=["No Stone", "Stone"]))

    # ── Generate Plots ───────────────────────────────────────────────────────
    print("── Generating Plots ──")
    plot_loss_curves(
        train_losses, val_losses, os.path.join(cfg.plot_dir, "loss_curves.png")
    )
    plot_confusion_matrix(
        test_labels, test_preds, os.path.join(cfg.plot_dir, "confusion_matrix.png")
    )
    plot_roc_curve(
        test_labels, test_probs, os.path.join(cfg.plot_dir, "roc_curve.png")
    )
    plot_feature_importance(
        model, feature_names, os.path.join(cfg.plot_dir, "feature_importance.png")
    )

    # ── Save results summary ─────────────────────────────────────────────────
    results = {
        "test_accuracy": round(acc, 4),
        "test_precision": round(prec, 4),
        "test_recall": round(rec, 4),
        "test_f1": round(f1, 4),
        "test_auc_roc": round(auc, 4),
        "best_epoch": best_epoch,
        "total_epochs_trained": len(train_losses),
        "training_time_seconds": round(elapsed, 1),
    }
    results_path = os.path.join(cfg.model_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n   📄 Results summary saved → {results_path}")

    print()
    print("=" * 60)
    print("✅ TRAINING COMPLETE!")
    print(f"   Model:   {best_model_path}")
    print(f"   Plots:   {cfg.plot_dir}/")
    print(f"   Results: {results_path}")
    print()
    print("   Next step: python -m src.predict")
    print("=" * 60)


if __name__ == "__main__":
    main()
