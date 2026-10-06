"""
model.py — Neural network architectures for gallstone prediction.

Canonical Architectures:
  1. `GallstoneNet` (Canonical Baseline / Deep Tabular MLP):
     Input(N features)
       → BatchNorm → Linear(128) → ReLU → Dropout(0.3)
       → Linear(64) → ReLU → Dropout(0.3)
       → Linear(32) → ReLU → Dropout(0.2)
       → Linear(1)   ← raw logit (use BCEWithLogitsLoss)

  2. `TabularResNet` (Deep Residual Variant):
     Input(N features)
       → Linear Projection + LayerNorm + ReLU
       → Dense Block with LayerNorm, Dropout, and Identity Skip-Connection
       → Linear(1)   ← raw logit
"""

import torch
import torch.nn as nn


class TabularResNet(nn.Module):
    """Deep residual tabular network with LayerNorm and identity skip-connections."""

    def __init__(self, in_features: int, hidden: int = 64, dropout: float = 0.25):
        super().__init__()
        self.proj = nn.Linear(in_features, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln3 = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)
        self.skip = nn.Linear(in_features, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.ln1(self.proj(x)))
        res = self.skip(x)
        h2 = torch.relu(self.ln2(self.fc1(h)))
        h2 = self.drop(torch.relu(self.ln3(self.fc2(h2))))
        out = self.out(h2 + res)
        return out.squeeze(1) if out.ndim > 1 else out

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


class GallstoneNet(nn.Module):
    """Feed-forward classifier for gallstone prediction."""

    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()

        self.network = nn.Sequential(
            # ── Normalise raw inputs ─────────────────────────────────
            nn.BatchNorm1d(input_dim),

            # ── Hidden layer 1 ───────────────────────────────────────
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # ── Hidden layer 2 ───────────────────────────────────────
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            # ── Hidden layer 3 ───────────────────────────────────────
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.66),  # lighter dropout near output

            # ── Output (raw logit) ───────────────────────────────────
            nn.Linear(32, 1),
        )

        # Initialise weights using Kaiming (good for ReLU networks)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, input_dim)

        Returns
        -------
        logits : (batch, 1)  — pass through sigmoid for probabilities
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities (0-1) instead of raw logits."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
