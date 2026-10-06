"""
model.py — Neural network architecture for gallstone prediction.

Architecture:
  Input(N features)
    → BatchNorm → Linear(128) → ReLU → Dropout(0.3)
    → Linear(64) → ReLU → Dropout(0.3)
    → Linear(32) → ReLU → Dropout(0.2)
    → Linear(1)   ← raw logit (use BCEWithLogitsLoss)

Why this architecture:
  • BatchNorm on input stabilises training across differently-scaled features
  • 3 hidden layers give enough capacity for ~35 features / 319 patients
  • Dropout regularises aggressively to prevent overfitting on a small dataset
  • We output a raw logit — sigmoid is applied inside BCEWithLogitsLoss for
    numerical stability, and manually during inference
"""

import torch
import torch.nn as nn


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
