"""
utils/models/cnn.py
===================
1-D CNN for tabular feature vectors reshaped as single-channel sequences.
Input shape: (batch, input_dim) — unsqueezed to (batch, 1, input_dim) inside forward().
"""

import torch.nn as nn


class SleepCNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (batch, input_dim) → (batch, 1, input_dim)
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x)
