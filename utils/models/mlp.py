"""
utils/models/mlp.py
===================
Multi-layer perceptron for tabular four-class sleep-stage classification.
"""

import torch.nn as nn


class SleepMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = (256, 128, 64),
        dropout: float = 0.3,
        num_classes: int = 4,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
