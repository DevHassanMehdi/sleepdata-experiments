"""
utils/models/lstm.py
====================
Bidirectional LSTM for sequence-of-epochs sleep-stage classification.
Input shape: (batch, seq_len, input_dim).
"""

import torch.nn as nn


class SleepLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 4,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        last_out    = lstm_out[:, -1, :]   # last timestep
        return self.classifier(last_out)
