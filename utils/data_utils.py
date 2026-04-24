"""
utils/data_utils.py
===================
PyTorch Dataset classes for sleep-stage classification.
"""

import torch
from torch.utils.data import Dataset


class EpochDataset(Dataset):
    """Single-epoch dataset for MLP, CNN, RandomForest, XGBoost."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X if not hasattr(X, "values") else X.values)
        self.y = torch.LongTensor(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SequenceDataset(Dataset):
    """
    Sliding-window sequence dataset for LSTM.

    Each sample is a window of seq_len consecutive epochs ending at index i,
    labelled by the stage of epoch i.  The first seq_len-1 epochs are skipped
    so every window is fully populated.
    """

    def __init__(self, X, y, seq_len: int = 10):
        self.seq_len   = seq_len
        self.X         = torch.FloatTensor(X if not hasattr(X, "values") else X.values)
        self.y         = torch.LongTensor(y)
        self.valid_idx = list(range(seq_len - 1, len(y)))

    def __len__(self) -> int:
        return len(self.valid_idx)

    def __getitem__(self, idx):
        end   = self.valid_idx[idx] + 1
        start = end - self.seq_len
        return self.X[start:end], self.y[self.valid_idx[idx]]
