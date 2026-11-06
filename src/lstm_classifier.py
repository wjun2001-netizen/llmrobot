"""Fixed, small binary classifier for five time steps of [E,C,S]."""
import torch
from torch import nn


class FailureRiskLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=16, num_layers=1, batch_first=True)
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (5, 3):
            raise ValueError("Expected input tensor [batch, 5, 3] in time-major feature order")
        if not torch.isfinite(x).all():
            raise ValueError("Non-finite model input")
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1]).squeeze(-1)  # Raw logits; sigmoid belongs to evaluation only.
