"""A small LSTM predicts only the change from the observed current log-flux."""
import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(self, hidden: int = 64, dropout: float = .2):
        super().__init__()
        self.lstm = nn.LSTM(6, hidden, num_layers=2, dropout=dropout, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(inputs)
        return self.head(sequence[:, -1]).squeeze(-1)
