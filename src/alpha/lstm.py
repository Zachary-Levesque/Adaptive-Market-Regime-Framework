"""Regime-specific LSTM alpha model."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from src.alpha.training import TrainingHistory, predict_dataset, train_model


class RegimeLSTM(nn.Module):
    """LSTM forecaster with attention over hidden states."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        batch_size: int = 32,
        patience: int = 10,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attention = nn.Linear(hidden_size, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.lstm(x)
        attention_scores = self.attention(sequence_output).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.sum(sequence_output * attention_weights.unsqueeze(-1), dim=1)
        return self.head(context).squeeze(-1)

    def fit(self, train_dataset: Dataset, val_dataset: Dataset, epochs: int, device: str = "cpu") -> TrainingHistory:
        return train_model(
            model=self,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            patience=self.patience,
            device=device,
        )

    def predict_dataset(self, dataset: Dataset, device: str = "cpu") -> np.ndarray:
        return predict_dataset(self, dataset=dataset, batch_size=self.batch_size, device=device)
