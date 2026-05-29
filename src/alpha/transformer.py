"""Regime-specific Transformer alpha model."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from src.alpha.training import TrainingHistory, predict_dataset, train_model


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        positions = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.size(1)]


class RegimeTransformer(nn.Module):
    """Transformer encoder forecaster for regime-specific alpha."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        batch_size: int = 32,
        patience: int = 10,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience

        self.input_projection = nn.Linear(input_size, hidden_size)
        self.position = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(x)
        encoded = self.encoder(self.position(projected))
        pooled = encoded.mean(dim=1)
        return self.head(pooled).squeeze(-1)

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
