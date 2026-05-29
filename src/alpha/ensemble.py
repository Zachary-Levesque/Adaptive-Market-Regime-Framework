"""Ensemble forecaster combining LSTM and Transformer alpha models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.alpha.dataset import RegimeDataset
from src.alpha.lstm import RegimeLSTM
from src.alpha.transformer import RegimeTransformer


def compute_prediction_sharpe(predictions: np.ndarray, actual_returns: np.ndarray) -> float:
    if len(predictions) == 0:
        return 0.0
    realized = predictions.reshape(-1) * actual_returns.reshape(-1)
    std = float(np.std(realized))
    if std == 0.0:
        return 0.0
    return float(np.mean(realized) / std)


@dataclass
class EnsembleValidation:
    lstm_sharpe: float
    transformer_sharpe: float
    lstm_weight: float
    transformer_weight: float


class RegimeAlphaEnsemble:
    """Blend LSTM and Transformer predictions for a given regime."""

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
        sequence_length: int = 60,
        target_regime: int | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.target_regime = target_regime
        self.feature_names = feature_names or []
        self.lstm = RegimeLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            patience=patience,
        )
        self.transformer = RegimeTransformer(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            patience=patience,
        )
        self.weights = {"lstm": 0.6, "transformer": 0.4}
        self.validation_: EnsembleValidation | None = None

    def fit(self, train_dataset: Dataset, val_dataset: Dataset, epochs: int, device: str = "cpu") -> EnsembleValidation:
        self.lstm.fit(train_dataset, val_dataset, epochs=epochs, device=device)
        self.transformer.fit(train_dataset, val_dataset, epochs=epochs, device=device)

        actuals = self._extract_targets(val_dataset)
        lstm_preds = self.lstm.predict_dataset(val_dataset, device=device)
        transformer_preds = self.transformer.predict_dataset(val_dataset, device=device)
        lstm_sharpe = compute_prediction_sharpe(lstm_preds, actuals)
        transformer_sharpe = compute_prediction_sharpe(transformer_preds, actuals)

        lstm_weight, transformer_weight = self._derive_weights(lstm_sharpe, transformer_sharpe)
        self.weights = {"lstm": lstm_weight, "transformer": transformer_weight}
        self.validation_ = EnsembleValidation(
            lstm_sharpe=lstm_sharpe,
            transformer_sharpe=transformer_sharpe,
            lstm_weight=lstm_weight,
            transformer_weight=transformer_weight,
        )
        return self.validation_

    def predict_dataset(self, dataset: Dataset, device: str = "cpu") -> np.ndarray:
        lstm_preds = self.lstm.predict_dataset(dataset, device=device)
        transformer_preds = self.transformer.predict_dataset(dataset, device=device)
        return self.weights["lstm"] * lstm_preds + self.weights["transformer"] * transformer_preds

    def save(self, model_dir: str | Path) -> None:
        output_dir = Path(model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.lstm.state_dict(), output_dir / "lstm.pt")
        torch.save(self.transformer.state_dict(), output_dir / "transformer.pt")
        metadata = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "sequence_length": self.sequence_length,
            "target_regime": self.target_regime,
            "feature_names": self.feature_names,
            "weights": self.weights,
        }
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    @classmethod
    def load(cls, model_dir: str | Path) -> "RegimeAlphaEnsemble":
        base = Path(model_dir)
        with (base / "metadata.json").open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        ensemble = cls(
            input_size=int(metadata["input_size"]),
            hidden_size=int(metadata.get("hidden_size", 128)),
            num_layers=int(metadata.get("num_layers", 2)),
            dropout=float(metadata.get("dropout", 0.2)),
            sequence_length=int(metadata["sequence_length"]),
            target_regime=metadata.get("target_regime"),
            feature_names=list(metadata.get("feature_names", [])),
        )
        ensemble.weights = {
            "lstm": float(metadata["weights"]["lstm"]),
            "transformer": float(metadata["weights"]["transformer"]),
        }
        ensemble.lstm.load_state_dict(torch.load(base / "lstm.pt", map_location="cpu"))
        ensemble.transformer.load_state_dict(torch.load(base / "transformer.pt", map_location="cpu"))
        return ensemble

    @staticmethod
    def _derive_weights(lstm_sharpe: float, transformer_sharpe: float) -> tuple[float, float]:
        positive_lstm = max(lstm_sharpe, 0.0)
        positive_transformer = max(transformer_sharpe, 0.0)
        total = positive_lstm + positive_transformer
        if total <= 0.0:
            return 0.6, 0.4
        return positive_lstm / total, positive_transformer / total

    @staticmethod
    def _extract_targets(dataset: Dataset) -> np.ndarray:
        if isinstance(dataset, RegimeDataset):
            return dataset.targets.detach().cpu().numpy()
        if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
            base_targets = dataset.dataset.targets.detach().cpu().numpy()
            return base_targets[np.array(dataset.indices)]
        raise TypeError("Unsupported dataset type for target extraction.")
