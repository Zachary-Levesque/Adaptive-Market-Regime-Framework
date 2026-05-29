"""Phase 3 alpha-model pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.alpha.dataset import RegimeDataset, extract_regime_series
from src.alpha.ensemble import RegimeAlphaEnsemble
from src.alpha.training import temporal_train_val_split
from src.alpha.walk_forward import WalkForwardValidator
from src.config import AlphaConfig, RegimeConfig

try:  # pragma: no cover - exercised indirectly depending on environment
    from loguru import logger
except ImportError:  # pragma: no cover - dependency may not be installed in CI/local env
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class AlphaArtifacts:
    alpha_signals: pd.DataFrame
    validation_metrics: pd.DataFrame
    trained_regimes: list[int]


class AlphaPipeline:
    """Train regime-specific alpha ensembles and persist forecasts."""

    def __init__(self, alpha_config: AlphaConfig, regime_config: RegimeConfig) -> None:
        self.alpha_config = alpha_config
        self.regime_config = regime_config

    def build(
        self,
        technical_features: pd.DataFrame,
        returns: pd.DataFrame,
        factors: pd.DataFrame,
        regime_labels: pd.DataFrame | pd.Series,
        epochs_override: int | None = None,
        run_validation: bool = True,
    ) -> AlphaArtifacts:
        regime_series = extract_regime_series(regime_labels)
        trained_regimes: list[int] = []
        signal_frame = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
        metrics_frames: list[pd.DataFrame] = []

        unique_regimes = sorted(int(regime) for regime in regime_series.dropna().unique())
        epochs = int(epochs_override) if epochs_override is not None else self.alpha_config.epochs

        for regime in unique_regimes:
            dataset = RegimeDataset(
                features=technical_features,
                returns=returns,
                regime_labels=regime_series,
                target_regime=regime,
                factors=factors,
                sequence_length=self.alpha_config.sequence_length,
                min_samples=self.alpha_config.min_samples_per_regime,
                augment_noise_std=self.alpha_config.augment_noise_std,
            )
            if len(dataset) < 2 or dataset.input_size == 0:
                logger.warning("Skipping regime {} due to insufficient samples", regime)
                continue

            trained_regimes.append(regime)
            train_dataset, val_dataset = temporal_train_val_split(dataset, self.alpha_config.validation_fraction)
            ensemble = RegimeAlphaEnsemble(
                input_size=dataset.input_size,
                hidden_size=self.alpha_config.hidden_size,
                num_layers=self.alpha_config.num_layers,
                dropout=self.alpha_config.dropout,
                learning_rate=self.alpha_config.learning_rate,
                weight_decay=self.alpha_config.weight_decay,
                batch_size=self.alpha_config.batch_size,
                patience=self.alpha_config.patience,
                sequence_length=self.alpha_config.sequence_length,
                target_regime=regime,
                feature_names=dataset.feature_names,
            )
            validation = ensemble.fit(train_dataset, val_dataset, epochs=epochs, device=self.alpha_config.device)
            ensemble_dir = self.alpha_config.model_dir / f"regime_{regime}"
            ensemble.save(ensemble_dir)

            predictions = ensemble.predict_dataset(dataset, device=self.alpha_config.device)
            signal_frame = self._write_predictions(signal_frame, dataset.sample_dates, dataset.sample_tickers, predictions)

            if run_validation:
                validator = WalkForwardValidator(
                    train_window=self.alpha_config.train_window,
                    test_window=self.alpha_config.test_window,
                    step_size=self.alpha_config.step_size,
                )
                metrics = validator.validate(
                    model_factory=lambda input_size: RegimeAlphaEnsemble(
                        input_size=input_size,
                        hidden_size=self.alpha_config.hidden_size,
                        num_layers=self.alpha_config.num_layers,
                        dropout=self.alpha_config.dropout,
                        learning_rate=self.alpha_config.learning_rate,
                        weight_decay=self.alpha_config.weight_decay,
                        batch_size=self.alpha_config.batch_size,
                        patience=self.alpha_config.patience,
                        sequence_length=self.alpha_config.sequence_length,
                        target_regime=regime,
                    ),
                    features=technical_features,
                    returns=returns,
                    regime_labels=regime_series,
                    target_regime=regime,
                    factors=factors,
                    sequence_length=self.alpha_config.sequence_length,
                    epochs=max(1, min(epochs, 5)),
                    validation_fraction=self.alpha_config.validation_fraction,
                    min_samples=self.alpha_config.min_samples_per_regime,
                    augment_noise_std=self.alpha_config.augment_noise_std,
                    device=self.alpha_config.device,
                )
            else:
                metrics = pd.DataFrame()

            if not metrics.empty:
                metrics.insert(0, "regime", regime)
                metrics["lstm_weight"] = validation.lstm_weight
                metrics["transformer_weight"] = validation.transformer_weight
                metrics_frames.append(metrics)

        alpha_signals = signal_frame.sort_index()
        validation_metrics = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
        self._persist(alpha_signals, validation_metrics)
        return AlphaArtifacts(
            alpha_signals=alpha_signals,
            validation_metrics=validation_metrics,
            trained_regimes=trained_regimes,
        )

    def _persist(self, alpha_signals: pd.DataFrame, validation_metrics: pd.DataFrame) -> None:
        self.alpha_config.model_dir.mkdir(parents=True, exist_ok=True)
        self.alpha_config.signals_path.parent.mkdir(parents=True, exist_ok=True)
        alpha_signals.to_parquet(self.alpha_config.signals_path)
        metrics_to_save = validation_metrics
        if metrics_to_save.empty and len(metrics_to_save.columns) == 0:
            metrics_to_save = pd.DataFrame(
                columns=[
                    "regime",
                    "fold",
                    "n_train",
                    "n_test",
                    "sharpe",
                    "ic",
                    "rank_ic",
                    "hit_rate",
                    "lstm_weight",
                    "transformer_weight",
                ]
            )
        metrics_to_save.to_parquet(self.alpha_config.metrics_path)
        logger.info("Saved alpha signals to {}", self.alpha_config.signals_path)

    @staticmethod
    def _write_predictions(
        frame: pd.DataFrame,
        dates: pd.DatetimeIndex,
        tickers: list[str],
        predictions: np.ndarray,
    ) -> pd.DataFrame:
        for date, ticker, prediction in zip(dates, tickers, predictions):
            frame.at[pd.Timestamp(date), ticker] = float(prediction)
        return frame
