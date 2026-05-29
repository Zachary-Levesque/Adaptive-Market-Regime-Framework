"""Compare phase 3 alpha models against simple sklearn baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.alpha.baselines import BaselineSpec, build_default_baseline_specs
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
class AlphaComparisonArtifacts:
    fold_metrics: pd.DataFrame
    leaderboard: pd.DataFrame
    best_model: str
    best_signal_path: Path | None
    signal_paths: dict[str, Path]


class AlphaModelComparator:
    """Score the existing ensemble and a set of baseline models on walk-forward splits."""

    def __init__(
        self,
        alpha_config: AlphaConfig,
        regime_config: RegimeConfig,
        baseline_specs: list[BaselineSpec] | None = None,
    ) -> None:
        self.alpha_config = alpha_config
        self.regime_config = regime_config
        self.baseline_specs = baseline_specs or build_default_baseline_specs()

    def build(
        self,
        technical_features: pd.DataFrame,
        returns: pd.DataFrame,
        factors: pd.DataFrame,
        regime_labels: pd.DataFrame | pd.Series,
        epochs_override: int | None = None,
        include_ensemble: bool = True,
    ) -> AlphaComparisonArtifacts:
        regime_series = extract_regime_series(regime_labels)
        unique_regimes = sorted(int(regime) for regime in regime_series.dropna().unique())
        epochs = int(epochs_override) if epochs_override is not None else self.alpha_config.epochs
        validator = WalkForwardValidator(
            train_window=self.alpha_config.train_window,
            test_window=self.alpha_config.test_window,
            step_size=self.alpha_config.step_size,
        )

        model_specs: list[tuple[str, Callable[[int], object]]] = [
            *[(spec.name, spec.factory) for spec in self.baseline_specs],
        ]
        if include_ensemble:
            model_specs.append(
                (
                    "ensemble",
                    lambda input_size: RegimeAlphaEnsemble(
                        input_size=input_size,
                        hidden_size=self.alpha_config.hidden_size,
                        num_layers=self.alpha_config.num_layers,
                        dropout=self.alpha_config.dropout,
                        learning_rate=self.alpha_config.learning_rate,
                        weight_decay=self.alpha_config.weight_decay,
                        batch_size=self.alpha_config.batch_size,
                        patience=self.alpha_config.patience,
                        sequence_length=self.alpha_config.sequence_length,
                    ),
                )
            )

        fold_frames: list[pd.DataFrame] = []
        signal_frames: dict[str, pd.DataFrame] = {
            model_name: pd.DataFrame(np.nan, index=returns.index, columns=returns.columns, dtype=float)
            for model_name, _ in model_specs
        }

        for model_name, factory in model_specs:
            for regime in unique_regimes:
                regime_signal_frame, regime_metrics = self._project_regime_model(
                    model_name=model_name,
                    model_factory=factory,
                    regime=regime,
                    technical_features=technical_features,
                    returns=returns,
                    factors=factors,
                    regime_series=regime_series,
                    validator=validator,
                    epochs=epochs,
                )
                if not regime_metrics.empty:
                    fold_frames.append(regime_metrics)
                if not regime_signal_frame.empty:
                    signal_frames[model_name] = signal_frames[model_name].combine_first(regime_signal_frame)

        fold_metrics = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
        leaderboard = self._summarize(fold_metrics)
        signal_paths = self.save(signal_frames, fold_metrics, leaderboard)
        best_model = str(leaderboard.index[0]) if not leaderboard.empty else ""
        best_signal_path = signal_paths.get(best_model)
        return AlphaComparisonArtifacts(
            fold_metrics=fold_metrics,
            leaderboard=leaderboard,
            best_model=best_model,
            best_signal_path=best_signal_path,
            signal_paths=signal_paths,
        )

    def save(
        self,
        signal_frames: dict[str, pd.DataFrame],
        fold_metrics: pd.DataFrame,
        leaderboard: pd.DataFrame,
    ) -> dict[str, Path]:
        output_path = self.alpha_config.comparison_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fold_metrics.to_parquet(output_path)
        signal_dir = self.alpha_config.signals_dir
        signal_dir.mkdir(parents=True, exist_ok=True)

        signal_paths: dict[str, Path] = {}
        for model_name, frame in signal_frames.items():
            signal_path = signal_dir / f"{model_name}.parquet"
            frame.sort_index().to_parquet(signal_path)
            signal_paths[model_name] = signal_path

        leaderboard_to_save = leaderboard.copy()
        leaderboard_to_save["signal_path"] = pd.Series(
            {model_name: str(path) for model_name, path in signal_paths.items()}
        )
        leaderboard_to_save.to_parquet(output_path.with_name("alpha_model_comparison_summary.parquet"))

        selection = pd.DataFrame(
            [
                {
                    "model": leaderboard_to_save.index[0] if not leaderboard_to_save.empty else "",
                    "signal_path": str(signal_paths.get(leaderboard_to_save.index[0], "")) if not leaderboard_to_save.empty else "",
                }
            ]
        )
        if not leaderboard_to_save.empty:
            top_model = leaderboard_to_save.index[0]
            for column in leaderboard_to_save.columns:
                selection.loc[0, column] = leaderboard_to_save.loc[top_model, column]
            selection.loc[0, "model"] = top_model
            selection.loc[0, "signal_path"] = str(signal_paths[top_model])
        selection.to_parquet(self.alpha_config.selection_path)
        logger.info("Saved alpha model comparison to {}", output_path)
        return signal_paths

    def _project_regime_model(
        self,
        model_name: str,
        model_factory,
        regime: int,
        technical_features: pd.DataFrame,
        returns: pd.DataFrame,
        factors: pd.DataFrame,
        regime_series: pd.Series,
        validator: WalkForwardValidator,
        epochs: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            return pd.DataFrame(), pd.DataFrame()

        regime_signal_frame = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns, dtype=float)
        fold_rows: list[dict[str, float | int]] = []
        usable_dates = technical_features.index.intersection(returns.index).intersection(regime_series.index)
        usable_dates = usable_dates.intersection(factors.index if factors is not None else returns.index)
        splits = validator.generate_splits(usable_dates)

        for fold, (train_dates, test_dates) in enumerate(splits):
            train_dataset = RegimeDataset(
                features=technical_features,
                returns=returns,
                regime_labels=regime_series,
                target_regime=regime,
                factors=factors,
                sequence_length=self.alpha_config.sequence_length,
                min_samples=self.alpha_config.min_samples_per_regime,
                augment_noise_std=self.alpha_config.augment_noise_std,
                allowed_dates=train_dates,
            )
            test_dataset = RegimeDataset(
                features=technical_features,
                returns=returns,
                regime_labels=regime_series,
                target_regime=regime,
                factors=factors,
                sequence_length=self.alpha_config.sequence_length,
                min_samples=0,
                augment_noise_std=self.alpha_config.augment_noise_std,
                allowed_dates=test_dates,
            )
            if len(train_dataset) < 2 or len(test_dataset) == 0:
                continue

            train_subset, val_subset = temporal_train_val_split(train_dataset, self.alpha_config.validation_fraction)
            model = model_factory(train_dataset.input_size)
            model.fit(train_subset, val_subset, epochs=epochs, device=self.alpha_config.device)
            predictions = model.predict_dataset(test_dataset, device=self.alpha_config.device)
            actuals = test_dataset.targets.detach().cpu().numpy()

            fold_rows.append(
                {
                    "model": model_name,
                    "regime": regime,
                    "fold": fold,
                    "n_train": len(train_dataset),
                    "n_test": len(test_dataset),
                    "sharpe": validator._compute_daily_sharpe(predictions, actuals, test_dataset.sample_dates),
                    "ic": validator._safe_corr(predictions, actuals, method="pearson"),
                    "rank_ic": validator._safe_corr(predictions, actuals, method="spearman"),
                    "hit_rate": float(np.mean(np.sign(predictions) == np.sign(actuals))),
                }
            )

            for date, ticker, prediction in zip(test_dataset.sample_dates, test_dataset.sample_tickers, predictions):
                regime_signal_frame.at[pd.Timestamp(date), ticker] = float(prediction)

        return regime_signal_frame, pd.DataFrame(fold_rows)

    @staticmethod
    def _summarize(fold_metrics: pd.DataFrame) -> pd.DataFrame:
        if fold_metrics.empty:
            empty = pd.DataFrame(
                columns=[
                    "n_rows",
                    "n_folds",
                    "n_regimes",
                    "mean_sharpe",
                    "median_sharpe",
                    "mean_ic",
                    "mean_rank_ic",
                    "mean_hit_rate",
                    "mean_train_size",
                    "mean_test_size",
                ]
            )
            empty.index.name = "model"
            return empty

        rows: list[dict[str, float | int | str]] = []
        for model_name, group in fold_metrics.groupby("model"):
            rows.append(
                {
                    "model": model_name,
                    "n_rows": int(len(group)),
                    "n_folds": int(group["fold"].nunique()) if "fold" in group.columns else int(len(group)),
                    "n_regimes": int(group["regime"].nunique()) if "regime" in group.columns else 0,
                    "mean_sharpe": float(group["sharpe"].mean()) if "sharpe" in group.columns else 0.0,
                    "median_sharpe": float(group["sharpe"].median()) if "sharpe" in group.columns else 0.0,
                    "mean_ic": float(group["ic"].mean()) if "ic" in group.columns else 0.0,
                    "mean_rank_ic": float(group["rank_ic"].mean()) if "rank_ic" in group.columns else 0.0,
                    "mean_hit_rate": float(group["hit_rate"].mean()) if "hit_rate" in group.columns else 0.0,
                    "mean_train_size": float(group["n_train"].mean()) if "n_train" in group.columns else 0.0,
                    "mean_test_size": float(group["n_test"].mean()) if "n_test" in group.columns else 0.0,
                }
            )

        leaderboard = pd.DataFrame(rows).set_index("model")
        leaderboard = leaderboard.sort_values(["mean_sharpe", "mean_ic", "mean_hit_rate"], ascending=False)
        return leaderboard
