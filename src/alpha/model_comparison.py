"""Compare phase 3 alpha models against simple sklearn baselines."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.alpha.baselines import BaselineSpec, build_default_baseline_specs
from src.alpha.dataset import RegimeDataset, extract_regime_series
from src.alpha.ensemble import RegimeAlphaEnsemble
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

        fold_frames: list[pd.DataFrame] = []
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

            for spec in self.baseline_specs:
                metrics = validator.validate(
                    model_factory=lambda _input_size, factory=spec.factory: factory(),
                    features=technical_features,
                    returns=returns,
                    regime_labels=regime_series,
                    target_regime=regime,
                    factors=factors,
                    sequence_length=self.alpha_config.sequence_length,
                    epochs=epochs,
                    validation_fraction=self.alpha_config.validation_fraction,
                    min_samples=self.alpha_config.min_samples_per_regime,
                    augment_noise_std=self.alpha_config.augment_noise_std,
                    device=self.alpha_config.device,
                )
                if metrics.empty:
                    continue
                metrics.insert(0, "model", spec.name)
                metrics.insert(1, "regime", regime)
                fold_frames.append(metrics)

        if include_ensemble and self.alpha_config.metrics_path.exists():
            ensemble_metrics = pd.read_parquet(self.alpha_config.metrics_path).copy()
            if not ensemble_metrics.empty:
                ensemble_metrics.insert(0, "model", "ensemble")
                if "regime" not in ensemble_metrics.columns:
                    ensemble_metrics.insert(1, "regime", pd.NA)
                fold_frames.append(ensemble_metrics)

        fold_metrics = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
        leaderboard = self._summarize(fold_metrics)
        best_model = str(leaderboard.index[0]) if not leaderboard.empty else ""
        self.save(fold_metrics, leaderboard)
        return AlphaComparisonArtifacts(
            fold_metrics=fold_metrics,
            leaderboard=leaderboard,
            best_model=best_model,
        )

    def save(self, fold_metrics: pd.DataFrame, leaderboard: pd.DataFrame) -> None:
        output_path = self.alpha_config.comparison_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fold_metrics.to_parquet(output_path)
        leaderboard.to_parquet(output_path.with_name("alpha_model_comparison_summary.parquet"))
        logger.info("Saved alpha model comparison to {}", output_path)

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
