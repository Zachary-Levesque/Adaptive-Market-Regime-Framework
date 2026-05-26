"""Phase 2 regime detection pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import RegimeConfig
from src.regime.bayesian import BayesianRegimeSmoothing
from src.regime.gmm import RegimeGMM
from src.regime.hmm import RegimeHMM
from src.regime.kalman import KalmanRegimeFilter
from src.regime.visualize import plot_regime_history

try:  # pragma: no cover - exercised indirectly depending on environment
    from loguru import logger
except ImportError:  # pragma: no cover - dependency may not be installed in CI/local env
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class RegimeArtifacts:
    regime_labels: pd.DataFrame
    regime_probs: pd.DataFrame
    transition_matrix: pd.DataFrame
    regime_summary: pd.DataFrame
    gmm_score: float


class RegimeDetectionPipeline:
    """Build and persist phase-two regime outputs from processed data."""

    def __init__(
        self,
        config: RegimeConfig,
        hmm: RegimeHMM | None = None,
        gmm: RegimeGMM | None = None,
        smoother: BayesianRegimeSmoothing | None = None,
        kalman_filter: KalmanRegimeFilter | None = None,
    ) -> None:
        self.config = config
        self.hmm = hmm or RegimeHMM(
            n_regimes=config.n_regimes,
            covariance_type=config.covariance_type,
            n_iter=config.n_iter,
            n_restarts=config.n_restarts,
            regime_names=config.regime_names,
        )
        self.gmm = gmm or RegimeGMM(n_components=config.n_regimes)
        self.smoother = smoother or BayesianRegimeSmoothing()
        self.kalman_filter = kalman_filter or KalmanRegimeFilter()

    def build(
        self,
        regime_features: pd.DataFrame,
        prices: pd.DataFrame,
        benchmark: str = "SPY",
    ) -> RegimeArtifacts:
        logger.info("Building regime detection outputs from {} observations", len(regime_features))

        self.hmm.fit(regime_features)
        raw_labels = self.hmm.predict_regimes(regime_features, apply_mapping=False)
        summary = self.hmm.label_regimes(raw_labels, regime_features.loc[raw_labels.index, "spy_return"])

        labels = self.hmm.predict_regimes(regime_features)
        probs = self.hmm.predict_proba(regime_features)
        transition_matrix = self.smoother.compute_transition_matrix(labels)
        smoothed_probs = self.smoother.smooth_probabilities(probs, transition_matrix)
        filtered_probs = self.kalman_filter.filter(smoothed_probs)
        full_probs = self._expand_probabilities(filtered_probs, regime_features.index)
        full_labels = full_probs.idxmax(axis=1).map(self._name_to_label())

        regime_label_frame = pd.DataFrame(
            {
                "regime": full_labels.astype(int),
                "regime_name": full_labels.map(self.config.regime_names),
            },
            index=regime_features.index,
        )

        self.gmm.fit(regime_features)
        gmm_labels = self.gmm.predict(regime_features)
        gmm_score = self.gmm.compare_with_hmm(labels, gmm_labels)

        artifacts = RegimeArtifacts(
            regime_labels=regime_label_frame,
            regime_probs=full_probs,
            transition_matrix=transition_matrix,
            regime_summary=summary,
            gmm_score=gmm_score,
        )
        self._persist(artifacts, prices=prices, benchmark=benchmark)
        return artifacts

    def _persist(self, artifacts: RegimeArtifacts, prices: pd.DataFrame, benchmark: str) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        artifacts.regime_labels.to_parquet(self.config.output_dir / "regime_labels.parquet")
        artifacts.regime_probs.to_parquet(self.config.output_dir / "regime_probs.parquet")
        artifacts.transition_matrix.to_parquet(self.config.output_dir / "transition_matrix.parquet")
        artifacts.regime_summary.to_parquet(self.config.output_dir / "regime_summary.parquet")
        pd.DataFrame({"adjusted_rand_score": [artifacts.gmm_score]}).to_parquet(
            self.config.output_dir / "gmm_validation.parquet"
        )
        self.hmm.save(self.config.model_path)
        figure = plot_regime_history(
            prices=prices,
            regime_labels=artifacts.regime_labels["regime"],
            regime_probs=artifacts.regime_probs,
            benchmark=benchmark,
            path=self.config.chart_path,
        )
        figure.clf()
        logger.info("Saved regime outputs to {}", self.config.output_dir)

    def _expand_probabilities(self, probabilities: pd.DataFrame, full_index: pd.Index) -> pd.DataFrame:
        expanded = probabilities.reindex(full_index).ffill().bfill()
        return expanded.div(expanded.sum(axis=1), axis=0)

    def _name_to_label(self) -> dict[str, int]:
        return {name: label for label, name in self.config.regime_names.items()}
