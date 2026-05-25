"""Phase 1 data pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import DataConfig
from src.data.factors import FactorLoader
from src.data.features import FeatureEngineer
from src.data.ingestion import MarketDataIngester

try:  # pragma: no cover - exercised indirectly depending on environment
    from loguru import logger
except ImportError:  # pragma: no cover - dependency may not be installed in CI/local env
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class PipelineArtifacts:
    prices: pd.DataFrame
    returns: pd.DataFrame
    factors: pd.DataFrame
    technical_features: pd.DataFrame
    regime_features: pd.DataFrame
    macro: pd.DataFrame


class DataPipeline:
    """Build the Phase 1 research datasets from config."""

    DEFAULT_MACRO_SERIES = {
        "DGS10": "DGS10",
        "DGS2": "DGS2",
        "VIXCLS": "VIXCLS",
    }

    def __init__(
        self,
        config: DataConfig,
        ingester: MarketDataIngester | None = None,
        factor_loader: FactorLoader | None = None,
        feature_engineer: FeatureEngineer | None = None,
    ) -> None:
        self.config = config
        self.ingester = ingester or MarketDataIngester(
            cache_dir=config.cache_dir,
            local_data_dir=config.local_data_dir,
        )
        self.factor_loader = factor_loader or FactorLoader()
        self.feature_engineer = feature_engineer or FeatureEngineer()

    def build(self) -> PipelineArtifacts:
        """Download, transform, and persist the Phase 1 datasets."""
        logger.info("Building Phase 1 datasets for {} tickers", len(self.config.universe))

        price_universe = [ticker for ticker in self.config.universe if ticker not in {"^VIX", "VIX"}]

        prices = self.ingester.download_prices(
            tickers=price_universe,
            start=self.config.start_date,
            end=self.config.end_date,
        )
        returns = self.ingester.compute_returns(prices)

        ff_factors = self.factor_loader.download_ff5()
        factors = self.factor_loader.align_with_returns(ff_factors, returns)

        macro = self.factor_loader.download_macro_series(
            series_map=self.DEFAULT_MACRO_SERIES,
            start=self.config.start_date,
            end=self.config.end_date,
        )
        macro = macro.reindex(prices.index).ffill()
        vix_series = macro["VIXCLS"] if "VIXCLS" in macro.columns else None

        technical_features = self.feature_engineer.compute_technical_features(prices, vix=vix_series)
        technical_features = self.feature_engineer.normalize(technical_features)
        regime_features = self.feature_engineer.compute_regime_features(
            prices=prices,
            vix=vix_series,
            macro=macro,
            benchmark=self.config.benchmark,
        )

        artifacts = PipelineArtifacts(
            prices=prices,
            returns=returns,
            factors=factors,
            technical_features=technical_features,
            regime_features=regime_features,
            macro=macro,
        )
        self._persist(artifacts)
        return artifacts

    def _persist(self, artifacts: PipelineArtifacts) -> None:
        output_dir = self.config.processed_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs: list[tuple[pd.DataFrame, Path]] = [
            (artifacts.prices, output_dir / "prices.parquet"),
            (artifacts.returns, output_dir / "returns.parquet"),
            (artifacts.factors, output_dir / "factors.parquet"),
            (artifacts.technical_features, output_dir / "technical_features.parquet"),
            (artifacts.regime_features, output_dir / "regime_features.parquet"),
            (artifacts.macro, output_dir / "macro.parquet"),
        ]

        for frame, path in outputs:
            self.ingester.save(frame, path)
