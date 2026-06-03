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
    data_quality: pd.DataFrame


class DataPipeline:
    """Build the Phase 1 research datasets from config."""

    DEFAULT_MACRO_SERIES = {
        "DGS10": "DGS10",
        "DGS2": "DGS2",
        "VIXCLS": "VIXCLS",
        "CPI": "CPIAUCSL",
        "UNRATE": "UNRATE",
        "FEDFUNDS": "FEDFUNDS",
        "M2": "M2SL",
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
            allow_remote_downloads=config.allow_remote_downloads,
            stooq_api_key=config.stooq_api_key,
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

        ff_factors = self._download_or_load_processed(
            dataset_name="factors",
            path=self.config.processed_dir / "factors.parquet",
            download=lambda: self.factor_loader.download_ff5(
                start=self.config.start_date,
                end=self.config.end_date,
            ),
        )
        factors = self.factor_loader.align_with_returns(ff_factors, returns)

        macro = self._download_or_load_processed(
            dataset_name="macro",
            path=self.config.processed_dir / "macro.parquet",
            download=lambda: self.factor_loader.download_macro_series(
                series_map=self.DEFAULT_MACRO_SERIES,
                start=self.config.start_date,
                end=self.config.end_date,
            ),
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
        data_quality = self._build_data_quality_report(
            prices=prices,
            returns=returns,
            factors=factors,
            macro=macro,
        )

        artifacts = PipelineArtifacts(
            prices=prices,
            returns=returns,
            factors=factors,
            technical_features=technical_features,
            regime_features=regime_features,
            macro=macro,
            data_quality=data_quality,
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
            (artifacts.data_quality, output_dir / "data_quality_report.parquet"),
        ]

        for frame, path in outputs:
            self.ingester.save(frame, path)

    def _download_or_load_processed(self, dataset_name: str, path: Path, download) -> pd.DataFrame:
        try:
            return download()
        except Exception as exc:
            if not path.exists():
                raise
            logger.warning(
                "Unable to refresh {} from provider: {}. Using existing processed artifact {}.",
                dataset_name,
                exc,
                path,
            )
            return pd.read_parquet(path)

    def _build_data_quality_report(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        factors: pd.DataFrame,
        macro: pd.DataFrame,
    ) -> pd.DataFrame:
        """Summarize coverage and missingness for downstream readiness checks."""
        if not isinstance(prices.columns, pd.MultiIndex):
            raise TypeError("Expected price data with (ticker, field) MultiIndex columns.")

        price_field = "Adj Close" if "Adj Close" in prices.columns.get_level_values(1) else "Close"
        close = prices.xs(price_field, axis=1, level=1).sort_index()
        normalized_returns = returns.copy()
        normalized_returns.index = pd.to_datetime(normalized_returns.index).tz_localize(None)
        gfc_start = pd.Timestamp("2008-09-01")
        gfc_end = pd.Timestamp("2009-03-31")

        rows: list[dict[str, object]] = []
        for ticker in close.columns:
            close_series = pd.to_numeric(close[ticker], errors="coerce")
            valid_close = close_series.dropna()
            return_series = (
                pd.to_numeric(normalized_returns[ticker], errors="coerce")
                if ticker in normalized_returns.columns
                else pd.Series(dtype=float)
            )
            first_date = valid_close.index.min() if not valid_close.empty else pd.NaT
            last_date = valid_close.index.max() if not valid_close.empty else pd.NaT
            gfc_window = close_series.loc[(close_series.index >= gfc_start) & (close_series.index <= gfc_end)]
            rows.append(
                {
                    "dataset": "prices",
                    "symbol": str(ticker),
                    "first_valid_date": "" if pd.isna(first_date) else str(pd.Timestamp(first_date).date()),
                    "last_valid_date": "" if pd.isna(last_date) else str(pd.Timestamp(last_date).date()),
                    "n_observations": int(close_series.notna().sum()),
                    "missing_fraction": float(close_series.isna().mean()) if len(close_series) else 1.0,
                    "return_observations": int(return_series.notna().sum()),
                    "return_missing_fraction": float(return_series.isna().mean()) if len(return_series) else 1.0,
                    "covers_gfc": bool(gfc_window.notna().any()),
                }
            )

        rows.extend(self._frame_quality_rows("factors", factors))
        rows.extend(self._frame_quality_rows("macro", macro))
        return pd.DataFrame(rows)

    @staticmethod
    def _frame_quality_rows(dataset: str, frame: pd.DataFrame) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        normalized = frame.copy()
        normalized.index = pd.to_datetime(normalized.index).tz_localize(None)
        for column in normalized.columns:
            series = pd.to_numeric(normalized[column], errors="coerce")
            valid = series.dropna()
            first_date = valid.index.min() if not valid.empty else pd.NaT
            last_date = valid.index.max() if not valid.empty else pd.NaT
            rows.append(
                {
                    "dataset": dataset,
                    "symbol": str(column),
                    "first_valid_date": "" if pd.isna(first_date) else str(pd.Timestamp(first_date).date()),
                    "last_valid_date": "" if pd.isna(last_date) else str(pd.Timestamp(last_date).date()),
                    "n_observations": int(series.notna().sum()),
                    "missing_fraction": float(series.isna().mean()) if len(series) else 1.0,
                    "return_observations": 0,
                    "return_missing_fraction": 1.0,
                    "covers_gfc": bool(
                        series.loc[
                            (series.index >= pd.Timestamp("2008-09-01"))
                            & (series.index <= pd.Timestamp("2009-03-31"))
                        ].notna().any()
                    ),
                }
            )
        return rows
