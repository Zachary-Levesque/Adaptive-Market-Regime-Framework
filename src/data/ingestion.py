"""Market data ingestion utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:  # pragma: no cover - exercised indirectly depending on environment
    import yfinance as yf
except ImportError:  # pragma: no cover - dependency may not be installed in CI/local env
    yf = None

try:  # pragma: no cover - exercised indirectly depending on environment
    from loguru import logger
except ImportError:  # pragma: no cover - dependency may not be installed in CI/local env
    import logging

    logger = logging.getLogger(__name__)


class MarketDataIngester:
    """Download, clean, and persist market data."""

    REQUIRED_FIELDS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")

    def __init__(self, max_forward_fill: int = 5, max_missing_fraction: float = 0.10):
        self.max_forward_fill = max_forward_fill
        self.max_missing_fraction = max_missing_fraction

    def download_prices(
        self,
        tickers: Iterable[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Download OHLCV data as a ticker-first MultiIndex DataFrame."""
        if yf is None:
            raise ImportError("yfinance is required for download_prices(). Install dependencies first.")

        frames: list[pd.DataFrame] = []

        for ticker in tickers:
            try:
                raw = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
            except Exception as exc:  # pragma: no cover - network/runtime dependent
                logger.exception("Failed to download {}: {}", ticker, exc)
                continue

            if raw.empty:
                logger.warning("No data returned for {}", ticker)
                continue

            formatted = self._format_single_ticker_frame(raw, ticker)
            if self._missing_fraction(formatted) > self.max_missing_fraction:
                logger.warning(
                    "Dropping {} due to missing fraction {:.2%}",
                    ticker,
                    self._missing_fraction(formatted),
                )
                continue

            cleaned = formatted.ffill(limit=self.max_forward_fill)
            frames.append(cleaned)

        if not frames:
            raise ValueError("No price data could be downloaded for the requested universe.")

        prices = pd.concat(frames, axis=1).sort_index(axis=1)
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        logger.info("Downloaded price history for {} tickers", len(prices.columns.levels[0]))
        return prices

    def compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute cleaned daily log returns from close prices."""
        close_prices = self._extract_close_prices(prices)
        returns = np.log(close_prices / close_prices.shift(1))

        mean = returns.mean()
        std = returns.std(ddof=0).replace(0.0, np.nan)
        classical_z = (returns - mean) / std

        median = returns.median()
        mad = (returns - median).abs().median().replace(0.0, np.nan)
        robust_std = 1.4826 * mad
        robust_z = (returns - median) / robust_std

        outlier_mask = (classical_z.abs() > 5.0) | (robust_z.abs() > 5.0)
        returns = returns.mask(outlier_mask)
        return returns.dropna(how="all")

    def save(self, data: pd.DataFrame, path: str | Path) -> None:
        """Persist a DataFrame to parquet."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(output_path)
        logger.info("Saved dataset to {}", output_path)

    def load(self, path: str | Path) -> pd.DataFrame:
        """Load a parquet dataset."""
        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"Dataset not found: {input_path}")
        return pd.read_parquet(input_path)

    def _format_single_ticker_frame(self, frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
        normalized = frame.copy()
        normalized.columns = [str(column) for column in normalized.columns]

        if "Adj Close" not in normalized.columns and "Close" in normalized.columns:
            normalized["Adj Close"] = normalized["Close"]

        normalized = normalized.reindex(columns=self.REQUIRED_FIELDS)
        normalized.columns = pd.MultiIndex.from_product([[ticker], normalized.columns])
        return normalized.sort_index()

    def _missing_fraction(self, formatted: pd.DataFrame) -> float:
        anchor = formatted.xs("Adj Close", axis=1, level=1).iloc[:, 0]
        return float(anchor.isna().mean())

    @staticmethod
    def _extract_close_prices(prices: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(prices.columns, pd.MultiIndex):
            raise TypeError("Expected a MultiIndex price DataFrame with ticker and field levels.")

        fields = prices.columns.get_level_values(1)
        field = "Adj Close" if "Adj Close" in fields else "Close"
        close_prices = prices.xs(field, axis=1, level=1)
        close_prices = close_prices.sort_index().dropna(how="all")
        close_prices.columns.name = None
        return close_prices
