"""Market data ingestion utilities."""

from __future__ import annotations

from pathlib import Path
import time
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

try:  # pragma: no cover - exercised indirectly depending on environment
    from pandas_datareader import data as web
except ImportError:  # pragma: no cover - dependency may not be installed in CI/local env
    web = None


class MarketDataIngester:
    """Download, clean, and persist market data."""

    REQUIRED_FIELDS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")

    def __init__(
        self,
        max_forward_fill: int = 5,
        max_missing_fraction: float = 0.10,
        cache_dir: str | Path | None = None,
        retry_attempts: int = 3,
        retry_delay_seconds: float = 2.0,
    ):
        self.max_forward_fill = max_forward_fill
        self.max_missing_fraction = max_missing_fraction
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.retry_attempts = retry_attempts
        self.retry_delay_seconds = retry_delay_seconds

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

        requested_tickers = list(dict.fromkeys(tickers))
        cached = self._load_cached_prices(requested_tickers, start, end, interval)
        if cached is not None:
            logger.info("Loaded cached price history for {} tickers", len(cached.columns.levels[0]))
            return cached

        frames = self._download_in_batches(requested_tickers, start, end, interval)

        if not frames:
            raise ValueError("No price data could be downloaded for the requested universe.")

        prices = pd.concat(frames, axis=1).sort_index(axis=1)
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        logger.info("Downloaded price history for {} tickers", len(prices.columns.levels[0]))
        self._save_cached_prices(prices, requested_tickers, start, end, interval)
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

    def _download_in_batches(
        self,
        tickers: list[str],
        start: str,
        end: str,
        interval: str,
    ) -> list[pd.DataFrame]:
        frames: list[pd.DataFrame] = []
        batched_raw = self._download_batch(tickers, start, end, interval)
        remaining: list[str] = []

        for ticker in tickers:
            formatted = self._extract_ticker_from_batch(batched_raw, ticker)
            if formatted is None or formatted.dropna(how="all").empty:
                remaining.append(ticker)
                continue

            cleaned = self._validate_and_clean(formatted, ticker)
            if cleaned is not None:
                frames.append(cleaned)

        if not frames and remaining:
            logger.warning("Yahoo returned no usable data for the batch. Falling back to Stooq.")
            return self._download_from_stooq(remaining, start, end)

        for ticker in remaining:
            raw = self._download_single_with_retry(ticker, start, end, interval)
            if raw is not None and not raw.empty:
                formatted = self._format_single_ticker_frame(raw, ticker)
                cleaned = self._validate_and_clean(formatted, ticker)
                if cleaned is not None:
                    frames.append(cleaned)
                    continue

            stooq_frame = self._download_single_from_stooq(ticker, start, end)
            if stooq_frame is None:
                logger.warning("No data returned for {}", ticker)
                continue

            cleaned = self._validate_and_clean(stooq_frame, ticker)
            if cleaned is not None:
                frames.append(cleaned)

        return frames

    def _download_batch(self, tickers: list[str], start: str, end: str, interval: str) -> pd.DataFrame:
        try:
            raw = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
                group_by="ticker",
            )
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            logger.warning("Batch download failed, falling back to selective retries: {}", exc)
            return pd.DataFrame()

        return raw if isinstance(raw, pd.DataFrame) else pd.DataFrame()

    def _download_single_with_retry(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str,
    ) -> pd.DataFrame | None:
        for attempt in range(1, self.retry_attempts + 1):
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
                logger.warning("Download attempt {}/{} failed for {}: {}", attempt, self.retry_attempts, ticker, exc)
                raw = pd.DataFrame()

            if not raw.empty:
                return raw

            if attempt < self.retry_attempts:
                time.sleep(self.retry_delay_seconds * attempt)

        return None

    def _download_from_stooq(self, tickers: list[str], start: str, end: str) -> list[pd.DataFrame]:
        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            stooq_frame = self._download_single_from_stooq(ticker, start, end)
            if stooq_frame is None:
                logger.warning("No data returned for {}", ticker)
                continue

            cleaned = self._validate_and_clean(stooq_frame, ticker)
            if cleaned is not None:
                frames.append(cleaned)
        return frames

    def _download_single_from_stooq(self, ticker: str, start: str, end: str) -> pd.DataFrame | None:
        if web is None:
            return None

        for symbol in self._stooq_symbol_candidates(ticker):
            try:
                raw = web.DataReader(symbol, "stooq", start, end)
            except Exception:  # pragma: no cover - network/runtime dependent
                continue

            if raw is None or raw.empty:
                continue

            ordered = raw.sort_index()
            formatted = self._format_stooq_frame(ordered, ticker)
            if formatted.dropna(how="all").empty:
                continue
            logger.info("Loaded {} from Stooq fallback using symbol {}", ticker, symbol)
            return formatted

        return None

    def _validate_and_clean(self, formatted: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
        missing_fraction = self._missing_fraction(formatted)
        if missing_fraction > self.max_missing_fraction:
            logger.warning("Dropping {} due to missing fraction {:.2%}", ticker, missing_fraction)
            return None

        return formatted.ffill(limit=self.max_forward_fill)

    def _extract_ticker_from_batch(self, batch_frame: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
        if batch_frame.empty:
            return None

        if isinstance(batch_frame.columns, pd.MultiIndex):
            level_zero = batch_frame.columns.get_level_values(0)
            level_one = batch_frame.columns.get_level_values(1)

            if ticker in level_zero:
                ticker_frame = batch_frame[ticker].copy()
                return self._format_single_ticker_frame(ticker_frame, ticker)

            if ticker in level_one:
                ticker_frame = batch_frame.xs(ticker, axis=1, level=1).copy()
                return self._format_single_ticker_frame(ticker_frame, ticker)

        if not batch_frame.empty:
            return self._format_single_ticker_frame(batch_frame.copy(), ticker)

        return None

    def _cache_path(self, tickers: list[str], start: str, end: str, interval: str) -> Path | None:
        if self.cache_dir is None:
            return None

        safe_name = "_".join(tickers).replace("^", "IDX_")
        filename = f"{safe_name}_{start}_{end}_{interval}.parquet"
        return self.cache_dir / filename

    def _load_cached_prices(
        self,
        tickers: list[str],
        start: str,
        end: str,
        interval: str,
    ) -> pd.DataFrame | None:
        cache_path = self._cache_path(tickers, start, end, interval)
        if cache_path is None or not cache_path.exists():
            return None

        return pd.read_parquet(cache_path)

    def _save_cached_prices(
        self,
        prices: pd.DataFrame,
        tickers: list[str],
        start: str,
        end: str,
        interval: str,
    ) -> None:
        cache_path = self._cache_path(tickers, start, end, interval)
        if cache_path is None:
            return

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache_path)

    def _format_single_ticker_frame(self, frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
        normalized = frame.copy()
        normalized.columns = [str(column) for column in normalized.columns]

        if "Adj Close" not in normalized.columns and "Close" in normalized.columns:
            normalized["Adj Close"] = normalized["Close"]

        normalized = normalized.reindex(columns=self.REQUIRED_FIELDS)
        normalized.columns = pd.MultiIndex.from_product([[ticker], normalized.columns])
        return normalized.sort_index()

    def _format_stooq_frame(self, frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
        normalized = frame.copy()
        normalized.columns = [str(column).title() for column in normalized.columns]
        if "Adj Close" not in normalized.columns and "Close" in normalized.columns:
            normalized["Adj Close"] = normalized["Close"]
        if "Volume" not in normalized.columns:
            normalized["Volume"] = np.nan
        normalized = normalized.reindex(columns=self.REQUIRED_FIELDS)
        normalized.columns = pd.MultiIndex.from_product([[ticker], normalized.columns])
        normalized.index = pd.to_datetime(normalized.index).tz_localize(None)
        return normalized.sort_index()

    @staticmethod
    def _stooq_symbol_candidates(ticker: str) -> list[str]:
        if ticker.startswith("^"):
            return [ticker]
        return [f"{ticker}.US", ticker]

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
