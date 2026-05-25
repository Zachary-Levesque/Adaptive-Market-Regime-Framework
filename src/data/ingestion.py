"""Market data ingestion utilities."""

from __future__ import annotations

from pathlib import Path
from io import StringIO
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import requests

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

    def __init__(
        self,
        max_forward_fill: int = 5,
        max_missing_fraction: float = 0.10,
        cache_dir: str | Path | None = None,
        local_data_dir: str | Path | None = None,
        retry_attempts: int = 3,
        retry_delay_seconds: float = 2.0,
    ):
        self.max_forward_fill = max_forward_fill
        self.max_missing_fraction = max_missing_fraction
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.local_data_dir = Path(local_data_dir) if local_data_dir else None
        self.retry_attempts = retry_attempts
        self.retry_delay_seconds = retry_delay_seconds

    @dataclass(frozen=True)
    class LocalTickerStatus:
        ticker: str
        found: bool
        path: Path | None

    def inspect_local_data(self, tickers: Iterable[str]) -> list[LocalTickerStatus]:
        """Report which tickers can be resolved from the local raw-data directory."""
        statuses: list[MarketDataIngester.LocalTickerStatus] = []
        for ticker in list(dict.fromkeys(tickers)):
            path = self._find_local_price_file(ticker)
            statuses.append(self.LocalTickerStatus(ticker=ticker, found=path is not None, path=path))
        return statuses

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

        local_frames, missing_tickers = self._load_local_price_frames(requested_tickers, start, end)
        if local_frames and not missing_tickers:
            prices = pd.concat(local_frames, axis=1).sort_index(axis=1)
            prices.index = pd.to_datetime(prices.index).tz_localize(None)
            logger.info("Loaded local price history for {} tickers", len(prices.columns.levels[0]))
            self._save_cached_prices(prices, requested_tickers, start, end, interval)
            return prices

        frames = list(local_frames)
        if missing_tickers:
            logger.info(
                "Missing {} tickers locally; attempting remote download for the remainder",
                len(missing_tickers),
            )
        remote_frames = self._download_in_batches(missing_tickers or requested_tickers, start, end, interval)
        frames.extend(remote_frames)

        if not frames:
            local_hint = self._format_local_data_hint(requested_tickers)
            raise ValueError(
                "No price data could be loaded. Remote providers failed, and no local raw files were found in "
                f"{self.local_data_dir or 'the configured raw-data directory'}. {local_hint}"
            )

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
        for symbol in self._stooq_symbol_candidates(ticker):
            try:
                raw = self._fetch_stooq_csv(symbol, start, end)
            except Exception as exc:  # pragma: no cover - network/runtime dependent
                logger.warning("Stooq request failed for {} using symbol {}: {}", ticker, symbol, exc)
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

    def _load_local_price_frames(
        self,
        tickers: list[str],
        start: str,
        end: str,
    ) -> tuple[list[pd.DataFrame], list[str]]:
        if self.local_data_dir is None or not self.local_data_dir.exists():
            return [], tickers

        frames: list[pd.DataFrame] = []
        missing: list[str] = []
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        for ticker in tickers:
            path = self._find_local_price_file(ticker)
            if path is None:
                missing.append(ticker)
                continue

            formatted = self._load_local_price_file(path, ticker, start_ts, end_ts)
            if formatted is None or formatted.empty:
                missing.append(ticker)
                continue

            cleaned = self._validate_and_clean(formatted, ticker)
            if cleaned is None:
                missing.append(ticker)
                continue

            logger.info("Loaded {} from local raw file {}", ticker, path)
            frames.append(cleaned)

        return frames, missing

    def _format_local_data_hint(self, tickers: list[str]) -> str:
        sample_candidates = []
        for ticker in tickers[:3]:
            sample_candidates.extend(self._local_filename_candidates(ticker)[:2])
        sample_text = ", ".join(sample_candidates[:6])
        return (
            "Expected local files with a Date column under the raw-data directory, for example: "
            f"{sample_text}."
        )

    def _find_local_price_file(self, ticker: str) -> Path | None:
        if self.local_data_dir is None:
            return None

        for candidate in self._local_filename_candidates(ticker):
            matches = sorted(self.local_data_dir.rglob(candidate))
            if matches:
                return matches[0]

        return None

    def _load_local_price_file(
        self,
        path: Path,
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame | None:
        frame = pd.read_csv(path)
        if frame.empty:
            return None

        lowered = {column.lower(): column for column in frame.columns}
        if "date" not in lowered:
            raise ValueError(f"Local price file {path} is missing a Date column.")

        rename_map = {}
        for canonical in ["date", "open", "high", "low", "close", "adj close", "adj_close", "volume"]:
            if canonical in lowered:
                rename_map[lowered[canonical]] = canonical
        frame = frame.rename(columns=rename_map)

        if "adj close" not in frame.columns and "close" in frame.columns:
            frame["adj close"] = frame["close"]
        if "volume" not in frame.columns:
            frame["volume"] = np.nan

        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date").sort_index()
        frame = frame.loc[(frame.index >= start) & (frame.index <= end)]

        normalized = frame.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "adj close": "Adj Close",
                "volume": "Volume",
            }
        )
        normalized = normalized.reindex(columns=self.REQUIRED_FIELDS)
        normalized.columns = pd.MultiIndex.from_product([[ticker], normalized.columns])
        normalized.index = pd.to_datetime(normalized.index).tz_localize(None)
        return normalized

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
        return [f"{ticker}.US", f"{ticker}.us", ticker]

    @staticmethod
    def _local_filename_candidates(ticker: str) -> list[str]:
        base = ticker.lower().replace("^", "")
        return [
            f"{base}.us.txt",
            f"{base}.us.csv",
            f"{base}.txt",
            f"{base}.csv",
        ]

    def _fetch_stooq_csv(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        response = requests.get(
            "https://stooq.com/q/d/l/",
            params={
                "s": symbol,
                "i": "d",
                "d1": start_ts.strftime("%Y%m%d"),
                "d2": end_ts.strftime("%Y%m%d"),
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        response.raise_for_status()

        body = response.text.strip()
        if not body or body.lower() == "no data":
            return pd.DataFrame()

        frame = pd.read_csv(StringIO(body))
        if frame.empty:
            return frame

        if "Date" not in frame.columns:
            raise ValueError(f"Unexpected Stooq response columns for {symbol}: {list(frame.columns)}")

        frame["Date"] = pd.to_datetime(frame["Date"])
        frame = frame.set_index("Date").sort_index()
        return frame

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
