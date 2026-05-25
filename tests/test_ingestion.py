import numpy as np
import pandas as pd
from pathlib import Path

from src.data.ingestion import MarketDataIngester


def _sample_prices() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=30, freq="B")
    columns = pd.MultiIndex.from_product(
        [["SPY", "QQQ"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]],
        names=["ticker", "field"],
    )
    frame = pd.DataFrame(index=index, columns=columns, dtype=float)

    spy = np.linspace(100.0, 110.0, len(index))
    spy[15] = 250.0
    qqq = np.linspace(50.0, 55.0, len(index))

    for field in ["Open", "High", "Low", "Close", "Adj Close"]:
        frame[("SPY", field)] = spy
        frame[("QQQ", field)] = qqq
    frame[("SPY", "Volume")] = np.arange(1, len(index) + 1)
    frame[("QQQ", "Volume")] = np.arange(len(index), 0, -1)
    return frame


def test_compute_returns_uses_adjusted_close():
    ingester = MarketDataIngester()
    returns = ingester.compute_returns(_sample_prices())

    expected = np.log((100.0 + (10.0 / 29.0)) / 100.0)
    assert np.isclose(returns.loc[pd.Timestamp("2024-01-02"), "SPY"], expected)


def test_compute_returns_removes_large_outlier():
    ingester = MarketDataIngester()
    returns = ingester.compute_returns(_sample_prices())

    assert np.isnan(returns.iloc[14]["SPY"]) or np.isnan(returns.iloc[15]["SPY"])


def test_extract_ticker_from_batch_ticker_first_columns():
    ingester = MarketDataIngester()
    prices = _sample_prices()
    batch = prices.swaplevel(axis=1).sort_index(axis=1).swaplevel(axis=1).sort_index(axis=1)

    extracted = ingester._extract_ticker_from_batch(batch, "SPY")

    assert extracted is not None
    assert ("SPY", "Adj Close") in extracted.columns


def test_load_cached_prices_returns_saved_frame(tmp_path: Path):
    ingester = MarketDataIngester(cache_dir=tmp_path)
    prices = _sample_prices()

    ingester._save_cached_prices(prices, ["SPY", "QQQ"], "2024-01-01", "2024-02-01", "1d")
    loaded = ingester._load_cached_prices(["SPY", "QQQ"], "2024-01-01", "2024-02-01", "1d")

    assert loaded is not None
    assert loaded.equals(prices)
