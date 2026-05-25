import numpy as np
import pandas as pd

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

    expected = np.log(101.0 / 100.0)
    assert np.isclose(returns.loc[pd.Timestamp("2024-01-02"), "SPY"], expected)


def test_compute_returns_removes_large_outlier():
    ingester = MarketDataIngester()
    returns = ingester.compute_returns(_sample_prices())

    assert np.isnan(returns.iloc[14]["SPY"]) or np.isnan(returns.iloc[15]["SPY"])
