import numpy as np
import pandas as pd
from pathlib import Path

from src.data.ingestion import MarketDataIngester
import src.data.ingestion as ingestion_module


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


def test_fetch_stooq_csv_parses_valid_csv(monkeypatch):
    ingester = MarketDataIngester()

    csv_body = "\n".join(
        [
            "Date,Open,High,Low,Close,Volume",
            "2024-01-02,100,101,99,100.5,1000",
            "2024-01-03,101,102,100,101.5,1100",
        ]
    )

    class DummyResponse:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        ingestion_module.requests,
        "get",
        lambda *args, **kwargs: DummyResponse(csv_body),
    )

    frame = ingester._fetch_stooq_csv("AAPL.US", "2024-01-01", "2024-01-31")

    assert not frame.empty
    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_load_local_price_frames_finds_stooq_style_txt(tmp_path: Path):
    raw_dir = tmp_path / "raw" / "stooq" / "us" / "nasdaq stocks" / "1"
    raw_dir.mkdir(parents=True)
    (raw_dir / "aapl.us.txt").write_text(
        "\n".join(
            [
                "Date,Open,High,Low,Close,Volume",
                "2024-01-02,100,101,99,100.5,1000",
                "2024-01-03,101,102,100,101.5,1100",
            ]
        ),
        encoding="utf-8",
    )

    ingester = MarketDataIngester(local_data_dir=tmp_path / "raw")
    frames, missing = ingester._load_local_price_frames(["AAPL"], "2024-01-01", "2024-01-31")

    assert missing == []
    assert len(frames) == 1
    assert ("AAPL", "Adj Close") in frames[0].columns


def test_inspect_local_data_reports_found_and_missing(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "spy.csv").write_text(
        "\n".join(
            [
                "Date,Open,High,Low,Close,Volume",
                "2024-01-02,100,101,99,100.5,1000",
            ]
        ),
        encoding="utf-8",
    )

    ingester = MarketDataIngester(local_data_dir=raw_dir)
    statuses = ingester.inspect_local_data(["SPY", "QQQ"])

    assert statuses[0].ticker == "SPY"
    assert statuses[0].found is True
    assert statuses[1].ticker == "QQQ"
    assert statuses[1].found is False


def test_download_prices_uses_local_files_without_yfinance(tmp_path: Path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "spy.csv").write_text(
        "\n".join(
            [
                "Date,Open,High,Low,Close,Volume",
                "2024-01-02,100,101,99,100.5,1000",
                "2024-01-03,101,102,100,101.5,1100",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(ingestion_module, "yf", None)
    ingester = MarketDataIngester(local_data_dir=raw_dir, allow_remote_downloads=False)
    prices = ingester.download_prices(["SPY"], "2024-01-01", "2024-01-31")

    assert not prices.empty
    assert ("SPY", "Adj Close") in prices.columns
