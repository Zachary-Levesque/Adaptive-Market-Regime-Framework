from pathlib import Path

import pandas as pd

from src.config import DataConfig
from src.data.pipeline import DataPipeline


class StubIngester:
    def __init__(self, prices: pd.DataFrame, returns: pd.DataFrame):
        self._prices = prices
        self._returns = returns
        self.saved_paths: list[Path] = []
        self.download_calls: list[list[str]] = []

    def download_prices(self, tickers, start, end):
        self.download_calls.append(list(tickers))
        return self._prices

    def compute_returns(self, prices):
        return self._returns

    def save(self, data, path):
        self.saved_paths.append(Path(path))


class StubFactorLoader:
    def __init__(self, ff5: pd.DataFrame, macro: pd.DataFrame):
        self._ff5 = ff5
        self._macro = macro

    def download_ff5(self):
        return self._ff5

    def align_with_returns(self, factors, returns):
        return factors.reindex(returns.index).ffill()

    def download_macro_series(self, series_map, start, end):
        return self._macro


class StubFeatureEngineer:
    def compute_technical_features(self, prices, vix=None):
        return pd.DataFrame(
            {("SPY", "feature_x"): [1.0, 2.0, 3.0]},
            index=prices.index,
        )

    def normalize(self, features):
        return features * 10.0

    def compute_regime_features(self, prices, macro, benchmark, vix=None):
        return pd.DataFrame(
            {
                "spy_return": [0.1, 0.2, 0.3],
                "yield_curve_slope": macro["DGS10"] - macro["DGS2"],
            },
            index=prices.index,
        )


def test_pipeline_build_persists_all_expected_outputs(tmp_path: Path):
    index = pd.date_range("2024-01-01", periods=3, freq="B")
    columns = pd.MultiIndex.from_product(
        [["SPY"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]],
        names=["ticker", "field"],
    )
    prices = pd.DataFrame(1.0, index=index, columns=columns)
    returns = pd.DataFrame({"SPY": [0.01, 0.02, 0.03]}, index=index)
    ff5 = pd.DataFrame({"Mkt-RF": [0.1, 0.2, 0.3]}, index=index)
    macro = pd.DataFrame({"DGS10": [4.0, 4.1, 4.2], "DGS2": [3.5, 3.6, 3.7]}, index=index)

    config = DataConfig(
        universe=["SPY", "^VIX"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        benchmark="SPY",
        cache_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        local_data_dir=tmp_path / "raw",
        allow_remote_downloads=False,
    )
    ingester = StubIngester(prices=prices, returns=returns)
    factor_loader = StubFactorLoader(ff5=ff5, macro=macro)
    feature_engineer = StubFeatureEngineer()

    pipeline = DataPipeline(
        config=config,
        ingester=ingester,
        factor_loader=factor_loader,
        feature_engineer=feature_engineer,
    )
    artifacts = pipeline.build()

    assert artifacts.prices.equals(prices)
    assert artifacts.returns.equals(returns)
    assert artifacts.technical_features.iloc[0, 0] == 10.0
    assert ingester.download_calls == [["SPY"]]
    assert len(ingester.saved_paths) == 6
    assert (tmp_path / "processed" / "prices.parquet") in ingester.saved_paths
    assert (tmp_path / "processed" / "regime_features.parquet") in ingester.saved_paths
