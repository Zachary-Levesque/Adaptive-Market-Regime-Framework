import numpy as np
import pandas as pd

from src.data.features import FeatureEngineer


def _price_frame() -> pd.DataFrame:
    index = pd.date_range("2023-01-02", periods=260, freq="B")
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    tickers = ["SPY", "QQQ", "HYG", "LQD", "^VIX"]
    columns = pd.MultiIndex.from_product([tickers, fields], names=["ticker", "field"])
    frame = pd.DataFrame(index=index, columns=columns, dtype=float)

    for i, ticker in enumerate(tickers):
        base = 100 + i * 10
        trend = np.linspace(base, base + 15, len(index))
        if ticker == "^VIX":
            trend = np.linspace(18, 24, len(index))
        for field in ["Open", "High", "Low", "Close", "Adj Close"]:
            frame[(ticker, field)] = trend
        frame[(ticker, "Volume")] = 1_000_000 + i * 1_000

    return frame


def test_compute_technical_features_creates_expected_columns():
    engineer = FeatureEngineer()
    features = engineer.compute_technical_features(_price_frame())

    assert ("SPY", "return_21d") in features.columns
    assert ("SPY", "momentum_12_1") in features.columns
    assert ("MARKET", "vix_level") in features.columns


def test_normalize_clips_cross_sectionally():
    engineer = FeatureEngineer()
    columns = pd.MultiIndex.from_tuples(
        [("A", "feature_x"), ("B", "feature_x"), ("C", "feature_x")],
        names=["ticker", "feature"],
    )
    data = pd.DataFrame([[1.0, 2.0, 100.0]], index=[pd.Timestamp("2024-01-01")], columns=columns)

    normalized = engineer.normalize(data)
    values = normalized.xs("feature_x", axis=1, level=1).iloc[0]
    assert values.max() <= 3.0
    assert values.min() >= -3.0


def test_compute_regime_features_uses_macro_and_credit_proxy():
    engineer = FeatureEngineer()
    prices = _price_frame()
    macro = pd.DataFrame(
        {"DGS10": [4.0] * len(prices.index), "DGS2": [3.5] * len(prices.index)},
        index=prices.index,
    )

    regime = engineer.compute_regime_features(prices, macro=macro)

    assert "yield_curve_slope" in regime.columns
    assert "credit_spread_proxy" in regime.columns
    assert np.isclose(regime["yield_curve_slope"].dropna().iloc[0], 0.5)
