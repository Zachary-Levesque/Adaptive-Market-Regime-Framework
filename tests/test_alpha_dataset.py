import numpy as np
import pandas as pd

from src.alpha.dataset import RegimeDataset


def _alpha_inputs():
    index = pd.date_range("2021-01-01", periods=90, freq="B")
    tickers = ["AAA", "BBB"]
    feature_names = ["return_1d", "volatility_21d"]
    columns = pd.MultiIndex.from_product([tickers + ["MARKET"], feature_names], names=["ticker", "feature"])
    features = pd.DataFrame(index=index, columns=columns, dtype=float)

    for offset, ticker in enumerate(tickers):
        features[(ticker, "return_1d")] = np.linspace(0.0 + offset, 1.0 + offset, len(index))
        features[(ticker, "volatility_21d")] = np.linspace(1.0 + offset, 2.0 + offset, len(index))
    features[("MARKET", "return_1d")] = np.linspace(-0.1, 0.2, len(index))
    features[("MARKET", "volatility_21d")] = np.linspace(0.2, 0.4, len(index))

    returns = pd.DataFrame(
        {
            "AAA": np.sin(np.linspace(0, 3, len(index))) * 0.01,
            "BBB": np.cos(np.linspace(0, 3, len(index))) * 0.01,
        },
        index=index,
    )
    factors = pd.DataFrame(
        {
            "Mkt-RF": np.linspace(0.0, 0.02, len(index)),
            "SMB": np.linspace(0.01, -0.01, len(index)),
        },
        index=index,
    )
    regime_labels = pd.DataFrame({"regime": [0] * 50 + [1] * 40}, index=index)
    return features, returns, factors, regime_labels


def test_regime_dataset_builds_sequences_and_augments():
    features, returns, factors, regime_labels = _alpha_inputs()
    dataset = RegimeDataset(
        features=features,
        returns=returns,
        regime_labels=regime_labels,
        target_regime=0,
        factors=factors,
        sequence_length=10,
        min_samples=120,
        augment_noise_std=0.001,
    )

    assert len(dataset) >= 120
    assert dataset.features.shape[1] == 10
    assert dataset.input_size > 0
    assert set(dataset.sample_tickers).issubset({"AAA", "BBB"})
