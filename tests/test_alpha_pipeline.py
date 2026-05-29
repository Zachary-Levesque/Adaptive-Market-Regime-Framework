from pathlib import Path

import numpy as np
import pandas as pd

from src.alpha.pipeline import AlphaPipeline
from src.config import AlphaConfig, RegimeConfig


def _pipeline_inputs():
    index = pd.date_range("2021-01-01", periods=140, freq="B")
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    technical_features = {}
    for ticker_idx, ticker in enumerate(tickers):
        technical_features[(ticker, "return_1d")] = np.sin(np.linspace(0, 6, len(index)) + ticker_idx) * 0.1
        technical_features[(ticker, "volatility_21d")] = 0.2 + 0.01 * ticker_idx + np.linspace(0, 0.05, len(index))
    technical_features[("MARKET", "vix_level")] = np.linspace(18, 24, len(index))
    technical_features[("MARKET", "vix_5d_change")] = np.linspace(-0.05, 0.05, len(index))
    features = pd.DataFrame(technical_features, index=index)
    features.columns = pd.MultiIndex.from_tuples(features.columns, names=["ticker", "feature"])

    returns = pd.DataFrame(
        {
            ticker: np.sin(np.linspace(0, 10, len(index)) + idx) * 0.01
            for idx, ticker in enumerate(tickers)
        },
        index=index,
    )
    factors = pd.DataFrame(
        {
            "Mkt-RF": np.linspace(0.0, 0.02, len(index)),
            "SMB": np.linspace(0.01, -0.01, len(index)),
            "HML": np.linspace(-0.005, 0.005, len(index)),
        },
        index=index,
    )
    regime_labels = pd.DataFrame({"regime": [0] * 70 + [1] * 70}, index=index)
    return features, returns, factors, regime_labels


def test_alpha_pipeline_build_persists_signals_and_models(tmp_path: Path):
    features, returns, factors, regime_labels = _pipeline_inputs()
    alpha_config = AlphaConfig(
        hidden_size=16,
        num_layers=1,
        dropout=0.1,
        sequence_length=15,
        batch_size=16,
        epochs=1,
        learning_rate=0.001,
        train_window=60,
        test_window=20,
        step_size=20,
        model_dir=tmp_path / "models",
        signals_path=tmp_path / "processed" / "alpha_signals.parquet",
        metrics_path=tmp_path / "processed" / "alpha_metrics.parquet",
        diagnostics_path=tmp_path / "processed" / "alpha_diagnostics.parquet",
        comparison_path=tmp_path / "processed" / "alpha_model_comparison.parquet",
        signals_dir=tmp_path / "processed" / "alpha_signals",
        selection_path=tmp_path / "processed" / "alpha_signal_selection.parquet",
        validation_fraction=0.2,
        min_samples_per_regime=30,
        augment_noise_std=0.001,
        weight_decay=1e-5,
        patience=2,
        device="cpu",
    )
    regime_config = RegimeConfig(
        n_regimes=4,
        n_iter=10,
        covariance_type="full",
        regime_names={0: "Bull Trending", 1: "Low-Vol Compression", 2: "Bear Trending", 3: "High-Vol Crisis"},
        n_restarts=1,
        model_path=tmp_path / "regime" / "hmm.pkl",
        output_dir=tmp_path / "regime",
        chart_path=tmp_path / "regime" / "chart.png",
    )

    artifacts = AlphaPipeline(alpha_config, regime_config).build(
        technical_features=features,
        returns=returns,
        factors=factors,
        regime_labels=regime_labels,
        epochs_override=1,
        run_validation=False,
    )

    assert artifacts.trained_regimes == [0, 1]
    assert artifacts.alpha_signals.notna().sum().sum() > 0
    assert (tmp_path / "models" / "regime_0" / "metadata.json").exists()
    assert alpha_config.signals_path.exists()
    assert alpha_config.metrics_path.exists()
