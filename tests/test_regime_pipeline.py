from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RegimeConfig
from src.regime.pipeline import RegimeDetectionPipeline


def _synthetic_inputs():
    rng = np.random.default_rng(3)
    index = pd.date_range("2020-01-01", periods=240, freq="B")
    hidden_states = np.repeat([0, 1, 2, 3], 60)
    means = np.array(
        [
            [0.011, 0.09, 12.0, 0.03, 0.0, 1.2, 0.8, 0.1],
            [0.001, 0.04, 10.0, 0.01, 0.02, 0.8, 1.1, 0.0],
            [-0.005, 0.16, 19.0, -0.02, 0.08, -0.1, 0.2, -0.1],
            [-0.013, 0.30, 30.0, -0.05, 0.18, -0.8, -0.7, -0.2],
        ]
    )
    noise = np.column_stack(
        [
            rng.normal(0.0, 0.002, size=len(hidden_states)),
            rng.normal(0.0, 0.01, size=len(hidden_states)),
            rng.normal(0.0, 0.8, size=len(hidden_states)),
            rng.normal(0.0, 0.01, size=len(hidden_states)),
            rng.normal(0.0, 0.02, size=len(hidden_states)),
            rng.normal(0.0, 0.05, size=len(hidden_states)),
            rng.normal(0.0, 0.05, size=len(hidden_states)),
            rng.normal(0.0, 0.03, size=len(hidden_states)),
        ]
    )
    features = pd.DataFrame(
        means[hidden_states] + noise,
        index=index,
        columns=[
            "spy_return",
            "spy_volatility_21d",
            "vix_level",
            "vix_5d_change",
            "spy_momentum_63d",
            "cross_sectional_dispersion",
            "yield_curve_slope",
            "credit_spread_proxy",
        ],
    )

    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    columns = pd.MultiIndex.from_product([["SPY"], fields], names=["ticker", "field"])
    prices = pd.DataFrame(index=index, columns=columns, dtype=float)
    close = 100.0 * np.exp(features["spy_return"].cumsum())
    for field in ["Open", "High", "Low", "Close", "Adj Close"]:
        prices[("SPY", field)] = close
    prices[("SPY", "Volume")] = 1_000_000
    return features, prices


def test_regime_pipeline_build_persists_outputs(tmp_path: Path):
    features, prices = _synthetic_inputs()
    config = RegimeConfig(
        n_regimes=4,
        n_iter=50,
        covariance_type="full",
        regime_names={
            0: "Bull Trending",
            1: "Low-Vol Compression",
            2: "Bear Trending",
            3: "High-Vol Crisis",
        },
        n_restarts=3,
        model_path=tmp_path / "models" / "hmm.pkl",
        output_dir=tmp_path / "regimes",
        chart_path=tmp_path / "regimes" / "history.png",
    )

    artifacts = RegimeDetectionPipeline(config).build(features, prices, benchmark="SPY")

    assert set(artifacts.regime_labels["regime"].unique()) == {0, 1, 2, 3}
    assert np.allclose(artifacts.regime_probs.sum(axis=1).to_numpy(), 1.0)
    assert (tmp_path / "regimes" / "regime_labels.parquet").exists()
    assert (tmp_path / "regimes" / "regime_probs.parquet").exists()
    assert (tmp_path / "regimes" / "history.png").exists()
    assert (tmp_path / "models" / "hmm.pkl").exists()
