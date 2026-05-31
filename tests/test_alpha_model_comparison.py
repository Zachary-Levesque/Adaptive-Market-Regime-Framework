from pathlib import Path

import numpy as np
import pandas as pd

from src.alpha.baselines import build_default_baseline_specs
from src.alpha.model_comparison import AlphaModelComparator
from src.config import AlphaConfig, RegimeConfig


def _comparison_inputs():
    index = pd.date_range("2024-01-01", periods=40, freq="B")
    tickers = ["AAA", "BBB"]

    technical_features = {}
    for ticker_idx, ticker in enumerate(tickers):
        technical_features[(ticker, "return_1d")] = np.sin(np.linspace(0, 4, len(index)) + ticker_idx) * 0.05
        technical_features[(ticker, "volatility_21d")] = 0.15 + 0.01 * ticker_idx + np.linspace(0, 0.03, len(index))
    technical_features[("MARKET", "vix_level")] = np.linspace(16, 24, len(index))
    technical_features[("MARKET", "vix_5d_change")] = np.linspace(-0.04, 0.04, len(index))
    features = pd.DataFrame(technical_features, index=index)
    features.columns = pd.MultiIndex.from_tuples(features.columns, names=["ticker", "feature"])

    returns = pd.DataFrame(
        {
            "AAA": np.sin(np.linspace(0, 4, len(index))) * 0.01,
            "BBB": np.cos(np.linspace(0, 4, len(index))) * 0.01,
        },
        index=index,
    )
    factors = pd.DataFrame(
        {
            "Mkt-RF": np.linspace(0.0, 0.01, len(index)),
            "SMB": np.linspace(0.01, -0.01, len(index)),
            "HML": np.linspace(-0.005, 0.005, len(index)),
        },
        index=index,
    )
    regime_labels = pd.DataFrame({"regime": [0] * 20 + [1] * 20}, index=index)
    return features, returns, factors, regime_labels


def test_alpha_model_comparison_builds_leaderboard_and_saves_artifacts(tmp_path: Path):
    features, returns, factors, regime_labels = _comparison_inputs()
    alpha_config = AlphaConfig(
        hidden_size=8,
        num_layers=1,
        dropout=0.1,
        sequence_length=4,
        batch_size=8,
        epochs=1,
        learning_rate=0.001,
        train_window=10,
        test_window=4,
        step_size=4,
        model_dir=tmp_path / "models",
        signals_path=tmp_path / "processed" / "alpha_signals.parquet",
        metrics_path=tmp_path / "processed" / "alpha_metrics.parquet",
        diagnostics_path=tmp_path / "processed" / "alpha_diagnostics.parquet",
        comparison_path=tmp_path / "processed" / "alpha_model_comparison.parquet",
        signals_dir=tmp_path / "processed" / "alpha_signals",
        selection_path=tmp_path / "processed" / "alpha_signal_selection.parquet",
        validation_fraction=0.25,
        min_samples_per_regime=8,
        augment_noise_std=0.0,
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

    ensemble_metrics = pd.DataFrame(
        {
            "regime": [0, 1],
            "fold": [0, 1],
            "n_train": [10, 10],
            "n_test": [4, 4],
            "sharpe": [0.12, 0.08],
            "ic": [0.02, 0.01],
            "rank_ic": [0.03, 0.02],
            "hit_rate": [0.55, 0.52],
            "lstm_weight": [0.6, 0.6],
            "transformer_weight": [0.4, 0.4],
        }
    )
    alpha_config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble_metrics.to_parquet(alpha_config.metrics_path)

    comparator = AlphaModelComparator(
        alpha_config=alpha_config,
        regime_config=regime_config,
        baseline_specs=build_default_baseline_specs()[:1],
    )
    artifacts = comparator.build(
        technical_features=features,
        returns=returns,
        factors=factors,
        regime_labels=regime_labels,
        epochs_override=1,
        include_ensemble=True,
    )

    assert not artifacts.fold_metrics.empty
    assert "ridge" in artifacts.leaderboard.index
    assert "ensemble" in artifacts.leaderboard.index
    assert "ridge_last_step" in build_default_baseline_specs()[2].name
    assert "active_signal_days" in artifacts.leaderboard.columns
    assert "mean_signal_coverage" in artifacts.leaderboard.columns
    assert (tmp_path / "processed" / "alpha_model_comparison.parquet").exists()
    assert (tmp_path / "processed" / "alpha_model_comparison_summary.parquet").exists()
    assert (tmp_path / "processed" / "alpha_signal_selection.parquet").exists()
    assert (tmp_path / "processed" / "alpha_signals" / "ridge.parquet").exists()
    assert (tmp_path / "processed" / "alpha_signals" / "ensemble.parquet").exists()
    assert artifacts.best_signal_path == tmp_path / "processed" / "alpha_signals" / f"{artifacts.best_model}.parquet"
    assert artifacts.best_model in {"ridge", "ensemble"}


def test_optional_tree_baselines_accept_model_factory_input_size():
    specs = build_default_baseline_specs(include_tree_models=True)
    tree_specs = [spec for spec in specs if spec.name in {"random_forest", "gradient_boosting"}]

    assert len(tree_specs) == 2
    for spec in tree_specs:
        model = spec.factory(3)
        assert model.name == spec.name
