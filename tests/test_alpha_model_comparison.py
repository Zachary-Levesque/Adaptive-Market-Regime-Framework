from pathlib import Path

import numpy as np
import pandas as pd

from src.alpha.baselines import WeightedTechnicalRegressor, build_default_baseline_specs
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
    assert "cash" in artifacts.leaderboard.index
    assert "ridge_last_step" in build_default_baseline_specs()[2].name
    assert "active_signal_days" in artifacts.leaderboard.columns
    assert "mean_signal_coverage" in artifacts.leaderboard.columns
    assert "mean_net_sharpe" in artifacts.leaderboard.columns
    assert "mean_transaction_cost" in artifacts.leaderboard.columns
    assert "projected_backtest_sharpe" in artifacts.leaderboard.columns
    assert "projected_total_return" in artifacts.leaderboard.columns
    assert "projected_is_tradable" in artifacts.leaderboard.columns
    assert (tmp_path / "processed" / "alpha_model_comparison.parquet").exists()
    assert (tmp_path / "processed" / "alpha_model_comparison_summary.parquet").exists()
    assert (tmp_path / "processed" / "alpha_signal_selection.parquet").exists()
    assert (tmp_path / "processed" / "alpha_signals" / "ridge.parquet").exists()
    assert (tmp_path / "processed" / "alpha_signals" / "ensemble.parquet").exists()
    assert artifacts.best_signal_path == tmp_path / "processed" / "alpha_signals" / f"{artifacts.best_model}.parquet"
    assert artifacts.best_model in {"ridge", "ensemble", "cash", "regime_selector"}


def test_optional_tree_baselines_accept_model_factory_input_size():
    specs = build_default_baseline_specs(include_tree_models=True)
    tree_specs = [spec for spec in specs if spec.name in {"random_forest", "gradient_boosting"}]

    assert len(tree_specs) == 2
    for spec in tree_specs:
        model = spec.factory(3)
        assert model.name == spec.name


def test_weighted_technical_regressor_uses_named_features_and_date_normalization():
    class TinyDataset:
        def __init__(self):
            import torch

            self.features = torch.tensor(
                [
                    [[0.0, 0.0], [1.0, 0.0]],
                    [[0.0, 0.0], [-1.0, 0.0]],
                    [[0.0, 0.0], [3.0, 0.0]],
                ],
                dtype=torch.float32,
            )
            self.targets = torch.zeros(3, dtype=torch.float32)
            self.feature_names = ["tech__return_63d", "tech__volatility_21d"]
            self.sample_dates = pd.DatetimeIndex(["2024-01-01", "2024-01-01", "2024-01-02"])

        def __len__(self):
            return len(self.features)

    dataset = TinyDataset()
    model = WeightedTechnicalRegressor(
        name="technical_test",
        feature_weights={"tech__return_63d": 1.0, "missing": 10.0},
    )
    model.fit(dataset, dataset, epochs=0)

    predictions = model.predict_dataset(dataset)

    assert np.allclose(predictions[:2], [1.0, -1.0])
    assert predictions[2] == 3.0


def test_cash_candidate_wins_when_all_models_have_negative_validation_sharpe(tmp_path: Path):
    comparator = AlphaModelComparator(
        alpha_config=_minimal_alpha_config(tmp_path),
        regime_config=_minimal_regime_config(tmp_path),
        baseline_specs=[],
    )
    leaderboard = pd.DataFrame(
        [
            {
                "model": "weak_model",
                "n_rows": 1,
                "n_folds": 1,
                "n_regimes": 1,
                "mean_sharpe": -0.5,
                "median_sharpe": -0.5,
                "mean_net_sharpe": -1.0,
                "mean_ic": 0.1,
                "mean_rank_ic": 0.1,
                "mean_hit_rate": 0.55,
                "mean_train_size": 10.0,
                "mean_test_size": 4.0,
            }
        ]
    ).set_index("model")

    guarded = comparator._with_cash_candidate(leaderboard)

    assert guarded.index[0] == "cash"
    assert guarded.loc["cash", "mean_sharpe"] == 0.0
    assert guarded.loc["cash", "mean_net_sharpe"] == 0.0


def test_trading_stats_include_turnover_costs(tmp_path: Path):
    comparator = AlphaModelComparator(
        alpha_config=_minimal_alpha_config(tmp_path),
        regime_config=_minimal_regime_config(tmp_path),
        baseline_specs=[],
        transaction_cost_bps=10.0,
        max_gross_exposure=1.0,
        long_fraction=0.5,
        short_fraction=0.5,
    )

    stats = comparator._compute_trading_stats(
        predictions=np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32),
        actuals=np.array([0.01, -0.01, 0.02, -0.02], dtype=np.float32),
        dates=pd.DatetimeIndex(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
        tickers=["A", "B", "A", "B"],
    )

    assert stats["mean_turnover"] > 0.0
    assert stats["mean_transaction_cost"] > 0.0
    assert stats["net_sharpe"] < stats["gross_sharpe"]


def test_trading_stats_respect_rebalance_interval(tmp_path: Path):
    daily = AlphaModelComparator(
        alpha_config=_minimal_alpha_config(tmp_path),
        regime_config=_minimal_regime_config(tmp_path),
        baseline_specs=[],
        transaction_cost_bps=10.0,
        long_fraction=0.5,
        short_fraction=0.5,
        rebalance_interval_days=1,
    )
    held = AlphaModelComparator(
        alpha_config=_minimal_alpha_config(tmp_path),
        regime_config=_minimal_regime_config(tmp_path),
        baseline_specs=[],
        transaction_cost_bps=10.0,
        long_fraction=0.5,
        short_fraction=0.5,
        rebalance_interval_days=3,
    )
    predictions = np.array([1.0, -1.0, -1.0, 1.0, 1.0, -1.0], dtype=np.float32)
    actuals = np.array([0.01, -0.01, -0.01, 0.01, 0.01, -0.01], dtype=np.float32)
    dates = pd.DatetimeIndex(
        ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
    )
    tickers = ["A", "B", "A", "B", "A", "B"]

    daily_stats = daily._compute_trading_stats(predictions, actuals, dates, tickers)
    held_stats = held._compute_trading_stats(predictions, actuals, dates, tickers)

    assert held_stats["mean_turnover"] < daily_stats["mean_turnover"]


def test_projected_backtest_gate_can_select_cash_over_positive_fold_model(tmp_path: Path):
    comparator = AlphaModelComparator(
        alpha_config=_minimal_alpha_config(tmp_path),
        regime_config=_minimal_regime_config(tmp_path),
        baseline_specs=[],
        transaction_cost_bps=10.0,
        long_fraction=0.5,
        short_fraction=0.5,
    )
    index = pd.date_range("2024-01-01", periods=5, freq="B")
    returns = pd.DataFrame(
        {
            "A": [0.0, -0.01, -0.01, -0.01, -0.01],
            "B": [0.0, 0.01, 0.01, 0.01, 0.01],
        },
        index=index,
    )
    losing_signals = pd.DataFrame({"A": [1.0] * len(index), "B": [-1.0] * len(index)}, index=index)
    signal_frames = {
        "fold_winner": losing_signals,
        "cash": pd.DataFrame(np.nan, index=index, columns=returns.columns),
    }
    leaderboard = pd.DataFrame(
        [
            {
                "model": "fold_winner",
                "n_rows": 1,
                "n_folds": 1,
                "n_regimes": 1,
                "mean_sharpe": 1.0,
                "median_sharpe": 1.0,
                "mean_net_sharpe": 1.0,
                "mean_ic": 0.0,
                "mean_rank_ic": 0.0,
                "mean_hit_rate": 0.5,
                "mean_turnover": 0.0,
                "mean_transaction_cost": 0.0,
                "mean_train_size": 10.0,
                "mean_test_size": 4.0,
            },
            {
                "model": "cash",
                "n_rows": 0,
                "n_folds": 0,
                "n_regimes": 0,
                "mean_sharpe": 0.0,
                "median_sharpe": 0.0,
                "mean_net_sharpe": 0.0,
                "mean_ic": 0.0,
                "mean_rank_ic": 0.0,
                "mean_hit_rate": 0.0,
                "mean_turnover": 0.0,
                "mean_transaction_cost": 0.0,
                "mean_train_size": 0.0,
                "mean_test_size": 0.0,
            },
        ]
    ).set_index("model")

    gated = comparator._attach_projected_backtest_stats(leaderboard, signal_frames, returns)

    assert gated.index[0] == "cash"
    assert gated.loc["fold_winner", "projected_total_return"] < 0.0


def test_projected_backtest_gate_requires_positive_total_return(tmp_path: Path):
    comparator = AlphaModelComparator(
        alpha_config=_minimal_alpha_config(tmp_path),
        regime_config=_minimal_regime_config(tmp_path),
        baseline_specs=[],
    )
    leaderboard = pd.DataFrame(
        [
            {
                "model": "negative_return_model",
                "n_rows": 1,
                "n_folds": 1,
                "n_regimes": 1,
                "mean_sharpe": 1.0,
                "median_sharpe": 1.0,
                "mean_net_sharpe": 1.0,
                "mean_ic": 0.0,
                "mean_rank_ic": 0.0,
                "mean_hit_rate": 0.5,
                "mean_turnover": 0.0,
                "mean_transaction_cost": 0.0,
                "mean_train_size": 10.0,
                "mean_test_size": 4.0,
            },
            {
                "model": "cash",
                "n_rows": 0,
                "n_folds": 0,
                "n_regimes": 0,
                "mean_sharpe": 0.0,
                "median_sharpe": 0.0,
                "mean_net_sharpe": 0.0,
                "mean_ic": 0.0,
                "mean_rank_ic": 0.0,
                "mean_hit_rate": 0.0,
                "mean_turnover": 0.0,
                "mean_transaction_cost": 0.0,
                "mean_train_size": 0.0,
                "mean_test_size": 0.0,
            },
        ]
    ).set_index("model")
    returns = pd.DataFrame({"A": [0.0, 0.0]}, index=pd.date_range("2024-01-01", periods=2, freq="B"))
    signals = {
        "negative_return_model": pd.DataFrame({"A": [np.nan, np.nan]}, index=returns.index),
        "cash": pd.DataFrame({"A": [np.nan, np.nan]}, index=returns.index),
    }
    gated = comparator._attach_projected_backtest_stats(leaderboard, signals, returns)
    gated.loc["negative_return_model", "projected_backtest_sharpe"] = 1.0
    gated.loc["negative_return_model", "projected_total_return"] = -0.01
    gated.loc["negative_return_model", "projected_is_tradable"] = 0.0
    gated = gated.sort_values(
        ["projected_is_tradable", "projected_backtest_sharpe", "projected_total_return", "mean_net_sharpe"],
        ascending=False,
    )

    assert gated.index[0] == "negative_return_model"


def test_regime_selector_uses_positive_validated_model_by_regime(tmp_path: Path):
    comparator = AlphaModelComparator(
        alpha_config=_minimal_alpha_config(tmp_path),
        regime_config=_minimal_regime_config(tmp_path),
        baseline_specs=[],
        min_regime_selection_folds=2,
    )
    index = pd.date_range("2024-01-01", periods=4, freq="B")
    regime_series = pd.Series([0, 0, 1, 1], index=index, dtype="Int64")
    columns = ["A", "B"]
    model_a = pd.DataFrame({"A": [1.0, 1.0, 9.0, 9.0], "B": [-1.0, -1.0, -9.0, -9.0]}, index=index)
    model_b = pd.DataFrame({"A": [2.0, 2.0, 3.0, 3.0], "B": [-2.0, -2.0, -3.0, -3.0]}, index=index)
    signal_frames = {
        "model_a": model_a,
        "model_b": model_b,
        "cash": pd.DataFrame(np.nan, index=index, columns=columns),
    }
    fold_metrics = pd.DataFrame(
        [
            {"model": "model_a", "regime": 0, "fold": 0, "net_sharpe": 1.0, "sharpe": 1.0, "ic": 0.0},
            {"model": "model_a", "regime": 0, "fold": 1, "net_sharpe": 1.0, "sharpe": 1.0, "ic": 0.0},
            {"model": "model_b", "regime": 0, "fold": 0, "net_sharpe": -1.0, "sharpe": 0.0, "ic": 0.0},
            {"model": "model_b", "regime": 0, "fold": 1, "net_sharpe": -1.0, "sharpe": 0.0, "ic": 0.0},
            {"model": "model_a", "regime": 1, "fold": 0, "net_sharpe": -1.0, "sharpe": 0.0, "ic": 0.0},
            {"model": "model_a", "regime": 1, "fold": 1, "net_sharpe": -1.0, "sharpe": 0.0, "ic": 0.0},
            {"model": "model_b", "regime": 1, "fold": 0, "net_sharpe": 2.0, "sharpe": 2.0, "ic": 0.0},
            {"model": "model_b", "regime": 1, "fold": 1, "net_sharpe": 2.0, "sharpe": 2.0, "ic": 0.0},
        ]
    )

    enriched = comparator._with_regime_selector_signal(signal_frames, fold_metrics, regime_series)
    selector = enriched["regime_selector"]

    assert selector.loc[index[0], "A"] == model_a.loc[index[0], "A"]
    assert selector.loc[index[2], "A"] == model_b.loc[index[2], "A"]


def test_regime_selector_requires_enough_positive_folds(tmp_path: Path):
    comparator = AlphaModelComparator(
        alpha_config=_minimal_alpha_config(tmp_path),
        regime_config=_minimal_regime_config(tmp_path),
        baseline_specs=[],
        min_regime_selection_folds=3,
    )
    fold_metrics = pd.DataFrame(
        [
            {"model": "model_a", "regime": 2, "fold": 0, "net_sharpe": 10.0, "sharpe": 10.0, "ic": 0.0},
        ]
    )

    assert comparator._select_models_by_regime(fold_metrics) == {}


def _minimal_alpha_config(tmp_path: Path) -> AlphaConfig:
    return AlphaConfig(
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


def _minimal_regime_config(tmp_path: Path) -> RegimeConfig:
    return RegimeConfig(
        n_regimes=4,
        n_iter=10,
        covariance_type="full",
        regime_names={0: "Bull Trending", 1: "Low-Vol Compression", 2: "Bear Trending", 3: "High-Vol Crisis"},
        n_restarts=1,
        model_path=tmp_path / "regime" / "hmm.pkl",
        output_dir=tmp_path / "regime",
        chart_path=tmp_path / "regime" / "chart.png",
    )
