import numpy as np
import pandas as pd

from src.risk.backtester import AMRFBacktester, BacktestConfig


def test_backtester_applies_prior_day_weights_and_transaction_costs():
    index = pd.date_range("2024-01-01", periods=5, freq="B")
    returns = pd.DataFrame(
        {
            "A": [0.00, 0.10, 0.00, 0.00, 0.00],
            "B": [0.00, -0.10, 0.00, 0.00, 0.00],
            "SPY": [0.00, 0.01, 0.01, 0.01, 0.01],
        },
        index=index,
    )
    signals = pd.DataFrame(
        {
            "A": [1.0, 1.0, 1.0, 1.0, 1.0],
            "B": [-1.0, -1.0, -1.0, -1.0, -1.0],
            "SPY": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=index,
    )

    backtester = AMRFBacktester(
        returns=returns,
        alpha_signals=signals,
        config=BacktestConfig(
            max_gross_exposure=1.0,
            long_fraction=1 / 3,
            short_fraction=1 / 3,
            transaction_cost_bps=0.0,
            benchmark="SPY",
        ),
    )
    artifacts = backtester.run()

    assert np.isclose(artifacts.daily_results.loc[index[0], "strategy_return"], 0.0)
    assert artifacts.weights.loc[index[1], "A"] > 0
    assert artifacts.weights.loc[index[1], "B"] < 0
    assert np.isclose(artifacts.daily_results.loc[index[1], "strategy_return_gross"], 0.10)
    assert "strategy" in artifacts.performance_report.index


def test_backtester_saves_outputs(tmp_path):
    index = pd.date_range("2024-01-01", periods=4, freq="B")
    returns = pd.DataFrame({"A": [0.0, 0.01, -0.01, 0.02], "B": [0.0, -0.01, 0.01, -0.02]}, index=index)
    signals = pd.DataFrame({"A": [1.0, 1.0, -1.0, -1.0], "B": [-1.0, -1.0, 1.0, 1.0]}, index=index)
    regimes = pd.DataFrame({"regime": [0, 0, 1, 1]}, index=index)

    backtester = AMRFBacktester(returns=returns, alpha_signals=signals, regime_labels=regimes)
    artifacts = backtester.run(stress_periods={"sample": ("2024-01-01", "2024-01-04")})
    backtester.save(artifacts, output_dir=tmp_path)

    assert (tmp_path / "backtest_results.parquet").exists()
    assert (tmp_path / "performance_report.parquet").exists()
    assert (tmp_path / "position_weights.parquet").exists()
    assert (tmp_path / "regime_performance.parquet").exists()
    assert (tmp_path / "stress_report.parquet").exists()


def test_backtester_defaults_to_first_available_signal_date():
    index = pd.date_range("2024-01-01", periods=4, freq="B")
    returns = pd.DataFrame({"A": [0.01, 0.01, 0.01, 0.01], "B": [-0.01, -0.01, -0.01, -0.01]}, index=index)
    signals = pd.DataFrame({"A": [np.nan, np.nan, 1.0, 1.0], "B": [np.nan, np.nan, -1.0, -1.0]}, index=index)

    artifacts = AMRFBacktester(returns=returns, alpha_signals=signals).run()

    assert artifacts.daily_results.index.min() == index[2]
