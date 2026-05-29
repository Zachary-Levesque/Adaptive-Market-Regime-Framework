import numpy as np
import pandas as pd

from src.risk.metrics import PerformanceMetrics


def test_performance_metrics_compute_summary_and_regime_breakdown():
    index = pd.date_range("2024-01-01", periods=6, freq="B")
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, 0.01], index=index)
    regimes = pd.DataFrame({"regime": [0, 0, 1, 1, 1, 0]}, index=index)

    metrics = PerformanceMetrics()
    summary = metrics.summarize(returns)
    regime_summary = metrics.regime_conditional_performance(returns, regimes)

    assert summary["total_return"] > 0
    assert summary["max_drawdown"] < 0
    assert np.isclose(summary["win_rate"], 4 / 6)
    assert set(regime_summary.index) == {0, 1}
    assert regime_summary.loc[1, "n_days"] == 3


def test_drawdown_duration_tracks_underwater_period():
    index = pd.date_range("2024-01-01", periods=5, freq="B")
    returns = pd.Series([0.1, -0.05, -0.05, 0.01, 0.2], index=index)

    profile = PerformanceMetrics().compute_drawdown_profile(returns)

    assert profile.max_drawdown < 0
    assert profile.max_duration == 3

