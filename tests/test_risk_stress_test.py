import numpy as np
import pandas as pd

from src.risk.stress_test import StressTester


def test_stress_tester_reports_named_windows_and_missing_periods():
    index = pd.date_range("2024-01-01", periods=10, freq="B")
    returns = pd.Series([0.01, -0.02, 0.005, -0.01, 0.02, 0.0, -0.03, 0.01, 0.01, -0.005], index=index)

    report = StressTester().run_stress_test(
        returns,
        {
            "observed": ("2024-01-02", "2024-01-08"),
            "missing": ("2020-01-01", "2020-01-31"),
        },
    )

    assert report.loc["observed", "n_days"] > 0
    assert report.loc["observed", "max_drawdown"] <= 0
    assert report.loc["missing", "n_days"] == 0
    assert np.isnan(report.loc["missing", "period_return"])


def test_scenario_analysis_handles_direct_asset_shocks():
    loss = StressTester.scenario_analysis(
        weights=np.array([0.5, -0.5]),
        factor_shocks=np.array([-0.10, 0.04]),
    )

    assert np.isclose(loss, -0.07)

