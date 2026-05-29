import numpy as np
import pandas as pd

from src.alpha.diagnostics import AlphaDiagnostics


def test_alpha_diagnostics_scores_forward_returns_and_regimes(tmp_path):
    index = pd.date_range("2024-01-01", periods=5, freq="B")
    signals = pd.DataFrame(
        {
            "A": [1.0, 1.0, 1.0, np.nan, 1.0],
            "B": [0.5, 0.5, 0.5, np.nan, 0.5],
            "C": [-1.0, -1.0, -1.0, np.nan, -1.0],
        },
        index=index,
    )
    returns = pd.DataFrame(
        {
            "A": [0.0, 0.03, 0.02, 0.01, 0.00],
            "B": [0.0, 0.01, 0.01, 0.00, 0.00],
            "C": [0.0, -0.02, -0.01, -0.01, 0.00],
        },
        index=index,
    )
    regimes = pd.DataFrame({"regime": [0, 0, 1, 1, 1]}, index=index)

    diagnostics = AlphaDiagnostics(min_assets_per_day=3)
    artifacts = diagnostics.evaluate(signals, returns, regimes)
    diagnostics.save(artifacts, tmp_path / "alpha_diagnostics.parquet")

    assert artifacts.summary.loc["overall", "n_days"] == 3
    assert artifacts.summary.loc["overall", "mean_rank_ic"] > 0
    assert set(artifacts.regime_summary.index) == {0, 1}
    assert (tmp_path / "alpha_diagnostics.parquet").exists()
    assert (tmp_path / "alpha_diagnostics_summary.parquet").exists()
    assert (tmp_path / "alpha_diagnostics_by_regime.parquet").exists()


def test_alpha_diagnostics_requires_overlap():
    diagnostics = AlphaDiagnostics()
    signals = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2024-01-01")])
    returns = pd.DataFrame({"B": [0.01]}, index=[pd.Timestamp("2024-01-01")])

    try:
        diagnostics.evaluate(signals, returns)
    except ValueError as exc:
        assert "overlapping tickers" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing ticker overlap")

