import pandas as pd

from src.alpha.readiness import AlphaReadinessChecker, ReadinessThresholds


def test_alpha_readiness_passes_when_all_thresholds_are_met(tmp_path):
    checker = AlphaReadinessChecker(ReadinessThresholds(min_active_days=2, min_sharpe=0.5))
    selection = pd.DataFrame([{"model": "ensemble"}])
    diagnostics = pd.DataFrame(
        [
            {
                "n_days": 3,
                "mean_rank_ic": 0.01,
                "ic_positive_rate": 0.6,
            }
        ],
        index=["overall"],
    )
    performance = pd.DataFrame(
        [{"sharpe": 1.0, "total_return": 0.05}],
        index=["strategy"],
    )
    stress = pd.DataFrame([{"n_days": 5}], index=["sample"])

    report = checker.evaluate(selection, diagnostics, performance, stress)
    checker.save(report, tmp_path / "readiness.parquet")

    assert report["passed"].all()
    assert report["ready_for_rl"].all()
    assert (tmp_path / "readiness.parquet").exists()


def test_alpha_readiness_fails_on_short_history_negative_ic_and_missing_stress():
    checker = AlphaReadinessChecker(ReadinessThresholds(min_active_days=100, min_sharpe=0.5))
    selection = pd.DataFrame([{"model": "ensemble"}])
    diagnostics = pd.DataFrame(
        [
            {
                "n_days": 10,
                "mean_rank_ic": -0.01,
                "ic_positive_rate": 0.4,
            }
        ],
        index=["overall"],
    )
    performance = pd.DataFrame(
        [{"sharpe": 0.9, "total_return": 0.05}],
        index=["strategy"],
    )
    stress = pd.DataFrame([{"n_days": 0}], index=["missing"])

    report = checker.evaluate(selection, diagnostics, performance, stress)

    failed = set(report.loc[~report["passed"], "check"])
    assert {"active_history", "rank_ic", "ic_positive_rate", "stress_overlap", "stress_all_scenarios"}.issubset(failed)
    assert not report["ready_for_rl"].any()


def test_alpha_readiness_fails_when_strategy_lags_benchmarks():
    checker = AlphaReadinessChecker(ReadinessThresholds(min_active_days=2, min_sharpe=0.1))
    selection = pd.DataFrame([{"model": "regime_selector"}])
    diagnostics = pd.DataFrame(
        [
            {
                "n_days": 10,
                "mean_rank_ic": 0.02,
                "ic_positive_rate": 0.6,
            }
        ],
        index=["overall"],
    )
    performance = pd.DataFrame(
        [
            {"sharpe": 0.2, "total_return": 0.05},
            {"sharpe": 1.0, "total_return": 0.50},
            {"sharpe": 0.7, "total_return": 0.30},
        ],
        index=["strategy", "SPY", "equal_weight"],
    )
    stress = pd.DataFrame([{"n_days": 5}], index=["sample"])

    report = checker.evaluate(selection, diagnostics, performance, stress)

    failed = set(report.loc[~report["passed"], "check"])
    assert "benchmark_sharpe_SPY" in failed
    assert "benchmark_total_return_equal_weight" in failed
    assert not report["ready_for_rl"].any()


def test_alpha_readiness_fails_when_any_regime_has_negative_rank_ic():
    checker = AlphaReadinessChecker(
        ReadinessThresholds(
            min_active_days=2,
            min_sharpe=0.1,
            benchmark_names=(),
            require_all_stress_scenarios=False,
        )
    )
    selection = pd.DataFrame([{"model": "regime_selector"}])
    diagnostics = pd.DataFrame(
        [{"n_days": 10, "mean_rank_ic": 0.02, "ic_positive_rate": 0.6}],
        index=["overall"],
    )
    performance = pd.DataFrame(
        [{"sharpe": 0.2, "total_return": 0.05}],
        index=["strategy"],
    )
    regime_diagnostics = pd.DataFrame(
        [
            {"mean_rank_ic": 0.03},
            {"mean_rank_ic": -0.02},
        ],
        index=[0, 3],
    )

    report = checker.evaluate(
        selection=selection,
        diagnostics_summary=diagnostics,
        performance_report=performance,
        stress_report=pd.DataFrame([{"n_days": 5}], index=["sample"]),
        regime_diagnostics=regime_diagnostics,
    )

    failed = set(report.loc[~report["passed"], "check"])
    assert "regime_rank_ic_3" in failed
    assert "regime_rank_ic_0" not in failed
    assert not report["ready_for_rl"].any()
