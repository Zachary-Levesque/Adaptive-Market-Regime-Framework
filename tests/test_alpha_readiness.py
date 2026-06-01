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
    assert {"active_history", "rank_ic", "ic_positive_rate", "stress_overlap"}.issubset(failed)
    assert not report["ready_for_rl"].any()
