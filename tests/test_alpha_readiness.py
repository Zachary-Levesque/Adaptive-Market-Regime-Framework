import pandas as pd
from types import SimpleNamespace

from src.alpha.build_readiness import resolve_effective_selection
from src.alpha.readiness import (
    AlphaReadinessChecker,
    ReadinessThresholds,
    load_readiness_status,
    readiness_report_passes,
)


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


def test_alpha_readiness_fails_when_price_data_lacks_gfc_coverage():
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
    data_quality = pd.DataFrame(
        [
            {
                "dataset": "prices",
                "symbol": "SPY",
                "first_valid_date": "2012-01-03",
                "covers_gfc": False,
            },
            {
                "dataset": "prices",
                "symbol": "QQQ",
                "first_valid_date": "2012-01-03",
                "covers_gfc": False,
            },
        ]
    )

    report = checker.evaluate(
        selection=selection,
        diagnostics_summary=diagnostics,
        performance_report=performance,
        stress_report=pd.DataFrame([{"n_days": 5}], index=["sample"]),
        data_quality_report=data_quality,
    )

    failed = set(report.loc[~report["passed"], "check"])
    assert "data_price_gfc_coverage" in failed
    assert "data_price_history_start" in failed
    assert not report["ready_for_rl"].any()


def test_readiness_status_loader_fails_closed_for_missing_or_invalid_report(tmp_path):
    missing_status, missing_report = load_readiness_status(tmp_path / "missing.parquet")
    invalid = pd.DataFrame([{"check": "selected_model", "passed": True}])
    invalid_path = tmp_path / "invalid.parquet"
    invalid.to_parquet(invalid_path)

    invalid_status, invalid_report = load_readiness_status(invalid_path)

    assert missing_status is False
    assert missing_report.empty
    assert invalid_status is False
    assert invalid_report.equals(invalid)
    assert readiness_report_passes(pd.DataFrame([{"ready_for_rl": True}])) is True
    assert readiness_report_passes(pd.DataFrame([{"ready_for_rl": True}, {"ready_for_rl": False}])) is False


def test_readiness_effective_selection_falls_back_from_rl_tilted_signal(tmp_path):
    fallback_path = tmp_path / "signals" / "defensive_regime_selector.parquet"
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    fallback_path.write_bytes(b"")
    comparison_path = tmp_path / "alpha_model_comparison.parquet"
    pd.DataFrame(
        [{"model": "defensive_regime_selector", "signal_path": str(fallback_path)}]
    ).to_parquet(tmp_path / "alpha_model_comparison_summary.parquet")
    config = SimpleNamespace(
        alpha=SimpleNamespace(
            comparison_path=comparison_path,
            signals_path=tmp_path / "fallback.parquet",
        )
    )
    selection = pd.DataFrame(
        [{"model": "rl_tilted", "signal_path": str(tmp_path / "alpha_signals_rl_tilted.parquet")}]
    )

    effective = resolve_effective_selection(config, selection)

    assert effective.loc[0, "model"] == "defensive_regime_selector"
    assert effective.loc[0, "signal_path"] == str(fallback_path)
    assert effective.loc[0, "selection_method"] == "pre_rl_fallback"
