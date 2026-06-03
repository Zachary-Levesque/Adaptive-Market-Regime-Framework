from types import SimpleNamespace

import pandas as pd

from src.completion import ProjectCompletionChecker


def _config(tmp_path):
    return SimpleNamespace(
        data=SimpleNamespace(processed_dir=tmp_path / "processed"),
        regime=SimpleNamespace(output_dir=tmp_path / "regimes"),
        alpha=SimpleNamespace(
            selection_path=tmp_path / "processed" / "alpha_signal_selection.parquet",
            diagnostics_path=tmp_path / "processed" / "alpha_diagnostics.parquet",
        ),
        risk=SimpleNamespace(output_dir=tmp_path / "results"),
    )


def _write_required_artifacts(config):
    config.data.processed_dir.mkdir(parents=True)
    config.regime.output_dir.mkdir(parents=True)
    config.risk.output_dir.mkdir(parents=True)
    for filename in ProjectCompletionChecker.REQUIRED_ARTIFACTS:
        if filename in {"alpha_signal_selection.parquet", "alpha_readiness_report.parquet", "data_quality_report.parquet"}:
            continue
        pd.DataFrame({"x": [1]}).to_parquet(config.data.processed_dir / filename)
    for filename in ProjectCompletionChecker.REQUIRED_RESULT_ARTIFACTS:
        if filename == "performance_report.parquet":
            continue
        pd.DataFrame({"x": [1]}).to_parquet(config.risk.output_dir / filename)
    for filename in ProjectCompletionChecker.REQUIRED_REGIME_ARTIFACTS:
        pd.DataFrame({"x": [1]}).to_parquet(config.regime.output_dir / filename)


def test_project_completion_passes_when_all_gates_pass(tmp_path):
    config = _config(tmp_path)
    _write_required_artifacts(config)
    signal_path = config.data.processed_dir / "alpha_signals" / "regime_selector.parquet"
    signal_path.parent.mkdir()
    pd.DataFrame({"SPY": [1.0]}).to_parquet(signal_path)
    pd.DataFrame([{"model": "regime_selector", "signal_path": str(signal_path)}]).to_parquet(
        config.alpha.selection_path
    )
    pd.DataFrame([{"check": "selected_model", "passed": True, "value": "ok", "detail": "", "ready_for_rl": True}]).to_parquet(
        config.data.processed_dir / "alpha_readiness_report.parquet"
    )
    pd.DataFrame([{"dataset": "prices", "symbol": "SPY", "covers_gfc": True}]).to_parquet(
        config.data.processed_dir / "data_quality_report.parquet"
    )
    pd.DataFrame(
        [
            {"sharpe": 1.0, "total_return": 0.2},
            {"sharpe": 0.5, "total_return": 0.1},
            {"sharpe": 0.4, "total_return": 0.1},
        ],
        index=["strategy", "SPY", "equal_weight"],
    ).to_parquet(config.risk.output_dir / "performance_report.parquet")

    artifacts = ProjectCompletionChecker(config).evaluate()

    assert artifacts.complete is True
    assert artifacts.report["complete"].all()


def test_project_completion_reports_readiness_and_data_blockers(tmp_path):
    config = _config(tmp_path)
    _write_required_artifacts(config)
    signal_path = config.data.processed_dir / "alpha_signals" / "regime_selector.parquet"
    signal_path.parent.mkdir()
    pd.DataFrame({"SPY": [1.0]}).to_parquet(signal_path)
    pd.DataFrame([{"model": "regime_selector", "signal_path": str(signal_path)}]).to_parquet(
        config.alpha.selection_path
    )
    pd.DataFrame(
        [
            {
                "check": "backtest_sharpe",
                "passed": False,
                "value": "0.1",
                "detail": "Sharpe too low.",
                "ready_for_rl": False,
            }
        ]
    ).to_parquet(config.data.processed_dir / "alpha_readiness_report.parquet")
    pd.DataFrame([{"dataset": "prices", "symbol": "SPY", "covers_gfc": False}]).to_parquet(
        config.data.processed_dir / "data_quality_report.parquet"
    )
    pd.DataFrame(
        [
            {"sharpe": 0.1, "total_return": 0.05},
            {"sharpe": 0.5, "total_return": 0.1},
        ],
        index=["strategy", "SPY"],
    ).to_parquet(config.risk.output_dir / "performance_report.parquet")

    artifacts = ProjectCompletionChecker(config).evaluate()
    failed = set(artifacts.report.loc[~artifacts.report["passed"], "check"])

    assert artifacts.complete is False
    assert "readiness_gate" in failed
    assert "readiness_blocker_backtest_sharpe" in failed
    assert "data_all_prices_cover_gfc" in failed
    assert "performance_beats_SPY_sharpe" in failed
