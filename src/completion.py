"""Final project completion checks for AMRF artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import AppConfig


@dataclass(frozen=True)
class CompletionArtifacts:
    report: pd.DataFrame
    complete: bool


class ProjectCompletionChecker:
    """Evaluate whether the project artifacts satisfy the end-to-end research completion gate."""

    REQUIRED_ARTIFACTS = (
        "prices.parquet",
        "returns.parquet",
        "factors.parquet",
        "technical_features.parquet",
        "regime_features.parquet",
        "data_quality_report.parquet",
        "alpha_signal_selection.parquet",
        "alpha_diagnostics_summary.parquet",
        "alpha_diagnostics_by_regime.parquet",
        "alpha_readiness_report.parquet",
    )
    REQUIRED_RESULT_ARTIFACTS = (
        "performance_report.parquet",
        "backtest_results.parquet",
        "position_weights.parquet",
        "stress_report.parquet",
    )
    REQUIRED_REGIME_ARTIFACTS = (
        "regime_labels.parquet",
        "regime_probs.parquet",
        "gmm_validation.parquet",
        "regime_summary.parquet",
        "transition_matrix.parquet",
    )

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def evaluate(self) -> CompletionArtifacts:
        rows: list[dict[str, object]] = []
        rows.extend(self._artifact_rows())
        rows.extend(self._selection_rows())
        rows.extend(self._readiness_rows())
        rows.extend(self._performance_rows())
        rows.extend(self._data_coverage_rows())
        rows.extend(self._rl_rows())
        rows.extend(self._dashboard_rows())

        report = pd.DataFrame(rows)
        if report.empty:
            report = pd.DataFrame(columns=["check", "passed", "value", "detail", "required"])
        if "required" not in report.columns:
            report["required"] = True
        report["complete"] = report["passed"].astype(bool) | ~report["required"].astype(bool)
        return CompletionArtifacts(report=report, complete=bool(report["complete"].all()) if not report.empty else False)

    def save(self, artifacts: CompletionArtifacts, path: str | Path | None = None) -> Path:
        output_path = Path(path) if path is not None else self.config.risk.output_dir / "project_completion_report.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        artifacts.report.to_parquet(output_path)
        return output_path

    def _artifact_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for filename in self.REQUIRED_ARTIFACTS:
            path = self.config.data.processed_dir / filename
            rows.append(self._row(f"artifact_processed_{filename}", path.exists(), str(path), "Required processed artifact must exist."))
        for filename in self.REQUIRED_RESULT_ARTIFACTS:
            path = self.config.risk.output_dir / filename
            rows.append(self._row(f"artifact_result_{filename}", path.exists(), str(path), "Required result artifact must exist."))
        for filename in self.REQUIRED_REGIME_ARTIFACTS:
            path = self.config.regime.output_dir / filename
            rows.append(self._row(f"artifact_regime_{filename}", path.exists(), str(path), "Required regime artifact must exist."))
        return rows

    def _selection_rows(self) -> list[dict[str, object]]:
        path = self.config.alpha.selection_path
        if not path.exists():
            return [self._row("selected_signal_manifest", False, str(path), "Selected-signal manifest must exist.")]

        selection = pd.read_parquet(path)
        if selection.empty:
            return [self._row("selected_signal_manifest", False, "empty", "Selected-signal manifest must contain one selected model.")]

        selected = selection.iloc[0]
        signal_path = Path(str(selected.get("signal_path", "")))
        model = str(selected.get("model", ""))
        is_rl = signal_path.name == "alpha_signals_rl_tilted.parquet" or model == "rl_tilted"
        return [
            self._row("selected_signal_manifest", bool(model and signal_path.exists()), model, "Selected model and signal path must be valid."),
            self._row("selected_signal_not_unready_rl", not is_rl, str(signal_path), "Completion cannot depend on RL-tilted signals before readiness passes."),
        ]

    def _readiness_rows(self) -> list[dict[str, object]]:
        path = self.config.alpha.diagnostics_path.with_name("alpha_readiness_report.parquet")
        if not path.exists():
            return [self._row("readiness_report", False, str(path), "Readiness report must exist.", required=False)]

        readiness = pd.read_parquet(path)
        if readiness.empty or "ready_for_rl" not in readiness.columns:
            return [self._row("readiness_report", False, "invalid", "Readiness report must include ready_for_rl.", required=False)]

        rows = [
            self._row(
                "readiness_gate",
                bool(readiness["ready_for_rl"].all()),
                bool(readiness["ready_for_rl"].all()),
                "All readiness checks must pass before RL deployment.",
                required=False,
            )
        ]
        if {"check", "passed", "value", "detail"}.issubset(readiness.columns):
            for _, row in readiness.loc[~readiness["passed"].astype(bool)].iterrows():
                rows.append(
                    self._row(
                        f"readiness_blocker_{row['check']}",
                        False,
                        row["value"],
                        row["detail"],
                        required=False,
                    )
                )
        return rows

    def _performance_rows(self) -> list[dict[str, object]]:
        path = self.config.risk.output_dir / "performance_report.parquet"
        if not path.exists():
            return []
        performance = pd.read_parquet(path)
        if "strategy" not in performance.index:
            return [self._row("performance_strategy_row", False, "missing", "Performance report must include strategy row.")]

        strategy = performance.loc["strategy"]
        rows = [
            self._row(
                "performance_strategy_positive_return",
                float(strategy.get("total_return", 0.0)) > 0.0,
                float(strategy.get("total_return", 0.0)),
                "Selected strategy total return must be positive.",
            )
        ]
        for benchmark in ("SPY", "equal_weight"):
            if benchmark not in performance.index:
                continue
            excess_sharpe = float(strategy.get("sharpe", 0.0)) - float(performance.loc[benchmark].get("sharpe", 0.0))
            rows.append(
                self._row(
                    f"performance_beats_{benchmark}_sharpe",
                    excess_sharpe >= 0.0,
                    excess_sharpe,
                    f"Selected strategy Sharpe must be at least {benchmark}'s Sharpe.",
                    required=False,
                )
            )
        return rows

    def _data_coverage_rows(self) -> list[dict[str, object]]:
        path = self.config.data.processed_dir / "data_quality_report.parquet"
        if not path.exists():
            return []
        quality = pd.read_parquet(path)
        if not {"dataset", "symbol", "covers_gfc"}.issubset(quality.columns):
            return [self._row("data_quality_schema", False, list(quality.columns), "Data quality report must include coverage fields.")]

        prices = quality.loc[quality["dataset"].eq("prices")]
        missing = prices.loc[~prices["covers_gfc"].astype(bool), "symbol"].astype(str).tolist()
        return [
            self._row(
                "data_all_prices_cover_gfc",
                len(missing) == 0,
                ", ".join(missing),
                "All configured price symbols must cover the GFC stress period for final completion.",
            )
        ]

    def _rl_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        if not hasattr(self.config, "rl"):
            return rows
        if self.config.rl.model_path.exists():
            rows.append(self._row("rl_model_saved", True, str(self.config.rl.model_path), "PPO model must be saved."))
        if self.config.rl.backtest_results_path.exists():
            rows.append(
                self._row(
                    "rl_backtest_results_saved",
                    True,
                    str(self.config.rl.backtest_results_path),
                    "RL backtest results must be saved.",
                )
            )
        if self.config.rl.positions_path.exists():
            rows.append(
                self._row(
                    "rl_positions_saved",
                    True,
                    str(self.config.rl.positions_path),
                    "RL position weights must be saved.",
                )
            )
        if self.config.rl.comparison_path.exists():
            rows.append(
                self._row(
                    "rl_comparison_saved",
                    True,
                    str(self.config.rl.comparison_path),
                    "RL vs baseline comparison must be saved.",
                )
            )
        return rows

    def _dashboard_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        dashboard_app = Path("dashboard/app.py")
        dashboard_requirements = Path("dashboard/requirements.txt")
        dashboard_readme = Path("README_dashboard.md")
        for check, path, detail in [
            ("dashboard_app_saved", dashboard_app, "Streamlit dashboard must exist."),
            ("dashboard_requirements_saved", dashboard_requirements, "Dashboard requirements must exist."),
            ("dashboard_readme_saved", dashboard_readme, "Dashboard README must exist."),
        ]:
            if path.exists():
                rows.append(self._row(check, True, str(path), detail))
        return rows

    @staticmethod
    def _row(check: str, passed: bool, value, detail: str) -> dict[str, object]:
        return {
            "check": check,
            "passed": bool(passed),
            "value": str(value),
            "detail": detail,
            "required": True,
        }
