"""Pre-RL readiness checks for the selected alpha signal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ReadinessThresholds:
    min_active_days: int = 504
    min_sharpe: float = 0.5
    min_total_return: float = 0.0
    min_rank_ic: float = 0.0
    min_ic_positive_rate: float = 0.5
    require_stress_overlap: bool = True


class AlphaReadinessChecker:
    """Evaluate whether selected alpha evidence is strong enough for RL work."""

    def __init__(self, thresholds: ReadinessThresholds | None = None) -> None:
        self.thresholds = thresholds or ReadinessThresholds()

    def evaluate(
        self,
        selection: pd.DataFrame,
        diagnostics_summary: pd.DataFrame,
        performance_report: pd.DataFrame,
        stress_report: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        selected = selection.iloc[0] if not selection.empty else pd.Series(dtype=object)
        diagnostics = diagnostics_summary.iloc[0] if not diagnostics_summary.empty else pd.Series(dtype=object)
        strategy = (
            performance_report.loc["strategy"]
            if "strategy" in performance_report.index
            else pd.Series(dtype=object)
        )

        rows = [
            self._row(
                "selected_model",
                bool(selected.get("model")),
                str(selected.get("model", "")),
                "A selected alpha model must be present.",
            ),
            self._row(
                "active_history",
                float(diagnostics.get("n_days", 0.0)) >= self.thresholds.min_active_days,
                float(diagnostics.get("n_days", 0.0)),
                f"Selected signal should have at least {self.thresholds.min_active_days} scored days.",
            ),
            self._row(
                "rank_ic",
                float(diagnostics.get("mean_rank_ic", 0.0)) > self.thresholds.min_rank_ic,
                float(diagnostics.get("mean_rank_ic", 0.0)),
                "Selected signal should have positive mean rank IC.",
            ),
            self._row(
                "ic_positive_rate",
                float(diagnostics.get("ic_positive_rate", 0.0)) > self.thresholds.min_ic_positive_rate,
                float(diagnostics.get("ic_positive_rate", 0.0)),
                "IC should be positive on more than half of scored days.",
            ),
            self._row(
                "backtest_sharpe",
                float(strategy.get("sharpe", 0.0)) >= self.thresholds.min_sharpe,
                float(strategy.get("sharpe", 0.0)),
                f"Selected strategy Sharpe should be at least {self.thresholds.min_sharpe}.",
            ),
            self._row(
                "backtest_total_return",
                float(strategy.get("total_return", 0.0)) > self.thresholds.min_total_return,
                float(strategy.get("total_return", 0.0)),
                "Selected strategy total return should be positive.",
            ),
        ]

        if stress_report is not None and not stress_report.empty:
            overlap_days = int(pd.to_numeric(stress_report.get("n_days", pd.Series(dtype=float)), errors="coerce").sum())
            rows.append(
                self._row(
                    "stress_overlap",
                    (not self.thresholds.require_stress_overlap) or overlap_days > 0,
                    overlap_days,
                    "At least one configured stress window should overlap the selected backtest.",
                )
            )

        report = pd.DataFrame(rows)
        report["ready_for_rl"] = bool(report["passed"].all()) if not report.empty else False
        return report

    def save(self, report: pd.DataFrame, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        report.to_parquet(output)

    @staticmethod
    def _row(check: str, passed: bool, value, detail: str) -> dict[str, object]:
        return {
            "check": check,
            "passed": bool(passed),
            "value": str(value),
            "detail": detail,
        }
