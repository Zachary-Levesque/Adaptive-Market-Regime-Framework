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
    min_regime_rank_ic: float = 0.0
    min_ic_positive_rate: float = 0.5
    require_stress_overlap: bool = True
    require_all_stress_scenarios: bool = True
    benchmark_names: tuple[str, ...] = ("SPY", "equal_weight")
    min_benchmark_excess_sharpe: float = 0.0
    min_benchmark_excess_total_return: float = 0.0


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
        regime_diagnostics: pd.DataFrame | None = None,
        data_quality_report: pd.DataFrame | None = None,
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
        rows.extend(self._benchmark_rows(performance_report, strategy))
        rows.extend(self._regime_rows(regime_diagnostics))
        rows.extend(self._data_quality_rows(data_quality_report))

        if stress_report is not None and not stress_report.empty:
            n_days = pd.to_numeric(stress_report.get("n_days", pd.Series(dtype=float)), errors="coerce").fillna(0)
            overlap_days = int(n_days.sum())
            rows.append(
                self._row(
                    "stress_overlap",
                    (not self.thresholds.require_stress_overlap) or overlap_days > 0,
                    overlap_days,
                    "At least one configured stress window should overlap the selected backtest.",
                )
            )
            if self.thresholds.require_all_stress_scenarios:
                missing = stress_report.index[n_days.eq(0)].astype(str).tolist()
                rows.append(
                    self._row(
                        "stress_all_scenarios",
                        len(missing) == 0,
                        ", ".join(missing),
                        "Every configured stress window should overlap the selected backtest.",
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

    def _benchmark_rows(self, performance_report: pd.DataFrame, strategy: pd.Series) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        strategy_sharpe = float(strategy.get("sharpe", 0.0))
        strategy_total_return = float(strategy.get("total_return", 0.0))

        for benchmark_name in self.thresholds.benchmark_names:
            if benchmark_name not in performance_report.index:
                continue

            benchmark = performance_report.loc[benchmark_name]
            excess_sharpe = strategy_sharpe - float(benchmark.get("sharpe", 0.0))
            excess_total_return = strategy_total_return - float(benchmark.get("total_return", 0.0))
            rows.append(
                self._row(
                    f"benchmark_sharpe_{benchmark_name}",
                    excess_sharpe >= self.thresholds.min_benchmark_excess_sharpe,
                    excess_sharpe,
                    f"Selected strategy Sharpe should beat {benchmark_name} by at least "
                    f"{self.thresholds.min_benchmark_excess_sharpe}.",
                )
            )
            rows.append(
                self._row(
                    f"benchmark_total_return_{benchmark_name}",
                    excess_total_return >= self.thresholds.min_benchmark_excess_total_return,
                    excess_total_return,
                    f"Selected strategy total return should beat {benchmark_name} by at least "
                    f"{self.thresholds.min_benchmark_excess_total_return}.",
                )
            )

        return rows

    def _regime_rows(self, regime_diagnostics: pd.DataFrame | None) -> list[dict[str, object]]:
        if regime_diagnostics is None or regime_diagnostics.empty:
            return []

        rows: list[dict[str, object]] = []
        if "mean_rank_ic" not in regime_diagnostics.columns:
            return rows

        for regime, row in regime_diagnostics.iterrows():
            mean_rank_ic = float(row.get("mean_rank_ic", 0.0))
            rows.append(
                self._row(
                    f"regime_rank_ic_{regime}",
                    mean_rank_ic > self.thresholds.min_regime_rank_ic,
                    mean_rank_ic,
                    f"Selected signal should have positive mean rank IC in regime {regime}.",
                )
            )

        return rows

    def _data_quality_rows(self, data_quality_report: pd.DataFrame | None) -> list[dict[str, object]]:
        if data_quality_report is None or data_quality_report.empty:
            return []
        if not {"dataset", "covers_gfc"}.issubset(data_quality_report.columns):
            return []

        price_rows = data_quality_report.loc[data_quality_report["dataset"].eq("prices")]
        if price_rows.empty:
            return []

        covers_gfc = price_rows["covers_gfc"].astype(bool)
        missing_symbols = price_rows.loc[~covers_gfc, "symbol"].astype(str).tolist()
        first_dates = pd.to_datetime(price_rows["first_valid_date"], errors="coerce")
        earliest = first_dates.min()
        latest = first_dates.max()
        date_range = ""
        if pd.notna(earliest) and pd.notna(latest):
            date_range = f"{earliest.date()} to {latest.date()}"

        return [
            self._row(
                "data_price_gfc_coverage",
                len(missing_symbols) == 0,
                ", ".join(missing_symbols),
                "Every configured price symbol should have observations during the GFC stress window.",
            ),
            self._row(
                "data_price_history_start",
                bool(pd.notna(earliest) and earliest <= pd.Timestamp("2008-09-01")),
                date_range,
                "Price history should begin before the first configured GFC stress date.",
            ),
        ]


def readiness_report_passes(report: pd.DataFrame) -> bool:
    """Return whether a saved readiness report clears the RL gate."""
    if report.empty or "ready_for_rl" not in report.columns:
        return False
    return bool(report["ready_for_rl"].all())


def load_readiness_status(path: str | Path) -> tuple[bool, pd.DataFrame]:
    """Load a readiness report and return its gate status."""
    report_path = Path(path)
    if not report_path.exists():
        return False, pd.DataFrame()
    report = pd.read_parquet(report_path)
    return readiness_report_passes(report), report
