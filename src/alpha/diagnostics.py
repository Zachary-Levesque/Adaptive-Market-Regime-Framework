"""Diagnostics for alpha forecast quality."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class AlphaDiagnosticsArtifacts:
    daily_diagnostics: pd.DataFrame
    summary: pd.DataFrame
    regime_summary: pd.DataFrame


class AlphaDiagnostics:
    """Evaluate whether alpha forecasts rank and sign next-day returns."""

    def __init__(self, min_assets_per_day: int = 3) -> None:
        self.min_assets_per_day = min_assets_per_day

    def evaluate(
        self,
        alpha_signals: pd.DataFrame,
        returns: pd.DataFrame,
        regime_labels: pd.DataFrame | pd.Series | None = None,
    ) -> AlphaDiagnosticsArtifacts:
        signals, forward_returns = self._align_signals_and_forward_returns(alpha_signals, returns)
        regimes = self._normalize_regime_labels(regime_labels).reindex(signals.index) if regime_labels is not None else None

        rows: list[dict[str, float | int | pd.Timestamp]] = []
        for date in signals.index:
            signal_row = signals.loc[date]
            return_row = forward_returns.loc[date]
            joined = pd.DataFrame({"signal": signal_row, "forward_return": return_row}).dropna()
            if len(joined) < self.min_assets_per_day:
                continue

            row: dict[str, float | int | pd.Timestamp] = {
                "date": date,
                "n_assets": int(len(joined)),
                "ic": self._safe_corr(joined["signal"], joined["forward_return"], method="pearson"),
                "rank_ic": self._safe_corr(joined["signal"], joined["forward_return"], method="spearman"),
                "hit_rate": float((np.sign(joined["signal"]) == np.sign(joined["forward_return"])).mean()),
                "long_short_spread": self._long_short_spread(joined),
                "mean_abs_signal": float(joined["signal"].abs().mean()),
                "signal_coverage": float(signal_row.notna().mean()),
            }
            if regimes is not None and pd.notna(regimes.loc[date]):
                row["regime"] = int(regimes.loc[date])
            rows.append(row)

        daily = pd.DataFrame(rows)
        if not daily.empty:
            daily = daily.set_index("date").sort_index()

        summary = self._summarize(daily, label="overall")
        regime_summary = self._summarize_by_regime(daily)
        return AlphaDiagnosticsArtifacts(
            daily_diagnostics=daily,
            summary=summary,
            regime_summary=regime_summary,
        )

    def save(self, artifacts: AlphaDiagnosticsArtifacts, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        artifacts.daily_diagnostics.to_parquet(output_path)
        artifacts.summary.to_parquet(output_path.with_name("alpha_diagnostics_summary.parquet"))
        if not artifacts.regime_summary.empty:
            artifacts.regime_summary.to_parquet(output_path.with_name("alpha_diagnostics_by_regime.parquet"))

    @staticmethod
    def _align_signals_and_forward_returns(
        alpha_signals: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        signals = AlphaDiagnostics._normalize_frame(alpha_signals)
        normalized_returns = AlphaDiagnostics._normalize_frame(returns)
        forward_returns = normalized_returns.shift(-1)

        common_index = signals.index.intersection(forward_returns.index).sort_values()
        common_columns = signals.columns.intersection(forward_returns.columns).sort_values()
        if common_index.empty:
            raise ValueError("alpha_signals and returns have no overlapping dates.")
        if common_columns.empty:
            raise ValueError("alpha_signals and returns have no overlapping tickers.")

        signals = signals.loc[common_index, common_columns]
        forward_returns = forward_returns.loc[common_index, common_columns]
        active = signals.notna().any(axis=1)
        return signals.loc[active], forward_returns.loc[active]

    @staticmethod
    def _summarize(daily: pd.DataFrame, label: str) -> pd.DataFrame:
        if daily.empty:
            return pd.DataFrame(
                [
                    {
                        "segment": label,
                        "n_days": 0,
                        "mean_ic": 0.0,
                        "mean_rank_ic": 0.0,
                        "ic_positive_rate": 0.0,
                        "mean_hit_rate": 0.0,
                        "mean_long_short_spread": 0.0,
                        "t_stat_ic": 0.0,
                        "mean_signal_coverage": 0.0,
                    }
                ]
            ).set_index("segment")

        ic = daily["ic"].dropna()
        ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else 0.0
        t_stat = float(ic.mean() / (ic_std / np.sqrt(len(ic)))) if ic_std > 0.0 else 0.0
        return pd.DataFrame(
            [
                {
                    "segment": label,
                    "n_days": int(len(daily)),
                    "mean_ic": float(daily["ic"].mean()),
                    "mean_rank_ic": float(daily["rank_ic"].mean()),
                    "ic_positive_rate": float((daily["ic"] > 0.0).mean()),
                    "mean_hit_rate": float(daily["hit_rate"].mean()),
                    "mean_long_short_spread": float(daily["long_short_spread"].mean()),
                    "t_stat_ic": t_stat,
                    "mean_signal_coverage": float(daily["signal_coverage"].mean()),
                }
            ]
        ).set_index("segment")

    def _summarize_by_regime(self, daily: pd.DataFrame) -> pd.DataFrame:
        if daily.empty or "regime" not in daily.columns:
            return pd.DataFrame()

        frames = []
        for regime, group in daily.dropna(subset=["regime"]).groupby("regime"):
            summary = self._summarize(group, label=f"regime_{int(regime)}")
            summary.insert(0, "regime", int(regime))
            frames.append(summary)

        return pd.concat(frames).set_index("regime").sort_index() if frames else pd.DataFrame()

    @staticmethod
    def _long_short_spread(joined: pd.DataFrame, fraction: float = 0.2) -> float:
        ranked = joined.sort_values("signal")
        bucket = max(1, int(np.ceil(len(ranked) * fraction)))
        long_leg = float(ranked.tail(bucket)["forward_return"].mean())
        short_leg = float(ranked.head(bucket)["forward_return"].mean())
        return long_leg - short_leg

    @staticmethod
    def _safe_corr(left: pd.Series, right: pd.Series, method: str) -> float:
        frame = pd.DataFrame({"left": left, "right": right}).dropna()
        if len(frame) < 2:
            return 0.0
        value = frame["left"].corr(frame["right"], method=method)
        return float(value) if pd.notna(value) else 0.0

    @staticmethod
    def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        normalized.index = pd.to_datetime(normalized.index).tz_localize(None)
        normalized = normalized.apply(pd.to_numeric, errors="coerce")
        return normalized.replace([np.inf, -np.inf], np.nan).sort_index()

    @staticmethod
    def _normalize_regime_labels(regime_labels: pd.DataFrame | pd.Series) -> pd.Series:
        if isinstance(regime_labels, pd.DataFrame):
            if "regime" not in regime_labels.columns:
                raise KeyError("Expected regime label DataFrame to contain a 'regime' column.")
            labels = regime_labels["regime"]
        else:
            labels = regime_labels

        normalized = pd.Series(labels).copy()
        normalized.index = pd.to_datetime(normalized.index).tz_localize(None)
        return pd.to_numeric(normalized, errors="coerce").sort_index()

