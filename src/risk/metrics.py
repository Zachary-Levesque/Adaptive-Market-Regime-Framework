"""Performance metrics for daily-return strategy evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TRADING_DAYS = 252


@dataclass(frozen=True)
class DrawdownProfile:
    drawdown: pd.Series
    max_drawdown: float
    max_duration: int


class PerformanceMetrics:
    """Compute standard portfolio and strategy diagnostics."""

    def __init__(self, periods_per_year: int = TRADING_DAYS) -> None:
        self.periods_per_year = periods_per_year

    def annualized_return(self, returns: pd.Series) -> float:
        clean = self._clean_returns(returns)
        if clean.empty:
            return 0.0
        total_return = float((1.0 + clean).prod() - 1.0)
        years = len(clean) / self.periods_per_year
        if years <= 0:
            return 0.0
        return float((1.0 + total_return) ** (1.0 / years) - 1.0)

    def annualized_volatility(self, returns: pd.Series) -> float:
        clean = self._clean_returns(returns)
        if clean.empty:
            return 0.0
        return float(clean.std(ddof=0) * np.sqrt(self.periods_per_year))

    def sharpe_ratio(self, returns: pd.Series, rf: float = 0.0) -> float:
        clean = self._clean_returns(returns)
        if clean.empty:
            return 0.0
        excess = clean - rf / self.periods_per_year
        volatility = float(excess.std(ddof=0))
        if volatility == 0.0:
            return 0.0
        return float(np.sqrt(self.periods_per_year) * excess.mean() / volatility)

    def sortino_ratio(self, returns: pd.Series, rf: float = 0.0) -> float:
        clean = self._clean_returns(returns)
        if clean.empty:
            return 0.0
        excess = clean - rf / self.periods_per_year
        downside = excess[excess < 0.0]
        downside_deviation = float(np.sqrt((downside**2).mean())) if not downside.empty else 0.0
        if downside_deviation == 0.0:
            return 0.0
        return float(np.sqrt(self.periods_per_year) * excess.mean() / downside_deviation)

    def calmar_ratio(self, returns: pd.Series) -> float:
        annual_return = self.annualized_return(returns)
        drawdown = abs(self.max_drawdown(returns))
        if drawdown == 0.0:
            return 0.0
        return float(annual_return / drawdown)

    def max_drawdown(self, returns: pd.Series) -> float:
        return self.compute_drawdown_profile(returns).max_drawdown

    def compute_drawdown_profile(self, returns: pd.Series) -> DrawdownProfile:
        clean = self._clean_returns(returns)
        if clean.empty:
            empty = pd.Series(dtype=float, name="drawdown")
            return DrawdownProfile(drawdown=empty, max_drawdown=0.0, max_duration=0)

        equity = (1.0 + clean).cumprod()
        peak = equity.cummax()
        drawdown = equity / peak - 1.0
        drawdown.name = "drawdown"

        max_duration = 0
        current_duration = 0
        for value in drawdown:
            if value < 0.0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return DrawdownProfile(
            drawdown=drawdown,
            max_drawdown=float(drawdown.min()),
            max_duration=max_duration,
        )

    def win_rate(self, returns: pd.Series) -> float:
        clean = self._clean_returns(returns)
        if clean.empty:
            return 0.0
        return float((clean > 0.0).mean())

    def profit_factor(self, returns: pd.Series) -> float:
        clean = self._clean_returns(returns)
        if clean.empty:
            return 0.0
        gains = float(clean[clean > 0.0].sum())
        losses = abs(float(clean[clean < 0.0].sum()))
        if losses == 0.0:
            return np.inf if gains > 0.0 else 0.0
        return float(gains / losses)

    def monthly_returns(self, returns: pd.Series) -> pd.Series:
        clean = self._clean_returns(returns)
        if clean.empty:
            return pd.Series(dtype=float, name="monthly_return")
        monthly = clean.resample("ME").apply(lambda values: (1.0 + values).prod() - 1.0)
        monthly.name = "monthly_return"
        return monthly

    def annual_returns(self, returns: pd.Series) -> pd.Series:
        clean = self._clean_returns(returns)
        if clean.empty:
            return pd.Series(dtype=float, name="annual_return")
        annual = clean.resample("YE").apply(lambda values: (1.0 + values).prod() - 1.0)
        annual.index = annual.index.year
        annual.name = "annual_return"
        return annual

    def summarize(self, returns: pd.Series, rf: float = 0.0) -> dict[str, float]:
        clean = self._clean_returns(returns)
        if clean.empty:
            return {
                "annual_return": 0.0,
                "annual_volatility": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "calmar": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_duration": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_return": 0.0,
            }

        drawdown = self.compute_drawdown_profile(clean)
        return {
            "annual_return": self.annualized_return(clean),
            "annual_volatility": self.annualized_volatility(clean),
            "sharpe": self.sharpe_ratio(clean, rf=rf),
            "sortino": self.sortino_ratio(clean, rf=rf),
            "calmar": self.calmar_ratio(clean),
            "max_drawdown": drawdown.max_drawdown,
            "max_drawdown_duration": float(drawdown.max_duration),
            "win_rate": self.win_rate(clean),
            "profit_factor": self.profit_factor(clean),
            "total_return": float((1.0 + clean).prod() - 1.0),
        }

    def regime_conditional_performance(
        self,
        returns: pd.Series,
        regime_labels: pd.Series | pd.DataFrame,
        rf: float = 0.0,
    ) -> pd.DataFrame:
        clean = self._clean_returns(returns)
        labels = self._extract_regime_series(regime_labels).reindex(clean.index)
        rows: list[dict[str, float | int]] = []

        for regime in sorted(labels.dropna().unique()):
            mask = labels == regime
            regime_returns = clean.loc[mask]
            summary = self.summarize(regime_returns, rf=rf)
            summary["regime"] = int(regime)
            summary["n_days"] = int(mask.sum())
            rows.append(summary)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("regime").sort_index()

    def generate_report(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series | None = None,
        regime_labels: pd.Series | pd.DataFrame | None = None,
        rf: float = 0.0,
    ) -> pd.DataFrame:
        rows = {"strategy": self.summarize(returns, rf=rf)}
        if benchmark_returns is not None:
            rows["benchmark"] = self.summarize(benchmark_returns, rf=rf)
        report = pd.DataFrame(rows).T

        if regime_labels is not None:
            regime_report = self.regime_conditional_performance(returns, regime_labels, rf=rf)
            for regime, values in regime_report.iterrows():
                report.loc[f"strategy_regime_{regime}", values.index] = values

        return report

    @staticmethod
    def _clean_returns(returns: pd.Series) -> pd.Series:
        series = pd.Series(returns).copy()
        series.index = pd.to_datetime(series.index).tz_localize(None)
        return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().sort_index()

    @staticmethod
    def _extract_regime_series(regime_labels: pd.Series | pd.DataFrame) -> pd.Series:
        if isinstance(regime_labels, pd.DataFrame):
            if "regime" not in regime_labels.columns:
                raise KeyError("Expected regime label DataFrame to contain a 'regime' column.")
            labels = regime_labels["regime"]
        else:
            labels = regime_labels

        normalized = pd.Series(labels).copy()
        normalized.index = pd.to_datetime(normalized.index).tz_localize(None)
        return pd.to_numeric(normalized, errors="coerce")

