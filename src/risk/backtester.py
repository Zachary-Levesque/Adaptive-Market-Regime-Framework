"""Artifact-driven AMRF backtester."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.risk.metrics import PerformanceMetrics
from src.risk.stress_test import StressTester

try:  # pragma: no cover - exercised indirectly depending on environment
    from loguru import logger
except ImportError:  # pragma: no cover - dependency may not be installed in CI/local env
    import logging

    logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestConfig:
    max_gross_exposure: float = 1.0
    long_fraction: float = 0.2
    short_fraction: float = 0.2
    transaction_cost_bps: float = 10.0
    benchmark: str = "SPY"
    momentum_lookback: int = 63


@dataclass
class BacktestArtifacts:
    daily_results: pd.DataFrame
    performance_report: pd.DataFrame
    regime_report: pd.DataFrame
    stress_report: pd.DataFrame
    weights: pd.DataFrame


class AMRFBacktester:
    """Run a no-lookahead long/short backtest from saved alpha signals."""

    def __init__(
        self,
        returns: pd.DataFrame,
        alpha_signals: pd.DataFrame,
        regime_labels: pd.DataFrame | pd.Series | None = None,
        config: BacktestConfig | None = None,
    ) -> None:
        self.returns = self._normalize_frame(returns)
        self.alpha_signals = self._normalize_frame(alpha_signals)
        self.regime_labels = self._normalize_regime_labels(regime_labels) if regime_labels is not None else None
        self.config = config or BacktestConfig()
        self.metrics = PerformanceMetrics()
        self.stress_tester = StressTester()

    def run(
        self,
        start: str | None = None,
        end: str | None = None,
        stress_periods: Mapping[str, tuple[str, str] | list[str]] | None = None,
    ) -> BacktestArtifacts:
        returns, signals = self._aligned_inputs(start=start, end=end)
        raw_weights = self.construct_signal_weights(signals)
        applied_weights = raw_weights.shift(1).reindex(returns.index).fillna(0.0)
        pnl_returns = returns.fillna(0.0)

        gross_returns = (applied_weights * pnl_returns).sum(axis=1)
        turnover = applied_weights.diff().abs().sum(axis=1).fillna(applied_weights.abs().sum(axis=1))
        transaction_cost = turnover * (self.config.transaction_cost_bps / 10_000.0)
        strategy_returns = gross_returns - transaction_cost

        benchmark_returns = (
            pnl_returns[self.config.benchmark]
            if self.config.benchmark in pnl_returns.columns
            else pd.Series(0.0, index=returns.index)
        )
        equal_weight_returns = self._equal_weight_returns(pnl_returns)
        momentum_returns = self._momentum_baseline_returns(returns, pnl_returns)

        daily_results = pd.DataFrame(
            {
                "strategy_return_gross": gross_returns,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "strategy_return": strategy_returns,
                "benchmark_return": benchmark_returns,
                "equal_weight_return": equal_weight_returns,
                "momentum_return": momentum_returns,
                "equity": (1.0 + strategy_returns).cumprod(),
                "benchmark_equity": (1.0 + benchmark_returns).cumprod(),
                "equal_weight_equity": (1.0 + equal_weight_returns).cumprod(),
                "momentum_equity": (1.0 + momentum_returns).cumprod(),
            },
            index=returns.index,
        )
        daily_results["drawdown"] = self.metrics.compute_drawdown_profile(strategy_returns).drawdown

        if self.regime_labels is not None:
            daily_results["regime"] = self.regime_labels.reindex(daily_results.index)

        performance_report = self.compare_benchmarks(daily_results)
        regime_report = (
            self.metrics.regime_conditional_performance(strategy_returns, daily_results["regime"])
            if "regime" in daily_results.columns
            else pd.DataFrame()
        )
        stress_report = (
            self.stress_tester.run_stress_test(strategy_returns, stress_periods)
            if stress_periods
            else pd.DataFrame()
        )

        return BacktestArtifacts(
            daily_results=daily_results,
            performance_report=performance_report,
            regime_report=regime_report,
            stress_report=stress_report,
            weights=applied_weights,
        )

    def construct_signal_weights(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Convert cross-sectional alpha forecasts into daily long/short weights."""
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        for date, row in signals.iterrows():
            clean = pd.to_numeric(row, errors="coerce").dropna()
            if clean.empty:
                continue

            n_assets = len(clean)
            n_long = max(1, int(np.ceil(n_assets * self.config.long_fraction)))
            n_short = max(1, int(np.ceil(n_assets * self.config.short_fraction)))
            ranked = clean.sort_values()

            short_names = ranked.head(n_short).index.tolist()
            long_names = ranked.tail(n_long).index.tolist()

            if set(long_names) & set(short_names):
                # Degenerate one-asset universe; stay flat.
                continue

            gross_side = self.config.max_gross_exposure / 2.0
            if long_names:
                weights.loc[date, long_names] = gross_side / len(long_names)
            if short_names:
                weights.loc[date, short_names] = -gross_side / len(short_names)

        return weights

    def compare_benchmarks(self, daily_results: pd.DataFrame) -> pd.DataFrame:
        rows = {
            "strategy": self.metrics.summarize(daily_results["strategy_return"]),
            self.config.benchmark: self.metrics.summarize(daily_results["benchmark_return"]),
        }
        if "equal_weight_return" in daily_results.columns:
            rows["equal_weight"] = self.metrics.summarize(daily_results["equal_weight_return"])
        if "momentum_return" in daily_results.columns:
            rows[f"momentum_{self.config.momentum_lookback}d"] = self.metrics.summarize(
                daily_results["momentum_return"]
            )
        return pd.DataFrame(rows).T

    def save(
        self,
        artifacts: BacktestArtifacts,
        output_dir: str | Path = "data/results",
    ) -> None:
        base = Path(output_dir)
        base.mkdir(parents=True, exist_ok=True)
        artifacts.daily_results.to_parquet(base / "backtest_results.parquet")
        artifacts.performance_report.to_parquet(base / "performance_report.parquet")
        artifacts.weights.to_parquet(base / "position_weights.parquet")
        if not artifacts.regime_report.empty:
            artifacts.regime_report.to_parquet(base / "regime_performance.parquet")
        if not artifacts.stress_report.empty:
            artifacts.stress_report.to_parquet(base / "stress_report.parquet")
        logger.info("Saved Phase 4 backtest outputs to {}", base)

    def _aligned_inputs(self, start: str | None, end: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
        common_index = self.returns.index.intersection(self.alpha_signals.index).sort_values()
        common_columns = self.returns.columns.intersection(self.alpha_signals.columns).sort_values()
        if common_index.empty:
            raise ValueError("returns and alpha_signals have no overlapping dates.")
        if common_columns.empty:
            raise ValueError("returns and alpha_signals have no overlapping tickers.")

        returns = self.returns.loc[common_index, common_columns]
        signals = self.alpha_signals.loc[common_index, common_columns]
        start_timestamp = pd.Timestamp(start) if start is not None else self._first_signal_date(signals)
        if start_timestamp is not None:
            returns = returns.loc[returns.index >= start_timestamp]
            signals = signals.loc[signals.index >= start_timestamp]
        if end is not None:
            returns = returns.loc[returns.index <= pd.Timestamp(end)]
            signals = signals.loc[signals.index <= pd.Timestamp(end)]

        if returns.empty:
            raise ValueError("No backtest dates remain after applying start/end filters.")
        return returns, signals

    @staticmethod
    def _first_signal_date(signals: pd.DataFrame) -> pd.Timestamp | None:
        active_rows = signals.notna().any(axis=1)
        if not active_rows.any():
            return None
        return pd.Timestamp(active_rows[active_rows].index[0])

    def _equal_weight_returns(self, returns: pd.DataFrame) -> pd.Series:
        asset_returns = returns.drop(columns=[self.config.benchmark], errors="ignore")
        if asset_returns.empty:
            asset_returns = returns
        equal_weight = asset_returns.mean(axis=1)
        equal_weight.name = "equal_weight_return"
        return equal_weight

    def _momentum_baseline_returns(self, raw_returns: pd.DataFrame, pnl_returns: pd.DataFrame) -> pd.Series:
        lookback = self.config.momentum_lookback
        raw_asset_returns = raw_returns.drop(columns=[self.config.benchmark], errors="ignore")
        pnl_asset_returns = pnl_returns.drop(columns=[self.config.benchmark], errors="ignore")
        if raw_asset_returns.empty:
            raw_asset_returns = raw_returns
            pnl_asset_returns = pnl_returns

        min_periods = min(lookback, max(5, lookback // 3))
        momentum_scores = (1.0 + raw_asset_returns).rolling(lookback, min_periods=min_periods).apply(
            np.prod,
            raw=True,
        ) - 1.0
        momentum_weights = self.construct_signal_weights(momentum_scores)
        applied_weights = momentum_weights.shift(1).reindex(pnl_asset_returns.index).fillna(0.0)
        gross_returns = (applied_weights * pnl_asset_returns).sum(axis=1)
        turnover = applied_weights.diff().abs().sum(axis=1).fillna(applied_weights.abs().sum(axis=1))
        transaction_cost = turnover * (self.config.transaction_cost_bps / 10_000.0)
        momentum_returns = gross_returns - transaction_cost
        momentum_returns.name = "momentum_return"
        return momentum_returns

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
