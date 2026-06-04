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
    rebalance_interval_days: int = 1
    weighting_method: str = "equal"
    volatility_lookback: int = 21
    volatility_floor: float = 0.005
    max_position_weight: float = 1.0


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
        raw_weights = self.construct_signal_weights(signals, returns=returns)
        target_weights = self.apply_rebalance_schedule(raw_weights)
        applied_weights = target_weights.shift(1).reindex(returns.index).fillna(0.0)
        pnl_returns = returns.fillna(0.0)
        signal_coverage = signals.notna().mean(axis=1).reindex(returns.index).fillna(0.0)
        active_signal_count = signals.notna().sum(axis=1).reindex(returns.index).fillna(0).astype(int)

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
                "gross_exposure": applied_weights.abs().sum(axis=1),
                "net_exposure": applied_weights.sum(axis=1),
                "signal_coverage": signal_coverage,
                "active_signal_count": active_signal_count,
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

    def construct_signal_weights(self, signals: pd.DataFrame, returns: pd.DataFrame | None = None) -> pd.DataFrame:
        """Convert cross-sectional alpha forecasts into daily long/short weights."""
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
        volatility = self._rolling_volatility(returns, signals.index, signals.columns) if returns is not None else None

        for date, row in signals.iterrows():
            clean = pd.to_numeric(row, errors="coerce").dropna()
            if clean.empty:
                continue
            if float(clean.abs().sum()) == 0.0:
                continue

            n_assets = len(clean)
            n_long = self._side_count(n_assets, self.config.long_fraction)
            n_short = self._side_count(n_assets, self.config.short_fraction)
            if n_long == 0 and n_short == 0:
                continue
            ranked = clean.sort_values()

            short_names = ranked.head(n_short).index.tolist() if n_short else []
            long_names = ranked.tail(n_long).index.tolist() if n_long else []

            if set(long_names) & set(short_names):
                # Degenerate one-asset universe; stay flat.
                continue

            long_gross, short_gross = self._side_gross_exposures(bool(long_names), bool(short_names))
            if long_names:
                weights.loc[date, long_names] = self._side_weights(
                    names=long_names,
                    date=date,
                    gross_side=long_gross,
                    volatility=volatility,
                )
            if short_names:
                weights.loc[date, short_names] = -self._side_weights(
                    names=short_names,
                    date=date,
                    gross_side=short_gross,
                    volatility=volatility,
                )

        return weights

    def apply_rebalance_schedule(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Hold target weights between scheduled rebalance dates."""
        interval = max(1, int(self.config.rebalance_interval_days))
        if interval <= 1 or weights.empty:
            return weights

        scheduled = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
        current = pd.Series(0.0, index=weights.columns, dtype=float)
        last_rebalance_pos: int | None = None

        for pos, (date, row) in enumerate(weights.iterrows()):
            has_signal = bool(row.abs().sum() > 0.0)
            has_flat_target = not has_signal and last_rebalance_pos is not None
            should_rebalance = (
                has_flat_target
                or has_signal
                and (last_rebalance_pos is None or pos - last_rebalance_pos >= interval)
            )
            if should_rebalance:
                current = row.copy()
                last_rebalance_pos = pos
            scheduled.loc[date] = current

        return scheduled

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
        momentum_weights = self.construct_signal_weights(momentum_scores, returns=raw_asset_returns)
        momentum_weights = self.apply_rebalance_schedule(momentum_weights)
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

    def _side_weights(
        self,
        names: list[str],
        date: pd.Timestamp,
        gross_side: float,
        volatility: pd.DataFrame | None,
    ) -> pd.Series:
        if self.config.weighting_method != "inverse_volatility" or volatility is None:
            return self._cap_side_weights(pd.Series(gross_side / len(names), index=names, dtype=float), gross_side)

        if date not in volatility.index:
            return self._cap_side_weights(pd.Series(gross_side / len(names), index=names, dtype=float), gross_side)

        vols = volatility.loc[date, names].astype(float).replace([np.inf, -np.inf], np.nan)
        inverse = 1.0 / vols.clip(lower=self.config.volatility_floor)
        inverse = inverse.replace([np.inf, -np.inf], np.nan).dropna()
        if inverse.empty or float(inverse.sum()) <= 0.0:
            return self._cap_side_weights(pd.Series(gross_side / len(names), index=names, dtype=float), gross_side)

        side = pd.Series(0.0, index=names, dtype=float)
        side.loc[inverse.index] = gross_side * inverse / float(inverse.sum())
        missing = side.index[side.eq(0.0)]
        if len(missing) and float(side.sum()) < gross_side:
            side.loc[missing] = (gross_side - float(side.sum())) / len(missing)
        return self._cap_side_weights(side, gross_side)

    def _cap_side_weights(self, weights: pd.Series, gross_side: float) -> pd.Series:
        cap = float(self.config.max_position_weight)
        if cap <= 0.0 or cap >= gross_side or weights.empty:
            return weights

        capped = weights.clip(upper=cap)
        available = capped[capped < cap].index
        leftover = gross_side - float(capped.sum())
        while leftover > 1e-12 and len(available):
            increment = leftover / len(available)
            capped.loc[available] = (capped.loc[available] + increment).clip(upper=cap)
            new_leftover = gross_side - float(capped.sum())
            if abs(new_leftover - leftover) < 1e-12:
                break
            leftover = new_leftover
            available = capped[capped < cap].index
        return capped

    def _rolling_volatility(
        self,
        returns: pd.DataFrame,
        index: pd.Index,
        columns: pd.Index,
    ) -> pd.DataFrame:
        lookback = max(1, int(self.config.volatility_lookback))
        min_periods = min(lookback, max(2, lookback // 3))
        volatility = returns.reindex(columns=columns).rolling(lookback, min_periods=min_periods).std(ddof=0)
        return volatility.reindex(index)

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

    @staticmethod
    def _side_count(n_assets: int, fraction: float) -> int:
        fraction = max(0.0, float(fraction))
        if fraction == 0.0 or n_assets <= 0:
            return 0
        return max(1, int(np.ceil(n_assets * fraction)))

    def _side_gross_exposures(self, has_longs: bool, has_shorts: bool) -> tuple[float, float]:
        gross = float(self.config.max_gross_exposure)
        if has_longs and has_shorts:
            return gross / 2.0, gross / 2.0
        if has_longs:
            return gross, 0.0
        if has_shorts:
            return 0.0, gross
        return 0.0, 0.0
