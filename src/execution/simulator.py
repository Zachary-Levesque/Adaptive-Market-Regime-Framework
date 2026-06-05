"""Realistic execution simulation for AMRF portfolio weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ETF_TICKERS = {"SPY", "QQQ", "TLT", "GLD", "HYG", "LQD"}
SMALL_MID_TICKERS = {"CIEN"}


@dataclass(frozen=True)
class ExecutionSummary:
    daily_results: pd.DataFrame
    analytics: pd.DataFrame
    executed_weights: pd.DataFrame


class ExecutionSimulator:
    """Convert target portfolio weights into executed weights and costs."""

    def __init__(
        self,
        *,
        transaction_cost_bps: float = 10.0,
        max_single_trade_size: float = 0.10,
        slippage_coefficient: float = 0.35,
        slippage_floor_bps: float = 0.5,
        etf_cost_bps: float = 5.0,
        large_cap_cost_bps: float = 12.0,
        small_mid_cost_bps: float = 18.0,
        adv_lookback: int = 30,
        initial_portfolio_value: float = 1.0,
    ) -> None:
        self.transaction_cost_bps = float(transaction_cost_bps)
        self.max_single_trade_size = float(max_single_trade_size)
        self.slippage_coefficient = float(slippage_coefficient)
        self.slippage_floor_bps = float(slippage_floor_bps)
        self.etf_cost_bps = float(etf_cost_bps)
        self.large_cap_cost_bps = float(large_cap_cost_bps)
        self.small_mid_cost_bps = float(small_mid_cost_bps)
        self.adv_lookback = max(1, int(adv_lookback))
        self.initial_portfolio_value = float(initial_portfolio_value)

    def optimize_rebalance(
        self,
        *,
        current_weights: pd.Series,
        target_weights: pd.Series,
        date: pd.Timestamp,
        prices: pd.DataFrame,
    ) -> pd.Series:
        current = current_weights.reindex(target_weights.index).fillna(0.0).astype(float)
        target = target_weights.reindex(current.index).fillna(0.0).astype(float)

        delta = target - current
        clipped = delta.clip(lower=-self.max_single_trade_size, upper=self.max_single_trade_size)
        executed = current + clipped
        executed = executed.clip(lower=0.0)

        total = float(executed.sum())
        if total <= 0.0 or not np.isfinite(total):
            executed = pd.Series(1.0 / len(executed), index=executed.index, dtype=float)
        else:
            executed = executed / total
        return executed.astype(float)

    def simulate(
        self,
        target_weights: pd.DataFrame,
        returns: pd.DataFrame,
        prices: pd.DataFrame,
        *,
        initial_weights: pd.Series | None = None,
    ) -> ExecutionSummary:
        aligned_returns, aligned_targets, aligned_prices = self._align_inputs(target_weights, returns, prices)
        assets = list(aligned_targets.columns)
        current_weights = (
            initial_weights.reindex(assets).fillna(0.0).astype(float)
            if initial_weights is not None
            else pd.Series(1.0 / len(assets), index=assets, dtype=float)
        )

        daily_rows: list[dict[str, float | str]] = []
        executed_rows: list[pd.Series] = []
        analytics_rows: list[dict[str, float | str]] = []

        equity = self.initial_portfolio_value
        equity_curve: list[float] = []
        peak = self.initial_portfolio_value

        for i, date in enumerate(aligned_targets.index[:-1]):
            target = aligned_targets.loc[date].astype(float)
            executed = self.optimize_rebalance(
                current_weights=current_weights,
                target_weights=target,
                date=date,
                prices=aligned_prices,
            )
            next_date = aligned_targets.index[i + 1]
            next_returns = aligned_returns.loc[next_date].fillna(0.0).astype(float)

            turnover = float(np.abs(executed - current_weights).sum())
            asset_costs = self._cost_breakdown(date=date, executed=executed, current_weights=current_weights, prices=aligned_prices)
            gross_return = float((executed * next_returns).sum())
            transaction_cost = float(asset_costs["total_cost"]) + turnover * (self.transaction_cost_bps / 10_000.0)
            net_return = gross_return - transaction_cost

            equity *= 1.0 + net_return
            peak = max(peak, equity)
            drawdown = 1.0 - equity / max(peak, 1e-12)
            equity_curve.append(equity)

            daily_rows.append(
                {
                    "date": next_date,
                    "target_return": gross_return,
                    "gross_return": gross_return,
                    "transaction_cost": transaction_cost,
                    "net_return": net_return,
                    "turnover": turnover,
                    "slippage_bps": float(asset_costs["slippage_bps_mean"]),
                    "portfolio_value": equity,
                    "drawdown": drawdown,
                }
            )
            executed_rows.append(executed.rename(index=str))
            current_weights = executed

            for asset in assets:
                analytics_rows.append(
                    {
                        "date": next_date,
                        "asset": asset,
                        "tier": asset_costs["tiers"].get(asset, "large_cap"),
                        "turnover": float(abs(executed.get(asset, 0.0) - current_weights.get(asset, 0.0))),
                        "slippage_bps": float(asset_costs["asset_slippage_bps"].get(asset, 0.0)),
                        "cost_bps": float(asset_costs["asset_cost_bps"].get(asset, 0.0)),
                        "transaction_cost": float(asset_costs["asset_costs"].get(asset, 0.0)),
                        "adv_dollar": float(asset_costs["adv_dollar"].get(asset, np.nan)),
                    }
                )

        daily_results = pd.DataFrame(daily_rows).set_index("date")
        executed_weights = pd.DataFrame(executed_rows, index=daily_results.index, columns=assets)
        analytics = pd.DataFrame(analytics_rows)
        return ExecutionSummary(daily_results=daily_results, analytics=analytics, executed_weights=executed_weights)

    def _cost_breakdown(
        self,
        *,
        date: pd.Timestamp,
        executed: pd.Series,
        current_weights: pd.Series,
        prices: pd.DataFrame,
    ) -> dict[str, object]:
        asset_costs: dict[str, float] = {}
        asset_cost_bps: dict[str, float] = {}
        asset_slippage_bps: dict[str, float] = {}
        adv_dollar: dict[str, float] = {}
        tiers: dict[str, str] = {}

        for asset in executed.index:
            tier_bps, tier = self._tier_cost(asset)
            tiers[asset] = tier
            adv = self._adv_dollar(prices, asset, date)
            adv_dollar[asset] = adv
            trade_weight = abs(float(executed[asset] - current_weights.get(asset, 0.0)))
            slippage_bps = self._slippage_bps(trade_weight, adv)
            asset_slippage_bps[asset] = slippage_bps
            cost_bps = tier_bps + slippage_bps
            asset_cost_bps[asset] = cost_bps
            asset_costs[asset] = trade_weight * cost_bps / 10_000.0

        total_cost = float(sum(asset_costs.values()))
        mean_slippage = float(np.mean(list(asset_slippage_bps.values()))) if asset_slippage_bps else 0.0
        return {
            "asset_costs": asset_costs,
            "asset_cost_bps": asset_cost_bps,
            "asset_slippage_bps": asset_slippage_bps,
            "slippage_bps_mean": mean_slippage,
            "total_cost": total_cost,
            "adv_dollar": adv_dollar,
            "tiers": tiers,
        }

    def _tier_cost(self, asset: str) -> tuple[float, str]:
        if asset in ETF_TICKERS:
            return self.etf_cost_bps, "etf"
        if asset in SMALL_MID_TICKERS:
            return self.small_mid_cost_bps, "small_mid"
        return self.large_cap_cost_bps, "large_cap"

    def _adv_dollar(self, prices: pd.DataFrame, asset: str, date: pd.Timestamp) -> float:
        if not isinstance(prices.columns, pd.MultiIndex):
            return 0.0
        if asset not in prices.columns.get_level_values(0):
            return 0.0
        try:
            frame = prices[asset][["Close", "Volume"]].copy()
        except Exception:
            return 0.0
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        window = frame.loc[:date].tail(self.adv_lookback)
        if window.empty:
            return 0.0
        adv = (window["Close"].fillna(method="ffill") * window["Volume"].fillna(0.0)).mean()
        return float(adv if np.isfinite(adv) else 0.0)

    def _slippage_bps(self, trade_weight: float, adv_dollar: float) -> float:
        if trade_weight <= 0.0 or adv_dollar <= 0.0:
            return self.slippage_floor_bps
        order_value = trade_weight * self.initial_portfolio_value
        impact = self.slippage_coefficient * (order_value / max(adv_dollar, 1e-12)) * 10_000.0
        return float(max(self.slippage_floor_bps, impact))

    @staticmethod
    def _align_inputs(
        target_weights: pd.DataFrame,
        returns: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        common_index = target_weights.index.intersection(returns.index).sort_values()
        common_assets = target_weights.columns.intersection(returns.columns)
        if common_index.empty or common_assets.empty:
            raise ValueError("No overlap between target weights and returns for execution simulation.")
        aligned_targets = target_weights.loc[common_index, common_assets].copy()
        aligned_returns = returns.loc[common_index, common_assets].copy()
        aligned_prices = prices.loc[common_index, prices.columns.get_level_values(0).isin(common_assets)].copy()
        return aligned_returns, aligned_targets, aligned_prices
