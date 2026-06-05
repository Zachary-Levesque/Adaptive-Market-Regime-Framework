"""Gymnasium environment for AMRF PPO position sizing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.risk.backtester import AMRFBacktester, BacktestConfig


@dataclass(frozen=True)
class ObservationStats:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray


class TradingEnvironment(gym.Env):
    """Long-only portfolio sizing environment with regime-aware state."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns: pd.DataFrame,
        signal_scores: pd.DataFrame,
        regime_probs: pd.DataFrame,
        prices: pd.DataFrame,
        regime_features: pd.DataFrame,
        *,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        stats: ObservationStats | None = None,
        episode_length: int | None = None,
        random_start: bool = False,
        initial_portfolio_value: float = 1.0,
        transaction_cost_bps: float = 10.0,
        max_gross_exposure: float = 1.0,
        long_fraction: float = 0.2,
        short_fraction: float = 0.2,
        weighting_method: str = "equal",
        volatility_lookback: int = 21,
        volatility_floor: float = 0.005,
        max_position_weight: float = 1.0,
        rebalance_interval_days: int = 1,
        drawdown_penalty_threshold: float = 0.15,
        drawdown_penalty_scale: float = 4.0,
        action_regularization_scale: float = 0.1,
        reward_scale: float = 10.0,
        sharpe_window: int = 63,
        max_drawdown_stop: float = 0.40,
        rebalance_deadband: float = 0.0025,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.returns = self._align_frame(returns, start, end)
        self.signal_scores = self._align_frame(signal_scores, start, end).reindex(columns=self.returns.columns)
        self.regime_probs = self._align_frame(regime_probs, start, end)
        self.prices = self._align_prices(prices, self.returns.index, self.returns.columns)
        self.regime_features = self._align_frame(regime_features, start, end)
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.assets = list(self.returns.columns)
        self.n_assets = len(self.assets)
        self.episode_length = episode_length
        self.random_start = random_start
        self.initial_portfolio_value = float(initial_portfolio_value)
        self.transaction_cost_bps = float(transaction_cost_bps)
        self.max_gross_exposure = float(max_gross_exposure)
        self.long_fraction = float(long_fraction)
        self.short_fraction = float(short_fraction)
        self.weighting_method = str(weighting_method)
        self.volatility_lookback = max(1, int(volatility_lookback))
        self.volatility_floor = float(volatility_floor)
        self.max_position_weight = float(max_position_weight)
        self.rebalance_interval_days = max(1, int(rebalance_interval_days))
        self.drawdown_penalty_threshold = float(drawdown_penalty_threshold)
        self.drawdown_penalty_scale = float(drawdown_penalty_scale)
        self.action_regularization_scale = float(action_regularization_scale)
        self.reward_scale = float(reward_scale)
        self.sharpe_window = max(1, int(sharpe_window))
        self.max_drawdown_stop = float(max_drawdown_stop)
        self.rebalance_deadband = float(rebalance_deadband)
        self.n_regimes = int(self.regime_probs.shape[1])
        self._rng = np.random.default_rng(seed)

        self.backtest_config = BacktestConfig(
            max_gross_exposure=self.max_gross_exposure,
            long_fraction=self.long_fraction,
            short_fraction=self.short_fraction,
            transaction_cost_bps=self.transaction_cost_bps,
            rebalance_interval_days=self.rebalance_interval_days,
            weighting_method=self.weighting_method,
            volatility_lookback=self.volatility_lookback,
            volatility_floor=self.volatility_floor,
            max_position_weight=self.max_position_weight,
        )
        self.signal_weights = self._build_static_signal_weights()
        self.rolling_vol = self.returns.rolling(self.volatility_lookback, min_periods=5).std(ddof=0).fillna(0.0)
        self.vix_series = self._resolve_vix_series(self.regime_features, self.returns.index)
        self._dates = self.returns.index.to_list()
        self._observation_feature_names = tuple(self._build_observation_feature_names())
        self._stats = stats or self._build_stats()

        obs_dim = len(self._observation_feature_names)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.5, high=1.5, shape=(self.n_assets,), dtype=np.float32)

        self._start_pos = 0
        self._end_pos = len(self._dates) - 1
        self._episode_start = 0
        self._episode_end = len(self._dates) - 1
        self._position = 0
        self._portfolio_weights = pd.Series(1.0 / self.n_assets, index=self.assets, dtype=float)
        self._portfolio_value = self.initial_portfolio_value
        self._portfolio_return_history: list[float] = []
        self._days_since_rebalance = 0
        self._days_since_regime_change = 0
        self._last_regime = None
        self._last_trade_weights = self._portfolio_weights.copy()
        self._last_drawdown = 0.0

        self.reset(seed=seed)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._set_episode_bounds()
        self._position = self._episode_start
        self._portfolio_weights = pd.Series(1.0 / self.n_assets, index=self.assets, dtype=float)
        self._portfolio_value = self.initial_portfolio_value
        self._portfolio_return_history = []
        self._days_since_rebalance = 0
        self._days_since_regime_change = 0
        self._last_regime = self._current_regime_label(self._position)
        self._last_trade_weights = self._portfolio_weights.copy()
        self._last_drawdown = 0.0
        return self._get_observation(self._position), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.n_assets:
            raise ValueError(f"Expected action shape {(self.n_assets,)}, got {action.shape}.")

        current_date = self._dates[self._position]
        baseline_weights = self._baseline_weights(current_date)
        target_weights = self._action_to_weights(action, current_date)
        turnover = float(np.abs(target_weights - self._portfolio_weights).sum())
        trade_weights = self._apply_rebalance_deadband(target_weights)
        turnover = float(np.abs(trade_weights - self._portfolio_weights).sum())

        next_pos = self._position + 1
        if next_pos >= len(self._dates):
            observation = self._get_observation(self._position)
            return observation, 0.0, True, False, {"portfolio_value": self._portfolio_value}

        next_returns = self.returns.iloc[next_pos].fillna(0.0)
        gross_return = float((trade_weights * next_returns).sum())
        baseline_return = float((baseline_weights * next_returns).sum())
        transaction_cost = turnover * (self.transaction_cost_bps / 10_000.0)
        portfolio_return = gross_return - transaction_cost

        prev_value = self._portfolio_value
        self._portfolio_value *= 1.0 + portfolio_return
        self._portfolio_return_history.append(portfolio_return)
        self._portfolio_weights = trade_weights
        self._last_trade_weights = trade_weights
        # `_apply_rebalance_deadband` already updates the counter.

        current_drawdown = self._current_drawdown()
        if self._current_regime_label(next_pos) == self._last_regime:
            self._days_since_regime_change += 1
        else:
            self._days_since_regime_change = 0
            self._last_regime = self._current_regime_label(next_pos)
        self._last_drawdown = current_drawdown

        rolling_vol = self._rolling_strategy_vol()
        risk_adjusted_return = portfolio_return
        baseline_risk_adjusted = baseline_return
        drawdown_penalty = max(0.0, current_drawdown - self.drawdown_penalty_threshold) * self.drawdown_penalty_scale
        action_penalty = self.action_regularization_scale * float(
            np.mean(np.square(target_weights.values - baseline_weights.values))
        )
        reward = (
            self.reward_scale * (risk_adjusted_return + 0.5 * (risk_adjusted_return - baseline_risk_adjusted))
            - drawdown_penalty
            - action_penalty
        )
        reward = float(np.clip(reward, -1.0, 1.0))

        terminated = next_pos >= self._episode_end
        truncated = bool(current_drawdown >= self.max_drawdown_stop)
        self._position = next_pos
        observation = self._get_observation(self._position)
        info = {
            "portfolio_value": self._portfolio_value,
            "portfolio_return": portfolio_return,
            "gross_return": gross_return,
            "transaction_cost": transaction_cost,
            "turnover": turnover,
            "drawdown": current_drawdown,
            "rolling_vol": rolling_vol,
            "risk_adjusted_return": risk_adjusted_return,
            "action_penalty": action_penalty,
            "prev_portfolio_value": prev_value,
            "target_turnover": turnover,
        }
        return observation, reward, terminated, truncated, info

    def render(self):  # pragma: no cover - no interactive render path
        return None

    def _set_episode_bounds(self) -> None:
        eligible_positions = np.arange(0, len(self._dates) - 1)
        if len(eligible_positions) == 0:
            raise ValueError("TradingEnvironment requires at least two aligned observations.")

        if self.random_start and self.episode_length is not None and self.episode_length > 1:
            max_start = max(0, len(self._dates) - self.episode_length - 1)
            self._episode_start = int(self._rng.integers(0, max_start + 1)) if max_start > 0 else 0
            self._episode_end = min(len(self._dates) - 1, self._episode_start + self.episode_length - 1)
        else:
            self._episode_start = 0
            self._episode_end = len(self._dates) - 1

    def _get_observation(self, position: int) -> np.ndarray:
        if position >= len(self._dates):
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        raw = self._raw_observation(position)
        if raw.shape[0] != self._stats.mean.shape[0]:
            raise ValueError(
                f"Observation stats dimension mismatch: expected {raw.shape[0]}, got {self._stats.mean.shape[0]}"
            )
        obs = (raw - self._stats.mean) / np.where(self._stats.std == 0.0, 1.0, self._stats.std)
        return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)

    def _action_to_weights(self, action: np.ndarray, date: pd.Timestamp) -> pd.Series:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        base_weights = self._baseline_weights(date)
        scaled = base_weights.values * action
        gross = float(np.abs(scaled).sum())
        base_gross = float(np.abs(base_weights.values).sum())
        if gross <= 0.0 or not np.isfinite(gross) or base_gross <= 0.0 or not np.isfinite(base_gross):
            weights = base_weights.values.astype(float)
        else:
            weights = scaled / gross * base_gross
        return pd.Series(weights, index=self.assets, dtype=float)

    def _apply_rebalance_deadband(self, target_weights: pd.Series) -> pd.Series:
        delta = target_weights - self._portfolio_weights
        if float(delta.abs().sum()) <= self.rebalance_deadband:
            self._days_since_rebalance += 1
            return self._portfolio_weights.copy()
        self._days_since_rebalance = 0
        return target_weights.copy()

    def _rolling_strategy_vol(self) -> float:
        history = pd.Series(self._portfolio_return_history, dtype=float)
        if history.empty:
            return 1.0
        window = history.tail(self.sharpe_window)
        vol = float(window.std(ddof=0))
        return vol if vol > 0.0 else 1.0

    def _rolling_sharpe(self) -> float:
        history = pd.Series(self._portfolio_return_history, dtype=float)
        if history.size < 2:
            return 0.0
        window = history.tail(self.sharpe_window)
        vol = float(window.std(ddof=0))
        if vol <= 0.0:
            return 0.0
        return float(np.sqrt(252.0) * window.mean() / vol)

    def _current_drawdown(self) -> float:
        return self._drawdown_from_history(self._portfolio_return_history)

    @staticmethod
    def _drawdown_from_history(history: list[float]) -> float:
        if not history:
            return 0.0
        equity = np.cumprod(1.0 + np.asarray(history, dtype=float))
        peak = np.maximum.accumulate(equity)
        drawdown = 1.0 - equity[-1] / max(peak[-1], 1e-12)
        return float(max(0.0, drawdown))

    def _rolling_sharpe_from_history(self, history: list[float]) -> float:
        series = pd.Series(history, dtype=float)
        if series.size < 2:
            return 0.0
        window = series.tail(self.sharpe_window)
        vol = float(window.std(ddof=0))
        if vol <= 0.0:
            return 0.0
        return float(np.sqrt(252.0) * window.mean() / vol)

    def _current_regime_label(self, position: int) -> int:
        probs = self.regime_probs.iloc[position].fillna(0.0).to_numpy(dtype=float)
        return int(np.argmax(probs))

    def _current_regime_one_hot(self, position: int) -> np.ndarray:
        label = self._current_regime_label(position)
        one_hot = np.zeros(self.n_regimes, dtype=float)
        one_hot[label] = 1.0
        return one_hot

    def _vix_value(self, date: pd.Timestamp) -> float:
        if date in self.vix_series.index:
            return float(self.vix_series.loc[date])
        return float(self.vix_series.iloc[-1]) if not self.vix_series.empty else 0.0

    def _build_static_signal_weights(self) -> pd.DataFrame:
        backtester = AMRFBacktester(
            returns=self.returns,
            alpha_signals=self.signal_scores,
            config=self.backtest_config,
        )
        raw_weights = backtester.construct_signal_weights(self.signal_scores.loc[self.returns.index], returns=self.returns)
        target_weights = backtester.apply_rebalance_schedule(raw_weights)
        target_weights = target_weights.reindex(index=self.returns.index, columns=self.assets).fillna(0.0)
        return target_weights.astype(float)

    def _build_stats(self) -> ObservationStats:
        portfolio_weights = pd.Series(1.0 / self.n_assets, index=self.assets, dtype=float)
        portfolio_value = self.initial_portfolio_value
        portfolio_history: list[float] = []
        days_since_rebalance = 0
        days_since_regime_change = 0
        last_regime = self._current_regime_label(0) if self._dates else 0
        rows: list[np.ndarray] = []

        for position, date in enumerate(self._dates):
            rows.append(
                self._raw_observation(
                    position,
                    portfolio_weights=portfolio_weights,
                    portfolio_value=portfolio_value,
                    portfolio_history=portfolio_history,
                    days_since_rebalance=days_since_rebalance,
                    days_since_regime_change=days_since_regime_change,
                )
            )
            if position >= len(self._dates) - 1:
                break

            baseline = self._baseline_weights(date)
            turnover = float(np.abs(baseline - portfolio_weights).sum())
            if turnover <= self.rebalance_deadband:
                executed = portfolio_weights.copy()
                days_since_rebalance += 1
            else:
                executed = baseline.copy()
                days_since_rebalance = 0

            next_returns = self.returns.iloc[position + 1].fillna(0.0)
            portfolio_return = float((executed * next_returns).sum() - turnover * (self.transaction_cost_bps / 10_000.0))
            portfolio_value *= 1.0 + portfolio_return
            portfolio_history.append(portfolio_return)

            next_regime = self._current_regime_label(position + 1)
            if next_regime == last_regime:
                days_since_regime_change += 1
            else:
                days_since_regime_change = 0
                last_regime = next_regime
            portfolio_weights = executed

        frame = pd.DataFrame(rows, columns=self._observation_feature_names)
        mean = frame.mean(axis=0).to_numpy(dtype=float)
        std = frame.std(axis=0, ddof=0).replace(0.0, 1.0).to_numpy(dtype=float)
        return ObservationStats(feature_names=self._observation_feature_names, mean=mean, std=std)

    def _raw_observation(
        self,
        position: int,
        *,
        portfolio_weights: pd.Series | None = None,
        portfolio_value: float | None = None,
        portfolio_history: list[float] | None = None,
        days_since_rebalance: int | None = None,
        days_since_regime_change: int | None = None,
    ) -> np.ndarray:
        if position >= len(self._dates):
            return np.zeros(len(self._observation_feature_names), dtype=float)

        date = self._dates[position]
        portfolio_weights = (
            portfolio_weights.reindex(self.assets).fillna(0.0).astype(float)
            if portfolio_weights is not None
            else self._portfolio_weights.reindex(self.assets).fillna(0.0).astype(float)
        )
        _ = float(portfolio_value if portfolio_value is not None else self._portfolio_value)
        portfolio_history = list(portfolio_history if portfolio_history is not None else self._portfolio_return_history)
        days_since_rebalance = int(days_since_rebalance if days_since_rebalance is not None else self._days_since_rebalance)
        days_since_regime_change = int(
            days_since_regime_change if days_since_regime_change is not None else self._days_since_regime_change
        )

        return np.concatenate(
            [
                self._current_regime_one_hot(position),
                self._baseline_weights(date).values.astype(float),
                self.rolling_vol.loc[date].reindex(self.assets).fillna(0.0).astype(float).values,
                portfolio_weights.values.astype(float),
                np.array([self._drawdown_from_history(portfolio_history)], dtype=float),
                np.array([float(days_since_rebalance)], dtype=float),
                np.array([self._rolling_sharpe_from_history(portfolio_history)], dtype=float),
                np.array([self._vix_value(date)], dtype=float),
                np.array([float(days_since_regime_change)], dtype=float),
            ]
        )

    def _baseline_weights(self, date: pd.Timestamp) -> pd.Series:
        base_weights = self.signal_weights.loc[date].reindex(self.assets).fillna(0.0).astype(float)
        gross = float(np.abs(base_weights.values).sum())
        if gross <= 0.0 or not np.isfinite(gross):
            return pd.Series(1.0 / self.n_assets, index=self.assets, dtype=float)
        return base_weights

    def _build_observation_feature_names(self) -> list[str]:
        return (
            [f"regime_{idx}" for idx in range(self.n_regimes)]
            + [f"signal_{asset}" for asset in self.assets]
            + [f"vol_{asset}" for asset in self.assets]
            + [f"portfolio_{asset}" for asset in self.assets]
            + ["drawdown", "days_since_rebalance", "rolling_sharpe", "vix", "days_since_regime_change"]
        )

    @staticmethod
    def _align_frame(frame: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
        aligned = frame.copy().sort_index()
        aligned.index = pd.to_datetime(aligned.index).tz_localize(None)
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        return aligned.loc[(aligned.index >= start_ts) & (aligned.index <= end_ts)]

    @staticmethod
    def _align_prices(prices: pd.DataFrame, dates: pd.Index, assets: pd.Index) -> pd.DataFrame:
        if not isinstance(prices.columns, pd.MultiIndex):
            raise TypeError("Expected prices to be a MultiIndex frame.")
        subset = prices.loc[dates, prices.columns.get_level_values(0).isin(assets)].copy()
        return subset.sort_index(axis=1)

    @staticmethod
    def _resolve_vix_series(regime_features: pd.DataFrame, index: pd.Index) -> pd.Series:
        if "vix_level" in regime_features.columns:
            series = pd.to_numeric(regime_features["vix_level"], errors="coerce")
        elif "VIXCLS" in regime_features.columns:
            series = pd.to_numeric(regime_features["VIXCLS"], errors="coerce")
        else:
            series = pd.Series(0.0, index=index)
        series.index = pd.to_datetime(series.index).tz_localize(None)
        return series.reindex(index).ffill().bfill().fillna(0.0)
