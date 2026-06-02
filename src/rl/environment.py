"""Reinforcement Learning environment for position sizing tilts."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class TradingEnv(gym.Env):
    """
    Gymnasium environment that learns to tilt alpha-signal weights 
    based on regime conviction and market state.
    """
    def __init__(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
        regime_probs: pd.DataFrame,
        initial_capital: float = 100000.0,
        max_tilt: float = 0.5, # Max +/- 50% tilt on alpha weights
    ):
        super().__init__()
        
        # Align data
        common_idx = signals.index.intersection(returns.index).intersection(regime_probs.index).sort_values()
        self.signals = signals.loc[common_idx]
        self.returns = returns.loc[common_idx]
        self.regime_probs = regime_probs.loc[common_idx]
        self.tickers = signals.columns.tolist()
        
        self.initial_capital = initial_capital
        self.max_tilt = max_tilt
        
        # Observation space: 
        # [Alpha Signals for each ticker] + [Regime Probabilities]
        num_features = len(self.tickers) + self.regime_probs.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32
        )
        
        # Action space: Continuous tilts for each ticker
        self.action_space = spaces.Box(
            low=-max_tilt, high=max_tilt, shape=(len(self.tickers),), dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.history = []
        
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # 1. Get current alpha weights
        raw_signals = self.signals.iloc[self.current_step].fillna(0).values
        # Simple normalization to sum to 1.0 gross (0.5 long, 0.5 short)
        pos_sum = np.sum(np.maximum(0, raw_signals))
        neg_sum = np.abs(np.sum(np.minimum(0, raw_signals)))
        
        weights = np.zeros_like(raw_signals)
        if pos_sum > 0: weights += np.maximum(0, raw_signals) / (2.0 * pos_sum)
        if neg_sum > 0: weights -= np.abs(np.minimum(0, raw_signals)) / (2.0 * neg_sum)
        
        # 2. Apply RL tilts
        # Tilt formula: weight * (1 + action)
        # We ensure actions are clipped to [-max_tilt, max_tilt]
        tilted_weights = weights * (1.0 + action)
        
        # Re-normalize to ensure we don't exceed max leverage
        gross_exposure = np.sum(np.abs(tilted_weights))
        if gross_exposure > 1.0:
            tilted_weights /= gross_exposure
            
        # 3. Calculate step returns
        asset_returns = self.returns.iloc[self.current_step].fillna(0).values
        step_return = np.sum(tilted_weights * asset_returns)
        
        # 4. Update state
        old_value = self.portfolio_value
        self.portfolio_value *= (1.0 + step_return)
        self.history.append(step_return)
        
        self.current_step += 1
        done = self.current_step >= len(self.signals) - 1
        
        # 5. Reward function: Cumulative Sharpe-like reward
        reward = step_return
        if done:
            hist = np.array(self.history)
            if len(hist) > 1 and np.std(hist) > 0:
                sharpe = (np.mean(hist) / np.std(hist)) * np.sqrt(252)
                reward += sharpe * 10.0 # Significant terminal reward for Sharpe
        
        return self._get_obs(), float(reward), done, False, {}

    def _get_obs(self):
        if self.current_step >= len(self.signals):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
            
        alpha_obs = self.signals.iloc[self.current_step].fillna(0).values
        regime_obs = self.regime_probs.iloc[self.current_step].fillna(0).values
        
        # Combine and ensure no NaNs or Infs
        obs = np.concatenate([alpha_obs, regime_obs]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs
