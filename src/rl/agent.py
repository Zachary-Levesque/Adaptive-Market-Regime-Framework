"""PPO agent orchestration for AMRF Module 6."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.config import AppConfig
from src.risk.metrics import PerformanceMetrics
from src.rl.data import RLDataset, split_by_date
from src.rl.environment import ObservationStats, TradingEnvironment


@dataclass(frozen=True)
class RolloutArtifacts:
    positions: pd.DataFrame
    results: pd.DataFrame
    comparison: pd.DataFrame


class TrainingMetricsCallback(BaseCallback):
    """Record validation metrics to tensorboard and a parquet-friendly history."""

    def __init__(self, eval_env: TradingEnvironment, eval_freq: int = 10_000, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.metrics = PerformanceMetrics()
        self.history: list[dict[str, float]] = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        summary = evaluate_agent(self.model, self.eval_env)
        row = {
            "step": float(self.num_timesteps),
            "episode_return": float(summary["total_return"]),
            "portfolio_sharpe": float(summary["sharpe"]),
            "max_drawdown": float(summary["max_drawdown"]),
            "sortino": float(summary["sortino"]),
            "calmar": float(summary["calmar"]),
        }
        self.history.append(row)
        for key, value in row.items():
            self.logger.record(f"validation/{key}", value)
        return True


class PPOPositionSizingAgent:
    """Train and evaluate a PPO agent over the AMRF selected alpha signal."""

    def __init__(self, config: AppConfig, dataset: RLDataset) -> None:
        self.config = config
        self.dataset = dataset
        self.metrics = PerformanceMetrics()

    def build_environment(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        *,
        random_start: bool,
        episode_length: int | None,
        stats: ObservationStats | None = None,
    ) -> TradingEnvironment:
        return TradingEnvironment(
            returns=self.dataset.returns,
            signal_scores=self.dataset.selected_signal,
            regime_probs=self.dataset.regime_probs,
            prices=self.dataset.prices,
            regime_features=self.dataset.regime_features,
            start=start,
            end=end,
            stats=stats,
            episode_length=episode_length,
            random_start=random_start,
            initial_portfolio_value=self.config.rl.initial_capital,
            transaction_cost_bps=self.config.risk.transaction_cost_bps,
            drawdown_penalty_threshold=0.15,
            drawdown_penalty_scale=4.0,
            rolling_vol_window=self.config.risk.volatility_lookback,
            sharpe_window=63,
            max_drawdown_stop=0.40,
        )

    def build_stats(self) -> ObservationStats:
        train_returns = split_by_date(self.dataset.returns, self.config.rl.train_start, self.config.rl.train_end)
        train_signals = split_by_date(self.dataset.selected_signal, self.config.rl.train_start, self.config.rl.train_end)
        train_regime_features = split_by_date(
            self.dataset.regime_features, self.config.rl.train_start, self.config.rl.train_end
        )
        train_vix = train_regime_features["vix_level"] if "vix_level" in train_regime_features.columns else pd.Series(0.0, index=train_returns.index)
        signal_mean = train_signals.mean(axis=0).fillna(0.0)
        signal_std = train_signals.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        vol = train_returns.rolling(self.config.risk.volatility_lookback, min_periods=5).std(ddof=0)
        vol_mean = vol.mean(axis=0).fillna(0.0)
        vol_std = vol.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        return ObservationStats(
            signal_mean=signal_mean,
            signal_std=signal_std,
            vol_mean=vol_mean,
            vol_std=vol_std,
            vix_mean=float(train_vix.mean()),
            vix_std=float(train_vix.std(ddof=0) or 1.0),
        )

    def train(self, total_timesteps: int | None = None) -> tuple[PPO, pd.DataFrame]:
        stats = self.build_stats()
        def make_train_env() -> Monitor:
            return Monitor(
                self.build_environment(
                    self.config.rl.train_start,
                    self.config.rl.train_end,
                    random_start=True,
                    episode_length=252,
                    stats=stats,
                )
            )

        def make_validation_env() -> Monitor:
            return Monitor(
                self.build_environment(
                    self.config.rl.validation_start,
                    self.config.rl.validation_end,
                    random_start=False,
                    episode_length=None,
                    stats=stats,
                )
            )

        train_eval_env = self.build_environment(
            self.config.rl.validation_start,
            self.config.rl.validation_end,
            random_start=False,
            episode_length=None,
            stats=stats,
        )

        vec_env = DummyVecEnv([make_train_env])
        eval_vec_env = DummyVecEnv([make_validation_env])
        policy_kwargs = {"net_arch": dict(pi=[256, 256, 128], vf=[256, 256, 128])}

        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=self.config.rl.learning_rate,
            n_steps=self.config.rl.n_steps,
            batch_size=self.config.rl.batch_size,
            n_epochs=self.config.rl.n_epochs,
            gamma=self.config.rl.gamma,
            ent_coef=0.01,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(self.config.rl.model_path.parent / "tensorboard"),
            device="cpu",
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=50_000,
            save_path=str(self.config.rl.model_path.parent / "checkpoints"),
            name_prefix="ppo_position_sizer",
        )
        eval_callback = EvalCallback(
            eval_vec_env,
            best_model_save_path=str(self.config.rl.model_path.parent / "best_model"),
            log_path=str(self.config.rl.model_path.parent / "eval_logs"),
            eval_freq=10_000,
            deterministic=True,
            render=False,
            n_eval_episodes=1,
        )
        metrics_callback = TrainingMetricsCallback(train_eval_env, eval_freq=10_000)
        callback = CallbackList([checkpoint_callback, eval_callback, metrics_callback])

        timesteps = int(total_timesteps or self.config.rl.total_timesteps)
        model.learn(total_timesteps=timesteps, callback=callback)

        best_model_path = self.config.rl.model_path.parent / "best_model" / "best_model.zip"
        if best_model_path.exists():
            model = PPO.load(best_model_path, env=vec_env)

        self.save_model(model)
        history = pd.DataFrame(metrics_callback.history)
        if not history.empty:
            history.to_parquet(self.config.rl.training_history_path)
        return model, history

    def save_model(self, model: PPO) -> Path:
        path = self.config.rl.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        return path

    def load_model(self, env: DummyVecEnv | None = None) -> PPO:
        path = self.config.rl.model_path
        if not path.exists() and not path.with_suffix(".zip").exists():
            raise FileNotFoundError(f"RL model not found: {path}")
        return PPO.load(path, env=env)

    def backtest(
        self,
        model: PPO,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        *,
        stats: ObservationStats | None = None,
        execution_layer: Any | None = None,
    ) -> RolloutArtifacts:
        env = self.build_environment(start, end, random_start=False, episode_length=None, stats=stats)
        return rollout_policy(model, env, execution_layer=execution_layer)


def rollout_policy(
    model: PPO,
    env: TradingEnvironment,
    *,
    execution_layer: Any | None = None,
) -> RolloutArtifacts:
    """Roll out a deterministic policy on a TradingEnvironment."""
    obs, _ = env.reset()
    done = False
    positions: list[pd.Series] = []
    results: list[dict[str, float | str]] = []
    dates: list[pd.Timestamp] = []
    current_weights = pd.Series(1.0 / env.n_assets, index=env.assets, dtype=float)

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        target_weights = env._action_to_weights(np.asarray(action), env._dates[env._position])
        if execution_layer is not None:
            executed = execution_layer.optimize_rebalance(
                current_weights=current_weights,
                target_weights=target_weights,
                date=env._dates[env._position],
                prices=env.prices,
            )
        else:
            executed = target_weights

        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        current_weights = executed.reindex(env.assets).fillna(0.0)
        dates.append(env._dates[env._position])
        positions.append(current_weights.copy())
        results.append(
            {
                "date": env._dates[env._position].isoformat(),
                "reward": float(reward),
                "portfolio_return": float(info.get("portfolio_return", 0.0)),
                "gross_return": float(info.get("gross_return", 0.0)),
                "transaction_cost": float(info.get("transaction_cost", 0.0)),
                "turnover": float(info.get("turnover", 0.0)),
                "portfolio_value": float(info.get("portfolio_value", env.initial_portfolio_value)),
                "drawdown": float(info.get("drawdown", 0.0)),
                "rolling_vol": float(info.get("rolling_vol", 0.0)),
                "risk_adjusted_return": float(info.get("risk_adjusted_return", 0.0)),
            }
        )

    positions_df = pd.DataFrame(positions, index=pd.DatetimeIndex(dates), columns=env.assets)
    results_df = pd.DataFrame(results)
    results_df.index = pd.DatetimeIndex(dates)
    results_df.index.name = "date"
    metrics = PerformanceMetrics()
    daily_returns = results_df["portfolio_return"]
    benchmark = env.returns.loc[results_df.index, "SPY"] if "SPY" in env.returns.columns else pd.Series(0.0, index=results_df.index)
    comparison = pd.DataFrame(
        {
            "rl_agent": metrics.summarize(daily_returns),
            "static_signal": metrics.summarize(env.returns.loc[results_df.index].mul(env.signal_weights.loc[results_df.index]).sum(axis=1)),
            "SPY": metrics.summarize(benchmark),
        }
    ).T
    return RolloutArtifacts(positions=positions_df, results=results_df, comparison=comparison)


def evaluate_agent(model: PPO, env: TradingEnvironment) -> dict[str, float]:
    """Run a deterministic evaluation rollout and summarize performance."""
    rollout = rollout_policy(model, env)
    metrics = PerformanceMetrics()
    summary = metrics.summarize(rollout.results["portfolio_return"])
    summary["total_return"] = float((1.0 + rollout.results["portfolio_return"]).prod() - 1.0)
    summary["sharpe"] = metrics.sharpe_ratio(rollout.results["portfolio_return"])
    summary["sortino"] = metrics.sortino_ratio(rollout.results["portfolio_return"])
    summary["calmar"] = metrics.calmar_ratio(rollout.results["portfolio_return"])
    summary["max_drawdown"] = metrics.max_drawdown(rollout.results["portfolio_return"])
    return summary
