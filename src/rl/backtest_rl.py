"""Backtest the trained PPO agent against the selected static alpha signal."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.alpha.readiness import load_readiness_status
from src.config import load_config
from src.execution.simulator import ExecutionSimulator
from src.risk.backtester import AMRFBacktester, BacktestConfig
from src.risk.metrics import PerformanceMetrics
from src.rl.agent import PPOPositionSizingAgent, rollout_policy
from src.rl.data import load_rl_dataset, split_by_date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest the AMRF PPO agent.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML config file.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if the selected alpha readiness gate has not passed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ready_path = config.alpha.diagnostics_path.with_name("alpha_readiness_report.parquet")
    ready, _ = load_readiness_status(ready_path)
    if not ready and not args.force:
        raise SystemExit(
            "Refusing to backtest RL: selected alpha is not ready. "
            f"Run `python -m src.alpha.build_readiness --config {args.config}` first."
        )

    dataset = load_rl_dataset(config)
    agent = PPOPositionSizingAgent(config, dataset)
    model = agent.load_model()
    stats = agent.build_stats()
    test_env = agent.build_environment(
        config.rl.test_start,
        config.rl.test_end,
        random_start=False,
        episode_length=None,
        stats=stats,
    )

    rollout = rollout_policy(model, test_env)
    simulator = ExecutionSimulator(
        transaction_cost_bps=config.risk.transaction_cost_bps,
        max_single_trade_size=config.execution.max_single_trade_size,
        slippage_coefficient=config.execution.slippage_coefficient,
        slippage_floor_bps=config.execution.slippage_floor_bps,
        etf_cost_bps=config.execution.etf_cost_bps,
        large_cap_cost_bps=config.execution.large_cap_cost_bps,
        small_mid_cost_bps=config.execution.small_mid_cost_bps,
        initial_portfolio_value=config.rl.initial_capital,
    )

    rl_execution = simulator.simulate(rollout.positions, dataset.returns, dataset.prices)
    rl_results = rl_execution.daily_results.copy()
    rl_positions = rl_execution.executed_weights.copy()
    regime_labels = pd.DataFrame(
        {"regime": dataset.regime_probs.idxmax(axis=1).map({name: idx for idx, name in enumerate(dataset.regime_probs.columns)})}
    )
    rl_results["portfolio_return"] = rl_results["net_return"]
    rl_results["equity"] = rl_results["portfolio_value"]
    rl_results["benchmark_return"] = dataset.returns.loc[rl_results.index, "SPY"].reindex(rl_results.index).fillna(0.0)
    rl_results["equal_weight_return"] = dataset.returns.loc[rl_results.index].mean(axis=1)
    rl_results["regime"] = regime_labels.loc[rl_results.index, "regime"].reindex(rl_results.index).astype(float)

    static_signal = dataset.selected_signal.loc[config.rl.test_start : config.rl.test_end]
    static_backtester = AMRFBacktester(
        returns=split_by_date(dataset.returns, config.rl.test_start, config.rl.test_end),
        alpha_signals=static_signal,
        regime_labels=split_by_date(regime_labels, config.rl.test_start, config.rl.test_end),
        config=BacktestConfig(
            transaction_cost_bps=config.risk.transaction_cost_bps,
            benchmark=config.data.benchmark,
            rebalance_interval_days=config.risk.rebalance_interval_days,
            weighting_method=config.risk.weighting_method,
            volatility_lookback=config.risk.volatility_lookback,
            volatility_floor=config.risk.volatility_floor,
            max_position_weight=config.risk.max_position_weight,
            max_gross_exposure=config.risk.max_gross_exposure,
            long_fraction=config.risk.long_fraction,
            short_fraction=config.risk.short_fraction,
        ),
    )
    static_artifacts = static_backtester.run(start=config.rl.test_start, end=config.rl.test_end, stress_periods=config.risk.stress_periods)
    static_execution = simulator.simulate(static_artifacts.weights, dataset.returns, dataset.prices)
    static_results = static_execution.daily_results.copy()
    metrics = PerformanceMetrics()
    benchmark = dataset.returns.loc[rl_results.index, "SPY"].reindex(rl_results.index).fillna(0.0)
    static_gross = static_artifacts.daily_results["strategy_return_gross"].reindex(rl_results.index).fillna(0.0)
    static_net = static_results["net_return"].reindex(rl_results.index).fillna(0.0)
    rl_gross = rl_results["gross_return"].reindex(rl_results.index).fillna(0.0)
    rl_net = rl_results["net_return"].reindex(rl_results.index).fillna(0.0)
    comparison = pd.DataFrame(
        {
            "rl_agent_policy": metrics.summarize(rl_gross),
            "rl_agent_execution": metrics.summarize(rl_net),
            "static_signal_policy": metrics.summarize(static_gross),
            "static_signal_execution": metrics.summarize(static_net),
            "SPY": metrics.summarize(benchmark),
        }
    ).T

    output_dir = Path(config.risk.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.rl.positions_path.parent.mkdir(parents=True, exist_ok=True)

    rl_positions.to_parquet(config.rl.positions_path)
    rl_results.to_parquet(config.rl.backtest_results_path)
    comparison.to_parquet(config.rl.comparison_path)

    execution_summary = rl_execution.analytics.copy()
    if not execution_summary.empty:
        execution_summary.to_parquet(output_dir / "execution_analytics.parquet")

    print("RL comparison table:")
    print(comparison)
    print(f"Saved RL backtest results to {config.rl.backtest_results_path}")
    print(f"Saved RL positions to {config.rl.positions_path}")
    print(f"Saved RL vs baseline comparison to {config.rl.comparison_path}")


if __name__ == "__main__":
    main()
