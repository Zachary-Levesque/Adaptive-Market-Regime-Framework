"""CLI entrypoint for the Phase 4 risk and backtest build."""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import load_config
from src.risk.backtester import AMRFBacktester, BacktestConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AMRF Phase 4 backtest and risk outputs.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--start", default=None, help="Optional inclusive backtest start date.")
    parser.add_argument("--end", default=None, help="Optional inclusive backtest end date.")
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=None,
        help="One-way turnover cost in basis points.",
    )
    parser.add_argument(
        "--max-gross-exposure",
        type=float,
        default=None,
        help="Total absolute long plus short exposure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    returns = pd.read_parquet(config.data.processed_dir / "returns.parquet")
    alpha_signals = pd.read_parquet(config.alpha.signals_path)
    regime_labels = pd.read_parquet(config.regime.output_dir / "regime_labels.parquet")

    backtester = AMRFBacktester(
        returns=returns,
        alpha_signals=alpha_signals,
        regime_labels=regime_labels,
        config=BacktestConfig(
            max_gross_exposure=args.max_gross_exposure or config.risk.max_gross_exposure,
            long_fraction=config.risk.long_fraction,
            short_fraction=config.risk.short_fraction,
            transaction_cost_bps=args.transaction_cost_bps
            if args.transaction_cost_bps is not None
            else config.risk.transaction_cost_bps,
            benchmark=config.data.benchmark,
        ),
    )
    artifacts = backtester.run(start=args.start, end=args.end, stress_periods=config.risk.stress_periods)
    backtester.save(artifacts, output_dir=config.risk.output_dir)

    print(artifacts.performance_report.round(4).to_string())


if __name__ == "__main__":
    main()
