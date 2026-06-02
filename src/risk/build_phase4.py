"""CLI entrypoint for the Phase 4 risk and backtest build."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.risk.backtester import AMRFBacktester, BacktestConfig


@dataclass(frozen=True)
class SignalSelection:
    signal_path: Path
    transaction_cost_bps: float | None = None
    rebalance_interval_days: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AMRF Phase 4 backtest and risk outputs.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--start", default=None, help="Optional inclusive backtest start date.")
    parser.add_argument("--end", default=None, help="Optional inclusive backtest end date.")
    parser.add_argument(
        "--signal-source",
        default=None,
        help="Optional path to a parquet file containing alpha signals to backtest.",
    )
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
    parser.add_argument(
        "--rebalance-interval-days",
        type=int,
        default=None,
        help="Number of trading days between scheduled rebalances.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    returns = pd.read_parquet(config.data.processed_dir / "returns.parquet")
    selection = resolve_signal_selection(config, override=args.signal_source)
    alpha_signals = pd.read_parquet(selection.signal_path)
    regime_labels = pd.read_parquet(config.regime.output_dir / "regime_labels.parquet")
    transaction_cost_bps = (
        args.transaction_cost_bps
        if args.transaction_cost_bps is not None
        else selection.transaction_cost_bps
        if selection.transaction_cost_bps is not None
        else config.risk.transaction_cost_bps
    )
    rebalance_interval_days = (
        args.rebalance_interval_days
        if args.rebalance_interval_days is not None
        else selection.rebalance_interval_days
        if selection.rebalance_interval_days is not None
        else config.risk.rebalance_interval_days
    )

    backtester = AMRFBacktester(
        returns=returns,
        alpha_signals=alpha_signals,
        regime_labels=regime_labels,
        config=BacktestConfig(
            max_gross_exposure=args.max_gross_exposure or config.risk.max_gross_exposure,
            long_fraction=config.risk.long_fraction,
            short_fraction=config.risk.short_fraction,
            transaction_cost_bps=transaction_cost_bps,
            benchmark=config.data.benchmark,
            rebalance_interval_days=rebalance_interval_days,
            weighting_method=config.risk.weighting_method,
            volatility_lookback=config.risk.volatility_lookback,
            volatility_floor=config.risk.volatility_floor,
        ),
    )
    artifacts = backtester.run(start=args.start, end=args.end, stress_periods=config.risk.stress_periods)
    backtester.save(artifacts, output_dir=config.risk.output_dir)

    print(artifacts.performance_report.round(4).to_string())
    print(f"\nSignals used: {selection.signal_path}")
    print(f"Transaction cost bps: {transaction_cost_bps}")
    print(f"Rebalance interval days: {rebalance_interval_days}")


def resolve_signal_path(config, override: str | None = None):
    return resolve_signal_selection(config, override=override).signal_path


def resolve_signal_selection(config, override: str | None = None) -> SignalSelection:
    if override is not None:
        return SignalSelection(signal_path=Path(override))

    selection_path = config.alpha.selection_path
    if selection_path.exists():
        selection = pd.read_parquet(selection_path)
        if not selection.empty and "signal_path" in selection.columns:
            selected_path = Path(str(selection.iloc[0]["signal_path"]))
            if selected_path.exists():
                return SignalSelection(
                    signal_path=selected_path,
                    transaction_cost_bps=_optional_float(selection.iloc[0].get("transaction_cost_bps")),
                    rebalance_interval_days=_optional_int(selection.iloc[0].get("rebalance_interval_days")),
                )

    return SignalSelection(signal_path=config.alpha.signals_path)


def _optional_float(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _optional_int(value) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


if __name__ == "__main__":
    main()
