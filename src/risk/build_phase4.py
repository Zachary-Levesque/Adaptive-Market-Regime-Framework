"""CLI entrypoint for the Phase 4 risk and backtest build."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.alpha.readiness import load_readiness_status
from src.config import load_config
from src.risk.backtester import AMRFBacktester, BacktestConfig


@dataclass(frozen=True)
class SignalSelection:
    signal_path: Path
    transaction_cost_bps: float | None = None
    rebalance_interval_days: int | None = None
    weighting_method: str | None = None
    long_fraction: float | None = None
    short_fraction: float | None = None
    max_position_weight: float | None = None


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
    weighting_method = selection.weighting_method or config.risk.weighting_method
    long_fraction = selection.long_fraction if selection.long_fraction is not None else config.risk.long_fraction
    short_fraction = selection.short_fraction if selection.short_fraction is not None else config.risk.short_fraction
    max_position_weight = (
        selection.max_position_weight if selection.max_position_weight is not None else config.risk.max_position_weight
    )

    backtester = AMRFBacktester(
        returns=returns,
        alpha_signals=alpha_signals,
        regime_labels=regime_labels,
        config=BacktestConfig(
            max_gross_exposure=args.max_gross_exposure or config.risk.max_gross_exposure,
            long_fraction=long_fraction,
            short_fraction=short_fraction,
            transaction_cost_bps=transaction_cost_bps,
            benchmark=config.data.benchmark,
            rebalance_interval_days=rebalance_interval_days,
            weighting_method=weighting_method,
            volatility_lookback=config.risk.volatility_lookback,
            volatility_floor=config.risk.volatility_floor,
            max_position_weight=max_position_weight,
        ),
    )
    artifacts = backtester.run(start=args.start, end=args.end, stress_periods=config.risk.stress_periods)
    backtester.save(artifacts, output_dir=config.risk.output_dir)

    print(artifacts.performance_report.round(4).to_string())
    print(f"\nSignals used: {selection.signal_path}")
    print(f"Transaction cost bps: {transaction_cost_bps}")
    print(f"Rebalance interval days: {rebalance_interval_days}")
    print(f"Weighting method: {weighting_method}")
    print(f"Long fraction: {long_fraction}")
    print(f"Short fraction: {short_fraction}")
    print(f"Max position weight: {max_position_weight}")


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
                if _is_rl_tilted_signal(selected_path):
                    ready, _ = load_readiness_status(
                        config.alpha.diagnostics_path.with_name("alpha_readiness_report.parquet")
                    )
                    if not ready:
                        fallback_path = _resolve_non_rl_fallback_signal(config)
                        return SignalSelection(signal_path=fallback_path)
                return SignalSelection(
                    signal_path=selected_path,
                    transaction_cost_bps=_optional_float(selection.iloc[0].get("transaction_cost_bps")),
                    rebalance_interval_days=_optional_int(selection.iloc[0].get("rebalance_interval_days")),
                    weighting_method=_optional_str(selection.iloc[0].get("weighting_method")),
                    long_fraction=_optional_float(selection.iloc[0].get("long_fraction")),
                    short_fraction=_optional_float(selection.iloc[0].get("short_fraction")),
                    max_position_weight=_optional_float(selection.iloc[0].get("max_position_weight")),
                )

    return SignalSelection(signal_path=config.alpha.signals_path)


def _is_rl_tilted_signal(path: Path) -> bool:
    return path.name == "alpha_signals_rl_tilted.parquet"


def _resolve_non_rl_fallback_signal(config) -> Path:
    summary_path = config.alpha.comparison_path.with_name("alpha_model_comparison_summary.parquet")
    if summary_path.exists():
        summary = pd.read_parquet(summary_path)
        if "signal_path" in summary.columns:
            candidates = summary.copy()
            if "model" not in candidates.columns:
                candidates = candidates.reset_index()
            if "model" in candidates.columns:
                candidates = candidates[candidates["model"].astype(str).ne("rl_tilted")]
            candidates["signal_path"] = candidates["signal_path"].astype(str)
            candidates = candidates[~candidates["signal_path"].str.endswith("alpha_signals_rl_tilted.parquet")]
            for signal_path in candidates["signal_path"]:
                path = Path(signal_path)
                if path.exists():
                    return path

    return config.alpha.signals_path


def _optional_float(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _optional_int(value) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _optional_str(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value)
    return text if text else None


if __name__ == "__main__":
    main()
