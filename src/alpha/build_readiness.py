"""CLI entrypoint for selected-alpha readiness checks."""

from __future__ import annotations

import argparse

import pandas as pd

from src.alpha.build_diagnostics import resolve_non_rl_signal_path
from src.alpha.readiness import AlphaReadinessChecker, ReadinessThresholds
from src.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AMRF selected-alpha readiness report.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML config file.")
    parser.add_argument(
        "--min-active-days",
        type=int,
        default=504,
        help="Minimum scored selected-signal days required before RL work.",
    )
    parser.add_argument(
        "--min-sharpe",
        type=float,
        default=0.5,
        help="Minimum selected-strategy Sharpe required before RL work.",
    )
    return parser.parse_args()


def resolve_effective_selection(config, selection: pd.DataFrame) -> pd.DataFrame:
    if selection.empty or "signal_path" not in selection.columns:
        return selection

    selected_path = selection.iloc[0].get("signal_path")
    if pd.isna(selected_path) or not str(selected_path).endswith("alpha_signals_rl_tilted.parquet"):
        return selection

    fallback_path = resolve_non_rl_signal_path(config)
    effective = selection.copy()
    effective.loc[effective.index[0], "model"] = fallback_path.stem
    effective.loc[effective.index[0], "signal_path"] = str(fallback_path)
    effective.loc[effective.index[0], "selection_method"] = "pre_rl_fallback"
    return effective


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    selection = pd.read_parquet(config.alpha.selection_path)
    selection = resolve_effective_selection(config, selection)
    diagnostics_summary = pd.read_parquet(config.alpha.diagnostics_path.with_name("alpha_diagnostics_summary.parquet"))
    regime_diagnostics_path = config.alpha.diagnostics_path.with_name("alpha_diagnostics_by_regime.parquet")
    regime_diagnostics = (
        pd.read_parquet(regime_diagnostics_path)
        if regime_diagnostics_path.exists()
        else pd.DataFrame()
    )
    performance_report = pd.read_parquet(config.risk.output_dir / "performance_report.parquet")
    stress_path = config.risk.output_dir / "stress_report.parquet"
    stress_report = pd.read_parquet(stress_path) if stress_path.exists() else pd.DataFrame()

    checker = AlphaReadinessChecker(
        ReadinessThresholds(
            min_active_days=args.min_active_days,
            min_sharpe=args.min_sharpe,
        )
    )
    report = checker.evaluate(
        selection=selection,
        diagnostics_summary=diagnostics_summary,
        performance_report=performance_report,
        stress_report=stress_report,
        regime_diagnostics=regime_diagnostics,
    )
    output_path = config.alpha.diagnostics_path.with_name("alpha_readiness_report.parquet")
    checker.save(report, output_path)
    print(report.to_string(index=False))
    print(f"\nReady for RL: {bool(report['ready_for_rl'].all()) if not report.empty else False}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
