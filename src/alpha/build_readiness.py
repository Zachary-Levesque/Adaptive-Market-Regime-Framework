"""CLI entrypoint for selected-alpha readiness checks."""

from __future__ import annotations

import argparse

import pandas as pd

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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    selection = pd.read_parquet(config.alpha.selection_path)
    diagnostics_summary = pd.read_parquet(config.alpha.diagnostics_path.with_name("alpha_diagnostics_summary.parquet"))
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
    )
    output_path = config.alpha.diagnostics_path.with_name("alpha_readiness_report.parquet")
    checker.save(report, output_path)
    print(report.to_string(index=False))
    print(f"\nReady for RL: {bool(report['ready_for_rl'].all()) if not report.empty else False}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
