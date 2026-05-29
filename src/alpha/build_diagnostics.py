"""CLI entrypoint for alpha signal diagnostics."""

from __future__ import annotations

import argparse

import pandas as pd

from src.alpha.diagnostics import AlphaDiagnostics
from src.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AMRF alpha diagnostics.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML config file.")
    parser.add_argument(
        "--min-assets-per-day",
        type=int,
        default=3,
        help="Minimum paired forecasts/returns required to score a day.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    alpha_signals = pd.read_parquet(config.alpha.signals_path)
    returns = pd.read_parquet(config.data.processed_dir / "returns.parquet")
    regime_labels = pd.read_parquet(config.regime.output_dir / "regime_labels.parquet")

    diagnostics = AlphaDiagnostics(min_assets_per_day=args.min_assets_per_day)
    artifacts = diagnostics.evaluate(
        alpha_signals=alpha_signals,
        returns=returns,
        regime_labels=regime_labels,
    )
    diagnostics.save(artifacts, config.alpha.diagnostics_path)

    print(artifacts.summary.round(4).to_string())
    if not artifacts.regime_summary.empty:
        print("\nBy regime:")
        print(artifacts.regime_summary.round(4).to_string())


if __name__ == "__main__":
    main()

