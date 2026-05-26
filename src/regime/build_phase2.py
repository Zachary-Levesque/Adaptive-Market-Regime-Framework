"""CLI entrypoint for the Phase 2 regime-detection build."""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import load_config
from src.regime.pipeline import RegimeDetectionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AMRF Phase 2 regime detection outputs.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prices = pd.read_parquet(config.data.processed_dir / "prices.parquet")
    regime_features = pd.read_parquet(config.data.processed_dir / "regime_features.parquet")

    pipeline = RegimeDetectionPipeline(config.regime)
    pipeline.build(
        regime_features=regime_features,
        prices=prices,
        benchmark=config.data.benchmark,
    )


if __name__ == "__main__":
    main()
