"""CLI entrypoint for the Phase 3 alpha-model build."""

from __future__ import annotations

import argparse

import pandas as pd

from src.alpha.pipeline import AlphaPipeline
from src.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AMRF Phase 3 alpha-model outputs.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--epochs-override",
        type=int,
        default=None,
        help="Optional override for training epochs to enable faster verification runs.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip walk-forward validation when only signal generation is needed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    technical_features = pd.read_parquet(config.data.processed_dir / "technical_features.parquet")
    returns = pd.read_parquet(config.data.processed_dir / "returns.parquet")
    factors = pd.read_parquet(config.data.processed_dir / "factors.parquet")
    regime_labels = pd.read_parquet(config.regime.output_dir / "regime_labels.parquet")

    pipeline = AlphaPipeline(config.alpha, config.regime)
    pipeline.build(
        technical_features=technical_features,
        returns=returns,
        factors=factors,
        regime_labels=regime_labels,
        epochs_override=args.epochs_override,
        run_validation=not args.skip_validation,
    )


if __name__ == "__main__":
    main()
