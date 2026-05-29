"""CLI entrypoint for alpha model comparison."""

from __future__ import annotations

import argparse

import pandas as pd

from src.alpha.baselines import build_default_baseline_specs
from src.alpha.model_comparison import AlphaModelComparator
from src.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare AMRF alpha models and baselines.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML config file.")
    parser.add_argument(
        "--epochs-override",
        type=int,
        default=None,
        help="Optional override for training epochs to speed up comparison runs.",
    )
    parser.add_argument(
        "--skip-ensemble",
        action="store_true",
        help="Only compare baseline models and skip the saved ensemble metrics.",
    )
    parser.add_argument(
        "--include-tree-baselines",
        action="store_true",
        help="Include random forest and gradient boosting baselines in addition to the linear models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    technical_features = pd.read_parquet(config.data.processed_dir / "technical_features.parquet")
    returns = pd.read_parquet(config.data.processed_dir / "returns.parquet")
    factors = pd.read_parquet(config.data.processed_dir / "factors.parquet")
    regime_labels = pd.read_parquet(config.regime.output_dir / "regime_labels.parquet")

    comparator = AlphaModelComparator(
        config.alpha,
        config.regime,
        baseline_specs=build_default_baseline_specs(include_tree_models=args.include_tree_baselines),
    )
    artifacts = comparator.build(
        technical_features=technical_features,
        returns=returns,
        factors=factors,
        regime_labels=regime_labels,
        epochs_override=args.epochs_override,
        include_ensemble=not args.skip_ensemble,
    )

    print(artifacts.leaderboard.round(4).to_string())
    if artifacts.best_model:
        print(f"\nBest model: {artifacts.best_model}")


if __name__ == "__main__":
    main()
