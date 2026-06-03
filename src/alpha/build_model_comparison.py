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
    parser.add_argument(
        "--baseline",
        action="append",
        default=None,
        help="Baseline model name to include. Can be passed multiple times.",
    )
    parser.add_argument(
        "--baseline-limit",
        type=int,
        default=None,
        help="Optional maximum number of baseline specs to run after name filtering.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the comparison and print results without saving artifacts or selection manifests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    technical_features = pd.read_parquet(config.data.processed_dir / "technical_features.parquet")
    returns = pd.read_parquet(config.data.processed_dir / "returns.parquet")
    factors = pd.read_parquet(config.data.processed_dir / "factors.parquet")
    regime_labels = pd.read_parquet(config.regime.output_dir / "regime_labels.parquet")

    baseline_specs = build_default_baseline_specs(include_tree_models=args.include_tree_baselines)
    if args.baseline:
        requested = set(args.baseline)
        baseline_specs = [spec for spec in baseline_specs if spec.name in requested]
        missing = sorted(requested.difference({spec.name for spec in baseline_specs}))
        if missing:
            raise ValueError(f"Unknown baseline model(s): {', '.join(missing)}")
    if args.baseline_limit is not None:
        if args.baseline_limit < 1:
            raise ValueError("--baseline-limit must be at least 1.")
        baseline_specs = baseline_specs[: args.baseline_limit]

    comparator = AlphaModelComparator(
        config.alpha,
        config.regime,
        baseline_specs=baseline_specs,
        transaction_cost_bps=config.risk.transaction_cost_bps,
        max_gross_exposure=config.risk.max_gross_exposure,
        long_fraction=config.risk.long_fraction,
        short_fraction=config.risk.short_fraction,
        rebalance_interval_days=config.risk.rebalance_interval_days,
        weighting_method=config.risk.weighting_method,
        volatility_lookback=config.risk.volatility_lookback,
        volatility_floor=config.risk.volatility_floor,
        max_position_weight=config.risk.max_position_weight,
    )
    artifacts = comparator.build(
        technical_features=technical_features,
        returns=returns,
        factors=factors,
        regime_labels=regime_labels,
        epochs_override=args.epochs_override,
        include_ensemble=not args.skip_ensemble,
        save_outputs=not args.dry_run,
    )

    print(artifacts.leaderboard.round(4).to_string())
    if artifacts.best_model:
        print(f"\nBest model: {artifacts.best_model}")


if __name__ == "__main__":
    main()
