"""CLI entrypoint for alpha signal diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    parser.add_argument(
        "--signal-source",
        default=None,
        help="Optional path to a parquet file containing alpha signals to diagnose.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    signal_path = resolve_signal_path(config, override=args.signal_source)
    alpha_signals = pd.read_parquet(signal_path)
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
    print(f"\nSignals diagnosed: {signal_path}")


def resolve_signal_path(config, override: str | None = None) -> Path:
    if override is not None:
        return Path(override)

    selection_path = config.alpha.selection_path
    if selection_path.exists():
        selection = pd.read_parquet(selection_path)
        if not selection.empty and "signal_path" in selection.columns:
            selected_path = Path(str(selection.iloc[0]["signal_path"]))
            if selected_path.exists():
                return selected_path

    return config.alpha.signals_path


if __name__ == "__main__":
    main()
