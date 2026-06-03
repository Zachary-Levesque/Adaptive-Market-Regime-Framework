"""CLI entrypoint for alpha signal diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.alpha.readiness import load_readiness_status
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
    parser.add_argument(
        "--forward-return-horizon",
        type=int,
        default=None,
        help="Number of future trading days to sum when scoring forward returns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    signal_path = resolve_signal_path(config, override=args.signal_source)
    alpha_signals = pd.read_parquet(signal_path)
    returns = pd.read_parquet(config.data.processed_dir / "returns.parquet")
    regime_labels = pd.read_parquet(config.regime.output_dir / "regime_labels.parquet")

    diagnostics = AlphaDiagnostics(
        min_assets_per_day=args.min_assets_per_day,
        forward_return_horizon=args.forward_return_horizon or config.alpha.target_horizon,
    )
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
                if selected_path.name == "alpha_signals_rl_tilted.parquet":
                    ready, _ = load_readiness_status(
                        config.alpha.diagnostics_path.with_name("alpha_readiness_report.parquet")
                    )
                    if not ready:
                        return resolve_non_rl_signal_path(config)
                return selected_path

    return config.alpha.signals_path


def resolve_non_rl_signal_path(config) -> Path:
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


if __name__ == "__main__":
    main()
