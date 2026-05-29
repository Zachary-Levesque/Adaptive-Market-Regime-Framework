"""Validate local Phase 1 raw-data availability before running the full build."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.data.ingestion import MarketDataIngester


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate AMRF Phase 1 local raw-data inputs.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    price_universe = [ticker for ticker in config.data.universe if ticker not in {"^VIX", "VIX"}]

    ingester = MarketDataIngester(local_data_dir=config.data.local_data_dir)
    cache_ingester = MarketDataIngester(cache_dir=config.data.cache_dir)
    cached_prices = cache_ingester._load_cached_prices(
        price_universe,
        config.data.start_date,
        config.data.end_date,
        "1d",
    )
    statuses = ingester.inspect_local_data(price_universe)

    found = [status for status in statuses if status.found]
    missing = [status for status in statuses if not status.found]

    print(f"Local data directory: {config.data.local_data_dir}")
    print(f"Cache directory: {config.data.cache_dir}")
    if cached_prices is not None:
        cached_tickers = sorted(cached_prices.columns.get_level_values(0).unique())
        print(f"Cached multi-ticker parquet: found ({len(cached_tickers)} tickers)")
        print("Phase 1 price ingestion can use this cache without individual ticker files.")
    else:
        print("Cached multi-ticker parquet: not found")
    print(f"Tickers checked: {len(statuses)}")
    print(f"Resolved locally: {len(found)}")
    print(f"Missing locally: {len(missing)}")

    if found:
        print("\nResolved:")
        for status in found:
            print(f"  {status.ticker}: {status.path}")

    if missing:
        print("\nMissing individual ticker files:")
        for status in missing:
            candidates = ", ".join(ingester._local_filename_candidates(status.ticker))
            print(f"  {status.ticker}: expected one of [{candidates}] under {config.data.local_data_dir}")

    macro_note = "VIX, DGS10, and DGS2 are loaded separately from macro sources during the build."
    print(f"\nNotes: {macro_note}")


if __name__ == "__main__":
    main()
