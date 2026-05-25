"""CLI entrypoint for the Phase 1 data build."""

from __future__ import annotations

import argparse
from dataclasses import replace

from src.config import load_config
from src.data.pipeline import DataPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AMRF Phase 1 research datasets.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--allow-remote-downloads",
        action="store_true",
        help="Override config and allow remote market-data downloads for missing local tickers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.allow_remote_downloads:
        data_config = replace(config.data, allow_remote_downloads=True)
    else:
        data_config = config.data
    pipeline = DataPipeline(data_config)
    pipeline.build()


if __name__ == "__main__":
    main()
