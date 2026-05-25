"""CLI entrypoint for the Phase 1 data build."""

from __future__ import annotations

import argparse

from src.config import load_config
from src.data.pipeline import DataPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AMRF Phase 1 research datasets.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline = DataPipeline(config.data)
    pipeline.build()


if __name__ == "__main__":
    main()
