"""CLI entrypoint for AMRF final completion checks."""

from __future__ import annotations

import argparse

from src.completion import ProjectCompletionChecker
from src.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AMRF final project completion report.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML config file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output parquet path. Defaults to data/results/project_completion_report.parquet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checker = ProjectCompletionChecker(config)
    artifacts = checker.evaluate()
    output_path = checker.save(artifacts, args.output)

    print(artifacts.report.to_string(index=False))
    print(f"\nProject complete: {artifacts.complete}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
