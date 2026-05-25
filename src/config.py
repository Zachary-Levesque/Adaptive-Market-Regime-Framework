"""Project configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    universe: list[str]
    start_date: str
    end_date: str
    benchmark: str
    cache_dir: Path
    processed_dir: Path
    local_data_dir: Path
    allow_remote_downloads: bool


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig


def load_config(path: str | Path) -> AppConfig:
    """Load the application config from YAML."""
    config_path = Path(path)
    raw = _read_yaml(config_path)
    data_section = raw.get("data", {})

    cache_dir = Path(data_section.get("cache_dir", "data/raw"))
    processed_dir = Path(data_section.get("processed_dir", "data/processed"))
    local_data_dir = Path(data_section.get("local_data_dir", "data/raw"))
    allow_remote_downloads = bool(data_section.get("allow_remote_downloads", False))

    return AppConfig(
        data=DataConfig(
            universe=list(data_section["universe"]),
            start_date=str(data_section["start_date"]),
            end_date=str(data_section["end_date"]),
            benchmark=str(data_section["benchmark"]),
            cache_dir=cache_dir,
            processed_dir=processed_dir,
            local_data_dir=local_data_dir,
            allow_remote_downloads=allow_remote_downloads,
        )
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle) or {}

    if "data" not in parsed:
        raise KeyError("Config file must contain a top-level 'data' section.")

    return parsed
