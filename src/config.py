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
class RegimeConfig:
    n_regimes: int
    n_iter: int
    covariance_type: str
    regime_names: dict[int, str]
    n_restarts: int
    model_path: Path
    output_dir: Path
    chart_path: Path


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    regime: RegimeConfig


def load_config(path: str | Path) -> AppConfig:
    """Load the application config from YAML."""
    config_path = Path(path)
    raw = _read_yaml(config_path)
    data_section = raw.get("data", {})

    cache_dir = Path(data_section.get("cache_dir", "data/raw"))
    processed_dir = Path(data_section.get("processed_dir", "data/processed"))
    local_data_dir = Path(data_section.get("local_data_dir", "data/raw"))
    allow_remote_downloads = bool(data_section.get("allow_remote_downloads", False))
    regime_section = raw.get("regime", {})
    regime_names = {
        int(key): str(value)
        for key, value in regime_section.get(
            "regime_names",
            {
                0: "Bull Trending",
                1: "Low-Vol Compression",
                2: "Bear Trending",
                3: "High-Vol Crisis",
            },
        ).items()
    }

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
        ),
        regime=RegimeConfig(
            n_regimes=int(regime_section.get("n_regimes", 4)),
            n_iter=int(regime_section.get("n_iter", 1000)),
            covariance_type=str(regime_section.get("covariance_type", "full")),
            regime_names=regime_names,
            n_restarts=int(regime_section.get("n_restarts", 10)),
            model_path=Path(regime_section.get("model_path", "src/regime/hmm_model.pkl")),
            output_dir=Path(regime_section.get("output_dir", "data/regimes")),
            chart_path=Path(regime_section.get("chart_path", "data/regimes/regime_history.png")),
        ),
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle) or {}

    if "data" not in parsed:
        raise KeyError("Config file must contain a top-level 'data' section.")

    return parsed
