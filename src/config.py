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
class AlphaConfig:
    hidden_size: int
    num_layers: int
    dropout: float
    sequence_length: int
    batch_size: int
    epochs: int
    learning_rate: float
    train_window: int
    test_window: int
    step_size: int
    model_dir: Path
    signals_path: Path
    metrics_path: Path
    diagnostics_path: Path
    comparison_path: Path
    signals_dir: Path
    selection_path: Path
    validation_fraction: float
    min_samples_per_regime: int
    augment_noise_std: float
    weight_decay: float
    patience: int
    device: str


@dataclass(frozen=True)
class RiskConfig:
    n_simulations: int
    confidence_level: float
    stress_periods: dict[str, tuple[str, str]]
    output_dir: Path
    transaction_cost_bps: float
    max_gross_exposure: float
    long_fraction: float
    short_fraction: float
    rebalance_interval_days: int


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    regime: RegimeConfig
    alpha: AlphaConfig
    risk: RiskConfig


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
    alpha_section = raw.get("alpha", {})
    risk_section = raw.get("risk", {})
    lstm_section = alpha_section.get("lstm", {})
    walk_forward_section = alpha_section.get("walk_forward", {})
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
        alpha=AlphaConfig(
            hidden_size=int(lstm_section.get("hidden_size", 128)),
            num_layers=int(lstm_section.get("num_layers", 2)),
            dropout=float(lstm_section.get("dropout", 0.2)),
            sequence_length=int(lstm_section.get("sequence_length", 60)),
            batch_size=int(lstm_section.get("batch_size", 32)),
            epochs=int(lstm_section.get("epochs", 100)),
            learning_rate=float(lstm_section.get("learning_rate", 0.001)),
            train_window=int(walk_forward_section.get("train_window", 756)),
            test_window=int(walk_forward_section.get("test_window", 126)),
            step_size=int(walk_forward_section.get("step_size", 63)),
            model_dir=Path(alpha_section.get("model_dir", "src/alpha/models")),
            signals_path=Path(alpha_section.get("signals_path", "data/processed/alpha_signals.parquet")),
            metrics_path=Path(alpha_section.get("metrics_path", "data/processed/alpha_metrics.parquet")),
            diagnostics_path=Path(
                alpha_section.get("diagnostics_path", "data/processed/alpha_diagnostics.parquet")
            ),
            comparison_path=Path(
                alpha_section.get("comparison_path", "data/processed/alpha_model_comparison.parquet")
            ),
            signals_dir=Path(alpha_section.get("signals_dir", "data/processed/alpha_signals")),
            selection_path=Path(
                alpha_section.get("selection_path", "data/processed/alpha_signal_selection.parquet")
            ),
            validation_fraction=float(alpha_section.get("validation_fraction", 0.2)),
            min_samples_per_regime=int(alpha_section.get("min_samples_per_regime", 200)),
            augment_noise_std=float(alpha_section.get("augment_noise_std", 0.01)),
            weight_decay=float(alpha_section.get("weight_decay", 1e-5)),
            patience=int(alpha_section.get("patience", 10)),
            device=str(alpha_section.get("device", "cpu")),
        ),
        risk=RiskConfig(
            n_simulations=int(risk_section.get("n_simulations", 10000)),
            confidence_level=float(risk_section.get("confidence_level", 0.95)),
            stress_periods={
                str(name): (str(bounds[0]), str(bounds[1]))
                for name, bounds in risk_section.get(
                    "stress_periods",
                    {
                        "covid": ("2020-02-19", "2020-03-23"),
                        "rate_hike": ("2022-01-03", "2022-12-31"),
                    },
                ).items()
            },
            output_dir=Path(risk_section.get("output_dir", "data/results")),
            transaction_cost_bps=float(risk_section.get("transaction_cost_bps", 10.0)),
            max_gross_exposure=float(risk_section.get("max_gross_exposure", 1.0)),
            long_fraction=float(risk_section.get("long_fraction", 0.2)),
            short_fraction=float(risk_section.get("short_fraction", 0.2)),
            rebalance_interval_days=max(1, int(risk_section.get("rebalance_interval_days", 1))),
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
