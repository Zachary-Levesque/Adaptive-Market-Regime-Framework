"""Data loading and preprocessing helpers for the RL stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RLSplits:
    train: tuple[pd.Timestamp, pd.Timestamp]
    validation: tuple[pd.Timestamp, pd.Timestamp]
    test: tuple[pd.Timestamp, pd.Timestamp]


@dataclass(frozen=True)
class RLDataset:
    prices: pd.DataFrame
    returns: pd.DataFrame
    regime_probs: pd.DataFrame
    regime_features: pd.DataFrame
    selected_signal: pd.DataFrame
    signal_path: Path
    splits: RLSplits


def resolve_selected_signal_path(config) -> Path:
    """Resolve the deployable alpha signal selected by the research pipeline."""
    selection_path = Path(config.alpha.selection_path)
    if selection_path.exists():
        selection = pd.read_parquet(selection_path)
        if not selection.empty and "signal_path" in selection.columns:
            selected_path = Path(str(selection.iloc[0]["signal_path"]))
            if selected_path.exists():
                return selected_path

    candidate = Path(config.alpha.signals_dir) / "regime_portfolio_selector.parquet"
    if candidate.exists():
        return candidate

    return Path(config.alpha.signals_path)


def load_rl_dataset(config) -> RLDataset:
    """Load the static artifacts required by the RL pipeline."""
    prices = pd.read_parquet(Path(config.data.processed_dir) / "prices.parquet").sort_index()
    returns = pd.read_parquet(Path(config.data.processed_dir) / "returns.parquet").sort_index()
    regime_probs = pd.read_parquet(Path(config.regime.output_dir) / "regime_probs.parquet").sort_index()
    regime_features = pd.read_parquet(Path(config.data.processed_dir) / "regime_features.parquet").sort_index()
    selected_path = resolve_selected_signal_path(config)
    selected_signal = pd.read_parquet(selected_path).sort_index()

    common_index = (
        prices.index.intersection(returns.index)
        .intersection(regime_probs.index)
        .intersection(selected_signal.index)
        .intersection(regime_features.index)
        .sort_values()
    )
    common_assets = returns.columns.intersection(selected_signal.columns).intersection(prices.columns.get_level_values(0))

    if common_index.empty:
        raise ValueError("RL data artifacts do not share a common date index.")
    if common_assets.empty:
        raise ValueError("RL data artifacts do not share a common asset universe.")

    prices = _subset_prices(prices, common_assets, common_index)
    returns = returns.loc[common_index, common_assets]
    regime_probs = regime_probs.loc[common_index]
    regime_features = regime_features.loc[common_index]
    selected_signal = selected_signal.loc[common_index, common_assets]

    return RLDataset(
        prices=prices,
        returns=returns,
        regime_probs=regime_probs,
        regime_features=regime_features,
        selected_signal=selected_signal,
        signal_path=selected_path,
        splits=RLSplits(
            train=(pd.Timestamp(config.rl.train_start), pd.Timestamp(config.rl.train_end)),
            validation=(pd.Timestamp(config.rl.validation_start), pd.Timestamp(config.rl.validation_end)),
            test=(pd.Timestamp(config.rl.test_start), pd.Timestamp(config.rl.test_end)),
        ),
    )


def split_by_date(frame: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """Return a copy of ``frame`` restricted to the requested date window."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)].copy()


def normalize_signal_scores(signal_row: pd.Series, temperature: float = 1.0) -> pd.Series:
    """Convert a score vector to long-only portfolio weights."""
    clean = pd.to_numeric(signal_row, errors="coerce").fillna(0.0).astype(float)
    clean = clean.replace([np.inf, -np.inf], 0.0)
    scaled = clean / max(1e-8, float(temperature))
    shifted = scaled - scaled.max()
    weights = np.exp(np.clip(shifted, -50.0, 50.0))
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        return pd.Series(1.0 / len(clean), index=clean.index, dtype=float)
    return pd.Series(weights / total, index=clean.index, dtype=float)


def safe_forward_fill(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.copy().sort_index().ffill().bfill()


def _subset_prices(prices: pd.DataFrame, assets: pd.Index, dates: pd.Index) -> pd.DataFrame:
    if not isinstance(prices.columns, pd.MultiIndex):
        raise TypeError("Expected prices to be a MultiIndex frame.")

    subset = prices.loc[dates, prices.columns.get_level_values(0).isin(assets)].copy()
    return subset.sort_index(axis=1)
