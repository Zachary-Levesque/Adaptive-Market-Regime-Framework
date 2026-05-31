"""Datasets and sample builders for regime-specific alpha models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SampleMetadata:
    date: pd.Timestamp
    ticker: str


def extract_regime_series(regime_labels: pd.DataFrame | pd.Series) -> pd.Series:
    """Normalize regime labels to a nullable integer Series."""
    if isinstance(regime_labels, pd.DataFrame):
        if "regime" not in regime_labels.columns:
            raise KeyError("Expected regime_labels DataFrame to contain a 'regime' column.")
        series = regime_labels["regime"]
    else:
        series = regime_labels

    normalized = pd.Series(series.copy())
    normalized.index = pd.to_datetime(normalized.index).tz_localize(None)
    return normalized.astype("Int64").sort_index()


class RegimeDataset(Dataset):
    """Create sliding-window samples for a specific market regime."""

    def __init__(
        self,
        features: pd.DataFrame,
        returns: pd.DataFrame,
        regime_labels: pd.DataFrame | pd.Series,
        target_regime: int,
        factors: pd.DataFrame | None = None,
        sequence_length: int = 60,
        min_samples: int = 200,
        augment_noise_std: float = 0.01,
        allowed_dates: Iterable[pd.Timestamp] | None = None,
    ) -> None:
        self.target_regime = target_regime
        self.sequence_length = sequence_length
        self.min_samples = min_samples
        self.augment_noise_std = augment_noise_std
        self.allowed_dates = None if allowed_dates is None else {pd.Timestamp(date) for date in allowed_dates}

        (
            features_array,
            targets_array,
            sample_dates,
            sample_tickers,
            feature_names,
        ) = self._build_samples(
            features=features,
            returns=returns,
            regime_labels=extract_regime_series(regime_labels),
            factors=factors,
            target_regime=target_regime,
            sequence_length=sequence_length,
            allowed_dates=self.allowed_dates,
        )

        if len(features_array) == 0:
            self.features = torch.empty((0, sequence_length, 0), dtype=torch.float32)
            self.targets = torch.empty((0,), dtype=torch.float32)
            self.sample_dates = pd.DatetimeIndex([])
            self.sample_tickers: list[str] = []
            self.feature_names: list[str] = list(feature_names)
            return

        augmented_features, augmented_targets, augmented_dates, augmented_tickers = self._augment_if_needed(
            features_array,
            targets_array,
            sample_dates,
            sample_tickers,
        )

        self.features = torch.tensor(augmented_features, dtype=torch.float32)
        self.targets = torch.tensor(augmented_targets, dtype=torch.float32)
        self.sample_dates = pd.DatetimeIndex(augmented_dates)
        self.sample_tickers = list(augmented_tickers)
        self.feature_names = list(feature_names)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

    @property
    def input_size(self) -> int:
        return int(self.features.shape[-1]) if len(self.features) else 0

    def _augment_if_needed(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sample_dates: list[pd.Timestamp],
        sample_tickers: list[str],
    ) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp], list[str]]:
        if len(features) == 0 or len(features) >= self.min_samples:
            return features, targets, sample_dates, sample_tickers

        rng = np.random.default_rng(42)
        additional = self.min_samples - len(features)
        extra_indices = rng.integers(0, len(features), size=additional)
        noise = rng.normal(0.0, self.augment_noise_std, size=features[extra_indices].shape)

        augmented_features = np.concatenate([features, features[extra_indices] + noise], axis=0)
        augmented_targets = np.concatenate([targets, targets[extra_indices]], axis=0)
        augmented_dates = sample_dates + [sample_dates[idx] for idx in extra_indices]
        augmented_tickers = sample_tickers + [sample_tickers[idx] for idx in extra_indices]
        return augmented_features, augmented_targets, augmented_dates, augmented_tickers

    @staticmethod
    def _build_samples(
        features: pd.DataFrame,
        returns: pd.DataFrame,
        regime_labels: pd.Series,
        factors: pd.DataFrame | None,
        target_regime: int,
        sequence_length: int,
        allowed_dates: set[pd.Timestamp] | None,
    ) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp], list[str], list[str]]:
        if not isinstance(features.columns, pd.MultiIndex):
            raise TypeError("Expected technical features with MultiIndex columns.")

        normalized_returns = returns.copy()
        normalized_returns.index = pd.to_datetime(normalized_returns.index).tz_localize(None)
        normalized_returns = normalized_returns.sort_index()

        normalized_factors = factors.copy() if factors is not None else pd.DataFrame(index=normalized_returns.index)
        normalized_factors.index = pd.to_datetime(normalized_factors.index).tz_localize(None)
        normalized_factors = normalized_factors.sort_index()

        feature_index = pd.to_datetime(features.index).tz_localize(None)
        feature_frame = features.copy()
        feature_frame.index = feature_index
        feature_frame = feature_frame.sort_index()

        common_dates = feature_frame.index.intersection(normalized_returns.index).intersection(regime_labels.index)
        common_dates = common_dates.sort_values()

        if len(common_dates) <= sequence_length:
            return np.empty((0, sequence_length, 0)), np.empty((0,)), [], [], []

        market_block = (
            feature_frame.xs("MARKET", axis=1, level="ticker").reindex(common_dates)
            if "MARKET" in feature_frame.columns.get_level_values(0)
            else pd.DataFrame(index=common_dates)
        )
        factor_block = normalized_factors.reindex(common_dates).ffill().fillna(0.0)

        tickers = [ticker for ticker in feature_frame.columns.get_level_values(0).unique() if ticker != "MARKET"]
        ticker_matrices: dict[str, np.ndarray] = {}
        feature_names: list[str] | None = None

        for ticker in tickers:
            ticker_block = feature_frame.xs(ticker, axis=1, level="ticker").reindex(common_dates)
            combined = pd.concat(
                [
                    ticker_block.add_prefix("tech__"),
                    market_block.add_prefix("market__"),
                    factor_block.add_prefix("factor__"),
                ],
                axis=1,
            ).apply(pd.to_numeric, errors="coerce")
            if feature_names is None:
                feature_names = combined.columns.astype(str).tolist()
            ticker_matrices[ticker] = np.nan_to_num(combined.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        feature_names = feature_names or []
        regime_on_dates = regime_labels.reindex(common_dates)

        samples_x: list[np.ndarray] = []
        samples_y: list[float] = []
        sample_dates: list[pd.Timestamp] = []
        sample_tickers: list[str] = []

        for end_pos in range(sequence_length - 1, len(common_dates) - 1):
            current_date = common_dates[end_pos]
            next_date = common_dates[end_pos + 1]

            current_regime = regime_on_dates.loc[current_date]
            if pd.isna(current_regime) or int(current_regime) != target_regime:
                continue
            if allowed_dates is not None and current_date not in allowed_dates:
                continue

            start_pos = end_pos - sequence_length + 1
            for ticker in tickers:
                target_return = normalized_returns.at[next_date, ticker] if ticker in normalized_returns.columns else np.nan
                if pd.isna(target_return):
                    continue

                samples_x.append(ticker_matrices[ticker][start_pos : end_pos + 1])
                samples_y.append(float(target_return))
                sample_dates.append(current_date)
                sample_tickers.append(ticker)

        if not samples_x:
            return np.empty((0, sequence_length, len(feature_names))), np.empty((0,)), [], [], feature_names

        return (
            np.stack(samples_x),
            np.array(samples_y, dtype=np.float32),
            sample_dates,
            sample_tickers,
            feature_names,
        )
