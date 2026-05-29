"""Walk-forward validation for regime-specific alpha models."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.alpha.dataset import RegimeDataset
from src.alpha.training import temporal_train_val_split


class WalkForwardValidator:
    """Time-series validation without lookahead bias."""

    def __init__(self, train_window: int = 756, test_window: int = 126, step_size: int = 63) -> None:
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size

    def generate_splits(self, dates: pd.Index) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        ordered_dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
        splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []

        start = 0
        while start + self.train_window + self.test_window <= len(ordered_dates):
            train_dates = ordered_dates[start : start + self.train_window]
            test_dates = ordered_dates[start + self.train_window : start + self.train_window + self.test_window]
            splits.append((train_dates, test_dates))
            start += self.step_size

        return splits

    def validate(
        self,
        model_factory: Callable[[int], object],
        features: pd.DataFrame,
        returns: pd.DataFrame,
        regime_labels: pd.DataFrame | pd.Series,
        target_regime: int,
        factors: pd.DataFrame | None = None,
        sequence_length: int = 60,
        epochs: int = 10,
        validation_fraction: float = 0.2,
        min_samples: int = 200,
        augment_noise_std: float = 0.01,
        device: str = "cpu",
    ) -> pd.DataFrame:
        regime_series = regime_labels["regime"] if isinstance(regime_labels, pd.DataFrame) else regime_labels
        factor_index = factors.index if factors is not None else returns.index
        usable_dates = features.index.intersection(returns.index).intersection(regime_series.index).intersection(factor_index)
        splits = self.generate_splits(usable_dates)

        rows: list[dict[str, float | int]] = []
        for fold, (train_dates, test_dates) in enumerate(splits):
            train_dataset = RegimeDataset(
                features=features,
                returns=returns,
                regime_labels=regime_series,
                target_regime=target_regime,
                factors=factors,
                sequence_length=sequence_length,
                min_samples=min_samples,
                augment_noise_std=augment_noise_std,
                allowed_dates=train_dates,
            )
            test_dataset = RegimeDataset(
                features=features,
                returns=returns,
                regime_labels=regime_series,
                target_regime=target_regime,
                factors=factors,
                sequence_length=sequence_length,
                min_samples=0,
                augment_noise_std=augment_noise_std,
                allowed_dates=test_dates,
            )

            if len(train_dataset) < 2 or len(test_dataset) == 0:
                continue

            train_subset, val_subset = temporal_train_val_split(train_dataset, validation_fraction)
            model = model_factory(train_dataset.input_size)
            model.fit(train_subset, val_subset, epochs=epochs, device=device)
            predictions = model.predict_dataset(test_dataset, device=device)
            actuals = test_dataset.targets.detach().cpu().numpy()

            rows.append(
                {
                    "fold": fold,
                    "n_train": len(train_dataset),
                    "n_test": len(test_dataset),
                    "sharpe": self._compute_daily_sharpe(predictions, actuals, test_dataset.sample_dates),
                    "ic": self._safe_corr(predictions, actuals, method="pearson"),
                    "rank_ic": self._safe_corr(predictions, actuals, method="spearman"),
                    "hit_rate": float(np.mean(np.sign(predictions) == np.sign(actuals))),
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def _compute_daily_sharpe(predictions: np.ndarray, actuals: np.ndarray, dates: pd.DatetimeIndex) -> float:
        frame = pd.DataFrame({"date": pd.DatetimeIndex(dates), "prediction": predictions, "actual": actuals})
        daily_returns: list[float] = []
        for _, group in frame.groupby("date"):
            if len(group) < 2:
                continue
            ranked = group.sort_values("prediction")
            bucket = max(1, len(ranked) // 5)
            long_leg = ranked.tail(bucket)["actual"].mean()
            short_leg = ranked.head(bucket)["actual"].mean()
            daily_returns.append(float(long_leg - short_leg))

        if not daily_returns:
            return 0.0
        series = pd.Series(daily_returns)
        std = float(series.std(ddof=0))
        if std == 0.0:
            return 0.0
        return float(np.sqrt(252.0) * series.mean() / std)

    @staticmethod
    def _safe_corr(left: np.ndarray, right: np.ndarray, method: str) -> float:
        frame = pd.DataFrame({"left": left, "right": right}).dropna()
        if len(frame) < 2:
            return 0.0
        return float(frame["left"].corr(frame["right"], method=method) or 0.0)
