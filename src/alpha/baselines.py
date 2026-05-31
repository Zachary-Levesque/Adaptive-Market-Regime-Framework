"""Simple sklearn baselines for regime-specific alpha modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, Subset


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    factory: Callable[[int], "SequenceRegressor"]


class SequenceRegressor:
    """Minimal interface used by walk-forward validation."""

    def fit(self, train_dataset: Dataset, val_dataset: Dataset, epochs: int, device: str = "cpu") -> "SequenceRegressor":
        raise NotImplementedError

    def predict_dataset(self, dataset: Dataset, device: str = "cpu") -> np.ndarray:
        raise NotImplementedError


class SklearnSequenceRegressor(SequenceRegressor):
    """Flatten sequence samples and fit a standard sklearn regressor."""

    def __init__(self, estimator, name: str) -> None:
        self.estimator = estimator
        self.name = name
        self._fallback_mean: float = 0.0
        self._is_fitted = False

    def fit(self, train_dataset: Dataset, val_dataset: Dataset, epochs: int, device: str = "cpu") -> "SklearnSequenceRegressor":
        features, targets = self._dataset_to_arrays(train_dataset)
        if len(features) == 0:
            self._fallback_mean = 0.0
            self._is_fitted = True
            return self

        self._fallback_mean = float(np.mean(targets)) if len(targets) else 0.0
        try:
            self.estimator.fit(features, targets)
        except Exception:
            self.estimator = DummyMeanRegressor(self._fallback_mean)
            self.estimator.fit(features, targets)
        self._is_fitted = True
        return self

    def predict_dataset(self, dataset: Dataset, device: str = "cpu") -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fit before prediction.")

        features, _ = self._dataset_to_arrays(dataset)
        if len(features) == 0:
            return np.array([], dtype=np.float32)

        predictions = self.estimator.predict(features)
        return np.asarray(predictions, dtype=np.float32).reshape(-1)

    @staticmethod
    def _dataset_to_arrays(dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
        features_tensor, targets_tensor = _resolve_dataset_tensors(dataset)
        features = features_tensor.detach().cpu().numpy()
        targets = targets_tensor.detach().cpu().numpy().reshape(-1)
        if len(features) == 0:
            return np.empty((0, 0), dtype=np.float32), np.array([], dtype=np.float32)
        flattened = features.reshape(features.shape[0], -1).astype(np.float32, copy=False)
        return flattened, targets.astype(np.float32, copy=False)


class SklearnLastStepRegressor(SklearnSequenceRegressor):
    """Fit a sklearn regressor using only the most recent feature vector."""

    @staticmethod
    def _dataset_to_arrays(dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
        features_tensor, targets_tensor = _resolve_dataset_tensors(dataset)
        features = features_tensor.detach().cpu().numpy()
        targets = targets_tensor.detach().cpu().numpy().reshape(-1)
        if len(features) == 0:
            return np.empty((0, 0), dtype=np.float32), np.array([], dtype=np.float32)
        latest = features[:, -1, :].astype(np.float32, copy=False)
        return latest, targets.astype(np.float32, copy=False)


class SklearnSequenceSummaryRegressor(SklearnSequenceRegressor):
    """Fit a sklearn regressor on compact sequence summary statistics."""

    @staticmethod
    def _dataset_to_arrays(dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
        features_tensor, targets_tensor = _resolve_dataset_tensors(dataset)
        features = features_tensor.detach().cpu().numpy()
        targets = targets_tensor.detach().cpu().numpy().reshape(-1)
        if len(features) == 0:
            return np.empty((0, 0), dtype=np.float32), np.array([], dtype=np.float32)

        latest = features[:, -1, :]
        sequence_mean = features.mean(axis=1)
        sequence_std = features.std(axis=1)
        summary = np.concatenate([latest, sequence_mean, sequence_std], axis=1)
        return summary.astype(np.float32, copy=False), targets.astype(np.float32, copy=False)


class WeightedTechnicalRegressor(SequenceRegressor):
    """Deterministic technical-alpha baseline using named latest-step features."""

    def __init__(self, name: str, feature_weights: dict[str, float], cross_sectional_normalize: bool = True) -> None:
        self.name = name
        self.feature_weights = dict(feature_weights)
        self.cross_sectional_normalize = cross_sectional_normalize
        self._is_fitted = False

    def fit(self, train_dataset: Dataset, val_dataset: Dataset, epochs: int, device: str = "cpu") -> "WeightedTechnicalRegressor":
        self._is_fitted = True
        return self

    def predict_dataset(self, dataset: Dataset, device: str = "cpu") -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fit before prediction.")

        features_tensor, _ = _resolve_dataset_tensors(dataset)
        features = features_tensor.detach().cpu().numpy()
        if len(features) == 0:
            return np.array([], dtype=np.float32)

        feature_names = _resolve_feature_names(dataset)
        latest = features[:, -1, :]
        scores = np.zeros(len(features), dtype=np.float32)
        for feature_name, weight in self.feature_weights.items():
            if feature_name in feature_names:
                scores += float(weight) * latest[:, feature_names.index(feature_name)]

        if self.cross_sectional_normalize:
            scores = _normalize_by_date(scores, _resolve_sample_dates(dataset))

        return scores.astype(np.float32, copy=False)


class DummyMeanRegressor:
    """Fallback regressor used when a baseline cannot be fit robustly."""

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "DummyMeanRegressor":
        if len(targets):
            self.value = float(np.mean(targets))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), self.value, dtype=np.float32)


def build_default_baseline_specs(random_state: int = 42, include_tree_models: bool = False) -> list[BaselineSpec]:
    """Return the default baseline model set used for comparison."""

    specs = [
        BaselineSpec(
            name="ridge",
            factory=lambda _input_size: SklearnSequenceRegressor(
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", Ridge(alpha=1.0)),
                    ]
                ),
                name="ridge",
            ),
        ),
        BaselineSpec(
            name="elastic_net",
            factory=lambda _input_size: SklearnSequenceRegressor(
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=random_state)),
                    ]
                ),
                name="elastic_net",
            ),
        ),
        BaselineSpec(
            name="ridge_last_step",
            factory=lambda _input_size: SklearnLastStepRegressor(
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", Ridge(alpha=1.0)),
                    ]
                ),
                name="ridge_last_step",
            ),
        ),
        BaselineSpec(
            name="elastic_net_last_step",
            factory=lambda _input_size: SklearnLastStepRegressor(
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=random_state)),
                    ]
                ),
                name="elastic_net_last_step",
            ),
        ),
        BaselineSpec(
            name="ridge_summary",
            factory=lambda _input_size: SklearnSequenceSummaryRegressor(
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", Ridge(alpha=1.0)),
                    ]
                ),
                name="ridge_summary",
            ),
        ),
        BaselineSpec(
            name="technical_trend",
            factory=lambda _input_size: WeightedTechnicalRegressor(
                name="technical_trend",
                feature_weights={
                    "tech__momentum_12_1": 0.35,
                    "tech__return_63d": 0.25,
                    "tech__return_21d": 0.20,
                    "tech__price_to_ma200": 0.15,
                    "tech__price_to_ma50": 0.10,
                    "tech__volatility_21d": -0.10,
                },
            ),
        ),
        BaselineSpec(
            name="technical_reversal",
            factory=lambda _input_size: WeightedTechnicalRegressor(
                name="technical_reversal",
                feature_weights={
                    "tech__bollinger_zscore": -0.45,
                    "tech__return_5d": -0.30,
                    "tech__return_1d": -0.15,
                    "tech__volatility_21d": -0.10,
                },
            ),
        ),
        BaselineSpec(
            name="technical_blend",
            factory=lambda _input_size: WeightedTechnicalRegressor(
                name="technical_blend",
                feature_weights={
                    "tech__momentum_12_1": 0.25,
                    "tech__return_63d": 0.15,
                    "tech__return_21d": 0.10,
                    "tech__bollinger_zscore": -0.25,
                    "tech__return_5d": -0.15,
                    "tech__volatility_21d": -0.10,
                },
            ),
        ),
    ]

    if include_tree_models:
        specs.extend(
            [
                BaselineSpec(
                    name="random_forest",
                    factory=lambda _input_size: SklearnLastStepRegressor(
                        RandomForestRegressor(
                            n_estimators=200,
                            min_samples_leaf=5,
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                        name="random_forest",
                    ),
                ),
                BaselineSpec(
                    name="gradient_boosting",
                    factory=lambda _input_size: SklearnLastStepRegressor(
                        GradientBoostingRegressor(random_state=random_state),
                        name="gradient_boosting",
                    ),
                ),
            ]
        )

    return specs


def _resolve_dataset_tensors(dataset: Dataset) -> tuple:
    if isinstance(dataset, Subset):
        features, targets = _resolve_dataset_tensors(dataset.dataset)
        indices = np.asarray(dataset.indices, dtype=int)
        return features[indices], targets[indices]

    if hasattr(dataset, "features") and hasattr(dataset, "targets"):
        return dataset.features, dataset.targets

    raise TypeError("Unsupported dataset type for sklearn baseline training.")


def _resolve_feature_names(dataset: Dataset) -> list[str]:
    if isinstance(dataset, Subset):
        return _resolve_feature_names(dataset.dataset)
    return list(getattr(dataset, "feature_names", []))


def _resolve_sample_dates(dataset: Dataset) -> list:
    if isinstance(dataset, Subset):
        sample_dates = _resolve_sample_dates(dataset.dataset)
        return [sample_dates[idx] for idx in dataset.indices]
    return list(getattr(dataset, "sample_dates", []))


def _normalize_by_date(scores: np.ndarray, sample_dates: list) -> np.ndarray:
    if len(scores) == 0 or len(sample_dates) != len(scores):
        return scores

    normalized = scores.astype(np.float32, copy=True)
    for date in set(sample_dates):
        indices = [idx for idx, sample_date in enumerate(sample_dates) if sample_date == date]
        if len(indices) < 2:
            continue
        date_scores = normalized[indices]
        std = float(date_scores.std())
        if std == 0.0:
            normalized[indices] = 0.0
        else:
            normalized[indices] = (date_scores - float(date_scores.mean())) / std
    return normalized
