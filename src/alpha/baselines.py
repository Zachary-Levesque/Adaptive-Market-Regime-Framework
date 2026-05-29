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
    factory: Callable[[], "SequenceRegressor"]


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
            factory=lambda: SklearnSequenceRegressor(
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
            factory=lambda: SklearnSequenceRegressor(
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=random_state)),
                    ]
                ),
                name="elastic_net",
            ),
        ),
    ]

    if include_tree_models:
        specs.extend(
            [
                BaselineSpec(
                    name="random_forest",
                    factory=lambda: SklearnSequenceRegressor(
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
                    factory=lambda: SklearnSequenceRegressor(
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
