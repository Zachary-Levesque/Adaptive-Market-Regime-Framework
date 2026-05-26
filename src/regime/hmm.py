"""Hidden Markov Model utilities for market regime detection."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:  # pragma: no cover - exercised indirectly depending on environment
    from loguru import logger
except ImportError:  # pragma: no cover - dependency may not be installed in CI/local env
    import logging

    logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegimeDefinition:
    canonical_label: int
    canonical_name: str
    mean_return: float
    volatility: float


class RegimeHMM:
    """Fit and apply a Gaussian HMM to regime features."""

    DEFAULT_REGIME_NAMES = {
        0: "Bull Trending",
        1: "Low-Vol Compression",
        2: "Bear Trending",
        3: "High-Vol Crisis",
    }

    def __init__(
        self,
        n_regimes: int = 4,
        covariance_type: str = "full",
        n_iter: int = 1000,
        random_state: int = 42,
        n_restarts: int = 10,
        regime_names: dict[int, str] | None = None,
    ) -> None:
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.n_restarts = n_restarts
        self.regime_names = regime_names or self.DEFAULT_REGIME_NAMES.copy()
        self.imputer_ = SimpleImputer(strategy="median")
        self.scaler_ = StandardScaler()
        self.model_: GaussianHMM | None = None
        self.feature_columns_: list[str] | None = None
        self.state_mapping_: dict[int, int] = {state: state for state in range(n_regimes)}
        self.state_definitions_: dict[int, RegimeDefinition] = {}
        self.best_score_: float | None = None

    def preprocess(self, features: pd.DataFrame, fit: bool = False) -> tuple[np.ndarray, pd.Index]:
        """Clean, align, and standardize features before model application."""
        frame = self._coerce_features(features)
        valid = frame.replace([np.inf, -np.inf], np.nan).dropna(how="any")
        if valid.empty:
            raise ValueError("No fully-observed feature rows available for regime modeling.")

        if fit:
            self.feature_columns_ = list(valid.columns)
            transformed = self.imputer_.fit_transform(valid)
            scaled = self.scaler_.fit_transform(transformed)
            return scaled, valid.index

        self._ensure_fitted()
        if self.feature_columns_ is None:
            raise RuntimeError("Model feature columns are unavailable.")
        transformed = self.imputer_.transform(valid.reindex(columns=self.feature_columns_))
        scaled = self.scaler_.transform(transformed)
        return scaled, valid.index

    def fit(self, features: pd.DataFrame) -> "RegimeHMM":
        """Fit the HMM using multiple random restarts and keep the best score."""
        x, _ = self.preprocess(features, fit=True)
        best_model: GaussianHMM | None = None
        best_score = -np.inf

        for restart in range(self.n_restarts):
            candidate = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state + restart,
            )
            candidate.fit(x)
            score = float(candidate.score(x))
            logger.info("HMM restart {}/{} log-likelihood: {:.4f}", restart + 1, self.n_restarts, score)
            if score > best_score:
                best_score = score
                best_model = candidate

        if best_model is None:
            raise RuntimeError("Failed to fit any HMM candidate.")

        self.model_ = best_model
        self.best_score_ = best_score
        logger.info("Selected HMM with log-likelihood {:.4f}", best_score)
        return self

    def predict_regimes(self, features: pd.DataFrame, apply_mapping: bool = True) -> pd.Series:
        """Predict hard regime labels for valid feature rows."""
        self._ensure_fitted()
        x, index = self.preprocess(features, fit=False)
        labels = pd.Series(self.model_.predict(x), index=index, name="regime", dtype=int)
        return self.remap_labels(labels) if apply_mapping else labels

    def predict_proba(self, features: pd.DataFrame, apply_mapping: bool = True) -> pd.DataFrame:
        """Predict per-regime probabilities for valid feature rows."""
        self._ensure_fitted()
        x, index = self.preprocess(features, fit=False)
        columns = [f"state_{idx}" for idx in range(self.n_regimes)]
        probs = pd.DataFrame(self.model_.predict_proba(x), index=index, columns=columns)
        return self.remap_probabilities(probs) if apply_mapping else probs

    def label_regimes(
        self,
        regime_labels: pd.Series,
        returns: pd.Series,
        set_mapping: bool = True,
    ) -> pd.DataFrame:
        """Assign canonical regime semantics based on return and volatility characteristics."""
        aligned_returns = pd.to_numeric(returns, errors="coerce")
        aligned_labels, aligned_returns = regime_labels.align(aligned_returns, join="inner")
        stats = (
            pd.DataFrame({"regime": aligned_labels.astype(int), "return": aligned_returns.astype(float)})
            .dropna()
            .groupby("regime")["return"]
            .agg(mean_return="mean", volatility="std")
        )
        stats["volatility"] = stats["volatility"].fillna(0.0)

        if stats.empty or len(stats.index) != self.n_regimes:
            raise ValueError(
                f"Expected statistics for {self.n_regimes} regimes, received {len(stats.index)}."
            )

        remaining = set(stats.index.tolist())
        crisis = min(remaining, key=lambda regime: (stats.loc[regime, "mean_return"], -stats.loc[regime, "volatility"]))
        remaining.remove(crisis)
        bull = max(remaining, key=lambda regime: (stats.loc[regime, "mean_return"], -stats.loc[regime, "volatility"]))
        remaining.remove(bull)
        low_vol = min(
            remaining,
            key=lambda regime: (stats.loc[regime, "volatility"], abs(stats.loc[regime, "mean_return"])),
        )
        remaining.remove(low_vol)
        bear = remaining.pop()

        mapping = {
            bull: 0,
            low_vol: 1,
            bear: 2,
            crisis: 3,
        }
        labeled = stats.copy()
        labeled["canonical_label"] = labeled.index.map(mapping)
        labeled["canonical_name"] = labeled["canonical_label"].map(self.regime_names)
        labeled = labeled.sort_values("canonical_label")

        if set_mapping:
            self.state_mapping_ = mapping
            self.state_definitions_ = {
                raw_label: RegimeDefinition(
                    canonical_label=int(mapping[raw_label]),
                    canonical_name=self.regime_names[int(mapping[raw_label])],
                    mean_return=float(stats.loc[raw_label, "mean_return"]),
                    volatility=float(stats.loc[raw_label, "volatility"]),
                )
                for raw_label in stats.index
            }

        return labeled

    def remap_labels(self, labels: pd.Series) -> pd.Series:
        """Convert raw state labels into canonical regime labels."""
        remapped = labels.map(self.state_mapping_).astype(int)
        remapped.name = labels.name
        return remapped

    def remap_probabilities(self, probabilities: pd.DataFrame) -> pd.DataFrame:
        """Reorder raw probability columns into canonical regime order."""
        inverse = {canonical: raw for raw, canonical in self.state_mapping_.items()}
        remapped = pd.DataFrame(index=probabilities.index)
        for canonical in range(self.n_regimes):
            raw = inverse[canonical]
            remapped[self.regime_names[canonical]] = probabilities.iloc[:, raw]
        return remapped

    def save(self, path: str | Path) -> None:
        """Persist the fitted HMM, preprocessing state, and regime mapping."""
        self._ensure_fitted()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            pickle.dump(self, handle)
        logger.info("Saved regime HMM to {}", output_path)

    @classmethod
    def load(cls, path: str | Path) -> "RegimeHMM":
        """Load a persisted HMM model."""
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"Expected serialized {cls.__name__}, received {type(model).__name__}.")
        return model

    def _ensure_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError("RegimeHMM must be fitted before use.")

    @staticmethod
    def _coerce_features(features: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("Expected regime features as a pandas DataFrame.")
        frame = features.copy()
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        return numeric.sort_index()
