"""Gaussian Mixture validation utilities for regime clustering."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class RegimeGMM:
    """Validate regime separability with a Gaussian Mixture Model."""

    def __init__(self, n_components: int = 4, random_state: int = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.imputer_ = SimpleImputer(strategy="median")
        self.scaler_ = StandardScaler()
        self.model_: GaussianMixture | None = None
        self.feature_columns_: list[str] | None = None

    def fit(self, features: pd.DataFrame) -> "RegimeGMM":
        frame = self._coerce_features(features)
        valid = frame.replace([np.inf, -np.inf], np.nan).dropna(how="any")
        if valid.empty:
            raise ValueError("No fully-observed feature rows available for GMM fitting.")

        self.feature_columns_ = list(valid.columns)
        x = self.scaler_.fit_transform(self.imputer_.fit_transform(valid))
        self.model_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            random_state=self.random_state,
        )
        self.model_.fit(x)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.model_ is None or self.feature_columns_ is None:
            raise RuntimeError("RegimeGMM must be fitted before use.")
        frame = self._coerce_features(features)
        valid = frame.replace([np.inf, -np.inf], np.nan).dropna(how="any")
        x = self.scaler_.transform(self.imputer_.transform(valid.reindex(columns=self.feature_columns_)))
        labels = self.model_.predict(x)
        return pd.Series(labels, index=valid.index, name="gmm_regime", dtype=int)

    @staticmethod
    def compare_with_hmm(hmm_labels: pd.Series, gmm_labels: pd.Series) -> float:
        aligned_hmm, aligned_gmm = hmm_labels.align(gmm_labels, join="inner")
        if aligned_hmm.empty:
            raise ValueError("No overlapping labels available for HMM/GMM comparison.")
        return float(adjusted_rand_score(aligned_hmm, aligned_gmm))

    def plot_clusters(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        path: str | Path | None = None,
    ):
        if self.feature_columns_ is None:
            raise RuntimeError("RegimeGMM must be fitted before plotting clusters.")

        frame = self._coerce_features(features)
        valid = frame.replace([np.inf, -np.inf], np.nan).dropna(how="any")
        aligned_labels = labels.reindex(valid.index)
        x = self.scaler_.transform(self.imputer_.transform(valid.reindex(columns=self.feature_columns_)))
        embedding = PCA(n_components=2, random_state=self.random_state).fit_transform(x)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=aligned_labels.to_numpy(), cmap="tab10", s=18)
        ax.set_title("Regime Feature Clusters (PCA)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(*scatter.legend_elements(), title="Cluster")
        fig.tight_layout()

        if path is not None:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150)

        return fig

    @staticmethod
    def _coerce_features(features: pd.DataFrame) -> pd.DataFrame:
        frame = features.copy()
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        return frame.apply(pd.to_numeric, errors="coerce").sort_index()
