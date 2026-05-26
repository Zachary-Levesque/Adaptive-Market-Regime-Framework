"""Bayesian smoothing for regime probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


class BayesianRegimeSmoothing:
    """Smooth regime probabilities with an empirical transition matrix."""

    def __init__(self, prior_count: float = 1.0, epsilon: float = 1e-9) -> None:
        self.prior_count = prior_count
        self.epsilon = epsilon

    def compute_transition_matrix(self, regime_labels: pd.Series) -> pd.DataFrame:
        labels = pd.Series(regime_labels).dropna().astype(int)
        states = sorted(labels.unique().tolist())
        counts = pd.DataFrame(
            self.prior_count,
            index=states,
            columns=states,
            dtype=float,
        )

        for current, nxt in zip(labels.iloc[:-1], labels.iloc[1:]):
            counts.loc[int(current), int(nxt)] += 1.0

        transition = counts.div(counts.sum(axis=1), axis=0)
        transition.index.name = "from_regime"
        transition.columns.name = "to_regime"
        return transition

    def smooth_probabilities(
        self,
        raw_probs: pd.DataFrame,
        transition_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        probs = raw_probs.copy().astype(float)
        transitions = transition_matrix.to_numpy(dtype=float)
        emissions = np.clip(probs.to_numpy(dtype=float), self.epsilon, 1.0)
        emissions = emissions / emissions.sum(axis=1, keepdims=True)

        forward = np.zeros_like(emissions)
        backward = np.zeros_like(emissions)

        forward[0] = self._normalize(emissions[0])
        for idx in range(1, len(emissions)):
            prior = forward[idx - 1] @ transitions
            forward[idx] = self._normalize(emissions[idx] * prior)

        backward[-1] = np.ones(emissions.shape[1], dtype=float)
        for idx in range(len(emissions) - 2, -1, -1):
            message = transitions @ (emissions[idx + 1] * backward[idx + 1])
            backward[idx] = self._normalize(message)

        smoothed = np.vstack(
            [self._normalize(forward[idx] * backward[idx]) for idx in range(len(emissions))]
        )
        return pd.DataFrame(smoothed, index=probs.index, columns=probs.columns)

    @staticmethod
    def compute_regime_duration(regime_labels: pd.Series) -> dict[int, float]:
        labels = pd.Series(regime_labels).dropna().astype(int)
        if labels.empty:
            return {}

        durations: dict[int, list[int]] = {}
        run_label = int(labels.iloc[0])
        run_length = 1

        for label in labels.iloc[1:]:
            label = int(label)
            if label == run_label:
                run_length += 1
                continue
            durations.setdefault(run_label, []).append(run_length)
            run_label = label
            run_length = 1

        durations.setdefault(run_label, []).append(run_length)
        return {label: float(np.mean(lengths)) for label, lengths in durations.items()}

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        total = float(vector.sum())
        if total <= 0.0:
            return np.full_like(vector, 1.0 / len(vector))
        return vector / total
