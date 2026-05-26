"""Simple Kalman filtering for regime probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


class KalmanRegimeFilter:
    """Apply a random-walk Kalman filter to probability vectors."""

    def __init__(self, process_variance: float = 1e-4, observation_variance: float = 1e-2) -> None:
        self.process_variance = process_variance
        self.observation_variance = observation_variance

    def filter(self, observations: pd.DataFrame) -> pd.DataFrame:
        probs = observations.copy().astype(float)
        values = probs.to_numpy(dtype=float)
        n_steps, n_states = values.shape

        identity = np.eye(n_states)
        process_noise = self.process_variance * identity
        observation_noise = self.observation_variance * identity

        state = self._normalize(values[0])
        covariance = identity.copy()
        smoothed = np.zeros_like(values)
        smoothed[0] = state

        for idx in range(1, n_steps):
            predicted_state = state
            predicted_covariance = covariance + process_noise
            innovation = values[idx] - predicted_state
            innovation_covariance = predicted_covariance + observation_noise
            kalman_gain = predicted_covariance @ np.linalg.inv(innovation_covariance)
            state = predicted_state + kalman_gain @ innovation
            covariance = (identity - kalman_gain) @ predicted_covariance
            smoothed[idx] = self._normalize(state)

        return pd.DataFrame(smoothed, index=probs.index, columns=probs.columns)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        clipped = np.clip(vector, 1e-9, None)
        return clipped / clipped.sum()
