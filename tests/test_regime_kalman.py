import numpy as np
import pandas as pd

from src.regime.kalman import KalmanRegimeFilter


def test_kalman_filter_preserves_probability_simplex():
    observations = pd.DataFrame(
        [
            [0.75, 0.10, 0.10, 0.05],
            [0.72, 0.12, 0.11, 0.05],
            [0.20, 0.30, 0.40, 0.10],
            [0.15, 0.20, 0.50, 0.15],
        ],
        index=pd.date_range("2024-01-01", periods=4, freq="B"),
        columns=["Bull Trending", "Low-Vol Compression", "Bear Trending", "High-Vol Crisis"],
    )

    filtered = KalmanRegimeFilter().filter(observations)

    assert filtered.shape == observations.shape
    assert np.allclose(filtered.sum(axis=1).to_numpy(), 1.0)
    assert (filtered.to_numpy() >= 0.0).all()
