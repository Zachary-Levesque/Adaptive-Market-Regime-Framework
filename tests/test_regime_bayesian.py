import numpy as np
import pandas as pd

from src.regime.bayesian import BayesianRegimeSmoothing


def test_bayesian_smoothing_outputs_normalized_probabilities():
    index = pd.date_range("2024-01-01", periods=5, freq="B")
    raw_probs = pd.DataFrame(
        [
            [0.80, 0.10, 0.05, 0.05],
            [0.70, 0.15, 0.10, 0.05],
            [0.20, 0.50, 0.20, 0.10],
            [0.10, 0.20, 0.60, 0.10],
            [0.05, 0.10, 0.20, 0.65],
        ],
        index=index,
        columns=["Bull Trending", "Low-Vol Compression", "Bear Trending", "High-Vol Crisis"],
    )
    labels = pd.Series([0, 0, 1, 2, 3], index=index)
    smoother = BayesianRegimeSmoothing()

    transition = smoother.compute_transition_matrix(labels)
    smoothed = smoother.smooth_probabilities(raw_probs, transition)
    durations = smoother.compute_regime_duration(labels)

    assert transition.shape == (4, 4)
    assert np.allclose(smoothed.sum(axis=1).to_numpy(), 1.0)
    assert durations[0] == 2.0
    assert durations[1] == 1.0
