import pandas as pd

from src.regime.gmm import RegimeGMM


def test_compare_with_hmm_returns_perfect_score_for_identical_labels():
    index = pd.date_range("2024-01-01", periods=6, freq="B")
    hmm_labels = pd.Series([0, 0, 1, 1, 2, 3], index=index)
    gmm_labels = pd.Series([0, 0, 1, 1, 2, 3], index=index)

    score = RegimeGMM.compare_with_hmm(hmm_labels, gmm_labels)

    assert score == 1.0
