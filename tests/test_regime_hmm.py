from pathlib import Path

import numpy as np
import pandas as pd

from src.regime.hmm import RegimeHMM


def _synthetic_regime_features() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    index = pd.date_range("2021-01-01", periods=240, freq="B")
    hidden_states = np.repeat([0, 1, 2, 3], 60)
    means = np.array(
        [
            [0.012, 0.10, 12.0],
            [0.001, 0.04, 10.0],
            [-0.006, 0.18, 20.0],
            [-0.015, 0.35, 32.0],
        ]
    )
    noise = np.column_stack(
        [
            rng.normal(0.0, 0.002, size=len(hidden_states)),
            rng.normal(0.0, 0.01, size=len(hidden_states)),
            rng.normal(0.0, 1.0, size=len(hidden_states)),
        ]
    )
    values = means[hidden_states] + noise
    return pd.DataFrame(
        values,
        index=index,
        columns=["spy_return", "spy_volatility_21d", "vix_level"],
    )


def test_regime_hmm_fit_predict_and_persist(tmp_path: Path):
    features = _synthetic_regime_features()
    model = RegimeHMM(n_regimes=4, n_iter=50, n_restarts=3, random_state=11)

    model.fit(features)
    raw_labels = model.predict_regimes(features, apply_mapping=False)
    summary = model.label_regimes(raw_labels, features["spy_return"])
    labels = model.predict_regimes(features)
    probs = model.predict_proba(features)

    assert len(set(raw_labels)) == 4
    assert summary["canonical_name"].tolist() == [
        "Bull Trending",
        "Low-Vol Compression",
        "Bear Trending",
        "High-Vol Crisis",
    ]
    assert set(labels.unique()) == {0, 1, 2, 3}
    assert np.allclose(probs.sum(axis=1).to_numpy(), 1.0)

    path = tmp_path / "hmm.pkl"
    model.save(path)
    loaded = RegimeHMM.load(path)
    reloaded_probs = loaded.predict_proba(features)

    assert reloaded_probs.shape == probs.shape
    assert np.allclose(reloaded_probs.to_numpy(), probs.to_numpy())
