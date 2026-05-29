import numpy as np
import pandas as pd

from src.risk.monte_carlo import MonteCarloRiskEngine


def test_monte_carlo_simulates_terminal_returns_and_tail_metrics():
    engine = MonteCarloRiskEngine(random_state=7)
    means = pd.Series([0.001, 0.0005], index=["A", "B"])
    cov = pd.DataFrame([[0.0004, 0.0001], [0.0001, 0.0002]], index=means.index, columns=means.index)

    result = engine.simulate_returns(means, cov, n_simulations=500, horizon=20, weights=np.array([0.6, 0.4]))
    var = engine.compute_var(result.terminal_returns, confidence=0.95)
    cvar = engine.compute_cvar(result.terminal_returns, confidence=0.95)

    assert result.daily_returns.shape == (500, 20)
    assert result.terminal_returns.shape == (500,)
    assert var > 0
    assert cvar >= var


def test_parametric_var_returns_positive_loss():
    engine = MonteCarloRiskEngine()
    var = engine.compute_parametric_var(
        weights=np.array([1.0]),
        mean_returns=np.array([0.0]),
        cov_matrix=np.array([[0.0001]]),
        confidence=0.95,
    )

    assert var > 0

