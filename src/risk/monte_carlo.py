"""Monte Carlo portfolio risk simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MonteCarloResult:
    daily_returns: np.ndarray
    terminal_returns: np.ndarray


class MonteCarloRiskEngine:
    """Simulate portfolio return paths and compute tail-risk metrics."""

    def __init__(self, random_state: int | None = 42) -> None:
        self.random_state = random_state

    def simulate_returns(
        self,
        mean_returns: pd.Series | np.ndarray,
        cov_matrix: pd.DataFrame | np.ndarray,
        n_simulations: int = 10_000,
        horizon: int = 252,
        weights: pd.Series | np.ndarray | None = None,
    ) -> MonteCarloResult:
        """Generate simulated daily and terminal portfolio returns."""
        means = np.asarray(mean_returns, dtype=float).reshape(-1)
        covariance = np.asarray(cov_matrix, dtype=float)
        if covariance.shape != (len(means), len(means)):
            raise ValueError("cov_matrix must be square and match mean_returns length.")

        if weights is None:
            portfolio_weights = np.full(len(means), 1.0 / len(means))
        else:
            portfolio_weights = np.asarray(weights, dtype=float).reshape(-1)
            if len(portfolio_weights) != len(means):
                raise ValueError("weights must match mean_returns length.")

        rng = np.random.default_rng(self.random_state)
        asset_returns = rng.multivariate_normal(
            mean=means,
            cov=covariance,
            size=(n_simulations, horizon),
            check_valid="ignore",
        )
        portfolio_daily = asset_returns @ portfolio_weights
        terminal_returns = np.prod(1.0 + portfolio_daily, axis=1) - 1.0
        return MonteCarloResult(daily_returns=portfolio_daily, terminal_returns=terminal_returns)

    @staticmethod
    def compute_var(simulated_returns: np.ndarray, confidence: float = 0.95) -> float:
        """Return positive loss VaR for terminal or daily simulated returns."""
        returns = np.asarray(simulated_returns, dtype=float).reshape(-1)
        if returns.size == 0:
            return 0.0
        return float(-np.quantile(returns, 1.0 - confidence))

    @staticmethod
    def compute_cvar(simulated_returns: np.ndarray, confidence: float = 0.95) -> float:
        """Return positive loss CVaR/expected shortfall beyond VaR."""
        returns = np.asarray(simulated_returns, dtype=float).reshape(-1)
        if returns.size == 0:
            return 0.0
        threshold = np.quantile(returns, 1.0 - confidence)
        tail = returns[returns <= threshold]
        if tail.size == 0:
            return 0.0
        return float(-tail.mean())

    @staticmethod
    def compute_parametric_var(
        weights: pd.Series | np.ndarray,
        mean_returns: pd.Series | np.ndarray,
        cov_matrix: pd.DataFrame | np.ndarray,
        confidence: float = 0.95,
    ) -> float:
        """Compute one-period delta-normal VaR as a positive loss."""
        from scipy.stats import norm

        portfolio_weights = np.asarray(weights, dtype=float).reshape(-1)
        means = np.asarray(mean_returns, dtype=float).reshape(-1)
        covariance = np.asarray(cov_matrix, dtype=float)
        if len(portfolio_weights) != len(means):
            raise ValueError("weights must match mean_returns length.")

        portfolio_mean = float(portfolio_weights @ means)
        portfolio_std = float(np.sqrt(portfolio_weights @ covariance @ portfolio_weights))
        quantile = norm.ppf(1.0 - confidence, loc=portfolio_mean, scale=portfolio_std)
        return float(-quantile)

    def summarize(
        self,
        mean_returns: pd.Series | np.ndarray,
        cov_matrix: pd.DataFrame | np.ndarray,
        weights: pd.Series | np.ndarray | None = None,
        n_simulations: int = 10_000,
        horizon: int = 252,
        confidence: float = 0.95,
    ) -> dict[str, float]:
        result = self.simulate_returns(
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            weights=weights,
            n_simulations=n_simulations,
            horizon=horizon,
        )
        terminal = result.terminal_returns
        return {
            "mean_terminal_return": float(terminal.mean()),
            "terminal_volatility": float(terminal.std(ddof=0)),
            "var": self.compute_var(terminal, confidence=confidence),
            "cvar": self.compute_cvar(terminal, confidence=confidence),
        }

    @staticmethod
    def plot_return_distribution(
        simulated_returns: np.ndarray,
        var: float,
        cvar: float,
        path: str | Path | None = None,
    ):
        returns = np.asarray(simulated_returns, dtype=float).reshape(-1)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(returns, bins=60, color="#4263eb", alpha=0.75)
        ax.axvline(-var, color="#c92a2a", linestyle="--", label=f"VaR: {var:.2%}")
        ax.axvline(-cvar, color="#7f1d1d", linestyle="--", label=f"CVaR: {cvar:.2%}")
        ax.set_title("Simulated Return Distribution")
        ax.set_xlabel("Return")
        ax.set_ylabel("Frequency")
        ax.legend()
        fig.tight_layout()

        if path is not None:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150)

        return fig

