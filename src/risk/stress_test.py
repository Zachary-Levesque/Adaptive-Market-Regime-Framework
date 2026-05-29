"""Historical and scenario stress testing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from src.risk.metrics import DrawdownProfile, PerformanceMetrics


@dataclass(frozen=True)
class StressResult:
    period_return: float
    max_drawdown: float
    volatility: float
    n_days: int


class StressTester:
    """Evaluate strategy behavior during named stress windows."""

    def __init__(self, periods_per_year: int = 252) -> None:
        self.metrics = PerformanceMetrics(periods_per_year=periods_per_year)
        self.periods_per_year = periods_per_year

    def run_stress_test(
        self,
        strategy_returns: pd.Series,
        stress_periods: Mapping[str, tuple[str, str] | list[str]],
    ) -> pd.DataFrame:
        clean = self.metrics._clean_returns(strategy_returns)
        rows: list[dict[str, float | int | str]] = []

        for name, bounds in stress_periods.items():
            if len(bounds) != 2:
                raise ValueError(f"Stress period {name} must contain start and end dates.")
            start, end = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])
            window = clean.loc[(clean.index >= start) & (clean.index <= end)]
            if window.empty:
                rows.append(
                    {
                        "scenario": name,
                        "period_return": np.nan,
                        "max_drawdown": np.nan,
                        "volatility": np.nan,
                        "n_days": 0,
                    }
                )
                continue

            rows.append(
                {
                    "scenario": name,
                    "period_return": float((1.0 + window).prod() - 1.0),
                    "max_drawdown": self.metrics.max_drawdown(window),
                    "volatility": self.metrics.annualized_volatility(window),
                    "n_days": int(len(window)),
                }
            )

        return pd.DataFrame(rows).set_index("scenario")

    def compute_drawdown_profile(self, returns: pd.Series) -> DrawdownProfile:
        return self.metrics.compute_drawdown_profile(returns)

    @staticmethod
    def scenario_analysis(
        weights: pd.Series | np.ndarray,
        factor_shocks: pd.Series | np.ndarray,
        exposures: pd.DataFrame | np.ndarray | None = None,
    ) -> float:
        """Estimate portfolio return under direct asset or factor shocks.

        If exposures are supplied, factor_shocks are mapped to asset shocks by
        exposures @ factor_shocks. Otherwise factor_shocks are interpreted as
        asset-level shocks aligned with weights.
        """
        portfolio_weights = np.asarray(weights, dtype=float).reshape(-1)
        shocks = np.asarray(factor_shocks, dtype=float).reshape(-1)

        if exposures is not None:
            exposure_matrix = np.asarray(exposures, dtype=float)
            if exposure_matrix.shape[1] != len(shocks):
                raise ValueError("exposures columns must match factor_shocks length.")
            shocks = exposure_matrix @ shocks

        if len(portfolio_weights) != len(shocks):
            raise ValueError("weights must match asset shock length.")

        return float(portfolio_weights @ shocks)

