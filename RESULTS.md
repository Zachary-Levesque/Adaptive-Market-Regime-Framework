# AMRF Results

This document is the short version of the project results. It is based on the saved parquet artifacts in `data/`.

## Full-Period Backtest

Selected signal: `regime_portfolio_selector`

| Series | Annual Return | Sharpe | Sortino | Calmar | Max Drawdown | Total Return |
|---|---:|---:|---:|---:|---:|---:|
| AMRF Strategy | 17.32% | 0.9366 | 0.8907 | 0.6979 | -24.82% | 16.44x |
| SPY | 13.34% | 0.8742 | 0.8225 | 0.3956 | -33.73% | 8.41x |
| Equal Weight | 11.75% | 0.7421 | 0.7066 | 0.2389 | -49.19% | 6.30x |
| 63D Momentum | 8.11% | 0.6023 | 0.5620 | 0.2675 | -30.31% | 3.04x |

The selected AMRF strategy beats SPY by `0.0624` Sharpe on the full backtest window.

## Regime-Conditional Performance

From `data/results/regime_performance.parquet`.

| Regime | Annual Return | Sharpe | Sortino | Calmar | Max Drawdown | Total Return | n_days |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bull Trending (0) | 31.44% | 1.3825 | 1.2407 | 2.6622 | -11.81% | 0.58x | 421 |
| Low-Vol Compression (1) | 20.49% | 1.2135 | 1.1508 | 1.4612 | -14.02% | 2.10x | 1530 |
| Bear Trending (2) | 16.19% | 0.8388 | 0.7874 | 0.6556 | -24.69% | 1.19x | 1313 |
| High-Vol Crisis (3) | 10.40% | 0.6056 | 0.5959 | 0.4189 | -24.82% | 0.63x | 1245 |

Selected-signal diagnostics from `data/processed/alpha_diagnostics_by_regime.parquet`:

| Regime | Mean IC | Mean Rank IC | IC Positive Rate | Mean Hit Rate |
|---|---:|---:|---:|---:|
| Bull Trending | 0.030132 | 0.058133 | 0.525140 | 0.512389 |
| Low-Vol Compression | 0.056953 | 0.044020 | 0.557745 | 0.506781 |
| Bear Trending | -0.000562 | 0.011471 | 0.501268 | 0.493473 |
| High-Vol Crisis | 0.052957 | 0.033731 | 0.563847 | 0.529365 |

## RL vs Static Signal

Test slice: 2022-2024

| Series | Sharpe | Total Return | Note |
|---|---:|---:|---|
| RL agent policy | 0.4976 | 0.2772 | Policy-level output before execution costs |
| RL agent execution | -0.1090 | -0.1246 | Post-execution result |
| Static signal policy | 0.7871 | 0.5188 | Static signal before execution costs |
| Static signal execution | 0.3781 | 0.1842 | Post-execution benchmark |
| SPY | 0.7228 | 0.3718 | Buy-and-hold benchmark |

Honest reading: the PPO agent learned a positive policy, but execution costs and turnover still reduce the realized test result below the static signal. The static signal remains the better production baseline.

## Stress Tests

From `data/results/stress_report.parquet`.

| Scenario | Period Return | Max Drawdown | Volatility | n_days | Status |
|---|---:|---:|---:|---:|---|
| GFC | -0.51% | -10.30% | 19.89% | 146 | ok |
| COVID | -7.00% | -14.13% | 28.69% | 24 | ok |
| Rate Hike | -13.37% | -23.30% | 19.80% | 251 | ok |

