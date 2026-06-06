# AMRF Results

This document is the short version of the project results. It is based on the saved parquet artifacts in `data/`.

## Full-Period Backtest

Selected signal: `regime_portfolio_selector`

Portfolio layer: regime-aware alpha/SPY blend

| Series | Annual Return | Sharpe | Sortino | Calmar | Max Drawdown | Total Return |
|---|---:|---:|---:|---:|---:|---:|
| AMRF Strategy | 16.35% | 1.0866 | 1.0412 | 0.7887 | -20.73% | 17.61x |
| SPY | 14.53% | 0.9500 | 0.8966 | 0.4660 | -31.18% | 12.72x |
| Equal Weight | 13.94% | 0.8549 | 0.8149 | 0.2814 | -49.54% | 11.42x |
| 63D Momentum | 9.87% | 0.7036 | 0.6705 | 0.3534 | -27.93% | 5.15x |

The selected AMRF portfolio beats SPY by `0.1366` Sharpe on the full backtest window.

Allocation policy:

| Regime | Alpha Sleeve | SPY Sleeve |
|---|---:|---:|
| 0 | 0% | 100% |
| 1 | 25% | 75% |
| 2 | 25% | 75% |
| 3 | 75% | 25% |

## Regime-Conditional Performance

From `data/results/regime_performance.parquet`.

| Regime | Annual Return | Sharpe | Sortino | Calmar | Max Drawdown | Total Return | n_days |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bull Trending (0) | 45.48% | 2.3154 | 2.1806 | 4.7289 | -9.62% | 0.68x | 347 |
| Low-Vol Compression (1) | 14.37% | 1.0953 | 1.0357 | 0.9338 | -15.39% | 1.66x | 1839 |
| Bear Trending (2) | 14.60% | 0.9605 | 0.9198 | 0.7943 | -18.39% | 0.81x | 1095 |
| High-Vol Crisis (3) | 14.21% | 0.8992 | 0.8724 | 0.8660 | -16.41% | 1.31x | 1584 |

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
| GFC | 0.82% | -11.10% | 20.09% | 146 | ok |
| COVID | -15.32% | -16.22% | 19.45% | 24 | ok |
| Rate Hike | -17.50% | -20.29% | 21.93% | 251 | ok |
