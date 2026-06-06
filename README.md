# AMRF: Adaptive Market Regime Framework

AMRF is a completed quantitative research system that detects market regimes, selects regime-aware alpha signals, converts them into a risk-managed portfolio, and presents the result in a polished dashboard.

The project is designed to demonstrate end-to-end ability across quantitative finance, machine learning, backtesting, risk engineering, data pipelines, and product-grade dashboard development.

---

## What This Project Does

Most trading strategies assume markets behave consistently. AMRF treats markets as non-stationary: a strategy that works in a calm bull market may fail in a crisis, a drawdown, or a sideways volatility-compression regime.

AMRF solves that problem by building a full research pipeline:

1. Ingest historical prices, factors, macro inputs, and technical features.
2. Detect latent market regimes using an HMM-backed regime engine.
3. Compare alpha models and technical baselines with walk-forward validation.
4. Select the strongest deployable alpha sleeve.
5. Blend the alpha sleeve with SPY using a deterministic regime-aware allocation layer.
6. Backtest the final portfolio against SPY, equal-weight, and momentum baselines.
7. Run stress tests and readiness checks.
8. Surface the system state in a modern React dashboard.

The final portfolio is not just a raw signal. It is a regime-aware allocation policy:

| Regime | Alpha Sleeve | SPY Sleeve |
|---|---:|---:|
| 0 | 0% | 100% |
| 1 | 25% | 75% |
| 2 | 25% | 75% |
| 3 | 75% | 25% |

---

## Current Results

Current saved artifact snapshot from `data/results/performance_report.parquet`:

| Series | Annual Return | Sharpe | Sortino | Calmar | Max Drawdown | Total Return |
|---|---:|---:|---:|---:|---:|---:|
| AMRF Strategy | 16.35% | 1.0866 | 1.0412 | 0.7887 | -20.73% | 17.61x |
| SPY | 14.53% | 0.9500 | 0.8966 | 0.4660 | -31.18% | 12.72x |
| Equal Weight | 13.94% | 0.8549 | 0.8149 | 0.2814 | -49.54% | 11.42x |
| 63D Momentum | 9.87% | 0.7036 | 0.6705 | 0.3534 | -27.93% | 5.15x |

Current gates:

```text
Strategy Sharpe: 1.0866
SPY Sharpe:      0.9500
Ready for RL:    True
Project complete: True
```

The final strategy beats SPY on Sharpe and has a materially lower max drawdown in the saved backtest artifacts.

---

## Skills Demonstrated

This project demonstrates:

- quantitative research pipeline design
- financial data ingestion and validation
- feature engineering for cross-sectional assets
- market regime modeling
- Hidden Markov Models, GMM validation, Bayesian smoothing, and Kalman filtering
- walk-forward model validation
- signal diagnostics with IC, rank IC, hit rate, and regime-level performance
- transaction-cost-aware backtesting
- stress testing across GFC, COVID, and rate-hike windows
- risk metrics including Sharpe, Sortino, Calmar, drawdown, win rate, and profit factor
- deterministic portfolio construction and benchmark blending
- reinforcement-learning infrastructure with PPO
- execution simulation with costs and slippage
- FastAPI backend development
- React, TypeScript, Tailwind, Recharts, and responsive dashboard design
- automated test coverage and project completion gates

---

## Architecture

```text
Raw Data
  |
  v
Data Pipeline
  prices, returns, factors, macro data, technical features
  |
  v
Regime Engine
  HMM probabilities, GMM validation, Bayesian transitions, Kalman smoothing
  |
  v
Alpha Layer
  technical baselines, linear models, regime selectors, model comparison
  |
  v
Portfolio Layer
  selected alpha sleeve + regime-aware SPY blend
  |
  v
Risk + Readiness
  backtest, benchmarks, stress tests, diagnostics, completion report
  |
  v
Dashboards
  React product UI + Streamlit research viewer
```

---

## Core Modules

| Area | Path | Purpose |
|---|---|---|
| Configuration | `src/config.py` | Loads YAML config and typed runtime settings |
| Data pipeline | `src/data/` | Imports prices, builds returns, factors, macro, and technical features |
| Regime engine | `src/regime/` | Fits HMM/GMM regime models and writes regime probabilities |
| Alpha models | `src/alpha/` | Builds signals, compares models, diagnoses alpha quality |
| Risk engine | `src/risk/` | Backtests selected portfolio and computes risk metrics |
| RL layer | `src/rl/` | PPO position-sizing infrastructure |
| Execution layer | `src/execution/` | Execution simulation, costs, slippage, Alpaca hooks |
| React dashboard | `src/dashboard/frontend/` | Polished product UI |
| FastAPI backend | `src/dashboard/backend/` | Serves dashboard data from parquet artifacts |
| Streamlit dashboard | `dashboard/app.py` | Research artifact viewer |
| Tests | `tests/` | Regression coverage for data, alpha, regime, risk, dashboard, and completion logic |

---

## Important Artifacts

| Artifact | Meaning |
|---|---|
| `data/processed/prices.parquet` | Cleaned market prices |
| `data/processed/returns.parquet` | Daily asset returns |
| `data/regimes/regime_labels.parquet` | Current and historical regime labels |
| `data/regimes/regime_probs.parquet` | HMM regime probabilities |
| `data/processed/alpha_signal_selection.parquet` | Selected alpha model manifest |
| `data/results/position_weights.parquet` | Final blended portfolio weights |
| `data/results/alpha_sleeve_position_weights.parquet` | Alpha-only sleeve weights |
| `data/results/allocation_policy.parquet` | Regime-aware alpha/SPY allocation rule |
| `data/results/performance_report.parquet` | Strategy and benchmark performance metrics |
| `data/processed/alpha_readiness_report.parquet` | Readiness gate checks |
| `data/results/project_completion_report.parquet` | Final completion gate |

---

## How To Run

### 1. Install and build artifacts

From the repo root:

```bash
./run_pipeline.sh --no-dashboard
```

Useful variants:

```bash
# Import a local vendor archive first
./run_pipeline.sh --source ./d_us_txt.zip --no-dashboard

# Allow remote downloads for missing data
./run_pipeline.sh --allow-remote-downloads --source ./d_us_txt.zip --no-dashboard

# Include PPO training and RL backtest
./run_pipeline.sh --with-rl --source ./d_us_txt.zip --no-dashboard
```

### 2. Run the polished React dashboard

Start the FastAPI backend:

```bash
.venv/bin/python -m uvicorn src.dashboard.backend.main:app --host 127.0.0.1 --port 8000
```

Start the React frontend:

```bash
cd src/dashboard/frontend
npm run dev -- --host 127.0.0.1
```

Open:

```text
http://127.0.0.1:5173
```

### 3. Run the Streamlit research dashboard

```bash
.venv/bin/python -m streamlit run dashboard/app.py --server.headless true --server.port 8502
```

Open:

```text
http://localhost:8502
```

---

## How To Validate

Run the full test suite:

```bash
.venv/bin/python -m pytest
```

Expected current result:

```text
98 passed
```

Rebuild the key final reports:

```bash
.venv/bin/python -m src.risk.build_phase4 --config configs/config.yaml
.venv/bin/python -m src.alpha.build_diagnostics --config configs/config.yaml
.venv/bin/python -m src.alpha.build_readiness --config configs/config.yaml
.venv/bin/python -m src.build_completion_report --config configs/config.yaml
```

Expected final status:

```text
Ready for RL: True
Project complete: True
```

Build the React dashboard:

```bash
cd src/dashboard/frontend
npm run build
```

The build currently succeeds. Vite may report a non-blocking bundle-size warning.

---

## How To Interpret The Dashboard

The React dashboard is the main product surface.

- **Current Regime**: The market state detected by the HMM.
- **Strategy Sharpe**: Risk-adjusted performance of AMRF.
- **Max Drawdown**: Worst historical peak-to-trough decline.
- **Readiness**: Whether the strategy clears all quality gates.
- **Equity Curve**: AMRF versus SPY.
- **Portfolio Allocation**: Current alpha/SPY blend.
- **Regime Probabilities**: HMM confidence across market states.
- **Readiness Gate**: Individual checks that determine whether the strategy is research-ready.

If readiness fails, the strategy should not be treated as deployable.

---

## Raw Data Notes

The pipeline can load local vendor files before trying remote providers. It looks recursively under `data.local_data_dir` for Stooq-style files such as:

```text
aapl.us.txt
aapl.txt
spy.us.txt
qqq.us.txt
```

Required columns:

```text
Date, Open, High, Low, Close
```

Optional columns:

```text
Adj Close, Volume
```

Import a local ZIP or folder:

```bash
.venv/bin/python -m src.data.import_price_files --config configs/config.yaml --source /path/to/vendor/files-or-zip
```

Validate inputs:

```bash
.venv/bin/python -m src.data.validate_phase1_inputs --config configs/config.yaml
```

---

## Trading Disclaimer

AMRF is not a live trading bot and does not provide personalized financial advice. It is a research framework. Before any real-money usage, the system would need paper trading, broker integration hardening, monitoring, position limits, kill switches, compliance review, and operational risk controls.

---

## Research Influences

- Hamilton, J.D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series*
- Ang, A. & Bekaert, G. (2002). *Regime Switches in Interest Rates*
- Fama, E. & French, K. (2015). *A Five-Factor Asset Pricing Model*
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*
- Gu, S., Kelly, B. & Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning*
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms*
