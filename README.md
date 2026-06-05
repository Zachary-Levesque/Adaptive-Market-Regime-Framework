# AMRF — Adaptive Market Regime Framework

A research-grade quantitative research pipeline that detects market regimes, compares regime-aware alpha models, and backtests the selected signal with explicit risk and readiness reporting.
 
---
 
## Overview
 
Most trading strategies are built assuming markets behave consistently. They don't.
 
A momentum strategy that thrives in a bull trend destroys capital in a sideways mean-reverting market. A volatility strategy optimized for calm periods blows up in a crisis. The fundamental problem is that virtually all retail and academic quant models are **regime-blind** — they apply a single static strategy to a dynamic, non-stationary market. This leads to catastrophic drawdowns precisely when capital preservation matters most.
 
**AMRF addresses this research problem** by modeling financial markets as a dynamic hidden system with four distinct regimes, training or selecting alpha models by regime, and testing whether the resulting signal is strong enough to justify later portfolio-construction or reinforcement-learning work.
 
---
 
## Current Status

Implemented and tested:

1. Builds historical prices, returns, factors, macro data, technical features, and regime features
2. Detects market regimes with HMM probabilities, GMM validation, Bayesian smoothing, and Kalman filtering
3. Trains and compares regime-specific alpha models against simpler baselines
4. Selects an alpha signal using walk-forward, transaction-cost, rebalance, forward-return horizon, and projected-backtest evidence
5. Backtests the selected signal against SPY, equal-weight, and momentum baselines
6. Reports alpha diagnostics, regime-conditional results, stress tests, readiness checks, RL artifacts, and a dashboard

Implemented but still research-gated:

1. PPO reinforcement-learning position sizing
2. Intraday execution through Alpaca

The current selected signal is `regime_portfolio_selector`. The readiness gate is still blocked by a small benchmark Sharpe miss versus SPY, so the RL layer remains an exploratory extension rather than the production baseline.

The repo now includes a one-command refresh-and-launch path via `./run_pipeline.sh`. It bootstraps a clean `.venv`, imports local Stooq data when available, rebuilds the artifacts, and launches the Streamlit dashboard. Rerun it whenever you want the saved artifacts refreshed to the latest available data source.
 
---
 
## Architecture
 
```
┌─────────────────────────────────────────────────────────────┐
│                        AMRF PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  RAW DATA          REGIME ENGINE        ALPHA MODELS        │
│  ──────────        ──────────────       ─────────────       │
│  yfinance    ───►  Hidden Markov  ───►  LSTM per regime     │
│  Alpaca API         Model (HMM)         Fama-French         │
│  FRED API          Gaussian Mix         factors             │
│                     Model (GMM)         Walk-forward CV     │
│                    Bayesian trans.                          │
│                                                             │
│  RISK ENGINE       READINESS GATE       OUTPUT              │
│  ──────────        ──────────────       ──────              │
│  Monte Carlo ◄───  Alpha quality ◄───  Signals             │
│  CVaR/VaR          Benchmark tests      Rankings            │
│  Stress tests      Stress coverage      CLI reports         │
│  Backtester        Pre-RL decision      Parquet artifacts   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
 
---
 
## The Four Market Regimes
 
| Regime | Description | Dominant Strategy | Typical Period |
|---|---|---|---|
| **Bull Trending** | Rising prices, low volatility, positive momentum | Momentum, factor tilt to quality/growth | 2013–2019, 2020 recovery |
| **Bear Trending** | Falling prices, rising volatility, negative breadth | Short bias, defensive, hedge | 2008, 2022 |
| **High-Vol Crisis** | Extreme volatility, correlation spike, deleveraging | Risk-off, cash, short vol | Mar 2020, Sep 2008 |
| **Low-Vol Compression** | Sideways, mean-reverting, tight ranges | Mean reversion, sell volatility | 2015–2016, mid-2019 |
 
---
 
## Technical Stack
 
### Core ML & Statistical Models
 
| Model | Module | Purpose |
|---|---|---|
| Hidden Markov Model (HMM) | Regime Detection | Unsupervised regime identification |
| Gaussian Mixture Model (GMM) | Regime Detection | Regime clustering & validation |
| Bayesian Inference | Regime Detection | Regime transition probabilities |
| Kalman Filter | Signal Processing | State estimation & noise reduction |
| LSTM (PyTorch) | Alpha Generation | Regime-specific return forecasting |
| Transformer (PyTorch) | Alpha Generation | Attention-based factor modeling |
| PPO (Stable-Baselines3) | RL Position Sizing | Dynamic weight tilting agent |
| Fama-French 5-Factor | Alpha Generation | Systematic risk factor exposure |
| Monte Carlo Simulation | Risk Engine | VaR & CVaR estimation |
| Markowitz MVO | Portfolio Construction | Efficient frontier optimization |
 
### Infrastructure
 
| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| PyTorch | Deep learning |
| hmmlearn | Hidden Markov Models |
| scikit-learn | Classical ML, GMM |
| stable-baselines3 | Reinforcement learning |
| pandas / numpy | Data manipulation |
| yfinance | Historical market data |
| alpaca-trade-api | Intraday data & execution |
| pandas-datareader | Fama-French factor data |
| scipy | Statistical functions |
| matplotlib / plotly | Visualization |
| Streamlit | Dashboard UI |
| Docker | Containerization |
 
---
 
## Modules
 
### Module 1 — Data Pipeline
Ingests and normalizes historical price, volume, and factor data. Computes returns, volatility features, and Fama-French factor exposures across a configurable stock universe. Enhanced with **Fractional Differentiation** and Macro-economic features.
 
### Module 2 — Regime Detection Engine
Fits a Hidden Markov Model with Gaussian emissions to identify 4 latent market regimes. Validated and cross-checked with a Gaussian Mixture Model. Bayesian smoothing applied to regime transition probabilities. Kalman filter used for state estimation.
 
### Module 3 — Regime-Specific Alpha Models
For each of the 4 regimes, a dedicated LSTM + Transformer model is trained on in-regime data only, using Fama-French factors and technical features as inputs. Walk-forward cross-validation prevents lookahead bias.
 
### Module 4 — Alpha Selection, Diagnostics, and Readiness
Compares alpha candidates against baselines, writes a selected signal manifest, diagnoses IC/rank-IC quality, and checks whether the selected signal is strong enough for RL work. The readiness gate includes active-history, alpha-quality, benchmark-relative, and stress-coverage checks.
 
### Module 5 — Risk Engine & Backtester
Full backtesting engine with Monte Carlo VaR/CVaR, historical stress testing (2008, COVID-19, 2022), and performance attribution. Reports Sharpe, Sortino, Calmar, max drawdown, win rate, and regime-conditional performance.
 
### Module 6 — Reinforcement Learning Position Sizing Agent
A PPO-based agent trained in a custom Gymnasium environment. The agent learns to dynamically "tilt" alpha signals (up to +/- 50%) based on current regime probabilities and risk-adjusted return targets.

### Module 7 — Intraday Execution Layer
Uses 5-minute bar data from Alpaca API for intraday entry timing. VWAP deviation signals, volume spikes, and momentum confirmation filters are applied to daily signals to improve execution quality and reduce slippage.
 
### Module 8 — Interactive Dashboard
Streamlit dashboard for regime states, equity curves, readiness checks, and risk diagnostics. It reads saved parquet artifacts and does not fetch live market data.
 
---
 
## Results

Current local artifact snapshot from `data/results/performance_report.parquet`:

| Metric | AMRF Strategy | Buy & Hold SPY | Equal Weight | 63D Momentum |
|---|---|---|---|---|
| Annual Return | 17.32% | 13.34% | 11.75% | 8.11% |
| Sharpe Ratio | 0.94 | 0.87 | 0.74 | 0.60 |
| Sortino Ratio | 0.89 | 0.82 | 0.71 | 0.56 |
| Calmar Ratio | 0.70 | 0.40 | 0.24 | 0.27 |
| Max Drawdown | -24.82% | -33.73% | -49.19% | -30.31% |
| Total Return | 16.44x | 8.41x | 6.30x | 3.04x |

Selected-signal diagnostics from `data/processed/alpha_diagnostics_by_regime.parquet`:

| Regime | Mean IC | Mean Rank IC | IC Positive Rate | Mean Hit Rate |
|---|---:|---:|---:|---:|
| Bull Trending | 0.030132 | 0.058133 | 0.525140 | 0.512389 |
| Low-Vol Compression | 0.056953 | 0.044020 | 0.557745 | 0.506781 |
| Bear Trending | -0.000562 | 0.011471 | 0.501268 | 0.493473 |
| High-Vol Crisis | 0.052957 | 0.033731 | 0.563847 | 0.529365 |

RL backtest comparison on the 2022-2024 test slice:

| Series | Sharpe | Total Return |
|---|---:|---:|
| RL agent policy | 0.4976 | 0.2772 |
| RL agent execution | -0.1090 | -0.1246 |
| Static signal policy | 0.7871 | 0.5188 |
| Static signal execution | 0.3781 | 0.1842 |
| SPY | 0.7228 | 0.3718 |

The honest takeaway is that the static signal is the production baseline. The PPO agent learned a positive policy on paper, but execution costs and turnover still reduce realized test performance below the static signal.

> Note: Results are from the current local walk-forward artifacts, not final claimed performance. Past performance does not guarantee future results.
 
---
 
## Project Structure
 
```
AMRF/
├── data/
│   ├── raw/                    # Raw price and factor data
│   ├── processed/              # Engineered features
│   └── regimes/                # Historical regime labels
├── src/
│   ├── data/
│   │   ├── ingestion.py        # Data pipeline
│   │   ├── features.py         # Feature engineering
│   │   └── factors.py          # Fama-French factor loading
│   ├── regime/
│   │   ├── hmm.py              # Hidden Markov Model
│   │   ├── gmm.py              # Gaussian Mixture Model
│   │   ├── bayesian.py         # Bayesian transition model
│   │   └── kalman.py           # Kalman filter
│   ├── alpha/
│   │   ├── lstm.py             # LSTM model per regime
│   │   ├── transformer.py      # Transformer model
│   │   ├── model_comparison.py # Baselines, selector, cost/rebalance sensitivity
│   │   ├── diagnostics.py      # Alpha IC/rank-IC diagnostics
│   │   ├── readiness.py        # Pre-RL readiness gate
│   │   └── walk_forward.py     # Walk-forward CV
│   ├── risk/
│   │   ├── monte_carlo.py      # Monte Carlo VaR/CVaR
│   │   ├── stress_test.py      # Historical stress tests
│   │   ├── backtester.py       # Selected-signal backtester
│   │   └── metrics.py          # Performance metrics
├── tests/
├── configs/
│   └── config.yaml
├── requirements.txt
├── README.md
└── instructions.md
```
 
---
 
## Quick Start

```bash
git clone https://github.com/Zachary-Levesque/Adaptive-Market-Regime-Framework.git
cd Adaptive-Market-Regime-Framework
./run_pipeline.sh --source ./d_us_txt.zip
```

Useful variants:

```bash
# Refresh using remote downloads for missing data
./run_pipeline.sh --allow-remote-downloads --source ./d_us_txt.zip

# Include PPO training and backtest in the refresh
./run_pipeline.sh --with-rl --source ./d_us_txt.zip

# Rebuild everything but do not launch the dashboard
./run_pipeline.sh --no-dashboard --source ./d_us_txt.zip
```

If you only want the dashboard and the artifacts already exist, run:

```bash
.venv/bin/python -m streamlit run dashboard/app.py
```
 
---
 
## Research Foundation
 
This project is informed by the following academic literature:
 
- Ang, A. & Bekaert, G. (2002). *Regime Switches in Interest Rates*
- Hamilton, J.D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series*
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*
- Fama, E. & French, K. (2015). *A Five-Factor Asset Pricing Model*
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms*
- Gu, S., Kelly, B. & Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning*
---
