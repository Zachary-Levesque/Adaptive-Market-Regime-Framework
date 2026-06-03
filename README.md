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

1. **Builds** historical prices, returns, factors, macro data, technical features, and regime features
2. **Detects** market regimes with HMM probabilities, GMM validation, Bayesian smoothing, and Kalman filtering
3. **Trains and compares** regime-specific LSTM/Transformer ensembles against simpler baseline alpha models
4. **Selects** an alpha signal using walk-forward, transaction-cost, rebalance, forward-return horizon, and projected-backtest evidence
5. **Backtests** the selected signal against SPY, equal-weight, and momentum baselines
6. **Reports** alpha diagnostics, regime-conditional results, stress tests, and pre-RL readiness checks

Planned but not implemented:

1. PPO reinforcement-learning position sizing
2. Intraday execution through Alpaca
3. FastAPI/React dashboard and live daily recommendation workflow

The project should not move to RL until `python -m src.alpha.build_readiness` passes. The current local artifacts show a positive but weak alpha signal that underperforms simple benchmarks.

Target daily output once the planned live layer exists:
 
```
═══════════════════════════════════════════════════
  AMRF DAILY SIGNAL — 2026-05-19
═══════════════════════════════════════════════════
  REGIME: Bull Trending (81%) | Low-Vol (13%) | Bear (4%) | Crisis (2%)
 
  TRADE SIGNALS:
  ┌─────────┬──────────┬──────────┬─────────────┬──────────┐
  │ Ticker  │ Signal   │ Size     │ Conviction  │ Stop     │
  ├─────────┼──────────┼──────────┼─────────────┼──────────┤
  │ NVDA    │ LONG     │ $2,340   │ 82%         │ -5.2%    │
  │ TSM     │ LONG     │ $1,800   │ 74%         │ -4.8%    │
  │ AMD     │ LONG     │ $1,100   │ 67%         │ -5.5%    │
  │ MCHI    │ SHORT    │ $800     │ 71%         │ +4.1%    │
  │ IBIT    │ FLAT     │ $0       │ N/A         │ N/A      │
  └─────────┴──────────┴──────────┴─────────────┴──────────┘
 
  PORTFOLIO METRICS:
  Expected Sharpe: 1.84 | CVaR (95%): -2.3% | Max Position: 23.4%
═══════════════════════════════════════════════════
```
 
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
| Python 3.11 | Core language |
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
| FastAPI | Dashboard backend |
| React | Dashboard frontend |
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
Full-stack dashboard with a FastAPI backend and React/Tailwind frontend. Provides real-time visualization of regime states, equity curves, readiness checklists, and risk diagnostics.
 
---
 
## Results
 
Current local artifact snapshot from `data/results/performance_report.parquet`:

| Metric | AMRF Strategy | Buy & Hold SPY | Equal Weight | 63D Momentum |
|---|---|---|---|---|
| Annual Return | 2.28% | 19.81% | 19.22% | -3.44% |
| Sharpe Ratio | 0.25 | 1.32 | 1.10 | -0.19 |
| Max Drawdown | -25.18% | -20.49% | -37.16% | -37.87% |
| Calmar Ratio | 0.09 | 0.97 | 0.52 | -0.09 |
| Win Rate | 51.00% | 54.62% | 56.88% | 49.60% |
 
Alpha diagnostics currently score the selected signal against a 5-trading-day forward-return target. The latest selected signal is `defensive_regime_selector`: mean IC 0.0193, mean rank IC 0.0243, IC positive on 51.06% of scored days. The signal is active in regimes 1, 2, and 3, with positive rank IC in each active regime, but realized Sharpe and benchmark-relative performance remain below the readiness gate. This means the next project step should improve data coverage and risk-adjusted portfolio construction before adding reinforcement learning position sizing.

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
# Clone the repository
git clone https://github.com/Zachary-Levesque/AMRF.git
cd AMRF
 
# Install dependencies
pip install -r requirements.txt
 
# Run data pipeline
python -m src.data.build_phase1
 
# Train regime model
python -m src.regime.build_phase2
 
# Train regime-specific alpha ensemble
python -m src.alpha.build_phase3

# Compare alpha models against baselines and select signal
python -m src.alpha.build_model_comparison

# Run backtest
python -m src.risk.build_phase4

# Diagnose selected signal quality
python -m src.alpha.build_diagnostics

# Check whether the selected alpha is ready for RL work
python -m src.alpha.build_readiness
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
