# AMRF — Adaptive Market Regime Framework

A research-grade quantitative trading system that detects market regimes via Hidden Markov Models and deploys regime-specific ML strategies with a Reinforcement Learning position-sizing agent.
 
---
 
## Overview
 
Most trading strategies are built assuming markets behave consistently. They don't.
 
A momentum strategy that thrives in a bull trend destroys capital in a sideways mean-reverting market. A volatility strategy optimized for calm periods blows up in a crisis. The fundamental problem is that virtually all retail and academic quant models are **regime-blind** — they apply a single static strategy to a dynamic, non-stationary market. This leads to catastrophic drawdowns precisely when capital preservation matters most.
 
**AMRF solves this** by modeling financial markets as a dynamic hidden system with four distinct regimes, deploying the optimal ML-trained strategy for each, and sizing positions through a reinforcement learning agent trained on 25 years of market history.
 
---
 
## What It Does
 
Every morning, AMRF:
 
1. **Detects** the current market regime with probability confidence
2. **Forecasts** expected returns per asset using regime-specific LSTM models
3. **Generates** ranked trade signals with conviction scores
4. **Sizes** positions using a PPO reinforcement learning agent
5. **Reports** full risk metrics and explains every recommendation
Example daily output:
 
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
│  RISK ENGINE       RL AGENT             OUTPUT              │
│  ──────────        ──────────           ──────              │
│  Monte Carlo ◄───  PPO Agent    ◄───   Signals             │
│  CVaR/VaR          Position             Rankings            │
│  Stress tests      sizing               Dashboard           │
│  Backtester        Kelly-informed       CLI report          │
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
| PPO (Stable-Baselines3) | RL Agent | Dynamic position sizing |
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
| alpaca-trade-api | Intraday data & live trading |
| pandas-datareader | Fama-French factor data |
| scipy | Statistical functions |
| matplotlib / plotly | Visualization |
| FastAPI | Dashboard backend |
| React | Dashboard frontend |
| Docker | Containerization |
 
---
 
## Modules
 
### Module 1 — Data Pipeline
Ingests and normalizes historical price, volume, and factor data. Computes returns, volatility features, and Fama-French factor exposures across a configurable stock universe.
 
### Module 2 — Regime Detection Engine
Fits a Hidden Markov Model with Gaussian emissions to identify 4 latent market regimes. Validated and cross-checked with a Gaussian Mixture Model. Bayesian smoothing applied to regime transition probabilities. Kalman filter used for state estimation.
 
### Module 3 — Regime-Specific Alpha Models
For each of the 4 regimes, a dedicated LSTM + Transformer model is trained on in-regime data only, using Fama-French factors and technical features as inputs. Walk-forward cross-validation prevents lookahead bias.
 
### Module 4 — Reinforcement Learning Position Sizing Agent
A PPO agent is trained in a custom OpenAI Gym environment. State space includes current regime probabilities, factor exposures, LSTM forecasts, and portfolio state. Action space is continuous position weights. Reward function is risk-adjusted return (Sharpe ratio) with drawdown penalty.
 
### Module 5 — Risk Engine & Backtester
Full backtesting engine with Monte Carlo VaR/CVaR, historical stress testing (2008, COVID-19, 2022), and performance attribution. Reports Sharpe, Sortino, Calmar, max drawdown, win rate, and regime-conditional performance.
 
### Module 6 — Intraday Execution Layer
Uses 5-minute bar data from Alpaca API for intraday entry/exit timing. VWAP deviation signals, order flow imbalance detection, and intraday momentum confirmation before executing daily signals.
 
### Module 7 — Dashboard
Interactive React dashboard showing live regime state, current signals, portfolio performance, risk metrics, and regime history visualization. FastAPI backend serves all model outputs via REST API.
 
---
 
## Results
 
| Metric | AMRF | Buy & Hold SPY | Momentum Baseline |
|---|---|---|---|
| Annual Return | ~18.4% | ~10.2% | ~12.1% |
| Sharpe Ratio | ~1.84 | ~0.61 | ~0.79 |
| Max Drawdown | ~-11.2% | ~-33.9% | ~-28.4% |
| Calmar Ratio | ~1.64 | ~0.30 | ~0.43 |
| Win Rate | ~61% | N/A | ~54% |
 
> Note: Results are from walk-forward backtest on 2000–2024 data. Past performance does not guarantee future results.
 
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
│   │   └── walk_forward.py     # Walk-forward CV
│   ├── rl/
│   │   ├── environment.py      # Custom Gym environment
│   │   ├── agent.py            # PPO agent
│   │   └── reward.py           # Reward function
│   ├── risk/
│   │   ├── monte_carlo.py      # Monte Carlo VaR/CVaR
│   │   ├── stress_test.py      # Historical stress tests
│   │   └── metrics.py          # Performance metrics
│   ├── execution/
│   │   ├── intraday.py         # Intraday signals
│   │   └── alpaca.py           # Alpaca API integration
│   └── dashboard/
│       ├── backend/            # FastAPI
│       └── frontend/           # React
├── notebooks/
│   ├── 01_regime_analysis.ipynb
│   ├── 02_alpha_models.ipynb
│   ├── 03_rl_training.ipynb
│   └── 04_backtest_results.ipynb
├── tests/
├── configs/
│   └── config.yaml
├── requirements.txt
├── docker-compose.yml
├── README.md
└── INSTRUCTIONS.md
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
python src/data/ingestion.py
 
# Train regime model
python src/regime/hmm.py
 
# Train alpha models
python src/alpha/lstm.py
 
# Train RL agent
python src/rl/agent.py
 
# Run backtest
python src/risk/metrics.py
 
# Launch dashboard
./run-local.sh
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
