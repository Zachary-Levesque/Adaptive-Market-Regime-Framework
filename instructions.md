# AMRF — Build Instructions

> This document is a complete, step-by-step technical guide for building the Adaptive Market Regime Framework from scratch. It is written to be used alongside an AI coding assistant. Each module is self-contained with clear objectives, inputs, outputs, and implementation guidance.

---

## How To Use This Document

Each module has:
- **Goal** — what you are building and why
- **Inputs** — what data or outputs from previous modules it needs
- **Outputs** — what it produces
- **Dependencies** — libraries to install
- **Step-by-step instructions** — exactly what to build
- **Prompts for AI** — copy-paste prompts to use with your AI coding assistant
- **Validation** — how to verify it works correctly

Work through modules in order. Do not skip ahead — each module feeds the next.

---

## Environment Setup

### Prerequisites
- Python 3.11+
- Git
- Node.js 18+ (for dashboard)
- Docker (optional but recommended)

### Initial Setup

```bash
# Create project directory
git clone https://github.com/Zachary-Levesque/AMRF.git
cd AMRF

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Create folder structure
mkdir -p data/raw data/processed data/regimes
mkdir -p src/data src/regime src/alpha src/rl src/risk src/execution src/dashboard
mkdir -p notebooks tests configs
```

### Install All Dependencies

Create `requirements.txt`:

```txt
# Data
yfinance==0.2.38
pandas==2.2.0
numpy==1.26.4
pandas-datareader==0.10.0
alpaca-trade-api==3.0.2
fredapi==0.5.1

# ML & Statistics
scikit-learn==1.4.0
hmmlearn==0.3.2
scipy==1.12.0
statsmodels==0.14.1
pytorch==2.2.0
torchvision==0.17.0

# Reinforcement Learning
stable-baselines3==2.2.1
gymnasium==0.29.1

# Risk & Finance
PyPortfolioOpt==1.5.5
empyrical==0.5.5

# Visualization
matplotlib==3.8.2
plotly==5.18.0
seaborn==0.13.2

# Dashboard
fastapi==0.109.0
uvicorn==0.27.0
pydantic==2.5.3

# Utilities
pyyaml==6.0.1
python-dotenv==1.0.0
loguru==0.7.2
tqdm==4.66.1
pytest==7.4.4
```

```bash
pip install -r requirements.txt
```

### Configuration File

Create `configs/config.yaml`:

```yaml
data:
  universe:
    - NVDA
    - TSM
    - AMD
    - CIEN
    - LITE
    - AAPL
    - MSFT
    - GOOGL
    - META
    - AMZN
    - JPM
    - GS
    - SPY
    - QQQ
    - TLT
    - GLD
    - VIX
  start_date: "2000-01-01"
  end_date: "2024-12-31"
  benchmark: "SPY"

regime:
  n_regimes: 4
  n_iter: 1000
  covariance_type: "full"
  regime_names:
    0: "Bull Trending"
    1: "Low-Vol Compression"
    2: "Bear Trending"
    3: "High-Vol Crisis"

alpha:
  lstm:
    hidden_size: 128
    num_layers: 2
    dropout: 0.2
    sequence_length: 60
    batch_size: 32
    epochs: 100
    learning_rate: 0.001
  walk_forward:
    train_window: 756    # 3 years
    test_window: 126     # 6 months
    step_size: 63        # 3 months

rl:
  total_timesteps: 1000000
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  initial_capital: 100000

risk:
  n_simulations: 10000
  confidence_level: 0.95
  stress_periods:
    gfc: ["2008-09-01", "2009-03-31"]
    covid: ["2020-02-19", "2020-03-23"]
    rate_hike: ["2022-01-03", "2022-12-31"]
```

---

## Module 1 — Data Pipeline

### Goal
Build a robust data ingestion and feature engineering pipeline that downloads historical price data, computes returns and technical features, and loads Fama-French factor data. This is the foundation everything else depends on.

### Inputs
- Config file (ticker universe, date range)

### Outputs
- `data/processed/prices.parquet` — cleaned daily prices
- `data/processed/returns.parquet` — daily log returns
- `data/processed/features.parquet` — engineered feature matrix
- `data/processed/factors.parquet` — Fama-French 5 factors

### Dependencies
```bash
pip install yfinance pandas-datareader fredapi
```

### Step-by-Step Instructions

**Step 1.1 — Price Ingestion (`src/data/ingestion.py`)**

Build a class `MarketDataIngester` with methods:
- `download_prices(tickers, start, end)` — downloads OHLCV data for all tickers using yfinance, handles missing data, forward-fills gaps up to 5 days, drops tickers with >10% missing data
- `compute_returns(prices)` — computes daily log returns, removes outliers beyond 5 standard deviations
- `save(data, path)` — saves to parquet format
- `load(path)` — loads from parquet

Handle errors gracefully — yfinance sometimes fails for individual tickers. Use try/except and log failures.

**Step 1.2 — Fama-French Factors (`src/data/factors.py`)**

Build a class `FactorLoader` with methods:
- `download_ff5()` — downloads Fama-French 5 factors (Mkt-RF, SMB, HML, RMW, CMA) plus momentum (UMD) from Ken French's data library via pandas-datareader
- `align_with_returns(factors, returns)` — aligns factor dates with return dates
- The 6 factors you need: Market excess return, Size (SMB), Value (HML), Profitability (RMW), Investment (CMA), Momentum (UMD)

**Step 1.3 — Feature Engineering (`src/data/features.py`)**

Build a class `FeatureEngineer` with methods:
- `compute_technical_features(prices)` — computes:
  - Returns: 1d, 5d, 21d, 63d log returns
  - Volatility: 21d and 63d rolling realized volatility
  - Momentum: 12-1 month momentum (skip last month)
  - Volume: normalized volume ratio (today vs 20d average)
  - Trend: price relative to 50d and 200d moving average
  - Mean reversion: distance from 20d Bollinger Band mean (z-score)
  - VIX level and VIX change (use ^VIX)
- `compute_market_features(returns, benchmark='SPY')` — computes:
  - Market return (SPY)
  - Market volatility (21d rolling)
  - Cross-sectional dispersion (std of all stock returns)
  - Correlation regime (average pairwise correlation rolling 63d)
- `normalize(features)` — cross-sectional z-score normalization, clip at ±3

**Step 1.4 — Market Feature Matrix for Regime Detection**

The HMM needs a matrix of market-level features (not stock-level). Build `compute_regime_features(prices, vix)` that returns a single time series matrix with columns:
- SPY daily return
- SPY 21d realized volatility
- VIX level
- VIX 5d change
- SPY 63d momentum
- Cross-sectional return dispersion
- Yield curve slope (10Y - 2Y Treasury, via FRED API)
- Credit spread (HY - IG, approximate via ETFs HYG and LQD)

This matrix is what the HMM ingests.

### AI Prompts

```
Prompt 1:
"I am building a quantitative finance data pipeline in Python. 
Build me a class called MarketDataIngester in src/data/ingestion.py that:
1. Downloads OHLCV price data for a list of tickers using yfinance
2. Handles missing data by forward-filling up to 5 days
3. Drops tickers with more than 10% missing data
4. Computes daily log returns
5. Removes return outliers beyond 5 standard deviations
6. Saves and loads data to/from parquet format
7. Uses loguru for logging
Include proper error handling and docstrings."

Prompt 2:
"Build me a FeatureEngineer class in src/data/features.py that computes 
the following features for a quantitative trading system:
- 1d, 5d, 21d, 63d log returns
- 21d and 63d rolling realized volatility
- 12-1 month momentum
- Volume ratio (today vs 20d average)
- Price relative to 50d and 200d moving average
- Bollinger Band z-score (distance from 20d mean)
- VIX level and 5d VIX change
- Cross-sectional return dispersion
- Rolling 63d average pairwise correlation
Apply cross-sectional z-score normalization clipped at ±3.
Input is a pandas DataFrame of prices. Output is a feature matrix."

Prompt 3:
"Build me a FactorLoader class in src/data/factors.py that downloads 
the Fama-French 5 factors plus momentum from Ken French's data library 
using pandas-datareader. The factors needed are: Mkt-RF, SMB, HML, RMW, 
CMA, and UMD. Align them with a given returns DataFrame by date. 
Handle the case where factor dates don't perfectly match return dates."
```

### Validation

```python
# Run this to validate Module 1
from src.data.ingestion import MarketDataIngester
from src.data.features import FeatureEngineer
from src.data.factors import FactorLoader

ingester = MarketDataIngester()
prices = ingester.download_prices(['SPY', 'NVDA', 'AMD'], '2020-01-01', '2024-01-01')
returns = ingester.compute_returns(prices)

engineer = FeatureEngineer()
features = engineer.compute_technical_features(prices)
regime_features = engineer.compute_regime_features(prices)

loader = FactorLoader()
factors = loader.download_ff5()

# Assert no NaN in regime features after dropping first 200 rows
assert regime_features.iloc[200:].isna().sum().sum() == 0
print("Module 1 validation passed!")
print(f"Regime feature shape: {regime_features.shape}")
print(f"Features: {list(regime_features.columns)}")
```

---

## Module 2 — Regime Detection Engine

### Goal
Build the core of AMRF — a Hidden Markov Model that identifies 4 distinct latent market regimes from the feature matrix built in Module 1. Validate with a Gaussian Mixture Model. Apply Bayesian smoothing to transition probabilities. Use a Kalman Filter for state estimation.

### Inputs
- `data/processed/regime_features.parquet` — market feature matrix from Module 1

### Outputs
- `data/regimes/regime_labels.parquet` — daily regime label (0-3) for entire history
- `data/regimes/regime_probs.parquet` — regime probability vector per day (4 columns)
- `src/regime/hmm_model.pkl` — trained HMM model
- Visualization: regime history plot showing all 4 regimes colored over SPY price

### Dependencies
```bash
pip install hmmlearn scikit-learn scipy
```

### Step-by-Step Instructions

**Step 2.1 — Hidden Markov Model (`src/regime/hmm.py`)**

Build a class `RegimeHMM` with methods:

- `__init__(n_regimes=4)` — initialize with GaussianHMM from hmmlearn
- `preprocess(features)` — standardize features using StandardScaler, handle any remaining NaN
- `fit(features)` — fit the HMM model. Important settings:
  - `covariance_type='full'`
  - `n_iter=1000`
  - `random_state=42`
  - Run 10 random restarts and keep the model with highest log-likelihood (HMM is sensitive to initialization)
- `predict_regimes(features)` — returns hard regime labels (0-3)
- `predict_proba(features)` — returns soft regime probabilities (n_samples × 4 matrix)
- `label_regimes(regime_labels, returns)` — automatically identify which regime number corresponds to which market condition by computing:
  - Mean return per regime
  - Mean volatility per regime
  - Then assign: highest return + low vol = Bull, lowest return + high vol = Crisis, etc.
- `save(path)` and `load(path)` — pickle the model

**Step 2.2 — Gaussian Mixture Model Validation (`src/regime/gmm.py`)**

Build a class `RegimeGMM` with methods:
- `fit(features)` — fit sklearn GaussianMixture with n_components=4
- `predict(features)` — returns cluster labels
- `compare_with_hmm(hmm_labels, gmm_labels)` — computes adjusted rand score to measure agreement. Target: >0.65 agreement
- `plot_clusters(features, labels)` — PCA to 2D and plot colored clusters

**Step 2.3 — Bayesian Smoothing (`src/regime/bayesian.py`)**

Build a class `BayesianRegimeSmoothing` with methods:
- `compute_transition_matrix(regime_labels)` — compute empirical transition probability matrix from historical labels
- `smooth_probabilities(raw_probs, transition_matrix)` — apply forward-backward algorithm smoothing to regime probabilities (reduces noisy regime switching)
- `compute_regime_duration(regime_labels)` — compute average duration of each regime in days

**Step 2.4 — Kalman Filter (`src/regime/kalman.py`)**

Build a class `KalmanRegimeFilter` with methods:
- `filter(observations)` — implement a simple linear Kalman filter for smoothing the regime probability time series
- Use a random walk state transition model
- The Kalman filter smooths out noisy day-to-day regime probability fluctuations

**Step 2.5 — Regime Visualization**

Build `src/regime/visualize.py` with:
- `plot_regime_history(prices, regime_labels, regime_probs)` — creates a 3-panel chart:
  - Panel 1: SPY price colored by regime (4 colors)
  - Panel 2: Regime probability stacked area chart over time
  - Panel 3: Regime distribution pie chart
- Clearly label the 2008 crisis, 2020 COVID crash, 2022 bear market

### AI Prompts

```
Prompt 1:
"I am building a market regime detection system for a quantitative 
trading framework in Python. Build me a class RegimeHMM in src/regime/hmm.py 
using the hmmlearn library that:
1. Fits a Gaussian HMM with 4 hidden states to a feature matrix
2. Uses full covariance type and 1000 iterations
3. Runs 10 random restarts and keeps the best model by log-likelihood
4. Predicts hard regime labels and soft regime probabilities
5. Automatically labels regimes by their return/volatility characteristics:
   - Bull Trending: highest mean return, low volatility
   - Low-Vol Compression: low return, lowest volatility
   - Bear Trending: negative return, high volatility
   - High-Vol Crisis: most negative return, highest volatility
6. Saves and loads the model using pickle
Include proper logging and error handling."

Prompt 2:
"Build me a Kalman Filter class in src/regime/kalman.py for smoothing 
a time series of regime probabilities. The filter should:
1. Use a random walk state transition model
2. Accept a (T x 4) matrix of raw regime probabilities as input
3. Return a smoothed (T x 4) matrix of regime probabilities
4. Ensure smoothed probabilities sum to 1 at each timestep
This is used to reduce noisy day-to-day regime switching in a 
Hidden Markov Model output."

Prompt 3:
"Build me a visualization function in src/regime/visualize.py that 
creates a 3-panel matplotlib figure showing:
1. SPY price history (2000-2024) with background colored by market regime 
   (4 regimes: Bull=green, Bear=red, Crisis=dark red, Low-Vol=yellow)
2. Stacked area chart of regime probabilities over time
3. Bar chart showing average return and volatility per regime
Mark these specific dates with vertical lines:
- Sep 15 2008 (Lehman collapse)
- Mar 23 2020 (COVID bottom)
- Jan 3 2022 (Rate hike cycle begins)
Use matplotlib with a dark theme."
```

### Validation

```python
from src.regime.hmm import RegimeHMM
from src.regime.gmm import RegimeGMM
import pandas as pd

regime_features = pd.read_parquet('data/processed/regime_features.parquet')

# Fit HMM
hmm = RegimeHMM(n_regimes=4)
hmm.fit(regime_features)
labels = hmm.predict_regimes(regime_features)
probs = hmm.predict_proba(regime_features)

# Validate 4 regimes exist
assert len(set(labels)) == 4, "Should have exactly 4 regimes"

# Validate probabilities sum to 1
assert abs(probs.sum(axis=1) - 1.0).max() < 1e-6

# Validate regime durations are reasonable (not flickering every day)
from src.regime.bayesian import BayesianRegimeSmoothing
smoother = BayesianRegimeSmoothing()
durations = smoother.compute_regime_duration(labels)
assert min(durations.values()) > 5, "Regimes should last more than 5 days on average"

print("Module 2 validation passed!")
print(f"Regime distribution: {pd.Series(labels).value_counts().to_dict()}")
print(f"Average regime duration (days): {durations}")
```

---

## Module 3 — Regime-Specific Alpha Models

### Goal
Train a separate LSTM + Transformer model for each of the 4 market regimes. Each model learns to forecast next-day returns using Fama-French factors and technical features, but only on data from its specific regime. This is the core insight — a model trained only on bull market data will be better at predicting bull market returns than a model trained on all data.

### Inputs
- `data/processed/features.parquet` — stock-level feature matrix
- `data/processed/factors.parquet` — Fama-French factors
- `data/regimes/regime_labels.parquet` — regime labels per day

### Outputs
- 4 trained LSTM models (one per regime) saved to `src/alpha/models/`
- `data/processed/alpha_signals.parquet` — daily alpha forecasts per stock
- Walk-forward backtest performance metrics per regime

### Dependencies
```bash
pip install torch torchvision
```

### Step-by-Step Instructions

**Step 3.1 — Dataset Builder (`src/alpha/dataset.py`)**

Build a class `RegimeDataset` (inherits `torch.utils.data.Dataset`):
- `__init__(features, returns, regime_labels, target_regime, sequence_length=60)` — filters data to only include days in target_regime, creates sliding windows of length 60 days
- `__getitem__(idx)` — returns (X, y) where X is a (60 × n_features) tensor and y is next-day return
- Handle the case where a regime has fewer than 200 samples — in this case, extend with synthetic augmentation (add small Gaussian noise to existing samples)

**Step 3.2 — LSTM Model (`src/alpha/lstm.py`)**

Build a class `RegimeLSTM(nn.Module)`:
- Architecture:
  - Input layer: n_features
  - 2-layer LSTM with hidden_size=128, dropout=0.2
  - Attention mechanism over LSTM hidden states (compute attention weights, weighted sum)
  - Fully connected: 128 → 64 → 1
  - Output: predicted next-day return (regression)
- Training:
  - Loss: Sharpe ratio loss (maximize Sharpe, not minimize MSE — this is critical for quant finance)
  - Sharpe loss = -mean(returns) / std(returns) where returns = predicted × actual_returns
  - Optimizer: Adam with lr=0.001, weight_decay=1e-5
  - Learning rate scheduler: ReduceLROnPlateau
  - Early stopping: patience=10 epochs

**Step 3.3 — Transformer Model (`src/alpha/transformer.py`)**

Build a class `RegimeTransformer(nn.Module)`:
- Architecture:
  - Input projection: n_features → 128
  - Positional encoding
  - 2 Transformer encoder layers (nhead=8, dim_feedforward=256, dropout=0.1)
  - Global average pooling over sequence
  - Fully connected: 128 → 64 → 1
- Use same Sharpe ratio loss as LSTM

**Step 3.4 — Ensemble (`src/alpha/ensemble.py`)**

Build a class `RegimeAlphaEnsemble`:
- Trains both LSTM and Transformer for each regime
- Final prediction = 0.6 × LSTM_pred + 0.4 × Transformer_pred
- Weights determined by validation Sharpe ratio

**Step 3.5 — Walk-Forward Cross-Validation (`src/alpha/walk_forward.py`)**

This is critical — without it your backtest is meaningless.

Build a class `WalkForwardValidator`:
- `__init__(train_window=756, test_window=126, step_size=63)` — 3yr train, 6mo test, step 3mo
- `generate_splits(dates)` — yields (train_idx, test_idx) pairs sliding forward through time
- `validate(model_class, features, returns, regime_labels)` — for each split:
  1. Train model on train_window days of in-regime data only
  2. Predict on test_window days
  3. Compute Sharpe, IC (Information Coefficient), and Rank IC
- Report: mean Sharpe across folds, mean IC, hit rate (% of days with correct direction)

### AI Prompts

```
Prompt 1:
"I am building a regime-specific return forecasting model for a quant 
trading system. Build me an LSTM model class RegimeLSTM in PyTorch with:
1. 2-layer LSTM with hidden_size=128, dropout=0.2
2. Attention mechanism over LSTM hidden states
3. Fully connected output layers: 128 → 64 → 1 (return forecast)
4. Custom Sharpe ratio loss function:
   loss = -mean(predicted_returns * actual_returns) / std(predicted_returns * actual_returns)
   where predicted_returns are the model outputs scaled to unit variance
5. Adam optimizer with lr=0.001 and weight_decay=1e-5
6. ReduceLROnPlateau scheduler and early stopping with patience=10
Include train() and predict() methods. Input is (batch, sequence=60, features) tensor."

Prompt 2:
"Build me a WalkForwardValidator class in src/alpha/walk_forward.py for 
validating a time series ML model without lookahead bias. It should:
1. Generate train/test splits with train_window=756 days, test_window=126 days, step=63 days
2. For each split, train the model only on training data
3. Predict on test data and compute:
   - Sharpe ratio of long-short portfolio (long top quintile, short bottom quintile)
   - Information Coefficient (Spearman rank correlation of predictions vs actual returns)
   - Hit rate (% of predictions with correct direction)
4. Return a DataFrame of performance metrics per fold
5. Plot the cumulative out-of-sample returns across all folds
This is for a regime-specific model so also filter train/test data by regime label."
```

### Validation

```python
# Quick validation - train on just one regime
from src.alpha.lstm import RegimeLSTM
from src.alpha.walk_forward import WalkForwardValidator
import pandas as pd

features = pd.read_parquet('data/processed/features.parquet')
returns = pd.read_parquet('data/processed/returns.parquet')
regime_labels = pd.read_parquet('data/regimes/regime_labels.parquet')

validator = WalkForwardValidator()
results = validator.validate(RegimeLSTM, features, returns, regime_labels, target_regime=0)

print(f"Bull regime - Mean IC: {results['ic'].mean():.4f}")
print(f"Bull regime - Mean Sharpe: {results['sharpe'].mean():.4f}")
print(f"Bull regime - Hit Rate: {results['hit_rate'].mean():.2%}")

# IC > 0.02 is considered meaningful in quant finance
assert results['ic'].mean() > 0.0, "IC should be positive"
print("Module 3 validation passed!")
```

---

## Module 4 — Reinforcement Learning Position Sizing Agent

### Goal
Train a PPO reinforcement learning agent to dynamically size portfolio positions. The agent learns optimal risk-taking behavior across all market regimes — being aggressive in bull markets, defensive in crises, and exploiting mean reversion in low-vol periods.

### Inputs
- `data/processed/features.parquet`
- `data/processed/alpha_signals.parquet` — from Module 3
- `data/regimes/regime_probs.parquet` — regime probabilities from Module 2

### Outputs
- Trained PPO agent saved to `src/rl/ppo_agent.zip`
- `data/processed/rl_positions.parquet` — daily position weights per stock
- Training curves: episode reward, portfolio Sharpe, max drawdown

### Dependencies
```bash
pip install stable-baselines3 gymnasium
```

### Step-by-Step Instructions

**Step 4.1 — Trading Environment (`src/rl/environment.py`)**

Build a class `TradingEnvironment(gymnasium.Env)`:

- **State space** (observation):
  - Current regime probabilities (4 values)
  - Alpha signals for each stock (n_stocks values)
  - Current portfolio weights (n_stocks values)
  - Current portfolio Sharpe ratio (rolling 63d)
  - Current drawdown from peak
  - VIX level normalized
  - Days since last regime change
  - Total: ~(4 + n_stocks×2 + 4) dimensional

- **Action space**:
  - Continuous: portfolio weight vector of size n_stocks
  - Constrained: weights sum to 1, each weight between -0.3 and 0.3 (allow shorting)
  - Applied via softmax normalization

- **Reward function** (this is the most important part):
  ```
  daily_return = portfolio_return(weights, actual_returns)
  sharpe_contribution = daily_return / rolling_vol
  drawdown_penalty = -2.0 × max(0, drawdown - 0.05)  # Penalize >5% drawdown
  turnover_penalty = -0.001 × sum(abs(weight_change))  # Transaction costs
  
  reward = sharpe_contribution + drawdown_penalty + turnover_penalty
  ```

- **Episode structure**:
  - Each episode: 252 trading days (1 year) sampled randomly from history
  - Reset: random start date, equal-weight portfolio
  - Done: episode ends after 252 steps or if drawdown > 25%

- **`step(action)`**: applies position weights, computes next-day return, updates state
- **`reset()`**: initializes new episode at random start date

**Step 4.2 — PPO Agent Training (`src/rl/agent.py`)**

Build a class `RegimeAwareAgent`:
- Use `stable_baselines3.PPO` with:
  - policy='MlpPolicy'
  - Custom policy network: [256, 256, 128] layers
  - learning_rate=0.0003
  - n_steps=2048
  - batch_size=64
  - n_epochs=10
  - gamma=0.99
  - clip_range=0.2
  - ent_coef=0.01 (encourages exploration)
  - total_timesteps=1,000,000

- Add callbacks:
  - `EvalCallback` — evaluates on held-out 2023-2024 data every 10,000 steps
  - `CheckpointCallback` — saves model every 50,000 steps
  - Custom callback to log portfolio Sharpe ratio

**Step 4.3 — Reward Function Tuning**

This is iterative. Train for 100,000 steps first. Check:
- Is the agent learning? (reward should increase over training)
- Is it too conservative? (increase Sharpe coefficient)
- Is it over-trading? (increase turnover penalty)
- Is it taking too much risk? (increase drawdown penalty)

Adjust reward coefficients and retrain. This is normal in RL.

### AI Prompts

```
Prompt 1:
"Build me a custom OpenAI Gymnasium trading environment called 
TradingEnvironment in src/rl/environment.py for training a reinforcement 
learning agent on portfolio allocation. The environment should:

State space:
- 4 market regime probabilities
- Alpha signals for 10 stocks (normalized forecasted returns)
- Current portfolio weights for 10 stocks
- Rolling 63-day Sharpe ratio
- Current drawdown from peak
- Normalized VIX level
Total: 28-dimensional observation vector

Action space:
- Continuous vector of 10 portfolio weights
- Constrained to sum to 1, each between -0.3 and 0.3
- Apply softmax to normalize actions

Reward function:
- Daily portfolio return scaled by rolling volatility (Sharpe contribution)
- Penalty of -2.0 × max(0, drawdown - 0.05) for drawdowns > 5%
- Penalty of -0.001 × sum(abs(weight_changes)) for turnover/transaction costs

Episodes: 252 trading steps (1 year), random start date from 2000-2022.
Done condition: 252 steps elapsed OR drawdown > 25%.

Input data: pandas DataFrames of returns, alpha signals, regime probs.
Use numpy for fast computation inside step()."

Prompt 2:
"Build a RegimeAwareAgent class in src/rl/agent.py that trains a PPO 
reinforcement learning agent using stable-baselines3 on the TradingEnvironment. 
Requirements:
1. PPO with MlpPolicy, 3-layer network [256, 256, 128]
2. Hyperparameters: lr=0.0003, n_steps=2048, batch_size=64, gamma=0.99
3. EvalCallback on held-out 2023-2024 data, evaluating every 10,000 steps
4. Custom callback that logs portfolio Sharpe ratio and max drawdown to tensorboard
5. Train for 1,000,000 total timesteps
6. Save the trained model and plot training curves (reward vs timesteps)
7. predict(observation) method that returns portfolio weights for a given state"
```

### Validation

```python
from src.rl.environment import TradingEnvironment
from src.rl.agent import RegimeAwareAgent
import pandas as pd

# Quick environment test
returns = pd.read_parquet('data/processed/returns.parquet')
alpha = pd.read_parquet('data/processed/alpha_signals.parquet')
regime_probs = pd.read_parquet('data/regimes/regime_probs.parquet')

env = TradingEnvironment(returns, alpha, regime_probs)
obs, _ = env.reset()

# Validate observation shape
assert obs.shape == (28,), f"Expected (28,) got {obs.shape}"

# Run random episode
done = False
total_reward = 0
while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward

print(f"Random agent episode reward: {total_reward:.4f}")
print("Module 4 environment validation passed!")
```

---

## Module 5 — Risk Engine & Backtester

### Goal
Build a comprehensive risk engine and backtesting framework. This ties everything together — using regime detection, alpha models, and RL agent to run a full 25-year backtest with proper risk metrics.

### Inputs
- All outputs from Modules 1-4

### Outputs
- Full backtest results DataFrame
- Performance report PDF
- Comparison vs benchmarks (SPY buy-and-hold, momentum)
- `data/results/backtest_results.parquet`

### Step-by-Step Instructions

**Step 5.1 — Monte Carlo Risk Engine (`src/risk/monte_carlo.py`)**

Build a class `MonteCarloRiskEngine`:
- `simulate_returns(mean_returns, cov_matrix, n_simulations=10000, horizon=252)` — generates 10,000 portfolio return paths using multivariate normal distribution
- `compute_var(simulated_returns, confidence=0.95)` — Value at Risk at 95% confidence
- `compute_cvar(simulated_returns, confidence=0.95)` — Conditional VaR (Expected Shortfall) — average loss beyond VaR
- `compute_parametric_var(weights, mean_returns, cov_matrix)` — parametric VaR using normal distribution assumption
- `plot_return_distribution(simulated_returns, var, cvar)` — histogram with VaR/CVaR marked

**Step 5.2 — Historical Stress Tests (`src/risk/stress_test.py`)**

Build a class `StressTester`:
- `run_stress_test(strategy_returns, stress_periods)` — applies historical factor shocks from GFC, COVID, 2022
- `compute_drawdown_profile(returns)` — full drawdown time series, max drawdown, drawdown duration
- `scenario_analysis(weights, factor_shocks)` — estimate portfolio loss under custom factor shock scenarios

**Step 5.3 — Performance Metrics (`src/risk/metrics.py`)**

Build a class `PerformanceMetrics`:
- `sharpe_ratio(returns, rf=0.04)` — annualized Sharpe
- `sortino_ratio(returns, rf=0.04)` — like Sharpe but only penalizes downside vol
- `calmar_ratio(returns)` — annualized return / max drawdown
- `max_drawdown(returns)` — maximum peak-to-trough decline
- `win_rate(returns)` — % of positive days
- `profit_factor(returns)` — total gains / total losses
- `regime_conditional_performance(returns, regime_labels)` — all metrics broken down by regime
- `generate_report(returns, benchmark_returns, regime_labels)` — full performance report

**Step 5.4 — Full Backtester (`src/risk/backtester.py`)**

Build a class `AMRFBacktester`:
- Runs the complete AMRF pipeline in walk-forward mode:
  1. At each step: detect current regime
  2. Get alpha signals from regime-specific model
  3. Get position weights from RL agent
  4. Apply transaction cost model (0.1% per trade)
  5. Record daily portfolio return
- `run(start='2000-01-01', end='2024-12-31')` — full backtest
- `compare_benchmarks()` — compare vs SPY, equal-weight, momentum strategy
- `plot_results()` — equity curve, drawdown chart, rolling Sharpe, regime breakdown

### AI Prompts

```
Prompt 1:
"Build a MonteCarloRiskEngine class in src/risk/monte_carlo.py for 
portfolio risk analysis. It should:
1. Simulate 10,000 portfolio return paths using multivariate normal distribution
   given mean returns vector and covariance matrix
2. Compute Value at Risk (VaR) at 95% and 99% confidence levels
3. Compute Conditional VaR / Expected Shortfall (average loss beyond VaR)
4. Compute parametric VaR using the delta-normal method
5. Plot the return distribution as a histogram with VaR and CVaR marked
6. Return all results as a clean dictionary with proper annualization
Use numpy for simulation performance."

Prompt 2:
"Build a comprehensive PerformanceMetrics class in src/risk/metrics.py 
for evaluating a quantitative trading strategy. Include:
1. Sharpe ratio (annualized, with configurable risk-free rate)
2. Sortino ratio (downside deviation only)
3. Calmar ratio (return / max drawdown)
4. Maximum drawdown and drawdown duration
5. Win rate and profit factor
6. Monthly and annual return breakdown
7. Regime-conditional performance (all metrics broken down by regime label)
8. A generate_full_report() method that prints a formatted performance 
   summary comparing the strategy vs a benchmark
Input: pandas Series of daily returns, benchmark returns, regime labels."
```

---

## Module 6 — Intraday Execution Layer

### Goal
Add intraday timing to the daily signals. Instead of blindly buying at market open, AMRF waits for intraday confirmation before entering. This dramatically improves entry prices and reduces false signals.

### Dependencies
```bash
pip install alpaca-trade-api
```

### Step-by-Step Instructions

**Step 6.1 — Alpaca Data Feed (`src/execution/alpaca.py`)**

Build a class `AlpacaDataFeed`:
- Connect to Alpaca paper trading API (free account at alpaca.markets)
- `get_intraday_bars(ticker, timeframe='5Min', days=5)` — fetches 5-minute OHLCV bars
- `get_latest_quote(ticker)` — real-time bid/ask
- Note: Store API keys in `.env` file, never in code

**Step 6.2 — Intraday Signals (`src/execution/intraday.py`)**

Build a class `IntradaySignalGenerator`:
- `compute_vwap(bars)` — Volume Weighted Average Price from intraday bars
- `vwap_signal(price, vwap, threshold=0.005)` — long signal when price crosses above VWAP + threshold
- `volume_confirmation(bars, window=10)` — True if current volume > 1.5× average volume
- `momentum_confirmation(bars, window=5)` — True if 5-bar momentum is positive
- `generate_entry_signal(ticker, daily_signal)` — combines all intraday confirmations:
  - Daily signal must be LONG
  - Price must be above VWAP
  - Volume must confirm
  - Returns: (entry_price, stop_loss, take_profit)
- `compute_stop_loss(entry_price, atr, multiplier=2.0)` — ATR-based stop loss
- `compute_take_profit(entry_price, risk, reward_ratio=2.5)` — 2.5:1 reward/risk

### AI Prompts

```
Prompt 1:
"Build an IntradaySignalGenerator class in src/execution/intraday.py 
that generates intraday trade entry signals. Given a pandas DataFrame 
of 5-minute OHLCV bars:
1. Compute VWAP (Volume Weighted Average Price)
2. Generate a long entry signal when: price crosses above VWAP AND 
   volume is 1.5x the 10-bar average AND 5-bar momentum is positive
3. Compute ATR-based stop loss: entry_price - 2.0 × ATR(14)
4. Compute take-profit at 2.5:1 reward/risk ratio
5. Return entry price, stop loss, take profit, and signal confidence score
This is for a day trading system — signals are generated at 9:45 AM 
and positions are closed by 3:45 PM."
```

---

## Module 7 — Dashboard

### Goal
Build an interactive dashboard that displays regime state, trade signals, portfolio performance, and risk metrics in real time.

### Stack
- Backend: FastAPI (Python)
- Frontend: React + Recharts
- Refresh: every 5 minutes during market hours

### Step-by-Step Instructions

**Step 7.1 — FastAPI Backend (`src/dashboard/backend/main.py`)**

Build endpoints:
- `GET /api/regime/current` — current regime probabilities and label
- `GET /api/signals/today` — today's trade signals with conviction scores
- `GET /api/portfolio/performance` — equity curve, Sharpe, drawdown
- `GET /api/risk/metrics` — VaR, CVaR, current drawdown
- `GET /api/regime/history` — full regime history for chart
- `GET /api/backtest/results` — full backtest comparison

**Step 7.2 — React Frontend**

Build components:
- `RegimeGauge` — circular gauge showing current regime with probability
- `SignalTable` — sortable table of today's signals with conviction bars
- `EquityCurve` — Recharts line chart with regime background coloring
- `RiskMetrics` — cards showing Sharpe, CVaR, max drawdown
- `RegimeHistory` — stacked area chart of regime probabilities over time

### AI Prompts

```
Prompt 1:
"Build a FastAPI backend in src/dashboard/backend/main.py for a 
quantitative trading dashboard. It should serve these endpoints:
- GET /api/regime/current: returns current regime name, probability vector, 
  and regime duration in days
- GET /api/signals/today: returns list of {ticker, signal, size, conviction, 
  stop_loss, take_profit} for today
- GET /api/portfolio/performance: returns equity curve as list of 
  {date, portfolio_value, benchmark_value, drawdown}
- GET /api/risk/metrics: returns {sharpe, sortino, calmar, max_drawdown, 
  var_95, cvar_95, win_rate}
Load all data from parquet files in data/results/. Use pydantic models 
for all response types. Include CORS middleware."

Prompt 2:
"Build a React dashboard component for a quantitative trading system.
Create these components using Recharts:
1. RegimeGauge: a PieChart showing 4 regime probabilities with colors
   (green=Bull, yellow=Low-Vol, orange=Bear, red=Crisis)
2. SignalTable: a table with columns [Ticker, Signal, Size, Conviction%,
   Stop Loss, Take Profit] with color coding (green=LONG, red=SHORT, gray=FLAT)
3. EquityCurve: a LineChart with two lines (AMRF vs SPY) and background
   coloring based on current market regime
4. RiskMetrics: 4 metric cards showing Sharpe, CVaR, Max Drawdown, Win Rate
Fetch data from FastAPI endpoints at localhost:8000/api/..."
```

---

## Final Integration

### Putting It All Together

Create `src/main.py` — the daily runner:

```python
"""
AMRF Daily Runner
Run every morning at 8:30 AM to generate today's trade signals.
"""

import yaml
from src.data.ingestion import MarketDataIngester
from src.data.features import FeatureEngineer
from src.regime.hmm import RegimeHMM
from src.alpha.ensemble import RegimeAlphaEnsemble
from src.rl.agent import RegimeAwareAgent
from src.risk.monte_carlo import MonteCarloRiskEngine
from src.execution.intraday import IntradaySignalGenerator

def run_daily():
    # Load config
    with open('configs/config.yaml') as f:
        config = yaml.safe_load(f)
    
    # 1. Update data
    ingester = MarketDataIngester()
    prices = ingester.download_prices(config['data']['universe'])
    
    # 2. Engineer features
    engineer = FeatureEngineer()
    features = engineer.compute_regime_features(prices)
    
    # 3. Detect current regime
    hmm = RegimeHMM.load('src/regime/hmm_model.pkl')
    regime = hmm.predict_regimes(features)[-1]
    regime_probs = hmm.predict_proba(features)[-1]
    
    # 4. Generate alpha signals
    alpha_model = RegimeAlphaEnsemble.load('src/alpha/models/')
    signals = alpha_model.predict(features, regime)
    
    # 5. Size positions with RL agent
    agent = RegimeAwareAgent.load('src/rl/ppo_agent.zip')
    weights = agent.predict(regime_probs, signals)
    
    # 6. Compute risk metrics
    risk = MonteCarloRiskEngine()
    var, cvar = risk.compute_var(weights, features)
    
    # 7. Get intraday entry timing
    intraday = IntradaySignalGenerator()
    entries = intraday.generate_entry_signal(signals, weights)
    
    # 8. Print report
    print_daily_report(regime, regime_probs, entries, var, cvar)

if __name__ == "__main__":
    run_daily()
```

### Docker Setup

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  amrf-backend:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./src:/app/src
    command: uvicorn src.dashboard.backend.main:app --host 0.0.0.0 --port 8000

  amrf-frontend:
    build: ./src/dashboard/frontend
    ports:
      - "5173:5173"
    depends_on:
      - amrf-backend
```

---

## Build Order Summary

Build in this exact order — each depends on the previous:

| Step | Module | Est. Time | Key Output |
|---|---|---|---|
| 1 | Data Pipeline | 1-2 days | Feature matrix, Fama-French factors |
| 2 | Regime Detection | 2-3 days | Regime labels + probabilities |
| 3 | Alpha Models | 3-5 days | Return forecasts per regime |
| 4 | RL Agent | 3-4 days | Position sizing weights |
| 5 | Risk Engine | 2-3 days | Backtest results + metrics |
| 6 | Intraday Layer | 1-2 days | Entry/exit timing signals |
| 7 | Dashboard | 2-3 days | Interactive web UI |
| **Total** | | **~2-3 weeks** | **Full AMRF system** |

---

## Tips For Working With AI

1. **Always give context first** — start every prompt with "I am building AMRF, a regime-aware quant trading system" so the AI understands the project
2. **One module at a time** — don't ask the AI to build multiple modules in one prompt
3. **Paste error messages** — when something breaks, paste the full error and ask "how do I fix this in the context of AMRF"
4. **Ask for tests** — after every module ask "write pytest tests for this module"
5. **Validate as you go** — run the validation script at the end of each module before moving on
6. **Commit often** — git commit after every working module so you can roll back

---

## Common Issues & Solutions

| Issue | Solution |
|---|---|
| HMM converges to fewer than 4 regimes | Increase random restarts to 20, try different initialization |
| LSTM loss is NaN | Reduce learning rate to 0.0001, add gradient clipping |
| RL agent doesn't learn | Check reward scale (should be ~[-1, 1]), reduce action space |
| yfinance rate limit | Add time.sleep(0.5) between downloads |
| Backtest Sharpe too high | Check for lookahead bias in feature computation |

---

## Resources

- [hmmlearn documentation](https://hmmlearn.readthedocs.io)
- [stable-baselines3 documentation](https://stable-baselines3.readthedocs.io)
- [Advances in Financial Machine Learning — Lopez de Prado](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
- [Alpaca Paper Trading API](https://alpaca.markets/docs)
- [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
- [PyTorch Documentation](https://pytorch.org/docs)
