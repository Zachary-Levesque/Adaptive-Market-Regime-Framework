# AMRF Project Summary

This document is your personal guide to understanding and explaining the Adaptive Market Regime Framework. Use it to prepare for interviews, resume bullets, demos, and future development.

---

## One-Sentence Explanation

AMRF is an end-to-end quantitative research platform that detects market regimes, selects regime-aware alpha signals, blends them with SPY into a risk-managed portfolio, and validates the strategy through backtesting, stress testing, readiness gates, and dashboards.

---

## Resume Version

Built a research-grade quantitative trading framework in Python and TypeScript that ingests historical market data, detects market regimes with HMM/GMM/Bayesian/Kalman methods, compares alpha models with walk-forward validation, constructs a regime-aware alpha/SPY portfolio, backtests against SPY/equal-weight/momentum baselines, and presents risk/readiness diagnostics in a polished React dashboard.

Suggested resume bullets:

- Built an end-to-end market-regime research pipeline using Python, pandas, scikit-learn, hmmlearn, PyTorch, FastAPI, React, TypeScript, and Streamlit.
- Implemented HMM-based market regime detection with GMM validation, Bayesian transition smoothing, and Kalman filtering.
- Designed walk-forward alpha model comparison with signal diagnostics including IC, rank IC, hit rate, turnover, and regime-conditional performance.
- Added a deterministic regime-aware portfolio layer that blends alpha exposure with SPY and improved saved backtest Sharpe from below-benchmark to `1.0866` versus SPY `0.9500`.
- Built transaction-cost-aware backtesting, stress testing, readiness gates, and a final completion report over parquet research artifacts.
- Designed and implemented a polished responsive React dashboard backed by FastAPI endpoints for regime state, allocation, performance, and readiness.

---

## The Core Problem

Most trading systems are regime-blind. They assume that one strategy can work across:

- bull trends,
- sideways low-volatility markets,
- bear markets,
- high-volatility crisis periods.

That assumption is usually wrong. Momentum, reversal, defensive, and risk-off behavior all work differently depending on the market state.

AMRF solves this by asking:

1. What regime is the market currently in?
2. Which alpha signal works best in that regime?
3. How much risk should the final portfolio allocate to the alpha sleeve versus SPY?
4. Is the result strong enough to beat a benchmark after costs?

---

## Final Product State

Current selected model:

```text
regime_portfolio_selector
```

Current final portfolio:

```text
selected alpha sleeve + regime-aware SPY blend
```

Current allocation rule:

| Regime | Alpha Sleeve | SPY Sleeve |
|---|---:|---:|
| 0 | 0% | 100% |
| 1 | 25% | 75% |
| 2 | 25% | 75% |
| 3 | 75% | 25% |

Current saved performance:

```text
Strategy Sharpe: 1.0866
SPY Sharpe:      0.9500
Strategy MDD:    -20.73%
SPY MDD:         -31.18%
Ready for RL:    True
Project complete: True
```

---

## How The System Was Built

### 1. Data Pipeline

The project starts by loading historical price data and building clean research inputs.

Main responsibilities:

- import local Stooq-style vendor files,
- optionally fetch missing data remotely,
- normalize prices and returns,
- compute technical features,
- load factor data,
- build macro/regime features,
- validate data quality and GFC coverage.

Important files:

```text
src/data/ingestion.py
src/data/import_price_files.py
src/data/features.py
src/data/factors.py
src/data/pipeline.py
src/data/build_phase1.py
src/data/validate_phase1_inputs.py
```

Important artifacts:

```text
data/processed/prices.parquet
data/processed/returns.parquet
data/processed/factors.parquet
data/processed/technical_features.parquet
data/processed/regime_features.parquet
data/processed/data_quality_report.parquet
```

Skill demonstrated:

```text
Financial data engineering, validation, reproducible artifact pipelines.
```

### 2. Regime Detection

The regime engine models markets as latent states instead of one static environment.

Models used:

- Hidden Markov Model for primary regime classification,
- Gaussian Mixture Model for clustering validation,
- Bayesian smoothing for transition probabilities,
- Kalman filtering for noisy state estimates.

Important files:

```text
src/regime/hmm.py
src/regime/gmm.py
src/regime/bayesian.py
src/regime/kalman.py
src/regime/pipeline.py
src/regime/build_phase2.py
```

Important artifacts:

```text
data/regimes/regime_labels.parquet
data/regimes/regime_probs.parquet
data/regimes/regime_summary.parquet
data/regimes/transition_matrix.parquet
data/regimes/gmm_validation.parquet
```

Skill demonstrated:

```text
Unsupervised market-state modeling and interpretable ML for non-stationary time series.
```

### 3. Alpha Signal Generation

The alpha layer generates and compares candidate signals.

Signal families include:

- technical trend,
- technical reversal,
- technical multi-horizon,
- volatility-adjusted momentum,
- volatility-adjusted reversal,
- ridge,
- elastic net,
- regime selector,
- defensive regime selector,
- risk-managed variants,
- regime portfolio selector.

Important files:

```text
src/alpha/baselines.py
src/alpha/model_comparison.py
src/alpha/walk_forward.py
src/alpha/diagnostics.py
src/alpha/readiness.py
src/alpha/build_model_comparison.py
src/alpha/build_diagnostics.py
src/alpha/build_readiness.py
```

Important artifacts:

```text
data/processed/alpha_signal_selection.parquet
data/processed/alpha_model_comparison.parquet
data/processed/alpha_model_comparison_summary.parquet
data/processed/alpha_diagnostics.parquet
data/processed/alpha_diagnostics_by_regime.parquet
data/processed/alpha_signals/
```

Skill demonstrated:

```text
Signal research, walk-forward validation, cross-sectional diagnostics, model selection.
```

### 4. Portfolio Construction

This is the final improvement that made the project complete.

The raw alpha sleeve had useful return but too much risk. The fix was a deterministic portfolio layer:

```text
final portfolio = alpha exposure * alpha sleeve + SPY exposure * SPY
```

The exposure changes by regime. This reduced drawdown and improved Sharpe versus SPY.

Important files:

```text
src/risk/backtester.py
src/risk/build_phase4.py
configs/config.yaml
```

Important artifacts:

```text
data/results/position_weights.parquet
data/results/alpha_sleeve_position_weights.parquet
data/results/allocation_policy.parquet
data/results/allocation_exposure.parquet
```

Skill demonstrated:

```text
Portfolio construction, risk control, benchmark blending, auditable strategy design.
```

### 5. Backtesting And Risk

The risk engine tests the final portfolio against benchmarks.

It reports:

- annual return,
- annual volatility,
- Sharpe,
- Sortino,
- Calmar,
- max drawdown,
- drawdown duration,
- win rate,
- profit factor,
- total return,
- stress-period performance,
- regime-conditional performance.

Important files:

```text
src/risk/backtester.py
src/risk/metrics.py
src/risk/stress_test.py
src/risk/monte_carlo.py
src/risk/build_phase4.py
```

Important artifacts:

```text
data/results/performance_report.parquet
data/results/backtest_results.parquet
data/results/regime_performance.parquet
data/results/stress_report.parquet
```

Skill demonstrated:

```text
Backtesting, risk analytics, transaction costs, benchmark evaluation.
```

### 6. Readiness Gate

The readiness gate decides whether the strategy is strong enough for downstream RL research.

It checks:

- selected model exists,
- active signal history is long enough,
- mean rank IC is positive,
- IC positive rate is above 50%,
- backtest Sharpe clears threshold,
- total return is positive,
- strategy beats SPY,
- strategy beats equal-weight,
- regime-level rank IC is positive,
- data covers stress periods,
- all stress scenarios overlap.

Important file:

```text
src/alpha/readiness.py
```

Important artifact:

```text
data/processed/alpha_readiness_report.parquet
```

Skill demonstrated:

```text
Research governance, quality gates, model acceptance criteria.
```

### 7. Completion Checker

The completion checker verifies that the whole project is truly done.

It checks:

- all required processed artifacts,
- all required regime artifacts,
- all required result artifacts,
- selected signal manifest,
- allocation artifacts,
- readiness status,
- benchmark Sharpe comparison,
- data coverage,
- RL artifacts,
- dashboard files.

Important file:

```text
src/completion.py
```

Important artifact:

```text
data/results/project_completion_report.parquet
```

Skill demonstrated:

```text
Production-style artifact validation and end-to-end project acceptance testing.
```

### 8. Dashboard Product

The polished dashboard is built with:

- React,
- TypeScript,
- Tailwind CSS,
- Recharts,
- lucide-react,
- FastAPI.

It shows:

- current regime,
- strategy Sharpe,
- max drawdown,
- readiness status,
- strategy versus SPY equity curve,
- current alpha/SPY allocation,
- allocation policy by regime,
- HMM regime probabilities,
- readiness checks.

Important files:

```text
src/dashboard/frontend/src/App.tsx
src/dashboard/frontend/src/index.css
src/dashboard/backend/main.py
```

Skill demonstrated:

```text
Full-stack product UI, API design, responsive dashboard design, data visualization.
```

---

## Why Each Design Choice Matters

### Why HMM?

HMMs are interpretable for market regimes. They provide:

- regime labels,
- regime probabilities,
- transition matrices,
- explicit latent state behavior.

That is better for this project than a black-box sequence model as the first regime layer.

### Why walk-forward validation?

Financial data is temporal. Random splits can leak future information. Walk-forward validation trains on the past and tests on the future, which better matches deployment reality.

### Why rank IC?

The alpha signal is cross-sectional. Rank IC measures whether the model ranks assets correctly, not just whether it predicts exact returns.

### Why blend with SPY?

The raw alpha sleeve had return but too much risk. SPY is a liquid benchmark stabilizer. Blending alpha with SPY by regime improved risk-adjusted performance while keeping the rule simple and explainable.

### Why keep RL downstream?

RL can overtrade and add complexity. It should only be used after the base deterministic portfolio is strong. In this project, the deterministic blend is the production baseline.

---

## How To Demo The Project

### 1. Start with the problem

Say:

```text
Most strategies are regime-blind. I built AMRF to detect market regimes and adapt portfolio exposure based on the regime.
```

### 2. Show the dashboard

Open:

```text
http://127.0.0.1:5173
```

Point out:

- current regime,
- Sharpe,
- drawdown,
- readiness,
- allocation,
- equity curve,
- regime probabilities.

### 3. Explain the final result

Say:

```text
The final portfolio uses a selected alpha sleeve, then blends it with SPY based on the detected regime. That raised saved backtest Sharpe to 1.0866 versus SPY at 0.9500 and reduced max drawdown to -20.73%.
```

### 4. Show the validation

Run:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m src.build_completion_report --config configs/config.yaml
```

Expected:

```text
98 passed
Project complete: True
```

---

## Commands You Should Know

Run everything:

```bash
./run_pipeline.sh --no-dashboard
```

Rebuild key final reports:

```bash
.venv/bin/python -m src.risk.build_phase4 --config configs/config.yaml
.venv/bin/python -m src.alpha.build_diagnostics --config configs/config.yaml
.venv/bin/python -m src.alpha.build_readiness --config configs/config.yaml
.venv/bin/python -m src.build_completion_report --config configs/config.yaml
```

Run tests:

```bash
.venv/bin/python -m pytest
```

Run React dashboard:

```bash
.venv/bin/python -m uvicorn src.dashboard.backend.main:app --host 127.0.0.1 --port 8000
cd src/dashboard/frontend
npm run dev -- --host 127.0.0.1
```

Run Streamlit dashboard:

```bash
.venv/bin/python -m streamlit run dashboard/app.py --server.headless true --server.port 8502
```

Inspect final metrics:

```bash
.venv/bin/python - <<'PY'
import pandas as pd

perf = pd.read_parquet("data/results/performance_report.parquet")
ready = pd.read_parquet("data/processed/alpha_readiness_report.parquet")
complete = pd.read_parquet("data/results/project_completion_report.parquet")
policy = pd.read_parquet("data/results/allocation_policy.parquet")

print(perf)
print(policy)
print("Ready for RL:", bool(ready["ready_for_rl"].all()))
print("Project complete:", bool(complete["complete"].all()))
PY
```

---

## Interview Questions And Strong Answers

### What is the main idea?

AMRF adapts portfolio exposure to market regimes instead of using one static strategy in every environment.

### What was the hardest technical problem?

The hardest part was not generating alpha; it was converting a noisy alpha signal into a portfolio that beat SPY on risk-adjusted terms. The final fix was a regime-aware allocation layer.

### Why did you not just use RL?

RL should not rescue a weak base strategy. I used readiness gates to ensure the deterministic portfolio was competitive first. RL is downstream research only if it improves after-cost results.

### How did you avoid lookahead bias?

The alpha layer uses walk-forward validation, the backtester applies prior-day weights, and the allocation layer is applied before the one-day shift so decisions are lagged before returns are realized.

### What proves the project is complete?

The completion checker verifies required artifacts, selected signal validity, readiness, benchmark Sharpe, data coverage, allocation artifacts, dashboards, and RL artifacts. The saved completion report currently passes.

### What would you improve next?

I would paper trade the final blended weights, compare live paper results against backtest expectations, and only then evaluate PPO against the deterministic blend after costs.

---

## Limitations To Be Honest About

- Backtest performance does not guarantee future returns.
- The final allocation map is deterministic and should be revalidated over time.
- Transaction costs and slippage are modeled, but live execution can differ.
- The RL layer currently remains optional research.
- This should not be connected to live trading without paper trading, controls, monitoring, and kill switches.

---

## Mental Model Of The Whole Project

Think of AMRF as four stacked decisions:

```text
1. What market regime are we in?
2. Which alpha signal should we trust in this regime?
3. How much of the portfolio should use that alpha signal?
4. Does the resulting portfolio beat a benchmark after costs and risk?
```

That is the project.

Everything else is infrastructure to make those four decisions reproducible, testable, and explainable.
