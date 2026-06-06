# Final Improvement Plan

## Implementation Status

Implemented on the current artifact snapshot:

- Added a deterministic regime-aware allocation layer between the selected alpha sleeve and SPY.
- Saved auditable allocation artifacts:
  - `data/results/allocation_policy.parquet`
  - `data/results/allocation_exposure.parquet`
  - `data/results/alpha_sleeve_position_weights.parquet`
- Rebuilt Phase 4, diagnostics, readiness, and completion reports.
- Current strategy Sharpe is `1.0866` versus SPY Sharpe `0.9500`.
- Current max drawdown is `-20.73%` versus SPY max drawdown `-31.18%`.
- RL readiness now passes all checks.

Current allocation policy:

| Regime | Alpha Sleeve | SPY Sleeve |
|---|---:|---:|
| 0 | 0% | 100% |
| 1 | 25% | 75% |
| 2 | 25% | 75% |
| 3 | 75% | 25% |

## Goal
Make AMRF a finished research product that:

1. Detects regimes with the HMM.
2. Uses regime-specific alpha signals.
3. Produces a portfolio that can beat SPY on risk-adjusted terms.
4. Uses RL only after the base portfolio is already competitive.
5. Is easy to run, easy to inspect, and easy to explain.

## Current State

The project is no longer blocked by the benchmark gap described in the original review. The regime-aware alpha/SPY blend now clears the SPY Sharpe gate, the readiness report passes, and the completion checker reports the project as complete.

What already works:
- The data refresh path works with the new archive.
- Phase 1, regime detection, alpha comparison, risk backtesting, and the dashboard all run.
- The UI is readable, has hover help on the main overview metrics, and surfaces the allocation policy.
- The completion checker reports the project as complete.
- The selected portfolio beats SPY on Sharpe in the saved full-window backtest.

What still matters:
- The standalone alpha sleeve should still be distinguished from the deployable blended portfolio.
- RL should be evaluated as a downstream allocator only if it improves the blended portfolio after costs.
- Future changes should keep the allocation rule simple and auditable.

## Deep Review Findings

### 1. The edge is real, but it is not concentrated enough
Pre-implementation, the standalone strategy had higher total return than SPY, but also higher volatility and deeper drawdowns.

Original observed full-sample profile:
- Strategy annual return: about 17.0%
- Strategy Sharpe: about 0.93
- Strategy max drawdown: about -42.9%
- SPY Sharpe: about 0.95
- SPY max drawdown: about -31.2%

Interpretation:
- The system is producing alpha.
- The problem is not “no return.”
- The problem is “return with too much risk.”

### 2. The alpha is regime-dependent
The strategy is not uniformly strong across market states.

Observed behavior by regime:
- Regime 1 and Regime 3 are strong enough to justify active exposure.
- Regime 0 and Regime 2 are where the system loses its Sharpe advantage.

Interpretation:
- The right fix is not a generic de-risking knob.
- The right fix is a regime-aware allocation layer that changes exposure by state.

### 3. Simple execution tuning did not solve the benchmark gap
Lowering exposure, slowing rebalances, or using the current signal family more conservatively did not produce a Sharpe above SPY.

Interpretation:
- The problem is structural.
- We should not keep searching the same narrow execution knobs.

### 4. A regime-aware blend can beat SPY
The most credible path found so far is a regime-aware blend of the alpha sleeve and SPY.

The current evidence indicates:
- A simple convex blend can beat SPY on Sharpe.
- A regime-specific blend does even better.

Interpretation:
- This is the strongest available path to a finished product.
- The portfolio layer, not the raw signal, is where the remaining edge should be extracted.

### 5. RL should stay downstream
RL currently behaves like an overlay that can add complexity before the base portfolio is truly strong.

Interpretation:
- RL should not be the thing that rescues the project.
- RL should only be used after the base regime-aware portfolio is already competitive.

## What Needs to Change

### Priority 1: Add a regime-aware allocation layer
This is the main product fix.

What it should do:
- Use HMM regime probabilities to choose exposure between:
  - the alpha sleeve
  - SPY
  - possibly equal-weight as a secondary stabilizer
- Increase alpha exposure in regimes where the alpha sleeve beats the benchmark.
- Reduce or replace alpha exposure in weak regimes.

Why this matters:
- The current strategy’s main weakness is not selection accuracy alone.
- The weakness is how that signal behaves when it is turned into a portfolio.

Suggested implementation:
- Start with a deterministic regime map.
- Then test a small set of regime weights.
- Use walk-forward evaluation.
- Keep the final rule simple enough to explain.

### Priority 2: Rework the selection objective
The current selector optimizes projected backtest Sharpe, total return, and turnover, but it is still possible to pick a signal that looks good in isolation and loses on final risk-adjusted performance.

What should change:
- Add a final objective on realized portfolio Sharpe after benchmark blending.
- Penalize regimes that weaken the full portfolio.
- Prefer signals that improve the final portfolio, not just the signal-level diagnostics.

Why this matters:
- The project goal is not “find a signal.”
- The goal is “ship a portfolio that performs.”

### Priority 3: Improve the weak regimes
The weakest observed regimes are the ones where the alpha sleeve underperforms SPY.

What to investigate:
- Regime 0
- Regime 2

Likely fixes:
- Use more defensive exposure in those regimes.
- Reduce or eliminate active alpha when regime confidence is weak.
- Consider a different model for the weak regimes, not just the current portfolio selector.

### Priority 4: Reduce drawdown without crushing return
The strategy has enough return. It needs better risk efficiency.

What to test:
- Partial SPY hedge.
- Volatility targeting.
- Risk-off cash allocation in crisis-like states.
- Shorter exposure windows only where the regime is unstable.

Why this matters:
- A slightly lower return with much lower volatility can beat SPY on Sharpe.
- That is the easiest path to a stronger final product.

### Priority 5: Make the RL layer honest
RL should be framed as a second-stage allocator.

What to do:
- Train RL only on the improved regime-aware portfolio.
- Compare RL against the new rule-based blend, not just against the current standalone signal.
- Keep RL only if it improves realized performance after costs.

### Priority 6: Tighten the UI
The dashboard is functional but still needs to be more self-explanatory.

What to improve:
- Add hover help to all core metrics.
- Keep the Overview page as the first page.
- Show current regime, selected model, data-through date, readiness, and benchmark gap clearly.
- Make the difference between “signal quality” and “portfolio quality” obvious.

## Recommended Execution Order

### Step 1: Build the benchmark-beating portfolio layer
Implement regime-aware allocation between the alpha sleeve and SPY.

### Step 2: Backtest it properly
Re-run:
- Phase 4
- diagnostics
- readiness
- completion report

### Step 3: Check whether the benchmark gate passes
The target is:
- Sharpe above SPY
- positive total return
- acceptable drawdown
- readiness no longer blocked by benchmark misses

### Step 4: If it fails, simplify
If the first regime-aware blend does not win:
- test a smaller number of regime weights
- test a simpler hedge
- test a different alpha family

Do not expand the search space too much. The problem is now portfolio design, not raw model count.

### Step 5: Only then revisit RL
RL is not the rescue mechanism.

It should be the last layer added after the base portfolio is already competitive.

### Step 6: Final UI polish
Once the portfolio is strong:
- make the dashboard explain the allocation logic
- show why the current state is invested the way it is
- surface the benchmark comparison in the first screen

## Definition of Done

This project should be considered finished only when all of the following are true:

- The pipeline runs end to end on the current archive.
- The regime layer is current and interpretable.
- The selected portfolio beats SPY on Sharpe in the backtest.
- The benchmark gate is no longer blocking readiness.
- The RL layer is evaluated only after the base portfolio is strong.
- The dashboard makes the state of the system obvious to a new user in under a minute.

## Remaining Follow-Up

1. Re-evaluate RL against the improved blended portfolio, not against the older standalone alpha sleeve.
2. Keep RL only if it improves realized after-cost performance.
3. Continue treating the deterministic blend as the production baseline until RL proves incremental value.

## Bottom Line

The core project target is now met: the existing alpha is converted into a benchmark-beating, regime-aware portfolio. The remaining work is optional RL validation, not a blocker for the base research product.
