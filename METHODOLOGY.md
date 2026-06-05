# AMRF Methodology

AMRF is built around one idea: market behavior is not stationary, so a single static trading rule should not be expected to work well across all periods. The project addresses that by separating regime detection, alpha generation, portfolio construction, risk control, and reinforcement learning into distinct modules.

## Why HMM for regime detection

The regime engine uses a Hidden Markov Model instead of an LSTM or other sequence learner for the first-stage regime label. That choice is deliberate.

An HMM gives interpretable latent states and an explicit transition matrix. Those two properties matter in a regime system:

- You can inspect state persistence and switching behavior directly.
- The regime probabilities are usable as structured inputs to later modules.
- The model is unsupervised, so it does not require labels that are themselves derived from the target strategy.

An LSTM can fit sequence patterns, but it is harder to interpret and easier to turn into a black box. For a research framework that needs auditability and downstream gating, the HMM is the better first layer.

## Why walk-forward validation

The alpha selection pipeline uses walk-forward validation to mimic how the strategy would actually be deployed.

That avoids the main failure mode of financial ML: lookahead bias hidden inside random cross-validation. A random split can leak regime structure, volatility regimes, and macro context across train and test. Walk-forward keeps the temporal order intact, so each fold only trains on past data and evaluates on future data.

This is particularly important for AMRF because:

- regime structure changes over time,
- turnover and transaction costs matter,
- the selected signal is later used as the base policy for RL and execution.

## Why inverse-volatility weighting

The selected signal uses inverse-volatility weighting because it is a simple risk-parity style construction that reduces concentration in high-volatility assets.

The intuition is practical:

- stable assets get larger weights,
- noisy assets get smaller weights,
- the portfolio tends to behave more consistently across regimes.

That does not guarantee alpha, but it is a strong default for a regime-aware allocation layer because it is easy to explain and tends to reduce avoidable drawdowns.

## Why PPO for position sizing

PPO is used for the RL layer because the action is continuous: the agent is not choosing from a small discrete menu, it is scaling a full portfolio of weights.

That makes PPO a better fit than a discrete-action policy for three reasons:

- it can express gradual tilts instead of hard switches,
- it can be trained with a custom reward that mixes return, drawdown, and transaction cost,
- it is robust enough for a non-stationary financial environment when the observation space is normalized and the policy is anchored to a known-good baseline.

In this project, PPO is not replacing the alpha model. It is learning how aggressively to size the selected signal.

## Risk and execution design

The execution layer models transaction costs and slippage explicitly. That is important because a strategy that looks good gross can fail net once turnover is charged.

The project also keeps the static signal and RL agent comparable by recording both policy-level and execution-aware performance. That distinction matters:

- policy-level results show what the model learned,
- execution-aware results show what would plausibly happen in practice.

## Honest limitation

The current PPO agent does not outperform the static signal on the 2022-2024 test slice after execution costs. It learns a positive policy, but realized test performance is still reduced by turnover and execution drag.

The most likely reason is distribution shift: the test period contains a regime mix and cost structure that differs from the training distribution, and the RL policy has not yet learned enough incremental edge over the static signal to justify those extra trades.

That is not a failure of the framework. It is a useful boundary condition. The static signal remains the production baseline, and the RL layer should be treated as an experimental sizing overlay until it consistently improves the execution-aware result.

