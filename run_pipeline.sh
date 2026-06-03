#!/bin/bash
# AMRF End-to-End Pipeline Execution Script

set -e

echo "--- 1. Data Ingestion & Feature Engineering ---"
python -m src.data.recompute_features

echo "--- 2. Regime Detection ---"
python -m src.regime.build_phase2 --config configs/config.yaml

echo "--- 3. Alpha Ensemble Training (Fast Run) ---"
python -m src.alpha.build_phase3 --config configs/config.yaml --epochs-override 10 --skip-validation

echo "--- 4. Model Selection & Comparison ---"
python -m src.alpha.build_model_comparison --config configs/config.yaml --skip-ensemble --include-tree-baselines

echo "--- 5. RL Position Sizing Optimization ---"
python -m src.rl.training --config configs/config.yaml --timesteps 10000
python -m src.rl.build_phase5 --config configs/config.yaml --mode predict

echo "--- 6. Final Backtesting & Risk Reporting ---"
python -m src.risk.build_phase4 --config configs/config.yaml

echo "--- 7. Readiness Diagnostics ---"
python -m src.alpha.build_readiness --config configs/config.yaml

echo "--- 8. Daily Trade Ticket Generation (Intraday) ---"
# Note: Requires ALPACA_API_KEY environment variable
python -m src.execution.build_phase7 --config configs/config.yaml

echo "--- Framework Execution Complete ---"
echo "To start the dashboard:"
echo "1. Terminal A: uvicorn src.dashboard.backend.main:app --reload"
echo "2. Terminal B: cd src/dashboard/frontend && npm run dev"
