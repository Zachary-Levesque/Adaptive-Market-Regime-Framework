"""Backward-compatible wrapper for the RL PPO pipeline."""

from __future__ import annotations

from src.rl.backtest_rl import main as backtest_main
from src.rl.train_ppo import main as train_main


def run_rl_pipeline():
    train_main()
    backtest_main()


if __name__ == "__main__":
    run_rl_pipeline()
