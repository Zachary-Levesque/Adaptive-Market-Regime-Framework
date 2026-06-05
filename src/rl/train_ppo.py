"""Train the PPO position sizing agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.rl.agent import PPOPositionSizingAgent
from src.rl.data import load_rl_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AMRF PPO position sizing agent.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset = load_rl_dataset(config)
    agent = PPOPositionSizingAgent(config, dataset)
    model, history = agent.train(total_timesteps=args.timesteps)
    print(f"Saved PPO model to {config.rl.model_path}")
    print(f"Training history rows: {len(history)}")
    if not history.empty and "validation_sharpe" in history.columns:
        best_row = history.loc[history["validation_sharpe"].idxmax()]
        print(
            "Best validation Sharpe: "
            f"{float(best_row['validation_sharpe']):.6f} at step {int(best_row['step'])}"
        )


if __name__ == "__main__":
    main()
