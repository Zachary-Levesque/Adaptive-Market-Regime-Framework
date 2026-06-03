"""RL Training script for AMRF Module 6."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.alpha.readiness import load_readiness_status
from src.config import load_config
from src.rl.environment import TradingEnv

def train_rl():
    parser = argparse.ArgumentParser(description="Train AMRF RL Agent.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--timesteps", type=int, default=100000, help="Number of timesteps to train.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Train even if the selected alpha readiness gate has not passed.",
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    ready, _ = load_readiness_status(config.alpha.diagnostics_path.with_name("alpha_readiness_report.parquet"))
    if not ready and not args.force:
        raise SystemExit(
            "Refusing to train RL: selected alpha is not ready. "
            "Run `python -m src.alpha.build_readiness --config "
            f"{args.config}` and only continue when it reports `Ready for RL: True`. "
            "Use --force only for explicit experiments."
        )
    
    # Load data artifacts
    processed_dir = Path(config.data.processed_dir)
    regime_dir = Path(config.regime.output_dir)
    
    signals = pd.read_parquet(config.alpha.signals_path)
    returns = pd.read_parquet(processed_dir / "returns.parquet")
    regime_probs = pd.read_parquet(regime_dir / "regime_probs.parquet")
    
    # Create environment
    def make_env():
        return TradingEnv(
            signals=signals,
            returns=returns,
            regime_probs=regime_probs,
            initial_capital=config.rl.initial_capital
        )
    
    env = DummyVecEnv([make_env])
    
    # Initialize PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=config.rl.learning_rate,
        n_steps=config.rl.n_steps,
        batch_size=config.rl.batch_size,
        n_epochs=config.rl.n_epochs,
        gamma=config.rl.gamma,
        device="cpu"
    )
    
    print(f"Starting training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps)
    
    # Save model
    model_path = Path("models/rl_ppo_agent")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_rl()
