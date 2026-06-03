"""CLI entrypoint for Module 6 — RL Position Sizing."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from stable_baselines3 import PPO

from src.alpha.readiness import load_readiness_status
from src.config import load_config
from src.rl.environment import TradingEnv

def run_rl_pipeline():
    parser = argparse.ArgumentParser(description="Run AMRF RL Position Sizing Pipeline.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--mode", choices=["train", "predict"], default="predict", help="Train a new agent or generate signals.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if the selected alpha readiness gate has not passed.",
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    ready, _ = load_readiness_status(config.alpha.diagnostics_path.with_name("alpha_readiness_report.parquet"))
    if not ready and not args.force:
        raise SystemExit(
            "Refusing to run RL position sizing: selected alpha is not ready. "
            "Run `python -m src.alpha.build_readiness --config "
            f"{args.config}` and only continue when it reports `Ready for RL: True`. "
            "Use --force only for explicit experiments."
        )
    
    # Load data
    processed_dir = Path(config.data.processed_dir)
    regime_dir = Path(config.regime.output_dir)
    signals = pd.read_parquet(config.alpha.signals_path)
    returns = pd.read_parquet(processed_dir / "returns.parquet")
    regime_probs = pd.read_parquet(regime_dir / "regime_probs.parquet")
    
    model_path = Path("models/rl_ppo_agent")
    
    if args.mode == "train":
        from src.rl.training import train_rl
        # We call the function we already tested
        print("Training RL agent...")
        # Since train_rl uses its own argparse, we'll just re-implement the call here for simplicity
        # or we could refactor. For this CLI, we'll just use the existing script.
        import subprocess
        command = ["python", "-m", "src.rl.training", "--config", args.config, "--timesteps", str(config.rl.total_timesteps)]
        if args.force:
            command.append("--force")
        subprocess.run(command, check=True)
    
    # Generate RL-tilted signals
    if not model_path.exists() and not (model_path.with_suffix(".zip")).exists():
        print(f"RL model not found at {model_path}. Please train first.")
        return

    print("Generating RL-tilted signals...")
    model = PPO.load(model_path)
    env = TradingEnv(signals, returns, regime_probs)
    
    obs, _ = env.reset()
    all_tilted_signals = []
    
    # Run through the dataset
    for i in range(len(signals)):
        action, _ = model.predict(obs, deterministic=True)
        
        # Apply tilt to current signals
        current_signals = signals.iloc[i].fillna(0).values
        tilted = current_signals * (1.0 + action)
        all_tilted_signals.append(tilted)
        
        # Step env (though we only care about the predictions here)
        if i < len(signals) - 1:
            obs, _, _, _, _ = env.step(action)
            
    tilted_df = pd.DataFrame(all_tilted_signals, index=signals.index, columns=signals.columns)
    
    output_path = processed_dir / "alpha_signals_rl_tilted.parquet"
    tilted_df.to_parquet(output_path)
    print(f"RL-tilted signals saved to {output_path}")

if __name__ == "__main__":
    run_rl_pipeline()
