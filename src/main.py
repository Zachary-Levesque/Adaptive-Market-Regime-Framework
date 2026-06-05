"""Legacy AMRF daily runner prototype.

Use ``./run_pipeline.sh`` for the supported refresh flow and
``.venv/bin/python -m streamlit run dashboard/app.py`` for the dashboard.
"""

import argparse
import sys
from pathlib import Path
from loguru import logger
import pandas as pd

from src.config import load_config
from src.data.ingestion import MarketDataIngester
from src.data.features import FeatureEngineer
from src.regime.hmm import RegimeHMM
from src.alpha.ensemble import RegimeAlphaEnsemble
from src.rl.environment import TradingEnv
from stable_baselines3 import PPO

def run_daily():
    parser = argparse.ArgumentParser(description="AMRF Daily Runner")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.")
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info("Starting AMRF Daily Signal Generation")

    # 1. Update Market Data
    ingester = MarketDataIngester()
    logger.info("Downloading latest prices...")
    # Fetch last 2 years for feature computation stability
    prices = ingester.download_prices(config.data.universe)
    if prices.empty:
        logger.error("Failed to download prices. Aborting.")
        sys.exit(1)

    # 2. Engineer Features
    engineer = FeatureEngineer()
    logger.info("Computing features...")
    # We only need market-level for regime and stock-level for alpha
    regime_features = engineer.compute_regime_features(prices)
    technical_features = engineer.compute_technical_features(prices)
    technical_features = engineer.normalize(technical_features)

    # 3. Detect Current Regime
    hmm = RegimeHMM()
    hmm.load(config.regime.model_path)
    current_regime = hmm.predict_regimes(regime_features)[-1]
    regime_probs = hmm.predict_proba(regime_features)[-1]
    
    regime_name = config.regime.regime_names.get(int(current_regime), str(current_regime))
    logger.info(f"DETECTED REGIME: {regime_name} (Prob: {regime_probs[int(current_regime)]:.2%})")

    # 4. Generate Alpha Signals
    # We use the selected regime-specific model
    ensemble_dir = config.alpha.model_dir / f"regime_{int(current_regime)}"
    if not ensemble_dir.exists():
        logger.warning(f"No trained model for regime {current_regime}. Falling back to Bull model.")
        ensemble_dir = config.alpha.model_dir / "regime_0"
        
    ensemble = RegimeAlphaEnsemble(
        input_size=technical_features.shape[1], # Approximate
        target_regime=int(current_regime)
    )
    ensemble.load(ensemble_dir)
    
    # We need a RegimeDataset to format the inputs for the ensemble
    # For a daily runner, we just need the latest window
    latest_signals = {}
    tickers = [t for t in technical_features.columns.get_level_values(0).unique() if t != "MARKET"]
    
    # This is a simplification - in a real runner we'd use the full dataset logic
    # Here we just output the raw alpha for diagnostics
    logger.info("Generating alpha forecasts...")
    # (Simplified for the prototype runner)
    
    # 5. Apply RL Tilts
    rl_model_path = Path("models/ppo_position_sizer.zip")
    if rl_model_path.exists() or (rl_model_path.with_suffix(".zip")).exists():
        logger.info("Applying RL position sizing tilts...")
        rl_model = PPO.load(rl_model_path)
        # Construct current observation
        # [Alpha Signals] + [Regime Probs]
        # dummy_alpha = np.zeros(len(tickers)) 
        # obs = np.concatenate([dummy_alpha, regime_probs])
        # action, _ = rl_model.predict(obs)
    else:
        logger.warning("RL agent not found. Using raw alpha weights.")

    logger.success("Daily Signal Run Complete. Results available in Dashboard.")

if __name__ == "__main__":
    run_daily()
