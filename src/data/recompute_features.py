"""Utility to recompute features from existing price data, bypassing yfinance downloads."""

import yaml
from pathlib import Path
import pandas as pd
from src.config import load_config
from src.data.pipeline import DataPipeline
from src.data.ingestion import MarketDataIngester
from src.data.features import FeatureEngineer
from src.data.factors import FactorLoader
from loguru import logger

def main():
    config_path = "configs/config.yaml"
    config = load_config(config_path)
    
    processed_dir = Path(config.data.processed_dir)
    prices_path = processed_dir / "prices.parquet"
    
    if not prices_path.exists():
        logger.error("No prices.parquet found in {}. Run build_phase1 first.", processed_dir)
        return

    logger.info("Loading existing prices from {}", prices_path)
    prices = pd.read_parquet(prices_path)
    
    ingester = MarketDataIngester()
    returns = ingester.compute_returns(prices)
    
    factor_loader = FactorLoader()
    logger.info("Downloading Fama-French factors...")
    ff_factors = factor_loader.download_ff5(start=config.data.start_date, end=config.data.end_date)
    factors = factor_loader.align_with_returns(ff_factors, returns)
    
    pipeline = DataPipeline(config.data)
    logger.info("Downloading macro series from FRED...")
    macro = factor_loader.download_macro_series(
        series_map=pipeline.DEFAULT_MACRO_SERIES,
        start=config.data.start_date,
        end=config.data.end_date
    )
    macro = macro.reindex(prices.index).ffill()
    vix_series = macro["VIXCLS"] if "VIXCLS" in macro.columns else None
    
    feature_engineer = FeatureEngineer()
    logger.info("Computing enhanced technical features (including FracDiff)...")
    technical_features = feature_engineer.compute_technical_features(prices, vix=vix_series)
    technical_features = feature_engineer.normalize(technical_features)
    
    logger.info("Computing enhanced regime features...")
    regime_features = feature_engineer.compute_regime_features(
        prices=prices,
        vix=vix_series,
        macro=macro,
        benchmark=config.data.benchmark
    )
    
    # Save artifacts
    ingester.save(returns, processed_dir / "returns.parquet")
    ingester.save(factors, processed_dir / "factors.parquet")
    ingester.save(technical_features, processed_dir / "technical_features.parquet")
    ingester.save(regime_features, processed_dir / "regime_features.parquet")
    ingester.save(macro, processed_dir / "macro.parquet")
    
    logger.info("Features recomputed successfully.")

if __name__ == "__main__":
    main()
