"""CLI entrypoint for Module 7 — Intraday Execution Layer."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import load_config
from src.execution.alpaca import AlpacaDataFeed
from src.execution.intraday import IntradaySignalGenerator

def run_intraday_layer():
    parser = argparse.ArgumentParser(description="Generate intraday trade entry tickets.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--ticker", action="append", help="Specific tickers to check (optional).")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # 1. Load the latest RL-tilted signals
    signals_path = Path(config.data.processed_dir) / "alpha_signals_rl_tilted.parquet"
    if not signals_path.exists():
        # Fallback to base signals if RL not run
        signals_path = config.alpha.signals_path
        
    if not signals_path.exists():
        logger.error("No alpha signals found. Run the research pipeline first.")
        return

    signals = pd.read_parquet(signals_path)
    latest_signals = signals.iloc[-1].dropna()
    
    if args.ticker:
        latest_signals = latest_signals.reindex(args.ticker).dropna()
        
    if latest_signals.empty:
        logger.info("No active alpha signals for today.")
        return

    # 2. Connect to Alpaca
    try:
        feed = AlpacaDataFeed()
    except Exception as e:
        logger.error(f"Failed to initialize Alpaca feed: {e}")
        return

    generator = IntradaySignalGenerator()
    tickets = []

    logger.info(f"Checking intraday confirmation for {len(latest_signals)} tickers...")

    for ticker, daily_signal in latest_signals.items():
        if ticker == "^VIX" or ticker == "VIX":
            continue
            
        try:
            # Fetch last 50 bars (5Min)
            bars = feed.get_intraday_bars(ticker, limit=50)
            if bars.empty:
                logger.warning(f"No intraday data for {ticker}")
                continue
                
            ticket = generator.generate_entry_signal(ticker, bars, daily_signal)
            tickets.append(ticket)
            
            if ticket["signal"] != "FLAT":
                logger.success(f"TRADE TICKET: {ticker} {ticket['signal']} @ {ticket['entry_price']:.2f}")
            else:
                logger.info(f"WAIT: {ticker} - {ticket['reason']}")
                
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")

    # 3. Save tickets
    output_path = Path(config.risk.output_dir) / "intraday_tickets.parquet"
    pd.DataFrame(tickets).to_parquet(output_path)
    logger.info(f"Saved {len(tickets)} tickets to {output_path}")

if __name__ == "__main__":
    run_intraday_layer()
