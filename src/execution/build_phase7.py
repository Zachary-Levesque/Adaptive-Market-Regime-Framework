"""CLI entrypoint for Module 7 — Intraday Execution Layer."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from src.alpha.readiness import load_readiness_status
from src.config import load_config
from src.execution.alpaca import AlpacaDataFeed
from src.execution.intraday import IntradaySignalGenerator

def run_intraday_layer():
    parser = argparse.ArgumentParser(description="Generate intraday trade entry tickets.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--ticker", action="append", help="Specific tickers to check (optional).")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # 1. Load the latest signal approved by the research gate.
    signals_path = Path(config.data.processed_dir) / "alpha_signals_rl_tilted.parquet"
    ready, _ = load_readiness_status(config.alpha.diagnostics_path.with_name("alpha_readiness_report.parquet"))
    if not ready or not signals_path.exists():
        if signals_path.exists() and not ready:
            logger.warning("Ignoring RL-tilted signals because selected alpha is not ready for RL.")
        signals_path = resolve_selected_signal_path(config)
        
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


def resolve_selected_signal_path(config) -> Path:
    selection_path = config.alpha.selection_path
    if selection_path.exists():
        selection = pd.read_parquet(selection_path)
        if not selection.empty and "signal_path" in selection.columns:
            selected_path = Path(str(selection.iloc[0]["signal_path"]))
            if selected_path.exists() and selected_path.name != "alpha_signals_rl_tilted.parquet":
                return selected_path

    return config.alpha.signals_path


if __name__ == "__main__":
    run_intraday_layer()
