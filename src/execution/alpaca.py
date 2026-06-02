"""Alpaca API integration for intraday data."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
try:
    import alpaca_trade_api as tradeapi
except ImportError:
    tradeapi = None

from dotenv import load_dotenv

load_dotenv()

class AlpacaDataFeed:
    """Fetch intraday data from Alpaca."""

    def __init__(self, api_key: str | None = None, secret_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        self.base_url = base_url or os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        
        if tradeapi is None:
            raise ImportError("alpaca-trade-api is required. Install it first.")
            
        if not self.api_key or not self.secret_key:
            # We allow initialization without keys for testing, but methods will fail
            self.api = None
        else:
            self.api = tradeapi.REST(self.api_key, self.secret_key, self.base_url, api_version='v2')

    def get_intraday_bars(self, ticker: str, timeframe: str = '5Min', limit: int = 1000) -> pd.DataFrame:
        """Fetch historical intraday bars for a ticker."""
        if self.api is None:
            raise ValueError("Alpaca API keys not configured.")
            
        # Map timeframe to Alpaca format if needed
        # Alpaca-trade-api v3 uses TimeFrame objects or strings like '5Min', '1Day'
        bars = self.api.get_bars(ticker, timeframe, limit=limit).df
        if bars.empty:
            return pd.DataFrame()
            
        bars.index = pd.to_datetime(bars.index)
        return bars

    def get_latest_quote(self, ticker: str) -> dict:
        """Fetch the latest bid/ask quote."""
        if self.api is None:
            raise ValueError("Alpaca API keys not configured.")
            
        quote = self.api.get_latest_quote(ticker)
        return {
            "bid": quote.bp,
            "ask": quote.ap,
            "bid_size": quote.bs,
            "ask_size": quote.as_
        }
