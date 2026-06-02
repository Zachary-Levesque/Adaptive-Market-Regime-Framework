"""Intraday signal confirmation logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

class IntradaySignalGenerator:
    """Generate entry signals with intraday confirmation."""

    def __init__(self, volume_threshold: float = 1.5, momentum_window: int = 5, vwap_threshold: float = 0.005):
        self.volume_threshold = volume_threshold
        self.momentum_window = momentum_window
        self.vwap_threshold = vwap_threshold

    def compute_vwap(self, bars: pd.DataFrame) -> pd.Series:
        """Compute Volume Weighted Average Price."""
        v = bars['volume']
        p = (bars['high'] + bars['low'] + bars['close']) / 3.0
        return (p * v).cumsum() / v.cumsum()

    def generate_entry_signal(self, ticker: str, bars: pd.DataFrame, daily_signal: float) -> dict:
        """Combine daily alpha with intraday confirmation."""
        if bars.empty:
            return {"ticker": ticker, "signal": "FLAT", "reason": "No data"}

        vwap = self.compute_vwap(bars)
        current_price = bars['close'].iloc[-1]
        current_vwap = vwap.iloc[-1]
        
        # Volume confirmation
        avg_vol = bars['volume'].tail(10).mean()
        curr_vol = bars['volume'].iloc[-1]
        vol_confirm = curr_vol > (avg_vol * self.volume_threshold)
        
        # Momentum confirmation
        mom = bars['close'].diff(self.momentum_window).iloc[-1]
        mom_confirm = mom > 0 if daily_signal > 0 else mom < 0
        
        # VWAP confirmation
        vwap_confirm = current_price > current_vwap * (1 + self.vwap_threshold) if daily_signal > 0 else current_price < current_vwap * (1 - self.vwap_threshold)
        
        is_entry = daily_signal != 0 and vol_confirm and mom_confirm and vwap_confirm
        
        if is_entry:
            signal_type = "LONG" if daily_signal > 0 else "SHORT"
            atr = self._compute_atr(bars)
            stop_loss = current_price - (2.0 * atr) if daily_signal > 0 else current_price + (2.0 * atr)
            take_profit = current_price + (2.5 * (current_price - stop_loss)) if daily_signal > 0 else current_price - (2.5 * (stop_loss - current_price))
            
            return {
                "ticker": ticker,
                "signal": signal_type,
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "confidence": abs(daily_signal),
                "reason": "Confirmed by VWAP, Volume, and Momentum"
            }
        
        return {
            "ticker": ticker, 
            "signal": "FLAT", 
            "reason": "Daily signal exists but intraday confirmation failed" if daily_signal != 0 else "No daily signal"
        }

    def _compute_atr(self, bars: pd.DataFrame, window: int = 14) -> float:
        """Compute Average True Range."""
        high = bars['high']
        low = bars['low']
        close_prev = bars['close'].shift(1)
        
        tr = pd.concat([
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs()
        ], axis=1).max(axis=1)
        
        return float(tr.tail(window).mean())
