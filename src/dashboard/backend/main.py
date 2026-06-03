"""FastAPI backend for AMRF Dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="AMRF Dashboard API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to data artifacts
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
REGIMES_DIR = DATA_DIR / "regimes"
RESULTS_DIR = DATA_DIR / "results"

class RegimeStatus(BaseModel):
    current_regime: str
    probabilities: dict[str, float]
    duration_days: int

class Signal(BaseModel):
    ticker: str
    signal: str
    size: float
    conviction: float
    stop_loss: str
    take_profit: str

class PerformanceMetrics(BaseModel):
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    win_rate: float
    annual_return: float

def load_parquet_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Data file not found: {path}")
    return pd.read_parquet(path)

@app.get("/api/regime/current")
async def get_current_regime():
    probs = load_parquet_safe(REGIMES_DIR / "regime_probs.parquet")
    labels = load_parquet_safe(REGIMES_DIR / "regime_labels.parquet")
    summary = load_parquet_safe(REGIMES_DIR / "regime_summary.parquet")
    
    current_label = int(labels.iloc[-1]["regime"])
    current_probs = probs.iloc[-1].to_dict()
    
    # Calculate duration
    last_regime = labels.iloc[-1]["regime"]
    count = 0
    for val in reversed(labels["regime"].values):
        if val == last_regime:
            count += 1
        else:
            break
            
    regime_names = {
        0: "Bull Trending",
        1: "Low-Vol Compression",
        2: "Bear Trending",
        3: "High-Vol Crisis"
    }
    
    return {
        "current_regime": regime_names.get(current_label, f"Regime {current_label}"),
        "probabilities": current_probs,
        "duration_days": count
    }

@app.get("/api/signals/today")
async def get_today_signals():
    selection = load_parquet_safe(PROCESSED_DIR / "alpha_signal_selection.parquet")
    if selection.empty:
        return []
    
    signal_path = Path(selection.iloc[0]["signal_path"])
    signals = load_parquet_safe(signal_path)
    
    if signals.empty:
        return []
    
    latest_signals = signals.iloc[-1].dropna()
    results = []
    for ticker, val in latest_signals.items():
        results.append({
            "ticker": ticker,
            "signal": "LONG" if val > 0 else "SHORT",
            "size": abs(val) * 10000, # Dummy scaling for now
            "conviction": abs(val),
            "stop_loss": "-5.0%", # Placeholder
            "take_profit": "+12.0%" # Placeholder
        })
    return results

@app.get("/api/portfolio/performance")
async def get_performance():
    results = load_parquet_safe(RESULTS_DIR / "backtest_results.parquet")
    df = results.reset_index()
    if 'date' not in df.columns and 'index' in df.columns:
        df = df.rename(columns={'index': 'date'})
    if "portfolio_value" not in df.columns and "equity" in df.columns:
        df["portfolio_value"] = df["equity"]
    if "benchmark_value" not in df.columns and "benchmark_equity" in df.columns:
        df["benchmark_value"] = df["benchmark_equity"]
        
    # Simplify for the chart (return last 500 days)
    chart_data = df.tail(500).to_dict(orient="records")
    return chart_data

@app.get("/api/risk/metrics")
async def get_risk_metrics():
    report = load_parquet_safe(RESULTS_DIR / "performance_report.parquet")
    if "strategy" not in report.index:
        raise HTTPException(status_code=404, detail="Strategy metrics not found")
        
    strategy = report.loc["strategy"]
    return {
        "sharpe": float(strategy.get("sharpe", 0.0)),
        "sortino": float(strategy.get("sortino", 0.0)),
        "calmar": float(strategy.get("calmar", 0.0)),
        "max_drawdown": float(strategy.get("max_drawdown", 0.0)),
        "win_rate": float(strategy.get("win_rate", 0.0)),
        "annual_return": float(strategy.get("annual_return", 0.0))
    }

@app.get("/api/readiness")
async def get_readiness():
    report = load_parquet_safe(PROCESSED_DIR / "alpha_readiness_report.parquet")
    return report.to_dict(orient="records")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
