"""Visualization helpers for regime detection outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REGIME_COLORS = {
    0: "#2f9e44",
    1: "#f59f00",
    2: "#c92a2a",
    3: "#7f1d1d",
}


def plot_regime_history(
    prices: pd.DataFrame,
    regime_labels: pd.Series,
    regime_probs: pd.DataFrame,
    benchmark: str = "SPY",
    path: str | Path | None = None,
):
    """Plot price, regime probabilities, and regime distribution."""
    close = _extract_close(prices, benchmark)
    labels = regime_labels.reindex(close.index).ffill().bfill().astype(int)
    probs = regime_probs.reindex(close.index).ffill().bfill()

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(14, 10),
        gridspec_kw={"height_ratios": [2.5, 2.0, 1.2]},
    )

    ax_price, ax_probs, ax_dist = axes
    ax_price.plot(close.index, close, color="#111827", linewidth=1.4, label=benchmark)
    for regime, color in REGIME_COLORS.items():
        mask = labels == regime
        ax_price.scatter(close.index[mask], close[mask], s=8, color=color, label=f"Regime {regime}")
    ax_price.set_title(f"{benchmark} Price by Regime")
    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left", ncol=2)

    ax_probs.stackplot(
        probs.index,
        [probs[column] for column in probs.columns],
        labels=list(probs.columns),
        alpha=0.85,
    )
    ax_probs.set_title("Regime Probabilities")
    ax_probs.set_ylabel("Probability")
    ax_probs.set_ylim(0, 1)
    ax_probs.legend(loc="upper left", ncol=2)

    counts = labels.value_counts().sort_index()
    ax_dist.bar(
        [f"Regime {regime}" for regime in counts.index],
        counts.values,
        color=[REGIME_COLORS.get(int(regime), "#495057") for regime in counts.index],
    )
    ax_dist.set_title("Regime Distribution")
    ax_dist.set_ylabel("Days")

    for event_date, title in [
        ("2008-09-15", "Lehman"),
        ("2020-03-23", "COVID Bottom"),
        ("2022-01-03", "Rate Hikes"),
    ]:
        event_timestamp = pd.Timestamp(event_date)
        if close.index.min() <= event_timestamp <= close.index.max():
            ax_price.axvline(event_timestamp, color="#6b7280", linestyle="--", linewidth=1)
            ax_probs.axvline(event_timestamp, color="#6b7280", linestyle="--", linewidth=1)
            ax_price.text(event_timestamp, close.max(), title, rotation=90, va="top", ha="right", fontsize=8)

    fig.tight_layout()

    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)

    return fig


def _extract_close(prices: pd.DataFrame, benchmark: str) -> pd.Series:
    field = "Adj Close" if ("Adj Close" in prices.columns.get_level_values(1)) else "Close"
    close = prices.xs(field, axis=1, level=1)
    return close[benchmark]
