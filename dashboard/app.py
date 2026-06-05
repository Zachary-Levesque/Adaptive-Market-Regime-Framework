"""AMRF research dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.rl.data import resolve_selected_signal_path
from src.config import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


@st.cache_data(show_spinner=False)
def load_frame(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> dict[str, pd.DataFrame | Path]:
    config = load_config(CONFIG_PATH)
    selected_signal_path = resolve_selected_signal_path(config)
    data = {
        "config": config,
        "selected_signal_path": selected_signal_path,
        "regime_probs": load_frame(str(config.regime.output_dir / "regime_probs.parquet")),
        "transition_matrix": load_frame(str(config.regime.output_dir / "transition_matrix.parquet")),
        "rl_positions": load_frame(str(config.rl.positions_path)),
        "rl_backtest": load_frame(str(config.rl.backtest_results_path)),
        "rl_comparison": load_frame(str(config.rl.comparison_path)),
        "static_backtest": load_frame(str(config.risk.output_dir / "backtest_results.parquet")),
        "performance_report": load_frame(str(config.risk.output_dir / "performance_report.parquet")),
        "regime_performance": load_frame(str(config.risk.output_dir / "regime_performance.parquet")),
        "alpha_diag": load_frame(str(config.alpha.diagnostics_path)),
        "alpha_diag_regime": load_frame(str(config.alpha.diagnostics_path.with_name("alpha_diagnostics_by_regime.parquet"))),
        "alpha_model_summary": load_frame(str(config.alpha.comparison_path.with_name("alpha_model_comparison_summary.parquet"))),
        "selected_signal": load_frame(str(selected_signal_path)),
        "static_signal": load_frame(str(config.alpha.signals_dir / "regime_portfolio_selector.parquet")),
    }
    return data


def current_regime_block(regime_probs: pd.DataFrame) -> tuple[str, float, pd.Series]:
    if regime_probs.empty:
        return "Unavailable", 0.0, pd.Series(dtype=float)
    latest = regime_probs.iloc[-1].fillna(0.0)
    regime = latest.idxmax()
    return str(regime), float(latest.max()), latest


def build_regime_area_chart(regime_probs: pd.DataFrame) -> go.Figure:
    if regime_probs.empty:
        return go.Figure()
    fig = px.area(
        regime_probs.reset_index().rename(columns={"index": "date"}),
        x=regime_probs.index.name or "date",
        y=regime_probs.columns,
        title="Regime Probabilities",
        color_discrete_map={
            "Bull Trending": "#2ca02c",
            "Low-Vol Compression": "#bcbd22",
            "Bear Trending": "#ff7f0e",
            "High-Vol Crisis": "#d62728",
        },
    )
    fig.update_layout(height=350, legend_title_text="Regime")
    return fig


def build_transition_heatmap(matrix: pd.DataFrame) -> go.Figure:
    if matrix.empty:
        return go.Figure()
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns.astype(str),
            y=matrix.index.astype(str),
            colorscale="Viridis",
            text=matrix.round(3).values,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(title="Regime Transition Matrix", height=350)
    return fig


def build_drawdown_gauge(drawdown: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=drawdown * 100.0,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#d62728"},
                "steps": [
                    {"range": [0, 5], "color": "#e8f5e9"},
                    {"range": [5, 15], "color": "#fff3cd"},
                    {"range": [15, 100], "color": "#f8d7da"},
                ],
            },
        )
    )
    fig.update_layout(title="Current Drawdown", height=260)
    return fig


def portfolio_page(data: dict[str, pd.DataFrame | Path]) -> None:
    rl_positions = data["rl_positions"]
    selected_signal = data["selected_signal"]
    if rl_positions.empty or selected_signal.empty:
        st.warning("Portfolio artifacts are not available yet.")
        return

    latest_rl = rl_positions.iloc[-1]
    latest_signal = selected_signal.reindex(columns=rl_positions.columns).iloc[-1].fillna(0.0)
    recent_rl = rl_positions.tail(30).reset_index().rename(columns={"index": "date"}).melt(id_vars=rl_positions.index.name or "date", var_name="asset", value_name="weight")
    recent_signal = (
        selected_signal.reindex(columns=rl_positions.columns).tail(30)
        .reset_index()
        .rename(columns={"index": "date"})
        .melt(id_vars=selected_signal.index.name or "date", var_name="asset", value_name="weight")
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Signal Weights")
        st.dataframe(latest_signal.sort_values(ascending=False).to_frame("weight"), use_container_width=True)
    with col2:
        st.subheader("Current RL Weights")
        st.dataframe(latest_rl.sort_values(ascending=False).to_frame("weight"), use_container_width=True)

    history = pd.concat(
        [
            recent_signal.assign(source="Signal"),
            recent_rl.assign(source="RL"),
        ],
        ignore_index=True,
    )
    fig = px.line(history, x="date", y="weight", color="asset", line_dash="source", title="30-Day Weight History")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    rl_backtest = data["rl_backtest"]
    drawdown = float(rl_backtest["drawdown"].iloc[-1]) if not rl_backtest.empty and "drawdown" in rl_backtest.columns else 0.0
    st.plotly_chart(build_drawdown_gauge(drawdown), use_container_width=True)


def backtest_page(data: dict[str, pd.DataFrame | Path]) -> None:
    rl_backtest = data["rl_backtest"]
    static_backtest = data["static_backtest"]
    perf = data["performance_report"]
    comparison = data["rl_comparison"]

    curves = []
    if not rl_backtest.empty:
        curves.append(rl_backtest[["portfolio_value"]].rename(columns={"portfolio_value": "RL Agent"}))
    if not static_backtest.empty and "equity" in static_backtest.columns:
        curves.append(static_backtest[["equity"]].rename(columns={"equity": "Static Signal"}))
    if not static_backtest.empty and "benchmark_equity" in static_backtest.columns:
        curves.append(static_backtest[["benchmark_equity"]].rename(columns={"benchmark_equity": "SPY"}))
    if not static_backtest.empty and "equal_weight_equity" in static_backtest.columns:
        curves.append(static_backtest[["equal_weight_equity"]].rename(columns={"equal_weight_equity": "Equal Weight"}))

    if curves:
        equity = pd.concat(curves, axis=1)
        fig = px.line(equity, x=equity.index, y=equity.columns, title="Equity Curves")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    table = pd.DataFrame()
    if not perf.empty:
        table = perf.copy()
    if not comparison.empty:
        table = pd.concat([table, comparison], axis=0, sort=False)
    if not table.empty:
        st.subheader("Performance Summary")
        st.dataframe(table, use_container_width=True)

    regime_perf = data["regime_performance"]
    if not regime_perf.empty:
        st.subheader("Regime-Conditional Performance")
        st.dataframe(regime_perf, use_container_width=True)


def diagnostics_page(data: dict[str, pd.DataFrame | Path]) -> None:
    diag = data["alpha_diag"]
    diag_regime = data["alpha_diag_regime"]
    model_summary = data["alpha_model_summary"]
    static_backtest = data["static_backtest"]

    if not diag.empty and "ic" in diag.columns:
        fig = px.line(diag.reset_index(), x="date", y="ic", title="Information Coefficient Time Series")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if not diag_regime.empty and "mean_ic" in diag_regime.columns:
            fig = px.bar(
                diag_regime.reset_index(),
                x="regime",
                y="mean_ic",
                title="IC by Regime",
                color="mean_ic",
                color_continuous_scale="RdYlGn",
            )
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if not model_summary.empty and {"backtest_sharpe", "benchmark_sharpe_SPY"}.issubset(model_summary.columns):
            fig = px.scatter(
                model_summary.reset_index(),
                x="benchmark_sharpe_SPY",
                y="backtest_sharpe",
                text=model_summary.reset_index().get("model"),
                title="Signal vs Benchmark Sharpe",
            )
            st.plotly_chart(fig, use_container_width=True)

    if not static_backtest.empty and {"benchmark_return", "strategy_return"}.issubset(static_backtest.columns):
        fig = px.scatter(
            static_backtest.reset_index(),
            x="benchmark_return",
            y="strategy_return",
            title="Daily Signal vs Benchmark Scatter",
            opacity=0.5,
        )
        st.plotly_chart(fig, use_container_width=True)


def regime_page(data: dict[str, pd.DataFrame | Path]) -> None:
    regime_probs = data["regime_probs"]
    transition_matrix = data["transition_matrix"]
    if regime_probs.empty:
        st.warning("Regime artifacts are not available yet.")
        return

    regime, prob, latest = current_regime_block(regime_probs)
    st.metric("Current Regime", regime, f"{prob:.1%}")
    st.plotly_chart(build_regime_area_chart(regime_probs), use_container_width=True)
    st.plotly_chart(build_transition_heatmap(transition_matrix), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="AMRF Dashboard", layout="wide")
    st.title("Adaptive Market Regime Framework")
    data = load_dashboard_data()

    page = st.sidebar.radio(
        "Page",
        ["Regime Monitor", "Portfolio", "Backtest Results", "Alpha Diagnostics"],
        index=0,
    )

    if page == "Regime Monitor":
        regime_page(data)
    elif page == "Portfolio":
        portfolio_page(data)
    elif page == "Backtest Results":
        backtest_page(data)
    else:
        diagnostics_page(data)


if __name__ == "__main__":
    main()
