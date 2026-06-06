"""AMRF research dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
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
        "selection_manifest": load_frame(str(config.alpha.selection_path)),
        "regime_probs": load_frame(str(config.regime.output_dir / "regime_probs.parquet")),
        "transition_matrix": load_frame(str(config.regime.output_dir / "transition_matrix.parquet")),
        "rl_positions": load_frame(str(config.rl.positions_path)),
        "rl_backtest": load_frame(str(config.rl.backtest_results_path)),
        "rl_comparison": load_frame(str(config.rl.comparison_path)),
        "strategy_backtest": load_frame(str(config.risk.output_dir / "backtest_results.parquet")),
        "performance_report": load_frame(str(config.risk.output_dir / "performance_report.parquet")),
        "allocation_exposure": load_frame(str(config.risk.output_dir / "allocation_exposure.parquet")),
        "allocation_policy": load_frame(str(config.risk.output_dir / "allocation_policy.parquet")),
        "regime_performance": load_frame(str(config.risk.output_dir / "regime_performance.parquet")),
        "alpha_diag": load_frame(str(config.alpha.diagnostics_path)),
        "alpha_diag_regime": load_frame(str(config.alpha.diagnostics_path.with_name("alpha_diagnostics_by_regime.parquet"))),
        "alpha_model_summary": load_frame(str(config.alpha.comparison_path.with_name("alpha_model_comparison_summary.parquet"))),
        "readiness_report": load_frame(str(config.alpha.diagnostics_path.with_name("alpha_readiness_report.parquet"))),
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


def latest_index_date(*frames: pd.DataFrame) -> str:
    latest: pd.Timestamp | None = None
    for frame in frames:
        if frame.empty:
            continue
        idx = pd.to_datetime(pd.Index(frame.index), errors="coerce")
        idx = idx[~idx.isna()]
        if idx.empty:
            continue
        candidate = idx.max()
        if latest is None or candidate > latest:
            latest = candidate
    return latest.date().isoformat() if latest is not None else "Unavailable"


def selected_model_name(selection_manifest: pd.DataFrame) -> str:
    if selection_manifest.empty:
        return "Unavailable"
    row = selection_manifest.iloc[0]
    model = str(row.get("model", "")).strip()
    if model:
        return model
    signal_path = str(row.get("signal_path", "")).strip()
    return Path(signal_path).stem if signal_path else "Unavailable"


def readiness_status(readiness_report: pd.DataFrame) -> tuple[str, str]:
    if readiness_report.empty or "ready_for_rl" not in readiness_report.columns:
        return "Unavailable", "No readiness report"
    ready = bool(readiness_report["ready_for_rl"].all())
    passed = int(readiness_report["passed"].astype(bool).sum()) if "passed" in readiness_report.columns else 0
    total = int(len(readiness_report))
    status = "Ready" if ready else "Blocked"
    detail = f"{passed}/{total} checks passed"
    return status, detail


def hover_text(label: str, help_text: str) -> str:
    return (
        f'<span title="{help_text}" style="cursor: help; border-bottom: 1px dotted #777;">'
        f"{label}</span>"
    )


def metric_block(
    column,
    label: str,
    value,
    help_text: str,
    delta: str | None = None,
) -> None:
    column.metric(label, value, delta, help=help_text)


def build_regime_area_chart(regime_probs: pd.DataFrame) -> go.Figure:
    if regime_probs.empty:
        return go.Figure()
    colors = {
        "Bull Trending": "#2ca02c",
        "Low-Vol Compression": "#bcbd22",
        "Bear Trending": "#ff7f0e",
        "High-Vol Crisis": "#d62728",
    }
    fig = go.Figure()
    x = regime_probs.index
    for i, column in enumerate(regime_probs.columns):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=regime_probs[column],
                name=column,
                mode="lines",
                line={"color": colors.get(column, None), "width": 1.5},
                stackgroup="regimes",
                opacity=0.8,
            )
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
    drawdown_pct = abs(float(drawdown)) * 100.0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=drawdown_pct,
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


def overview_page(data: dict[str, pd.DataFrame | Path]) -> None:
    regime_probs = data["regime_probs"]
    performance = data["performance_report"]
    readiness = data["readiness_report"]
    selection = data["selection_manifest"]
    selected_signal = data["selected_signal"]
    rl_backtest = data["rl_backtest"]
    strategy_backtest = data["strategy_backtest"]
    model_summary = data["alpha_model_summary"]
    allocation_exposure = data["allocation_exposure"]
    allocation_policy = data["allocation_policy"]

    current_regime, regime_prob, _ = current_regime_block(regime_probs)
    selected_model = selected_model_name(selection)
    status, status_detail = readiness_status(readiness)
    latest_date = latest_index_date(regime_probs, selected_signal, strategy_backtest, rl_backtest)
    latest_alpha_exposure = (
        float(allocation_exposure["alpha_exposure"].dropna().iloc[-1])
        if not allocation_exposure.empty and "alpha_exposure" in allocation_exposure.columns and allocation_exposure["alpha_exposure"].notna().any()
        else None
    )

    strategy_row = performance.loc["strategy"] if not performance.empty and "strategy" in performance.index else pd.Series(dtype=float)
    spy_row = performance.loc["SPY"] if not performance.empty and "SPY" in performance.index else pd.Series(dtype=float)
    excess_sharpe = float(strategy_row.get("sharpe", 0.0)) - float(spy_row.get("sharpe", 0.0))

    st.markdown("### Snapshot")
    metric_cols = st.columns(5)
    metric_block(
        metric_cols[0],
        "Current Regime",
        current_regime,
        "The latest HMM regime with the highest probability. This is the model’s current market-state estimate.",
        f"{regime_prob:.1%}",
    )
    metric_block(
        metric_cols[1],
        "Selected Model",
        selected_model,
        "The alpha signal chosen by the model-comparison and selection pipeline. This is the signal the backtest uses.",
    )
    metric_block(
        metric_cols[2],
        "Data Through",
        latest_date,
        "The latest date available in the saved artifacts powering the dashboard. The app is reading parquet files, not live quotes.",
    )
    metric_block(
        metric_cols[3],
        "RL Readiness",
        status,
        "Whether the selected signal clears the pre-RL quality gate. Blocked means one or more advisory checks failed.",
        status_detail,
    )
    metric_block(
        metric_cols[4],
        "Sharpe vs SPY",
        f"{excess_sharpe:+.2f}",
        "Strategy Sharpe minus SPY Sharpe. Positive means the selected strategy is beating SPY on risk-adjusted return.",
    )

    if not performance.empty and "strategy" in performance.index:
        st.markdown("### Performance")
        perf_cols = st.columns(4)
        metric_block(
            perf_cols[0],
            "Strategy Sharpe",
            f"{float(strategy_row.get('sharpe', 0.0)):.2f}",
            "Annualized risk-adjusted return for the selected strategy.",
        )
        metric_block(
            perf_cols[1],
            "SPY Sharpe",
            f"{float(spy_row.get('sharpe', 0.0)):.2f}",
            "Annualized risk-adjusted return for the benchmark ETF.",
        )
        metric_block(
            perf_cols[2],
            "Strategy Total Return",
            f"{float(strategy_row.get('total_return', 0.0)):.2f}",
            "Total compounded return for the selected strategy over the backtest window.",
        )
        metric_block(
            perf_cols[3],
            "Max Drawdown",
            f"{abs(float(strategy_row.get('max_drawdown', 0.0))):.2%}",
            "Largest peak-to-trough decline in the selected strategy’s equity curve.",
        )

    if latest_alpha_exposure is not None:
        st.markdown("### Portfolio Allocation")
        alloc_cols = st.columns(3)
        metric_block(
            alloc_cols[0],
            "Alpha Sleeve",
            f"{latest_alpha_exposure:.0%}",
            "Current portfolio exposure assigned to the selected alpha sleeve after the regime-aware allocation rule.",
        )
        metric_block(
            alloc_cols[1],
            "SPY Sleeve",
            f"{1.0 - latest_alpha_exposure:.0%}",
            "Current portfolio exposure assigned to SPY by the regime-aware allocation rule.",
        )
        if not allocation_policy.empty:
            metric_block(
                alloc_cols[2],
                "Allocation Rule",
                "Regime Blend",
                "The portfolio layer blends the alpha sleeve with SPY using explicit regime exposure settings.",
            )
            visible = [c for c in ["regime", "alpha_exposure", "benchmark", "benchmark_exposure"] if c in allocation_policy.columns]
            st.dataframe(allocation_policy[visible], width="stretch")

    if not model_summary.empty:
        st.markdown(f"### {hover_text('Selected Alpha Candidates', 'The highest-ranked alpha models from the comparison stage.')} ", unsafe_allow_html=True)
        visible = [c for c in ["mean_sharpe", "mean_rank_ic", "projected_backtest_sharpe", "projected_total_return"] if c in model_summary.columns]
        if visible:
            st.dataframe(model_summary[visible].head(8).round(4), width="stretch")

    if not readiness.empty and "passed" in readiness.columns:
        failed = readiness.loc[~readiness["passed"].astype(bool)]
        if not failed.empty:
            st.markdown(
                f"### {hover_text('Current Blockers', 'The checks currently preventing RL deployment or signaling where the chosen strategy is weak.')}",
                unsafe_allow_html=True,
            )
            st.dataframe(failed[["check", "value", "detail"]], width="stretch")


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
        st.subheader("Selected Strategy Weights")
        st.dataframe(latest_signal.sort_values(ascending=False).to_frame("weight"), width="stretch")
    with col2:
        st.subheader("RL Weights")
        st.dataframe(latest_rl.sort_values(ascending=False).to_frame("weight"), width="stretch")

    history = pd.concat(
        [
            recent_signal.assign(source="Signal"),
            recent_rl.assign(source="RL"),
        ],
        ignore_index=True,
    )
    fig = go.Figure()
    dash_map = {"Signal": "solid", "RL": "dash"}
    for (source, asset), subset in history.groupby(["source", "asset"], sort=False):
        fig.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset["weight"],
                mode="lines",
                name=f"{asset} ({source})",
                line={"dash": dash_map.get(source, "solid")},
            )
        )
    fig.update_layout(title="30-Day Weight History", height=420, legend_title_text="Asset / Source")
    fig.update_layout(height=420)
    st.plotly_chart(fig, width="stretch")

    rl_backtest = data["rl_backtest"]
    drawdown = float(rl_backtest["drawdown"].iloc[-1]) if not rl_backtest.empty and "drawdown" in rl_backtest.columns else 0.0
    st.plotly_chart(build_drawdown_gauge(drawdown), width="stretch")


def backtest_page(data: dict[str, pd.DataFrame | Path]) -> None:
    rl_backtest = data["rl_backtest"]
    strategy_backtest = data["strategy_backtest"]
    perf = data["performance_report"]
    comparison = data["rl_comparison"]

    curves = []
    if not rl_backtest.empty:
        curves.append(rl_backtest[["portfolio_value"]].rename(columns={"portfolio_value": "RL Agent"}))
    if not strategy_backtest.empty and "equity" in strategy_backtest.columns:
        curves.append(strategy_backtest[["equity"]].rename(columns={"equity": "Selected Strategy"}))
    if not strategy_backtest.empty and "benchmark_equity" in strategy_backtest.columns:
        curves.append(strategy_backtest[["benchmark_equity"]].rename(columns={"benchmark_equity": "SPY"}))
    if not strategy_backtest.empty and "equal_weight_equity" in strategy_backtest.columns:
        curves.append(strategy_backtest[["equal_weight_equity"]].rename(columns={"equal_weight_equity": "Equal Weight"}))

    if curves:
        equity = pd.concat(curves, axis=1)
        fig = go.Figure()
        for column in equity.columns:
            fig.add_trace(go.Scatter(x=equity.index, y=equity[column], mode="lines", name=column))
        fig.update_layout(title="Equity Curves", height=420, legend_title_text="Series")
        fig.update_layout(height=420)
        st.plotly_chart(fig, width="stretch")

    table = pd.DataFrame()
    if not perf.empty:
        table = perf.copy()
    if not comparison.empty:
        table = pd.concat([table, comparison], axis=0, sort=False)
    if not table.empty:
        st.subheader("Performance Summary")
        st.dataframe(table, width="stretch")

    regime_perf = data["regime_performance"]
    if not regime_perf.empty:
        st.subheader("Regime-Conditional Performance")
        st.dataframe(regime_perf, width="stretch")


def diagnostics_page(data: dict[str, pd.DataFrame | Path]) -> None:
    diag = data["alpha_diag"]
    diag_regime = data["alpha_diag_regime"]
    model_summary = data["alpha_model_summary"]
    strategy_backtest = data["strategy_backtest"]
    readiness = data["readiness_report"]

    if not diag.empty and "ic" in diag.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=diag.index, y=diag["ic"], mode="lines", name="IC"))
        fig.update_layout(title="Information Coefficient Time Series", height=350)
        st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        if not diag_regime.empty and "mean_ic" in diag_regime.columns:
            colors = ["#2ca02c" if v >= 0 else "#d62728" for v in diag_regime["mean_ic"]]
            fig = go.Figure(
                data=go.Bar(
                    x=diag_regime.index.astype(str),
                    y=diag_regime["mean_ic"],
                    marker={"color": colors},
                )
            )
            fig.update_layout(title="IC by Regime", height=350)
            st.plotly_chart(fig, width="stretch")
    with col2:
        if not model_summary.empty and {"backtest_sharpe", "benchmark_sharpe_SPY"}.issubset(model_summary.columns):
            labels = model_summary.index.astype(str).tolist()
            fig = go.Figure(
                data=go.Scatter(
                    x=model_summary["benchmark_sharpe_SPY"],
                    y=model_summary["backtest_sharpe"],
                    mode="markers+text",
                    text=labels,
                    textposition="top center",
                )
            )
            fig.update_layout(title="Signal vs Benchmark Sharpe", height=350)
            st.plotly_chart(fig, width="stretch")

    if not strategy_backtest.empty and {"benchmark_return", "strategy_return"}.issubset(strategy_backtest.columns):
        fig = go.Figure(
            data=go.Scatter(
                x=strategy_backtest["benchmark_return"],
                y=strategy_backtest["strategy_return"],
                mode="markers",
                opacity=0.5,
            )
        )
        fig.update_layout(title="Daily Signal vs Benchmark Scatter", height=350)
        st.plotly_chart(fig, width="stretch")

    if not readiness.empty:
        st.subheader("Readiness Report")
        cols = [c for c in ["check", "passed", "value", "detail"] if c in readiness.columns]
        st.dataframe(readiness[cols], width="stretch")


def regime_page(data: dict[str, pd.DataFrame | Path]) -> None:
    regime_probs = data["regime_probs"]
    transition_matrix = data["transition_matrix"]
    if regime_probs.empty:
        st.warning("Regime artifacts are not available yet.")
        return

    regime, prob, latest = current_regime_block(regime_probs)
    st.metric("Current Regime", regime, f"{prob:.1%}")
    st.plotly_chart(build_regime_area_chart(regime_probs), width="stretch")
    st.plotly_chart(build_transition_heatmap(transition_matrix), width="stretch")


def main() -> None:
    st.set_page_config(page_title="AMRF Dashboard", layout="wide")
    st.title("Adaptive Market Regime Framework")
    data = load_dashboard_data()

    page = st.sidebar.radio(
        "Page",
        ["Overview", "Regime Monitor", "Portfolio", "Backtest Results", "Alpha Diagnostics"],
        index=0,
    )

    if page == "Overview":
        overview_page(data)
    elif page == "Regime Monitor":
        regime_page(data)
    elif page == "Portfolio":
        portfolio_page(data)
    elif page == "Backtest Results":
        backtest_page(data)
    else:
        diagnostics_page(data)


if __name__ == "__main__":
    main()
