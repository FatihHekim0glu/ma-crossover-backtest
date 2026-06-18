"""Plotly figures for the backtest report.

Every function returns a ``go.Figure`` rather than rendering. The notebook /
CLI handles display + saving so this module stays pure.

Performance: ``go.Scattergl`` (WebGL) is used for any price/return series
likely to exceed ~2,000 points. SVG-based ``go.Scatter`` is reserved for
short series and annotations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

_THEME_NAME = "ma_backtester"


def install_theme() -> None:
    """Register a polished default plotly template. Idempotent."""
    template = go.layout.Template()
    template.layout = go.Layout(
        font={"family": "Inter, Helvetica, Arial, sans-serif", "size": 12, "color": "#2a2a2a"},
        title={"x": 0.02, "xanchor": "left", "font": {"size": 16}},
        margin={"l": 60, "r": 30, "t": 60, "b": 50},
        xaxis={"showgrid": True, "gridcolor": "rgba(0,0,0,0.06)", "zeroline": False},
        yaxis={"showgrid": True, "gridcolor": "rgba(0,0,0,0.06)", "zeroline": False},
        plot_bgcolor="white",
        paper_bgcolor="white",
        colorway=["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"],
        hovermode="x unified",
    )
    pio.templates[_THEME_NAME] = template
    pio.templates.default = _THEME_NAME


def equity_curve(
    *,
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series | None = None,
    title: str = "Equity Curve",
    log_scale: bool = True,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=strategy_equity.index,
            y=strategy_equity.to_numpy(),
            mode="lines",
            name="Strategy",
            line={"width": 2},
        )
    )
    if benchmark_equity is not None:
        fig.add_trace(
            go.Scattergl(
                x=benchmark_equity.index,
                y=benchmark_equity.to_numpy(),
                mode="lines",
                name="Buy & Hold",
                line={"width": 1.5, "dash": "dash"},
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Equity",
        yaxis_type="log" if log_scale else "linear",
        legend={"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.8)"},
    )
    return fig


def underwater_drawdown(
    *,
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series | None = None,
    title: str = "Drawdown (underwater)",
) -> go.Figure:
    def _dd(eq: pd.Series) -> pd.Series:
        return eq / eq.cummax() - 1.0

    fig = go.Figure()
    strat_dd = _dd(strategy_equity)
    # Strategy = blue, Buy & Hold = red (dashed) - matches the equity curve
    # so the two charts read as a coherent pair.
    fig.add_trace(
        go.Scattergl(
            x=strat_dd.index,
            y=strat_dd.to_numpy(),
            fill="tozeroy",
            name="Strategy",
            line={"color": "rgb(31,119,180)", "width": 1.4},
            fillcolor="rgba(31,119,180,0.30)",
        )
    )
    if benchmark_equity is not None:
        bench_dd = _dd(benchmark_equity)
        fig.add_trace(
            go.Scattergl(
                x=bench_dd.index,
                y=bench_dd.to_numpy(),
                fill="tozeroy",
                name="Buy & Hold",
                line={"color": "rgb(214,39,40)", "width": 1.0, "dash": "dash"},
                fillcolor="rgba(214,39,40,0.15)",
            )
        )

    worst_idx = strat_dd.idxmin()
    fig.add_annotation(
        x=worst_idx,
        y=float(strat_dd.min()),
        text=f"{float(strat_dd.min()):.1%}",
        arrowhead=2,
        ax=40,
        ay=-30,
    )
    fig.update_layout(title=title, yaxis_tickformat=".0%", xaxis_title="Date")
    return fig


def returns_distribution(
    *,
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    title: str = "Daily Returns Distribution",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=strategy_returns.dropna(),
            histnorm="probability density",
            opacity=0.6,
            nbinsx=80,
            name="Strategy",
        )
    )
    if benchmark_returns is not None:
        fig.add_trace(
            go.Histogram(
                x=benchmark_returns.dropna(),
                histnorm="probability density",
                opacity=0.6,
                nbinsx=80,
                name="Buy & Hold",
            )
        )

    r = strategy_returns.dropna()
    if len(r) > 0:
        stats_text = (
            f"μ = {float(r.mean()):.3%}<br>"
            f"σ = {float(r.std(ddof=1)):.3%}<br>"  # noqa: RUF001
            f"skew = {float(r.skew()):.2f}<br>"
            f"kurt = {float(r.kurtosis()):.2f}"
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            text=stats_text,
            align="left",
            bordercolor="gray",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.8)",
        )

    fig.update_layout(title=title, barmode="overlay", xaxis_tickformat=".2%", yaxis_title="Density")
    return fig


def rolling_sharpe(
    *,
    strategy_returns: pd.Series,
    window: int = 252,
    title: str | None = None,
) -> go.Figure:
    r = strategy_returns.dropna()
    rolling_mean = r.rolling(window).mean()
    rolling_std = r.rolling(window).std(ddof=1)
    rs = rolling_mean / rolling_std * np.sqrt(252)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(x=rs.index, y=rs.to_numpy(), mode="lines", name=f"Rolling Sharpe ({window}d)")
    )
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(0,0,0,0.4)")
    fig.add_hline(y=1, line_dash="dot", line_color="rgba(0,150,0,0.4)", annotation_text="SR=1")
    fig.update_layout(
        title=title or f"Rolling Sharpe ({window}-day window)",
        xaxis_title="Date",
        yaxis_title="Annualised Sharpe",
    )
    return fig


def signal_overlay(
    *,
    close: pd.Series,
    positions: pd.Series,
    title: str = "Price with Entry / Exit Markers",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=close.index,
            y=close.to_numpy(),
            mode="lines",
            name="Close",
            line={"width": 1.2},
        )
    )
    delta = positions.diff().fillna(positions.iloc[0])
    entries = delta > 0
    exits = delta < 0
    fig.add_trace(
        go.Scattergl(
            x=close.index[entries],
            y=close[entries].to_numpy(),
            mode="markers",
            name="Entry",
            marker={"symbol": "triangle-up", "size": 9, "color": "green"},
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=close.index[exits],
            y=close[exits].to_numpy(),
            mode="markers",
            name="Exit",
            marker={"symbol": "triangle-down", "size": 9, "color": "red"},
        )
    )
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price")
    return fig


def parameter_heatmap(
    *,
    sharpe_grid: pd.DataFrame,
    title: str | None = None,
) -> go.Figure:
    """Sharpe heatmap with the overfitting caveat baked into the title.

    The caveat is part of the chart on purpose: this plot is widely
    misinterpreted, and a static caption in a notebook is too easy to skip.
    """
    safe_title = title or (
        "In-sample parameter sweep - DIAGNOSTIC ONLY"
        "<br><sub>High-Sharpe regions reflect overfitting to one historical path. "
        "Use walk-forward results for selection.</sub>"
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=sharpe_grid.to_numpy(),
            x=sharpe_grid.columns.to_list(),
            y=sharpe_grid.index.to_list(),
            colorscale="RdBu",
            zmid=0.0,
            colorbar={"title": "Sharpe"},
        )
    )
    fig.update_layout(
        title=safe_title,
        xaxis_title="slow window (bars)",
        yaxis_title="fast window (bars)",
        height=600,
    )
    return fig


def report_dashboard(
    *,
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    strategy_returns: pd.Series,
    title: str = "Backtest Report",
) -> go.Figure:
    """Three-panel: equity (log), drawdown, rolling Sharpe - shared x-axis."""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Equity (log scale)", "Drawdown", "Rolling Sharpe (252d)"),
    )

    fig.add_trace(
        go.Scattergl(x=strategy_equity.index, y=strategy_equity.to_numpy(), name="Strategy"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=benchmark_equity.index,
            y=benchmark_equity.to_numpy(),
            name="Buy & Hold",
            line={"dash": "dash"},
        ),
        row=1,
        col=1,
    )

    strat_dd = strategy_equity / strategy_equity.cummax() - 1.0
    fig.add_trace(
        go.Scattergl(
            x=strat_dd.index,
            y=strat_dd.to_numpy(),
            fill="tozeroy",
            name="DD",
            line={"color": "rgb(31,119,180)"},
            fillcolor="rgba(31,119,180,0.30)",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    r = strategy_returns.dropna()
    rs = r.rolling(252).mean() / r.rolling(252).std(ddof=1) * np.sqrt(252)
    fig.add_trace(
        go.Scattergl(x=rs.index, y=rs.to_numpy(), name="Rolling SR", showlegend=False),
        row=3,
        col=1,
    )

    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_layout(title=title, height=900, hovermode="x unified")
    return fig
