"""Structural tests for plotly figure builders - no PNG rendering.

Every public function in ``ma_backtester.plotting`` returns a
``plotly.graph_objects.Figure``. Tests assert on:

- ``len(fig.data)`` and trace types
- ``fig.layout`` properties (axis type, title text, tickformat)
- ``fig.layout.annotations`` / ``fig.layout.shapes`` for added artefacts

This is intentionally cheaper than rendering and image-diffing: we are
guarding against structural regressions (missing trace, wrong axis type)
not pixel-perfect visuals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pytest

from ma_backtester import plotting
from ma_backtester.backtester import run_backtest, run_buy_and_hold
from ma_backtester.config import StrategyConfig

_STRAT = StrategyConfig(fast_window=10, slow_window=30)


@pytest.fixture
def bt_artifacts(seeded_gbm_prices: pd.Series) -> tuple:
    """Bundle of (prices, strategy result, buy-and-hold result) for plot inputs."""
    strat = run_backtest(close=seeded_gbm_prices, strategy_config=_STRAT)
    bench = run_buy_and_hold(close=seeded_gbm_prices, initial_cash=1.0)
    return seeded_gbm_prices, strat, bench


def test_install_theme_is_idempotent_and_sets_default() -> None:
    plotting.install_theme()
    plotting.install_theme()  # second call must not raise
    assert pio.templates.default == "ma_backtester"
    assert "ma_backtester" in pio.templates


def test_equity_curve_two_traces_log_axis(bt_artifacts: tuple) -> None:
    _, strat, bench = bt_artifacts
    fig = plotting.equity_curve(strategy_equity=strat.equity, benchmark_equity=bench.equity)
    assert len(fig.data) == 2
    assert all(isinstance(t, go.Scattergl) for t in fig.data)
    assert fig.layout.yaxis.type == "log"
    assert "Equity Curve" in fig.layout.title.text


def test_equity_curve_single_series_linear(bt_artifacts: tuple) -> None:
    _, strat, _ = bt_artifacts
    fig = plotting.equity_curve(strategy_equity=strat.equity, log_scale=False)
    assert len(fig.data) == 1
    assert fig.data[0].name == "Strategy"
    assert fig.layout.yaxis.type == "linear"


def test_underwater_drawdown_annotates_worst(bt_artifacts: tuple) -> None:
    _, strat, bench = bt_artifacts
    fig = plotting.underwater_drawdown(strategy_equity=strat.equity, benchmark_equity=bench.equity)
    assert len(fig.data) == 2
    assert all(isinstance(t, go.Scattergl) for t in fig.data)
    assert len(fig.layout.annotations) == 1
    assert "%" in fig.layout.annotations[0].text
    assert fig.layout.yaxis.tickformat == ".0%"


def test_returns_distribution_two_histograms_with_stats(bt_artifacts: tuple) -> None:
    _, strat, bench = bt_artifacts
    fig = plotting.returns_distribution(
        strategy_returns=strat.daily_returns, benchmark_returns=bench.daily_returns
    )
    assert len(fig.data) == 2
    assert all(isinstance(t, go.Histogram) for t in fig.data)
    assert fig.layout.barmode == "overlay"
    assert len(fig.layout.annotations) == 1


def test_returns_distribution_empty_series_skips_annotation() -> None:
    empty = pd.Series([], dtype="float64", index=pd.DatetimeIndex([]))
    fig = plotting.returns_distribution(strategy_returns=empty)
    assert len(fig.data) == 1
    assert len(fig.layout.annotations) == 0


def test_rolling_sharpe_one_trace_default_title(bt_artifacts: tuple) -> None:
    _, strat, _ = bt_artifacts
    fig = plotting.rolling_sharpe(strategy_returns=strat.daily_returns, window=126)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Scattergl)
    assert "126" in fig.layout.title.text
    # add_hline produces shapes on layout, not traces
    assert len(fig.layout.shapes) == 2


def test_signal_overlay_three_traces_with_markers(bt_artifacts: tuple) -> None:
    prices, strat, _ = bt_artifacts
    fig = plotting.signal_overlay(close=prices, positions=strat.positions)
    assert len(fig.data) == 3
    names = [t.name for t in fig.data]
    assert names == ["Close", "Entry", "Exit"]
    assert fig.data[1].marker.symbol == "triangle-up"
    assert fig.data[2].marker.symbol == "triangle-down"


def test_parameter_heatmap_type_and_caveat() -> None:
    grid = pd.DataFrame(
        np.random.default_rng(0).normal(size=(4, 5)),
        index=[5, 10, 15, 20],
        columns=[20, 40, 60, 80, 100],
    )
    fig = plotting.parameter_heatmap(sharpe_grid=grid)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Heatmap)
    assert fig.data[0].zmid == 0.0
    assert "DIAGNOSTIC ONLY" in fig.layout.title.text


def test_parameter_heatmap_custom_title_overrides_caveat() -> None:
    grid = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=[5, 10], columns=[20, 40])
    fig = plotting.parameter_heatmap(sharpe_grid=grid, title="Custom")
    assert fig.layout.title.text == "Custom"


def test_report_dashboard_four_traces_log_top_panel(bt_artifacts: tuple) -> None:
    _, strat, bench = bt_artifacts
    fig = plotting.report_dashboard(
        strategy_equity=strat.equity,
        benchmark_equity=bench.equity,
        strategy_returns=strat.daily_returns,
    )
    assert len(fig.data) == 4
    assert fig.layout.yaxis.type == "log"
    assert fig.layout.yaxis2.tickformat == ".0%"
    assert fig.layout.height == 900
    assert "Backtest Report" in fig.layout.title.text
