"""Streamlit dashboard for the MA crossover backtester.

Pure presentation layer. All backtest logic lives in ``ma_backtester`` and is
covered by the unit suite; this file routes widget inputs to those functions
and renders the results. If you find yourself adding business logic here,
push it down into ``ma_backtester`` instead so the test suite can guard it.

Run with::

    uv run streamlit run app.py
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ma_backtester.backtester import run_backtest, run_buy_and_hold
from ma_backtester.backtester import sweep as run_sweep
from ma_backtester.benchmark import compare_strategies
from ma_backtester.config import (
    DEFAULT_COST_BPS_GRID,
    DEFAULT_SWEEP,
    DEFAULT_TICKERS,
    CostConfig,
    StrategyConfig,
    WalkForwardConfig,
)
from ma_backtester.costs import FixedBpsCost
from ma_backtester.data import DataQualityError, load_close
from ma_backtester.data_providers import (
    PolygonProvider,
    YFinanceProvider,
    make_provider,
)
from ma_backtester.data_snooping import (
    deflated_sharpe_ratio,
    effective_number_of_trials,
)
from ma_backtester.metrics import (
    annualised_volatility,
    cagr,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from ma_backtester.plotting import (
    equity_curve,
    install_theme,
    parameter_heatmap,
    underwater_drawdown,
)
from ma_backtester.walk_forward import run_walk_forward

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Page config and theme
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="MA Crossover Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
install_theme()


# --------------------------------------------------------------------------- #
# Cached data layer
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading price data...", ttl=60 * 60)
def cached_load_close(ticker: str, start: str, end: str) -> pd.Series:
    return load_close(ticker, start=start, end=end)


@st.cache_resource(show_spinner=False)
def _resolve_provider_label() -> str:
    """Return ``"polygon"`` or ``"yfinance"`` for the result-panel badge.

    Cached at the resource level so the provider isn't reconstructed on
    every widget interaction. The factory's branching cost is trivial, but
    repeated httpx-client construction for the Polygon path is not.
    """
    provider = make_provider()
    if isinstance(provider, PolygonProvider):
        provider.close()
        return "polygon"
    if isinstance(provider, YFinanceProvider):
        return "yfinance"
    return "unknown"


@st.cache_data(show_spinner=False, ttl=60 * 30)
def cached_cost_sensitivity(
    ticker: str, start: str, end: str, fast: int, slow: int, bps_grid: tuple[float, ...]
) -> pd.DataFrame:
    """Cost-sensitivity table — cached because it costs N backtests per refresh."""
    close = cached_load_close(ticker, start, end)
    cfg = StrategyConfig(fast_window=fast, slow_window=slow)
    rows: list[dict[str, float]] = []
    for bps in bps_grid:
        c = FixedBpsCost(CostConfig(per_side_bps=bps))
        r = run_backtest(close=close, strategy_config=cfg, cost_model=c)
        rows.append(
            {
                "Cost (bps/side)": bps,
                "Round-trip (bps)": bps * 2,
                "Sharpe": sharpe_ratio(r.daily_returns),
                "CAGR": cagr(r.equity),
                "Max DD": max_drawdown(r.equity),
            }
        )
    return pd.DataFrame(rows).set_index("Cost (bps/side)")


@st.cache_data(show_spinner="Running sweep...", ttl=60 * 30)
def cached_sweep(
    ticker: str, start: str, end: str, cost_bps: float
) -> tuple[dict[str, float], pd.DataFrame, str]:
    """Return per-(fast, slow) Sharpe map, returns matrix, and best key."""
    close = cached_load_close(ticker, start, end)
    cost = FixedBpsCost(CostConfig(per_side_bps=cost_bps))
    grid = DEFAULT_SWEEP.grid()
    results = run_sweep(close=close, grid=grid, cost_model=cost)
    sharpes = {
        f"{cfg.fast_window}_{cfg.slow_window}": sharpe_ratio(r.daily_returns)
        for cfg, r in results.items()
    }
    best = max(sharpes, key=lambda k: sharpes[k])
    matrix = pd.DataFrame(
        {f"{cfg.fast_window}_{cfg.slow_window}": r.daily_returns for cfg, r in results.items()}
    )
    return sharpes, matrix, best


@st.cache_data(show_spinner="Running walk-forward (this is the slow one)...", ttl=60 * 30)
def cached_walk_forward(
    ticker: str, start: str, end: str, cost_bps: float, train_years: int, test_years: int
) -> dict[str, object]:
    close = cached_load_close(ticker, start, end)
    cost = FixedBpsCost(CostConfig(per_side_bps=cost_bps))
    wf_config = WalkForwardConfig(train_years=train_years, test_years=test_years, step_years=1)
    result = run_walk_forward(
        close=close,
        ticker=ticker,
        sweep=DEFAULT_SWEEP,
        wf_config=wf_config,
        cost_model=cost,
    )
    return {
        "folds": [
            {
                "fold": f.fold_index,
                "train_end": f.train_end.date(),
                "test_end": f.test_end.date(),
                "fast": f.selected_fast,
                "slow": f.selected_slow,
                "IS_sharpe": f.in_sample_sharpe,
                "OOS_sharpe": f.out_of_sample_sharpe,
                "OOS_return": f.out_of_sample_return,
            }
            for f in result.folds
        ],
        "equity": result.concatenated_equity,
        "returns": result.concatenated_returns,
    }


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Configuration")

    ticker = st.selectbox(
        "Ticker",
        options=[*list(DEFAULT_TICKERS), "Other..."],
        index=0,
        help="The five default ETFs span asset classes (US large/tech/small, gold, long bonds).",
    )
    _TICKER_RE = re.compile(r"^[A-Z0-9.\-^]{1,15}$")
    if ticker == "Other...":
        ticker = st.text_input("Custom ticker", value="VTI").strip().upper()
        if not _TICKER_RE.fullmatch(ticker):
            st.error("Ticker must match [A-Z0-9.\\-^], max 15 chars (e.g. BRK-B, BF.B).")
            st.stop()

    start_date = st.date_input("Start", value=date(2010, 1, 1))
    end_date = st.date_input("End", value=date(2024, 12, 31))
    if end_date <= start_date:
        st.error("End date must be after start date.")
        st.stop()

    st.divider()
    st.caption("Strategy parameters")
    fast = st.slider("Fast window (bars)", min_value=5, max_value=100, value=50, step=5)
    slow = st.slider("Slow window (bars)", min_value=20, max_value=300, value=200, step=10)
    if fast >= slow:
        st.error("Fast window must be strictly less than slow window.")
        st.stop()

    cost_bps = st.slider(
        "Cost per side (bps)",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5,
        help="5 bps per side ≈ 10 bps round-trip — conservative for liquid US ETFs.",
    )


# --------------------------------------------------------------------------- #
# Run live backtest (cheap — no button needed)
# --------------------------------------------------------------------------- #
try:
    strategy_cfg = StrategyConfig(fast_window=fast, slow_window=slow)
    cost_model = FixedBpsCost(CostConfig(per_side_bps=cost_bps))

    close = cached_load_close(ticker, str(start_date), str(end_date))
    strat = run_backtest(close=close, strategy_config=strategy_cfg, cost_model=cost_model)
    bench = run_buy_and_hold(close=close, cost_model=cost_model)

    # --------------------------------------------------------------------------- #
    # Header
    # --------------------------------------------------------------------------- #
    st.title("Moving-Average Crossover Backtester")
    _provider_label = _resolve_provider_label()
    st.markdown(
        f"**SMA({fast}, {slow})** on **{ticker}** from "
        f"**{start_date}** to **{end_date}** with **{cost_bps} bps/side** transaction cost. "
        f"All numbers update live as you move the sliders."
    )
    st.caption(f"data: {_provider_label}")

    # --------------------------------------------------------------------------- #
    # Headline metrics
    # --------------------------------------------------------------------------- #
    def _metric_row(
        label: str, fmt: str, strat_val: float, bench_val: float
    ) -> tuple[str, str, str, str]:
        return (
            label,
            fmt.format(strat_val),
            fmt.format(bench_val),
            fmt.format(strat_val - bench_val),
        )

    metrics_df = pd.DataFrame(
        [
            _metric_row("CAGR", "{:.2%}", cagr(strat.equity), cagr(bench.equity)),
            _metric_row(
                "Annual vol",
                "{:.2%}",
                annualised_volatility(strat.daily_returns),
                annualised_volatility(bench.daily_returns),
            ),
            _metric_row(
                "Sharpe",
                "{:.3f}",
                sharpe_ratio(strat.daily_returns),
                sharpe_ratio(bench.daily_returns),
            ),
            _metric_row(
                "Sortino",
                "{:.3f}",
                sortino_ratio(strat.daily_returns),
                sortino_ratio(bench.daily_returns),
            ),
            _metric_row(
                "Max drawdown", "{:.2%}", max_drawdown(strat.equity), max_drawdown(bench.equity)
            ),
        ],
        columns=["Metric", "Strategy", "Buy & Hold", "Δ (Strat − B&H)"],  # noqa: RUF001
    )

    st.subheader("Headline metrics")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------- #
    # Equity + drawdown
    # --------------------------------------------------------------------------- #
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            equity_curve(
                strategy_equity=strat.equity,
                benchmark_equity=bench.equity,
                title="Equity curve (log scale)",
            ),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            underwater_drawdown(
                strategy_equity=strat.equity,
                benchmark_equity=bench.equity,
            ),
            use_container_width=True,
        )

    # --------------------------------------------------------------------------- #
    # Statistical comparison
    # --------------------------------------------------------------------------- #
    st.subheader("Statistical comparison")
    st.caption(
        "CAPM regression with Newey-West HAC standard errors (bandwidth via Andrews 1991), "
        "information ratio, and Memmel-corrected Jobson-Korkie Sharpe-difference test."
    )

    cmp = compare_strategies(
        strategy_returns=strat.daily_returns,
        benchmark_returns=bench.daily_returns,
    )
    cmp_dict = asdict(cmp)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Jensen's α (annual)",  # noqa: RUF001
        f"{cmp.alpha_annual:.2%}",
        help=f"t = {cmp.alpha_t_stat:.2f}, p = {cmp.alpha_p_value:.3f}",
    )
    c2.metric(
        "β vs benchmark", f"{cmp.beta:.3f}", help="< 1 means strategy is in cash some of the time"
    )
    c3.metric(
        "Information ratio", f"{cmp.information_ratio:.3f}", help=">0.5 good, >1.0 exceptional"
    )
    c4.metric(
        "Sharpe difference", f"{cmp.sharpe_diff:.3f}", help=f"p = {cmp.sharpe_diff_p_value:.3f}"
    )

    if cmp.alpha_p_value > 0.05:
        st.info(
            f"**Alpha is NOT statistically distinguishable from zero** "
            f"(p = {cmp.alpha_p_value:.3f}, HAC SEs, {cmp.hac_lags} lags). "
            "This is the methodologically honest reading."
        )

    with st.expander("Full comparison table"):
        st.dataframe(pd.Series(cmp_dict).to_frame("value"), use_container_width=True)

    # --------------------------------------------------------------------------- #
    # Cost sensitivity
    # --------------------------------------------------------------------------- #
    st.subheader("Cost sensitivity")
    sensitivity_df = cached_cost_sensitivity(
        ticker, str(start_date), str(end_date), fast, slow, tuple(DEFAULT_COST_BPS_GRID)
    )
    st.dataframe(
        sensitivity_df.style.format(
            {"Sharpe": "{:.3f}", "CAGR": "{:.2%}", "Max DD": "{:.2%}", "Round-trip (bps)": "{:.0f}"}
        ),
        use_container_width=True,
    )

    # --------------------------------------------------------------------------- #
    # Heavy analyses — gated behind buttons
    # --------------------------------------------------------------------------- #
    st.divider()
    tab_sweep, tab_wf = st.tabs(["Parameter sweep + DSR", "Walk-forward (the honest one)"])

    with tab_sweep:
        st.markdown(
            "Sweep ~320 (fast, slow) combinations on the **in-sample** window and "
            "apply the Deflated Sharpe Ratio (Bailey & López de Prado, 2014). "
            "The heatmap is *diagnostic only* — see the caption on the chart."
        )
        if st.button("Run sweep", key="run_sweep"):
            sharpes, matrix, best_key = cached_sweep(
                ticker, str(start_date), str(end_date), cost_bps
            )
            best_fast, best_slow = map(int, best_key.split("_"))
            st.success(
                f"Best in-sample: SMA({best_fast}, {best_slow}) "
                f"with Sharpe = {sharpes[best_key]:.3f}"
            )

            grid_df = pd.DataFrame(
                index=sorted(DEFAULT_SWEEP.fast_windows),
                columns=sorted(DEFAULT_SWEEP.slow_windows),
                dtype="float64",
            )
            for key, s in sharpes.items():
                f_w, s_w = map(int, key.split("_"))
                grid_df.loc[f_w, s_w] = s
            st.plotly_chart(parameter_heatmap(sharpe_grid=grid_df), use_container_width=True)

            n_eff = effective_number_of_trials(returns_matrix=matrix)
            best_returns = matrix[best_key]
            dsr = deflated_sharpe_ratio(
                daily_returns=best_returns,
                n_trials=len(sharpes),
                n_effective_trials=n_eff,
            )

            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Observed Sharpe", f"{dsr.observed_sharpe:.3f}")
            dc2.metric("Expected max under null", f"{dsr.expected_max_sharpe_under_null:.3f}")
            dc3.metric(
                "Deflated Sharpe (probability)",
                f"{dsr.deflated_sharpe:.3f}",
                delta="reject null" if dsr.can_reject_null else "cannot reject null",
            )
            st.caption(
                f"Effective trials: {dsr.n_effective_trials} of {dsr.n_trials} (PCA, 95% var). "
                f"Return distribution skew = {dsr.skew:.2f}, kurtosis = {dsr.kurtosis:.2f}."
            )

    with tab_wf:
        st.markdown(
            "Anchored expanding-window walk-forward: train on growing history, "
            "evaluate on a non-overlapping out-of-sample slice, re-optimise each year. "
            "This takes ~30s for one ticker."
        )
        wf_train = st.slider("Train window (years)", min_value=3, max_value=10, value=5)
        wf_test = st.slider("OOS window (years)", min_value=1, max_value=3, value=1)
        if st.button("Run walk-forward", key="run_wf"):
            wf = cached_walk_forward(
                ticker, str(start_date), str(end_date), cost_bps, wf_train, wf_test
            )
            folds_df = pd.DataFrame(wf["folds"])
            st.dataframe(
                folds_df.style.format(
                    {
                        "IS_sharpe": "{:.3f}",
                        "OOS_sharpe": "{:.3f}",
                        "OOS_return": "{:.2%}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            # In-sample vs out-of-sample Sharpe scatter — the diagnostic chart.
            fig = go.Figure()
            fig.add_scatter(
                x=folds_df["IS_sharpe"],
                y=folds_df["OOS_sharpe"],
                mode="markers+text",
                text=folds_df["fold"].astype(str),
                textposition="top right",
                marker={"size": 12},
            )
            lim = float(
                max(folds_df["IS_sharpe"].abs().max(), folds_df["OOS_sharpe"].abs().max()) + 0.3
            )
            fig.add_shape(type="line", x0=-lim, y0=-lim, x1=lim, y1=lim, line={"dash": "dash"})
            fig.add_annotation(
                x=lim * 0.7,
                y=lim * 0.7,
                text="points near y=x → IS Sharpe predicts OOS Sharpe<br>(strategy generalises)",
                showarrow=False,
                font={"size": 10, "color": "rgba(0,0,0,0.55)"},
                align="center",
            )
            fig.update_layout(
                title="In-sample vs out-of-sample Sharpe (one point per fold)",
                xaxis_title="In-sample Sharpe",
                yaxis_title="OOS Sharpe",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

            if wf["returns"] is not None and len(wf["returns"]) >= 30:
                dsr_oos = deflated_sharpe_ratio(
                    daily_returns=wf["returns"],
                    n_trials=DEFAULT_SWEEP.size,
                )
                wc1, wc2, wc3 = st.columns(3)
                wc1.metric("Concatenated OOS Sharpe", f"{dsr_oos.observed_sharpe:.3f}")
                wc2.metric(
                    "Expected max under null", f"{dsr_oos.expected_max_sharpe_under_null:.3f}"
                )
                wc3.metric(
                    "OOS Deflated Sharpe",
                    f"{dsr_oos.deflated_sharpe:.3f}",
                    delta="reject null" if dsr_oos.can_reject_null else "cannot reject null",
                )

            if wf["equity"] is not None:
                bench_oos = bench.equity.loc[wf["equity"].index]
                bench_oos = bench_oos / bench_oos.iloc[0] * float(wf["equity"].iloc[0])
                st.plotly_chart(
                    equity_curve(
                        strategy_equity=wf["equity"],
                        benchmark_equity=bench_oos,
                        title="Concatenated out-of-sample equity vs B&H",
                    ),
                    use_container_width=True,
                )

    # --------------------------------------------------------------------------- #
    # Footer
    # --------------------------------------------------------------------------- #
    st.divider()
    st.caption(
        "Engine: vectorised pandas with the `position = signal.shift(1)` discipline, "
        "property-tested for no-lookahead. "
        "Statistical machinery: Newey-West HAC standard errors, Memmel-corrected JK Sharpe-diff, "
        "Deflated Sharpe Ratio. See README for references."
    )
except DataQualityError as e:
    log.warning("data quality error: %s", e)
    st.error(f"Couldn't load {ticker}: {e}")
    st.stop()
except Exception:
    log.exception("unexpected error in dashboard")
    st.error(
        "Something went wrong loading data for this ticker/date range. "
        "Please try a different combination, or see the logs for details."
    )
    st.stop()
