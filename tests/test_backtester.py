"""End-to-end backtester invariants on synthetic series."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings

from ma_backtester.backtester import run_backtest, run_buy_and_hold
from ma_backtester.config import CostConfig, StrategyConfig
from ma_backtester.costs import FixedBpsCost
from ma_backtester.metrics import sharpe_ratio
from tests.conftest import price_series_strategy

_STRATEGY = StrategyConfig(fast_window=10, slow_window=30)


def test_buy_and_hold_matches_price_ratio(seeded_gbm_prices: pd.Series) -> None:
    result = run_buy_and_hold(close=seeded_gbm_prices, initial_cash=1.0)
    expected_total_return = float(seeded_gbm_prices.iloc[-1] / seeded_gbm_prices.iloc[0] - 1.0)
    actual_total_return = float(result.equity.iloc[-1] - 1.0)
    assert actual_total_return == pytest.approx(expected_total_return, rel=1e-9)


def test_constant_prices_yield_constant_equity(constant_prices: pd.Series) -> None:
    """No price movement -> no PnL, no trades."""
    result = run_backtest(close=constant_prices, strategy_config=_STRATEGY)
    assert (result.equity == result.equity.iloc[0]).all()
    assert result.trades.empty
    assert (result.gross_returns == 0.0).all()


def test_positions_are_always_in_long_flat_domain(seeded_gbm_prices: pd.Series) -> None:
    result = run_backtest(close=seeded_gbm_prices, strategy_config=_STRATEGY)
    unique = set(result.positions.unique().tolist())
    assert unique.issubset({0.0, 1.0})


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(prices=price_series_strategy(min_size=200, max_size=400))
def test_cost_monotonicity(prices: pd.Series) -> None:
    """Doubling the cost can only hurt the Sharpe."""
    low_cost = run_backtest(
        close=prices,
        strategy_config=_STRATEGY,
        cost_model=FixedBpsCost(CostConfig(per_side_bps=2.0)),
    )
    high_cost = run_backtest(
        close=prices,
        strategy_config=_STRATEGY,
        cost_model=FixedBpsCost(CostConfig(per_side_bps=20.0)),
    )
    s_low = sharpe_ratio(low_cost.daily_returns)
    s_high = sharpe_ratio(high_cost.daily_returns)
    # Either both NaN, or high <= low
    if math.isnan(s_low) and math.isnan(s_high):
        return
    assert s_high <= s_low + 1e-9


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(prices=price_series_strategy(min_size=200, max_size=400))
def test_scale_invariance(prices: pd.Series) -> None:
    """Multiplying all prices by alpha leaves positions and returns unchanged."""
    base = run_backtest(close=prices, strategy_config=_STRATEGY)
    scaled = run_backtest(close=prices * 3.7, strategy_config=_STRATEGY)
    pd.testing.assert_series_equal(base.positions, scaled.positions, check_names=False)
    pd.testing.assert_series_equal(
        base.daily_returns, scaled.daily_returns, check_names=False, rtol=1e-12, atol=1e-12
    )


def test_equity_stays_positive(seeded_gbm_prices: pd.Series) -> None:
    result = run_backtest(close=seeded_gbm_prices, strategy_config=_STRATEGY)
    assert (result.equity > 0).all()


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(prices=price_series_strategy(min_size=150, max_size=300))
def test_backtest_idempotence_and_input_unmutated(prices: pd.Series) -> None:
    """Repeated calls yield identical results and never mutate the input.

    Guards against any future regression that introduces module-level
    caching, in-place fillna, or RNG state leakage.
    """
    snapshot = prices.copy()
    r1 = run_backtest(close=prices, strategy_config=_STRATEGY)
    r2 = run_backtest(close=prices, strategy_config=_STRATEGY)
    pd.testing.assert_series_equal(r1.equity, r2.equity, check_names=False)
    pd.testing.assert_series_equal(r1.positions, r2.positions, check_names=False)
    pd.testing.assert_series_equal(prices, snapshot)


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(prices=price_series_strategy(min_size=150, max_size=300))
def test_buy_and_hold_terminal_equity_equals_price_ratio(prices: pd.Series) -> None:
    """Under zero-cost B&H, equity[-1]/equity[0] must equal price[-1]/price[0] exactly."""
    res = run_buy_and_hold(close=prices, initial_cash=100_000.0)
    expected = float(prices.iloc[-1] / prices.iloc[0])
    actual = float(res.equity.iloc[-1] / res.equity.iloc[0])
    assert actual == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_trades_have_consistent_dates(seeded_gbm_prices: pd.Series) -> None:
    result = run_backtest(close=seeded_gbm_prices, strategy_config=_STRATEGY)
    if not result.trades.empty:
        assert (result.trades["exit_date"] >= result.trades["entry_date"]).all()
        assert (result.trades["bars_held"] > 0).all()
