"""The no-lookahead invariant — the project's central correctness claim.

If any of these tests fails, the backtester is silently consuming future
information. Do not weaken these tests to make them pass; fix the engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ma_backtester.backtester import run_backtest
from ma_backtester.config import StrategyConfig
from ma_backtester.strategy import generate_signal, signal_to_position
from tests.conftest import make_gbm_series, price_series_strategy

_STRATEGY = StrategyConfig(fast_window=10, slow_window=30)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    prices=price_series_strategy(min_size=120, max_size=300),
    t=st.integers(min_value=60, max_value=119),
)
def test_prefix_determinism_positions(prices: pd.Series, t: int) -> None:
    """Truncating input at bar t must not change positions up to bar t.

    This is the gold-standard no-lookahead test: any leak from the future
    will cause the prefix run to differ from the full run on the shared
    bars.
    """
    full = run_backtest(close=prices, strategy_config=_STRATEGY)
    prefix = run_backtest(close=prices.iloc[: t + 1], strategy_config=_STRATEGY)

    pd.testing.assert_series_equal(
        prefix.positions,
        full.positions.iloc[: t + 1],
        check_names=False,
    )


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    prices=price_series_strategy(min_size=120, max_size=300),
    t=st.integers(min_value=60, max_value=119),
)
def test_future_perturbation_invariance(prices: pd.Series, t: int) -> None:
    """Mutating prices strictly after bar t must not change positions up to t."""
    rng = np.random.default_rng(int(prices.iloc[0] * 1000) % (2**31))
    perturbed = prices.copy()
    perturbed.iloc[t + 1 :] = perturbed.iloc[t + 1 :] * (
        1.0 + rng.normal(0, 0.05, size=len(perturbed) - t - 1)
    )

    full = run_backtest(close=prices, strategy_config=_STRATEGY)
    perturbed_run = run_backtest(close=perturbed, strategy_config=_STRATEGY)

    pd.testing.assert_series_equal(
        full.positions.iloc[: t + 1],
        perturbed_run.positions.iloc[: t + 1],
        check_names=False,
    )


@pytest.mark.parametrize("k", [1, 3, 7])
def test_shift_equivariance(k: int) -> None:
    """Shifting the input by k bars must shift positions by exactly k bars."""
    prices = make_gbm_series(seed=7, n=300)
    base = run_backtest(close=prices, strategy_config=_STRATEGY).positions

    shifted_prices = prices.shift(k)
    shifted_prices.iloc[:k] = prices.iloc[0]
    shifted = run_backtest(close=shifted_prices, strategy_config=_STRATEGY).positions

    # After the first k bars of the shifted series (warm-up artefact), the
    # shifted positions should match the base shifted by k.
    expected = base.shift(k).fillna(0.0)
    np.testing.assert_array_equal(
        shifted.iloc[k + _STRATEGY.slow_window :].to_numpy(),
        expected.iloc[k + _STRATEGY.slow_window :].to_numpy(),
    )


def test_warmup_period_is_flat() -> None:
    """The first slow_window-1 bars must have zero position (warm-up)."""
    prices = make_gbm_series(seed=1, n=200)
    cfg = StrategyConfig(fast_window=10, slow_window=30)

    signal = generate_signal(prices, cfg)
    position = signal_to_position(signal)

    assert (position.iloc[: cfg.slow_window] == 0).all()
    # Signal itself is NaN during true warm-up
    assert signal.iloc[: cfg.slow_window - 1].isna().all()


def test_no_negative_shift_in_strategy() -> None:
    """Signal at bar t must depend only on close[0..t]."""
    prices = make_gbm_series(seed=99, n=200)
    cfg = StrategyConfig(fast_window=10, slow_window=30)

    full_signal = generate_signal(prices, cfg)
    # Truncate at bar 100 and confirm signal[100] matches full signal[100]
    truncated_signal = generate_signal(prices.iloc[:101], cfg)
    assert (
        pd.isna(full_signal.iloc[100]) and pd.isna(truncated_signal.iloc[100])
    ) or full_signal.iloc[100] == truncated_signal.iloc[100]
