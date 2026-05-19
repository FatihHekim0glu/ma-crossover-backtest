"""Signal generation correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ma_backtester.config import StrategyConfig
from ma_backtester.strategy import (
    buy_and_hold_position,
    generate_signal,
    signal_to_position,
)


def test_constant_price_never_signals_long(constant_prices: pd.Series) -> None:
    cfg = StrategyConfig(fast_window=10, slow_window=30)
    signal = generate_signal(constant_prices, cfg)
    # fast == slow == 100 always, so fast > slow is False everywhere
    nonzero_signal = signal.dropna()
    assert (nonzero_signal == 0).all()


def test_pure_uptrend_eventually_signals_long(linear_uptrend: pd.Series) -> None:
    cfg = StrategyConfig(fast_window=5, slow_window=20)
    signal = generate_signal(linear_uptrend, cfg)
    # After warm-up + a few bars of trend, fast SMA > slow SMA
    assert (signal.dropna() == 1).sum() > 200


def test_signal_to_position_shifts_by_one() -> None:
    raw = pd.Series([np.nan, np.nan, 0, 0, 1, 1, 0, 1], dtype="float64")
    pos = signal_to_position(raw)
    expected = pd.Series([0.0, np.nan, np.nan, 0.0, 0.0, 1.0, 1.0, 0.0]).fillna(0)
    np.testing.assert_array_equal(pos.to_numpy(), expected.to_numpy())


def test_buy_and_hold_is_always_one(seeded_gbm_prices: pd.Series) -> None:
    pos = buy_and_hold_position(seeded_gbm_prices)
    assert (pos == 1.0).all()
    assert len(pos) == len(seeded_gbm_prices)


def test_invalid_strategy_config_rejected() -> None:
    with pytest.raises(ValueError, match="strictly less"):
        StrategyConfig(fast_window=30, slow_window=10)
    with pytest.raises(ValueError, match="strictly less"):
        StrategyConfig(fast_window=20, slow_window=20)
    with pytest.raises(ValueError, match="positive"):
        StrategyConfig(fast_window=0, slow_window=10)
