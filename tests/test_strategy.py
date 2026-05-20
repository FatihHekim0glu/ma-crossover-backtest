"""Signal generation correctness."""

from __future__ import annotations

from datetime import date as _date

import numpy as np
import pandas as pd
import pytest

from ma_backtester.config import (
    CostConfig as _CostConfig,
)
from ma_backtester.config import (
    RunConfig as _RunConfig,
)
from ma_backtester.config import (
    StrategyConfig,
)
from ma_backtester.config import (
    SweepConfig as _SweepConfig,
)
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


def test_generate_signal_rejects_non_series() -> None:
    """Defensive guard: numpy array or list must not silently pass through."""
    import numpy as np

    cfg = StrategyConfig(fast_window=10, slow_window=30)
    with pytest.raises(TypeError, match="pandas Series"):
        generate_signal(np.arange(100, dtype=float), cfg)  # type: ignore[arg-type]


def test_run_config_validates_inputs() -> None:
    """RunConfig.__post_init__ rejects empty ticker / reversed dates / non-positive cash."""
    valid_strategy = StrategyConfig(fast_window=10, slow_window=30)
    valid_cost = _CostConfig(per_side_bps=5.0)

    with pytest.raises(ValueError, match="non-empty"):
        _RunConfig(
            ticker="  ",
            start=_date(2010, 1, 1),
            end=_date(2020, 1, 1),
            strategy=valid_strategy,
            cost=valid_cost,
        )

    with pytest.raises(ValueError, match="after start"):
        _RunConfig(
            ticker="SPY",
            start=_date(2020, 1, 1),
            end=_date(2010, 1, 1),
            strategy=valid_strategy,
            cost=valid_cost,
        )

    with pytest.raises(ValueError, match="initial_cash"):
        _RunConfig(
            ticker="SPY",
            start=_date(2010, 1, 1),
            end=_date(2020, 1, 1),
            strategy=valid_strategy,
            cost=valid_cost,
            initial_cash=0.0,
        )


def test_sweep_config_rejects_empty_grid() -> None:
    """SweepConfig.__post_init__ refuses degenerate grids."""
    with pytest.raises(ValueError, match="non-empty"):
        _SweepConfig(fast_windows=(), slow_windows=(20,))
    with pytest.raises(ValueError, match="empty grid"):
        # fast >= slow for every pair -> no usable configs
        _SweepConfig(fast_windows=(50,), slow_windows=(10,))
