"""Walk-forward harness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ma_backtester.config import StrategyConfig, SweepConfig, WalkForwardConfig
from ma_backtester.walk_forward import _select_best, generate_folds, run_walk_forward
from tests.conftest import make_gbm_series


def test_generate_folds_non_overlapping() -> None:
    prices = make_gbm_series(seed=0, n=252 * 10)  # 10 years
    folds = generate_folds(prices, WalkForwardConfig(train_years=5, test_years=1, step_years=1))
    assert len(folds) >= 4
    for i in range(1, len(folds)):
        assert folds[i].test_start > folds[i - 1].test_end


def test_generate_folds_anchored() -> None:
    """All folds start at the same train start (anchored = expanding window)."""
    prices = make_gbm_series(seed=0, n=252 * 8)
    folds = generate_folds(prices, WalkForwardConfig(train_years=3, test_years=1, step_years=1))
    train_starts = {f.train_start for f in folds}
    assert len(train_starts) == 1


def test_walk_forward_config_rejects_overlapping_oos() -> None:
    """step_years < test_years means OOS windows overlap — refuse at construction."""
    with pytest.raises(ValueError, match="non-overlapping"):
        WalkForwardConfig(train_years=5, test_years=2, step_years=1)


def test_walk_forward_config_rejects_zero_years() -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        WalkForwardConfig(train_years=0, test_years=1, step_years=1)


def test_generate_folds_rejects_non_datetime_index() -> None:
    s = pd.Series(np.arange(2000, dtype=float))
    with pytest.raises(TypeError, match="DatetimeIndex"):
        generate_folds(s, WalkForwardConfig())


def test_select_best_raises_when_all_nan() -> None:
    sweep = SweepConfig(fast_windows=(5,), slow_windows=(20,))
    nan_map = {StrategyConfig(fast_window=5, slow_window=20): float("nan")}
    with pytest.raises(ValueError, match="All sweep results were NaN"):
        _select_best(nan_map, sweep, use_neighbourhood=True)


def test_select_best_picks_max_not_min() -> None:
    """Mutation guard: if someone swaps max() for min(), this catches it."""
    sweep = SweepConfig(fast_windows=(5, 10), slow_windows=(20, 50))
    sharpes = {
        StrategyConfig(fast_window=5, slow_window=20): -1.0,
        StrategyConfig(fast_window=5, slow_window=50): 2.5,  # the winner
        StrategyConfig(fast_window=10, slow_window=20): 0.5,
        StrategyConfig(fast_window=10, slow_window=50): 1.2,
    }
    chosen = _select_best(sharpes, sweep, use_neighbourhood=False)
    assert (chosen.fast_window, chosen.slow_window) == (5, 50)


def test_select_best_deterministic_tie_break() -> None:
    """When multiple configs tie on Sharpe, sort lexicographically."""
    sweep = SweepConfig(fast_windows=(5, 10), slow_windows=(20, 50))
    sharpes = {
        StrategyConfig(fast_window=10, slow_window=50): 1.0,
        StrategyConfig(fast_window=5, slow_window=20): 1.0,
        StrategyConfig(fast_window=5, slow_window=50): 1.0,
        StrategyConfig(fast_window=10, slow_window=20): 1.0,
    }
    chosen = _select_best(sharpes, sweep, use_neighbourhood=False)
    # Lowest (fast, slow) wins the tie
    assert (chosen.fast_window, chosen.slow_window) == (5, 20)


def test_walk_forward_smoke() -> None:
    prices = make_gbm_series(seed=1, n=252 * 8)
    tiny_sweep = SweepConfig(fast_windows=(10, 20), slow_windows=(50, 100))
    result = run_walk_forward(
        close=prices,
        ticker="TEST",
        sweep=tiny_sweep,
        wf_config=WalkForwardConfig(train_years=3, test_years=1, step_years=1),
    )
    assert result.ticker == "TEST"
    assert len(result.folds) >= 4
    assert result.concatenated_equity is not None
    assert isinstance(result.concatenated_equity, pd.Series)
    for fold in result.folds:
        assert fold.selected_fast < fold.selected_slow
