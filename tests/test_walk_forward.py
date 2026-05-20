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


def test_select_best_neighbourhood_tiebreak_picks_plateau() -> None:
    """The neighbourhood tie-break (the default) should prefer plateaus over spikes.

    The plateau winner sits at (10, 50) in the middle of the grid with
    eight 0.9-Sharpe neighbours. The spike sits at (50, 200) in the far
    corner with all-zero neighbours. Both score 1.0 in-sample, so the
    tie-break must come down to neighbourhood mean.
    """
    sweep = SweepConfig(
        fast_windows=(5, 10, 20, 30, 50),
        slow_windows=(30, 50, 100, 150, 200),
    )
    sharpes: dict[StrategyConfig, float] = dict.fromkeys(sweep.grid(), 0.0)
    sharpes[StrategyConfig(fast_window=10, slow_window=50)] = 1.0  # plateau winner
    sharpes[StrategyConfig(fast_window=50, slow_window=200)] = 1.0  # isolated spike
    plateau_neighbours = {
        (5, 30),
        (5, 50),
        (5, 100),
        (10, 30),
        (10, 100),
        (20, 30),
        (20, 50),
        (20, 100),
    }
    for cfg in sweep.grid():
        if (cfg.fast_window, cfg.slow_window) in plateau_neighbours:
            sharpes[cfg] = 0.9
    chosen = _select_best(sharpes, sweep, use_neighbourhood=True)
    assert (chosen.fast_window, chosen.slow_window) == (10, 50)


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


# --------------------------------------------------------------------------- #
# Coverage push — edge cases in fold logic + selection (cycle 6)
# --------------------------------------------------------------------------- #


def test_grid_neighbour_mean_sharpe_all_nan_neighbours() -> None:
    """Tied config whose 8 neighbours are NaN: deterministic lex tie-break still wins."""
    sweep = SweepConfig(fast_windows=(5, 10, 20), slow_windows=(20, 50, 100))
    sharpes = dict.fromkeys(sweep.grid(), float("nan"))
    sharpes[StrategyConfig(fast_window=5, slow_window=20)] = 1.0
    sharpes[StrategyConfig(fast_window=20, slow_window=100)] = 1.0
    chosen = _select_best(sharpes, sweep, use_neighbourhood=True)
    # Both candidates have NaN neighbourhood mean; lex order picks (5, 20).
    assert (chosen.fast_window, chosen.slow_window) == (5, 20)


def test_run_walk_forward_skips_fold_when_all_nan(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If _select_best raises (all-NaN grid), fold is skipped + warning logged."""
    import ma_backtester.walk_forward as wf_mod

    monkeypatch.setattr(wf_mod, "sharpe_ratio", lambda r: float("nan"))
    prices = make_gbm_series(seed=3, n=252 * 7)
    sweep = SweepConfig(fast_windows=(10,), slow_windows=(50,))
    with caplog.at_level("WARNING", logger="ma_backtester.walk_forward"):
        result = run_walk_forward(close=prices, ticker="X", sweep=sweep)
    assert result.folds == []
    assert any("skipped" in r.message for r in caplog.records)


def test_run_walk_forward_default_cost_model_runs() -> None:
    """cost_model=None falls through to zero_cost_model without error."""
    prices = make_gbm_series(seed=4, n=252 * 7)
    sweep = SweepConfig(fast_windows=(10,), slow_windows=(50,))
    result = run_walk_forward(close=prices, ticker="X", sweep=sweep, cost_model=None)
    assert result.concatenated_equity is not None


def test_run_walk_forward_empty_folds_returns_none_concats() -> None:
    """Series shorter than one train+test window → 0 folds, both concats are None."""
    prices = make_gbm_series(seed=5, n=252 * 4)  # 4y < 5y train default
    sweep = SweepConfig(fast_windows=(10,), slow_windows=(50,))
    result = run_walk_forward(close=prices, ticker="X", sweep=sweep)
    assert result.folds == []
    assert result.concatenated_equity is None
    assert result.concatenated_returns is None


def test_run_walk_forward_deterministic() -> None:
    """Two identical runs produce identical fold selections and equity."""
    prices = make_gbm_series(seed=6, n=252 * 8)
    sweep = SweepConfig(fast_windows=(10, 20), slow_windows=(50, 100))
    r1 = run_walk_forward(close=prices, ticker="X", sweep=sweep)
    r2 = run_walk_forward(close=prices, ticker="X", sweep=sweep)
    assert [(f.selected_fast, f.selected_slow) for f in r1.folds] == [
        (f.selected_fast, f.selected_slow) for f in r2.folds
    ]
    assert r1.concatenated_equity is not None
    assert r2.concatenated_equity is not None
    pd.testing.assert_series_equal(r1.concatenated_equity, r2.concatenated_equity)
