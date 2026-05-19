"""Anchored walk-forward evaluation.

Anchored = the train window starts at the first available bar and *expands*;
the out-of-sample window slides forward by one step at a time.

For each fold:

1. Run the full strategy grid on the train slice.
2. Select the (fast, slow) pair with the highest in-sample Sharpe.
   With ``use_neighbourhood_tiebreak`` (default), break ties by the mean
   Sharpe over the 8 grid neighbours — picks plateaus rather than spikes.
3. Evaluate the selected strategy on the OOS slice with no re-tuning.

Out-of-sample slices are **non-overlapping** so the concatenated OOS series
is a clean, contiguous return series suitable for downstream statistics
(Sharpe, t-test on alpha, DSR).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ma_backtester.backtester import run_backtest
from ma_backtester.config import (
    TRADING_DAYS_PER_YEAR,
    StrategyConfig,
    SweepConfig,
    WalkForwardConfig,
)
from ma_backtester.costs import CostModel, zero_cost_model
from ma_backtester.metrics import sharpe_ratio
from ma_backtester.results import WalkForwardFold, WalkForwardResult

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_folds(close: pd.Series, config: WalkForwardConfig) -> list[_Fold]:
    """Generate anchored fold boundaries."""
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close must have a DatetimeIndex")

    train_bars = config.train_years * TRADING_DAYS_PER_YEAR
    test_bars = config.test_years * TRADING_DAYS_PER_YEAR
    step_bars = config.step_years * TRADING_DAYS_PER_YEAR

    folds: list[_Fold] = []
    test_start_idx = train_bars
    fold_idx = 0

    while test_start_idx + test_bars <= len(close):
        train_start = close.index[0]
        train_end = close.index[test_start_idx - 1]
        test_start = close.index[test_start_idx]
        test_end = close.index[min(test_start_idx + test_bars - 1, len(close) - 1)]
        folds.append(
            _Fold(
                index=fold_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        test_start_idx += step_bars
        fold_idx += 1

    return folds


def _grid_neighbour_mean_sharpe(
    sharpe_grid: pd.DataFrame,
    chosen: StrategyConfig,
) -> float:
    """Mean Sharpe of the 8 neighbours of (fast, slow) on the grid."""
    fasts = sharpe_grid.index.to_list()
    slows = sharpe_grid.columns.to_list()
    f_idx = fasts.index(chosen.fast_window)
    s_idx = slows.index(chosen.slow_window)
    values: list[float] = []
    for df in (-1, 0, 1):
        for ds in (-1, 0, 1):
            if df == 0 and ds == 0:
                continue
            ni, nj = f_idx + df, s_idx + ds
            if 0 <= ni < len(fasts) and 0 <= nj < len(slows):
                v = float(sharpe_grid.iat[ni, nj])
                if not math.isnan(v):
                    values.append(v)
    return float(np.mean(values)) if values else float("nan")


def _select_best(
    sharpes: dict[StrategyConfig, float],
    sweep: SweepConfig,
    use_neighbourhood: bool,
) -> StrategyConfig:
    valid = {cfg: s for cfg, s in sharpes.items() if not math.isnan(s)}
    if not valid:
        raise ValueError("All sweep results were NaN — check input data length")

    top_sharpe = max(valid.values())
    tied = [cfg for cfg, s in valid.items() if s == top_sharpe]
    if len(tied) == 1 or not use_neighbourhood:
        return tied[0]

    grid_df = pd.DataFrame(
        index=sorted(sweep.fast_windows), columns=sorted(sweep.slow_windows), dtype="float64"
    )
    for cfg, s in sharpes.items():
        if cfg.fast_window in grid_df.index and cfg.slow_window in grid_df.columns:
            grid_df.loc[cfg.fast_window, cfg.slow_window] = s

    return max(tied, key=lambda c: _grid_neighbour_mean_sharpe(grid_df, c))


def run_walk_forward(
    *,
    close: pd.Series,
    ticker: str,
    sweep: SweepConfig,
    wf_config: WalkForwardConfig | None = None,
    cost_model: CostModel | None = None,
    initial_cash: float = 100_000.0,
) -> WalkForwardResult:
    wf_config = wf_config or WalkForwardConfig()
    cost_model = cost_model or zero_cost_model()
    grid = sweep.grid()

    folds_meta = generate_folds(close, wf_config)
    fold_results: list[WalkForwardFold] = []
    concatenated_returns_chunks: list[pd.Series] = []
    concatenated_equity_chunks: list[pd.Series] = []
    running_equity = initial_cash

    for fold in folds_meta:
        train_slice = close.loc[fold.train_start : fold.train_end]
        test_slice = close.loc[fold.test_start : fold.test_end]
        if len(test_slice) < 10:
            continue

        in_sample_sharpes: dict[StrategyConfig, float] = {}
        for cfg in grid:
            res = run_backtest(
                close=train_slice,
                strategy_config=cfg,
                cost_model=cost_model,
                initial_cash=initial_cash,
            )
            in_sample_sharpes[cfg] = sharpe_ratio(res.daily_returns)

        try:
            chosen = _select_best(in_sample_sharpes, sweep, wf_config.use_neighbourhood_tiebreak)
        except ValueError as exc:
            _log.warning("fold %d skipped (%s)", fold.index, exc)
            continue

        test_result = run_backtest(
            close=test_slice,
            strategy_config=chosen,
            cost_model=cost_model,
            initial_cash=running_equity,
        )

        fold_results.append(
            WalkForwardFold(
                fold_index=fold.index,
                train_start=fold.train_start,
                train_end=fold.train_end,
                test_start=fold.test_start,
                test_end=fold.test_end,
                selected_fast=chosen.fast_window,
                selected_slow=chosen.slow_window,
                in_sample_sharpe=float(in_sample_sharpes[chosen]),
                out_of_sample_sharpe=sharpe_ratio(test_result.daily_returns),
                out_of_sample_return=float(test_result.equity.iloc[-1] / running_equity - 1.0),
                test_equity=test_result.equity,
                test_positions=test_result.positions,
            )
        )
        concatenated_returns_chunks.append(test_result.daily_returns)
        concatenated_equity_chunks.append(test_result.equity)
        running_equity = float(test_result.equity.iloc[-1])

    concatenated_returns = (
        pd.concat(concatenated_returns_chunks) if concatenated_returns_chunks else None
    )
    concatenated_equity = (
        pd.concat(concatenated_equity_chunks) if concatenated_equity_chunks else None
    )

    return WalkForwardResult(
        ticker=ticker,
        folds=fold_results,
        concatenated_equity=concatenated_equity,
        concatenated_returns=concatenated_returns,
    )
