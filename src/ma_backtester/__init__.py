"""Vectorised moving-average crossover backtester.

The public surface is re-exported here so callers can write::

    from ma_backtester import run_backtest, StrategyConfig, CostConfig

Submodule imports remain available for fine-grained access.
"""

from ma_backtester.backtester import run_backtest, run_buy_and_hold, sweep
from ma_backtester.benchmark import compare_strategies
from ma_backtester.config import (
    CostConfig,
    RunConfig,
    StrategyConfig,
    SweepConfig,
    WalkForwardConfig,
)
from ma_backtester.costs import FixedBpsCost, zero_cost_model
from ma_backtester.data_snooping import (
    deflated_sharpe_ratio,
    effective_number_of_trials,
    probabilistic_sharpe_ratio,
)
from ma_backtester.metrics import compute_metrics_table, sharpe_ratio
from ma_backtester.results import (
    BacktestResult,
    BenchmarkComparison,
    DeflatedSharpeResult,
    MetricsTable,
    WalkForwardFold,
    WalkForwardResult,
)
from ma_backtester.walk_forward import run_walk_forward

__version__ = "0.1.0"

__all__ = [
    "BacktestResult",
    "BenchmarkComparison",
    "CostConfig",
    "DeflatedSharpeResult",
    "FixedBpsCost",
    "MetricsTable",
    "RunConfig",
    "StrategyConfig",
    "SweepConfig",
    "WalkForwardConfig",
    "WalkForwardFold",
    "WalkForwardResult",
    "__version__",
    "compare_strategies",
    "compute_metrics_table",
    "deflated_sharpe_ratio",
    "effective_number_of_trials",
    "probabilistic_sharpe_ratio",
    "run_backtest",
    "run_buy_and_hold",
    "run_walk_forward",
    "sharpe_ratio",
    "sweep",
    "zero_cost_model",
]
