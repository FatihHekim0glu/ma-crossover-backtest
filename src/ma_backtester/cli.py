"""Typer CLI for the backtester.

Three commands:

    ma-backtester run         # single backtest, prints metrics table
    ma-backtester sweep       # parameter sweep, prints in-sample heatmap stats
    ma-backtester walk-forward  # anchored walk-forward with DSR adjustment
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from ma_backtester.backtester import run_backtest, run_buy_and_hold
from ma_backtester.backtester import sweep as run_sweep
from ma_backtester.benchmark import compare_strategies
from ma_backtester.config import (
    DEFAULT_SWEEP,
    CostConfig,
    StrategyConfig,
    WalkForwardConfig,
)
from ma_backtester.costs import FixedBpsCost
from ma_backtester.data import load_close
from ma_backtester.data_snooping import deflated_sharpe_ratio, effective_number_of_trials
from ma_backtester.metrics import compute_metrics_table, sharpe_ratio
from ma_backtester.walk_forward import run_walk_forward

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Vectorised MA-crossover backtester with honest evaluation.",
)
console = Console()


def _print_metrics(title: str, metrics_dict: dict[str, float | int]) -> None:
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            if any(k in key for k in ("return", "drawdown", "rate", "vol", "turnover")):
                cell = f"{value:.2%}"
            else:
                cell = f"{value:.4f}"
        else:
            cell = str(value)
        table.add_row(key, cell)
    console.print(table)


@app.command()
def run(
    ticker: str = typer.Option("SPY", "--ticker", "-t"),
    fast: int = typer.Option(20, "--fast", "-f"),
    slow: int = typer.Option(50, "--slow", "-s"),
    start: str = typer.Option("2010-01-01", "--start"),
    end: str = typer.Option("2024-12-31", "--end"),
    cost_bps: float = typer.Option(5.0, "--cost-bps", help="Per-side cost in basis points"),
    initial_cash: float = typer.Option(100_000.0, "--cash"),
) -> None:
    """Single backtest with side-by-side comparison vs buy-and-hold."""
    console.print(f"[bold]Loading[/bold] {ticker} {start} -> {end}")
    close = load_close(ticker, start=start, end=end)

    strategy = StrategyConfig(fast_window=fast, slow_window=slow)
    cost_model = FixedBpsCost(CostConfig(per_side_bps=cost_bps))

    strat = run_backtest(
        close=close,
        strategy_config=strategy,
        cost_model=cost_model,
        initial_cash=initial_cash,
    )
    bench = run_buy_and_hold(close=close, cost_model=cost_model, initial_cash=initial_cash)

    strat_metrics = compute_metrics_table(
        equity=strat.equity,
        daily_returns=strat.daily_returns,
        positions=strat.positions,
        trades=strat.trades,
    )
    bench_metrics = compute_metrics_table(
        equity=bench.equity,
        daily_returns=bench.daily_returns,
        positions=bench.positions,
        trades=bench.trades,
    )

    _print_metrics(f"{ticker} — SMA({fast},{slow}) — {cost_bps}bps/side", asdict(strat_metrics))
    _print_metrics(f"{ticker} — Buy & Hold benchmark", asdict(bench_metrics))

    comparison = compare_strategies(
        strategy_returns=strat.daily_returns,
        benchmark_returns=bench.daily_returns,
    )
    _print_metrics("Strategy vs Buy & Hold", asdict(comparison))


@app.command()
def sweep(
    ticker: str = typer.Option("SPY", "--ticker", "-t"),
    start: str = typer.Option("2010-01-01", "--start"),
    end: str = typer.Option("2024-12-31", "--end"),
    cost_bps: float = typer.Option(5.0, "--cost-bps"),
    output_dir: Path = typer.Option(Path("results"), "--output-dir"),
) -> None:
    """Run the default 20x20 grid and report best in-sample + DSR."""
    output_dir.mkdir(parents=True, exist_ok=True)
    close = load_close(ticker, start=start, end=end)
    cost_model = FixedBpsCost(CostConfig(per_side_bps=cost_bps))

    grid = DEFAULT_SWEEP.grid()
    console.print(f"[bold]Sweeping {len(grid)} configs[/bold] on {ticker}")
    results = run_sweep(close=close, grid=grid, cost_model=cost_model)

    sharpes = {cfg: sharpe_ratio(res.daily_returns) for cfg, res in results.items()}
    best_cfg = max(sharpes, key=lambda c: sharpes[c])
    best_res = results[best_cfg]

    console.print(
        f"Best in-sample: SMA({best_cfg.fast_window},{best_cfg.slow_window}) "
        f"Sharpe={sharpes[best_cfg]:.3f}"
    )

    returns_matrix = pd.DataFrame(
        {f"{cfg.fast_window}_{cfg.slow_window}": res.daily_returns for cfg, res in results.items()}
    )
    n_eff = effective_number_of_trials(returns_matrix=returns_matrix)
    console.print(f"Effective number of trials (PCA, 95% var): {n_eff} of {len(grid)}")

    dsr = deflated_sharpe_ratio(
        daily_returns=best_res.daily_returns,
        n_trials=len(grid),
        n_effective_trials=n_eff,
    )
    _print_metrics("Deflated Sharpe Ratio", asdict(dsr))


@app.command(name="walk-forward")
def walk_forward(
    ticker: str = typer.Option("SPY", "--ticker", "-t"),
    start: str = typer.Option("2005-01-01", "--start"),
    end: str = typer.Option("2024-12-31", "--end"),
    cost_bps: float = typer.Option(5.0, "--cost-bps"),
    train_years: int = typer.Option(5, "--train-years"),
    test_years: int = typer.Option(1, "--test-years"),
) -> None:
    """Anchored walk-forward, then DSR over the concatenated OOS series."""
    close = load_close(ticker, start=start, end=end)
    cost_model = FixedBpsCost(CostConfig(per_side_bps=cost_bps))
    wf_config = WalkForwardConfig(train_years=train_years, test_years=test_years)

    result = run_walk_forward(
        close=close,
        ticker=ticker,
        sweep=DEFAULT_SWEEP,
        wf_config=wf_config,
        cost_model=cost_model,
    )

    table = Table(title=f"{ticker} Walk-Forward Folds", show_header=True, header_style="bold")
    for col in (
        "fold",
        "train_end",
        "test_end",
        "fast",
        "slow",
        "IS Sharpe",
        "OOS Sharpe",
        "OOS return",
    ):
        table.add_column(col)
    for f in result.folds:
        table.add_row(
            str(f.fold_index),
            str(f.train_end.date()),
            str(f.test_end.date()),
            str(f.selected_fast),
            str(f.selected_slow),
            f"{f.in_sample_sharpe:.3f}",
            f"{f.out_of_sample_sharpe:.3f}",
            f"{f.out_of_sample_return:.2%}",
        )
    console.print(table)

    if result.concatenated_returns is not None and len(result.concatenated_returns) >= 30:
        dsr = deflated_sharpe_ratio(
            daily_returns=result.concatenated_returns,
            n_trials=DEFAULT_SWEEP.size,
        )
        _print_metrics("Concatenated OOS — Deflated Sharpe", asdict(dsr))
