"""Generate static documentation assets (equity-curve PNG, dashboard placeholder).

Run with::

    uv run python scripts/generate_assets.py

Requires the ``notebooks`` optional dependency group (provides ``kaleido``),
installed via ``uv sync --all-extras``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ma_backtester import backtester, config, costs, data, plotting


def generate_equity_curve_png(output_dir: Path) -> None:
    """Run a default SPY backtest and write equity-curve / dashboard PNGs.

    Parameters
    ----------
    output_dir : Path
        Directory to write ``equity_curve.png`` and ``dashboard.png`` into.
        Must already exist.
    """
    close = data.load_close("SPY", start="2010-01-01", end="2024-12-31")

    cost_model = costs.FixedBpsCost(config.CostConfig(per_side_bps=5.0))

    strategy_result = backtester.run_backtest(
        close=close,
        strategy_config=config.StrategyConfig(fast_window=50, slow_window=200),
        cost_model=cost_model,
    )
    benchmark_result = backtester.run_buy_and_hold(
        close=close,
        cost_model=cost_model,
    )

    fig = plotting.equity_curve(
        strategy_equity=strategy_result.equity,
        benchmark_equity=benchmark_result.equity,
        title="SPY 50/200 MA Crossover vs Buy & Hold (2010-2024, 5 bps/side)",
    )

    equity_path = output_dir / "equity_curve.png"
    dashboard_path = output_dir / "dashboard.png"

    fig.write_image(equity_path, width=1200, height=600, scale=2)
    shutil.copyfile(equity_path, dashboard_path)


if __name__ == "__main__":
    output_dir: Path = Path(__file__).resolve().parent.parent / "docs" / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_equity_curve_png(output_dir)
    print(f"Wrote {output_dir / 'equity_curve.png'}")
    print(f"Wrote {output_dir / 'dashboard.png'}")
