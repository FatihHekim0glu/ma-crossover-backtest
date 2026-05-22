#!/usr/bin/env bash
# Regenerate the README headline numbers and figures from scratch.
#
#   bash scripts/reproduce.sh
#
# Outputs:
#   - results/headline.json        Strategy + benchmark metrics for SMA(50,200) on SPY 2010-2024
#   - results/comparison.json      Jensen alpha / beta / IR / Sharpe-diff
#   - docs/assets/equity_curve.png Refreshed equity curve image
#
# Exit codes:
#   0 - success, results match the README within rounding
#   1 - any step failed

set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> uv sync"
uv sync --all-extras --dev

echo "==> running headline backtest"
uv run python - <<'PY'
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

from ma_backtester import (
    CostConfig,
    FixedBpsCost,
    StrategyConfig,
    compare_strategies,
    compute_metrics_table,
    run_backtest,
    run_buy_and_hold,
)
from ma_backtester.data import load_close

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

close = load_close("SPY", start="2010-01-01", end="2024-12-31")
strategy = StrategyConfig(fast_window=50, slow_window=200)
cost = FixedBpsCost(CostConfig(per_side_bps=5.0))

strat = run_backtest(close=close, strategy_config=strategy, cost_model=cost)
bench = run_buy_and_hold(close=close, cost_model=cost)

strat_metrics = asdict(compute_metrics_table(
    equity=strat.equity, daily_returns=strat.daily_returns,
    positions=strat.positions, trades=strat.trades,
))
bench_metrics = asdict(compute_metrics_table(
    equity=bench.equity, daily_returns=bench.daily_returns,
    positions=bench.positions, trades=bench.trades,
))
comparison = asdict(compare_strategies(
    strategy_returns=strat.daily_returns,
    benchmark_returns=bench.daily_returns,
))

headline = {
    "ticker": "SPY",
    "start": "2010-01-01",
    "end": "2024-12-31",
    "strategy": {"fast": 50, "slow": 200, "cost_bps_per_side": 5.0},
    "strategy_metrics": strat_metrics,
    "buy_and_hold_metrics": bench_metrics,
}
(results_dir / "headline.json").write_text(json.dumps(headline, indent=2, default=str))
(results_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, default=str))

# README claims (Strategy / Buy&Hold), tolerance 5e-4 absolute:
EXPECTED = {
    "strategy_cagr":     0.0951,
    "buy_and_hold_cagr": 0.1370,
    "strategy_sharpe":   0.7332,
    "buy_and_hold_sharpe": 0.8386,
    "alpha_annual":      0.0084,
    "alpha_p_value":     0.6926,
}
TOL = 5e-4

checks = [
    ("strategy_cagr",      strat_metrics["cagr"]),
    ("buy_and_hold_cagr",  bench_metrics["cagr"]),
    ("strategy_sharpe",    strat_metrics["sharpe"]),
    ("buy_and_hold_sharpe", bench_metrics["sharpe"]),
    ("alpha_annual",       comparison["alpha_annual"]),
    ("alpha_p_value",      comparison["alpha_p_value"]),
]

mismatches = []
for name, observed in checks:
    expected = EXPECTED[name]
    delta = abs(observed - expected)
    status = "OK" if delta < TOL else "DRIFT"
    print(f"  {status:5s}  {name:22s} observed={observed:.4f}  expected={expected:.4f}  delta={delta:.2e}")
    if delta >= TOL:
        mismatches.append(name)

if mismatches:
    print(f"\nFAIL: {len(mismatches)} metric(s) drifted beyond {TOL}: {mismatches}", file=sys.stderr)
    sys.exit(1)

print("\nAll README headline metrics reproduce within tolerance.")
PY

echo "==> refreshing equity curve PNG"
uv run python scripts/generate_assets.py

echo "==> done; results/ + docs/assets/ updated"
