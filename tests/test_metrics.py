"""Performance metrics — KIKO tests on hand-built series."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ma_backtester.metrics import (
    annualised_volatility,
    average_drawdown,
    cagr,
    calmar_ratio,
    max_drawdown,
    max_drawdown_duration,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)


def _series(values: list[float]) -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.Series(values, index=idx, dtype="float64")


def test_total_return_simple_case() -> None:
    equity = _series([100, 110, 121])
    assert total_return(equity) == pytest.approx(0.21)


def test_cagr_for_one_year_doubling() -> None:
    equity = _series([100.0] * 252 + [200.0])
    # Equity is flat then jumps; CAGR uses geometric formula over N-1 bars
    val = cagr(equity)
    assert math.isfinite(val)
    assert val > 0


def test_sharpe_zero_for_zero_returns() -> None:
    rets = _series([0.0] * 100)
    assert math.isnan(sharpe_ratio(rets))


def test_sharpe_is_positive_for_positive_drift() -> None:
    rng = np.random.default_rng(42)
    rets = pd.Series(rng.normal(0.001, 0.01, 1000))
    s = sharpe_ratio(rets)
    assert s > 0


def test_max_drawdown_known_path() -> None:
    equity = _series([100, 120, 90, 80, 110])
    assert max_drawdown(equity) == pytest.approx(-1 / 3, rel=1e-9)


def test_max_drawdown_zero_for_monotone() -> None:
    equity = _series([100, 110, 120, 130])
    assert max_drawdown(equity) == pytest.approx(0.0)


def test_drawdown_duration() -> None:
    # 100 -> 80 -> 100 -> 110 -> 100 -> 115
    # Drawdown runs (dd < 0): indices 1,2,3 (length 3), then 6,7 (length 2)
    equity = _series([100, 90, 80, 90, 100, 110, 105, 100, 115])
    assert max_drawdown_duration(equity) == 3


def test_average_drawdown_zero_for_monotone() -> None:
    equity = _series([100, 110, 120, 130])
    assert average_drawdown(equity) == pytest.approx(0.0)


def test_calmar_handles_monotone_up() -> None:
    equity = _series(list(range(100, 200)))
    # Max DD is zero -> Calmar is undefined; we return NaN, not inf
    assert math.isnan(calmar_ratio(equity))


def test_calmar_nan_when_equity_collapses() -> None:
    """If equity reaches zero, CAGR is undefined → Calmar must propagate NaN."""
    equity = _series([100.0, 50.0, 0.0])
    assert math.isnan(calmar_ratio(equity))


def test_annualised_volatility_known() -> None:
    daily = pd.Series([0.01, -0.01, 0.01, -0.01] * 100)
    sigma = daily.std(ddof=1)
    expected = sigma * math.sqrt(252)
    assert annualised_volatility(daily) == pytest.approx(expected)


def test_sharpe_annualization_factor_is_sqrt_252() -> None:
    """Closed-form lock: catches accidental ``* 252`` instead of ``* sqrt(252)``."""
    rng = np.random.default_rng(123)
    rets = pd.Series(rng.normal(0.0008, 0.012, 5000))
    mu = float(rets.mean())
    sigma = float(rets.std(ddof=1))
    expected = mu / sigma * math.sqrt(252)
    assert sharpe_ratio(rets) == pytest.approx(expected, rel=1e-12)


def test_sortino_annualization_factor_is_sqrt_252() -> None:
    rng = np.random.default_rng(456)
    rets = pd.Series(rng.normal(0.0008, 0.012, 5000))
    excess = rets
    downside = np.minimum(excess, 0.0)
    dd = math.sqrt(float(np.mean(downside**2)))
    expected = float(excess.mean()) / dd * math.sqrt(252)
    assert sortino_ratio(rets) == pytest.approx(expected, rel=1e-12)


def test_trade_statistics_non_empty() -> None:
    """The whole win-rate / profit-factor / avg-holding arithmetic is exercised."""
    from ma_backtester.metrics import trade_statistics

    trades = pd.DataFrame(
        {
            "net_return": [0.05, -0.02, 0.10, -0.01],
            "bars_held": [10, 5, 20, 3],
        }
    )
    s = trade_statistics(trades)
    assert s["n_trades"] == 4
    assert s["win_rate"] == 0.5
    # Gross wins = 0.15, gross losses = 0.03 -> PF = 5.0
    assert s["profit_factor"] == pytest.approx(5.0, rel=1e-9)
    assert s["avg_win"] == pytest.approx(0.075)
    assert s["avg_loss"] == pytest.approx(-0.015)
    assert s["avg_holding_period_days"] == pytest.approx(9.5)


def test_sortino_higher_than_sharpe_when_skewed_positive() -> None:
    """Right-skewed positive-mean returns should make Sortino > Sharpe.

    Sortino divides by downside deviation only; Sharpe by total deviation.
    When upside dispersion exceeds downside, Sortino is the larger.
    """
    rng = np.random.default_rng(7)
    base = rng.normal(0.0002, 0.004, 1000)
    jump_idx = rng.choice(1000, 60, replace=False)
    base[jump_idx] += 0.05  # 60 large positive jumps -> right skew
    rets = pd.Series(base)
    sharpe = sharpe_ratio(rets)
    sortino = sortino_ratio(rets)
    assert sortino > sharpe
