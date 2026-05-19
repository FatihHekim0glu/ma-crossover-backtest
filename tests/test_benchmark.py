"""Benchmark comparison statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ma_backtester.benchmark import (
    capm_regression,
    compare_strategies,
    information_ratio,
    memmel_sharpe_difference_test,
)


def _bdays(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


def test_capm_recovers_beta_one() -> None:
    rng = np.random.default_rng(0)
    bench = pd.Series(rng.normal(0.0005, 0.01, 1000), index=_bdays(1000))
    strat = bench.copy()  # identical -> beta should be 1, alpha 0
    alpha_ann, _alpha_t, _, beta, _, _ = capm_regression(
        strategy_returns=strat, benchmark_returns=bench
    )
    assert beta == pytest.approx(1.0, abs=1e-9)
    assert alpha_ann == pytest.approx(0.0, abs=1e-9)


def test_capm_recovers_beta_half() -> None:
    rng = np.random.default_rng(1)
    bench = pd.Series(rng.normal(0.0005, 0.01, 1000), index=_bdays(1000))
    strat = 0.5 * bench
    _, _, _, beta, _, _ = capm_regression(strategy_returns=strat, benchmark_returns=bench)
    assert beta == pytest.approx(0.5, abs=1e-9)


def test_capm_detects_alpha() -> None:
    rng = np.random.default_rng(2)
    bench = pd.Series(rng.normal(0.0005, 0.01, 1000), index=_bdays(1000))
    daily_alpha = 0.0002
    strat = bench + daily_alpha  # constant alpha
    alpha_ann, _, _, beta, _, _ = capm_regression(strategy_returns=strat, benchmark_returns=bench)
    assert beta == pytest.approx(1.0, abs=1e-9)
    assert alpha_ann == pytest.approx(daily_alpha * 252, rel=1e-6)


def test_information_ratio_zero_when_returns_equal() -> None:
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 0.01, 500), index=_bdays(500))
    ir, _, _ = information_ratio(strategy_returns=r, benchmark_returns=r)
    assert np.isnan(ir) or ir == pytest.approx(0.0, abs=1e-9)


def test_sharpe_diff_test_symmetric() -> None:
    rng = np.random.default_rng(4)
    a = pd.Series(rng.normal(0.001, 0.01, 500), index=_bdays(500))
    b = pd.Series(rng.normal(0.001, 0.01, 500), index=_bdays(500))
    diff_ab, p_ab = memmel_sharpe_difference_test(returns_a=a, returns_b=b)
    diff_ba, p_ba = memmel_sharpe_difference_test(returns_a=b, returns_b=a)
    assert diff_ab == pytest.approx(-diff_ba, abs=1e-9)
    assert p_ab == pytest.approx(p_ba, abs=1e-9)


def test_capm_rejects_short_series() -> None:
    r = pd.Series(np.zeros(20), index=_bdays(20))
    with pytest.raises(ValueError, match="at least 30"):
        capm_regression(strategy_returns=r, benchmark_returns=r)


def test_memmel_identical_series_pvalue_high() -> None:
    """For identical return series the Sharpe diff is exactly zero and the
    two-sided p-value should be 1.0 (or very close).
    """
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 500), index=_bdays(500))
    diff, p = memmel_sharpe_difference_test(returns_a=r, returns_b=r)
    assert abs(diff) < 1e-12
    assert p == pytest.approx(1.0, abs=1e-9)


def test_compare_strategies_smoke() -> None:
    rng = np.random.default_rng(5)
    bench = pd.Series(rng.normal(0.0005, 0.01, 500), index=_bdays(500))
    strat = bench + rng.normal(0.0, 0.001, 500)
    cmp = compare_strategies(strategy_returns=strat, benchmark_returns=bench)
    assert cmp.n_observations == 500
    assert cmp.hac_lags >= 1
    assert cmp.beta == pytest.approx(1.0, abs=0.2)
