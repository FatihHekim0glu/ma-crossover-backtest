"""Deflated Sharpe Ratio and N_effective."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as _stats

from ma_backtester.data_snooping import (
    deflated_sharpe_ratio,
    effective_number_of_trials,
    probabilistic_sharpe_ratio,
)


def _bdays(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


def test_n_effective_lower_for_correlated_strategies() -> None:
    rng = np.random.default_rng(0)
    common = rng.normal(0.0005, 0.01, 1000)
    # 10 highly-correlated strategies (each a small perturbation of common)
    mat = pd.DataFrame(
        {f"s{i}": common + rng.normal(0, 0.0005, 1000) for i in range(10)},
        index=_bdays(1000),
    )
    n_eff = effective_number_of_trials(returns_matrix=mat)
    assert 1 <= n_eff < 10  # should be much smaller than 10


def test_n_effective_full_for_uncorrelated() -> None:
    rng = np.random.default_rng(1)
    mat = pd.DataFrame(
        {f"s{i}": rng.normal(0.0005, 0.01, 1000) for i in range(8)},
        index=_bdays(1000),
    )
    n_eff = effective_number_of_trials(returns_matrix=mat)
    # Independent series -> N_eff close to (but possibly < ) full count
    assert n_eff >= 6


def test_psr_above_half_for_positive_sharpe() -> None:
    rng = np.random.default_rng(7)
    rets = pd.Series(rng.normal(0.001, 0.01, 1000), index=_bdays(1000))
    psr = probabilistic_sharpe_ratio(daily_returns=rets, benchmark_sharpe_annual=0.0)
    assert psr > 0.5


def test_dsr_deflates_sharpe() -> None:
    rng = np.random.default_rng(2)
    rets = pd.Series(rng.normal(0.001, 0.01, 1000), index=_bdays(1000))
    dsr = deflated_sharpe_ratio(daily_returns=rets, n_trials=400, n_effective_trials=30)
    assert dsr.observed_sharpe > 0
    assert dsr.expected_max_sharpe_under_null > 0
    # DSR is a probability
    assert 0.0 <= dsr.deflated_sharpe <= 1.0


def test_dsr_rejects_with_huge_sharpe() -> None:
    rng = np.random.default_rng(3)
    rets = pd.Series(rng.normal(0.005, 0.005, 1000), index=_bdays(1000))  # Sharpe ~ 16 ann.
    dsr = deflated_sharpe_ratio(daily_returns=rets, n_trials=400, n_effective_trials=30)
    assert dsr.can_reject_null


def test_dsr_short_series_rejected() -> None:
    rets = pd.Series([0.001] * 20, index=_bdays(20))
    with pytest.raises(ValueError, match="at least 30"):
        deflated_sharpe_ratio(daily_returns=rets, n_trials=10)


def test_expected_max_sharpe_matches_hand_calculation() -> None:
    """Closed-form lock on the Bailey-LdP extreme-value approximation.

    For N=10 trials with daily Sharpe variance = 1, the formula gives
    (1 - gamma) * Phi^-1(0.9) + gamma * Phi^-1(1 - 1/(10e))
      = 0.4228 * 1.2816 + 0.5772 * 1.7894
      = 1.5746...
    """
    from ma_backtester.data_snooping import _expected_max_sharpe

    assert _expected_max_sharpe(n_trials=10, sharpe_variance_daily=1.0) == pytest.approx(
        1.5746, abs=1e-3
    )


def test_expected_max_sharpe_zero_for_single_trial() -> None:
    from ma_backtester.data_snooping import _expected_max_sharpe

    assert _expected_max_sharpe(n_trials=1, sharpe_variance_daily=1.0) == 0.0


def test_dsr_raises_on_zero_volatility() -> None:
    rets = pd.Series([0.0] * 100, index=_bdays(100))
    with pytest.raises(ValueError, match="zero return-volatility"):
        deflated_sharpe_ratio(daily_returns=rets, n_trials=10)


def test_psr_normal_returns_matches_paper_formula() -> None:
    """For ~Gaussian returns the Mertens variance collapses to (1 + 0.5*SR^2)/(n-1).

    This guards against the recurring bug of dropping the +2 from the
    excess-kurtosis adjustment. The drift must be large enough that
    ``sr^2`` materially contributes to the variance — with zero drift
    the ``+2`` correction is numerically invisible and the test would
    pass under the bug.
    """
    rng = np.random.default_rng(0)
    # Daily drift 0.05 with sigma 0.01 gives SR_daily ~ 5; sr^2 ~ 25 dominates.
    rets = pd.Series(rng.normal(0.05, 0.01, 2000), index=_bdays(2000))
    sigma = float(rets.std(ddof=1))
    sr_daily = float(rets.mean()) / sigma
    expected_var = (1.0 + 0.5 * sr_daily**2) / (len(rets) - 1)
    expected_z = sr_daily / math.sqrt(expected_var)
    expected_psr = float(_stats.norm.cdf(expected_z))

    assert probabilistic_sharpe_ratio(daily_returns=rets) == pytest.approx(
        expected_psr, rel=1e-3, abs=1e-3
    )


# --------------------------------------------------------------------------- #
# Coverage push — fallback branches in N_eff and PSR (cycle 6)
# --------------------------------------------------------------------------- #


def test_n_effective_single_row_falls_back_to_column_count() -> None:
    """clean.shape[0] < 2 → fall through to column count."""
    mat = pd.DataFrame({"a": [0.01], "b": [0.02], "c": [0.03]}, index=_bdays(1))
    assert effective_number_of_trials(returns_matrix=mat) == 3


def test_n_effective_single_column_falls_back() -> None:
    """Single-column matrix has no covariance → fall through to column count."""
    mat = pd.DataFrame({"only": np.linspace(0, 0.01, 200)}, index=_bdays(200))
    assert effective_number_of_trials(returns_matrix=mat) == 1


def test_n_effective_zero_variance_falls_back() -> None:
    """All-constant returns → covariance is zero → fallback to column count."""
    mat = pd.DataFrame({f"s{i}": [0.001] * 100 for i in range(4)}, index=_bdays(100))
    assert effective_number_of_trials(returns_matrix=mat) == 4


def test_psr_short_series_returns_nan() -> None:
    rets = pd.Series([0.001] * 20, index=_bdays(20))
    assert math.isnan(probabilistic_sharpe_ratio(daily_returns=rets))


def test_psr_zero_volatility_returns_nan() -> None:
    rets = pd.Series([0.0] * 100, index=_bdays(100))
    assert math.isnan(probabilistic_sharpe_ratio(daily_returns=rets))


def test_dsr_defaults_n_effective_to_n_trials() -> None:
    """When n_effective_trials is None, DSR uses n_trials directly."""
    rng = np.random.default_rng(11)
    rets = pd.Series(rng.normal(0.001, 0.01, 500), index=_bdays(500))
    result = deflated_sharpe_ratio(daily_returns=rets, n_trials=42)
    assert result.n_effective_trials == 42
