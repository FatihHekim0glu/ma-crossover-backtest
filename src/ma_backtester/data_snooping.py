"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

When you sweep N strategies on the same history, the *maximum* observed
Sharpe is biased upward even under a null of zero skill. The Deflated
Sharpe Ratio adjusts the observed Sharpe for:

1. selection bias across N effectively-independent trials,
2. non-normality of returns (skew and kurtosis),
3. sample length.

It returns the probability that the strategy's *true* Sharpe is positive
after the selection adjustment. Conventional threshold: DSR > 0.95.

Variance formula
----------------
Mertens (2002) / Bailey & Lopez de Prado (2014) give:

    Var[SR_hat] = (1 - g3 * SR + ((g4 - 1) / 4) * SR^2) / (T - 1)

where g3 is sample skewness and g4 is the **non-excess** sample kurtosis
(i.e. g4 = 3 for a normal distribution). ``scipy.stats.kurtosis`` with
``fisher=True`` returns *excess* kurtosis k = g4 - 3, so the equivalent
expression in terms of k is::

    (1 - g3 * SR + ((k + 2) / 4) * SR^2) / (T - 1)

which is what this module implements. Mistakenly using ``k / 4`` instead
of ``(k + 2) / 4`` collapses to ``1 / (T - 1)`` for normal returns and
silently under-estimates the variance, biasing PSR/DSR optimistic.

Effective trials are estimated via PCA on the strategy-return matrix
across the parameter grid: the number of principal components needed to
explain 95% of variance. This is a heuristic; the original paper clusters
trials by correlation. Documented as a deliberate simplification.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from ma_backtester.config import TRADING_DAYS_PER_YEAR
from ma_backtester.results import DeflatedSharpeResult

_EULER_MASCHERONI: float = 0.5772156649015328
DSR_REJECT_THRESHOLD: float = 0.95


def effective_number_of_trials(
    *,
    returns_matrix: pd.DataFrame,
    variance_threshold: float = 0.95,
) -> int:
    """PCA-based estimate of independent trials.

    Returns the number of principal components needed to explain
    ``variance_threshold`` of variance in the strategy-return matrix.
    Falls back to total column count for a degenerate input.
    """
    clean = returns_matrix.dropna(axis=0, how="any")
    if clean.shape[0] < 2 or clean.shape[1] < 2:
        return max(1, returns_matrix.shape[1])

    centred = clean - clean.mean(axis=0)
    cov = np.cov(centred.to_numpy(), rowvar=False)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    total = eigvals.sum()
    if total <= 0:
        return returns_matrix.shape[1]
    cumulative = np.cumsum(eigvals) / total
    n_eff = int(np.searchsorted(cumulative, variance_threshold) + 1)
    return max(1, min(n_eff, returns_matrix.shape[1]))


def _expected_max_sharpe(*, n_trials: int, sharpe_variance_daily: float) -> float:
    """Bailey/Lopez de Prado extreme-value approximation."""
    if n_trials <= 1:
        return 0.0
    inv_n = 1.0 / n_trials
    inv_ne = 1.0 / (n_trials * math.e)
    term_a = (1.0 - _EULER_MASCHERONI) * float(stats.norm.ppf(1.0 - inv_n))
    term_b = _EULER_MASCHERONI * float(stats.norm.ppf(1.0 - inv_ne))
    return math.sqrt(sharpe_variance_daily) * (term_a + term_b)


def probabilistic_sharpe_ratio(
    *,
    daily_returns: pd.Series,
    benchmark_sharpe_annual: float = 0.0,
) -> float:
    """Probability that the true (annualised) Sharpe exceeds the benchmark.

    Adjusts for non-normality via skew and kurtosis (Bailey-LdP 2012).
    """
    r = daily_returns.dropna()
    n = len(r)
    if n < 30:
        return float("nan")

    sigma = float(r.std(ddof=1))
    if sigma < 1e-12:
        return float("nan")
    sr_daily = float(r.mean()) / sigma
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=True))  # excess kurtosis

    # See module docstring: (k + 2) / 4 in terms of excess kurtosis k,
    # equivalent to (g4 - 1) / 4 in terms of non-excess kurtosis g4.
    var = (1.0 - skew * sr_daily + ((kurt + 2.0) / 4.0) * sr_daily**2) / (n - 1)
    if var <= 0:  # pragma: no cover — variance is non-negative for valid return samples
        return float("nan")

    sr_bench_daily = benchmark_sharpe_annual / math.sqrt(TRADING_DAYS_PER_YEAR)
    z = (sr_daily - sr_bench_daily) / math.sqrt(var)
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    *,
    daily_returns: pd.Series,
    n_trials: int,
    n_effective_trials: int | None = None,
) -> DeflatedSharpeResult:
    """Compute the Deflated Sharpe Ratio.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily returns of the *selected* (best-in-sample) strategy.
    n_trials : int
        Raw number of strategy variants tested. Used only when
        ``n_effective_trials`` is None.
    n_effective_trials : int | None
        PCA-derived effective independent trials. Pass this when you have it.
    """
    r = daily_returns.dropna()
    n = len(r)
    if n < 30:
        raise ValueError(f"Need at least 30 returns, got {n}")

    sigma = float(r.std(ddof=1))
    if sigma < 1e-12:
        raise ValueError("zero return-volatility — cannot deflate")
    sr_daily = float(r.mean()) / sigma
    sr_annual = sr_daily * math.sqrt(TRADING_DAYS_PER_YEAR)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=True))  # excess kurtosis

    # See module docstring: (k + 2) / 4 in terms of excess kurtosis k,
    # equivalent to (g4 - 1) / 4 in terms of non-excess kurtosis g4.
    sr_var_daily = (1.0 - skew * sr_daily + ((kurt + 2.0) / 4.0) * sr_daily**2) / (n - 1)
    if sr_var_daily <= 0:  # pragma: no cover — non-negative by construction for valid samples
        raise ValueError("non-positive Sharpe variance — check input")

    n_eff = n_effective_trials if n_effective_trials is not None else max(1, n_trials)
    expected_max_daily = _expected_max_sharpe(n_trials=n_eff, sharpe_variance_daily=sr_var_daily)
    expected_max_annual = expected_max_daily * math.sqrt(TRADING_DAYS_PER_YEAR)

    z = (sr_daily - expected_max_daily) / math.sqrt(sr_var_daily)
    dsr = float(stats.norm.cdf(z))
    psr = probabilistic_sharpe_ratio(daily_returns=r, benchmark_sharpe_annual=0.0)

    return DeflatedSharpeResult(
        observed_sharpe=sr_annual,
        expected_max_sharpe_under_null=expected_max_annual,
        deflated_sharpe=dsr,
        probabilistic_sharpe=psr,
        n_trials=int(n_trials),
        n_effective_trials=int(n_eff),
        skew=skew,
        kurtosis=kurt,
        can_reject_null=bool(dsr > DSR_REJECT_THRESHOLD),
    )
