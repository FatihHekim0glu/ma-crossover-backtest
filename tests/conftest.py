"""Shared test fixtures and Hypothesis strategies.

All synthetic price series here are deterministic (seeded) — no network calls,
no yfinance, no flake. The unit suite must run offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import strategies as st

_DEFAULT_START = "2010-01-03"


def _bdays(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(_DEFAULT_START, periods=n)


@pytest.fixture
def constant_prices() -> pd.Series:
    """Flat 100.0 for 300 business days. Returns are all zero."""
    idx = _bdays(300)
    return pd.Series(100.0, index=idx, name="Close", dtype="float64")


@pytest.fixture
def linear_uptrend() -> pd.Series:
    """Linearly rising prices: 100 -> 200 over 300 bars."""
    idx = _bdays(300)
    values = np.linspace(100.0, 200.0, 300)
    return pd.Series(values, index=idx, name="Close", dtype="float64")


@pytest.fixture
def sine_wave_prices() -> pd.Series:
    """Sinusoid with mean 100, amplitude 10, period ~ 60 bars."""
    idx = _bdays(600)
    t = np.arange(600)
    values = 100.0 + 10.0 * np.sin(2 * np.pi * t / 60.0)
    return pd.Series(values, index=idx, name="Close", dtype="float64")


@pytest.fixture
def seeded_gbm_prices() -> pd.Series:
    """Geometric Brownian Motion with mu=0.08/yr, sigma=0.20/yr, seed=42."""
    rng = np.random.default_rng(42)
    n = 1000
    mu = 0.08 / 252.0
    sigma = 0.20 / np.sqrt(252.0)
    log_rets = rng.normal(loc=mu - 0.5 * sigma**2, scale=sigma, size=n)
    prices = 100.0 * np.exp(np.cumsum(log_rets))
    return pd.Series(prices, index=_bdays(n), name="Close", dtype="float64")


def make_gbm_series(
    *, seed: int, n: int, mu_annual: float = 0.07, sigma_annual: float = 0.18
) -> pd.Series:
    rng = np.random.default_rng(seed)
    mu = mu_annual / 252.0
    sigma = sigma_annual / np.sqrt(252.0)
    log_rets = rng.normal(loc=mu - 0.5 * sigma**2, scale=sigma, size=n)
    prices = 100.0 * np.exp(np.cumsum(log_rets))
    return pd.Series(prices, index=_bdays(n), name="Close", dtype="float64")


@st.composite
def price_series_strategy(
    draw: st.DrawFn,
    min_size: int = 100,
    max_size: int = 400,
) -> pd.Series:
    """Hypothesis strategy producing positive, finite OHLCV-close-like series."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    return make_gbm_series(seed=seed, n=n)
