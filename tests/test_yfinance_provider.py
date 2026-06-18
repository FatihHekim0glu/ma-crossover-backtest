"""Unit tests for ``ma_backtester.data_providers.yfinance.YFinanceProvider``.

The provider is a thin wrapper around ``ma_backtester.data.load_ohlcv``; we
mock that function so the tests stay offline.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ma_backtester.data_providers import YFinanceProvider
from ma_backtester.data_providers.exceptions import PolygonError

pytestmark = pytest.mark.unit


def _fake_ohlcv_frame() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=5, name="Date")
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0.1, 0.5, 5))
    return pd.DataFrame(
        {
            "Open": close - 0.1,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": 1_000_000.0,
        },
        index=idx,
        dtype="float64",
    )


def test_get_eod_delegates_to_load_ohlcv() -> None:
    fake = _fake_ohlcv_frame()
    provider = YFinanceProvider()
    with patch("ma_backtester.data.load_ohlcv", return_value=fake) as mock_load:
        df = provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 8))
    mock_load.assert_called_once_with("AAPL", start="2024-01-02", end="2024-01-08")
    pd.testing.assert_frame_equal(df, fake)


def test_get_eod_returns_titlecase_columns() -> None:
    fake = _fake_ohlcv_frame()
    provider = YFinanceProvider()
    with patch("ma_backtester.data.load_ohlcv", return_value=fake):
        df = provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 8))
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index.name == "Date"
    assert df.index.tz is None


def test_get_eod_empty_frame_passes_through() -> None:
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    empty.index = pd.DatetimeIndex([], name="Date")
    provider = YFinanceProvider()
    with patch("ma_backtester.data.load_ohlcv", return_value=empty):
        df = provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 8))
    assert df.empty
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_get_eod_rejects_reversed_dates() -> None:
    provider = YFinanceProvider()
    with pytest.raises(ValueError, match="must be <= end"):
        provider.get_eod("AAPL", date(2024, 1, 8), date(2024, 1, 2))


def test_get_ticker_meta_raises_polygon_error() -> None:
    provider = YFinanceProvider()
    with pytest.raises(PolygonError, match="requires POLYGON_API_KEY"):
        provider.get_ticker_meta("AAPL")


def test_get_grouped_daily_raises_polygon_error() -> None:
    provider = YFinanceProvider()
    with pytest.raises(PolygonError, match="requires POLYGON_API_KEY"):
        provider.get_grouped_daily(date(2024, 1, 2))


def test_close_is_noop() -> None:
    """yfinance manages its own session - close must be a safe no-op."""
    provider = YFinanceProvider()
    assert provider.close() is None
    # Idempotent.
    assert provider.close() is None
