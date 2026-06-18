"""yfinance-backed fallback provider.

Wraps :func:`ma_backtester.data.load_ohlcv` so the rest of the codebase can
target the same ``get_eod`` shape regardless of which backend is active.
Strictly delegates - no re-implementation of caching, retries, or validation.

Methods that yfinance cannot service in the spirit of point-in-time accuracy
(``get_ticker_meta``, ``get_grouped_daily``) raise :class:`PolygonError` so
callers can degrade gracefully or branch on provider capability.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd

from .exceptions import PolygonError

logger = logging.getLogger(__name__)


class YFinanceProvider:
    """Drop-in fallback that mirrors :class:`PolygonProvider`'s public surface."""

    def __init__(self) -> None:
        # Marker that distinguishes this provider in the UI badge / logs
        # without callers needing isinstance checks across packages.
        self.name = "yfinance"

    def get_ticker_meta(self, ticker: str) -> dict[str, Any]:
        raise PolygonError(
            "get_ticker_meta requires POLYGON_API_KEY; "
            "yfinance fallback has no equivalent reference endpoint"
        )

    def get_eod(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Delegate to ``ma_backtester.data.load_ohlcv``.

        Returns the same TitleCase OHLCV / tz-naive Date-indexed frame the
        rest of the backtester already expects.
        """
        if start > end:
            raise ValueError(f"start ({start}) must be <= end ({end})")
        # Lazy import: ma_backtester.data pulls in yfinance, parquet, and
        # urllib at module load time. Deferring keeps this module cheap to
        # import for callers that only need the factory's branching decision.
        from ma_backtester.data import load_ohlcv

        return load_ohlcv(ticker, start=start.isoformat(), end=end.isoformat())

    def get_grouped_daily(self, date_: date) -> pd.DataFrame:
        raise PolygonError(
            "get_grouped_daily requires POLYGON_API_KEY; yfinance has no grouped-daily equivalent"
        )

    def close(self) -> None:
        # Parity with PolygonProvider - yfinance manages its own session.
        return None
