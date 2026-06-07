"""Pluggable EOD data providers.

Public surface:

* :class:`PolygonProvider` — REST client for Polygon.io aggregates and
  reference data. Requires ``POLYGON_API_KEY``.
* :class:`YFinanceProvider` — thin wrapper around the repo's existing
  yfinance pipeline so the backtester can run standalone with no key.
* :func:`make_provider` — factory that picks the right backend.
* :class:`PolygonError` and friends — typed exception hierarchy callers
  can use to distinguish auth, rate-limit, and data-shape failures.
"""

from __future__ import annotations

from .exceptions import (
    PolygonAuthError,
    PolygonDataError,
    PolygonError,
    PolygonRateLimitError,
)
from .factory import make_provider
from .polygon import PolygonProvider
from .yfinance import YFinanceProvider

__all__ = [
    "PolygonAuthError",
    "PolygonDataError",
    "PolygonError",
    "PolygonProvider",
    "PolygonRateLimitError",
    "YFinanceProvider",
    "make_provider",
]
