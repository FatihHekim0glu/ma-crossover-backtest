"""Factory that selects Polygon when configured, yfinance otherwise.

Resolution order
----------------
1. Explicit ``api_key`` argument (highest priority).
2. ``POLYGON_API_KEY`` environment variable.
3. yfinance fallback (no API key required).

This keeps the standalone repo usable without a Polygon subscription while
letting the app upgrade to survivorship-bias-aware data the moment a key is
configured - no code changes required at the call site.
"""

from __future__ import annotations

import logging
import os

from .polygon import PolygonProvider
from .yfinance import YFinanceProvider

logger = logging.getLogger(__name__)


def make_provider(
    api_key: str | None = None,
) -> PolygonProvider | YFinanceProvider:
    """Return the Polygon provider when a key is available, else yfinance.

    Parameters
    ----------
    api_key
        Explicit Polygon API key. Takes precedence over the environment
        variable. Empty/whitespace strings are treated as "not provided".
    """
    resolved = (api_key or os.environ.get("POLYGON_API_KEY") or "").strip()
    if resolved:
        logger.info("data provider: polygon (key configured)")
        return PolygonProvider(api_key=resolved)
    logger.info("data provider: yfinance (no POLYGON_API_KEY set)")
    return YFinanceProvider()
