"""Polygon.io REST client for end-of-day equity data.

Public surface is :class:`PolygonProvider`. The factory in
:mod:`ma_backtester.data_providers.factory` selects this provider when
``POLYGON_API_KEY`` is set and otherwise falls back to the local yfinance
pipeline so this repo keeps working standalone without a key.

Why this matters for the backtester
-----------------------------------
yfinance silently survivorship-biases backtests because tickers that have
been delisted disappear from the API. Polygon's ``/v2/aggs/grouped/...``
endpoint returns *every actively-traded* symbol on a given historical date,
which is the input you need to build a point-in-time S&P 500 universe.

Rate-limit / retry
------------------
A sliding-window token bucket caps outbound traffic at ``rate_limit_rpm``
requests per 60 seconds (Polygon Starter ceiling is 100). Each transient
failure (429 / 5xx / network) triggers exponential backoff over three
attempts (1s, 2s, 4s + small jitter). The bucket is process-local.

Adjusted prices
---------------
We always request ``adjusted=true`` (split + dividend). Mirror the field as
``adjusted=True`` in the returned frame's metadata-friendly columns so a
future raw-feed consumer can coexist without column collisions.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from collections import deque
from datetime import date
from typing import Any

import httpx
import pandas as pd

from .exceptions import (
    PolygonAuthError,
    PolygonDataError,
    PolygonError,
    PolygonRateLimitError,
)

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.polygon.io"
_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 1.0
_STARTER_RPM = 100
_RATE_WINDOW_SECONDS = 60.0
_HTTP_TIMEOUT = 30.0


# --------------------------------------------------------------------------- #
# Token bucket
# --------------------------------------------------------------------------- #


class _TokenBucket:
    """Sliding-window rate limiter - at most ``rpm`` requests per 60 seconds.

    Thread-safe: a single instance can be shared across worker threads inside
    one process. Across processes you'd want a Redis-backed limiter instead.
    """

    def __init__(self, rpm: int = _STARTER_RPM) -> None:
        if rpm <= 0:
            raise ValueError(f"rpm must be positive, got {rpm}")
        self._rpm = rpm
        self._window: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            cutoff = now - _RATE_WINDOW_SECONDS
            while self._window and self._window[0] < cutoff:
                self._window.popleft()
            if len(self._window) >= self._rpm:
                wait = _RATE_WINDOW_SECONDS - (now - self._window[0]) + 0.05
                if wait > 0:
                    time.sleep(wait)
                now = time.monotonic()
                cutoff = now - _RATE_WINDOW_SECONDS
                while self._window and self._window[0] < cutoff:
                    self._window.popleft()
            self._window.append(now)


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #


class PolygonProvider:
    """Polygon.io REST adapter producing TitleCase OHLCV frames.

    Parameters
    ----------
    api_key
        Polygon API key. If empty, falls back to ``POLYGON_API_KEY`` env
        var. :class:`PolygonAuthError` is raised when neither is present.
    session
        Optional pre-constructed :class:`httpx.Client`. Tests inject a
        transport-mocked client this way. When omitted, the provider owns and
        closes its own client.
    rate_limit_rpm
        Sliding-window cap on outbound requests per 60 seconds. Defaults to
        the Polygon Starter tier value (100).
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        session: httpx.Client | None = None,
        rate_limit_rpm: int = _STARTER_RPM,
    ) -> None:
        resolved_key = (api_key or os.environ.get("POLYGON_API_KEY") or "").strip()
        if not resolved_key:
            raise PolygonAuthError("POLYGON_API_KEY is required to instantiate PolygonProvider")
        self._api_key = resolved_key
        self._owns_session = session is None
        self._session = session or httpx.Client(timeout=_HTTP_TIMEOUT)
        self._bucket = _TokenBucket(rpm=rate_limit_rpm)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_ticker_meta(self, ticker: str) -> dict[str, Any]:
        """Return the ``/v3/reference/tickers/{ticker}`` payload body as a dict."""
        ticker = ticker.strip().upper()
        payload = self._request("GET", f"/v3/reference/tickers/{ticker}")
        results = payload.get("results")
        if not isinstance(results, dict):
            raise PolygonDataError(f"No reference data returned for {ticker}")
        return results

    def get_eod(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Return daily OHLCV for ``ticker`` over ``[start, end]`` (inclusive).

        Output is TitleCase (Open/High/Low/Close/Volume) with a tz-naive
        ``DatetimeIndex`` named ``Date``. Close is split- and dividend-
        adjusted (Polygon ``adjusted=true``). When the API returns no
        results, an empty frame with the canonical OHLCV columns is returned.
        """
        ticker = ticker.strip().upper()
        if start > end:
            raise ValueError(f"start ({start}) must be <= end ({end})")
        return self._fetch_aggs(ticker, start, end)

    def get_grouped_daily(self, date_: date) -> pd.DataFrame:
        """Grouped-daily snapshot of every actively-traded US stock on ``date_``.

        Index is the ticker symbol (named ``ticker``); columns are TitleCase
        OHLCV. Used by survivorship-bias-aware universe construction.
        """
        path = f"/v2/aggs/grouped/locale/us/market/stocks/{date_.isoformat()}"
        payload = self._request("GET", path, params={"adjusted": "true"})
        results = payload.get("results") or []
        if not results:
            empty = pd.DataFrame(columns=list(_OHLCV_COLUMNS))
            empty.index.name = "ticker"
            return empty
        rows: dict[str, dict[str, float]] = {}
        for bar in results:
            symbol = bar.get("T")
            if not symbol:
                continue
            try:
                rows[symbol] = {
                    "Open": float(bar.get("o", 0.0)),
                    "High": float(bar.get("h", 0.0)),
                    "Low": float(bar.get("l", 0.0)),
                    "Close": float(bar.get("c", 0.0)),
                    "Volume": float(bar.get("v", 0.0)),
                }
            except (TypeError, ValueError) as exc:
                raise PolygonDataError(
                    f"grouped-daily bar for {symbol} on {date_} is malformed: {exc}"
                ) from exc
        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "ticker"
        return df

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    # ------------------------------------------------------------------ #
    # HTTP plumbing
    # ------------------------------------------------------------------ #

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{_BASE_URL}{path}"
        merged_params = dict(params or {})
        merged_params["apiKey"] = self._api_key
        last_exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            self._bucket.acquire()
            try:
                response = self._session.request(method, url, params=merged_params)
            except httpx.HTTPError as exc:
                last_exc = exc
                self._sleep_backoff(attempt)
                continue
            status_code = response.status_code
            if status_code in (401, 403):
                raise PolygonAuthError(
                    f"Polygon auth rejected ({status_code}); check POLYGON_API_KEY"
                )
            if status_code == 429:
                last_exc = PolygonRateLimitError(
                    f"Polygon rate limit hit (attempt {attempt + 1}/{_MAX_ATTEMPTS})"
                )
                self._sleep_backoff(attempt)
                continue
            if status_code >= 500:
                last_exc = PolygonError(
                    f"Polygon {status_code} on {path} (attempt {attempt + 1}/{_MAX_ATTEMPTS})"
                )
                self._sleep_backoff(attempt)
                continue
            if status_code >= 400:
                raise PolygonError(f"Polygon {status_code} on {path}: {response.text[:200]}")
            try:
                return response.json()
            except ValueError as exc:
                raise PolygonDataError(f"Polygon returned non-JSON for {path}") from exc
        if isinstance(last_exc, PolygonRateLimitError):
            raise last_exc
        raise PolygonError(
            f"Polygon request failed after {_MAX_ATTEMPTS} attempts on {path}"
        ) from last_exc

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        delay = _BACKOFF_BASE_SECONDS * (2**attempt) + random.uniform(0.0, 0.25)
        time.sleep(delay)

    # ------------------------------------------------------------------ #
    # Aggregates
    # ------------------------------------------------------------------ #

    def _fetch_aggs(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        path = f"/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}"
        payload = self._request(
            "GET",
            path,
            params={"adjusted": "true", "sort": "asc", "limit": 50_000},
        )
        results = payload.get("results") or []
        if not results:
            empty = pd.DataFrame(columns=list(_OHLCV_COLUMNS))
            empty.index = pd.DatetimeIndex([], name="Date")
            return empty
        records: list[dict[str, Any]] = []
        for bar in results:
            ts_ms = bar.get("t")
            if ts_ms is None:
                raise PolygonDataError(f"daily bar for {ticker} is missing timestamp 't': {bar!r}")
            try:
                records.append(
                    {
                        # Polygon's daily aggregates encode the trading day as
                        # the UTC midnight timestamp of that calendar date.
                        # Do NOT shift to America/New_York - that would move
                        # the bar back one day.
                        "Date": pd.Timestamp(ts_ms, unit="ms", tz="UTC")
                        .tz_localize(None)
                        .normalize(),
                        "Open": float(bar.get("o", 0.0)),
                        "High": float(bar.get("h", 0.0)),
                        "Low": float(bar.get("l", 0.0)),
                        "Close": float(bar.get("c", 0.0)),
                        "Volume": float(bar.get("v", 0.0)),
                    }
                )
            except (TypeError, ValueError) as exc:
                raise PolygonDataError(f"daily bar for {ticker} is malformed: {exc}") from exc
        df = pd.DataFrame.from_records(records).set_index("Date").sort_index()
        df.index = pd.DatetimeIndex(df.index, name="Date")
        return df.astype("float64")
