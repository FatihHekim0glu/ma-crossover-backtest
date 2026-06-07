"""Unit tests for ``ma_backtester.data_providers.polygon``.

All HTTP traffic is intercepted via :class:`httpx.MockTransport` so the
suite stays offline. Time-dependent paths (rate-limit waits, backoff) are
fast-forwarded via ``unittest.mock.patch`` on ``time.sleep`` / ``time.monotonic``.
"""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import patch

import httpx
import pandas as pd
import pytest

from ma_backtester.data_providers.exceptions import (
    PolygonAuthError,
    PolygonDataError,
    PolygonError,
    PolygonRateLimitError,
)
from ma_backtester.data_providers.polygon import (
    _RATE_WINDOW_SECONDS,
    PolygonProvider,
    _TokenBucket,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_provider(handler: Any, *, rpm: int = 1000) -> PolygonProvider:
    """Build a :class:`PolygonProvider` wired to a mocked transport."""
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="https://api.polygon.io")
    return PolygonProvider(api_key="test-key", session=client, rate_limit_rpm=rpm)


def _aggs_response(bars: list[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(
        200,
        json={"status": "OK", "resultsCount": len(bars), "results": bars},
    )


def _bar(ts_ms: int, **overrides: float) -> dict[str, Any]:
    base = {
        "t": ts_ms,
        "o": 100.0,
        "h": 101.0,
        "l": 99.0,
        "c": 100.5,
        "v": 1_000_000.0,
        "vw": 100.3,
        "n": 4321,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# get_ticker_meta
# --------------------------------------------------------------------------- #


def test_get_ticker_meta_calls_v3_endpoint() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        return httpx.Response(
            200,
            json={"status": "OK", "results": {"ticker": "AAPL", "name": "Apple Inc."}},
        )

    provider = _make_provider(handler)
    meta = provider.get_ticker_meta("aapl")
    assert "/v3/reference/tickers/AAPL" in captured["url"]
    assert captured["method"] == "GET"
    assert meta["ticker"] == "AAPL"
    assert meta["name"] == "Apple Inc."


def test_get_ticker_meta_raises_data_error_when_results_missing() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "OK"})

    provider = _make_provider(handler)
    with pytest.raises(PolygonDataError, match="No reference data"):
        provider.get_ticker_meta("AAPL")


# --------------------------------------------------------------------------- #
# get_eod
# --------------------------------------------------------------------------- #


def test_get_eod_returns_titlecase_columns() -> None:
    # Two trading days encoded as UTC midnight timestamps.
    ts_2024_01_02 = int(pd.Timestamp("2024-01-02", tz="UTC").value // 1_000_000)
    ts_2024_01_03 = int(pd.Timestamp("2024-01-03", tz="UTC").value // 1_000_000)

    def handler(_: httpx.Request) -> httpx.Response:
        return _aggs_response(
            [
                _bar(ts_2024_01_02, c=100.5),
                _bar(ts_2024_01_03, c=101.0),
            ]
        )

    provider = _make_provider(handler)
    df = provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 3))
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index.name == "Date"
    assert df.index.tz is None
    assert len(df) == 2
    assert df["Close"].iloc[0] == pytest.approx(100.5)
    assert df["Close"].iloc[1] == pytest.approx(101.0)


def test_get_eod_handles_empty_results() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "OK", "resultsCount": 0})

    provider = _make_provider(handler)
    df = provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 3))
    assert df.empty
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index.name == "Date"


def test_get_eod_retries_on_429() -> None:
    ts = int(pd.Timestamp("2024-01-02", tz="UTC").value // 1_000_000)
    call_count = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(429, json={"status": "ERROR", "error": "rate limit"})
        return _aggs_response([_bar(ts)])

    with patch("ma_backtester.data_providers.polygon.time.sleep"):
        provider = _make_provider(handler)
        df = provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 2))
    assert call_count["n"] == 2
    assert len(df) == 1


def test_get_eod_raises_auth_error_on_401() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "Unauthorized"})

    provider = _make_provider(handler)
    with pytest.raises(PolygonAuthError, match="401"):
        provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 3))


def test_get_eod_raises_auth_error_on_403() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "Forbidden"})

    provider = _make_provider(handler)
    with pytest.raises(PolygonAuthError, match="403"):
        provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 3))


def test_get_eod_exhausts_retries_on_persistent_429() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "Too many requests"})

    with patch("ma_backtester.data_providers.polygon.time.sleep"):
        provider = _make_provider(handler)
        with pytest.raises(PolygonRateLimitError):
            provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 3))


def test_get_eod_retries_on_5xx_then_succeeds() -> None:
    ts = int(pd.Timestamp("2024-01-02", tz="UTC").value // 1_000_000)
    call_count = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] < 2:
            return httpx.Response(502, json={"error": "bad gateway"})
        return _aggs_response([_bar(ts)])

    with patch("ma_backtester.data_providers.polygon.time.sleep"):
        provider = _make_provider(handler)
        df = provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 2))
    assert call_count["n"] == 2
    assert len(df) == 1


def test_get_eod_rejects_reversed_date_range() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return _aggs_response([])

    provider = _make_provider(handler)
    with pytest.raises(ValueError, match="must be <= end"):
        provider.get_eod("AAPL", date(2024, 1, 5), date(2024, 1, 2))


def test_get_eod_dates_are_normalized_and_tz_naive() -> None:
    ts = int(pd.Timestamp("2024-01-02", tz="UTC").value // 1_000_000)

    def handler(_: httpx.Request) -> httpx.Response:
        return _aggs_response([_bar(ts)])

    provider = _make_provider(handler)
    df = provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 2))
    assert df.index[0] == pd.Timestamp("2024-01-02")
    assert df.index.tz is None


# --------------------------------------------------------------------------- #
# get_grouped_daily
# --------------------------------------------------------------------------- #


def test_get_grouped_daily_indexes_by_ticker() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/v2/aggs/grouped/locale/us/market/stocks/2024-01-02" in str(request.url)
        return httpx.Response(
            200,
            json={
                "status": "OK",
                "results": [
                    {"T": "AAPL", "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 1e6},
                    {"T": "MSFT", "o": 200.0, "h": 202.0, "l": 199.0, "c": 201.5, "v": 5e5},
                ],
            },
        )

    provider = _make_provider(handler)
    df = provider.get_grouped_daily(date(2024, 1, 2))
    assert df.index.name == "ticker"
    assert set(df.index) == {"AAPL", "MSFT"}
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.loc["AAPL", "Close"] == pytest.approx(100.5)


def test_get_grouped_daily_empty_results_returns_empty_frame() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "OK", "results": []})

    provider = _make_provider(handler)
    df = provider.get_grouped_daily(date(2024, 1, 2))
    assert df.empty
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]


# --------------------------------------------------------------------------- #
# Token bucket / rate limiting
# --------------------------------------------------------------------------- #


def test_token_bucket_rate_limits() -> None:
    """When rpm requests are already in the window, the next call must sleep."""
    bucket = _TokenBucket(rpm=3)
    slept_for: list[float] = []

    fake_now = {"t": 0.0}

    def fake_monotonic() -> float:
        return fake_now["t"]

    def fake_sleep(seconds: float) -> None:
        slept_for.append(seconds)
        # Advance the clock so the sleeping call sees an updated window.
        fake_now["t"] += seconds

    with (
        patch("ma_backtester.data_providers.polygon.time.monotonic", fake_monotonic),
        patch("ma_backtester.data_providers.polygon.time.sleep", fake_sleep),
    ):
        # Three immediate acquires fill the window; none sleep.
        bucket.acquire()
        bucket.acquire()
        bucket.acquire()
        assert slept_for == []
        # The fourth must wait until the oldest entry rolls out of the 60s window.
        bucket.acquire()

    assert len(slept_for) == 1
    assert slept_for[0] > 0
    # Bound check: never wait longer than the configured window.
    assert slept_for[0] <= _RATE_WINDOW_SECONDS + 0.1


def test_token_bucket_rejects_non_positive_rpm() -> None:
    with pytest.raises(ValueError, match="rpm must be positive"):
        _TokenBucket(rpm=0)


# --------------------------------------------------------------------------- #
# Malformed payloads
# --------------------------------------------------------------------------- #


def test_provider_raises_data_error_on_malformed_payload() -> None:
    """A bar with a non-numeric close must raise PolygonDataError."""

    def handler(_: httpx.Request) -> httpx.Response:
        return _aggs_response(
            [
                {
                    "t": int(pd.Timestamp("2024-01-02", tz="UTC").value // 1_000_000),
                    "o": "not-a-number",
                    "h": 101.0,
                    "l": 99.0,
                    "c": 100.5,
                    "v": 1_000_000.0,
                }
            ]
        )

    provider = _make_provider(handler)
    with pytest.raises(PolygonDataError, match="malformed"):
        provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 2))


def test_provider_raises_data_error_on_missing_timestamp() -> None:
    """Daily bar with no ``t`` field is malformed."""

    def handler(_: httpx.Request) -> httpx.Response:
        return _aggs_response([{"o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 1_000_000.0}])

    provider = _make_provider(handler)
    with pytest.raises(PolygonDataError, match="missing timestamp"):
        provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 2))


def test_provider_raises_data_error_on_non_json_response() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not json</html>")

    provider = _make_provider(handler)
    with pytest.raises(PolygonDataError, match="non-JSON"):
        provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 2))


def test_provider_raises_generic_error_on_4xx_other_than_auth_or_rate_limit() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    provider = _make_provider(handler)
    with pytest.raises(PolygonError) as excinfo:
        provider.get_eod("XYZZY", date(2024, 1, 2), date(2024, 1, 2))
    assert "404" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Construction guards
# --------------------------------------------------------------------------- #


def test_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    with pytest.raises(PolygonAuthError, match="POLYGON_API_KEY is required"):
        PolygonProvider(api_key=None)


def test_provider_reads_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "env-key")
    provider = PolygonProvider()
    # Sanity: an authenticated request would include the env key. We don't
    # call the network — just construct successfully.
    assert provider is not None
    provider.close()


def test_provider_sends_api_key_as_query_param() -> None:
    captured: dict[str, Any] = {}
    ts = int(pd.Timestamp("2024-01-02", tz="UTC").value // 1_000_000)

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        return _aggs_response([_bar(ts)])

    provider = _make_provider(handler)
    provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 2))
    assert captured["params"].get("apiKey") == "test-key"
    assert captured["params"].get("adjusted") == "true"
