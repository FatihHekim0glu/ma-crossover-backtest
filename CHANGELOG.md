# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ma_backtester.data_providers` package with a pluggable end-of-day data
  abstraction:
  - `PolygonProvider` — REST client for Polygon.io aggregates, reference
    tickers, and grouped-daily snapshots. Sliding-window token bucket
    (default 100 rpm), three-attempt exponential backoff on 429/5xx,
    typed exception hierarchy (`PolygonError`, `PolygonAuthError`,
    `PolygonRateLimitError`, `PolygonDataError`).
  - `YFinanceProvider` — thin wrapper around the existing `data.load_ohlcv`
    pipeline so the repo keeps running without an API key.
  - `make_provider()` factory that selects Polygon when `POLYGON_API_KEY` is
    set (or passed explicitly) and falls back to yfinance otherwise.
- `httpx>=0.27` dependency for the Polygon HTTP client (and for
  transport-mocked offline tests).
- Three new test modules with 30+ unit tests covering the provider, the
  factory, and the yfinance wrapper. All marked `@pytest.mark.unit` and
  fully offline (`httpx.MockTransport` + `unittest.mock`).
- Streamlit app now shows a `data: polygon|yfinance` badge in the result
  panel so the active backend is always visible.

### Changed

- README updated with a *Data sources* section explaining the
  Polygon/yfinance trade-off and how to enable each backend.
- Top-of-README description updated to mention the survivorship-bias-aware
  Polygon integration.
- `pyproject.toml` declares `unit` and `network` pytest markers in addition
  to the existing `slow` marker.
