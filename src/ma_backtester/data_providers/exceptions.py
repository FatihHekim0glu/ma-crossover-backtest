"""Typed Polygon error hierarchy.

Callers can distinguish auth (401/403) and rate-limit (429) failures from
generic data/parse errors, but the base :class:`PolygonError` still catches
everything via ``except PolygonError``.
"""

from __future__ import annotations


class PolygonError(Exception):
    """Base class for every Polygon-originated failure."""


class PolygonAuthError(PolygonError):
    """Raised on HTTP 401/403 — missing or invalid API key."""


class PolygonRateLimitError(PolygonError):
    """Raised after exhausting retries on HTTP 429."""


class PolygonDataError(PolygonError):
    """Raised when the response payload is missing expected fields or is empty."""
