"""Unit tests for ``ma_backtester.data_providers.factory.make_provider``."""

from __future__ import annotations

import pytest

from ma_backtester.data_providers import (
    PolygonProvider,
    YFinanceProvider,
    make_provider,
)

pytestmark = pytest.mark.unit


def test_make_provider_with_explicit_key_returns_polygon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ensure the env var is NOT also set, so we can prove the explicit arg wins.
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    provider = make_provider(api_key="explicit-key")
    try:
        assert isinstance(provider, PolygonProvider)
    finally:
        provider.close()


def test_make_provider_without_key_returns_yfinance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    provider = make_provider()
    try:
        assert isinstance(provider, YFinanceProvider)
    finally:
        provider.close()


def test_make_provider_reads_env_var_when_arg_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "env-only-key")
    provider = make_provider()
    try:
        assert isinstance(provider, PolygonProvider)
    finally:
        provider.close()


def test_make_provider_explicit_key_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "env-key")
    provider = make_provider(api_key="explicit-key")
    try:
        assert isinstance(provider, PolygonProvider)
    finally:
        provider.close()


def test_make_provider_treats_whitespace_key_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    provider = make_provider(api_key="   ")
    try:
        assert isinstance(provider, YFinanceProvider)
    finally:
        provider.close()
