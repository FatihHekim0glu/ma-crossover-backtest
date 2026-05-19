"""Data-layer validation. Network calls live behind @pytest.mark.slow."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ma_backtester.data import (
    DataQualityError,
    _is_stale,
    _read_meta,
    _write_meta,
    validate,
)


def _good_frame() -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=250)
    rng = np.random.default_rng(0)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 250)))
    high = close * 1.005
    low = close * 0.995
    df = pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": 1_000_000.0},
        index=idx,
    )
    df.index.name = "Date"
    return df.astype("float64")


def test_validate_accepts_clean_frame() -> None:
    df = _good_frame()
    report = validate(df, "TEST")
    assert report.n_rows == 250
    assert report.n_jumps == 0


def test_validate_rejects_inverted_high_low() -> None:
    df = _good_frame()
    df.iloc[5, df.columns.get_loc("High")] = 1.0
    with pytest.raises(DataQualityError, match="OHLC"):
        validate(df, "TEST")


def test_validate_rejects_negative_close() -> None:
    df = _good_frame()
    df.iloc[3, df.columns.get_loc("Close")] = -1.0
    with pytest.raises(DataQualityError, match="OHLC"):
        validate(df, "TEST")


def test_validate_flags_large_jump() -> None:
    df = _good_frame()
    df.iloc[100, df.columns.get_loc("Close")] *= 10.0
    df.iloc[100, df.columns.get_loc("High")] *= 10.0
    report = validate(df, "TEST")
    assert report.n_jumps >= 1


def test_is_stale_no_meta() -> None:
    assert _is_stale(None, ttl_hours=24.0)


def test_is_stale_fresh_meta() -> None:
    fresh = {
        "last_fetched_utc": datetime.now(UTC).isoformat(),
        "n_rows": 100,
    }
    assert not _is_stale(fresh, ttl_hours=24.0)


def test_is_stale_old_meta() -> None:
    old_ts = (datetime.now(UTC) - timedelta(hours=72)).isoformat()
    assert _is_stale({"last_fetched_utc": old_ts, "n_rows": 100}, ttl_hours=24.0)


def test_meta_roundtrip(tmp_path: Path) -> None:
    meta = {"SPY": {"n_rows": 100, "last_fetched_utc": "2024-01-01T00:00:00+00:00"}}
    _write_meta(tmp_path, meta)
    loaded = _read_meta(tmp_path)
    assert loaded == meta


def test_meta_recovers_from_bad_json(tmp_path: Path) -> None:
    (tmp_path / "_meta.json").write_text("{not valid json", encoding="utf-8")
    assert _read_meta(tmp_path) == {}


@pytest.mark.slow
def test_load_ohlcv_live_spy() -> None:
    """Hits yfinance; only runs with `pytest -m slow`."""
    from ma_backtester.data import load_ohlcv

    df = load_ohlcv("SPY", start="2024-01-01", end="2024-03-01")
    assert len(df) > 30
    assert {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns)
