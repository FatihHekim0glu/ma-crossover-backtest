"""Data-layer validation. Network calls live behind @pytest.mark.slow."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ma_backtester.data import (
    DataQualityError,
    _is_stale,
    _normalise_yf_columns,
    _read_meta,
    _safe_ticker,
    _write_meta,
    load_ohlcv,
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


# --------------------------------------------------------------------------- #
# Coverage push — network paths via mock (cycle 6, runs offline)
# --------------------------------------------------------------------------- #


def _ohlcv_fixture() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=10, name="Date")
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0.1, 0.5, 10))
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


_STOOQ_CSV = (
    b"Date,Open,High,Low,Close,Volume\n"
    b"2024-01-02,100.0,101.0,99.0,100.5,1000\n"
    b"2024-01-03,100.5,101.5,99.5,101.0,1100\n"
    b"2024-01-04,101.0,102.0,100.0,101.5,1200\n"
)


def test_validate_rejects_empty_frame() -> None:
    """An entirely-empty OHLCV frame must be rejected before any analysis."""
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    with pytest.raises(DataQualityError, match="empty frame"):
        validate(empty, "TEST")


def test_load_close_returns_named_series(tmp_path: Path) -> None:
    """``load_close`` wraps ``load_ohlcv`` and exposes only the Close column."""
    from ma_backtester.data import load_close

    df = _ohlcv_fixture()
    df.to_parquet(tmp_path / "AAPL.parquet")
    (tmp_path / "_meta.json").write_text(
        '{"AAPL": {"last_fetched_utc": "'
        + datetime.now(UTC).isoformat()
        + '", "n_rows": '
        + str(len(df))
        + "}}",
        encoding="utf-8",
    )
    s = load_close("AAPL", start="2024-01-02", end="2024-01-15", cache_dir=tmp_path)
    assert isinstance(s, pd.Series)
    assert s.name == "AAPL"
    assert s.dtype == "float64"


def test_load_from_stooq_with_mocked_url() -> None:
    """Direct test of load_from_stooq with a mocked urllib response."""
    from ma_backtester.data import load_from_stooq

    mock_resp = MagicMock()
    mock_resp.__enter__ = MagicMock(return_value=MagicMock(read=lambda: _STOOQ_CSV))
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("ma_backtester.data.urlopen", return_value=mock_resp):
        df = load_from_stooq("AAPL")
    assert set(df.columns) >= {"Open", "High", "Low", "Close", "Volume"}
    assert df.index.name == "Date"
    assert len(df) == 3


def test_safe_ticker_accepts_common_forms() -> None:
    assert _safe_ticker("aapl") == "AAPL"
    assert _safe_ticker("BRK-B") == "BRK-B"
    assert _safe_ticker("^GSPC") == "^GSPC"


@pytest.mark.parametrize("bad", ["../etc", "", "AB C", "A" * 16, "abc;rm"])
def test_safe_ticker_rejects(bad: str) -> None:
    with pytest.raises(DataQualityError, match="invalid ticker"):
        _safe_ticker(bad)


def test_normalise_yf_columns_drops_multiindex() -> None:
    base = _ohlcv_fixture()
    df = base.copy()
    df.columns = pd.MultiIndex.from_product([base.columns, ["AAPL"]])
    out = _normalise_yf_columns(df)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_normalise_yf_columns_rejects_missing() -> None:
    df = _ohlcv_fixture().drop(columns=["Volume"])
    with pytest.raises(DataQualityError, match="missing columns"):
        _normalise_yf_columns(df)


def test_normalise_yf_columns_strips_timezone() -> None:
    df = _ohlcv_fixture()
    df.index = df.index.tz_localize("US/Eastern")
    out = _normalise_yf_columns(df)
    assert out.index.tz is None
    assert out.index.name == "Date"


def test_load_ohlcv_cache_hit_skips_yfinance(tmp_path: Path) -> None:
    """Fresh parquet + fresh meta → yfinance never called."""
    df = _ohlcv_fixture()
    df.to_parquet(tmp_path / "AAPL.parquet")
    meta = {
        "AAPL": {
            "last_fetched_utc": datetime.now(UTC).isoformat(),
            "n_rows": len(df),
        }
    }
    (tmp_path / "_meta.json").write_text(
        '{"AAPL": ' + str(meta["AAPL"]).replace("'", '"') + "}",
        encoding="utf-8",
    )
    with patch("yfinance.download") as yf_mock:
        result = load_ohlcv(
            "AAPL",
            start="2024-01-02",
            end="2024-01-15",
            cache_dir=tmp_path,
        )
    assert yf_mock.call_count == 0
    assert len(result) > 0


def test_load_ohlcv_yfinance_empty_raises(tmp_path: Path) -> None:
    """yfinance returns empty → DataQualityError after retries."""
    with (
        patch("yfinance.download", return_value=pd.DataFrame()),
        patch("ma_backtester.data.time.sleep"),
        pytest.raises(DataQualityError),
    ):
        load_ohlcv("AAPL", start="2024-01-01", end="2024-01-15", cache_dir=tmp_path)


def test_load_ohlcv_yfinance_failure_falls_back_to_stooq(tmp_path: Path) -> None:
    """yfinance raises → load_from_stooq is tried → success → source='stooq'."""
    fixture_df = _ohlcv_fixture()
    fixture_df.index.name = "Date"

    mock_resp = MagicMock()
    mock_resp.__enter__ = MagicMock(return_value=MagicMock(read=lambda: _STOOQ_CSV))
    mock_resp.__exit__ = MagicMock(return_value=False)

    with (
        patch("yfinance.download", side_effect=ConnectionError("boom")),
        patch("ma_backtester.data.time.sleep"),
        patch("ma_backtester.data.urlopen", return_value=mock_resp),
    ):
        df = load_ohlcv("AAPL", start="2024-01-02", end="2024-01-04", cache_dir=tmp_path)
    assert len(df) > 0
    meta_text = (tmp_path / "_meta.json").read_text(encoding="utf-8")
    assert "stooq" in meta_text


def test_load_ohlcv_both_sources_fail_chains_errors(tmp_path: Path) -> None:
    """If both yfinance AND Stooq fail, the yfinance error is raised, chained."""
    with (
        patch("yfinance.download", side_effect=RuntimeError("yf-down")),
        patch("ma_backtester.data.time.sleep"),
        patch("ma_backtester.data.urlopen", side_effect=OSError("stooq-down")),
    ):
        with pytest.raises(DataQualityError) as excinfo:
            load_ohlcv("AAPL", start="2024-01-02", end="2024-01-04", cache_dir=tmp_path)
        # The originating cause chain should mention Stooq somewhere
        chain_messages = []
        e: BaseException | None = excinfo.value
        while e is not None:
            chain_messages.append(str(e))
            e = e.__cause__ or e.__context__
        assert any("stooq" in m.lower() or "stooq-down" in m for m in chain_messages)
