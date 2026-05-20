"""End-to-end CLI tests via Typer's CliRunner.

``load_close`` is monkey-patched to a synthetic GBM series so the CLI tests
never hit yfinance. Sweep/walk-forward also monkeypatch ``DEFAULT_SWEEP`` to
a tiny grid so the full suite stays under ~5 seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from ma_backtester import cli
from ma_backtester.cli import app
from ma_backtester.config import SweepConfig
from ma_backtester.data import DataQualityError

runner = CliRunner()


@pytest.fixture
def synthetic_close() -> pd.Series:
    idx = pd.bdate_range("2010-01-04", "2024-12-31")
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0004, 0.01, len(idx))
    return pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx, name="close")


@pytest.fixture
def patched_load(monkeypatch: pytest.MonkeyPatch, synthetic_close: pd.Series) -> pd.Series:
    monkeypatch.setattr(
        cli, "load_close", lambda ticker, start, end: synthetic_close.loc[start:end]
    )
    return synthetic_close


@pytest.fixture
def tiny_sweep(monkeypatch: pytest.MonkeyPatch) -> SweepConfig:
    small = SweepConfig(fast_windows=(5, 10), slow_windows=(20, 30))
    monkeypatch.setattr(cli, "DEFAULT_SWEEP", small)
    return small


def test_run_happy_path(patched_load: pd.Series) -> None:
    result = runner.invoke(app, ["run", "--ticker", "SPY", "--fast", "20", "--slow", "50"])
    assert result.exit_code == 0
    assert "Strategy vs Buy & Hold" in result.stdout


def test_run_rejects_invalid_ticker() -> None:
    result = runner.invoke(app, ["run", "--ticker", "../etc"])
    assert result.exit_code != 0
    combined = result.stdout + (result.stderr or "")
    assert "invalid" in combined.lower()


def test_run_rejects_reversed_dates() -> None:
    result = runner.invoke(app, ["run", "--start", "2024-01-01", "--end", "2023-01-01"])
    assert result.exit_code != 0
    combined = result.stdout + (result.stderr or "")
    assert "strictly before" in combined


def test_run_data_quality_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> pd.Series:
        raise DataQualityError("no rows for ticker")

    monkeypatch.setattr(cli, "load_close", boom)
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 1
    combined = result.stdout + (result.stderr or "")
    assert "no rows for ticker" in combined


def test_sweep_happy_path(patched_load: pd.Series, tiny_sweep: SweepConfig) -> None:
    result = runner.invoke(app, ["sweep", "--ticker", "SPY"])
    assert result.exit_code == 0
    assert "Deflated Sharpe Ratio" in result.stdout
    assert "Best in-sample" in result.stdout


def test_walk_forward_happy_path(patched_load: pd.Series, tiny_sweep: SweepConfig) -> None:
    result = runner.invoke(app, ["walk-forward", "--train-years", "3", "--test-years", "1"])
    assert result.exit_code == 0
    assert "Walk-Forward Folds" in result.stdout


@pytest.mark.parametrize("cmd", ["run", "sweep", "walk-forward"])
def test_help_lists_options(cmd: str) -> None:
    result = runner.invoke(app, [cmd, "--help"])
    assert result.exit_code == 0
    assert "--ticker" in result.stdout
    assert "--cost-bps" in result.stdout


def test_sweep_data_quality_error_exits_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> pd.Series:
        raise DataQualityError("invalid ticker symbol")

    monkeypatch.setattr(cli, "load_close", boom)
    result = runner.invoke(app, ["sweep"])
    assert result.exit_code == 1
    combined = result.stdout + (result.stderr or "")
    assert "invalid ticker symbol" in combined


def test_walk_forward_data_quality_error_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: object, **kwargs: object) -> pd.Series:
        raise DataQualityError("no rows")

    monkeypatch.setattr(cli, "load_close", boom)
    result = runner.invoke(app, ["walk-forward"])
    assert result.exit_code == 1
    combined = result.stdout + (result.stderr or "")
    assert "no rows" in combined
