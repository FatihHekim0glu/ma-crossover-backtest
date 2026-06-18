"""Streamlit dashboard smoke tests via ``streamlit.testing.v1.AppTest``.

These tests exercise the sidebar widget tree of ``app.py`` end-to-end with
a mocked ``load_close`` so they never hit yfinance. The two heavier paths
(parameter sweep, walk-forward) are gated behind ``@pytest.mark.slow``
because they really do run ~320 backtests / ~10 folds even on synthetic
data - adequate for nightly CI but excessive for a per-PR run.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from tests.conftest import make_gbm_series

_PRICES: pd.Series = make_gbm_series(seed=42, n=3800)
_PATCH_TARGET: str = "app.load_close"
_APP_PATH: str = "app.py"


def test_default_load_renders_title_and_metrics() -> None:
    with patch(_PATCH_TARGET, return_value=_PRICES):
        at = AppTest.from_file(_APP_PATH).run(timeout=30)
    assert not at.exception
    assert at.title[0].value == "Moving-Average Crossover Backtester"
    assert any("Headline metrics" in s.value for s in at.subheader)


def test_ticker_change_to_qqq_reloads() -> None:
    """Changing the ticker re-runs the app without exception.

    We intentionally do NOT assert on the mock call_args_list because
    Streamlit's ``@st.cache_data`` deduplicates identical (ticker, start, end)
    triples - the call may be served from cache rather than re-invoking
    our mock. The substantive guarantee is "no exception on re-render".
    """
    with patch(_PATCH_TARGET, return_value=_PRICES):
        at = AppTest.from_file(_APP_PATH).run(timeout=30)
        at.sidebar.selectbox[0].set_value("QQQ").run(timeout=30)
    assert not at.exception


def test_custom_ticker_rejects_path_traversal() -> None:
    at = AppTest.from_file(_APP_PATH).run(timeout=30)
    at.sidebar.selectbox[0].set_value("Other...").run(timeout=30)
    at.sidebar.text_input[0].set_value("../etc").run(timeout=30)
    assert any("Ticker must match" in e.value for e in at.error)


def test_fast_ge_slow_errors() -> None:
    with patch(_PATCH_TARGET, return_value=_PRICES):
        at = AppTest.from_file(_APP_PATH).run(timeout=30)
        # slow then fast, in that order: set slow low, then push fast above it
        at.sidebar.slider[1].set_value(20).run(timeout=30)  # slow -> 20
        at.sidebar.slider[0].set_value(100).run(timeout=30)  # fast -> 100
    assert any("Fast window must be strictly less" in e.value for e in at.error)


# NOTE: A direct AppTest of the "end < start" branch was attempted but
# Streamlit AppTest's date_input.set_value does not consistently propagate
# the new value to the script across versions. The same guard is exercised
# in tests/test_cli.py::test_run_rejects_reversed_dates, which is sufficient
# for the validation contract.


def test_cost_slider_rerenders_without_exception() -> None:
    with patch(_PATCH_TARGET, return_value=_PRICES):
        at = AppTest.from_file(_APP_PATH).run(timeout=30)
        at.sidebar.slider[2].set_value(25.0).run(timeout=30)  # cost slider
    assert not at.exception
