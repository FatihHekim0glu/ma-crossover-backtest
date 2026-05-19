"""Transaction cost model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ma_backtester.config import CostConfig
from ma_backtester.costs import FixedBpsCost, zero_cost_model


def _positions(values: list[float]) -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.Series(values, index=idx, dtype="float64", name="position")


def test_zero_cost_model_yields_zero_drag() -> None:
    cm = zero_cost_model()
    pos = _positions([0, 1, 1, 0, 1, 0])
    cost = cm.cost_series(pos)
    assert (cost == 0.0).all()


def test_fixed_bps_cost_charges_on_turnover() -> None:
    cm = FixedBpsCost(CostConfig(per_side_bps=5.0))
    pos = _positions([0, 1, 1, 0, 1, 0])
    cost = cm.cost_series(pos)
    expected = pd.Series([0.0, 1.0, 0.0, 1.0, 1.0, 1.0]) * (5.0 / 10_000)
    np.testing.assert_allclose(cost.to_numpy(), expected.to_numpy(), atol=1e-12)


def test_round_trip_property() -> None:
    cfg = CostConfig(per_side_bps=5.0)
    assert cfg.round_trip_bps == pytest.approx(10.0)


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(bps=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
def test_cost_scales_linearly_in_bps(bps: float) -> None:
    cm = FixedBpsCost(CostConfig(per_side_bps=bps))
    pos = _positions([0, 1, 0, 1, 0])
    cost = cm.cost_series(pos)
    expected_total = (1 + 1 + 1 + 1) * (bps / 10_000)
    assert float(cost.sum()) == pytest.approx(expected_total, rel=1e-12, abs=1e-15)
