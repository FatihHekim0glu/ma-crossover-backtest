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


def test_cost_series_treats_nan_first_position_as_zero() -> None:
    """A raw un-shifted signal could start with NaN; the entry cost must
    not poison the downstream equity curve."""
    cm = FixedBpsCost(CostConfig(per_side_bps=5.0))
    pos = pd.Series([float("nan"), 1.0, 1.0, 0.0], index=pd.bdate_range("2020-01-01", periods=4))
    cost = cm.cost_series(pos)
    # cost.iloc[0] should be 0.0 (NaN start treated as zero); subsequent
    # bars should be finite.
    assert cost.iloc[0] == 0.0
    assert not cost.iloc[1:].isna().any()


def test_cost_config_rejects_negative_bps() -> None:
    with pytest.raises(ValueError, match="per_side_bps must be >= 0"):
        CostConfig(per_side_bps=-1.0)


def test_cost_config_rejects_absurd_bps() -> None:
    with pytest.raises(ValueError, match="looks wrong"):
        CostConfig(per_side_bps=5000.0)


def test_fixed_bps_cost_exposes_config() -> None:
    """The ``config`` property is the read-back accessor for the wrapped CostConfig."""
    cfg = CostConfig(per_side_bps=7.5)
    cm = FixedBpsCost(cfg)
    assert cm.config is cfg
    assert cm.config.per_side_bps == 7.5


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
