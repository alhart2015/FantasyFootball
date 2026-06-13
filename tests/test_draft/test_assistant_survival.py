"""Tests for the ADP survival model."""

from __future__ import annotations

import math

import numpy as np
import pytest

from projections.draft.assistant.survival import (
    LogisticSurvival,
    SurvivalModel,
    default_sigma,
    expected_best_by_position,
)


def test_is_survival_model() -> None:
    assert isinstance(LogisticSurvival(sigma=8.0), SurvivalModel)


@pytest.mark.parametrize("bad_sigma", [0.0, -1.0, math.nan])
def test_rejects_non_positive_sigma(bad_sigma: float) -> None:
    # NaN must be rejected too: `nan <= 0` is False, so without an explicit
    # guard it would silently produce an all-null p_available column.
    with pytest.raises(ValueError, match="sigma"):
        LogisticSurvival(sigma=bad_sigma)


def test_monotone_in_adp() -> None:
    model = LogisticSurvival(sigma=8.0)
    # Later ADP (drafted later) → more likely to survive to a fixed pick.
    p_early = model.p_available(adp=3.0, at_pick=18)
    p_late = model.p_available(adp=30.0, at_pick=18)
    assert 0.0 <= p_early <= p_late <= 1.0


def test_boundaries() -> None:
    model = LogisticSurvival(sigma=8.0)
    assert model.p_available(adp=1.0, at_pick=60) < 0.05  # long gone
    assert model.p_available(adp=200.0, at_pick=10) > 0.95  # nowhere near taken


def test_null_adp_survives() -> None:
    model = LogisticSurvival(sigma=8.0)
    assert model.p_available(adp=math.nan, at_pick=18) == 1.0


def test_default_sigma_scales_with_teams() -> None:
    assert default_sigma(12) == 8.0  # two-thirds of a 12-team round


def test_expected_best_by_position_hand_computed() -> None:
    # RB: values [50, 40], survival [0.5, 0.5] -> 50*0.5*1 + 40*0.5*(1-0.5) = 35
    # WR: value 52, survival 0.5 -> 52*0.5 = 26
    positions = np.array(["RB", "RB", "WR"])
    values = np.array([50.0, 40.0, 52.0])
    probs = np.array([0.5, 0.5, 0.5])
    tiebreak = np.array(["00-0000001", "00-0000002", "00-0000003"])
    out = expected_best_by_position(positions, values, probs, tiebreak)
    assert out == {"RB": 35.0, "WR": 26.0}


def test_expected_best_by_position_sorts_by_value_then_tiebreak() -> None:
    # Unsorted input: the higher value is consumed first regardless of row order.
    positions = np.array(["RB", "RB"])
    values = np.array([40.0, 50.0])
    probs = np.array([0.5, 0.5])
    tiebreak = np.array(["00-0000002", "00-0000001"])
    out = expected_best_by_position(positions, values, probs, tiebreak)
    assert out["RB"] == 35.0


def test_expected_best_by_position_zero_prob_best_player() -> None:
    # Best player (value 50) is certain to be GONE (p=0) -> contributes 0; the
    # second-best (40, p=0.5) is then the best survivor: 40 * 0.5 * (1 - 0) = 20.
    positions = np.array(["RB", "RB"])
    values = np.array([50.0, 40.0])
    probs = np.array([0.0, 0.5])
    tiebreak = np.array(["00-0000001", "00-0000002"])
    out = expected_best_by_position(positions, values, probs, tiebreak)
    assert out["RB"] == 20.0
