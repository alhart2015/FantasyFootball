"""D/ST has its own variance shape, and falling back to `default` is wrong for it.

Fitted by scripts/fit_dst_variance.py from 544 team-weeks of ESPN 2025 D/ST actuals scored
through score_dst. Spec §6.
"""

from __future__ import annotations

import pytest

from projections.draft.assistant.performance_variance import VarianceParams
from projections.schemas import Position


@pytest.fixture
def params() -> VarianceParams:
    return VarianceParams.load()


def test_dst_has_its_own_fitted_entries(params: VarianceParams) -> None:
    """Without these a defense silently falls through to `default`, an affine fitted on skill
    players."""
    assert Position.DST.value in params.weekly_std_affine
    assert f"{Position.DST.value}|veteran" in params.mean_mult_log_sd
    assert f"{Position.DST.value}|rookie" in params.mean_mult_log_sd


def test_a_defense_is_not_a_rookie(params: VarianceParams) -> None:
    """A team has no rookie year. Both tiers carry the same value so the lookup can never fall
    through to the skill default depending on a flag that means nothing here."""
    assert params.log_sd(Position.DST.value, is_rookie=True) == params.log_sd(
        Position.DST.value, is_rookie=False
    )


def test_dst_weekly_spread_is_nearly_flat_in_the_mean(params: VarianceParams) -> None:
    """The measured shape, and the reason DST needs its own entry: a defense's weekly swing is
    driven by touchdowns and shutouts — near-random events — not by how good it is. Skill
    positions scale strongly with the mean (RB a=0.26, WR a=0.31); DST fits a=0.025."""
    coef = params.weekly_std_affine[Position.DST.value]
    assert coef["a"] < 0.1
    assert 5.0 < coef["b"] < 7.0
    for skill in ("RB", "WR", "TE"):
        assert coef["a"] < params.weekly_std_affine[skill]["a"]


def test_the_default_would_understate_a_weak_defense(params: VarianceParams) -> None:
    """Where the fallback actually hurt. At a strong defense's scoring rate the default is
    close; at a streaming-candidate's rate it is far too confident — and streaming is exactly
    the decision this feeds (issue #166)."""
    weak_per_game = 1.9  # the worst 2025 defense's rate
    fitted = params.weekly_std(Position.DST.value, weak_per_game)
    default_coef = params.weekly_std_affine["default"]
    fallback = default_coef["a"] * weak_per_game + default_coef["b"]
    assert fitted > fallback * 1.25, (
        f"expected the fitted DST spread ({fitted:.2f}) to be materially wider than the skill "
        f"default ({fallback:.2f}) at a weak defense's scoring rate"
    )


def test_dst_is_the_least_predictable_position_season_to_season(params: VarianceParams) -> None:
    """log_sd 0.42 against RB's 0.40, the widest skill position. Measured, not asserted: the
    2025 actual/projected ratio ranged 0.32x to 2.08x across the 32 defenses."""
    dst = params.log_sd(Position.DST.value, is_rookie=False)
    for skill in ("QB", "RB", "WR", "TE"):
        assert dst >= params.log_sd(skill, is_rookie=False)
