"""Adding a D/ST roster slot must not move skill replacement levels.

This is the invariant spec §5.1 measured and §7.4 requires pinned. It holds because
`_select_pool` fills a position slot from that position alone and FLEX from RB/WR/TE, and
`Position.DST` is in neither — so the DST pass takes 16 defenses and touches no skill
selection.

It is pinned rather than trusted because it is *load-bearing and invisible*: adding DST to
`FLEX_ELIGIBLE`, or making the pool selector fill slots from a shared pass, would silently
change every RB's VORP. The auction diff a reviewer looks at would show movement and be read
as the expected repricing (spec §7.4) rather than as a bug.
"""

from __future__ import annotations

import pandas as pd
import pytest

from projections.draft.league_config import LeagueConfig
from projections.draft.vorp import generate_vorp_table
from projections.schemas import Position, RosterSlot, Ruleset

N_TEAMS = 16
_COUNTS = {Position.QB: 60, Position.RB: 120, Position.WR: 160, Position.TE: 60, Position.DST: 32}
_BASE = {
    Position.QB: 320.0,
    Position.RB: 260.0,
    Position.WR: 250.0,
    Position.TE: 180.0,
    Position.DST: 120.0,
}

_SLOTS_NO_DST = {
    RosterSlot.QB: 1,
    RosterSlot.RB: 2,
    RosterSlot.WR: 2,
    RosterSlot.TE: 1,
    RosterSlot.FLEX: 1,
    RosterSlot.BENCH: 6,
}
_SLOTS_WITH_DST = {**_SLOTS_NO_DST, RosterSlot.DST: 1}

_SKILL = (Position.QB, Position.RB, Position.WR, Position.TE)


@pytest.fixture
def projections() -> pd.DataFrame:
    rows = [
        {
            "gsis_id": f"00-{i + 1_000_000 * n:07d}",
            "season": 2026,
            "position": pos.value,
            "ruleset": "ESPN_HALF",
            "n_weeks": 17,
            "season_mean": _BASE[pos] - i * 1.5,
            "season_p10": 0.0,
            "season_p50": 0.0,
            "season_p90": 0.0,
            "model_id": "invariant-probe",
            "generated_at": pd.Timestamp("2026-09-06", tz="UTC"),
        }
        for n, (pos, count) in enumerate(_COUNTS.items())
        for i in range(count)
    ]
    return pd.DataFrame(rows)


def _replacement_by_position(
    projections: pd.DataFrame, slots: dict[RosterSlot, int]
) -> dict[str, float]:
    cfg = LeagueConfig(
        name="invariant-probe", n_teams=N_TEAMS, roster_slots=slots, ruleset=Ruleset.espn_half()
    )
    table = generate_vorp_table(projections, cfg)
    grouped = table.groupby("position")["replacement_fpts"].first()
    return {str(pos): float(value) for pos, value in grouped.items()}


def test_skill_replacement_is_identical_with_and_without_a_dst_slot(
    projections: pd.DataFrame,
) -> None:
    without = _replacement_by_position(projections, _SLOTS_NO_DST)
    with_dst = _replacement_by_position(projections, _SLOTS_WITH_DST)
    for pos in _SKILL:
        assert without[pos.value] == with_dst[pos.value], (
            f"{pos.value} replacement moved when a D/ST slot was added "
            f"({without[pos.value]} -> {with_dst[pos.value]}). A DST pick must not consume a "
            "skill player; check FLEX_ELIGIBLE and _select_pool."
        )


def test_dst_is_absent_without_the_slot_and_priced_with_it(projections: pd.DataFrame) -> None:
    assert Position.DST.value not in _replacement_by_position(projections, _SLOTS_NO_DST)
    assert Position.DST.value in _replacement_by_position(projections, _SLOTS_WITH_DST)


def test_dst_rows_reach_the_vorp_table(projections: pd.DataFrame) -> None:
    """The failure this replaces: a kept DST slot used to raise 'cannot fill 16 DST slots:
    only 0 eligible players remain' because no defense had a projection."""
    cfg = LeagueConfig(
        name="invariant-probe",
        n_teams=N_TEAMS,
        roster_slots=_SLOTS_WITH_DST,
        ruleset=Ruleset.espn_half(),
    )
    table = generate_vorp_table(projections, cfg)
    dst = table[table["position"] == Position.DST.value]
    assert len(dst) == _COUNTS[Position.DST]
    assert (dst["vorp"] > 0).any(), "the best defenses must beat replacement"


def test_a_dst_slot_with_no_defenses_still_raises(projections: pd.DataFrame) -> None:
    """Removing the projections must bring back the original error, not a silent empty slot."""
    skill_only = projections[projections["position"] != Position.DST.value]
    cfg = LeagueConfig(
        name="invariant-probe",
        n_teams=N_TEAMS,
        roster_slots=_SLOTS_WITH_DST,
        ruleset=Ruleset.espn_half(),
    )
    with pytest.raises(ValueError, match="cannot fill"):
        generate_vorp_table(skill_only, cfg)
