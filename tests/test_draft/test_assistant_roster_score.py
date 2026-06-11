"""Tests for optimal starting-lineup scoring."""

from __future__ import annotations

import pandas as pd

from projections.draft.assistant.roster_score import optimal_lineup_points
from projections.schemas import _PYARROW_STR, RosterSlot


def _roster(players: list[tuple[str, str, float]]) -> pd.DataFrame:
    """players = [(gsis_id, position, season_mean_fpts), ...]."""
    return pd.DataFrame(
        {
            "gsis_id": pd.array([p[0] for p in players], dtype=_PYARROW_STR),
            "position": pd.array([p[1] for p in players], dtype=_PYARROW_STR),
            "season_mean_fpts": [p[2] for p in players],
        }
    )


def test_strand_case_fills_flex_before_super_flex() -> None:
    # Spec §3.4 counterexample: SUPER_FLEX grabbing RB first would strand the QB.
    roster = _roster([("00-0000001", "RB", 100.0), ("00-0000002", "QB", 90.0)])
    slots = {RosterSlot.FLEX: 1, RosterSlot.SUPER_FLEX: 1}
    assert optimal_lineup_points(roster, slots) == 190.0


def test_single_position_and_flex_pick_best() -> None:
    roster = _roster(
        [
            ("00-0000001", "RB", 100.0),
            ("00-0000002", "RB", 80.0),
            ("00-0000003", "WR", 90.0),
        ]
    )
    # RB slot takes best RB (100); FLEX takes best remaining eligible (WR 90 > RB 80).
    slots = {RosterSlot.RB: 1, RosterSlot.FLEX: 1}
    assert optimal_lineup_points(roster, slots) == 190.0


def test_bench_and_ir_score_nothing() -> None:
    roster = _roster([("00-0000001", "RB", 100.0), ("00-0000002", "RB", 80.0)])
    slots = {RosterSlot.RB: 1, RosterSlot.BENCH: 5, RosterSlot.IR: 1}
    assert optimal_lineup_points(roster, slots) == 100.0


def test_unfillable_slot_scores_partial() -> None:
    # K starting slot on a roster with no kicker (skill-only pool) -> that slot = 0.
    roster = _roster([("00-0000001", "RB", 100.0)])
    slots = {RosterSlot.RB: 1, RosterSlot.K: 1}
    assert optimal_lineup_points(roster, slots) == 100.0


def test_tie_break_is_deterministic() -> None:
    # Equal points: which player fills which slot is gsis_id-stable, total invariant.
    roster = _roster(
        [
            ("00-0000002", "RB", 100.0),
            ("00-0000001", "RB", 100.0),
            ("00-0000003", "WR", 50.0),
        ]
    )
    slots = {RosterSlot.RB: 1, RosterSlot.FLEX: 1}
    # RB slot + FLEX both fillable by RB; best two RBs start (100+100); WR benched.
    assert optimal_lineup_points(roster, slots) == 200.0
    # same inputs -> same total; the impl's sorted() gives the cross-run guarantee.
    assert optimal_lineup_points(roster, slots) == 200.0
