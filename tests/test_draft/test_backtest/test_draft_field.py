"""Tests for src/projections/draft/backtest/draft_field.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.draft.backtest.draft_field import _MAXP, _MINP, draft_mixed_field, seat_layout
from projections.draft.league_config import LeagueConfig
from projections.draft.roster_eligibility import bot_eligible
from projections.schemas import _PYARROW_STR, Position, RosterSlot, VorpTableSchema


def _config_16_half() -> LeagueConfig:
    return LeagueConfig(
        name="test16half",
        n_teams=16,
        budget=200,
        min_bid=1,
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 4,
        },
        ruleset="espn_half",  # type: ignore[arg-type]
    )


def _synthetic_pool(n_per_pos: int = 50) -> pd.DataFrame:
    # Pos-offsets keep IDs unique and valid (\d{2}-\d{7})
    pos_offset = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
    rows = []
    for pos in ("QB", "RB", "WR", "TE"):
        off = pos_offset[pos]
        for i in range(n_per_pos):
            player_num = off * 1000 + i
            rows.append(
                {
                    "gsis_id": f"00-{player_num:07d}",
                    "position": pos,
                    "season_mean_fpts": 300.0 - i,
                    "vorp": 150.0 - i,
                    "replacement_fpts": 150.0,
                    "consensus_adp": float(i * 4 + {"QB": 3, "RB": 1, "WR": 2, "TE": 4}[pos]),
                }
            )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    return VorpTableSchema.validate(df)


def test_seat_layout_mirrors_on_paired_seeds() -> None:
    odd = seat_layout(1)  # base
    even = seat_layout(2)  # mirrored
    assert {s for s, k in odd.items() if k == "now_or_never"} == {2, 6, 10, 14}
    assert {s for s, k in odd.items() if k == "season_value"} == {4, 8, 12, 16}
    assert {s for s, k in even.items() if k == "now_or_never"} == {4, 8, 12, 16}
    assert {s for s, k in even.items() if k == "season_value"} == {2, 6, 10, 14}
    assert sum(1 for k in odd.values() if k == "bot") == 8


def test_draft_fills_every_roster_without_dupes() -> None:
    from projections.draft.assistant.strategy import DraftStrategy, RawVorpStrategy

    cfg = _config_16_half()
    pool = _synthetic_pool()
    seats: dict[int, DraftStrategy | None] = {
        s: (RawVorpStrategy() if s in (2, 4) else None) for s in range(1, 17)
    }
    rosters = draft_mixed_field(seats, pool, cfg, rng=np.random.default_rng(0), jitter=8.0)
    allp = [g for r in rosters.values() for g in r]
    assert len(allp) == len(set(allp))  # no dupes
    assert all(len(r) == cfg.roster_size for r in rosters.values())


def test_bots_satisfy_position_minimums() -> None:
    cfg = _config_16_half()
    pool = _synthetic_pool(n_per_pos=60)  # 240 players, deeper than 16 * MAXP per position
    seats: dict[int, object] = {s: None for s in range(1, 17)}  # all-bot field
    rosters = draft_mixed_field(seats, pool, cfg, rng=np.random.default_rng(0), jitter=8.0)  # type: ignore[arg-type]
    pos_by_id = {str(g): str(p) for g, p in zip(pool["gsis_id"], pool["position"], strict=False)}
    minp = {"QB": 1, "RB": 3, "WR": 3, "TE": 1}
    for seat, roster in rosters.items():
        counts = {pos: 0 for pos in minp}
        for g in roster:
            counts[pos_by_id[g]] = counts.get(pos_by_id[g], 0) + 1
        for pos, lo in minp.items():
            assert counts[pos] >= lo, f"seat {seat} has {counts[pos]} {pos}, need >= {lo}"


def test_seat_layout_defaults_unchanged() -> None:
    # Default labels reproduce the historical nn/sv layout exactly.
    odd = seat_layout(1)
    assert {s for s, lab in odd.items() if lab == "now_or_never"} == {2, 6, 10, 14}
    assert {s for s, lab in odd.items() if lab == "season_value"} == {4, 8, 12, 16}
    even = seat_layout(0)  # paired mirror swaps nn<->sv
    assert {s for s, lab in even.items() if lab == "now_or_never"} == {4, 8, 12, 16}


def test_seat_layout_custom_labels() -> None:
    lay = seat_layout(1, label_a="season_value_timing", label_b="season_value")
    assert {s for s, lab in lay.items() if lab == "season_value_timing"} == {2, 6, 10, 14}
    assert {s for s, lab in lay.items() if lab == "season_value"} == {4, 8, 12, 16}
    assert {s for s, lab in lay.items() if lab == "bot"} == {1, 3, 5, 7, 9, 11, 13, 15}


def test_hero_seat_layout_one_hero_rest_bots() -> None:
    from projections.draft.backtest.draft_field import hero_seat_layout

    layout = hero_seat_layout(hero_seat=3, hero_label="now_or_never", n_teams=12)
    assert layout[3] == "now_or_never"
    assert all(layout[s] == "bot" for s in range(1, 13) if s != 3)
    assert set(layout) == set(range(1, 13))


def test_hero_seat_layout_rejects_out_of_range_seat() -> None:
    import pytest

    from projections.draft.backtest.draft_field import hero_seat_layout

    with pytest.raises(ValueError, match="hero_seat"):
        hero_seat_layout(hero_seat=0, hero_label="x", n_teams=12)
    with pytest.raises(ValueError, match="hero_seat"):
        hero_seat_layout(hero_seat=13, hero_label="x", n_teams=12)


def test_snake_bounds_are_position_keyed_and_exclude_k_dst() -> None:
    # The snake field's tuned values, now Position-keyed; K/DST a bot holds are never returned.
    assert _MINP == {Position.QB: 1, Position.RB: 3, Position.WR: 3, Position.TE: 1}
    assert _MAXP == {Position.QB: 3, Position.RB: 6, Position.WR: 6, Position.TE: 3}
    counts = {Position.QB: 1, Position.RB: 3, Position.WR: 3, Position.TE: 1, Position.K: 1}
    elig = bot_eligible(counts, 5, minimums=_MINP, maximums=_MAXP)
    assert Position.K not in elig  # iteration domain = bound keysets, not counts
