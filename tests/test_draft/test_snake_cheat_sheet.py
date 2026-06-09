"""Tests for `projections.draft.snake_cheat_sheet`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.draft._pool import _select_pool
from projections.draft.league_config import LeagueConfig
from projections.draft.snake_cheat_sheet import (
    DISPLAY_NAME_FALLBACK,
    _assign_tiers,
    generate_snake_cheat_sheet,
)
from projections.schemas import (
    _PYARROW_STR,
    Position,
    RosterSlot,
    Ruleset,
    SnakeCheatSheetSchema,
    VorpTableSchema,
)

_POSITION_ID_PREFIX: dict[Position, int] = {
    Position.QB: 1,
    Position.RB: 2,
    Position.WR: 3,
    Position.TE: 4,
    Position.K: 5,
    Position.DST: 6,
}


def _make_config(
    n_teams: int = 4,
    roster_slots: dict[RosterSlot, int] | None = None,
    ruleset: Ruleset | None = None,
) -> LeagueConfig:
    return LeagueConfig(
        name="test",
        n_teams=n_teams,
        budget=100,
        min_bid=1,
        roster_slots=roster_slots
        or {
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 1,
        },
        ruleset=ruleset or Ruleset.espn_ppr(),
    )


def _make_vorp_table(positions: dict[Position, int], base_fpts: float = 300.0) -> pd.DataFrame:
    """Build a VorpTableSchema-validated frame with `count` rows per position.

    Per-row VORPs are arbitrary but monotonically decreasing within position
    (vorp = base_fpts - i). Replacement fpts is broadcast per-position from
    the row at rank `count` (the "first off the board" position-internal
    boundary — close enough for testing; algorithmic correctness lives in
    upstream VORP tests).
    """
    rows: list[dict[str, object]] = []
    for pos, count in positions.items():
        prefix = _POSITION_ID_PREFIX[pos]
        # Pick replacement = the worst player at this position (so all VORPs >= 0).
        replacement_fpts = base_fpts - (count - 1)
        for i in range(count):
            season_mean_fpts = base_fpts - i
            rows.append(
                {
                    "gsis_id": f"00-{prefix}{i:06d}",
                    "position": pos.value,
                    "season_mean_fpts": season_mean_fpts,
                    "vorp": season_mean_fpts - replacement_fpts,
                    "replacement_fpts": replacement_fpts,
                }
            )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    return VorpTableSchema.validate(df)


def test_assign_tiers_gap_based_correctness() -> None:
    """§5.1 #8 — synthetic gaps produce the documented tier partition."""
    vorps = np.array([100.0, 99.0, 98.0, 50.0, 49.0, 48.0, 10.0, 9.0, 8.0])
    tiers = _assign_tiers(vorps, n_tiers=3)
    assert list(tiers) == [1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_assign_tiers_fallback_when_n_in_pool_less_than_n() -> None:
    """§5.1 #9 — fewer in-pool players than tiers: each gets own tier."""
    vorps = np.array([50.0, 30.0, 20.0, 10.0, 5.0])  # 5 players
    tiers = _assign_tiers(vorps, n_tiers=8)
    assert list(tiers) == [1, 2, 3, 4, 5]


def test_assign_tiers_exact_when_n_in_pool_equals_n() -> None:
    """§5.1 #10 — exactly N in-pool players: 1-per-tier."""
    vorps = np.array([100.0, 80.0, 60.0, 40.0])
    tiers = _assign_tiers(vorps, n_tiers=4)
    assert list(tiers) == [1, 2, 3, 4]


def test_assign_tiers_with_n_equal_one_all_tier_one() -> None:
    """§5.1 #19 — tiers_per_position=1 collapses everyone into tier 1."""
    vorps = np.array([100.0, 50.0, 25.0, 10.0, 1.0])
    tiers = _assign_tiers(vorps, n_tiers=1)
    assert list(tiers) == [1, 1, 1, 1, 1]


def test_assign_tiers_tie_break_prefers_earlier_gap() -> None:
    """§5.1 #21 — when gaps are tied, the earlier (higher-rank) gap wins."""
    # gaps = [1, 4, 1, 4]: two gaps of 4 competing for the single allowed cut
    # under n_tiers=2. Earlier gap (index 1) wins; later gap (index 3) loses.
    vorps = np.array([10.0, 9.0, 5.0, 4.0, 0.0])
    tiers = _assign_tiers(vorps, n_tiers=2)
    assert list(tiers) == [1, 1, 2, 2, 2]


def test_assign_tiers_empty_input() -> None:
    vorps = np.array([], dtype=np.float64)
    tiers = _assign_tiers(vorps, n_tiers=8)
    assert tiers.shape == (0,)
    assert tiers.dtype == np.int64


def test_output_validates_against_schema() -> None:
    """§5.1 #1 — output is SnakeCheatSheetSchema-valid; column order matches."""
    cfg = _make_config()
    vorp = _make_vorp_table({Position.QB: 8, Position.RB: 12, Position.WR: 12, Position.TE: 8})
    out = generate_snake_cheat_sheet(vorp, cfg)
    SnakeCheatSheetSchema.validate(out)
    assert list(out.columns) == list(SnakeCheatSheetSchema.to_schema().columns)


def test_row_count_preserved() -> None:
    """§5.1 #2 — output row count == input row count."""
    cfg = _make_config()
    vorp = _make_vorp_table({Position.QB: 8, Position.RB: 12, Position.WR: 12, Position.TE: 8})
    out = generate_snake_cheat_sheet(vorp, cfg)
    assert len(out) == len(vorp)


def test_positional_rank_strictly_monotonic_within_position() -> None:
    """§5.1 #3 — positional_rank is 1, 2, 3, ... by vorp desc within each position."""
    cfg = _make_config()
    vorp = _make_vorp_table({Position.QB: 8, Position.RB: 12, Position.WR: 12, Position.TE: 8})
    out = generate_snake_cheat_sheet(vorp, cfg)
    for pos_value in out["position"].unique():
        sub = out[out["position"] == pos_value].sort_values("positional_rank")
        assert list(sub["positional_rank"]) == list(range(1, len(sub) + 1))
        # vorp must be non-increasing as positional_rank increases
        vorps = sub["vorp"].to_numpy()
        assert (vorps[:-1] >= vorps[1:]).all()


def test_positional_rank_tie_break_by_gsis_id() -> None:
    """§5.1 #4 — equal-vorp rows tie-break by gsis_id ascending."""
    # Construct two QBs with identical VORP and explicit gsis_id ordering.
    df = pd.DataFrame(
        {
            "gsis_id": pd.Series(["00-1000002", "00-1000001"], dtype=_PYARROW_STR),
            "position": pd.Series(["QB", "QB"], dtype=_PYARROW_STR),
            "season_mean_fpts": [300.0, 300.0],
            "vorp": [50.0, 50.0],
            "replacement_fpts": [250.0, 250.0],
        }
    )
    vorp = VorpTableSchema.validate(df)
    # n_teams=2 with QB:1 → pool needs 2 QBs; we have exactly 2. (LeagueConfig
    # rejects n_teams=1 — must be >1. BENCH would force the pool past 2.)
    cfg = _make_config(
        roster_slots={RosterSlot.QB: 1},
        n_teams=2,
    )
    out = generate_snake_cheat_sheet(vorp, cfg)
    ranked = out.sort_values("positional_rank").reset_index(drop=True)
    assert ranked.loc[0, "gsis_id"] == "00-1000001"  # lower gsis_id first on tie
    assert ranked.loc[1, "gsis_id"] == "00-1000002"


def test_is_in_pool_matches_select_pool() -> None:
    """§5.1 #5 — set of is_in_pool=True gsis_ids equals _select_pool's output."""
    cfg = _make_config()
    vorp = _make_vorp_table({Position.QB: 8, Position.RB: 12, Position.WR: 12, Position.TE: 8})
    out = generate_snake_cheat_sheet(vorp, cfg)
    expected_pool = set(_select_pool(vorp, cfg))
    actual_pool = set(out.loc[out["is_in_pool"], "gsis_id"])
    assert actual_pool == expected_pool


def test_tier_dtype_is_nullable_int64() -> None:
    """§5.1 #6 — tier is pd.Int64Dtype(); in-pool rows int, out-of-pool rows NA."""
    cfg = _make_config()
    # Deliberately oversized inputs so some are out of pool.
    vorp = _make_vorp_table({Position.QB: 20, Position.RB: 30, Position.WR: 30, Position.TE: 20})
    out = generate_snake_cheat_sheet(vorp, cfg)
    assert out["tier"].dtype == pd.Int64Dtype()
    assert out.loc[out["is_in_pool"], "tier"].notna().all()
    assert out.loc[~out["is_in_pool"], "tier"].isna().all()


def test_tier_monotonic_with_vorp_within_position() -> None:
    """§5.1 #7 — tier T's min vorp >= tier T+1's max vorp (contiguous partition)."""
    cfg = _make_config()
    vorp = _make_vorp_table({Position.QB: 20, Position.RB: 30, Position.WR: 30, Position.TE: 20})
    out = generate_snake_cheat_sheet(vorp, cfg)
    in_pool = out[out["is_in_pool"]]
    for pos_value in in_pool["position"].unique():
        sub = in_pool[in_pool["position"] == pos_value]
        for t in sorted(sub["tier"].dropna().unique())[:-1]:
            tier_t_min = sub.loc[sub["tier"] == t, "vorp"].min()
            tier_t1_max = sub.loc[sub["tier"] == t + 1, "vorp"].max()
            assert tier_t_min >= tier_t1_max, (
                f"tier {t} min vorp ({tier_t_min}) < tier {t + 1} max vorp ({tier_t1_max})"
            )


def test_display_name_join_happy_path() -> None:
    """§5.1 #12 — every gsis_id in display_names gets its mapped name."""
    cfg = _make_config()
    # Pool needs ≥8 RBs/WRs under default n_teams=4 roster (incl. FLEX); use
    # the same sizing as the schema/regression tests above.
    vorp = _make_vorp_table({Position.QB: 8, Position.RB: 12, Position.WR: 12, Position.TE: 8})
    display = pd.DataFrame(
        {
            "gsis_id": vorp["gsis_id"].astype(_PYARROW_STR),
            "display_name": pd.Series(
                [f"Player {i}" for i in range(len(vorp))], dtype=_PYARROW_STR
            ),
        }
    )
    out = generate_snake_cheat_sheet(vorp, cfg, display_names=display)
    for _, row in out.iterrows():
        expected = display.loc[display["gsis_id"] == row["gsis_id"], "display_name"].iloc[0]
        assert row["display_name"] == expected


def test_display_name_missing_rows_fall_back_to_em_dash() -> None:
    """§5.1 #13 — uncovered gsis_ids get '—'."""
    cfg = _make_config()
    vorp = _make_vorp_table({Position.QB: 8, Position.RB: 12, Position.WR: 12, Position.TE: 8})
    # Cover only the first half.
    half = vorp.head(len(vorp) // 2)
    display = pd.DataFrame(
        {
            "gsis_id": half["gsis_id"].astype(_PYARROW_STR),
            "display_name": pd.Series(
                [f"Player {i}" for i in range(len(half))], dtype=_PYARROW_STR
            ),
        }
    )
    out = generate_snake_cheat_sheet(vorp, cfg, display_names=display)
    covered_ids = set(display["gsis_id"])
    for _, row in out.iterrows():
        if row["gsis_id"] in covered_ids:
            assert row["display_name"] != DISPLAY_NAME_FALLBACK
        else:
            assert row["display_name"] == DISPLAY_NAME_FALLBACK


def test_display_name_none_yields_all_em_dash() -> None:
    """§5.1 #14 — display_names=None → every row has display_name '—'."""
    cfg = _make_config()
    vorp = _make_vorp_table({Position.QB: 8, Position.RB: 12, Position.WR: 12, Position.TE: 8})
    out = generate_snake_cheat_sheet(vorp, cfg, display_names=None)
    assert (out["display_name"] == DISPLAY_NAME_FALLBACK).all()


def test_position_with_no_in_pool_rows_emits_rank_but_no_tier() -> None:
    """§5.1 #11 — a position whose players are all squeezed out of the pool
    (no in-pool rows but rows do exist in input) still appears in output with
    positional_rank populated and tier = NA.

    Construct: 2-team league with roster {QB:1} consuming exactly 2 players.
    Provide 2 QBs (in-pool) and 2 RBs (out-of-pool — RB not in roster_slots).
    RB rows have is_in_pool=False, tier=NA, positional_rank 1 and 2.
    """
    cfg = _make_config(
        n_teams=2,
        roster_slots={RosterSlot.QB: 1},
    )
    vorp = _make_vorp_table({Position.QB: 2, Position.RB: 2})
    out = generate_snake_cheat_sheet(vorp, cfg)
    rb = out[out["position"] == "RB"]
    assert len(rb) == 2
    assert (~rb["is_in_pool"]).all()
    assert rb["tier"].isna().all()
    assert list(rb.sort_values("positional_rank")["positional_rank"]) == [1, 2]


def test_missing_required_position_raises() -> None:
    """§5.1 #17 — LeagueConfig requires K but VORP has no K rows → raises from _select_pool."""
    cfg = _make_config(
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.K: 1,
            RosterSlot.BENCH: 1,
        },
    )
    vorp = _make_vorp_table({Position.QB: 8})  # no K rows
    with pytest.raises(ValueError, match=r"cannot fill \d+ K slots"):
        generate_snake_cheat_sheet(vorp, cfg)


def test_empty_input_raises() -> None:
    """§5.1 #18 — empty VORP input + non-empty config raises via _select_pool.

    The function calls _select_pool first, which raises if it can't fill the
    config's required positions. Empty input can't fill anything. We fail
    loudly rather than silently emit an empty cheat sheet.
    """
    cfg = _make_config()
    empty_df = pd.DataFrame(
        {
            "gsis_id": pd.Series([], dtype=_PYARROW_STR),
            "position": pd.Series([], dtype=_PYARROW_STR),
            "season_mean_fpts": pd.Series([], dtype=float),
            "vorp": pd.Series([], dtype=float),
            "replacement_fpts": pd.Series([], dtype=float),
        }
    )
    empty_vorp = VorpTableSchema.validate(empty_df)
    with pytest.raises(ValueError, match=r"cannot fill"):
        generate_snake_cheat_sheet(empty_vorp, cfg)


def test_tiers_per_position_zero_or_negative_raises() -> None:
    """§5.1 #20 — invalid tiers_per_position raises before computation."""
    cfg = _make_config()
    vorp = _make_vorp_table({Position.QB: 8, Position.RB: 12, Position.WR: 12, Position.TE: 8})
    with pytest.raises(ValueError, match="tiers_per_position must be >= 1"):
        generate_snake_cheat_sheet(vorp, cfg, tiers_per_position=0)
    with pytest.raises(ValueError, match="tiers_per_position must be >= 1"):
        generate_snake_cheat_sheet(vorp, cfg, tiers_per_position=-3)


def test_output_sorted_by_position_canonical_then_rank() -> None:
    """§5.1 #15 — output sorted (QB, RB, WR, TE, K, DST), then positional_rank asc."""
    cfg = _make_config()
    vorp = _make_vorp_table({Position.WR: 12, Position.QB: 8, Position.RB: 12, Position.TE: 8})
    out = generate_snake_cheat_sheet(vorp, cfg)
    canonical = [Position.QB.value, Position.RB.value, Position.WR.value, Position.TE.value]
    # Filter canonical to only positions actually present, preserving order.
    expected_positions: list[str] = []
    for pos_value in canonical:
        sub = out[out["position"] == pos_value]
        expected_positions.extend([pos_value] * len(sub))
    assert list(out["position"]) == expected_positions
    # Within each position, positional_rank is ascending.
    for pos_value in out["position"].unique():
        sub = out[out["position"] == pos_value]
        assert list(sub["positional_rank"]) == sorted(sub["positional_rank"])


def _attach_consensus_adp(vorp: pd.DataFrame, adp_by_gsis: dict[str, float]) -> pd.DataFrame:
    """Return a copy of a VORP table with a consensus_adp column (re-validated)."""
    out = vorp.copy()
    out["consensus_adp"] = pd.array(
        [adp_by_gsis.get(g) for g in out["gsis_id"]], dtype=pd.Float64Dtype()
    )
    return VorpTableSchema.validate(out)


# NOTE on configs: generate_snake_cheat_sheet calls _select_pool, which RAISES if the
# LeagueConfig pool can't be filled. So each fixture is paired with a config whose pool it
# fills. adp_delta is computed independently of pool membership, so a minimal pool is fine.


def test_cheat_sheet_without_adp_leaves_new_columns_na() -> None:
    """Weekly-path VORP table (no consensus_adp) -> consensus_adp/adp_delta all-NA,
    every other column unchanged (backward compatible)."""
    vorp = _make_vorp_table({Position.QB: 2, Position.RB: 2})
    cfg = _make_config(n_teams=2, roster_slots={RosterSlot.QB: 1, RosterSlot.RB: 1})
    sheet = generate_snake_cheat_sheet(vorp, cfg)
    assert sheet["consensus_adp"].isna().all()
    assert sheet["adp_delta"].isna().all()


def test_cheat_sheet_adp_delta_value_and_reach() -> None:
    """A late-ADP, high-VORP player is a 'value' (+delta); an early-ADP, low-VORP
    player is a 'reach' (-delta). Within position."""
    # Two QBs: best VORP (00-1000000) but LATE ADP -> value; worst VORP but EARLY ADP -> reach.
    # Pool sized to the fixture (QB:1 x 2 teams = 2) so _select_pool fills exactly.
    vorp = _make_vorp_table({Position.QB: 2})
    cfg = _make_config(n_teams=2, roster_slots={RosterSlot.QB: 1})
    # gsis ids from _make_vorp_table: 00-1000000 (higher vorp), 00-1000001 (lower vorp)
    adp = {"00-1000000": 50.0, "00-1000001": 5.0}
    sheet = generate_snake_cheat_sheet(_attach_consensus_adp(vorp, adp), cfg)
    by_gsis = sheet.set_index("gsis_id")
    # 00-1000000: vorp_rank 1, adp_rank 2 -> delta +1 (value)
    assert by_gsis.loc["00-1000000", "adp_delta"] == 1
    # 00-1000001: vorp_rank 2, adp_rank 1 -> delta -1 (reach)
    assert by_gsis.loc["00-1000001", "adp_delta"] == -1


def test_cheat_sheet_adp_delta_multiplayer_permutation() -> None:
    """3 single-position players, ADP order != VORP order, |delta| > 1 — pins the
    index-aligned (ADP-rank - VORP-rank) math against a positional-subtraction regression."""
    vorp = _make_vorp_table({Position.QB: 3})
    cfg = _make_config(n_teams=2, roster_slots={RosterSlot.QB: 1})
    # VORP order: 00-1000000 (vorp 2) > 00-1000001 (1) > 00-1000002 (0) -> vorp_rank 1/2/3.
    # ADP:        00-1000001 (10) < 00-1000002 (20) < 00-1000000 (30)   -> adp_rank  3/1/2.
    adp = {"00-1000000": 30.0, "00-1000001": 10.0, "00-1000002": 20.0}
    sheet = generate_snake_cheat_sheet(_attach_consensus_adp(vorp, adp), cfg)
    by_gsis = sheet.set_index("gsis_id")
    assert by_gsis.loc["00-1000000", "adp_delta"] == 2  # adp_rank 3 - vorp_rank 1
    assert by_gsis.loc["00-1000001", "adp_delta"] == -1  # adp_rank 1 - vorp_rank 2
    assert by_gsis.loc["00-1000002", "adp_delta"] == -1  # adp_rank 2 - vorp_rank 3


def test_cheat_sheet_null_adp_row_gets_null_delta() -> None:
    """A player missing consensus_adp gets null adp_delta but keeps its (passed-through)
    null consensus_adp; other players' deltas are unaffected (population isolation)."""
    vorp = _make_vorp_table({Position.WR: 3})
    cfg = _make_config(n_teams=2, roster_slots={RosterSlot.WR: 1})
    # Only two of three WRs have an ADP.
    adp = {"00-3000000": 10.0, "00-3000001": 20.0}  # 00-3000002 has none
    sheet = generate_snake_cheat_sheet(_attach_consensus_adp(vorp, adp), cfg)
    by_gsis = sheet.set_index("gsis_id")
    assert pd.isna(by_gsis.loc["00-3000002", "adp_delta"])
    assert pd.isna(by_gsis.loc["00-3000002", "consensus_adp"])
    # The two ADP-bearing WRs both rank 1/2 by both keys -> delta 0 each.
    assert by_gsis.loc["00-3000000", "adp_delta"] == 0
    assert by_gsis.loc["00-3000001", "adp_delta"] == 0


def test_cheat_sheet_with_adp_validates_schema() -> None:
    vorp = _make_vorp_table({Position.QB: 4, Position.RB: 6})
    cfg = _make_config(n_teams=2, roster_slots={RosterSlot.QB: 1, RosterSlot.RB: 1})
    adp = {g: float(i + 1) for i, g in enumerate(vorp["gsis_id"])}
    sheet = generate_snake_cheat_sheet(_attach_consensus_adp(vorp, adp), cfg)
    SnakeCheatSheetSchema.validate(sheet)
    assert "consensus_adp" in sheet.columns
    assert "adp_delta" in sheet.columns


def test_determinism_byte_identical_reruns() -> None:
    """§5.1 #16 — same inputs → byte-identical output frame."""
    cfg = _make_config()
    vorp = _make_vorp_table({Position.QB: 8, Position.RB: 12, Position.WR: 12, Position.TE: 8})
    display = pd.DataFrame(
        {
            "gsis_id": vorp["gsis_id"].astype(_PYARROW_STR),
            "display_name": pd.Series(
                [f"Player {i}" for i in range(len(vorp))], dtype=_PYARROW_STR
            ),
        }
    )
    out1 = generate_snake_cheat_sheet(vorp, cfg, display_names=display, tiers_per_position=6)
    out2 = generate_snake_cheat_sheet(vorp, cfg, display_names=display, tiers_per_position=6)
    pd.testing.assert_frame_equal(out1, out2)
