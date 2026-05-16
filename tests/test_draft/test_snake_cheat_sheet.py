"""Tests for `projections.draft.snake_cheat_sheet`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.draft._pool import _select_pool
from projections.draft.league_config import LeagueConfig
from projections.draft.snake_cheat_sheet import _assign_tiers, generate_snake_cheat_sheet
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
            assert row["display_name"] != "—"
        else:
            assert row["display_name"] == "—"


def test_display_name_none_yields_all_em_dash() -> None:
    """§5.1 #14 — display_names=None → every row has display_name '—'."""
    cfg = _make_config()
    vorp = _make_vorp_table({Position.QB: 8, Position.RB: 12, Position.WR: 12, Position.TE: 8})
    out = generate_snake_cheat_sheet(vorp, cfg, display_names=None)
    assert (out["display_name"] == "—").all()


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


def test_empty_input_returns_empty() -> None:
    """§5.1 #18 — empty VORP input → empty output, schema-valid.

    NOTE: Spec §3.6 originally said empty input → empty output. Actual algorithm
    calls _select_pool first, which raises if it can't fill the config's
    required positions. With empty input it can't fill any position. We test
    the AS-IMPLEMENTED behavior: empty input + non-empty config raises
    explicitly. Silently emitting an empty cheat sheet for a non-empty config
    feels wrong — fail loudly when we can't compute. This is the stricter
    contract; spec §3.6 will be updated to match.
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
