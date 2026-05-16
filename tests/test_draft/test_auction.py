"""Unit tests for `projections.draft.auction.generate_auction_values` and helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.draft.auction import (
    _select_pool,
    generate_auction_values,  # noqa: F401 — re-exported for Task 5+ tests; smoke-imports the placeholder
)
from projections.draft.league_config import LeagueConfig
from projections.schemas import (
    AuctionValuesSchema,  # noqa: F401 — re-exported for Task 5+ tests; smoke-imports the schema
    Position,
    RosterSlot,
    Ruleset,
)


def _make_config(
    n_teams: int = 4,
    roster_slots: dict[RosterSlot, int] | None = None,
    budget: int = 100,
    min_bid: int = 1,
) -> LeagueConfig:
    return LeagueConfig(
        name="test",
        n_teams=n_teams,
        budget=budget,
        min_bid=min_bid,
        roster_slots=roster_slots
        or {
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 1,
        },
        ruleset=Ruleset.espn_ppr(),
    )


def _make_vorp_table(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a VORP-table-shaped DataFrame from a list of dicts.

    Required keys: gsis_id, position, season_mean_fpts, vorp.
    """
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(pd.StringDtype("pyarrow"))
    df["position"] = df["position"].astype(pd.StringDtype("pyarrow"))
    df["season_mean_fpts"] = df["season_mean_fpts"].astype("float64")
    df["vorp"] = df["vorp"].astype("float64")
    return df


def _bulk_position_rows(
    position: Position, count: int, base_fpts: float = 200.0
) -> list[dict[str, object]]:
    """Generate `count` rows for `position` with descending season_mean_fpts and matching VORP."""
    out: list[dict[str, object]] = []
    for i in range(count):
        out.append(
            {
                "gsis_id": f"00-{position.value}{i:05d}"[:10],
                "position": position.value,
                "season_mean_fpts": base_fpts - i,
                "vorp": (base_fpts - i) - 100.0,
            }
        )
    return out


def test_select_pool_size_matches_total_pool_size() -> None:
    cfg = _make_config()
    rows: list[dict[str, object]] = []
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE):
        rows.extend(_bulk_position_rows(pos, count=20))
    df = _make_vorp_table(rows)
    pool_ids = _select_pool(df, cfg)
    assert len(pool_ids) == cfg.total_pool_size


def test_select_pool_respects_position_quotas() -> None:
    cfg = _make_config()
    rows: list[dict[str, object]] = []
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE):
        rows.extend(_bulk_position_rows(pos, count=20))
    df = _make_vorp_table(rows)
    pool_ids = _select_pool(df, cfg)
    pool_df = df[df["gsis_id"].isin(pool_ids)]
    counts = pool_df["position"].value_counts().to_dict()
    # 4 teams x 1 QB = 4 QBs guaranteed; 4 x 2 = 8 RBs; 4 x 2 = 8 WRs; 4 x 1 = 4 TEs.
    # FLEX (4 slots) + BENCH (4 slots) fill from the best remaining.
    assert counts[Position.QB.value] >= 4
    assert counts[Position.RB.value] >= 8
    assert counts[Position.WR.value] >= 8
    assert counts[Position.TE.value] >= 4
    assert sum(counts.values()) == cfg.total_pool_size


def test_select_pool_omits_low_projection_players() -> None:
    """With only 12 RBs and a league that wants 12 in-pool RBs (8 strict + 4 FLEX-eligible),
    RB13+ should be out of pool."""
    cfg = _make_config(
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.BENCH: 0,
        }
    )
    rows: list[dict[str, object]] = []
    rows.extend(_bulk_position_rows(Position.QB, count=10))
    rows.extend(_bulk_position_rows(Position.RB, count=20))
    rows.extend(_bulk_position_rows(Position.WR, count=20))
    rows.extend(_bulk_position_rows(Position.TE, count=10))
    df = _make_vorp_table(rows)
    pool_ids = _select_pool(df, cfg)
    pool_df = df[df["gsis_id"].isin(pool_ids)]
    rb_pool = pool_df[pool_df["position"] == Position.RB.value]
    # 4 teams * 2 RB starters = 8 strict RBs. FLEX may add more.
    assert len(rb_pool) >= 8
    # The omitted RBs are the lowest-projection ones
    all_rb_ids = set(df[df["position"] == Position.RB.value]["gsis_id"])
    omitted_rb_ids = all_rb_ids - set(rb_pool["gsis_id"])
    if omitted_rb_ids:
        min_in_pool_fpts = rb_pool["season_mean_fpts"].min()
        omitted_max_fpts = df[df["gsis_id"].isin(omitted_rb_ids)]["season_mean_fpts"].max()
        assert omitted_max_fpts <= min_in_pool_fpts


def test_select_pool_omits_position_not_in_roster_slots() -> None:
    """A config without K or DST should not consume K or DST players into the pool."""
    cfg = _make_config(
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.BENCH: 2,
        }
    )
    rows: list[dict[str, object]] = []
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE, Position.K, Position.DST):
        rows.extend(_bulk_position_rows(pos, count=15))
    df = _make_vorp_table(rows)
    pool_ids = _select_pool(df, cfg)
    pool_df = df[df["gsis_id"].isin(pool_ids)]
    assert (pool_df["position"] != Position.K.value).all()
    assert (pool_df["position"] != Position.DST.value).all()


def test_select_pool_errors_on_missing_required_position() -> None:
    """If the config requires a position the VORP table doesn't cover, raise clearly."""
    cfg = _make_config(
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.K: 1,
            RosterSlot.BENCH: 0,
        }
    )
    rows: list[dict[str, object]] = []
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE):
        rows.extend(_bulk_position_rows(pos, count=10))
    df = _make_vorp_table(rows)
    with pytest.raises(ValueError, match=r"\bK\b"):
        _select_pool(df, cfg)
