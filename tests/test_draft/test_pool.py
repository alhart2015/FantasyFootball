"""Tests for `projections.draft._pool._select_pool` after vorp-optional generalization."""

from __future__ import annotations

import pandas as pd

from projections.draft._pool import _select_pool
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, Position, RosterSlot, Ruleset


def _make_config() -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=4,
        budget=100,
        min_bid=1,
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 1,
        },
        ruleset=Ruleset.espn_ppr(),
    )


_POSITION_ID_PREFIX = {
    Position.QB: 1,
    Position.RB: 2,
    Position.WR: 3,
    Position.TE: 4,
}


def _bulk_rows(position: Position, count: int) -> list[dict[str, object]]:
    prefix = _POSITION_ID_PREFIX[position]
    return [
        {
            "gsis_id": f"00-{prefix}{i:06d}",
            "position": position.value,
            "season_mean_fpts": 200.0 - i,
        }
        for i in range(count)
    ]


def _df(rows: list[dict[str, object]], with_vorp: bool) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df["season_mean_fpts"] = df["season_mean_fpts"].astype("float64")
    if with_vorp:
        df["vorp"] = 0.0
    return df


def test_select_pool_accepts_input_without_vorp() -> None:
    cfg = _make_config()
    rows: list[dict[str, object]] = []
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE):
        rows.extend(_bulk_rows(pos, count=20))
    df_no_vorp = _df(rows, with_vorp=False)
    pool_no_vorp = _select_pool(df_no_vorp, cfg)
    assert len(pool_no_vorp) == cfg.total_pool_size


def test_select_pool_vorp_optional_matches_zero_vorp() -> None:
    """A vorp-less input must produce the same pool as a vorp-all-zero input.

    Zero vorp degenerates the tie-break to gsis_id alone, matching the absent-vorp path.
    """
    cfg = _make_config()
    rows: list[dict[str, object]] = []
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE):
        rows.extend(_bulk_rows(pos, count=20))
    pool_no_vorp = _select_pool(_df(rows, with_vorp=False), cfg)
    pool_zero_vorp = _select_pool(_df(rows, with_vorp=True), cfg)
    assert pool_no_vorp == pool_zero_vorp
