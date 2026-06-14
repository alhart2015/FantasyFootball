"""Tests for src/projections/draft/backtest/draft_basis.py.

Two tests:
  1. sleeper_adp correctly returns Sleeper-only ADP, ignoring ESPN's ~170 sentinel.
  2. build_draft_basis returns half-PPR season_mean_fpts and Sleeper consensus_adp.
"""

from __future__ import annotations

import pandas as pd

from projections.schemas import ExternalProjectionSchema, ProjectionSource

_ESPN = ProjectionSource.ESPN.value
_SLEEPER = ProjectionSource.SLEEPER.value

_STAT_COLS = [
    "passing_yards",
    "passing_tds",
    "interceptions",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "fumbles_lost",
]


def _external(
    rows: list[tuple[str, str, str, float | None, dict[str, float]]],
) -> pd.DataFrame:
    """Build an ExternalProjectionSchema-valid DataFrame.

    Each tuple is (source, gsis_id, position, adp, stats_dict).
    Only stats_dict keys are filled; absent stat columns are None.
    """
    records = []
    for source, gsis, position, adp, stats in rows:
        rec: dict[str, object] = {
            "source": source,
            "source_player_id": f"{source}-{gsis}",
            "gsis_id": gsis,
            "is_placeholder_gsis": False,
            "full_name": gsis,
            "position": position,
            "season": 2025,
            "asof": "2025-08-01",
            "adp": adp,
            "espn_draft_rank": None,
        }
        for col in _STAT_COLS:
            rec[col] = stats.get(col, None)
        records.append(rec)

    df = pd.DataFrame(records)
    return ExternalProjectionSchema.validate(df)


# ---------------------------------------------------------------------------
# Test 1 — sleeper_adp ignores the ESPN ~170 sentinel
# ---------------------------------------------------------------------------


def test_sleeper_adp_ignores_espn_sentinel() -> None:
    from projections.draft.backtest.draft_basis import sleeper_adp

    ext = _external(
        [
            (_ESPN, "00-0030001", "RB", 170.0, {"rushing_yards": 1100.0, "rushing_tds": 9.0}),
            (_SLEEPER, "00-0030001", "RB", 3.0, {}),
        ]
    )
    s = sleeper_adp(ext)
    # Should be Sleeper's 3.0, NOT 170 and NOT the (170+3)/2 mean
    assert float(s.loc["00-0030001"]) == 3.0


# ---------------------------------------------------------------------------
# Test 2 — build_draft_basis: half-PPR season_mean_fpts + Sleeper consensus_adp
# ---------------------------------------------------------------------------


def test_build_draft_basis_half_ppr_and_sleeper_adp() -> None:
    from projections.draft.backtest.draft_basis import build_draft_basis
    from projections.draft.league_config import LeagueConfig
    from projections.schemas import RosterSlot

    cfg = LeagueConfig(
        name="t",
        n_teams=2,
        budget=200,
        min_bid=1,
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 1,
            RosterSlot.WR: 1,
            RosterSlot.TE: 1,
        },
        ruleset="espn_half",  # type: ignore[arg-type]  # str preset resolved by field_validator
    )

    rows: list[tuple[str, str, str, float | None, dict[str, float]]] = []

    # One scored WR we assert on: 90 rec, 1200 rec yds, 8 rec TD
    # half-PPR: 90*0.5 + 1200*0.1 + 8*6 = 45 + 120 + 48 = 213.0
    wr_gid = "00-0040002"
    rows += [
        (
            _ESPN,
            wr_gid,
            "WR",
            170.0,
            {"receptions": 90.0, "receiving_yards": 1200.0, "receiving_tds": 8.0},
        ),
        (_SLEEPER, wr_gid, "WR", 1.0, {}),
    ]

    # 7 filler players (1 more WR, 2 QB, 2 RB, 2 TE) — use prefix "00-005" to avoid id collision
    filler: list[tuple[str, float | None, float | None]] = [
        ("QB", 4000.0, 30.0),
        ("QB", 3500.0, 25.0),
        ("RB", None, None),
        ("RB", None, None),
        ("WR", None, None),
        ("TE", None, None),
        ("TE", None, None),
    ]
    for i, (pos, py, ptd) in enumerate(filler):
        gid = f"00-005{i:04d}"
        if pos == "QB":
            stats: dict[str, float] = {
                "passing_yards": float(py or 0.0),
                "passing_tds": float(ptd or 0.0),
            }
        else:
            # >=2 non-zero fields so the filler is stat-bearing (build_consensus excludes
            # degenerate <2-field lines; real RB/WR/TE projections always have several).
            stats = {"rushing_yards": 200.0, "rushing_tds": 2.0}
        rows += [
            (_ESPN, gid, pos, 170.0, stats),
            (_SLEEPER, gid, pos, float(i + 5), {}),
        ]

    ext = _external(rows)
    table = build_draft_basis(ext, league_config=cfg)

    wr = table[table["gsis_id"] == wr_gid].iloc[0]
    assert abs(float(wr["season_mean_fpts"]) - 213.0) < 1e-6  # half-PPR points
    assert float(wr["consensus_adp"]) == 1.0  # Sleeper ADP, not 170
