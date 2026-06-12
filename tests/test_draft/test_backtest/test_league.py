"""Tests for simulate_league — Task 10 of H2H backtest plan."""

import pandas as pd

from projections.draft.assistant.strategy import RawVorpStrategy
from projections.draft.backtest.league import Calendar, simulate_league
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, RosterSlot, VorpTableSchema


def _cfg6() -> LeagueConfig:
    return LeagueConfig(
        name="t6",
        n_teams=6,
        budget=200,
        min_bid=1,
        roster_slots={RosterSlot.QB: 1, RosterSlot.RB: 1, RosterSlot.BENCH: 1},
        ruleset="espn_half",  # type: ignore[arg-type]  # str preset resolved by field_validator
    )


def _pool6() -> pd.DataFrame:
    """6-team pool with 3 elite players seat 1 will draft and 30 bot-fodder players.

    Elite players (gsis 00-0009001..3):
      - Enormous season_mean_fpts (1000) and VORP (900) — RawVorpStrategy grabs all three.
      - Enormous consensus_adp (9999) — bots treat as undraftable (they pick lowest-ADP first).

    Bot-fodder players: low VORP + low ADP (bots draft these in picks 2-18, seat 1 ignores).

    Projection = actual = season_mean_fpts every week, so seat 1 scores 2000 pts/week
    (QB+RB both start at 1000 each) versus ~19 pts/week for every other seat.
    """
    rows: list[dict[str, object]] = []

    # 3 elite players: QB, RB, and a QB bench fill for seat 1
    for idx, pos in enumerate(("QB", "RB", "QB"), start=1):
        rows.append(
            {
                "gsis_id": f"00-{9000 + idx:07d}",
                "position": pos,
                "season_mean_fpts": 1000.0,
                "vorp": 900.0,
                "replacement_fpts": 100.0,
                "consensus_adp": 9999.0,  # bots skip (they pick lowest ADP)
            }
        )

    # 30 bot-fodder players: low VORP, low ADP (bots grab these)
    for i in range(30):
        pos = "QB" if i % 2 == 0 else "RB"
        rows.append(
            {
                "gsis_id": f"00-{i + 1:07d}",
                "position": pos,
                "season_mean_fpts": 10.0 - i * 0.1,
                "vorp": 5.0 - i * 0.1,
                "replacement_fpts": 5.0,
                "consensus_adp": float(i + 1),
            }
        )

    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    return VorpTableSchema.validate(df)


def test_dominant_seat_is_champion_and_top_record() -> None:
    """Seat 1 (RawVorpStrategy) drafts the three elite players and wins every matchup.

    Elite players score 1000 pts/wk each; seat 1 starts QB+RB = 2000 pts/wk.
    All other seats draft bot-fodder scoring ~19 pts/wk. Seat 1 wins all 5 regular
    season games, seeds #1, and wins the 6-team single-elimination bracket.
    """
    cfg, pool = _cfg6(), _pool6()
    cal = Calendar(
        regular_weeks=tuple(range(1, 6)),
        playoff_weeks=(6, 7, 8),
        playoff_size=6,
    )
    seat_strategies: dict[int, object] = {1: RawVorpStrategy(), **{s: None for s in range(2, 7)}}
    labels = {1: "now_or_never", **{s: "bot" for s in range(2, 7)}}
    # projection == actual == the player's season_mean_fpts, every week 1..8
    proj = {
        (g, wk): float(m)
        for g, m in zip(pool["gsis_id"], pool["season_mean_fpts"], strict=False)
        for wk in range(1, 9)
    }
    actual = dict(proj)
    results = simulate_league(
        0,
        seat_strategies=seat_strategies,  # type: ignore[arg-type]  # dict[int,object] not Mapping[int,DraftStrategy|None]
        strategy_labels=labels,
        pool=pool,
        config=cfg,
        proj_lookup=proj,
        actual_lookup=actual,
        calendar=cal,
        jitter=8.0,
    )
    by_seat = {r.seat: r for r in results}
    assert by_seat[1].is_champion  # dominant seat wins it all
    assert by_seat[1].wins == 5 and by_seat[1].losses == 0  # record sums to regular weeks
    assert all(r.wins + r.losses == 5 for r in results)
    assert by_seat[1].points_for > 0
