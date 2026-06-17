"""Tests for the projected-vs-projected league simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.league_projection import (
    gauntlet_schedule,
    project_draft,
    team_weekly_points,
)
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, RosterSlot, Ruleset

_SLOTS = {
    RosterSlot.QB: 1,
    RosterSlot.RB: 2,
    RosterSlot.WR: 3,
    RosterSlot.TE: 1,
    RosterSlot.FLEX: 1,
    RosterSlot.BENCH: 9,
}


def _roster(gsis: list[str], pos: list[str], mean: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": pd.array(gsis, dtype=_PYARROW_STR),
            "position": pd.array(pos, dtype=_PYARROW_STR),
            "season_mean_fpts": mean,
            "is_rookie": [False] * len(gsis),
        }
    )


def test_team_weekly_points_shape_and_higher_means_score_more() -> None:
    params = VarianceParams.load()
    weeks = list(range(1, 14))
    ids = [f"00-000{i:04d}" for i in range(1, 9)]
    pos = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "RB"]
    avail = PlayerAvailability(p={g: 1.0 for g in ids}, bye={})
    strong = team_weekly_points(
        _roster(ids, pos, [300.0] * 8),
        avail,
        params,
        n_sims=400,
        weeks=weeks,
        roster_slots=_SLOTS,
        rng=np.random.default_rng(0),
    )
    weak = team_weekly_points(
        _roster(ids, pos, [120.0] * 8),
        avail,
        params,
        n_sims=400,
        weeks=weeks,
        roster_slots=_SLOTS,
        rng=np.random.default_rng(0),
    )
    assert strong.shape == (400, 13)
    assert strong.mean() > weak.mean()  # higher projections -> more lineup points


@pytest.mark.parametrize("n_teams", [10, 12, 16])
def test_gauntlet_schedule_is_a_valid_round_robin(n_teams: int) -> None:
    sched = gauntlet_schedule(n_teams, n_weeks=13)
    assert len(sched) == 13
    for week in sched:
        seats = [s for pair in week for s in pair]
        assert sorted(seats) == list(range(1, n_teams + 1))  # everyone plays exactly once
    # slot 1 plays slot 2 in wk1, slot 3 in wk2 (the rotating gauntlet)
    assert (1, 2) in [tuple(sorted(p)) for p in sched[0]]
    assert (1, 3) in [tuple(sorted(p)) for p in sched[1]]


def test_gauntlet_schedule_rejects_odd() -> None:
    with pytest.raises(ValueError, match="even"):
        gauntlet_schedule(11, n_weeks=13)


def _symmetric_league(n_teams: int) -> tuple[dict[int, list[str]], pd.DataFrame]:
    """n_teams rosters with identical position/points multisets (distinct gsis) -> symmetric."""
    rows: list[dict[str, object]] = []
    rosters: dict[int, list[str]] = {}
    template_pos = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "RB", "WR", "QB"]
    template_mean = [280.0, 230.0, 200.0, 220.0, 190.0, 170.0, 150.0, 140.0, 130.0, 250.0]
    for slot in range(1, n_teams + 1):
        ids = []
        for j, (po, mn) in enumerate(zip(template_pos, template_mean, strict=True)):
            g = f"00-{slot:02d}{j:05d}"
            ids.append(g)
            rows.append({"gsis_id": g, "position": po, "season_mean_fpts": mn, "is_rookie": False})
        rosters[slot] = ids
    pool = pd.DataFrame(rows)
    pool["gsis_id"] = pool["gsis_id"].astype(_PYARROW_STR)
    pool["position"] = pool["position"].astype(_PYARROW_STR)
    return rosters, pool


def _config(n_teams: int) -> LeagueConfig:
    return LeagueConfig(name="t", n_teams=n_teams, roster_slots=_SLOTS, ruleset=Ruleset.espn_half())


def test_symmetric_league_metrics_near_uniform_baseline() -> None:
    n = 12
    rosters, pool = _symmetric_league(n)
    avail = PlayerAvailability(p={g: 1.0 for g in pool["gsis_id"].astype(str)}, bye={})
    res = project_draft(
        rosters,
        pool,
        avail,
        VarianceParams.load(),
        league_config=_config(n),
        n_sims=3000,
        rng=np.random.default_rng(1),
    )
    assert set(res) == set(range(1, n + 1))
    champ = np.array([res[s].champ_pct for s in range(1, n + 1)])
    assert abs(champ.sum() - 1.0) < 1e-9  # exactly one champion per sim
    # symmetric -> no seat dominates (1/12 ~ 0.083)
    assert champ.max() < 0.25 and champ.min() > 0.0
    assert all(0.35 < res[s].make_playoffs_pct < 0.65 for s in range(1, n + 1))  # ~6/12


def test_project_draft_is_deterministic() -> None:
    rosters, pool = _symmetric_league(10)
    avail = PlayerAvailability(p={g: 1.0 for g in pool["gsis_id"].astype(str)}, bye={})
    cfg, nsims = _config(10), 500
    a = project_draft(
        rosters,
        pool,
        avail,
        VarianceParams.load(),
        league_config=cfg,
        n_sims=nsims,
        rng=np.random.default_rng(7),
    )
    b = project_draft(
        rosters,
        pool,
        avail,
        VarianceParams.load(),
        league_config=cfg,
        n_sims=nsims,
        rng=np.random.default_rng(7),
    )
    assert a[1] == b[1]


def test_stronger_roster_wins_more() -> None:
    n = 12
    rosters, pool = _symmetric_league(n)
    boost = pool["gsis_id"].astype(str).isin(rosters[1])
    pool.loc[boost, "season_mean_fpts"] = pool.loc[boost, "season_mean_fpts"] + 120.0
    avail = PlayerAvailability(p={g: 1.0 for g in pool["gsis_id"].astype(str)}, bye={})
    res = project_draft(
        rosters,
        pool,
        avail,
        VarianceParams.load(),
        league_config=_config(n),
        n_sims=2000,
        rng=np.random.default_rng(2),
    )
    assert res[1].champ_pct > 0.3
    assert res[1].make_playoffs_pct > max(res[s].make_playoffs_pct for s in range(2, n + 1))


def test_project_draft_requires_at_least_six_teams() -> None:
    rosters, pool = _symmetric_league(4)
    avail = PlayerAvailability(p={g: 1.0 for g in pool["gsis_id"].astype(str)}, bye={})
    with pytest.raises(ValueError, match="at least"):
        project_draft(
            rosters,
            pool,
            avail,
            VarianceParams.load(),
            league_config=_config(4),
            n_sims=50,
            rng=np.random.default_rng(0),
        )
