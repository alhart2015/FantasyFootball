"""Tests for the full-draft simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.draft.assistant.simulation import _draft_picks, simulate_draft
from projections.draft.assistant.state import DraftState
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, RosterSlot, Ruleset


def _config(n_teams: int = 2) -> LeagueConfig:
    # roster_size = sum of non-IR slots = 3 -> 2 teams * 3 = 6 picks.
    return LeagueConfig(
        name="t",
        n_teams=n_teams,
        roster_slots={RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.BENCH: 1},
        ruleset=Ruleset.espn_ppr(),
    )


def _pool(n: int = 12) -> pd.DataFrame:
    # n players, descending fpts/vorp, ascending adp, alternating RB/WR.
    gsis = [f"00-00000{i:02d}" for i in range(1, n + 1)]
    return pd.DataFrame(
        {
            "gsis_id": pd.array(gsis, dtype=_PYARROW_STR),
            "position": pd.array(["RB" if i % 2 else "WR" for i in range(n)], dtype=_PYARROW_STR),
            "season_mean_fpts": [float(200 - i) for i in range(n)],
            "vorp": [float(100 - i) for i in range(n)],
            "replacement_fpts": [100.0] * n,
            "consensus_adp": pd.array([float(i + 1) for i in range(n)], dtype=pd.Float64Dtype()),
        }
    )


class _BestFpts:
    """Hero fake: draft the highest-season_mean_fpts not-yet-drafted player."""

    def recommend(
        self, state: DraftState, pool: pd.DataFrame, config: LeagueConfig
    ) -> pd.DataFrame:
        avail = pool[~pool["gsis_id"].isin(state.drafted_ids)]
        return avail.sort_values(["season_mean_fpts", "gsis_id"], ascending=[False, True])


class _WorstFpts:
    """Hero fake: draft the lowest-season_mean_fpts not-yet-drafted player."""

    def recommend(
        self, state: DraftState, pool: pd.DataFrame, config: LeagueConfig
    ) -> pd.DataFrame:
        avail = pool[~pool["gsis_id"].isin(state.drafted_ids)]
        return avail.sort_values(["season_mean_fpts", "gsis_id"], ascending=[True, True])


def test_hero_gets_exactly_roster_size_picks() -> None:
    cfg = _config(n_teams=2)
    rng = np.random.default_rng(0)
    roster = simulate_draft(
        _BestFpts(), my_slot=1, pool=_pool(), config=cfg, adp_jitter=2.0, rng=rng
    )
    assert len(roster) == cfg.roster_size  # 3


def test_snake_pick_order_hand_computed() -> None:
    # n_teams=2, roster_size=3 -> 6 picks; snake slots p1=s1,p2=s2,p3=s2,p4=s1,p5=s1,p6=s2.
    # Hero = slot 1 -> picks 1,4,5. adp_jitter=0 -> bots deterministically take lowest adp;
    # _BestFpts takes highest fpts (= lowest gsis here). Pool fpts desc, adp asc by gsis.
    # p1 hero->01, p2 bot->02, p3 bot->03, p4 hero->04, p5 hero->05, p6 bot->06.
    cfg = _config(n_teams=2)
    picks = _draft_picks(
        _BestFpts(),
        my_slot=1,
        pool=_pool(),
        config=cfg,
        adp_jitter=0.0,
        rng=np.random.default_rng(0),
    )
    assert picks == [
        "00-0000001",
        "00-0000002",
        "00-0000003",
        "00-0000004",
        "00-0000005",
        "00-0000006",
    ]
    # The harness returns exactly the hero's three snake seats (picks 1,4,5).
    roster = simulate_draft(
        _BestFpts(),
        my_slot=1,
        pool=_pool(),
        config=cfg,
        adp_jitter=0.0,
        rng=np.random.default_rng(0),
    )
    assert set(roster["gsis_id"]) == {"00-0000001", "00-0000004", "00-0000005"}


def test_determinism_same_seed_same_roster() -> None:
    cfg = _config()
    r1 = simulate_draft(
        _BestFpts(),
        my_slot=2,
        pool=_pool(),
        config=cfg,
        adp_jitter=3.0,
        rng=np.random.default_rng(7),
    )
    r2 = simulate_draft(
        _BestFpts(),
        my_slot=2,
        pool=_pool(),
        config=cfg,
        adp_jitter=3.0,
        rng=np.random.default_rng(7),
    )
    assert list(r1["gsis_id"]) == list(r2["gsis_id"])


def test_different_seed_changes_the_field() -> None:
    # Compare the FULL deterministic draft (12 picks) across two seeds, not just the
    # 3-player hero roster -- far lower chance of coincidental equality. Different bot
    # noise -> different board. (Each draft is itself deterministic given its seed.)
    cfg = _config()
    picks1 = _draft_picks(
        _BestFpts(),
        my_slot=2,
        pool=_pool(),
        config=cfg,
        adp_jitter=5.0,
        rng=np.random.default_rng(1),
    )
    picks2 = _draft_picks(
        _BestFpts(),
        my_slot=2,
        pool=_pool(),
        config=cfg,
        adp_jitter=5.0,
        rng=np.random.default_rng(2),
    )
    assert picks1 != picks2


def test_paired_field_identical_before_hero_diverges() -> None:
    # my_slot=2: pick #1 is a bot; same seed -> identical regardless of hero strategy.
    cfg = _config()
    picks_a = _draft_picks(
        _BestFpts(),
        my_slot=2,
        pool=_pool(),
        config=cfg,
        adp_jitter=4.0,
        rng=np.random.default_rng(11),
    )
    picks_b = _draft_picks(
        _WorstFpts(),
        my_slot=2,
        pool=_pool(),
        config=cfg,
        adp_jitter=4.0,
        rng=np.random.default_rng(11),
    )
    assert picks_a[0] == picks_b[0]  # the pre-divergence bot pick
