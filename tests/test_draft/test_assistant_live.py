"""Tests for the LiveDraftSession controller and its helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from projections.draft.assistant.live import (
    LiveDraftSession,
    attach_names,
    build_session_strategy,
)
from projections.draft.assistant.state import DraftState
from projections.draft.assistant.strategy import NowOrNeverStrategy, RawVorpStrategy
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, Position, RosterSlot, Ruleset, validate_gsis_id


def _league() -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=12,
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 5,
        },
        ruleset=Ruleset.espn_ppr(),
    )


def _id_map() -> pd.DataFrame:
    ids = [f"00-000{i:04d}" for i in range(1, 40)]
    return pd.DataFrame(
        {
            "gsis_id": pd.array(ids, dtype=_PYARROW_STR),
            "position": pd.array(
                ["RB", "WR", "QB", "TE"] * 9 + ["RB", "WR", "QB"], dtype=_PYARROW_STR
            ),
            "full_name": pd.array([f"P{i}" for i in range(1, 40)], dtype=_PYARROW_STR),
            "team": pd.array(["KC"] * 39, dtype=_PYARROW_STR),
        }
    )


def _pool() -> pd.DataFrame:
    ids = [f"00-000{i:04d}" for i in range(1, 40)]
    return pd.DataFrame(
        {
            "gsis_id": pd.array(ids, dtype=_PYARROW_STR),
            "position": pd.array(
                ["RB", "WR", "QB", "TE"] * 9 + ["RB", "WR", "QB"], dtype=_PYARROW_STR
            ),
            "season_mean_fpts": [300.0 - i for i in range(39)],
            "vorp": [150.0 - i for i in range(39)],
            "replacement_fpts": [100.0] * 39,
            "consensus_adp": pd.array([float(i + 1) for i in range(39)], dtype=pd.Float64Dtype()),
        }
    )


class _FakeStrategy:
    """A DraftStrategy that returns the eligible pool sorted by vorp (no MC)."""

    def recommend(self, state: DraftState, pool: pd.DataFrame, config) -> pd.DataFrame:  # type: ignore[no-untyped-def]
        avail = pool[~pool["gsis_id"].isin(state.drafted_ids)].copy()
        avail = avail.sort_values("vorp", ascending=False).reset_index(drop=True)
        avail["p_available_next"] = pd.array([pd.NA] * len(avail), dtype=pd.Float64Dtype())
        avail["fills_starting_slot"] = True
        avail["score"] = avail["vorp"]
        avail["rank"] = pd.array(range(1, len(avail) + 1), dtype=pd.Int64Dtype())
        return avail


def _session(
    picks: list[str] | None = None,
    mode: str = "copilot",
    *,
    adp_jitter: float = 0.0,  # default: pure ADP order → deterministic bot picks in tests
    base_seed: int = 0,
    pool: pd.DataFrame | None = None,
) -> LiveDraftSession:
    return LiveDraftSession(
        league=_league(),
        my_slot=7,
        id_map=_id_map(),
        pool=_pool() if pool is None else pool,
        strategy=_FakeStrategy(),
        strategy_name="fake",
        mode=mode,  # type: ignore[arg-type]
        adp_jitter=adp_jitter,
        base_seed=base_seed,
        picks=[validate_gsis_id(p) for p in (picks or [])],
    )


def test_build_session_strategy_analytic_types() -> None:
    league = _league()
    assert isinstance(
        build_session_strategy(
            "raw_vorp", league=league, sigma=None, availability=None, n_sims=1, base_seed=0
        ),
        RawVorpStrategy,
    )
    assert isinstance(
        build_session_strategy(
            "now_or_never", league=league, sigma=None, availability=None, n_sims=1, base_seed=0
        ),
        NowOrNeverStrategy,
    )


def test_build_session_strategy_mc_requires_availability() -> None:
    league = _league()
    with pytest.raises(ValueError, match="availability"):
        build_session_strategy(
            "season_value", league=league, sigma=None, availability=None, n_sims=1, base_seed=0
        )


def test_build_session_strategy_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        build_session_strategy(
            "nope", league=_league(), sigma=None, availability=None, n_sims=1, base_seed=0
        )


def test_current_pick_and_my_pick_progression() -> None:
    s = _session()
    assert s.current_pick == 1
    assert not s.is_my_pick  # slot 1 on the clock, I'm slot 7
    s = _session(picks=[f"00-000{i:04d}" for i in range(1, 7)])  # 6 picks → pick 7 is mine
    assert s.current_pick == 7
    assert s.is_my_pick
    assert s.round_and_slot() == (1, 7)


def test_next_pick_number_snakes() -> None:
    s = _session(picks=[f"00-000{i:04d}" for i in range(1, 7)])  # standing at pick 7 (mine)
    # 12 teams, slot 7: next pick after #7 is #18 (snake).
    assert s.next_pick_number == 18


def test_is_complete_when_roster_full() -> None:
    total = _league().n_teams * _league().roster_size
    s = _session(picks=[f"00-000{i:04d}" for i in range(1, total + 1)])
    assert s.is_complete


def test_record_pick_appends_and_rejects_duplicate() -> None:
    s = _session()
    s.record_pick("00-0000001")
    assert s.picks == ["00-0000001"]
    with pytest.raises(ValueError, match="already drafted"):
        s.record_pick("00-0000001")


def test_record_pick_rejects_absent_from_id_map_for_my_pick() -> None:
    # Stand at pick 7 (my slot) so the recorded pick is mine; my picks need an id_map position.
    s = _session(picks=[f"00-000{i:04d}" for i in range(1, 7)])
    assert s.is_my_pick
    with pytest.raises(ValueError, match="id_map"):
        s.record_pick("00-0009999")


def test_record_pick_allows_off_id_map_opponent_pick() -> None:
    # At pick 1 (an opponent's slot) a player absent from id_map is fine — only my picks
    # need a resolvable position. This is what lets mock_advance record placeholder-gsis
    # rookies that live in the VORP pool but not yet in id_map (instead of crashing).
    s = _session()
    assert not s.is_my_pick
    s.record_pick("00-0009999")
    assert "00-0009999" in s.picks


def test_undo_pops_last() -> None:
    s = _session()
    s.record_pick("00-0000001")
    s.record_pick("00-0000002")
    assert s.undo() == "00-0000002"
    assert s.picks == ["00-0000001"]
    s.undo()
    assert s.undo() is None  # empty → None


def test_available_pool_excludes_drafted() -> None:
    s = _session(picks=["00-0000001", "00-0000002"])
    avail = s.available_pool()
    assert "00-0000001" not in set(avail["gsis_id"])
    assert len(avail) == len(_pool()) - 2


def test_recommendation_delegates_to_strategy() -> None:
    s = _session()
    rec = s.recommendation()
    assert next(iter(rec["gsis_id"])) == "00-0000001"  # highest vorp, undrafted
    assert "rank" in rec.columns


def test_recommendation_empty_when_complete() -> None:
    s = _session(picks=list(_pool()["gsis_id"]))  # whole pool drafted → nothing available
    assert s.recommendation().empty


def test_suggested_pick_is_deterministic_and_low_adp() -> None:
    s = _session()
    first = s.suggested_pick()
    again = s.suggested_pick()
    assert first == again  # stable across reruns for one board state
    # _session uses adp_jitter=0.0 → pure ADP order → lowest-ADP player wins.
    assert first == "00-0000001"


def test_suggested_pick_none_when_pool_empty() -> None:
    s = _session(picks=list(_pool()["gsis_id"]))
    assert s.suggested_pick() is None


def test_suggested_pick_reproducible_and_seeded_under_jitter() -> None:
    # The shared fixture pins adp_jitter=0.0, which zeroes the rng draws; jitter > 0 makes
    # the seed matter — guarding determinism + that base_seed is honored.
    def sess(seed: int) -> LiveDraftSession:
        return _session(adp_jitter=8.0, base_seed=seed)

    assert sess(0).suggested_pick() == sess(0).suggested_pick()  # same seed → reproducible
    assert sess(0).suggested_pick() != sess(1).suggested_pick()  # base_seed changes the pick


def test_suggested_pick_backfills_missing_consensus_adp() -> None:
    # consensus_adp is Optional in VorpTableSchema; a pool without it must not KeyError —
    # suggested_pick back-fills all-NA (no market signal) and bot_pick breaks ties on gsis.
    s = _session(pool=_pool().drop(columns=["consensus_adp"]))
    assert s.suggested_pick() == "00-0000001"  # all +inf → lowest gsis_id wins


def test_my_roster_view_assigns_slots_and_open_needs() -> None:
    picks = [f"00-000{i:04d}" for i in range(1, 7)]  # 6 opponent picks (ids 1..6)
    s = _session(picks=picks)
    # Pick #7 is mine (slot 7 of 12). id 9 is an RB in the fixture (index 8 → "RB").
    s.record_pick("00-0000009")
    view = s.my_roster_view()
    assert len(view.filled) == 1
    assert view.filled.iloc[0]["position"] == "RB"
    assert view.filled.iloc[0]["full_name"] == "P9"
    # An RB slot is now consumed; one RB starter slot remains open (RB:2).
    assert view.open_slots[RosterSlot.RB] == 1


def test_best_available_by_position_top_n() -> None:
    s = _session()
    best = s.best_available_by_position(top=2)
    assert set(best) <= set(Position)
    rb = best[Position.RB]
    assert len(rb) == 2
    assert list(rb["vorp"]) == sorted(rb["vorp"], reverse=True)


def test_attach_names_inserts_full_name() -> None:
    s = _session()
    named = attach_names(s.recommendation(), s.player_names)
    assert "full_name" in named.columns
    assert named.iloc[0]["full_name"] == "P1"


def test_mock_advance_stops_at_my_pick() -> None:
    s = _session(mode="mock")  # my_slot=7
    made = s.mock_advance_to_my_pick()
    assert len(made) == 6  # bots take picks 1..6
    assert s.is_my_pick  # standing at pick 7 (mine)
    assert s.current_pick == 7


def test_mock_advance_raises_in_copilot() -> None:
    s = _session(mode="copilot")
    with pytest.raises(RuntimeError, match="mock"):
        s.mock_advance_to_my_pick()


def test_mock_advance_tolerates_pool_player_absent_from_id_map() -> None:
    # A placeholder-gsis rookie can sit in the VORP pool but not in id_map; the bot may
    # pick it. mock_advance must not crash on such a pick (regression guard).
    extra = "00-0009998"  # not present in _id_map()
    pool = pd.concat(
        [
            _pool(),
            pd.DataFrame(
                {
                    "gsis_id": pd.array([extra], dtype=_PYARROW_STR),
                    "position": pd.array(["RB"], dtype=_PYARROW_STR),
                    "season_mean_fpts": [999.0],
                    "vorp": [999.0],
                    "replacement_fpts": [100.0],
                    "consensus_adp": pd.array([0.5], dtype=pd.Float64Dtype()),  # lowest → first
                }
            ),
        ],
        ignore_index=True,
    )
    s = _session(pool=pool, mode="mock")
    made = s.mock_advance_to_my_pick()  # must not raise
    assert extra in made  # the off-id_map rookie (lowest ADP) was the first bot pick
    assert s.is_my_pick


def test_roster_scorecard_matches_optimal_lineup() -> None:
    from projections.draft.assistant.roster_score import optimal_lineup_points

    picks = [f"00-000{i:04d}" for i in range(1, 7)]
    s = _session(picks=picks)
    s.record_pick("00-0000007")  # my pick #7
    mine = s.pool[s.pool["gsis_id"].isin(s.state().my_pick_ids)]
    expected = optimal_lineup_points(mine, s.league.roster_slots)
    assert s.roster_scorecard() == expected


def test_to_state_dict_is_cli_compatible(tmp_path: Path) -> None:
    from projections.draft.assistant.state import load_draft_state

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(_league().model_dump_json())
    s = _session(picks=["00-0000001", "00-0000002"])
    s.league_config_path = cfg_path
    d = s.to_state_dict()
    assert set(d) >= {"league_config", "my_slot", "picks", "mode", "strategy_name"}

    # load_draft_state must accept the saved superset unchanged.
    state_path = tmp_path / "session.json"
    state_path.write_text(json.dumps(d))
    loaded_state, _ = load_draft_state(state_path, _id_map())
    assert list(loaded_state.picks) == ["00-0000001", "00-0000002"]


def test_save_load_round_trip(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(_league().model_dump_json())

    s = _session(picks=["00-0000001"])
    s.league_config_path = cfg_path
    s.strategy_name = "raw_vorp"  # analytic → load needs no availability
    # Set every persisted config field to a distinctive NON-default so a dropped/renamed
    # key in to_state_dict or a wrong fallback in load() is caught (not just picks/slot).
    s.mode = "mock"
    s.adp_jitter = 4.0
    s.n_sims = 175
    s.season = 2024
    s.sigma = 9.0
    save_path = tmp_path / "session.json"
    s.save(save_path)

    loaded = LiveDraftSession.load(save_path, id_map=_id_map(), pool=_pool())
    assert loaded.picks == ["00-0000001"]
    assert loaded.strategy_name == "raw_vorp"
    assert loaded.my_slot == 7
    assert loaded.mode == "mock"
    assert loaded.adp_jitter == 4.0
    assert loaded.n_sims == 175
    assert loaded.season == 2024
    assert loaded.sigma == 9.0


def test_build_session_strategy_now_or_never_floored() -> None:
    from projections.draft.assistant.live import BOARD_STRATEGIES, build_session_strategy
    from projections.draft.assistant.strategy import NowOrNeverFlooredStrategy

    strat = build_session_strategy(
        "now_or_never_floored",
        league=_league(),
        sigma=None,
        availability=None,  # analytic: must NOT require availability
        n_sims=300,
        base_seed=0,
        floor=55.0,
        floor_weight=2.0,
    )
    assert isinstance(strat, NowOrNeverFlooredStrategy)
    assert strat.floor == 55.0
    assert strat.floor_weight == 2.0
    assert "now_or_never_floored" in BOARD_STRATEGIES


def test_build_session_strategy_floored_defaults() -> None:
    from projections.draft.assistant.live import build_session_strategy
    from projections.draft.assistant.strategy import (
        _DEFAULT_FLOOR,
        _DEFAULT_FLOOR_WEIGHT,
        NowOrNeverFlooredStrategy,
    )

    strat = build_session_strategy(
        "now_or_never_floored",
        league=_league(),
        sigma=None,
        availability=None,
        n_sims=300,
        base_seed=0,
    )
    assert isinstance(strat, NowOrNeverFlooredStrategy)
    # Compare against the constants (not literals) so the A/B default change stays green.
    assert strat.floor == _DEFAULT_FLOOR
    assert strat.floor_weight == _DEFAULT_FLOOR_WEIGHT
