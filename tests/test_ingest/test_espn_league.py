"""Unit tests for the ESPN private-league client.

The network call is not exercised here; every parser is pure over a synthetic payload
shaped like a real ESPN multi-view response.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from projections.ingest.espn_league import (
    DEFAULT_VIEWS,
    ESPN_LINEUP_SLOTS,
    ESPN_PRO_TEAMS,
    EspnCredentials,
    EspnLeagueError,
    build_league_config,
    find_my_team_id,
    parse_draft_order,
    parse_draft_picks,
    parse_draft_settings,
    parse_roster_slots,
    parse_rosters,
    parse_ruleset,
    parse_schedule,
    parse_teams,
    pro_team_code,
    scoring_family,
    team_records,
    write_league_snapshot,
)
from projections.schemas import _RULESET_NAME_VALUES, RosterSlot, Team

_MY_SWID = "{AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE}"


def _payload(**overrides: Any) -> dict[str, Any]:
    """A minimal but realistically shaped half-PPR 12-team auction league."""
    base: dict[str, Any] = {
        "id": 856974,
        "settings": {
            "name": "GOAT Steins and Slow Combines",
            "size": 12,
            "draftSettings": {
                "type": "AUCTION",
                "auctionBudget": 200,
                # 2026-08-30T18:00:00Z
                "date": 1788112800000,
                "keeperCount": 0,
                "timePerSelection": 30,
            },
            "rosterSettings": {
                "lineupSlotCounts": {
                    "0": 1,  # QB
                    "1": 0,  # TQB — zero, must be skipped
                    "2": 2,  # RB
                    "4": 2,  # WR
                    "6": 1,  # TE
                    "16": 1,  # D/ST
                    "17": 1,  # K
                    "20": 5,  # BENCH
                    "21": 1,  # IR
                    "23": 2,  # FLEX
                }
            },
            "scoringSettings": {
                "scoringItems": [
                    {"statId": 3, "points": 0.04},  # 25 passing yds/pt
                    {"statId": 4, "points": 5.0},
                    {"statId": 20, "points": -2.0},
                    {"statId": 24, "points": 0.1},
                    {"statId": 25, "points": 6.0},
                    {"statId": 42, "points": 0.1},
                    {"statId": 43, "points": 6.0},
                    {"statId": 53, "points": 0.5},  # half-PPR
                    {"statId": 72, "points": -2.0},
                    {"statId": 19, "points": 2.0},
                    {"statId": 26, "points": 2.0},
                    {"statId": 44, "points": 2.0},
                    {"statId": 83, "points": 3.0},  # a kicking category Ruleset cannot model
                ]
            },
        },
        "members": [
            {"id": _MY_SWID, "displayName": "alden"},
            {"id": "{11111111-2222-3333-4444-555555555555}", "displayName": "will"},
        ],
        "teams": [
            {
                "id": 1,
                "name": "Two Kids Two Steins",
                "abbrev": "TK2S",
                "owners": [_MY_SWID],
                "roster": {
                    "entries": [
                        {
                            "lineupSlotId": 0,
                            "acquisitionType": "DRAFT",
                            "playerPoolEntry": {
                                "player": {
                                    "id": 3918298,
                                    "fullName": "Josh Allen",
                                    "defaultPositionId": 1,
                                    "proTeamId": 2,
                                }
                            },
                        },
                        {
                            "lineupSlotId": 2,
                            "acquisitionType": "DRAFT",
                            "playerPoolEntry": {
                                "player": {
                                    "id": 4429795,
                                    "fullName": "Jahmyr Gibbs",
                                    "defaultPositionId": 2,
                                    "proTeamId": 8,
                                }
                            },
                        },
                    ]
                },
            },
            {
                "id": 2,
                "location": "Lemon",
                "nickname": "Party",
                "abbrev": "LP",
                "owners": ["{11111111-2222-3333-4444-555555555555}"],
                "roster": {"entries": []},
            },
        ],
        "draftDetail": {
            "drafted": True,
            "picks": [
                {"overallPickNumber": 1, "teamId": 1, "playerId": 4429795, "bidAmount": 86},
                {"overallPickNumber": 2, "teamId": 1, "playerId": 3918298, "bidAmount": 25},
                # A player drafted then dropped: no roster entry to name him.
                {"overallPickNumber": 3, "teamId": 2, "playerId": 999999, "bidAmount": 4},
            ],
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["{ABC-DEF}", "ABC-DEF"])
def test_swid_is_normalized_to_braces(raw: str) -> None:
    assert EspnCredentials(swid=raw, espn_s2="s2").normalized_swid == "{ABC-DEF}"


def test_cookie_header_shape() -> None:
    creds = EspnCredentials(swid="ABC", espn_s2="  s2value  ")
    assert creds.cookie_header() == "SWID={ABC}; espn_s2=s2value"


def test_blank_swid_is_rejected_but_named_as_identity_only() -> None:
    """ESPN never validates SWID (probed against league 856974, 2026-08-24: espn_s2 plus a
    wrong, zeroed, or absent SWID all returned 200; the right SWID with no espn_s2 returned
    401). It is still required, because it is the only thing that identifies which team is
    the caller's — and the error has to say so, or a wrong SWID reads as a login problem."""
    with pytest.raises(EspnLeagueError, match="does not authenticate"):
        EspnCredentials(swid="   ", espn_s2="s2")


def test_blank_espn_s2_is_rejected_as_the_authenticator() -> None:
    with pytest.raises(EspnLeagueError, match="what ESPN checks"):
        EspnCredentials(swid="{ABC}", espn_s2="  ")


def test_from_env_requires_both(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ESPN_SWID", "{ABC}")
    monkeypatch.delenv("ESPN_S2", raising=False)
    assert EspnCredentials.from_env() is None
    monkeypatch.setenv("ESPN_S2", "s2")
    creds = EspnCredentials.from_env()
    assert creds is not None and creds.espn_s2 == "s2"


def test_from_file_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "creds.json"
    path.write_text(json.dumps({"swid": "{ABC}", "espn_s2": "s2"}), encoding="utf-8")
    creds = EspnCredentials.from_file(path)
    assert creds is not None and creds.normalized_swid == "{ABC}"
    assert EspnCredentials.from_file(tmp_path / "missing.json") is None


def test_from_file_rejects_incomplete_json(tmp_path: Path) -> None:
    path = tmp_path / "creds.json"
    path.write_text(json.dumps({"swid": "{ABC}"}), encoding="utf-8")
    with pytest.raises(EspnLeagueError, match="must contain both"):
        EspnCredentials.from_file(path)


def test_resolve_prefers_env_over_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "creds.json"
    path.write_text(json.dumps({"swid": "{FILE}", "espn_s2": "file"}), encoding="utf-8")
    monkeypatch.setenv("ESPN_SWID", "{ENV}")
    monkeypatch.setenv("ESPN_S2", "env")
    assert EspnCredentials.resolve(path).normalized_swid == "{ENV}"


def test_resolve_raises_with_actionable_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ESPN_SWID", raising=False)
    monkeypatch.delenv("ESPN_S2", raising=False)
    with pytest.raises(EspnLeagueError, match=r"fantasy\.espn\.com"):
        EspnCredentials.resolve(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# Id maps
# ---------------------------------------------------------------------------


def test_pro_team_ids_all_resolve_to_canonical_teams() -> None:
    """Every id in the map must survive normalize_team_code — this is what catches a
    typo'd or stale abbreviation (WSH, JAX, LV) at test time rather than mid-draft."""
    resolved = {pro_team_code(pid) for pid in ESPN_PRO_TEAMS}
    assert None not in resolved
    assert len(resolved) == 32, "ESPN_PRO_TEAMS must cover all 32 teams exactly once"


def test_pro_team_code_handles_free_agent_and_unknown() -> None:
    assert pro_team_code(0) is None
    assert pro_team_code(999) is None


def test_aliased_pro_teams_normalize() -> None:
    assert pro_team_code(28) is Team.WAS  # ESPN says WSH
    assert pro_team_code(30) is Team.JAC  # ESPN says JAX


def test_lineup_slot_map_collapses_all_three_flex_ids() -> None:
    assert ESPN_LINEUP_SLOTS[3] is RosterSlot.FLEX
    assert ESPN_LINEUP_SLOTS[5] is RosterSlot.FLEX
    assert ESPN_LINEUP_SLOTS[23] is RosterSlot.FLEX


# ---------------------------------------------------------------------------
# Settings parsers
# ---------------------------------------------------------------------------


def test_parse_roster_slots_skips_zero_counts() -> None:
    slots = parse_roster_slots(_payload())
    assert slots == {
        RosterSlot.QB: 1,
        RosterSlot.RB: 2,
        RosterSlot.WR: 2,
        RosterSlot.TE: 1,
        RosterSlot.DST: 1,
        RosterSlot.K: 1,
        RosterSlot.BENCH: 5,
        RosterSlot.IR: 1,
        RosterSlot.FLEX: 2,
    }


def test_parse_roster_slots_sums_distinct_flex_ids() -> None:
    """RB/WR (3), WR/TE (5) and RB/WR/TE (23) all map to FLEX; counts must add, not clobber."""
    payload = _payload()
    payload["settings"]["rosterSettings"]["lineupSlotCounts"] = {"0": 1, "3": 1, "5": 1, "23": 2}
    assert parse_roster_slots(payload)[RosterSlot.FLEX] == 4


def test_parse_roster_slots_warns_on_idp_slots(caplog: pytest.LogCaptureFixture) -> None:
    payload = _payload()
    payload["settings"]["rosterSettings"]["lineupSlotCounts"] = {"0": 1, "2": 2, "10": 3}
    slots = parse_roster_slots(payload)
    assert RosterSlot.QB in slots
    assert "10x3" in caplog.text


def test_parse_roster_slots_requires_the_settings_view() -> None:
    with pytest.raises(EspnLeagueError, match="mSettings"):
        parse_roster_slots({"settings": {}})


def test_parse_ruleset_inverts_yardage_and_reads_half_ppr() -> None:
    ruleset, _ = parse_ruleset(_payload(), name="test")
    assert ruleset.passing_yds_per_pt == pytest.approx(25.0)
    assert ruleset.rushing_yds_per_pt == pytest.approx(10.0)
    assert ruleset.receiving_yds_per_pt == pytest.approx(10.0)
    assert ruleset.reception_pts == pytest.approx(0.5)
    assert ruleset.passing_td_pts == pytest.approx(5.0)
    assert ruleset.interception_pts == pytest.approx(-2.0)
    assert ruleset.two_pt_pts == pytest.approx(2.0)


def test_parse_ruleset_reports_categories_it_cannot_model() -> None:
    _, notes = parse_ruleset(_payload(), name="test")
    assert any("statId 83" in note for note in notes), notes


def test_parse_ruleset_rejects_non_positive_yardage() -> None:
    """A zero points-per-yard cannot be inverted. Failing loudly beats a silent default —
    every downstream projection depends on this number."""
    payload = _payload()
    payload["settings"]["scoringSettings"]["scoringItems"] = [{"statId": 3, "points": 0.0}]
    with pytest.raises(EspnLeagueError, match="yards-per-point"):
        parse_ruleset(payload, name="test")


def test_parse_ruleset_flags_disagreeing_two_point_values() -> None:
    payload = _payload()
    payload["settings"]["scoringSettings"]["scoringItems"] = [
        {"statId": 19, "points": 4.0},
        {"statId": 26, "points": 2.0},
        {"statId": 44, "points": 2.0},
    ]
    ruleset, notes = parse_ruleset(payload, name="test")
    assert ruleset.two_pt_pts == pytest.approx(2.0)  # rushing wins
    assert any("Two-point" in note for note in notes), notes


def test_parse_ruleset_flags_position_overrides() -> None:
    payload = _payload()
    payload["settings"]["scoringSettings"]["scoringItems"] = [
        {"statId": 43, "points": 6.0, "pointsOverrides": {"16": 4.0}}
    ]
    _, notes = parse_ruleset(payload, name="test")
    assert any("pointsOverrides" in note for note in notes), notes


def test_parse_ruleset_notes_missing_categories() -> None:
    payload = _payload()
    payload["settings"]["scoringSettings"]["scoringItems"] = [{"statId": 53, "points": 1.0}]
    _, notes = parse_ruleset(payload, name="test")
    assert any("defaults apply" in note for note in notes), notes


@pytest.mark.parametrize(
    ("reception_pts", "expected"),
    [(1.0, "ESPN_PPR"), (0.5, "ESPN_HALF"), (0.0, "STANDARD")],
)
def test_scoring_family_tags_the_standard_values(reception_pts: float, expected: str) -> None:
    assert scoring_family(reception_pts) == (expected, True)


def test_scoring_family_snaps_an_unusual_value_to_the_nearest_family() -> None:
    family, exact = scoring_family(0.6)
    assert (family, exact) == ("ESPN_HALF", False)


def test_ruleset_name_is_always_a_whitelisted_family() -> None:
    """`Ruleset.name` is not free text: `_RULESET_NAME_VALUES` whitelists it and
    `ConsensusProjectionSchema` validates against that list, so a descriptive per-league
    name (e.g. "Critts 2025_scoring") makes `generate_league_vorp_table.py` fail on the
    config this module writes. Pinned to the real whitelist so a rename there breaks here.
    """
    for reception_pts in (0.0, 0.4, 0.5, 0.75, 1.0, 1.5):
        payload = _payload()
        payload["settings"]["scoringSettings"]["scoringItems"] = [
            {"statId": 53, "points": reception_pts}
        ]
        ruleset, _ = parse_ruleset(payload)
        assert ruleset.name in _RULESET_NAME_VALUES


def test_parse_ruleset_reports_a_non_standard_ppr_value() -> None:
    payload = _payload()
    payload["settings"]["scoringSettings"]["scoringItems"] = [{"statId": 53, "points": 0.6}]
    ruleset, notes = parse_ruleset(payload)
    assert ruleset.reception_pts == pytest.approx(0.6)  # the exact value still scores
    assert ruleset.name == "ESPN_HALF"  # only the tag is approximate
    assert any("nearest one" in note for note in notes), notes


def test_parse_ruleset_accepts_an_explicit_name_override() -> None:
    ruleset, _ = parse_ruleset(_payload(), name="DRAFTKINGS")
    assert ruleset.name == "DRAFTKINGS"


def test_parse_draft_settings_converts_epoch_millis() -> None:
    draft = parse_draft_settings(_payload())
    assert draft["type"] == "AUCTION"
    assert draft["auction_budget"] == 200
    assert draft["date_utc"] is not None
    assert draft["date_utc"].startswith("2026-08-30")


def test_parse_draft_settings_treats_zero_date_as_unscheduled() -> None:
    payload = _payload()
    payload["settings"]["draftSettings"]["date"] = 0
    assert parse_draft_settings(payload)["date_utc"] is None


# ---------------------------------------------------------------------------
# LeagueConfig
# ---------------------------------------------------------------------------


def test_build_league_config_matches_espn_settings() -> None:
    config = build_league_config(_payload())
    assert config.name == "GOAT Steins and Slow Combines"
    assert config.n_teams == 12
    assert config.budget == 200
    assert config.ruleset.reception_pts == pytest.approx(0.5)
    # roster_size excludes IR, and K/DST are dropped: QB1+RB2+WR2+TE1+FLEX2+BENCH5 = 13
    assert config.roster_size == 13


def test_build_league_config_drops_k_and_dst_slots() -> None:
    """The projections core ingests QB/RB/WR/TE only, so a kept D/ST slot makes
    generate_vorp_table raise "cannot fill 16 DST slots: only 0 eligible players remain"
    (hit live on league 856974, 2026-08-24). Dropping them is also the correct replacement
    level: a D/ST pick does not consume a skill player."""
    config = build_league_config(_payload())
    assert RosterSlot.K not in config.roster_slots
    assert RosterSlot.DST not in config.roster_slots
    # ESPN roster is 15 deep (QB1 RB2 WR2 TE1 FLEX2 K1 DST1 BENCH5); 13 are skill picks.
    assert config.roster_size == 13
    # parse_roster_slots still reports ESPN faithfully — only the config filters.
    assert parse_roster_slots(_payload())[RosterSlot.DST] == 1


def test_build_league_config_falls_back_to_team_count() -> None:
    payload = _payload()
    del payload["settings"]["size"]
    assert build_league_config(payload).n_teams == 2  # the two teams in the fixture


def test_build_league_config_ignores_the_budget_espn_reports_for_snake() -> None:
    """ESPN reports a non-zero auctionBudget even for snake leagues (verified live on
    league 856974: SNAKE with auctionBudget 200). Copying it into LeagueConfig would let a
    meaningless — or stale, if the league used to be an auction — number reach the
    auction-value math."""
    payload = _payload()
    payload["settings"]["draftSettings"] = {"type": "SNAKE", "auctionBudget": 500, "date": 0}
    assert build_league_config(payload).budget == 200  # LeagueConfig default, not ESPN's 500


def test_build_league_config_rejects_an_unsized_league() -> None:
    payload = _payload()
    payload["settings"]["size"] = 0
    payload["teams"] = []
    with pytest.raises(EspnLeagueError, match="team count"):
        build_league_config(payload)


# ---------------------------------------------------------------------------
# Teams / rosters / picks
# ---------------------------------------------------------------------------


def test_parse_teams_resolves_names_and_owners() -> None:
    teams = parse_teams(_payload())
    assert list(teams["team_id"]) == [1, 2]
    assert teams.loc[0, "team_name"] == "Two Kids Two Steins"
    assert teams.loc[0, "owner"] == "alden"
    # location + nickname is the older ESPN shape
    assert teams.loc[1, "team_name"] == "Lemon Party"


def test_written_teams_tsv_never_contains_a_swid(tmp_path: Path) -> None:
    """A SWID is a persistent ESPN account identifier for a real person and this repo is
    public, so it must not reach a committed file. Guarded by a test because the column is
    deliberately kept in memory for find_my_team_id."""
    creds = EspnCredentials(swid="{IRRELEVANT}", espn_s2="s2")
    write_league_snapshot(_payload(), tmp_path, creds)
    written = (tmp_path / "teams.tsv").read_text(encoding="utf-8")
    assert "owner_swid" not in written
    assert _MY_SWID not in written
    assert "Two Kids Two Steins" in written  # the useful content survives


def test_parse_teams_requires_the_team_view() -> None:
    with pytest.raises(EspnLeagueError, match="mTeam"):
        parse_teams({"teams": []})


def test_find_my_team_id_matches_on_swid() -> None:
    creds = EspnCredentials(swid=_MY_SWID.strip("{}").lower(), espn_s2="s2")
    assert find_my_team_id(_payload(), creds) == 1


def test_find_my_team_id_returns_none_for_a_non_owner() -> None:
    creds = EspnCredentials(swid="{NOT-IN-LEAGUE}", espn_s2="s2")
    assert find_my_team_id(_payload(), creds) is None


def test_parse_rosters_maps_positions_and_teams() -> None:
    rosters = parse_rosters(_payload())
    assert len(rosters) == 2
    allen = rosters[rosters["player"] == "Josh Allen"].iloc[0]
    assert allen["pos"] == "QB"
    assert allen["nfl_team"] == "BUF"
    assert allen["lineup_slot"] == "QB"


def test_parse_rosters_is_empty_but_typed_before_the_draft() -> None:
    """Pre-draft is the normal state during prep; the columns must still exist so
    downstream code does not KeyError on an empty league."""
    payload = _payload()
    for team in payload["teams"]:
        team["roster"] = {"entries": []}
    rosters = parse_rosters(payload)
    assert rosters.empty
    assert "player" in rosters.columns and "nfl_team" in rosters.columns


def test_parse_draft_picks_emits_the_sim_tsv_shape() -> None:
    payload = _payload()
    picks = parse_draft_picks(payload, parse_teams(payload))
    for column in ("pick", "salary", "player", "nfl_team", "pos", "fantasy_team"):
        assert column in picks.columns
    assert list(picks["pick"]) == [1, 2, 3]
    assert picks.loc[0, "player"] == "Jahmyr Gibbs"
    assert picks.loc[0, "salary"] == 86
    assert picks.loc[0, "fantasy_team"] == "Two Kids Two Steins"


def test_parse_draft_picks_keeps_unnameable_picks() -> None:
    """A player dropped after the draft has no roster entry. Dropping the row would
    silently shrink the pick count and corrupt any draft-history analysis."""
    payload = _payload()
    picks = parse_draft_picks(payload, parse_teams(payload))
    orphan = picks[picks["player_id"] == 999999].iloc[0]
    assert orphan["player"] == ""
    assert orphan["pos"] == "UNKNOWN"
    assert len(picks) == 3


def test_parse_draft_picks_is_empty_before_the_draft() -> None:
    payload = _payload()
    payload["draftDetail"] = {"drafted": False, "picks": []}
    picks = parse_draft_picks(payload, parse_teams(payload))
    assert picks.empty
    assert "salary" in picks.columns


def test_parse_draft_picks_drops_espn_placeholder_slots() -> None:
    """ESPN pre-creates every pick slot with `playerId: -1` as soon as the draft order is
    drawn — months before the draft. Verified live on league 856974 (2026-08-23): 208
    placeholder picks with `drafted: false`. Treating those as results would hand the sim
    a full board of blank picks.
    """
    payload = _payload()
    payload["draftDetail"] = {
        "drafted": False,
        "picks": [
            {
                "overallPickNumber": 1,
                "roundId": 1,
                "roundPickNumber": 1,
                "teamId": 2,
                "playerId": -1,
            },
            {
                "overallPickNumber": 2,
                "roundId": 1,
                "roundPickNumber": 2,
                "teamId": 1,
                "playerId": -1,
            },
        ],
    }
    assert parse_draft_picks(payload, parse_teams(payload)).empty


def test_parse_draft_order_keeps_placeholder_slots() -> None:
    """The mirror of the test above: the placeholders carry the draft order, which is the
    whole point of reading them before the draft."""
    payload = _payload()
    payload["draftDetail"] = {
        "drafted": False,
        "picks": [
            {
                "overallPickNumber": 2,
                "roundId": 1,
                "roundPickNumber": 2,
                "teamId": 1,
                "playerId": -1,
            },
            {
                "overallPickNumber": 1,
                "roundId": 1,
                "roundPickNumber": 1,
                "teamId": 2,
                "playerId": -1,
            },
            {
                "overallPickNumber": 3,
                "roundId": 2,
                "roundPickNumber": 1,
                "teamId": 1,
                "playerId": -1,
            },
        ],
    }
    order = parse_draft_order(payload, parse_teams(payload))
    assert list(order["overall"]) == [1, 2, 3]  # sorted, not payload order
    assert list(order["fantasy_team"]) == [
        "Lemon Party",
        "Two Kids Two Steins",
        "Two Kids Two Steins",
    ]
    mine = order[order["team_id"] == 1]
    assert list(mine["overall"]) == [2, 3]


def test_parse_draft_order_is_empty_when_no_order_is_drawn() -> None:
    payload = _payload()
    payload["draftDetail"] = {"picks": []}
    order = parse_draft_order(payload, parse_teams(payload))
    assert order.empty
    assert "round_pick" in order.columns


# --- schedule + records -------------------------------------------------------------------


def _matchup(
    week: int,
    home_id: int,
    away_id: int,
    *,
    home_pts: float = 0.0,
    away_pts: float = 0.0,
    winner: str = "UNDECIDED",
) -> dict[str, Any]:
    return {
        "matchupPeriodId": week,
        "winner": winner,
        "home": {"teamId": home_id, "totalPoints": home_pts},
        "away": {"teamId": away_id, "totalPoints": away_pts},
    }


def test_parse_schedule_reads_the_real_fixture_list() -> None:
    """The league's actual pairings, which in-season standings must simulate over rather than
    the synthetic `gauntlet_schedule` the preseason simulator uses."""
    payload = _payload(
        schedule=[
            _matchup(1, 1, 2, home_pts=120.5, away_pts=99.0, winner="HOME"),
            _matchup(1, 3, 4, home_pts=88.0, away_pts=101.25, winner="AWAY"),
            _matchup(2, 1, 3),
        ]
    )
    sched = parse_schedule(payload, parse_teams(payload))
    assert list(sched["week"]) == [1, 1, 2]
    assert list(sched["home_team_id"]) == [1, 3, 1]
    assert list(sched["is_played"]) == [True, True, False]
    assert sched.loc[0, "home_points"] == pytest.approx(120.5)
    assert sched.loc[1, "away_points"] == pytest.approx(101.25)


def test_is_played_comes_from_winner_not_from_points() -> None:
    """ESPN reports `totalPoints: 0.0` for an unplayed matchup, which is indistinguishable
    from a real scoreless one. `winner` is the authoritative signal, and reading points
    instead would mark a genuine 0-0 week as never played."""
    payload = _payload(
        schedule=[
            _matchup(1, 1, 2, home_pts=0.0, away_pts=0.0, winner="TIE"),
            _matchup(2, 1, 2, home_pts=0.0, away_pts=0.0),
        ]
    )
    sched = parse_schedule(payload, parse_teams(payload))
    assert list(sched["is_played"]) == [True, False]


def test_parse_schedule_is_empty_before_kickoff_but_still_lists_the_fixtures() -> None:
    """The expected state during draft prep: every matchup exists, none is played."""
    payload = _payload(schedule=[_matchup(w, 1, 2) for w in range(1, 15)])
    sched = parse_schedule(payload, parse_teams(payload))
    assert len(sched) == 14
    assert not sched["is_played"].any()


def test_parse_schedule_skips_a_bye_rather_than_half_recording_it() -> None:
    """An odd team count gives someone a bye, which ESPN reports as a matchup with no `away`.
    A bye is not a game, and a row with a null opponent would break every downstream join."""
    payload = _payload(
        schedule=[
            _matchup(1, 1, 2, winner="HOME"),
            {"matchupPeriodId": 1, "winner": "UNDECIDED", "home": {"teamId": 3}},
        ]
    )
    sched = parse_schedule(payload, parse_teams(payload))
    assert len(sched) == 1
    assert sched.loc[0, "home_team_id"] == 1


def test_team_records_tallies_wins_losses_and_points() -> None:
    payload = _payload(
        schedule=[
            _matchup(1, 1, 2, home_pts=120.0, away_pts=99.0, winner="HOME"),
            _matchup(2, 2, 1, home_pts=130.0, away_pts=110.0, winner="HOME"),
            _matchup(3, 1, 2, home_pts=100.0, away_pts=90.0, winner="HOME"),
        ]
    )
    recs = team_records(parse_schedule(payload, parse_teams(payload))).set_index("team_id")
    assert (recs.loc[1, "wins"], recs.loc[1, "losses"]) == (2, 1)
    assert (recs.loc[2, "wins"], recs.loc[2, "losses"]) == (1, 2)
    assert recs.loc[1, "points_for"] == pytest.approx(120.0 + 110.0 + 100.0)
    assert recs.loc[2, "points_for"] == pytest.approx(99.0 + 130.0 + 90.0)
    assert recs.loc[1, "games_played"] == 3


def test_team_records_carries_ties_as_neither_win_nor_loss() -> None:
    """Simulated weeks cannot tie -- the simulator breaks every matchup with `>=` -- but real
    played weeks can, and a tie must not be folded into a win or a loss for seeding."""
    payload = _payload(schedule=[_matchup(1, 1, 2, home_pts=100.0, away_pts=100.0, winner="TIE")])
    recs = team_records(parse_schedule(payload, parse_teams(payload))).set_index("team_id")
    for team_id in (1, 2):
        assert (recs.loc[team_id, "wins"], recs.loc[team_id, "losses"]) == (0, 0)
        assert recs.loc[team_id, "ties"] == 1
        assert recs.loc[team_id, "games_played"] == 1


def test_team_records_ignores_unplayed_matchups() -> None:
    payload = _payload(
        schedule=[
            _matchup(1, 1, 2, home_pts=120.0, away_pts=99.0, winner="HOME"),
            _matchup(2, 1, 2),
        ]
    )
    recs = team_records(parse_schedule(payload, parse_teams(payload))).set_index("team_id")
    assert recs.loc[1, "games_played"] == 1


def test_team_records_is_empty_before_kickoff() -> None:
    payload = _payload(schedule=[_matchup(1, 1, 2)])
    assert team_records(parse_schedule(payload, parse_teams(payload))).empty


def test_mmatchup_is_pulled_by_default() -> None:
    """Standings need the schedule, so it must not be an opt-in view a caller can forget."""
    assert "mMatchup" in DEFAULT_VIEWS


# --- the locked/simulated week partition ---------------------------------------------------


def test_team_records_can_be_bounded_to_the_weeks_the_simulator_will_not_replay() -> None:
    """`locked` and `simulated` must partition the season's weeks exactly once.

    `simulate_seasons` replays every week from `first_unplayed_week` onward, so anything
    `team_records` banks from those same weeks is counted twice. Two real cases hit this:
    a partially-played week (some matchups final, some not, which is every Sunday evening),
    and playoff weeks, which `parse_schedule` deliberately returns and which must never reach
    a REGULAR-season record. One boundary closes both.
    """
    payload = _payload(
        schedule=[
            _matchup(1, 1, 2, home_pts=120.0, away_pts=99.0, winner="HOME"),
            # Week 2 is partially played: this one final, the next still undecided.
            _matchup(2, 1, 2, home_pts=115.0, away_pts=80.0, winner="HOME"),
            _matchup(2, 3, 4, winner="UNDECIDED"),
            # A playoff week, already decided.
            _matchup(15, 1, 3, home_pts=140.0, away_pts=100.0, winner="HOME"),
        ]
    )
    schedule = parse_schedule(payload, parse_teams(payload))

    # Unbounded: banks week 2's finished game AND the playoff week -- 3 wins for team 1.
    assert team_records(schedule).set_index("team_id").loc[1, "wins"] == 3

    # Bounded to weeks strictly before the first unplayed week (2), only week 1 counts.
    bounded = team_records(schedule, through_week=1).set_index("team_id")
    assert bounded.loc[1, "wins"] == 1
    assert bounded.loc[1, "games_played"] == 1
    assert bounded.loc[1, "points_for"] == pytest.approx(120.0)


def test_team_records_through_week_excludes_playoff_results_from_a_regular_season_record() -> None:
    """A 14-week league whose playoffs are under way: the regular-season record must stay
    14 games. Folding a playoff win in makes a team 'x-y' in a league that plays fewer games
    than that, and contaminates the points-for that breaks seeding ties."""
    payload = _payload(
        schedule=[_matchup(w, 1, 2, home_pts=100.0, away_pts=90.0, winner="HOME") for w in (1, 2)]
        + [_matchup(15, 1, 3, home_pts=150.0, away_pts=80.0, winner="HOME")]
    )
    schedule = parse_schedule(payload, parse_teams(payload))
    recs = team_records(schedule, through_week=14).set_index("team_id")
    assert recs.loc[1, "wins"] == 2
    assert recs.loc[1, "points_for"] == pytest.approx(200.0)


def test_team_records_through_week_none_keeps_the_previous_unbounded_behaviour() -> None:
    payload = _payload(
        schedule=[_matchup(w, 1, 2, home_pts=100.0, away_pts=90.0, winner="HOME") for w in (1, 2)]
    )
    schedule = parse_schedule(payload, parse_teams(payload))
    assert team_records(schedule).equals(team_records(schedule, through_week=None))
