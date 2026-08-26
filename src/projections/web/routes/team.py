"""My Team page: year-to-date and rest-of-season stats, with league-wide rankings."""

from __future__ import annotations

import pandas as pd
from flask import Blueprint, current_app, render_template

from projections.draft.league_config import LeagueConfig
from projections.ingest.espn_league import (
    EspnCredentials,
    EspnLeagueError,
    fetch_league_payload,
    parse_rosters,
    parse_teams,
)
from projections.midseason.standings import SlotMap, rosters_to_slots
from projections.schemas import _PYARROW_STR, VorpTableSchema
from projections.scoring.actuals import actual_season_total
from projections.store import read_partition
from projections.web.app import DashboardConfig, dashboard_config
from projections.web.views.team_view import TeamPage, build_team_page, empty_team_page

bp = Blueprint("team", __name__)


@bp.route("/team")
def team() -> str:
    """Read, format, render. The assembly lives in `views.team_view`."""
    config = dashboard_config(current_app)
    if config.my_team_id is None:
        return render_template(
            "team.html",
            page=empty_team_page(
                "No team selected. Start the dashboard with --team-id to see your roster.",
                season=config.season,
            ),
        )
    try:
        page = _build(config, config.my_team_id)
    except (EspnLeagueError, FileNotFoundError) as exc:
        page = empty_team_page(str(exc), season=config.season)
    return render_template("team.html", page=page)


def _build(config: DashboardConfig, my_team_id: int) -> TeamPage:
    creds = EspnCredentials.resolve(config.credentials_path)
    payload = fetch_league_payload(config.league_id, config.season, creds=creds)

    teams = parse_teams(payload)
    rosters = parse_rosters(payload)
    if rosters.empty:
        return empty_team_page("No rosters yet — the draft has not happened.", season=config.season)

    pool = pd.read_parquet(config.pool_path)
    pool["gsis_id"] = pool["gsis_id"].astype(_PYARROW_STR)
    pool = VorpTableSchema.validate(pool)

    # Same ESPN-id -> gsis crosswalk the standings pipeline uses, so a player resolves
    # identically on both pages.
    slots = SlotMap.from_team_ids(list(teams["team_id"]))
    id_map = pd.read_parquet(config.data_root / "raw" / "id_map.parquet")
    by_slot, _ = rosters_to_slots(rosters, id_map, slots, set(pool["gsis_id"].astype(str)))
    mine = set(by_slot[slots.slot(my_team_id)])

    roster = rosters.assign(gsis_id=_resolve_gsis(rosters, id_map).astype(_PYARROW_STR))
    roster = roster[roster["gsis_id"].astype(str).isin(mine)]

    league_config = LeagueConfig.model_validate_json(
        (config.league_dir / "league_config.json").read_text()
    )
    team_name = _team_name(teams, my_team_id)
    return build_team_page(
        roster,
        _ytd(config, league_config),
        pool,
        team_name=team_name,
        season=config.season,
        week=_played_weeks(config) + 1,
    )


def _resolve_gsis(rosters: pd.DataFrame, id_map: pd.DataFrame) -> pd.Series:
    """ESPN `player_id` -> gsis, deduplicated on `espn_id` for the same reason
    `rosters_to_slots` does it: the live id_map holds ids that map to two players."""
    cross = (
        id_map[["espn_id", "gsis_id"]]
        .dropna()
        .astype({"espn_id": str})
        .drop_duplicates("espn_id")
        .set_index("espn_id")["gsis_id"]
    )
    return rosters["player_id"].astype(str).map(cross)


def _ytd(config: DashboardConfig, league_config: LeagueConfig) -> pd.DataFrame:
    """Season-to-date fantasy points, scored under THIS league's ruleset.

    Our own scoring rather than ESPN's applied totals: one number everywhere, and league-wide
    rankings are impossible from ESPN's per-matchup data regardless, since it only covers
    rostered players. An absent partition is the normal preseason state, not an error.
    """
    try:
        weekly = read_partition(config.data_root / "raw", "weekly_stats", season=config.season)
    except FileNotFoundError:
        return pd.DataFrame(
            {
                "gsis_id": pd.Series(dtype=_PYARROW_STR),
                "position": pd.Series(dtype=_PYARROW_STR),
                "actual_total": pd.Series(dtype="float64"),
            }
        )
    return actual_season_total(weekly, league_config.ruleset)


def _played_weeks(config: DashboardConfig) -> int:
    try:
        weekly = read_partition(config.data_root / "raw", "weekly_stats", season=config.season)
    except FileNotFoundError:
        return 0
    return int(weekly["week"].max()) if not weekly.empty else 0


def _team_name(teams: pd.DataFrame, my_team_id: int) -> str:
    match = teams[teams["team_id"] == my_team_id]
    return str(match.iloc[0]["team_name"]) if not match.empty else f"Team {my_team_id}"
