"""My Team page: year-to-date and rest-of-season stats, with league-wide rankings.

I/O only. Every rule about what the page shows lives in `views.team_view.assemble_team_page`,
which takes already-fetched data and is tested directly -- pass 1 of the review found eight
defects in this logic while it lived here, reachable only through an HTTP request that first
made a live ESPN call.
"""

from __future__ import annotations

import pandas as pd
from flask import Blueprint, current_app, render_template

from projections.draft.league_config import LeagueConfig
from projections.ingest.espn_league import EspnCredentials, EspnLeagueError, fetch_league_payload
from projections.midseason.standings import ProjectionInputError
from projections.schemas import _PYARROW_STR, IdMapSchema, VorpTableSchema
from projections.store import read_partition
from projections.web.app import DashboardConfig, dashboard_config
from projections.web.views.team_view import TeamPage, assemble_team_page, empty_team_page

bp = Blueprint("team", __name__)


@bp.route("/team")
def team() -> str:
    config = dashboard_config(current_app)
    if config.my_team_id is None:
        return render_template(
            "team.html",
            page=empty_team_page(
                "No team selected. Start the dashboard with --team-id to see your roster.",
                season=config.season,
            ),
        )
    missing = _missing_inputs(config)
    if missing:
        return render_template("team.html", page=empty_team_page(missing, season=config.season))
    try:
        page = _build(config, config.my_team_id)
    except (ProjectionInputError, EspnLeagueError, OSError) as exc:
        # OSError covers FileNotFoundError and the socket timeouts that `fetch_league_payload`
        # does not wrap -- a stalled ESPN read raises TimeoutError, which is not a URLError.
        page = empty_team_page(str(exc), season=config.season)
    return render_template("team.html", page=page)


def _missing_inputs(config: DashboardConfig) -> str | None:
    """Name what is absent, or None. Checked before the ESPN call so a missing local file does
    not cost a network round trip first."""
    required = {
        "the VORP pool": (config.pool_path, "scripts/generate_league_vorp_table.py"),
        "the id_map": (
            config.data_root / "raw" / "id_map.parquet",
            "projections.ingest.id_map.build_id_map",
        ),
        "the league config": (
            config.league_dir / "league_config.json",
            "python -m projections.ingest.espn_league",
        ),
    }
    absent = [
        f"{label} at {path} (build it with {how})"
        for label, (path, how) in required.items()
        if not path.exists()
    ]
    if not absent:
        return None
    return "Cannot show your team — missing " + "; ".join(absent) + "."


def _build(config: DashboardConfig, my_team_id: int) -> TeamPage:
    creds = EspnCredentials.resolve(config.credentials_path)
    payload = fetch_league_payload(config.league_id, config.season, creds=creds)

    pool = pd.read_parquet(config.pool_path)
    pool["gsis_id"] = pool["gsis_id"].astype(_PYARROW_STR)

    return assemble_team_page(
        payload,
        VorpTableSchema.validate(pool),
        IdMapSchema.validate(pd.read_parquet(config.data_root / "raw" / "id_map.parquet")),
        _weekly_stats(config),
        LeagueConfig.model_validate_json((config.league_dir / "league_config.json").read_text()),
        my_team_id=my_team_id,
        season=config.season,
    )


def _weekly_stats(config: DashboardConfig) -> pd.DataFrame:
    """This season's weekly stats, or an empty frame before the season starts.

    Read ONCE. The previous version read the same partition twice per page view -- once for
    the totals and once to derive the week -- and the week now comes from the schedule anyway.
    """
    try:
        return read_partition(config.data_root / "raw", "weekly_stats", season=config.season)
    except FileNotFoundError:
        return pd.DataFrame()
