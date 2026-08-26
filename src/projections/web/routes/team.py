"""My Team page: year-to-date and rest-of-season stats, with league-wide rankings.

I/O only. The assembly lives in `midseason.my_team.build_my_team` and the presentation in
`views.team_view.build_team_page`, both of which take already-fetched data and are tested
directly -- pass 1 of the review found eight defects in this logic while it lived here,
reachable only through an HTTP request that first made a live ESPN call.
"""

from __future__ import annotations

import pandas as pd
from flask import Blueprint, current_app, render_template
from pandera.errors import SchemaError

from projections.draft.league_config import LeagueConfig
from projections.ingest.espn_league import EspnCredentials, EspnLeagueError, fetch_league_payload
from projections.midseason.my_team import build_my_team
from projections.midseason.standings import ProjectionInputError
from projections.schemas import _PYARROW_STR, IdMapSchema, VorpTableSchema
from projections.store import read_partition
from projections.web.app import DashboardConfig, dashboard_config
from projections.web.inputs import missing_inputs, pool_and_id_map
from projections.web.views.team_view import TeamPage, build_team_page, empty_team_page

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
    except (ProjectionInputError, EspnLeagueError, OSError, SchemaError) as exc:
        # OSError covers FileNotFoundError and the socket timeouts that `fetch_league_payload`
        # does not wrap -- a stalled ESPN read raises TimeoutError, which is not a URLError.
        # SchemaError covers the pandera validations below: a pool or id_map that has drifted
        # from its schema is a data problem this page can name, not a traceback.
        page = empty_team_page(str(exc), season=config.season)
    return render_template("team.html", page=page)


def _missing_inputs(config: DashboardConfig) -> str | None:
    """Checked before the ESPN call so a missing local file does not cost a round trip first."""
    required = {
        **pool_and_id_map(config),
        "the league config": (
            config.league_dir / "league_config.json",
            "python -m projections.ingest.espn_league",
        ),
    }
    return missing_inputs(config, required, action="show your team")


def _build(config: DashboardConfig, my_team_id: int) -> TeamPage:
    creds = EspnCredentials.resolve(config.credentials_path)
    payload = fetch_league_payload(config.league_id, config.season, creds=creds)

    pool = pd.read_parquet(config.pool_path)
    pool["gsis_id"] = pool["gsis_id"].astype(_PYARROW_STR)

    run = build_my_team(
        payload,
        VorpTableSchema.validate(pool),
        IdMapSchema.validate(pd.read_parquet(config.data_root / "raw" / "id_map.parquet")),
        _weekly_stats(config),
        LeagueConfig.model_validate_json((config.league_dir / "league_config.json").read_text()),
        my_team_id=my_team_id,
        season=config.season,
    )
    return build_team_page(run, season=config.season)


def _weekly_stats(config: DashboardConfig) -> pd.DataFrame:
    """This season's weekly stats, or an empty frame before the season starts.

    Read ONCE. The previous version read the same partition twice per page view -- once for
    the totals and once to derive the week -- and the week now comes from the schedule anyway.
    """
    try:
        return read_partition(config.data_root / "raw", "weekly_stats", season=config.season)
    except FileNotFoundError:
        return pd.DataFrame()
