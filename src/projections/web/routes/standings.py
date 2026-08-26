"""Standings page: current record and projected finish for every team."""

from __future__ import annotations

import numpy as np
import pandas as pd
from flask import Blueprint, current_app, render_template

from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.assistant.rookies import attach_is_rookie
from projections.ingest.espn_league import EspnCredentials, EspnLeagueError, fetch_league_payload
from projections.midseason.standings import (
    ProjectionInputError,
    StandingsRun,
    project_league_standings,
)
from projections.schemas import _PYARROW_STR, VorpTableSchema
from projections.web.app import DashboardConfig, dashboard_config
from projections.web.views.standings_view import build_standings_page, empty_standings_page

bp = Blueprint("standings", __name__)


@bp.route("/")
@bp.route("/standings")
def standings() -> str:
    """Read, format, render. Every rule about what the table looks like lives in
    `views.standings_view`, which is testable without an app."""
    config = dashboard_config(current_app)
    missing = _missing_inputs(config)
    if missing:
        return render_template(
            "standings.html", page=empty_standings_page(missing, season=config.season)
        )
    try:
        run = _run_projection(config)
    except (ProjectionInputError, EspnLeagueError) as exc:
        # Both mean "there is nothing to project yet", which before the season is the normal
        # state rather than a fault. The reason goes on the page; an empty table would read as
        # "everyone is 0-0".
        page = empty_standings_page(str(exc), season=config.season)
    else:
        page = build_standings_page(run, season=config.season, my_team_id=config.my_team_id)
    return render_template("standings.html", page=page)


def _missing_inputs(config: DashboardConfig) -> str | None:
    """Name what is absent, or None when everything is present.

    Checked up front rather than caught as `FileNotFoundError`, because that exception carries
    a bare path and no indication of what the file is for or how to produce it. "No such file
    or directory: 'nonexistent'" is not something a reader can act on.
    """
    required = {
        "the VORP pool": (config.pool_path, "scripts/generate_league_vorp_table.py"),
        "the id_map": (
            config.data_root / "raw" / "id_map.parquet",
            "projections.ingest.id_map.build_id_map",
        ),
        "the ESPN credentials": (
            config.credentials_path,
            'a JSON file of {"swid": ..., "espn_s2": ...}',
        ),
    }
    absent = [
        f"{label} at {path} (build it with {how})"
        for label, (path, how) in required.items()
        if not path.exists()
    ]
    if not absent:
        return None
    return "Cannot project standings — missing " + "; ".join(absent) + "."


def _run_projection(config: DashboardConfig) -> StandingsRun:
    """Pull the league and project it. Kept beside the route rather than in the view model
    because it does I/O -- the view model stays a pure function over the result."""
    creds = EspnCredentials.resolve(config.credentials_path)
    payload = fetch_league_payload(config.league_id, config.season, creds=creds)

    pool = pd.read_parquet(config.pool_path)
    pool["gsis_id"] = pool["gsis_id"].astype(_PYARROW_STR)
    pool = attach_is_rookie(
        VorpTableSchema.validate(pool), season=config.season, data_root=config.data_root
    )
    return project_league_standings(
        payload,
        pool,
        pd.read_parquet(config.data_root / "raw" / "id_map.parquet"),
        load_store_availability(pool, season=config.season, data_root=config.data_root),
        VarianceParams.load(),
        season=config.season,
        n_sims=config.n_sims,
        rng=np.random.default_rng(0),
    )
