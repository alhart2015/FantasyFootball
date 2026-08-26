"""Standings page: current record and projected finish for every team."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from flask import Blueprint, current_app, render_template
from pandera.errors import SchemaError

from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.assistant.rookies import attach_is_rookie
from projections.ingest.espn_league import EspnCredentials, EspnLeagueError, fetch_league_payload
from projections.midseason.standings import (
    ProjectionInputError,
    StandingsRun,
    project_league_standings,
)
from projections.web.app import DashboardConfig, dashboard_config
from projections.web.inputs import (
    VARIANCE_PARAMS,
    load_id_map,
    load_pool,
    missing_inputs,
    pool_and_id_map,
)
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
    notes: list[str] = []
    try:
        run = _run_projection(config, notes)
    except (ProjectionInputError, EspnLeagueError, OSError, SchemaError) as exc:
        # ProjectionInputError and EspnLeagueError mean "there is nothing to project yet",
        # which before the season is the normal state rather than a fault. OSError covers the
        # two the pre-check cannot: load_store_availability raises FileNotFoundError when no
        # complete prior weekly_stats season is on disk, and a stalled ESPN read raises
        # TimeoutError, which is not a URLError and so is not wrapped. Either would otherwise
        # be a traceback on a page built to have an empty state.
        # Notes collected before the failure are still true and still worth showing: a thin
        # injury history is exactly the sort of thing that explains the failure below it.
        page = empty_standings_page(str(exc), season=config.season)
    else:
        page = build_standings_page(run, season=config.season, my_team_id=config.my_team_id)
    # The availability model's staleness warnings go to stderr, where nobody looking at a web
    # page will ever see them -- and a projection built on a thinner injury history than the
    # reader assumes is precisely what the notes slot is for. Applied on BOTH paths, so a run
    # that fails AFTER the loader still shows what the loader found; the template renders notes
    # on its empty branch too, which is what makes that worth doing.
    if notes:
        page = replace(page, notes=tuple(notes) + page.notes)
    return render_template("standings.html", page=page)


def _missing_inputs(config: DashboardConfig) -> str | None:
    """Checked before the ESPN call so a missing local file does not cost a round trip first."""
    required = {**pool_and_id_map(config), **VARIANCE_PARAMS}
    return missing_inputs(config, required, action="project standings")


def _run_projection(config: DashboardConfig, notes: list[str]) -> StandingsRun:
    """Pull the league and project it. Kept beside the route rather than in the view model
    because it does I/O -- the view model stays a pure function over the result."""
    creds = EspnCredentials.resolve(config.credentials_path)
    payload = fetch_league_payload(config.league_id, config.season, creds=creds)

    pool = attach_is_rookie(load_pool(config), season=config.season, data_root=config.data_root)
    return project_league_standings(
        payload,
        pool,
        load_id_map(config),
        load_store_availability(
            pool, season=config.season, data_root=config.data_root, notes=notes
        ),
        VarianceParams.load(),
        season=config.season,
        n_sims=config.n_sims,
        rng=np.random.default_rng(0),
    )
