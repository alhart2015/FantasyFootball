"""Standings page: current record and projected finish for every team."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from flask import Blueprint, current_app, render_template

from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.assistant.performance_variance import (
    _DEFAULT_PARAMS_PATH as DEFAULT_VARIANCE_PARAMS_PATH,
)
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
    notes: list[str] = []
    try:
        run = _run_projection(config, notes)
    except (ProjectionInputError, EspnLeagueError, OSError) as exc:
        # ProjectionInputError and EspnLeagueError mean "there is nothing to project yet",
        # which before the season is the normal state rather than a fault. OSError covers the
        # two the pre-check cannot: load_store_availability raises FileNotFoundError when no
        # complete prior weekly_stats season is on disk, and a stalled ESPN read raises
        # TimeoutError, which is not a URLError and so is not wrapped. Either would otherwise
        # be a traceback on a page built to have an empty state.
        page = empty_standings_page(str(exc), season=config.season)
    else:
        page = build_standings_page(run, season=config.season, my_team_id=config.my_team_id)
        # The availability model's staleness warnings go to stderr, where nobody looking at a
        # web page will ever see them -- and a projection built on a thinner injury history
        # than the reader assumes is precisely what the notes slot is for.
        if notes:
            page = replace(page, notes=tuple(notes) + page.notes)
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
        # `VarianceParams.load()` defaults to a RELATIVE path, resolved against the process
        # CWD rather than data_root, so starting the dashboard from anywhere but the repo root
        # used to 500 here. Named rather than left to raise.
        "the variance params": (
            DEFAULT_VARIANCE_PARAMS_PATH,
            "they ship with the repo — run from the repo root",
        ),
    }
    absent = [
        f"{label} at {path} (build it with {how})"
        for label, (path, how) in required.items()
        if not path.exists()
    ]
    # Credentials are checked separately: `EspnCredentials.resolve` reads ESPN_SWID/ESPN_S2
    # BEFORE the file, so requiring the file reported "missing the ESPN credentials" as a
    # falsehood on any machine using environment credentials.
    if EspnCredentials.from_env() is None and not config.credentials_path.exists():
        absent.append(
            f"the ESPN credentials — neither ESPN_SWID/ESPN_S2 in the environment nor a file "
            f"at {config.credentials_path}"
        )
    if not absent:
        return None
    return "Cannot project standings — missing " + "; ".join(absent) + "."


def _run_projection(config: DashboardConfig, notes: list[str]) -> StandingsRun:
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
        load_store_availability(
            pool, season=config.season, data_root=config.data_root, notes=notes
        ),
        VarianceParams.load(),
        season=config.season,
        n_sims=config.n_sims,
        rng=np.random.default_rng(0),
    )
