"""Flask app factory for the season dashboard.

Two read-only pages over data the repo already computes: projected standings, and my team's
year-to-date and rest-of-season stats. See
`docs/superpowers/specs/2026-08-26-season-web-ui-design.md`.

**This file does one thing.** Config, blueprint registration, and nothing else. The model for
this UI (`FantasyBaseball`) grew a single 2,307-line `register_routes(app)` holding ~30 handlers
plus nested helpers, and those helpers are unreachable from any test that is not an HTTP
request. Blueprints, one per page group, from the first commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flask import Flask


@dataclass(frozen=True)
class DashboardConfig:
    """Everything the app needs to find its data.

    Passed in rather than read from the environment so a test can point the whole app at a
    fixture tree and never touch the real `data/`.
    """

    #: Store root — the parent of `raw/` and `processed/`.
    data_root: Path
    #: The league snapshot directory holding `league_config.json`.
    league_dir: Path
    #: VORP pool parquet for this league.
    pool_path: Path
    season: int
    #: ESPN league id, for the live pull.
    league_id: int
    #: Which team is mine. None means the "you" highlight is simply absent.
    my_team_id: int | None = None
    #: Monte-Carlo draws per standings run. Lower in tests.
    n_sims: int = 2000


def create_app(config: DashboardConfig, **flask_kwargs: Any) -> Flask:
    """Build the dashboard app.

    `config` is required rather than defaulted: an app that silently falls back to the real
    `data/` is one import away from a test writing to it.
    """
    app = Flask(__name__)
    app.config["DASHBOARD"] = config
    app.config.update(flask_kwargs)

    from projections.web.routes.standings import bp as standings_bp
    from projections.web.routes.team import bp as team_bp

    app.register_blueprint(standings_bp)
    app.register_blueprint(team_bp)

    @app.route("/healthz")
    def healthz() -> dict[str, str]:
        """Liveness only. Deliberately does not touch the data root, so it stays useful for
        answering "is the server up" when the answer to "is the data there" is no."""
        return {"status": "ok"}

    return app


def dashboard_config(app: Flask) -> DashboardConfig:
    """The typed config off a live app. Routes use this rather than reaching into
    `app.config` with a bare string key."""
    config = app.config["DASHBOARD"]
    if not isinstance(config, DashboardConfig):  # pragma: no cover - misconfiguration
        raise TypeError(f"app.config['DASHBOARD'] is {type(config).__name__}, not DashboardConfig")
    return config
