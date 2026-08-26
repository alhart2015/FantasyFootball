"""Standings page: current record and projected finish for every team."""

from __future__ import annotations

from flask import Blueprint, current_app, render_template

from projections.web.app import dashboard_config

bp = Blueprint("standings", __name__)


@bp.route("/")
@bp.route("/standings")
def standings() -> str:
    """The league table. Renders nothing computed here -- see `views.standings_view`."""
    config = dashboard_config(current_app)
    return render_template("standings.html", season=config.season)
