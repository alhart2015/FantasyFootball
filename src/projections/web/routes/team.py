"""My Team page: year-to-date and rest-of-season stats, with rankings."""

from __future__ import annotations

from flask import Blueprint, current_app, render_template

from projections.web.app import dashboard_config

bp = Blueprint("team", __name__)


@bp.route("/team")
def team() -> str:
    """My roster. Renders nothing computed here -- see `views.team_view`."""
    config = dashboard_config(current_app)
    return render_template("team.html", season=config.season, my_team_id=config.my_team_id)
