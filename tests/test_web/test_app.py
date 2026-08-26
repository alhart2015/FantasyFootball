"""The app factory and its wiring.

Small on purpose. What is worth pinning at this layer is that the app cannot silently reach
production data, that every page is reachable, and that the layering rule the spec sets out is
actually true of the source -- the last of which no HTTP test can tell you.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from flask import Flask
from flask.testing import FlaskClient

from projections.web import DashboardConfig, create_app, dashboard_config

_WEB_ROOT = Path(__file__).resolve().parents[2] / "src" / "projections" / "web"


def test_every_page_is_reachable(client: FlaskClient) -> None:
    for route in ("/", "/standings", "/team", "/healthz"):
        assert client.get(route).status_code == 200, route


def test_health_does_not_depend_on_the_data_root(client: FlaskClient) -> None:
    """The fixture config points at paths that do not exist. `/healthz` answers "is the server
    up", which stays a useful question when the answer to "is the data there" is no."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_the_app_requires_a_config_rather_than_defaulting_to_real_data() -> None:
    """No ambient default means no test can accidentally read or write `data/`. The model repo
    needed an autouse fixture scrubbing production environment variables precisely because its
    app could reach a real store without being told to."""
    with pytest.raises(TypeError):
        create_app()  # type: ignore[call-arg]


def test_the_typed_config_comes_back_off_the_app(app: Flask) -> None:
    config = dashboard_config(app)
    assert isinstance(config, DashboardConfig)
    assert config.season == 2026
    assert config.my_team_id == 17


def test_a_wrongly_typed_config_is_rejected_at_the_accessor() -> None:
    """`app.config` is a plain dict, so nothing stops a caller putting a string under the key.
    The accessor exists so that lands as a clear TypeError rather than an AttributeError deep
    in a route."""
    app = Flask(__name__)
    app.config["DASHBOARD"] = "not a config"
    with pytest.raises(TypeError, match="DashboardConfig"):
        dashboard_config(app)


# --- layering, checked against the source ---------------------------------------------------


def test_no_view_model_imports_flask() -> None:
    """The rule the whole architecture rests on: view models are pure functions returning
    dataclasses, callable without an app or a request. That is what lets the interesting logic
    -- formatting, colour scales, rankings, empty states -- be tested directly.

    Checked against the source because it is exactly the kind of rule that holds until someone
    reaches for `flask.url_for` inside a formatter, and no HTTP test would notice.
    """
    offenders = [
        path.name
        for path in (_WEB_ROOT / "views").glob("*.py")
        if "flask" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"view models must not import flask: {offenders}"


def test_routes_do_not_grow_into_a_god_module() -> None:
    """The model repo put ~30 handlers plus nested helpers inside one 2,307-line function,
    which put those helpers out of reach of every test that is not an HTTP request. This is a
    smoke alarm, not a style rule: a route module past this length is holding logic that
    belongs in a view model.
    """
    too_long = {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in (_WEB_ROOT / "routes").glob("*.py")
        if len(path.read_text(encoding="utf-8").splitlines()) > 150
    }
    assert not too_long, f"route modules should stay thin; move logic to views: {too_long}"


def test_no_page_reaches_the_network_during_tests(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A test must never call the real ESPN API.

    `EspnCredentials.resolve` reads ESPN_SWID/ESPN_S2 from the environment BEFORE the file, so
    pointing the fixture at tmp_path is not enough: on a machine with those exported --
    which is this one -- `/team` reached `fetch_league_payload(856974, 2026)` over the wire.
    Verified by instrumentation before this test existed. It passed either way, because a
    failed call becomes an empty page and a 200, which is exactly why nobody would notice.
    """
    calls: list[tuple[int, int]] = []

    def spy(league_id: int, season: int, **kwargs: object) -> dict[str, object]:
        calls.append((league_id, season))
        raise AssertionError(f"test made a live ESPN call: league {league_id}, season {season}")

    monkeypatch.setattr("projections.ingest.espn_league.fetch_league_payload", spy)
    monkeypatch.setattr("projections.web.routes.team.fetch_league_payload", spy)
    monkeypatch.setattr("projections.web.routes.standings.fetch_league_payload", spy)

    for route in ("/", "/standings", "/team"):
        assert client.get(route).status_code == 200, route
    assert calls == [], f"pages attempted network calls: {calls}"
