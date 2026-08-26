"""The app factory and its wiring.

Small on purpose. What is worth pinning at this layer is that the app cannot silently reach
production data, that every page is reachable, and that the layering rule the spec sets out is
actually true of the source -- the last of which no HTTP test can tell you.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from flask import Flask
from flask.testing import FlaskClient

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.performance_variance import VarianceParams
from projections.midseason.standings import StandingsRun, project_league_standings
from projections.web import DashboardConfig, create_app, dashboard_config
from tests.test_midseason.conftest import espn_payload, id_map, vorp_pool

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
    offenders = [path.name for path in (_WEB_ROOT / "views").glob("*.py") if _imports_flask(path)]
    assert not offenders, f"view models must not import flask: {offenders}"


def _imports_flask(path: Path) -> bool:
    """Parsed rather than grepped.

    A substring search for "flask" over the whole file is wrong in both directions: it fires on
    a docstring saying "no Flask import belongs here" -- which these modules say, because the
    rule is worth writing down -- and it would keep firing after the offending import was
    removed as long as the sentence stayed. Import statements are the thing being ruled on.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "flask" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "flask":
                return True
    return False


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


def test_env_credentials_are_not_reported_as_missing(
    dashboard_config: DashboardConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`EspnCredentials.resolve` reads ESPN_SWID/ESPN_S2 before the file, so requiring the file
    reported "missing the ESPN credentials" as a falsehood on any machine using environment
    credentials -- and refused to render standings while /team worked fine."""
    from projections.web.routes.standings import _missing_inputs

    monkeypatch.setenv("ESPN_SWID", "{ABC}")
    monkeypatch.setenv("ESPN_S2", "s2value")
    message = _missing_inputs(dashboard_config)
    assert message is not None, "the pool and id_map are still absent"
    # Matched on the phrase, not the bare word: pytest's tmp_path is named after the test, so
    # "credentials" appears in every path in the message.
    assert "the ESPN credentials" not in message, message


def test_absent_credentials_are_reported_naming_both_sources(
    dashboard_config: DashboardConfig,
) -> None:
    from projections.web.routes.standings import _missing_inputs

    message = _missing_inputs(dashboard_config)
    assert message is not None
    assert "ESPN_SWID" in message and "the ESPN credentials" in message, message


def test_availability_staleness_reaches_the_page_not_just_stderr(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`load_store_availability` warns when the injury history is thinner than expected --
    an incomplete season on disk, or one that vanished between two that did not.

    `warnings.warn` reaches stderr, which nobody looking at a web page will ever see, so the
    page rendered a projection built on a thinner history than the reader believed and said
    nothing about it. The standings page has a notes slot for exactly this, and the route now
    hands its list down to the loader and reads it back.
    """
    from projections.web.routes import standings as standings_route

    stale = "weekly_stats season(s) [2024] are on disk but incomplete"

    def stub(config: object, notes: list[str]) -> StandingsRun:
        notes.append(stale)
        return _standings_run()

    monkeypatch.setattr(standings_route, "_missing_inputs", lambda config: None)
    monkeypatch.setattr(standings_route, "_run_projection", stub)

    body = client.get("/standings").get_data(as_text=True)
    assert stale in body, "a warning the reader cannot see is a warning that does not exist"


def _standings_run() -> StandingsRun:
    """A real run over the shared synthetic league, so the page under test is a real page."""
    pool = vorp_pool()
    return project_league_standings(
        espn_payload(played_weeks=2),
        pool,
        id_map(),
        PlayerAvailability(p={g: 1.0 for g in pool["gsis_id"].astype(str)}, bye={}),
        VarianceParams.load(),
        season=2026,
        n_sims=20,
        rng=np.random.default_rng(0),
    )
