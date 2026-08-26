"""Shared fixtures for the season dashboard tests.

Two rules enforced here:

1. **A test never touches the real `data/`.** `create_app` requires a `DashboardConfig`, so
   there is no ambient default, and every fixture points at `tmp_path`.
2. **A test never touches the real ESPN.** That one needs an autouse fixture after all --
   `EspnCredentials.resolve` reads the environment before the file, so a config pointing at
   `tmp_path` does not stop a developer machine with `ESPN_SWID`/`ESPN_S2` exported from making
   a live call. An earlier version of this docstring claimed requiring the config made
   env-scrubbing unnecessary; that was true of the data root and false of credentials, and a
   test was calling the user's real league because of it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from flask import Flask
from flask.testing import FlaskClient

from projections.web import DashboardConfig, create_app
from tests.test_midseason.conftest import MY_TEAM_ID

#: The league the dashboard tests render. The shape itself comes from
#: `tests/test_midseason/conftest.py`, which several modules here import directly -- the two
#: suites describe the same league on purpose, and a second copy of `17` was how they would
#: quietly stop doing so. Only the two ids the dashboard config needs and the domain fixtures
#: do not have are declared here.
TEST_SEASON = 2026
TEST_LEAGUE_ID = 856974
TEST_TEAM_ID = MY_TEAM_ID


@pytest.fixture(autouse=True)
def _no_ambient_espn_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip ESPN credentials from the environment for every web test.

    `EspnCredentials.resolve` reads ESPN_SWID/ESPN_S2 BEFORE falling back to the file, so
    pointing `credentials_path` at tmp_path does not stop a machine with those exported from
    reaching the real API. This conftest previously claimed that requiring the config made an
    env-scrubbing fixture unnecessary; for the data root that is true, for credentials it is
    not, and a test was making a live call to the user's league as a result.
    """
    monkeypatch.delenv("ESPN_SWID", raising=False)
    monkeypatch.delenv("ESPN_S2", raising=False)


@pytest.fixture
def dashboard_config(tmp_path: Path) -> DashboardConfig:
    """A config rooted entirely in `tmp_path`. Nothing exists at these paths yet -- a page
    asked to render against an empty data root must say so rather than raise, and several
    tests depend on exactly that."""
    return DashboardConfig(
        data_root=tmp_path / "data",
        league_dir=tmp_path / "leagues" / "test_league",
        pool_path=tmp_path / "pool.parquet",
        # Never the real creds file, which exists on this machine.
        credentials_path=tmp_path / "espn_credentials.json",
        season=TEST_SEASON,
        league_id=TEST_LEAGUE_ID,
        my_team_id=TEST_TEAM_ID,
        n_sims=20,
    )


@pytest.fixture
def app(dashboard_config: DashboardConfig) -> Flask:
    app = create_app(dashboard_config, TESTING=True)
    return app


@pytest.fixture
def client(app: Flask) -> Iterator[FlaskClient]:
    with app.test_client() as client:
        yield client
