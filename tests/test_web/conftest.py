"""Shared fixtures for the season dashboard tests.

The one rule enforced here: **a test never touches the real `data/`.** `create_app` requires a
`DashboardConfig`, so there is no ambient default to fall back to, and every fixture points at
`tmp_path`. The model repo needed a long fail-closed autouse fixture stripping production
environment variables because its app could reach a real store by default; requiring the config
is cheaper and harder to get wrong.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from flask import Flask
from flask.testing import FlaskClient

from projections.web import DashboardConfig, create_app

#: Matches the mid-season test league so fixtures can be shared conceptually with
#: `tests/test_midseason/conftest.py` without importing across the boundary.
TEST_SEASON = 2026
TEST_LEAGUE_ID = 856974
TEST_TEAM_ID = 17


@pytest.fixture
def dashboard_config(tmp_path: Path) -> DashboardConfig:
    """A config rooted entirely in `tmp_path`. Nothing exists at these paths yet -- a page
    asked to render against an empty data root must say so rather than raise, and several
    tests depend on exactly that."""
    return DashboardConfig(
        data_root=tmp_path / "data",
        league_dir=tmp_path / "leagues" / "test_league",
        pool_path=tmp_path / "pool.parquet",
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
