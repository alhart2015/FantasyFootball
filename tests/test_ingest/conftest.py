"""Ingest-test conftest.

The shared `fake_*_df` fixtures (raw `nfl_data_py` response mocks) are
defined in `tests/conftest.py` so the top-level smoke test can request
them too. Pytest hierarchical fixture resolution makes them available
to every test under this directory unchanged.
"""

from __future__ import annotations

from typing import Any

import pytest

from projections.ingest import external_projections


@pytest.fixture(autouse=True)
def _sleeper_offseason(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sleeper's calendar says "offseason" unless a test says otherwise.

    `refresh_external_projections` asks Sleeper whether the season is in progress before it
    decides whether to rebuild the season line from weekly data. Tests that stub the season
    fetch would otherwise make that one live request. A test of the in-season path patches
    `fetch_sleeper_state` itself, which overrides this.
    """
    state: dict[str, Any] = {"season": "1900", "season_type": "off", "week": 0}
    monkeypatch.setattr(external_projections, "fetch_sleeper_state", lambda: state)
