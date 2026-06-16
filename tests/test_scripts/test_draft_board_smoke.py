"""Headless smoke for the Streamlit draft board: it imports and runs without raising."""

from __future__ import annotations

import pytest


def test_draft_board_loads_without_session() -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("scripts/draft_board.py").run()
    assert not at.exception
    # Before any draft is started, the info prompt is shown.
    assert any("Start" in str(getattr(el, "value", "")) for el in at.info)
