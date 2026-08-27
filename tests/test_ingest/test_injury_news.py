"""The injury write-up fetch.

Every test here is offline — the parse takes a payload, and the fetch is exercised through a
monkeypatched opener. A test suite that reaches ESPN is a test suite that fails on a plane.
"""

from __future__ import annotations

import json
import urllib.error
from typing import Any

import pytest

from projections.ingest.injury_news import (
    InjuryNote,
    fetch_injury_note,
    fetch_injury_notes,
    parse_injury_notes,
)
from projections.schemas import InjuryStatus

#: Trimmed from a real response for Puka Nacua, 2026-08-24. Kept verbatim rather than
#: idealised, because the shape is undocumented and this is the evidence for what it is.
NACUA: dict[str, Any] = {
    "athlete": {
        "id": "4426515",
        "displayName": "Puka Nacua",
        "injuries": [
            {
                "longComment": (
                    "Nacua now is ticking toward two weeks off the practice field due to "
                    "soreness in his psoas, but he was able to work with a member of the Rams' "
                    "training staff on a side field Monday, indicating he's making some "
                    "progress."
                ),
                "shortComment": (
                    "Nacua (groin) isn't practicing Monday, Sarah Barshop of ESPN.com reports."
                ),
                "status": "Questionable",
                "date": "2026-08-24T17:50:00.000+00:00",
                "type": {"id": "2", "name": "INJURY_STATUS_QUESTIONABLE", "abbreviation": "Q"},
                "details": {
                    "fantasyStatus": {
                        "description": "QUESTIONABLE",
                        "abbreviation": "QUESTIONABLE",
                    },
                    "type": "Groin",
                    "location": "Groin",
                    "detail": "Soreness",
                    "side": "Not Specified",
                },
            }
        ],
    }
}

HEALTHY: dict[str, Any] = {"athlete": {"id": "4429795", "injuries": []}}


# --- the parse ----------------------------------------------------------------------------------


def test_the_write_up_survives_the_parse() -> None:
    """The whole reason this module exists: the one-word status is a guess multiplier, and the
    sentence underneath it is what lets a reader overrule the guess."""
    note = parse_injury_notes(NACUA, 4426515)
    assert note is not None
    assert note.status is InjuryStatus.QUESTIONABLE
    assert note.injury_type == "Groin"
    assert note.detail == "Soreness"
    assert "isn't practicing Monday" in note.short_comment
    assert "two weeks off the practice field" in note.long_comment


def test_the_summary_line_carries_body_part_severity_and_a_date() -> None:
    """Staleness is the first thing to check about injury news, so the date is in the one-line
    form rather than only in the full record."""
    note = parse_injury_notes(NACUA, 4426515)
    assert note is not None
    assert note.summary() == "Groin · Soreness · 2026-08-24"


def test_a_healthy_player_has_no_note() -> None:
    assert parse_injury_notes(HEALTHY, 4429795) is None


def test_a_payload_with_no_athlete_is_not_an_error() -> None:
    """ESPN returns odd shapes for retired and practice-squad players. An absent note is the
    right answer; an exception would take down a page over a missing footnote."""
    assert parse_injury_notes({}, 1) is None
    assert parse_injury_notes({"athlete": {}}, 1) is None
    assert parse_injury_notes({"athlete": {"injuries": None}}, 1) is None


def test_the_fantasy_status_wins_over_the_prose_one() -> None:
    """`status` is title-case prose ("Questionable"); `details.fantasyStatus.description` is the
    enum ESPN's own fantasy product uses. Preferring the latter keeps this column comparable
    with the one `parse_rosters` produces."""
    payload = json.loads(json.dumps(NACUA))
    payload["athlete"]["injuries"][0]["status"] = "Out"
    note = parse_injury_notes(payload, 4426515)
    assert note is not None
    assert note.status is InjuryStatus.QUESTIONABLE


def test_prose_status_is_used_when_there_is_no_fantasy_status() -> None:
    payload = json.loads(json.dumps(NACUA))
    del payload["athlete"]["injuries"][0]["details"]["fantasyStatus"]
    note = parse_injury_notes(payload, 4426515)
    assert note is not None
    assert note.status is InjuryStatus.QUESTIONABLE


def test_only_the_most_recent_injury_is_kept() -> None:
    """The list is newest-first, and a page showing three months of ankle history is a page
    nobody reads."""
    payload = json.loads(json.dumps(NACUA))
    older = json.loads(json.dumps(payload["athlete"]["injuries"][0]))
    older["details"]["type"] = "Ankle"
    payload["athlete"]["injuries"].append(older)
    note = parse_injury_notes(payload, 4426515)
    assert note is not None
    assert note.injury_type == "Groin"


# --- the fetch ------------------------------------------------------------------------------------


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_fetch_returns_a_parsed_note(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Response(NACUA))
    note = fetch_injury_note(4426515)
    assert isinstance(note, InjuryNote)
    assert note.injury_type == "Groin"


def test_a_network_failure_is_an_absent_note_not_an_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This is decoration on a recommendation that is already computed. A page that fails
    because a colour-commentary lookup timed out has traded something useful for something
    optional."""

    def boom(*args: object, **kwargs: object) -> None:
        raise urllib.error.URLError("no network")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert fetch_injury_note(4426515) is None


def test_a_timeout_is_also_an_absent_note(monkeypatch: pytest.MonkeyPatch) -> None:
    """`TimeoutError` is not a `URLError`, so it needs naming separately -- the same gap that
    took down the standings page once already."""

    def slow(*args: object, **kwargs: object) -> None:
        raise TimeoutError("too slow")

    monkeypatch.setattr("urllib.request.urlopen", slow)
    assert fetch_injury_note(4426515) is None


def test_malformed_json_is_an_absent_note(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Garbage:
        def read(self) -> bytes:
            return b"<html>rate limited</html>"

        def __enter__(self) -> _Garbage:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Garbage())
    assert fetch_injury_note(4426515) is None


def test_bulk_fetch_omits_the_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = {4426515: NACUA, 4429795: HEALTHY}

    def opener(request: Any, **kwargs: object) -> _Response:
        player_id = int(request.full_url.rstrip("/").rsplit("/", 1)[-1])
        return _Response(payloads[player_id])

    monkeypatch.setattr("urllib.request.urlopen", opener)
    notes = fetch_injury_notes([4426515, 4429795])
    assert set(notes) == {4426515}
