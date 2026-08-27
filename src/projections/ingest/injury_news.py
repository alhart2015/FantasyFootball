"""The beat-reporter write-up behind an injury designation.

`INJURY_RESERVE` is the one adjustment in `midseason.injuries` that is a guess rather than a
measurement — it is a roster designation, not a game status, so it does not appear in the
weekly injury report at all and four games is the NFL minimum stay. That guess is exactly where
a reader most needs to overrule the number, and the text that lets them do it exists:

    status:       Questionable
    type:         Groin · Soreness · 2026-08-24
    shortComment: "Nacua (groin) isn't practicing Monday, Sarah Barshop of ESPN.com reports."
    longComment:  "...ticking toward two weeks off the practice field... the team likely is
                   being as cautious as possible with Week 1 against the 49ers in mind."

Body part, severity, a beat reporter's read, and a date so staleness is visible.

**Not the fantasy API.** The `kona_player_info` view carries `injuryStatus` and nothing else —
no `injuries` array, no news body, just a `lastNewsDate` timestamp with no text attached. The
write-up lives on ESPN's public site API, which needs no cookies. The athlete id is the same
id the fantasy payloads use, so no crosswalk is involved.

**The text is fetched and shown, never parsed.** A regex for "four to six weeks" is wrong
whenever the sentence is about practice rather than games, or about a different player, and
the error would land inside a projection where nobody can see it. The number states what it
assumed; the text lets a human disagree.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from projections.schemas import InjuryStatus, display_str, parse_injury_status

#: Public site API. No authentication; `{player_id}` is ESPN's athlete id, the same one the
#: fantasy payloads carry.
_ATHLETE_URL = (
    "https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{player_id}"
)

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


@dataclass(frozen=True)
class InjuryNote:
    """One reported injury, as ESPN's beat coverage describes it."""

    player_id: int
    #: ESPN's own words for the status here, which can disagree with the fantasy payload's --
    #: the two feeds update independently. Both are shown rather than reconciled.
    status: InjuryStatus
    #: Body part, e.g. "Groin".
    injury_type: str
    #: Severity or mechanism, e.g. "Soreness".
    detail: str
    #: ISO timestamp. Staleness is the first thing to check about injury news.
    reported: str
    #: One-line wire summary.
    short_comment: str
    #: The full write-up, which is where a timeline usually appears.
    long_comment: str

    def summary(self) -> str:
        """A single line for a table cell, or empty when there is nothing to say."""
        parts = [part for part in (self.injury_type, self.detail) if part]
        head = " · ".join(parts)
        if self.reported:
            head = f"{head} · {self.reported[:10]}" if head else self.reported[:10]
        return head


def parse_athlete_injury(payload: Mapping[str, Any], player_id: int) -> InjuryNote | None:
    """The athlete payload -> its most recent injury note, or None when he is healthy.

    ESPN returns `injuries` as a list; a player with nothing wrong has an empty one. Only the
    first is taken — the list is newest-first and a page showing three months of ankle history
    is a page nobody reads.
    """
    athlete = payload.get("athlete", {}) or {}
    injuries = athlete.get("injuries", []) or []
    if not injuries:
        return None
    latest = injuries[0] or {}
    details = latest.get("details", {}) or {}
    fantasy = (details.get("fantasyStatus", {}) or {}).get("description")
    status, _ = parse_injury_status(fantasy or latest.get("status"))
    return InjuryNote(
        player_id=player_id,
        status=status,
        injury_type=display_str(details.get("type") or details.get("location")),
        detail=display_str(details.get("detail")),
        reported=display_str(latest.get("date")),
        short_comment=display_str(latest.get("shortComment")),
        long_comment=display_str(latest.get("longComment")),
    )


def fetch_injury_note(player_id: int, *, timeout: float = 15.0) -> InjuryNote | None:
    """One player's injury write-up. Returns None when he is healthy or ESPN has nothing.

    **Never raises for a network problem.** This is decoration on a recommendation that is
    already computed — a page that fails because a colour-commentary lookup timed out has
    traded something useful for something optional. A failed fetch is an absent note.
    """
    request = urllib.request.Request(
        _ATHLETE_URL.format(player_id=int(player_id)),
        headers={"User-Agent": _UA, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    return parse_athlete_injury(payload, player_id)


def fetch_injury_notes(
    player_ids: Iterable[int], *, timeout: float = 15.0
) -> dict[int, InjuryNote]:
    """Write-ups for several players, keyed by id. Absent players are simply absent.

    **Call this only for players whose fantasy status is not Active.** It is one request per
    player and the league-wide feed is 403, so a naive sweep of a 400-player free-agent list is
    400 requests for information about the ~15 who are hurt.

    (`midseason.injuries.is_multi_week` does NOT narrow this set -- the CLI uses it to decide
    whether to print the long write-up, which is a different question. An earlier version of
    this paragraph said otherwise.)
    """
    notes: dict[int, InjuryNote] = {}
    for player_id in player_ids:
        note = fetch_injury_note(player_id, timeout=timeout)
        if note is not None:
            notes[int(player_id)] = note
    return notes
