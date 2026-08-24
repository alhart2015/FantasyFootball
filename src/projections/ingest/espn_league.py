"""ESPN Fantasy league API client — reads a *private* league's settings, teams,
rosters and draft results.

Distinct from `external_projections.py`, which hits ESPN's public `leaguedefaults`
endpoint for preseason player projections. This module reads one specific league and
therefore needs the manager's browser cookies (`SWID` + `espn_s2`); ESPN answers 401
`AUTH_LEAGUE_NOT_VISIBLE` without them.

`fetch_league_payload` is the only network call; everything else is a pure parser over
the returned JSON, so the mapping logic is unit-tested with synthetic payloads.

The headline output is a `LeagueConfig` derived from ESPN's own settings, so the VORP /
auction / sim tooling runs against the league's real rules instead of a hand-copied
guess.

Usage:
    python -m projections.ingest.espn_league --league-id 856974 --season 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, RosterSlot, Ruleset, Team, normalize_team_code

_log = logging.getLogger(__name__)

_BASE_URL = (
    "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/"
    "{season}/segments/0/leagues/{league_id}"
)
_UA = "Mozilla/5.0"

#: Views worth pulling for draft prep. `mSettings` carries scoring + roster rules,
#: `mTeam` / `mRoster` the franchises and their players, `mDraftDetail` the pick log.
DEFAULT_VIEWS: tuple[str, ...] = ("mSettings", "mTeam", "mRoster", "mDraftDetail")

#: Default location for cookies when the env vars are unset. Gitignored — never commit.
DEFAULT_CREDS_PATH = Path("configs/espn_credentials.json")


class EspnLeagueError(RuntimeError):
    """Raised on an auth failure, HTTP error, or a payload we cannot interpret.
    The CLI converts it to SystemExit; programmatic callers can catch it normally."""


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EspnCredentials:
    """The two browser cookies ESPN uses to authorize a private-league read."""

    swid: str
    espn_s2: str

    def __post_init__(self) -> None:
        if not self.swid.strip() or not self.espn_s2.strip():
            raise EspnLeagueError("Both SWID and espn_s2 are required and must be non-empty.")

    @property
    def normalized_swid(self) -> str:
        """ESPN expects the SWID wrapped in braces; browsers show it either way."""
        raw = self.swid.strip()
        return raw if raw.startswith("{") else "{" + raw.strip("{}") + "}"

    def cookie_header(self) -> str:
        return f"SWID={self.normalized_swid}; espn_s2={self.espn_s2.strip()}"

    @classmethod
    def from_env(cls) -> EspnCredentials | None:
        """Read ESPN_SWID / ESPN_S2. Returns None when either is absent so callers can
        fall back to a creds file without catching an exception on the normal path."""
        swid, espn_s2 = os.environ.get("ESPN_SWID"), os.environ.get("ESPN_S2")
        if not swid or not espn_s2:
            return None
        return cls(swid=swid, espn_s2=espn_s2)

    @classmethod
    def from_file(cls, path: Path) -> EspnCredentials | None:
        """Read a JSON file of {"swid": ..., "espn_s2": ...}. Returns None if absent."""
        if not path.exists():
            return None
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EspnLeagueError(f"{path} is not valid JSON: {exc}") from exc
        try:
            return cls(swid=str(blob["swid"]), espn_s2=str(blob["espn_s2"]))
        except (KeyError, TypeError) as exc:
            raise EspnLeagueError(f"{path} must contain both 'swid' and 'espn_s2' keys.") from exc

    @classmethod
    def resolve(cls, path: Path = DEFAULT_CREDS_PATH) -> EspnCredentials:
        """Environment first (CI-friendly), then the creds file. Raises if neither works."""
        creds = cls.from_env() or cls.from_file(path)
        if creds is None:
            raise EspnLeagueError(
                "No ESPN credentials found. Set ESPN_SWID and ESPN_S2 in the environment, "
                f"or write {path} containing the keys 'swid' and 'espn_s2'. Both cookies come "
                "from fantasy.espn.com while logged in "
                "(DevTools -> Application -> Cookies -> espn.com)."
            )
        return creds


# ---------------------------------------------------------------------------
# ESPN id -> canonical enum maps
# ---------------------------------------------------------------------------

#: ESPN lineup-slot id -> canonical `RosterSlot`. ESPN has three distinct flex slots;
#: they collapse onto FLEX because downstream tooling models a single RB/WR/TE flex.
#: Slot 7 ("OP", any offensive player including QB) is the superflex.
ESPN_LINEUP_SLOTS: dict[int, RosterSlot] = {
    0: RosterSlot.QB,
    2: RosterSlot.RB,
    3: RosterSlot.FLEX,  # RB/WR
    4: RosterSlot.WR,
    5: RosterSlot.FLEX,  # WR/TE
    6: RosterSlot.TE,
    7: RosterSlot.SUPER_FLEX,  # OP — any offensive player
    16: RosterSlot.DST,
    17: RosterSlot.K,
    20: RosterSlot.BENCH,
    21: RosterSlot.IR,
    23: RosterSlot.FLEX,  # RB/WR/TE
}

#: ESPN defaultPositionId -> display string. Plain strings rather than `Position`
#: because `Position` covers skill positions only and rosters include K and D/ST.
ESPN_POSITION_NAMES: dict[int, str] = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}

#: ESPN proTeamId -> NFL team code. Id 0 means "no pro team" (free agent).
ESPN_PRO_TEAMS: dict[int, str] = {
    1: "ATL",
    2: "BUF",
    3: "CHI",
    4: "CIN",
    5: "CLE",
    6: "DAL",
    7: "DEN",
    8: "DET",
    9: "GB",
    10: "TEN",
    11: "IND",
    12: "KC",
    13: "LV",
    14: "LAR",
    15: "MIA",
    16: "MIN",
    17: "NE",
    18: "NO",
    19: "NYG",
    20: "NYJ",
    21: "PHI",
    22: "ARI",
    23: "PIT",
    24: "LAC",
    25: "SF",
    26: "SEA",
    27: "TB",
    28: "WSH",
    29: "CAR",
    30: "JAX",
    33: "BAL",
    34: "HOU",
}

#: ESPN scoring statId -> the `Ruleset` field it sets directly (points used as given).
#: Same id space as `external_projections.ESPN_STAT_IDS`, verified against real payloads.
_DIRECT_SCORING_IDS: dict[int, str] = {
    4: "passing_td_pts",
    20: "interception_pts",
    25: "rushing_td_pts",
    43: "receiving_td_pts",
    53: "reception_pts",
    72: "fumble_lost_pts",
}

#: ESPN scoring statId -> the per-yard `Ruleset` field. ESPN stores points-per-yard
#: (0.04); `Ruleset` stores yards-per-point (25.0), so these invert on parse.
_YARDAGE_SCORING_IDS: dict[int, str] = {
    3: "passing_yds_per_pt",
    24: "rushing_yds_per_pt",
    42: "receiving_yds_per_pt",
}

#: ESPN two-point-conversion statIds (passing / rushing / receiving). `Ruleset` models a
#: single `two_pt_pts`, so a disagreement between them is reported rather than collapsed.
_TWO_PT_SCORING_IDS: dict[int, str] = {19: "passing", 26: "rushing", 44: "receiving"}


def pro_team_code(pro_team_id: int) -> Team | None:
    """ESPN proTeamId -> canonical `Team`. None for free agents (id 0) or unknown ids."""
    raw = ESPN_PRO_TEAMS.get(int(pro_team_id))
    return None if raw is None else normalize_team_code(raw)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def fetch_league_payload(
    league_id: int,
    season: int,
    creds: EspnCredentials,
    views: tuple[str, ...] = DEFAULT_VIEWS,
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """GET one league, merging every requested view into a single JSON payload.

    ESPN honours repeated `view=` params, so one request covers all of them. A 401 means
    the cookies are missing, stale, or belong to an account outside the league; a 404
    means that league+season pair does not exist yet.
    """
    url = _BASE_URL.format(season=season, league_id=league_id)
    query = "&".join(f"view={view}" for view in views)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"User-Agent": _UA, "Cookie": creds.cookie_header(), "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise EspnLeagueError(
                f"ESPN returned 401 for league {league_id} season {season}: the SWID / espn_s2 "
                "cookies are missing, expired, or belong to an account that is not in this "
                "league. Log in at fantasy.espn.com and copy them again."
            ) from exc
        if exc.code == 404:
            raise EspnLeagueError(
                f"ESPN has no league {league_id} for season {season} (404). If the league was "
                "just renewed, the new season may not be published yet."
            ) from exc
        raise EspnLeagueError(f"ESPN HTTP {exc.code} for league {league_id}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise EspnLeagueError(f"Could not reach ESPN: {exc.reason}") from exc

    # A multi-view read returns an object, but some ESPN endpoints wrap it in a
    # one-element list. Normalize so the parsers only ever see one shape.
    if isinstance(payload, list):
        if not payload:
            raise EspnLeagueError(f"ESPN returned an empty payload for league {league_id}.")
        payload = payload[0]
    if not isinstance(payload, dict):
        raise EspnLeagueError(f"Unexpected ESPN payload shape: {type(payload).__name__}.")
    return payload


# ---------------------------------------------------------------------------
# Pure parsers
# ---------------------------------------------------------------------------


def parse_roster_slots(payload: dict[str, Any]) -> dict[RosterSlot, int]:
    """`settings.rosterSettings.lineupSlotCounts` -> canonical slot counts.

    ESPN reports every slot it supports, most with a count of 0; only non-zero slots are
    kept. The three flex ids collapse onto FLEX, so their counts are summed rather than
    overwritten.
    """
    counts = payload.get("settings", {}).get("rosterSettings", {}).get("lineupSlotCounts", {}) or {}
    if not counts:
        raise EspnLeagueError(
            "Payload has no settings.rosterSettings.lineupSlotCounts — was the mSettings "
            "view requested?"
        )
    slots: dict[RosterSlot, int] = {}
    unknown: list[str] = []
    for raw_id, raw_count in counts.items():
        count = int(raw_count)
        if count <= 0:
            continue
        slot = ESPN_LINEUP_SLOTS.get(int(raw_id))
        if slot is None:
            unknown.append(f"{raw_id}x{count}")
            continue
        slots[slot] = slots.get(slot, 0) + count
    if unknown:
        _log.warning(
            "Ignoring %d ESPN lineup slot(s) with no RosterSlot equivalent (IDP or unsupported): "
            "%s. The generated LeagueConfig will understate roster size.",
            len(unknown),
            ", ".join(sorted(unknown)),
        )
    if not slots:
        raise EspnLeagueError("No recognizable roster slots in lineupSlotCounts.")
    return slots


def parse_ruleset(payload: dict[str, Any], name: str) -> tuple[Ruleset, list[str]]:
    """`settings.scoringSettings.scoringItems` -> `Ruleset`, plus a list of human-readable
    notes about anything that did not map cleanly.

    Returning the notes rather than swallowing them matters: `Ruleset` models skill-position
    scoring only, so kicker and D/ST categories genuinely have nowhere to go. Reporting them
    keeps the drop visible instead of silent.
    """
    items = payload.get("settings", {}).get("scoringSettings", {}).get("scoringItems", []) or []
    if not items:
        raise EspnLeagueError(
            "Payload has no settings.scoringSettings.scoringItems — was the mSettings view "
            "requested?"
        )

    fields: dict[str, float] = {}
    two_pt: dict[str, float] = {}
    notes: list[str] = []
    unmodelled: list[str] = []

    for item in items:
        stat_id = int(item.get("statId", -1))
        points = float(item.get("points", 0.0))
        if item.get("pointsOverrides"):
            notes.append(
                f"statId {stat_id} has per-position pointsOverrides "
                f"{item['pointsOverrides']}; Ruleset applies one value to all positions."
            )
        if stat_id in _DIRECT_SCORING_IDS:
            fields[_DIRECT_SCORING_IDS[stat_id]] = points
        elif stat_id in _YARDAGE_SCORING_IDS:
            field = _YARDAGE_SCORING_IDS[stat_id]
            if points <= 0:
                raise EspnLeagueError(
                    f"ESPN scores statId {stat_id} at {points} points per yard. Ruleset stores "
                    f"yards-per-point and requires a positive value, so {field} cannot be "
                    "represented. Inspect the raw settings before trusting any projection."
                )
            fields[field] = 1.0 / points
        elif stat_id in _TWO_PT_SCORING_IDS:
            two_pt[_TWO_PT_SCORING_IDS[stat_id]] = points
        elif points != 0.0:
            unmodelled.append(f"statId {stat_id} = {points:g}")

    distinct_two_pt = set(two_pt.values())
    if len(distinct_two_pt) > 1:
        notes.append(
            f"Two-point conversions score differently by type ({two_pt}); Ruleset has one "
            f"two_pt_pts. Using the rushing value."
        )
        fields["two_pt_pts"] = two_pt.get("rushing", next(iter(distinct_two_pt)))
    elif distinct_two_pt:
        fields["two_pt_pts"] = distinct_two_pt.pop()

    if unmodelled:
        notes.append(
            f"{len(unmodelled)} scoring categories are not modelled by Ruleset (kicking, D/ST "
            f"and bonus categories have no skill-position equivalent): {', '.join(unmodelled)}."
        )

    expected = set(_DIRECT_SCORING_IDS.values()) | set(_YARDAGE_SCORING_IDS.values())
    missing = sorted(expected - fields.keys())
    if missing:
        notes.append(
            f"ESPN did not report these categories; Ruleset defaults apply: {', '.join(missing)}."
        )

    return Ruleset(name=name, **fields), notes


def parse_draft_settings(payload: dict[str, Any]) -> dict[str, Any]:
    """`settings.draftSettings` -> the fields draft prep cares about.

    `date` is epoch milliseconds; it is converted to an ISO-8601 UTC string, or None when
    ESPN has not scheduled the draft yet (it reports 0 in that case).
    """
    draft = payload.get("settings", {}).get("draftSettings", {}) or {}
    raw_date = int(draft.get("date", 0) or 0)
    return {
        "type": str(draft.get("type", "UNKNOWN")),
        "auction_budget": int(draft.get("auctionBudget", 0) or 0),
        "date_utc": (
            datetime.fromtimestamp(raw_date / 1000, tz=UTC).isoformat() if raw_date > 0 else None
        ),
        "keeper_count": int(draft.get("keeperCount", 0) or 0),
        "time_per_selection_sec": int(draft.get("timePerSelection", 0) or 0),
    }


def build_league_config(payload: dict[str, Any], *, name: str | None = None) -> LeagueConfig:
    """Derive a `LeagueConfig` from an ESPN payload.

    Auction budget comes from `draftSettings.auctionBudget`; a snake league reports 0
    there, in which case `LeagueConfig`'s own default budget is used (the field requires
    a positive value and is meaningless for snake drafts anyway).
    """
    settings = payload.get("settings", {}) or {}
    league_name = name or str(settings.get("name", f"espn_league_{payload.get('id', 'unknown')}"))
    n_teams = int(settings.get("size", 0) or 0)
    if n_teams <= 1:
        # `size` is occasionally absent; the team list is the reliable fallback.
        n_teams = len(payload.get("teams", []) or [])
    if n_teams <= 1:
        raise EspnLeagueError(f"Could not determine team count for league {payload.get('id')}.")

    draft = parse_draft_settings(payload)
    ruleset, notes = parse_ruleset(payload, name=f"{league_name}_scoring")
    for note in notes:
        _log.warning("Scoring: %s", note)

    kwargs: dict[str, Any] = {
        "name": league_name,
        "n_teams": n_teams,
        "roster_slots": parse_roster_slots(payload),
        "ruleset": ruleset,
    }
    # ESPN reports auctionBudget: 200 even for snake leagues, where it means nothing.
    # Only trust it when the draft is actually an auction; otherwise let LeagueConfig's
    # own default stand, so a stale budget from a previous format cannot leak into
    # auction-value math.
    if draft["type"] == "AUCTION" and draft["auction_budget"] > 0:
        kwargs["budget"] = draft["auction_budget"]
    return LeagueConfig(**kwargs)


def parse_teams(payload: dict[str, Any]) -> pd.DataFrame:
    """`teams` + `members` -> one row per franchise, with the owning member's display name.

    ESPN splits a team name across `location` and `nickname` in older leagues and puts the
    whole thing in `name` in newer ones; both are handled.
    """
    members_by_id = {
        str(member.get("id", "")): member for member in (payload.get("members", []) or [])
    }
    rows: list[dict[str, Any]] = []
    for team in payload.get("teams", []) or []:
        name = str(team.get("name") or "").strip()
        if not name:
            name = f"{team.get('location', '')} {team.get('nickname', '')}".strip()
        owners = [str(owner) for owner in (team.get("owners") or [])]
        owner_names = [
            str(members_by_id.get(owner, {}).get("displayName", owner)) for owner in owners
        ]
        rows.append(
            {
                "team_id": int(team.get("id", 0)),
                "team_name": name or f"Team {team.get('id')}",
                "abbrev": str(team.get("abbrev", "") or ""),
                "owner": ", ".join(owner_names),
                "owner_swid": owners[0] if owners else "",
            }
        )
    if not rows:
        raise EspnLeagueError("Payload has no teams — was the mTeam view requested?")
    frame = pd.DataFrame(rows).sort_values("team_id").reset_index(drop=True)
    for column in ("team_name", "abbrev", "owner", "owner_swid"):
        frame[column] = frame[column].astype(_PYARROW_STR)
    return frame


def find_my_team_id(payload: dict[str, Any], creds: EspnCredentials) -> int | None:
    """Identify which franchise belongs to the authenticated user, by SWID.

    Returns None rather than raising: a commissioner or a read-only account can legitimately
    see a league without owning a team in it.
    """
    target = creds.normalized_swid.upper()
    for team in payload.get("teams", []) or []:
        owners = {str(owner).strip().upper() for owner in (team.get("owners") or [])}
        if target in owners:
            return int(team.get("id", 0))
    return None


def parse_rosters(payload: dict[str, Any]) -> pd.DataFrame:
    """`teams[].roster.entries` -> one row per rostered player.

    Empty before the draft, which is the expected state during draft prep.
    """
    rows: list[dict[str, Any]] = []
    for team in payload.get("teams", []) or []:
        team_id = int(team.get("id", 0))
        entries = (team.get("roster", {}) or {}).get("entries", []) or []
        for entry in entries:
            player = (entry.get("playerPoolEntry", {}) or {}).get("player", {}) or {}
            pro_team = pro_team_code(int(player.get("proTeamId", 0) or 0))
            rows.append(
                {
                    "team_id": team_id,
                    "player_id": int(player.get("id", entry.get("playerId", 0)) or 0),
                    "player": str(player.get("fullName", "") or ""),
                    "pos": ESPN_POSITION_NAMES.get(
                        int(player.get("defaultPositionId", 0) or 0), "UNKNOWN"
                    ),
                    "nfl_team": str(pro_team) if pro_team is not None else "",
                    "lineup_slot": str(
                        ESPN_LINEUP_SLOTS.get(int(entry.get("lineupSlotId", -1)), "")
                    ),
                    "acquisition_type": str(entry.get("acquisitionType", "") or ""),
                }
            )
    frame = pd.DataFrame(
        rows,
        columns=[
            "team_id",
            "player_id",
            "player",
            "pos",
            "nfl_team",
            "lineup_slot",
            "acquisition_type",
        ],
    )
    for column in ("player", "pos", "nfl_team", "lineup_slot", "acquisition_type"):
        frame[column] = frame[column].astype(_PYARROW_STR)
    return frame


def parse_draft_picks(payload: dict[str, Any], teams: pd.DataFrame) -> pd.DataFrame:
    """`draftDetail.picks` -> the `pick salary player nfl_team pos fantasy_team` TSV shape
    that `scripts/_will_league_2026_outcomes.py --picks` consumes.

    ESPN pre-creates every pick slot as soon as the draft order is set, months before the
    draft: a 16-team/13-round league reports 208 picks with `playerId: -1` while
    `draftDetail.drafted` is still false. Those placeholders are *not* results and are
    dropped here — see `parse_draft_order` for the slot-by-slot order they do carry.

    ESPN's pick log carries ids only, so player names come from the roster view. A player
    traded or dropped since the draft will therefore be missing a name; those rows are kept
    with a blank name rather than dropped, so the pick count stays honest.
    """
    detail = payload.get("draftDetail", {}) or {}
    picks = [
        pick for pick in (detail.get("picks", []) or []) if int(pick.get("playerId", 0) or 0) > 0
    ]
    roster = parse_rosters(payload)
    players_by_id = {
        int(row.player_id): (str(row.player), str(row.pos), str(row.nfl_team))
        for row in roster.itertuples()
    }
    team_names = dict(zip(teams["team_id"], teams["team_name"], strict=True))

    rows: list[dict[str, Any]] = []
    for pick in picks:
        player_id = int(pick.get("playerId", 0) or 0)
        name, pos, nfl_team = players_by_id.get(player_id, ("", "UNKNOWN", ""))
        team_id = int(pick.get("teamId", 0) or 0)
        rows.append(
            {
                "pick": int(pick.get("overallPickNumber", 0) or 0),
                "salary": int(pick.get("bidAmount", 0) or 0),
                "player": name,
                "nfl_team": nfl_team,
                "pos": pos,
                "fantasy_team": str(team_names.get(team_id, f"Team {team_id}")),
                "player_id": player_id,
                "keeper": bool(pick.get("keeper", False)),
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "pick",
            "salary",
            "player",
            "nfl_team",
            "pos",
            "fantasy_team",
            "player_id",
            "keeper",
        ],
    )
    for column in ("player", "nfl_team", "pos", "fantasy_team"):
        frame[column] = frame[column].astype(_PYARROW_STR)
    return frame.sort_values("pick").reset_index(drop=True)


def parse_draft_order(payload: dict[str, Any], teams: pd.DataFrame) -> pd.DataFrame:
    """`draftDetail.picks` -> who picks where, independent of whether anyone has picked.

    ESPN publishes the full slot grid once the order is drawn, so this is the pre-draft
    artifact that actually matters: it answers "which slot am I, and when do I pick again".
    Auction leagues get a nomination order in the same shape.
    """
    picks = (payload.get("draftDetail", {}) or {}).get("picks", []) or []
    team_names = dict(zip(teams["team_id"], teams["team_name"], strict=True))
    rows = [
        {
            "overall": int(pick.get("overallPickNumber", 0) or 0),
            "round": int(pick.get("roundId", 0) or 0),
            "round_pick": int(pick.get("roundPickNumber", 0) or 0),
            "team_id": int(pick.get("teamId", 0) or 0),
            "fantasy_team": str(
                team_names.get(int(pick.get("teamId", 0) or 0), f"Team {pick.get('teamId')}")
            ),
        }
        for pick in picks
    ]
    frame = pd.DataFrame(
        rows, columns=["overall", "round", "round_pick", "team_id", "fantasy_team"]
    )
    frame["fantasy_team"] = frame["fantasy_team"].astype(_PYARROW_STR)
    return frame.sort_values("overall").reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _slugify(name: str, season: int) -> str:
    keep = [char.lower() if char.isalnum() else "_" for char in name]
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return f"{slug or 'espn_league'}_{season}"


def write_league_snapshot(
    payload: dict[str, Any],
    out_dir: Path,
    creds: EspnCredentials,
    *,
    my_team_id: int | None = None,
) -> dict[str, Any]:
    """Write the raw payload plus the derived config / teams / roster / draft files.

    The raw JSON is written first and unconditionally: if a parser is wrong, the evidence
    needed to fix it is already on disk.

    `my_team_id` overrides SWID auto-detection, which fails when the browser's SWID cookie
    belongs to a different ESPN profile than the account that owns the team.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "espn_raw.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    config = build_league_config(payload)
    (out_dir / "league_config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")

    teams = parse_teams(payload)
    # `owner_swid` stays in memory for find_my_team_id but never reaches disk: a SWID is a
    # persistent ESPN account identifier for a real person, and this repo is public. The
    # full raw payload does contain them, which is why espn_raw.json is gitignored.
    teams.drop(columns=["owner_swid"]).to_csv(out_dir / "teams.tsv", sep="\t", index=False)

    rosters = parse_rosters(payload)
    if not rosters.empty:
        rosters.to_csv(out_dir / "rosters.tsv", sep="\t", index=False)

    picks = parse_draft_picks(payload, teams)
    if not picks.empty:
        picks.to_csv(out_dir / "draft.tsv", sep="\t", index=False)

    order = parse_draft_order(payload, teams)
    if not order.empty:
        order.to_csv(out_dir / "draft_order.tsv", sep="\t", index=False)

    if my_team_id is not None and my_team_id not in set(teams["team_id"]):
        raise EspnLeagueError(
            f"--team-id {my_team_id} is not a team in this league. Valid ids: "
            f"{sorted(teams['team_id'])}."
        )
    resolved_team_id = my_team_id if my_team_id is not None else find_my_team_id(payload, creds)
    return {
        "config": config,
        "teams": teams,
        "rosters": rosters,
        "picks": picks,
        "order": order,
        "my_team_id": resolved_team_id,
        "draft_settings": parse_draft_settings(payload),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--league-id", type=int, required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Defaults to data/leagues/<league-slug>/.",
    )
    parser.add_argument(
        "--creds-file",
        type=Path,
        default=DEFAULT_CREDS_PATH,
        help="JSON file with 'swid' and 'espn_s2'. Used only if the env vars are unset.",
    )
    parser.add_argument(
        "--team-id",
        type=int,
        default=None,
        help=(
            "Your ESPN team id, overriding SWID auto-detection. Needed when the browser's "
            "SWID cookie belongs to a different ESPN profile than the team owner."
        ),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        creds = EspnCredentials.resolve(args.creds_file)
        payload = fetch_league_payload(args.league_id, args.season, creds)
        out_dir = args.out or Path("data/leagues") / _slugify(
            str((payload.get("settings", {}) or {}).get("name", f"league_{args.league_id}")),
            args.season,
        )
        result = write_league_snapshot(payload, out_dir, creds, my_team_id=args.team_id)
    except EspnLeagueError as exc:
        raise SystemExit(str(exc)) from exc

    config: LeagueConfig = result["config"]
    teams: pd.DataFrame = result["teams"]
    order: pd.DataFrame = result["order"]
    draft = result["draft_settings"]
    my_team_id = result["my_team_id"]

    print(f"\nLeague: {config.name}  (id {args.league_id}, season {args.season})")
    print(f"Teams: {config.n_teams}   Roster size: {config.roster_size}")
    print(f"Slots: {dict(sorted((str(k), v) for k, v in config.roster_slots.items()))}")
    print(
        f"Scoring: {config.ruleset.reception_pts} PPR, "
        f"{config.ruleset.passing_td_pts} pass TD, "
        f"{config.ruleset.passing_yds_per_pt:g} pass yds/pt"
    )
    budget = f"  budget ${draft['auction_budget']}" if draft["type"] == "AUCTION" else ""
    print(
        f"Draft: {draft['type']}{budget}  date {draft['date_utc'] or 'not scheduled'}  "
        f"keepers {draft['keeper_count']}"
    )
    if my_team_id is None:
        print(
            "\nMy team: NOT FOUND — no team is owned by this SWID. Re-run with --team-id "
            "<id> using one of:"
        )
        for row in teams.itertuples():
            print(f"  {row.team_id:>3}  {row.team_name:<30} {row.owner}")
    else:
        mine = teams.loc[teams["team_id"] == my_team_id]
        print(f"My team: {mine.iloc[0]['team_name']} (team_id {my_team_id})")
        if not order.empty:
            slots = order.loc[order["team_id"] == my_team_id]
            first = slots.iloc[0]
            print(f"Draft slot: {int(first['round_pick'])} of {config.n_teams}")
            print(f"My picks:   {', '.join(str(int(p)) for p in slots['overall'])}")
    print(f"Rostered players: {len(result['rosters'])}   Draft picks made: {len(result['picks'])}")
    print(f"\nWrote {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
