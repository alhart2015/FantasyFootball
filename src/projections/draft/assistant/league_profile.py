"""On-disk description of one real league's live-board setup.

A `LeagueConfig` says what the league's *rules* are. It does not say which VORP pool was
built for it, which seat is mine, or which strategy I decided to draft on — and those are
exactly the settings a user would otherwise retype into the board's sidebar on draft night,
under time pressure, from a plan document. A wrong one is not loud: the board comes up, looks
right, and scores every recommendation against the wrong roster or the wrong seat.

So a profile is a `board_profile.json` sitting beside the league's `league_config.json`:

    {
      "name": "Critts 2026",
      "league_config": "data/leagues/critts_2025_2026/league_config.json",
      "vorp_table": "data/vorp_2026/critts_half16_snake.parquet",
      "id_map": "data/raw/id_map.parquet",
      "my_slot": 8,
      "season": 2026,
      "strategy": "raw_vorp",
      "league_id": 856974,
      "team_id": 17
    }

Paths are resolved against the **current working directory** — the repo root, where
`streamlit run` is launched — matching every other path the board accepts. `id_map` and
`strategy` are optional and default to the board's own defaults.

`league_id` and `team_id` carry the same idea to the *in-season* tools (projected standings,
waiver recommender, trade analyzer), which address ESPN directly rather than a draft seat.
Both are optional: a profile written for the board alone predates them. They are also NOT
`my_slot` — that is a draft seat in `1..n_teams`, while `team_id` is ESPN's franchise id and
is arbitrary (17 in a 16-team league is normal). A CLI that needs one and cannot find it must
name the missing key rather than substitute a guess; pointing an in-season tool at the wrong
franchise produces a full, confident report about somebody else's roster.

`discover_profiles` deliberately reports malformed profiles instead of skipping them. A typo
in a profile that silently vanished would drop the board back to a generic preset, which is
the failure this module exists to prevent — the user would see a working board and no sign
that it is not their league.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from projections.draft.assistant.live import BOARD_STRATEGIES
from projections.draft.league_config import LeagueConfig

PROFILE_FILENAME = "board_profile.json"
DEFAULT_PROFILE_ROOT = Path("data/leagues")
DEFAULT_ID_MAP = Path("data/raw/id_map.parquet")
DEFAULT_STRATEGY = "raw_vorp"


@dataclass(frozen=True)
class LeagueProfile:
    """A ready-to-draft board configuration for one league."""

    key: str
    name: str
    league: LeagueConfig
    #: The directory the profile itself sits in. `league_config_path` may point anywhere,
    #: so this is the only reliable answer to "where do this league's rosters.tsv and
    #: schedule.tsv live".
    profile_dir: Path
    league_config_path: Path
    vorp_path: Path
    id_map_path: Path
    my_slot: int
    season: int
    strategy: str
    #: ESPN league / franchise ids for the in-season tools. `None` when the profile predates
    #: them, which every consumer must handle by asking for the argument explicitly.
    league_id: int | None = None
    team_id: int | None = None

    @property
    def label(self) -> str:
        """Sidebar text: enough to tell two of the user's leagues apart at a glance."""
        return (
            f"{self.name} — {self.league.n_teams} teams, "
            f"{self.league.roster_size} rounds, slot {self.my_slot}"
        )


@dataclass(frozen=True)
class ProfileError:
    """A profile file that exists but could not be loaded. Surfaced, never swallowed."""

    path: Path
    message: str


def _optional_positive_int(raw: dict[str, object], key: str) -> int | None:
    """Read an optional id. Absent is fine; present-and-nonsense is not — a `team_id` of `0` or
    `"seventeen"` would otherwise fail at the use site, far from the file that caused it."""
    if key not in raw or raw[key] is None:
        return None
    try:
        value = int(str(raw[key]).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} {raw[key]!r} is not an integer") from exc
    if value <= 0:
        raise ValueError(f"{key} {value} is not a positive integer")
    return value


def load_profile(path: Path) -> LeagueProfile:
    """Load and validate one `board_profile.json`. Raises `ValueError` on any problem."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON ({exc})") from exc
    if not isinstance(raw, dict):
        raise ValueError("expected a JSON object")

    for required in ("league_config", "vorp_table", "my_slot"):
        if required not in raw:
            raise ValueError(f"missing required key {required!r}")

    league_config_path = Path(str(raw["league_config"]))
    if not league_config_path.is_file():
        raise ValueError(f"league_config {league_config_path} does not exist")
    try:
        league = LeagueConfig.model_validate_json(league_config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"league_config {league_config_path} is invalid ({exc})") from exc

    # A slot outside the league is the one error that would otherwise produce a board that
    # runs happily and puts every one of the user's picks in the wrong place.
    try:
        my_slot = int(raw["my_slot"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"my_slot {raw['my_slot']!r} is not an integer") from exc
    if not 1 <= my_slot <= league.n_teams:
        raise ValueError(
            f"my_slot {my_slot} is outside 1..{league.n_teams} "
            f"for this {league.n_teams}-team league"
        )

    strategy = str(raw.get("strategy", DEFAULT_STRATEGY))
    if strategy not in BOARD_STRATEGIES:
        raise ValueError(f"strategy {strategy!r} is not one of {', '.join(BOARD_STRATEGIES)}")

    season = int(raw.get("season", 2026))
    id_map_path = Path(str(raw.get("id_map", DEFAULT_ID_MAP)))
    league_id = _optional_positive_int(raw, "league_id")
    team_id = _optional_positive_int(raw, "team_id")

    return LeagueProfile(
        key=path.parent.name,
        profile_dir=path.parent,
        name=str(raw.get("name", path.parent.name)),
        league=league,
        league_config_path=league_config_path,
        vorp_path=Path(str(raw["vorp_table"])),
        id_map_path=id_map_path,
        my_slot=my_slot,
        season=season,
        strategy=strategy,
        league_id=league_id,
        team_id=team_id,
    )


def discover_profiles(
    root: Path = DEFAULT_PROFILE_ROOT,
) -> tuple[list[LeagueProfile], list[ProfileError]]:
    """Find every `board_profile.json` under `root`, one directory deep.

    Returns `(profiles, errors)` sorted by directory name. A missing or unreadable `root` is
    not an error — it just means this checkout has no leagues configured, and the board falls
    back to its generic presets.
    """
    profiles: list[LeagueProfile] = []
    errors: list[ProfileError] = []
    if not root.is_dir():
        return profiles, errors
    for profile_path in sorted(root.glob(f"*/{PROFILE_FILENAME}")):
        try:
            profiles.append(load_profile(profile_path))
        except ValueError as exc:
            errors.append(ProfileError(path=profile_path, message=str(exc)))
    return profiles, errors


def resolve_profile(
    league_dir: Path | None = None, *, root: Path = DEFAULT_PROFILE_ROOT
) -> LeagueProfile:
    """The profile a CLI should default to when the user supplied no league arguments.

    `league_dir` names one league directory explicitly. With none given, this requires exactly
    one loadable profile under `root` and raises otherwise — including when a *broken* profile
    sits alongside a good one. Silently defaulting to "the one that happened to parse" is how a
    tool ends up reporting on the wrong league while looking like it worked.
    """
    if league_dir is not None:
        return load_profile(league_dir / PROFILE_FILENAME)

    profiles, errors = discover_profiles(root)
    if errors:
        detail = "; ".join(f"{e.path}: {e.message}" for e in errors)
        raise ValueError(f"{len(errors)} unreadable league profile(s) under {root} — {detail}")
    if not profiles:
        raise ValueError(
            f"no {PROFILE_FILENAME} found under {root}; pass the league arguments explicitly"
        )
    if len(profiles) > 1:
        keys = ", ".join(p.key for p in profiles)
        raise ValueError(
            f"{len(profiles)} league profiles under {root} ({keys}); "
            f"name one with --league-dir {root}/<key>"
        )
    return profiles[0]


# --------------------------------------------------------------------------------------
# CLI surface. The four in-season tools (projected standings, waiver recommender, trade
# analyzer, dashboard) all address the same thing — one league, one season, one franchise,
# one pool — and every one of them used to demand all four on every invocation. They share
# these helpers so the flags, the fallback rule and the announcement line cannot drift
# apart between tools.
# --------------------------------------------------------------------------------------

#: `--league-id --season --team-id --pool --league-dir`, as `add_league_arguments` names them.
LEAGUE_ARGUMENTS = ("league_id", "season", "team_id", "pool", "league_dir")


@dataclass(frozen=True)
class LeagueTarget:
    """Which league, season, franchise and pool one CLI run is about.

    `team_id` stays optional because most of these tools have a defined answer without one
    (standings for the whole league; waivers listing the teams to choose from). `league_dir`
    and `league_config_path` are `None` only on a fully-typed run that named no directory —
    a tool that needs either must say so rather than inventing a path.
    """

    league_id: int
    season: int
    team_id: int | None
    pool: Path
    league_dir: Path | None
    league_config_path: Path | None
    #: Profile display name when the file supplied any of the above; `None` when the user
    #: typed everything. CLIs print it, so a wrong default is visible rather than implicit.
    source: str | None

    def describe(self) -> str:
        """The one-line banner a CLI prints when a profile supplied any of this."""
        who = "" if self.team_id is None else f", team {self.team_id}"
        return f"league: {self.source} (id {self.league_id}{who}, {self.season}) — pool {self.pool}"

    def require_team(self) -> int:
        """The franchise id, for tools that have no answer without one. Callers that
        pass `require_team_id=True` to `resolve_league_target` have already failed with
        a better message naming the profile file; this is the type-level narrowing."""
        if self.team_id is None:
            raise ValueError("this tool needs --team-id <id>")
        return self.team_id

    def require_league_dir(self) -> Path:
        """For the tools that read `rosters.tsv` / `schedule.tsv` out of the league folder."""
        if self.league_dir is None:
            raise ValueError(
                "this tool needs the league directory; pass --league-dir <dir>, or drop the "
                "explicit arguments and let the league profile supply them"
            )
        return self.league_dir

    def require_league_config(self) -> Path:
        if self.league_config_path is None:
            raise ValueError(
                "this tool needs a league_config.json; pass --league-dir <dir>, or drop the "
                "explicit arguments and let the league profile supply them"
            )
        return self.league_config_path


def add_league_arguments(
    parser: argparse.ArgumentParser, *, team_id_help: str = "your team"
) -> None:
    """Declare the five shared flags. All default to `None`, which means "not typed" and is
    what `resolve_league_target` distinguishes from any value a profile may hold."""
    parser.add_argument("--league-id", type=int, default=None)
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--team-id", type=int, default=None, help=team_id_help)
    parser.add_argument("--pool", type=Path, default=None, help="VORP parquet for this league")
    parser.add_argument(
        "--league-dir",
        type=Path,
        default=None,
        help=f"league directory holding {PROFILE_FILENAME}; "
        "only needed when more than one league is configured",
    )


def resolve_league_target(
    args: argparse.Namespace,
    *,
    require_team_id: bool = False,
    root: Path = DEFAULT_PROFILE_ROOT,
) -> LeagueTarget:
    """Fill whatever the user did not type from the league's `board_profile.json`.

    Reads the attribute names `add_league_arguments` defines. Raises `ValueError` when the
    profile cannot be resolved unambiguously, or when it lacks a key with no other source.

    A run that already names `league_id`, `season` and `pool` never opens a profile at all —
    the fully-explicit invocation keeps working in a checkout with no `data/leagues/`, and a
    league folder that has no `board_profile.json` is still usable via `--league-dir`.
    `team_id` is not part of that test: it is optional for most of these tools, so requiring
    it would defeat the fallback for exactly the runs that need it least.
    """
    league_id, season = args.league_id, args.season
    team_id, pool, league_dir = args.team_id, args.pool, args.league_dir
    # Consumed off the Namespace on the way past. Every one of these now defaults to `None`,
    # so a use site that still read `args.season` instead of the resolved target would quietly
    # pass `None` down into a schema and fail far away — measured once as a `season` column of
    # nulls surfacing ninety seconds into a simulation. Removing them turns that class of
    # mistake into an immediate AttributeError naming the exact attribute.
    for flag in LEAGUE_ARGUMENTS:
        if hasattr(args, flag):
            delattr(args, flag)

    if league_id is not None and season is not None and pool is not None:
        return LeagueTarget(
            league_id=league_id,
            season=season,
            team_id=team_id,
            pool=pool,
            league_dir=league_dir,
            league_config_path=None if league_dir is None else league_dir / "league_config.json",
            source=None,
        )

    profile = resolve_profile(league_dir, root=root)
    league_id = league_id if league_id is not None else profile.league_id
    team_id = team_id if team_id is not None else profile.team_id
    if league_id is None or (require_team_id and team_id is None):
        missing = ["league_id"] if league_id is None else []
        if require_team_id and team_id is None:
            missing.append("team_id")
        keys = " and ".join(missing)
        flags = " ".join(f"--{n.replace('_', '-')}" for n in missing)
        raise ValueError(
            f"{profile.profile_dir / PROFILE_FILENAME} has no {keys}; "
            f"add it there, or pass {flags} on the command line"
        )
    return LeagueTarget(
        league_id=int(league_id),
        season=season if season is not None else profile.season,
        team_id=None if team_id is None else int(team_id),
        pool=pool if pool is not None else profile.vorp_path,
        league_dir=profile.profile_dir,
        league_config_path=profile.league_config_path,
        source=profile.name,
    )


__all__ = [
    "DEFAULT_ID_MAP",
    "DEFAULT_PROFILE_ROOT",
    "DEFAULT_STRATEGY",
    "LEAGUE_ARGUMENTS",
    "PROFILE_FILENAME",
    "LeagueProfile",
    "LeagueTarget",
    "ProfileError",
    "add_league_arguments",
    "discover_profiles",
    "load_profile",
    "resolve_league_target",
    "resolve_profile",
]
