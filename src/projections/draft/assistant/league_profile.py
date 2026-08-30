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
      "strategy": "raw_vorp"
    }

Paths are resolved against the **current working directory** — the repo root, where
`streamlit run` is launched — matching every other path the board accepts. `id_map` and
`strategy` are optional and default to the board's own defaults.

`discover_profiles` deliberately reports malformed profiles instead of skipping them. A typo
in a profile that silently vanished would drop the board back to a generic preset, which is
the failure this module exists to prevent — the user would see a working board and no sign
that it is not their league.
"""

from __future__ import annotations

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
    league_config_path: Path
    vorp_path: Path
    id_map_path: Path
    my_slot: int
    season: int
    strategy: str

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

    return LeagueProfile(
        key=path.parent.name,
        name=str(raw.get("name", path.parent.name)),
        league=league,
        league_config_path=league_config_path,
        vorp_path=Path(str(raw["vorp_table"])),
        id_map_path=id_map_path,
        my_slot=my_slot,
        season=season,
        strategy=strategy,
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


__all__ = [
    "DEFAULT_ID_MAP",
    "DEFAULT_PROFILE_ROOT",
    "DEFAULT_STRATEGY",
    "PROFILE_FILENAME",
    "LeagueProfile",
    "ProfileError",
    "discover_profiles",
    "load_profile",
]
