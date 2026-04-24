"""Single source of truth for canonical types: enums, NewTypes, pydantic models, pandera schemas."""

from __future__ import annotations

from enum import StrEnum


class Position(StrEnum):
    """NFL fantasy-relevant positions. Use Position.QB, never the string "QB"."""

    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    K = "K"
    DST = "DST"
    # Reserved for future IDP support; kept here so RosterSlot can refer to them
    # without a circular import. Not currently produced by ingest.


class Team(StrEnum):
    """Canonical NFL team codes. 32 teams.

    `nfl_data_py` historically uses some non-canonical aliases (JAX vs JAC,
    LA vs LAR). Use `normalize_team_code()` to coerce input before storing.
    """

    ARI = "ARI"
    ATL = "ATL"
    BAL = "BAL"
    BUF = "BUF"
    CAR = "CAR"
    CHI = "CHI"
    CIN = "CIN"
    CLE = "CLE"
    DAL = "DAL"
    DEN = "DEN"
    DET = "DET"
    GB = "GB"
    HOU = "HOU"
    IND = "IND"
    JAC = "JAC"
    KC = "KC"
    LAC = "LAC"
    LAR = "LAR"
    LV = "LV"
    MIA = "MIA"
    MIN = "MIN"
    NE = "NE"
    NO = "NO"
    NYG = "NYG"
    NYJ = "NYJ"
    PHI = "PHI"
    PIT = "PIT"
    SEA = "SEA"
    SF = "SF"
    TB = "TB"
    TEN = "TEN"
    WAS = "WAS"


# Aliases keyed lowercase for case-insensitive lookup.
_TEAM_ALIASES: dict[str, Team] = {
    "jax": Team.JAC,
    "la": Team.LAR,
    "stl": Team.LAR,   # Rams pre-2016
    "sd": Team.LAC,    # Chargers pre-2017
    "oak": Team.LV,    # Raiders pre-2020
    "wsh": Team.WAS,
    # Self-aliases for fast normalize_team_code passthrough:
    **{t.value.lower(): t for t in Team},
}


def normalize_team_code(code: str) -> Team:
    """Coerce a possibly-aliased team code to the canonical `Team`.

    Why: `nfl_data_py` and other sources use inconsistent codes across seasons.
    """
    try:
        return _TEAM_ALIASES[code.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown team code: {code!r}") from exc


class RosterSlot(StrEnum):
    """Roster slot identifiers used by downstream draft / lineup tools.

    SUPER_FLEX is included from day 1 even though the current league uses 1QB,
    so adding a superflex league later is a config flip rather than a rewrite.
    """

    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    FLEX = "FLEX"          # RB / WR / TE
    SUPER_FLEX = "SUPER_FLEX"  # QB / RB / WR / TE
    K = "K"
    DST = "DST"
    BENCH = "BENCH"
    IR = "IR"


class DistributionFamily(StrEnum):
    """Backing representation of a `Distribution`."""

    NORMAL = "NORMAL"
    GAMMA = "GAMMA"
    EMPIRICAL_QUANTILE = "EMPIRICAL_QUANTILE"  # quantile-regression output
    SAMPLED = "SAMPLED"                        # explicit sample array


class Stat(StrEnum):
    """Canonical column names for player stats. Reference these instead of literals
    in scoring rules and feature builders so typos fail at type-check time."""

    PASSING_YARDS = "passing_yards"
    PASSING_TDS = "passing_tds"
    INTERCEPTIONS = "interceptions"
    PASSING_2PT = "passing_2pt_conversions"
    RUSHING_YARDS = "rushing_yards"
    RUSHING_TDS = "rushing_tds"
    RUSHING_2PT = "rushing_2pt_conversions"
    RECEPTIONS = "receptions"
    RECEIVING_YARDS = "receiving_yards"
    RECEIVING_TDS = "receiving_tds"
    RECEIVING_2PT = "receiving_2pt_conversions"
    FUMBLES_LOST = "fumbles_lost"
    RETURN_TDS = "return_tds"
