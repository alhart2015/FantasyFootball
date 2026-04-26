"""Single source of truth for canonical types: enums, NewTypes, pydantic models, pandera schemas."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Final, NewType

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series
from pydantic import BaseModel, ConfigDict, Field

# Module constant: pyarrow-backed nullable string dtype.
# Used by every ingest module that needs to satisfy a pandera Series[str] field
# (object-dtype + plain strings will fail validation in pandera 0.31+).
_PYARROW_STR: Final = pd.StringDtype("pyarrow")


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
    "stl": Team.LAR,  # Rams pre-2016
    "sd": Team.LAC,  # Chargers pre-2017
    "oak": Team.LV,  # Raiders pre-2020
    "wsh": Team.WAS,
    # pro-football-reference 3-letter aliases (used by nfl_data_py.import_ids()).
    "gbp": Team.GB,
    "kan": Team.KC,
    "nwe": Team.NE,
    "nor": Team.NO,
    "sdg": Team.LAC,
    "tam": Team.TB,
    "sfo": Team.SF,
    "lvr": Team.LV,
    "rai": Team.LV,  # PFR's pre-Vegas Raiders
    "ram": Team.LAR,  # PFR's pre-2016 Rams
    "phx": Team.ARI,  # Phoenix Cardinals (pre-1994)
    "crd": Team.ARI,  # PFR's Cardinals
    "rav": Team.BAL,  # PFR's Ravens
    "clt": Team.IND,  # PFR's Colts (Baltimore -> Indianapolis legacy)
    "htx": Team.HOU,  # PFR's Texans
    "oti": Team.TEN,  # PFR's Titans (Oilers/Titans legacy)
    # Additional 3-letter team aliases observed in nfl_data_py.import_ids().
    "kcc": Team.KC,
    "nep": Team.NE,
    "nos": Team.NO,
    "sdc": Team.LAC,
    "tbb": Team.TB,
    # Self-aliases for fast normalize_team_code passthrough:
    **{t.value.lower(): t for t in Team},
}

# Sentinel codes meaning "no team" (free agent / not on a roster). These
# should map to None at ingest time rather than raise. Used by
# id_map ingest where ``team`` is nullable.
_NO_TEAM_CODES: frozenset[str] = frozenset({"fa", "fa*"})


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
    FLEX = "FLEX"  # RB / WR / TE
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
    SAMPLED = "SAMPLED"  # explicit sample array
    SAMPLED_SUMMARY = "SAMPLED_SUMMARY"  # per-stat dist params + summary in mean/p10/p50/p90


class Stat(StrEnum):
    """Canonical column names for player stats. Reference these instead of literals
    in scoring rules and feature builders so typos fail at type-check time."""

    PASSING_YARDS = "passing_yards"
    PASSING_TDS = "passing_tds"
    PASSING_ATTEMPTS = "attempts"
    COMPLETIONS = "completions"
    SACKS = "sacks"
    INTERCEPTIONS = "interceptions"
    PASSING_2PT = "passing_2pt_conversions"
    RUSHING_YARDS = "rushing_yards"
    RUSHING_TDS = "rushing_tds"
    RUSHING_2PT = "rushing_2pt_conversions"
    CARRIES = "carries"
    RECEPTIONS = "receptions"
    RECEIVING_YARDS = "receiving_yards"
    RECEIVING_TDS = "receiving_tds"
    RECEIVING_2PT = "receiving_2pt_conversions"
    RECEIVING_AIR_YARDS = "receiving_air_yards"
    TARGETS = "targets"
    FUMBLES_LOST = "fumbles_lost"
    RETURN_TDS = "return_tds"
    # Snap-counts column (not weekly_stats) — reserved so feature builders can
    # reference Stat.OFFENSE_PCT.value instead of a string literal.
    OFFENSE_PCT = "offense_pct"


# Each ID flavor is a distinct mypy type so passing one where another is expected
# is a type error. At runtime they are bare strings.
GsisId = NewType("GsisId", str)
EspnId = NewType("EspnId", str)
SleeperId = NewType("SleeperId", str)
PfrId = NewType("PfrId", str)

GSIS_ID_PATTERN: Final[str] = r"\d{2}-\d{7}"
_GSIS_ID_RE = re.compile(rf"^{GSIS_ID_PATTERN}$")


def validate_gsis_id(raw: str) -> GsisId:
    """Validate that `raw` matches the canonical gsis_id format and return it
    as a `GsisId`. The only sanctioned way to construct a `GsisId` from
    untrusted input."""
    if not _GSIS_ID_RE.fullmatch(raw):
        raise ValueError(f"Invalid gsis_id format: {raw!r}")
    return GsisId(raw)


class Ruleset(BaseModel):
    """Scoring ruleset. Defaults match ESPN standard PPR.

    Rulesets are immutable so we can hash/cache them and pass them around
    confidently. Use the named class methods for common presets, or pass field
    overrides for custom leagues.
    """

    model_config = ConfigDict(frozen=True)

    name: str = "ESPN_PPR"

    # Passing
    passing_yds_per_pt: float = Field(default=25.0, gt=0)
    passing_td_pts: float = 4.0
    interception_pts: float = -2.0

    # Rushing
    rushing_yds_per_pt: float = Field(default=10.0, gt=0)
    rushing_td_pts: float = 6.0

    # Receiving
    receiving_yds_per_pt: float = Field(default=10.0, gt=0)
    receiving_td_pts: float = 6.0
    reception_pts: float = 1.0  # PPR

    # Misc
    fumble_lost_pts: float = -2.0
    two_pt_pts: float = 2.0
    return_td_pts: float = 6.0

    @classmethod
    def espn_ppr(cls) -> Ruleset:
        return cls()

    @classmethod
    def espn_half(cls) -> Ruleset:
        return cls(name="ESPN_HALF", reception_pts=0.5)

    @classmethod
    def standard(cls) -> Ruleset:
        return cls(name="STANDARD", reception_pts=0.0)


# ---------------------------------------------------------------------------
# Pandera DataFrame schemas
# ---------------------------------------------------------------------------

_POSITION_VALUES = [p.value for p in Position]
_TEAM_VALUES = [t.value for t in Team]
_DIST_FAMILY_VALUES = [f.value for f in DistributionFamily]


class WeeklyStatsSchema(pa.DataFrameModel):
    """Canonical weekly stats — what `ingest.weekly_stats` produces."""

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=1999, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    opponent: Series[str] = pa.Field(isin=_TEAM_VALUES)
    passing_yards: Series[float] = pa.Field(ge=-100, le=800)
    passing_tds: Series[int] = pa.Field(ge=0, le=15)
    interceptions: Series[int] = pa.Field(ge=0, le=15)
    attempts: Series[int] = pa.Field(ge=0, le=80)
    completions: Series[int] = pa.Field(ge=0, le=60)
    sacks: Series[int] = pa.Field(ge=0, le=15)
    rushing_yards: Series[float] = pa.Field(ge=-50, le=400)
    rushing_tds: Series[int] = pa.Field(ge=0, le=10)
    carries: Series[int] = pa.Field(ge=0, le=50)
    receptions: Series[int] = pa.Field(ge=0, le=30)
    receiving_yards: Series[float] = pa.Field(ge=-50, le=400)
    receiving_tds: Series[int] = pa.Field(ge=0, le=10)
    receiving_air_yards: Series[float] = pa.Field(ge=-50, le=400)
    targets: Series[int] = pa.Field(ge=0, le=30)
    fumbles_lost: Series[int] = pa.Field(ge=0, le=10)

    class Config:
        strict = "filter"  # extra columns are dropped, not errored


class SchedulesSchema(pa.DataFrameModel):
    """Per-game schedule + Vegas line data — what `ingest.schedules` produces."""

    season: Series[int] = pa.Field(ge=1999, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    game_id: Series[str]
    home_team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    away_team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    kickoff: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"tz": "UTC", "unit": "us"}, nullable=True
    )
    spread_line: Series[float] = pa.Field(nullable=True)
    total_line: Series[float] = pa.Field(ge=0, le=100, nullable=True)
    home_moneyline: Series[int] = pa.Field(nullable=True)
    away_moneyline: Series[int] = pa.Field(nullable=True)
    surface: Series[str] = pa.Field(nullable=True)
    roof: Series[str] = pa.Field(nullable=True)
    temp: Series[int] = pa.Field(nullable=True)
    wind: Series[int] = pa.Field(nullable=True)

    class Config:
        strict = "filter"


class SnapCountsSchema(pa.DataFrameModel):
    """Per-player per-game snap counts — what `ingest.snap_counts` produces."""

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=1999, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    opponent: Series[str] = pa.Field(isin=_TEAM_VALUES)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    offense_snaps: Series[int] = pa.Field(ge=0, le=200)
    offense_pct: Series[float] = pa.Field(ge=0, le=1)
    defense_snaps: Series[int] = pa.Field(ge=0, le=200)
    defense_pct: Series[float] = pa.Field(ge=0, le=1)
    st_snaps: Series[int] = pa.Field(ge=0, le=100)
    st_pct: Series[float] = pa.Field(ge=0, le=1)

    class Config:
        strict = "filter"


class DepthChartsSchema(pa.DataFrameModel):
    """Per-team per-week depth chart — what `ingest.depth_charts` produces.

    `depth_team` is the raw slot label from nfl_data_py (e.g., "WR1", "LWR").
    `depth_rank` is parsed numeric rank within the position group (1 = starter).
    """

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=1999, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    depth_team: Series[str]
    depth_rank: Series[int] = pa.Field(ge=1, le=10)

    class Config:
        strict = "filter"


class NgsPassingSchema(pa.DataFrameModel):
    """NGS passing — season-to-date weekly snapshot per QB.
    Coverage starts 2016 (RFID-chip era)."""

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=2016, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    avg_time_to_throw: Series[float] = pa.Field(nullable=True)
    avg_completed_air_yards: Series[float] = pa.Field(nullable=True)
    avg_intended_air_yards: Series[float] = pa.Field(nullable=True)
    avg_air_yards_differential: Series[float] = pa.Field(nullable=True)
    aggressiveness: Series[float] = pa.Field(nullable=True)
    max_completed_air_distance: Series[float] = pa.Field(nullable=True)
    avg_air_yards_to_sticks: Series[float] = pa.Field(nullable=True)
    completion_percentage: Series[float] = pa.Field(nullable=True)
    expected_completion_percentage: Series[float] = pa.Field(nullable=True)
    completion_percentage_above_expectation: Series[float] = pa.Field(nullable=True)
    avg_air_distance: Series[float] = pa.Field(nullable=True)
    max_air_distance: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = "filter"


class NgsRushingSchema(pa.DataFrameModel):
    """NGS rushing — season-to-date weekly snapshot per ball-carrier.
    Coverage starts 2016."""

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=2016, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    efficiency: Series[float] = pa.Field(nullable=True)
    percent_attempts_gte_eight_defenders: Series[float] = pa.Field(nullable=True)
    avg_time_to_los: Series[float] = pa.Field(nullable=True)
    rush_attempts: Series[int] = pa.Field(ge=0, nullable=True)
    rush_yards: Series[int] = pa.Field(nullable=True)
    expected_rush_yards: Series[float] = pa.Field(nullable=True)
    rush_yards_over_expected: Series[float] = pa.Field(nullable=True)
    avg_rush_yards: Series[float] = pa.Field(nullable=True)
    rush_yards_over_expected_per_att: Series[float] = pa.Field(nullable=True)
    rush_pct_over_expected: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = "filter"


class NgsReceivingSchema(pa.DataFrameModel):
    """NGS receiving — season-to-date weekly snapshot per target-receiver.
    Coverage starts 2016."""

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=2016, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    avg_cushion: Series[float] = pa.Field(nullable=True)
    avg_separation: Series[float] = pa.Field(nullable=True)
    avg_intended_air_yards: Series[float] = pa.Field(nullable=True)
    percent_share_of_intended_air_yards: Series[float] = pa.Field(nullable=True)
    receptions: Series[int] = pa.Field(ge=0, nullable=True)
    targets: Series[int] = pa.Field(ge=0, nullable=True)
    catch_percentage: Series[float] = pa.Field(nullable=True)
    yards: Series[int] = pa.Field(nullable=True)
    rec_touchdowns: Series[int] = pa.Field(ge=0, nullable=True)
    avg_yac: Series[float] = pa.Field(nullable=True)
    avg_expected_yac: Series[float] = pa.Field(nullable=True)
    avg_yac_above_expectation: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = "filter"


class WrFeaturesSchema(pa.DataFrameModel):
    """WR feature DataFrame produced by `features.wr.build_wr_features`.
    Schema enforced at the module boundary — every column has a typed range
    so a feature regression surfaces at validate(), not three modules deep."""

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=1999, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    opponent: Series[str] = pa.Field(isin=_TEAM_VALUES)

    # Receiving usage (rolling)
    targets_per_game_l4: Series[float] = pa.Field(ge=0)
    targets_per_game_std: Series[float] = pa.Field(ge=0)
    target_share_l4: Series[float] = pa.Field(ge=0, le=1)
    air_yards_share_l4: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    receptions_per_game_l4: Series[float] = pa.Field(ge=0)
    # Trailing-mean yards: underlying weekly stat allows negatives (lost yardage,
    # sacks bookkept against passing). The mean of nonneg + (occasionally) negs
    # may itself be negative, so no ge=0 lower bound here.
    receiving_yards_per_game_l4: Series[float]
    receiving_tds_per_game_l4: Series[float] = pa.Field(ge=0)

    # Rushing usage (Deebo / jet-sweep WRs)
    rushing_attempts_per_game_l4: Series[float] = pa.Field(ge=0)
    rushing_yards_per_game_l4: Series[float]  # see receiving_yards_per_game_l4 note
    designed_rusher: Series[bool]

    # Snap / role
    snap_pct_l4: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    depth_rank: Series[int] = pa.Field(ge=1, le=10, nullable=True)

    # NGS receiving (season-to-date snapshot from prior week)
    avg_separation_std: Series[float] = pa.Field(nullable=True)
    avg_intended_air_yards_std: Series[float] = pa.Field(nullable=True)
    percent_share_intended_air_yards_std: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    avg_yac_above_expectation_std: Series[float] = pa.Field(nullable=True)

    # Game environment (from schedules)
    implied_team_total: Series[float] = pa.Field(ge=0, le=60, nullable=True)
    spread: Series[float] = pa.Field(nullable=True)
    is_home: Series[bool]
    roof_dome: Series[bool]

    # Opponent strength (proxy: opp's allowed WR fantasy points/game over trailing 4)
    opp_allowed_wr_fppg_l4: Series[float] = pa.Field(ge=0, nullable=True)

    class Config:
        strict = "filter"
        # Required so the empty-depth-chart fast path validates: a
        # `pd.DataFrame(columns=...)` produces object-dtype columns, which
        # pandera otherwise rejects against the typed Series declarations.
        coerce = True


class QbFeaturesSchema(pa.DataFrameModel):
    """QB feature DataFrame produced by `features.qb.build_qb_features`."""

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=1999, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    opponent: Series[str] = pa.Field(isin=_TEAM_VALUES)

    # Passing usage (rolling)
    pass_attempts_per_game_l4: Series[float] = pa.Field(ge=0)
    # Trailing-mean yards: underlying weekly stat allows negatives (sacks
    # subtract from passing_yards, scrambles can lose rushing yardage). The
    # mean over 4 games can therefore go negative, so no ge=0 lower bound here.
    passing_yards_per_game_l4: Series[float]
    passing_tds_per_game_l4: Series[float] = pa.Field(ge=0)
    interceptions_per_game_l4: Series[float] = pa.Field(ge=0)
    sacks_per_game_l4: Series[float] = pa.Field(ge=0)
    # Season-to-date mean of passing_yards: same negative-allowed underlying
    # rationale as passing_yards_per_game_l4 above.
    passing_yards_per_game_std: Series[float]

    # Rushing usage
    rushing_attempts_per_game_l4: Series[float] = pa.Field(ge=0)
    rushing_yards_per_game_l4: Series[float]  # see passing_yards_per_game_l4 note
    rushing_qb: Series[bool]

    # Snap / role
    snap_pct_l4: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    depth_rank: Series[int] = pa.Field(ge=1, le=10, nullable=True)

    # NGS passing (season-to-date snapshot from prior week)
    aggressiveness_std: Series[float] = pa.Field(nullable=True)
    completion_percentage_above_expectation_std: Series[float] = pa.Field(nullable=True)
    avg_intended_air_yards_std: Series[float] = pa.Field(nullable=True)
    avg_time_to_throw_std: Series[float] = pa.Field(nullable=True)

    # Game environment
    implied_team_total: Series[float] = pa.Field(ge=0, le=60, nullable=True)
    spread: Series[float] = pa.Field(nullable=True)
    is_home: Series[bool]
    roof_dome: Series[bool]

    # Opponent strength proxy
    opp_allowed_qb_fppg_l4: Series[float] = pa.Field(ge=0, nullable=True)

    class Config:
        strict = "filter"
        coerce = True  # see WrFeaturesSchema.Config for rationale


class RbFeaturesSchema(pa.DataFrameModel):
    """RB feature DataFrame produced by `features.rb.build_rb_features`."""

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=1999, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    opponent: Series[str] = pa.Field(isin=_TEAM_VALUES)

    # Rushing usage (rolling)
    carries_per_game_l4: Series[float] = pa.Field(ge=0)
    # Trailing-mean yards: underlying weekly stat allows negatives (lost
    # yardage on TFL/scrambles), so no ge=0 lower bound on the L4 mean.
    rushing_yards_per_game_l4: Series[float]
    rushing_tds_per_game_l4: Series[float] = pa.Field(ge=0)
    rush_share_l4: Series[float] = pa.Field(ge=0, le=1)

    # Receiving usage
    targets_per_game_l4: Series[float] = pa.Field(ge=0)
    receptions_per_game_l4: Series[float] = pa.Field(ge=0)
    receiving_yards_per_game_l4: Series[float]  # see rushing_yards_per_game_l4 note
    target_share_l4: Series[float] = pa.Field(ge=0, le=1)
    targets_per_game_std: Series[float] = pa.Field(ge=0)

    # Snap / role
    snap_pct_l4: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    depth_rank: Series[int] = pa.Field(ge=1, le=10, nullable=True)
    passing_down_back: Series[bool]

    # NGS rushing (season-to-date snapshot from prior week)
    efficiency_std: Series[float] = pa.Field(nullable=True)
    rush_yards_over_expected_per_att_std: Series[float] = pa.Field(nullable=True)
    percent_attempts_gte_eight_defenders_std: Series[float] = pa.Field(nullable=True)

    # Game environment
    implied_team_total: Series[float] = pa.Field(ge=0, le=60, nullable=True)
    spread: Series[float] = pa.Field(nullable=True)
    is_home: Series[bool]
    roof_dome: Series[bool]

    # Opponent strength proxy
    opp_allowed_rb_fppg_l4: Series[float] = pa.Field(ge=0, nullable=True)

    class Config:
        strict = "filter"
        coerce = True  # see WrFeaturesSchema.Config for rationale


class TeFeaturesSchema(pa.DataFrameModel):
    """TE feature DataFrame produced by `features.te.build_te_features`."""

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=1999, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    opponent: Series[str] = pa.Field(isin=_TEAM_VALUES)

    # Receiving usage (rolling)
    targets_per_game_l4: Series[float] = pa.Field(ge=0)
    targets_per_game_std: Series[float] = pa.Field(ge=0)
    target_share_l4: Series[float] = pa.Field(ge=0, le=1)
    receptions_per_game_l4: Series[float] = pa.Field(ge=0)
    # Trailing-mean yards: underlying weekly stat allows negatives, so no ge=0
    # lower bound on the L4 mean.
    receiving_yards_per_game_l4: Series[float]
    receiving_tds_per_game_l4: Series[float] = pa.Field(ge=0)

    # Rushing usage (rolling) — added Plan 3b for Taysom-Hill-shape TEs that
    # carry the ball; mirrors RB's rushing-feature shape.
    rushing_attempts_per_game_l4: Series[float] = pa.Field(ge=0)
    rushing_yards_per_game_l4: Series[float]  # see receiving_yards_per_game_l4 note

    # Snap / role
    snap_pct_l4: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    depth_rank: Series[int] = pa.Field(ge=1, le=10, nullable=True)

    # NGS receiving (season-to-date snapshot from prior week)
    avg_separation_std: Series[float] = pa.Field(nullable=True)
    avg_intended_air_yards_std: Series[float] = pa.Field(nullable=True)
    avg_yac_above_expectation_std: Series[float] = pa.Field(nullable=True)

    # Game environment
    implied_team_total: Series[float] = pa.Field(ge=0, le=60, nullable=True)
    spread: Series[float] = pa.Field(nullable=True)
    is_home: Series[bool]
    roof_dome: Series[bool]

    # Opponent strength proxy
    opp_allowed_te_fppg_l4: Series[float] = pa.Field(ge=0, nullable=True)

    class Config:
        strict = "filter"
        coerce = True  # see WrFeaturesSchema.Config for rationale


class IdMapSchema(pa.DataFrameModel):
    """Cross-platform player id translation table."""

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$", unique=True)
    espn_id: Series[str] = pa.Field(nullable=True)
    sleeper_id: Series[str] = pa.Field(nullable=True)
    pfr_id: Series[str] = pa.Field(nullable=True)
    full_name: Series[str]
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES, nullable=True)

    class Config:
        strict = "filter"


class ProjectionWeeklySchema(pa.DataFrameModel):
    """Published per-week projection (the consumer-facing contract)."""

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=1999, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    opponent: Series[str] = pa.Field(isin=_TEAM_VALUES)
    ruleset: Series[str]
    family: Series[str] = pa.Field(isin=_DIST_FAMILY_VALUES)
    params: Series[bytes]
    mean: Series[float]
    p10: Series[float]
    p50: Series[float]
    p90: Series[float]
    model_id: Series[str]
    # pandas >=2.0 stores timezone-aware timestamps as datetime64[us, UTC];
    # use unit='us' to match the actual dtype produced by pd.Timestamp(..., tz='UTC').
    generated_at: Series[pd.DatetimeTZDtype] = pa.Field(dtype_kwargs={"tz": "UTC", "unit": "us"})

    class Config:
        strict = "filter"
        coerce = True  # see WrFeaturesSchema.Config — empty-output fast path


class ProjectionSeasonSchema(pa.DataFrameModel):
    """Published per-season projection (consumer-facing contract for season totals).

    Aggregates per-week samples across the weeks a player has predictions for in a
    season. n_weeks reports how many weeks were aggregated; consumers may filter
    by it. position is the modal value from the input rows for the gsis_id (the
    rare in-season position change inherits the most-frequent value).
    """

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=1999, le=2100)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    ruleset: Series[str]
    n_weeks: Series[int] = pa.Field(ge=1, le=22)
    season_mean: Series[float]
    season_p10: Series[float]
    season_p50: Series[float]
    season_p90: Series[float]
    model_id: Series[str]
    generated_at: Series[pd.DatetimeTZDtype] = pa.Field(dtype_kwargs={"tz": "UTC", "unit": "us"})

    # `coerce = True` is required for empty-DataFrame validation (an empty
    # pd.DataFrame(columns=[...]) produces object-dtype columns); but pandera's
    # coerce will silently UTC-localize a naive datetime column rather than reject
    # it. We want naive timestamps treated as a producer bug, so we run a
    # pre-coerce parser that raises SchemaError when the input is naive.
    @pa.dataframe_parser
    @classmethod
    def _reject_naive_generated_at(cls, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0 or "generated_at" not in df.columns:
            return df
        col = df["generated_at"]
        if hasattr(col, "dt") and col.dt.tz is None:
            # pandera's SchemaError lacks type stubs; narrow ignore to the missing-stub call.
            raise pa.errors.SchemaError(  # type: ignore[no-untyped-call]
                schema=cls.to_schema(),
                data=df,
                message=(
                    "column 'generated_at': naive datetime not allowed; expected tz-aware UTC"
                ),
            )
        return df

    class Config:
        strict = "filter"
        coerce = True
