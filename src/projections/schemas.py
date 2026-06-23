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
    NEGATIVE_BINOMIAL = "NEGATIVE_BINOMIAL"  # NEW (Plan 3e Phase 1)
    STUDENT_T = "STUDENT_T"  # NEW (Plan 3e Phase 2) — heavy-tailed continuous
    SAMPLED = "SAMPLED"  # explicit sample array
    SAMPLED_SUMMARY = "SAMPLED_SUMMARY"  # per-stat dist params + summary in mean/p10/p50/p90
    QUANTILE = "QUANTILE"  # NEW (Plan 5) — Model C (LightGBM quantile regression)
    MIXED = "MIXED"  # NEW (Plan 5c) — per-row distribution mixes families per stat
    MIXTURE = "MIXTURE"  # Plan 6 — per-stat: weighted mixture of two child distributions


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


# The 9 canonical preseason stat-line fields carried by ExternalProjectionSchema and
# ConsensusProjectionSchema. Single source for the ingest producer (external_projections) and
# the consensus blend, which must stay in lockstep. Derived from Stat so a typo can't drift.
STAT_FIELDS: Final[tuple[str, ...]] = (
    Stat.PASSING_YARDS.value,
    Stat.PASSING_TDS.value,
    Stat.INTERCEPTIONS.value,
    Stat.RUSHING_YARDS.value,
    Stat.RUSHING_TDS.value,
    Stat.RECEPTIONS.value,
    Stat.RECEIVING_YARDS.value,
    Stat.RECEIVING_TDS.value,
    Stat.FUMBLES_LOST.value,
)

# ESPN-only auction-value columns (crowd average + PPR/STANDARD expert), carried through
# external_projections -> consensus. Single source of truth, imported by the ingest + blend
# layers (mirrors STAT_FIELDS).
ESPN_AUCTION_COLS: Final[tuple[str, ...]] = (
    "espn_auction_value_avg",
    "espn_auction_value_ppr",
    "espn_auction_value_std",
)


class ProjectionSource(StrEnum):
    """External preseason projection sources. Use ProjectionSource.ESPN, never "ESPN"."""

    ESPN = "ESPN"
    SLEEPER = "SLEEPER"


# Each ID flavor is a distinct mypy type so passing one where another is expected
# is a type error. At runtime they are bare strings.
GsisId = NewType("GsisId", str)
EspnId = NewType("EspnId", str)
SleeperId = NewType("SleeperId", str)
PfrId = NewType("PfrId", str)

GSIS_ID_PATTERN: Final[str] = r"\d{2}-\d{7}"
_GSIS_ID_RE = re.compile(rf"^{GSIS_ID_PATTERN}$")

# Canonical resolution for tz-aware datetime columns. polars/nflreadpy `.to_pandas()` and
# pyarrow parquet yield microseconds; pandas 2.x `merge_asof` and joins reject mixed
# datetime64[us]/[ns] keys, so every datetime column we validate or join is pinned to this
# unit (reference it; don't hardcode "us" at call sites). See ingest.depth_charts.
DATETIME_UNIT: Final[str] = "us"


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

    @classmethod
    def draftkings(cls) -> Ruleset:
        """DraftKings NFL Classic *base* scoring (skill positions, no yardage
        bonuses — those are a separate deterministic helper, see
        scoring.draftkings_bonus). Differs from ESPN PPR only in turnovers:
        INT and fumble lost are -1 (ESPN: -2)."""
        return cls(name="DRAFTKINGS", interception_pts=-1.0, fumble_lost_pts=-1.0)


# ---------------------------------------------------------------------------
# Pandera DataFrame schemas
# ---------------------------------------------------------------------------

_POSITION_VALUES = [p.value for p in Position]
_SKILL_POSITION_VALUES = [
    Position.QB.value,
    Position.RB.value,
    Position.WR.value,
    Position.TE.value,
]
_TEAM_VALUES = [t.value for t in Team]
_DIST_FAMILY_VALUES = [f.value for f in DistributionFamily]
_RULESET_NAME_VALUES = ["ESPN_PPR", "ESPN_HALF", "STANDARD", "DRAFTKINGS"]
_BACKTEST_VERDICT_VALUES = ["ADOPT", "NULL", "DO_NOT_ADOPT"]
_SOURCE_VALUES = [s.value for s in ProjectionSource]


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
    # 2025-W?? Tyler Higbee shows -92 receiving_air_yards (target on a behind-the-LOS
    # screen / shovel that lost ~92 yards). Empirical min on 2018-2024 was -33; the
    # wider bound here gives headroom for similar anomalies under the new nflverse
    # release schema.
    receiving_air_yards: Series[float] = pa.Field(ge=-100, le=400)
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
        dtype_kwargs={"tz": "UTC", "unit": DATETIME_UNIT}, nullable=True
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


class PbpSchema(pa.DataFrameModel):
    """Per-play data — what `ingest.pbp` produces. Curated subset of
    `nfl_data_py.import_pbp_data`'s ~370-column output."""

    play_id: Series[int] = pa.Field(ge=1)
    game_id: Series[str]
    season: Series[int] = pa.Field(ge=1999, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    posteam: Series[str] = pa.Field(isin=_TEAM_VALUES, nullable=True)
    defteam: Series[str] = pa.Field(isin=_TEAM_VALUES, nullable=True)
    play_type: Series[str] = pa.Field(nullable=True)
    qb_dropback: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    qb_scramble: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    sack: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    rush_attempt: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    pass_attempt: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    epa: Series[float] = pa.Field(nullable=True)
    wpa: Series[float] = pa.Field(nullable=True)
    success: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    air_yards: Series[float] = pa.Field(nullable=True)
    yards_after_catch: Series[float] = pa.Field(nullable=True)
    complete_pass: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    xpass: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    pass_oe: Series[float] = pa.Field(nullable=True)
    down: Series[float] = pa.Field(ge=1, le=4, nullable=True)
    ydstogo: Series[int] = pa.Field(ge=0, le=99, nullable=True)
    yardline_100: Series[float] = pa.Field(ge=0, le=100, nullable=True)
    half_seconds_remaining: Series[float] = pa.Field(ge=0, le=1800, nullable=True)
    passer_player_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$", nullable=True)
    rusher_player_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$", nullable=True)
    receiver_player_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$", nullable=True)

    class Config:
        strict = "filter"


class DraftPicksSchema(pa.DataFrameModel):
    """Per-player NFL draft pick metadata — what `ingest.draft_picks` produces.

    Snapshot semantics: a season's draft never changes after the draft completes.
    Source: `nfl_data_py.import_draft_picks`. UDFAs and pre-coverage players
    (drafts before 1980) are not present; downstream feature compute handles
    that with an inferred-draft-year fallback (see trajectory_features.py).
    """

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$", unique=True)
    draft_year: Series[int] = pa.Field(ge=1936, le=2100)
    draft_round: Series[int] = pa.Field(ge=1, le=15, nullable=True)
    draft_overall_pick: Series[int] = pa.Field(ge=1, le=500, nullable=True)
    pfr_id: Series[str] = pa.Field(nullable=True)
    draft_age: Series[float] = pa.Field(ge=18.0, le=40.0, nullable=True)

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

    # Vegas team-context (TODO #33c integration). Same shape as QbFeaturesSchema —
    # preseason_* broadcast from week 1; season_avg_* is expanding mean over
    # weeks 1..N-1 (NaN at week 1).
    preseason_implied_team_total: Series[float] = pa.Field(ge=0, le=60, nullable=True)
    preseason_spread: Series[float] = pa.Field(nullable=True)
    season_avg_implied_team_total: Series[float] = pa.Field(ge=0, le=60, nullable=True)
    season_avg_spread: Series[float] = pa.Field(nullable=True)

    # Opponent strength (proxy: opp's allowed WR fantasy points/game over trailing 4)
    opp_allowed_wr_fppg_l4: Series[float] = pa.Field(ge=0, nullable=True)

    # Trajectory features (PR #25 family probe + 2026-05-03 WR integration
    # spec). All four are structurally sparse: age + is_rookie need a
    # draft_picks lookup hit (or the inferred fallback) and so cover ~88-97%
    # of player-weeks; the trend cols need 8 prior active games and so cover
    # ~50% of player-weeks. NaN where coverage is missing; BaselineModel
    # imputes with feature mean, lightgbm consumes NaN natively.
    age: Series[float] = pa.Field(ge=15, le=50, nullable=True)
    is_rookie: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    volume_trend_l4_minus_prior_l4: Series[float] = pa.Field(nullable=True)
    snap_pct_change_l4_vs_prior_l4: Series[float] = pa.Field(ge=-1, le=1, nullable=True)

    # Weather features (PR #28 family probe + 2026-05-08 RB+WR integration
    # spec). Sourced from existing SchedulesSchema columns (wind, temp, roof,
    # surface) — no new ingest. Dome / closed-roof games filled with
    # (wind=0, temp=70) per compute_weather_features semantics. Outdoor NaN
    # wind/temp propagates; ~8% NaN rate concentrated in 2018-2019.
    wind_speed_mph: Series[float] = pa.Field(ge=0, nullable=True)
    is_high_wind: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    temperature_f: Series[float] = pa.Field(nullable=True)
    is_grass_surface: Series[float] = pa.Field(ge=0, le=1, nullable=True)

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

    # Vegas team-context (TODO #33c integration). Sourced from
    # SchedulesSchema.spread_line / total_line. preseason_* broadcast from
    # the team's week-1 game; season_avg_* is the expanding mean over
    # weeks 1..N-1 (NaN at week 1 by design).
    preseason_implied_team_total: Series[float] = pa.Field(ge=0, le=60, nullable=True)
    preseason_spread: Series[float] = pa.Field(nullable=True)
    season_avg_implied_team_total: Series[float] = pa.Field(ge=0, le=60, nullable=True)
    season_avg_spread: Series[float] = pa.Field(nullable=True)

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

    # PBP team-level features (PR #20 family probe + 2026-05-01 RB
    # integration spec). Trailing 4 prior games; NaN for early-season
    # weeks where fewer than 4 prior games exist (notably 2018 weeks 1-4,
    # the start of the curated PBP window).
    pace_l4: Series[float] = pa.Field(nullable=True)
    proe_l4: Series[float] = pa.Field(nullable=True)
    team_ayps_l4: Series[float] = pa.Field(ge=0, nullable=True)
    team_def_epa_resid_l4: Series[float] = pa.Field(nullable=True)

    # Weather features (PR #28 family probe + 2026-05-08 RB+WR integration
    # spec). Sourced from existing SchedulesSchema columns (wind, temp, roof,
    # surface) — no new ingest. Dome / closed-roof games filled with
    # (wind=0, temp=70) per compute_weather_features semantics. Outdoor NaN
    # wind/temp propagates; ~8% NaN rate concentrated in 2018-2019.
    wind_speed_mph: Series[float] = pa.Field(ge=0, nullable=True)
    is_high_wind: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    temperature_f: Series[float] = pa.Field(nullable=True)
    is_grass_surface: Series[float] = pa.Field(ge=0, le=1, nullable=True)

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

    # Trajectory features (PR #25 family probe + 2026-05-04 TE integration
    # spec). All four are structurally sparse: age + is_rookie need a
    # draft_picks lookup hit (or the inferred fallback) and so cover ~95%
    # of TE player-weeks per the probe; the trend cols need 8 prior active
    # games and so cover ~45-71% of TE player-weeks. NaN where coverage
    # is missing; BaselineModel imputes with feature mean, lightgbm
    # consumes NaN natively.
    age: Series[float] = pa.Field(ge=15, le=50, nullable=True)
    is_rookie: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    volume_trend_l4_minus_prior_l4: Series[float] = pa.Field(nullable=True)
    snap_pct_change_l4_vs_prior_l4: Series[float] = pa.Field(ge=-1, le=1, nullable=True)

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


class ExternalProjectionSchema(pa.DataFrameModel):
    """One row per (source, player, season, asof) of external preseason projection data.

    Stat line is nullable: ESPN provides it; Sleeper provides ADP only (null stat line).
    gsis_id is the real id for crosswalked veterans, else a synthetic 99-XXXXXXX placeholder
    (flagged is_placeholder_gsis) for pre-camp rookies; source_player_id is the stable
    cross-snapshot join key.
    """

    source: Series[str] = pa.Field(isin=_SOURCE_VALUES)
    source_player_id: Series[str]
    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    is_placeholder_gsis: Series[bool]
    full_name: Series[str]
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    season: Series[int] = pa.Field(ge=1999, le=2100)
    # ISO YYYY-MM-DD; also encoded in the partition path
    asof: Series[str] = pa.Field(str_matches=r"^\d{4}-\d{2}-\d{2}$")
    adp: Series[float] = pa.Field(nullable=True)
    espn_draft_rank: Series[float] = pa.Field(nullable=True)
    # Optional (not-required): ESPN-only auction values; absent on the Sleeper path and on
    # partitions written before this column existed. Float64 to avoid the NaN dtype-regression.
    espn_auction_value_avg: Series[pd.Float64Dtype] | None = pa.Field(nullable=True)
    espn_auction_value_ppr: Series[pd.Float64Dtype] | None = pa.Field(nullable=True)
    espn_auction_value_std: Series[pd.Float64Dtype] | None = pa.Field(nullable=True)
    passing_yards: Series[float] = pa.Field(nullable=True)
    passing_tds: Series[float] = pa.Field(nullable=True)
    interceptions: Series[float] = pa.Field(nullable=True)
    rushing_yards: Series[float] = pa.Field(nullable=True)
    rushing_tds: Series[float] = pa.Field(nullable=True)
    receptions: Series[float] = pa.Field(nullable=True)
    receiving_yards: Series[float] = pa.Field(nullable=True)
    receiving_tds: Series[float] = pa.Field(nullable=True)
    fumbles_lost: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = "filter"
        coerce = True


class ConsensusProjectionSchema(pa.DataFrameModel):
    """Published preseason consensus projection: one row per (gsis_id, season, asof).

    The consumer-facing contract downstream draft tooling reads. `consensus_adp` is the mean of
    available source ADPs (nullable -- a stat-line-only / unranked player can have none);
    `consensus_rank` is the ordinal over non-null `consensus_adp` (null when adp is null). The
    stat line + `projected_points_ppr` are present only for players a stat-line source covers
    (`has_points`). v1 sources: ESPN (stat line + ADP) + Sleeper (ADP only).

    Nullable floats use the pandas `Float64` extension dtype (the blend builder emits `pd.NA`),
    unlike the looser `ExternalProjectionSchema` raw-ingest counterpart. The `has_points` /
    stat-line consistency is a blend-builder invariant, not schema-enforced.
    """

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$", unique=True)
    season: Series[pd.Int64Dtype] = pa.Field(ge=1999, le=2100)
    # ISO YYYY-MM-DD; mirrors the raw external_projections snapshot this was derived from
    asof: Series[str] = pa.Field(str_matches=r"^\d{4}-\d{2}-\d{2}$")
    full_name: Series[str]
    position: Series[str] = pa.Field(isin=_SKILL_POSITION_VALUES)
    consensus_adp: Series[pd.Float64Dtype] = pa.Field(gt=0, nullable=True)
    consensus_rank: Series[pd.Int64Dtype] = pa.Field(ge=1, nullable=True)
    n_adp_sources: Series[pd.Int64Dtype] = pa.Field(ge=0)
    has_points: Series[bool]
    projected_points_ppr: Series[pd.Float64Dtype] = pa.Field(nullable=True)
    passing_yards: Series[pd.Float64Dtype] = pa.Field(nullable=True)
    passing_tds: Series[pd.Float64Dtype] = pa.Field(nullable=True)
    interceptions: Series[pd.Float64Dtype] = pa.Field(nullable=True)
    rushing_yards: Series[pd.Float64Dtype] = pa.Field(nullable=True)
    rushing_tds: Series[pd.Float64Dtype] = pa.Field(nullable=True)
    receptions: Series[pd.Float64Dtype] = pa.Field(nullable=True)
    receiving_yards: Series[pd.Float64Dtype] = pa.Field(nullable=True)
    receiving_tds: Series[pd.Float64Dtype] = pa.Field(nullable=True)
    fumbles_lost: Series[pd.Float64Dtype] = pa.Field(nullable=True)
    # Optional (not-required): ESPN-only auction values carried from external_projections.
    espn_auction_value_avg: Series[pd.Float64Dtype] | None = pa.Field(nullable=True)
    espn_auction_value_ppr: Series[pd.Float64Dtype] | None = pa.Field(nullable=True)
    espn_auction_value_std: Series[pd.Float64Dtype] | None = pa.Field(nullable=True)
    is_placeholder_gsis: Series[bool]
    ruleset: Series[str] = pa.Field(isin=_RULESET_NAME_VALUES)

    class Config:
        strict = "filter"
        coerce = True


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
    # DATETIME_UNIT matches the actual dtype produced by pd.Timestamp(..., tz='UTC').
    generated_at: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"tz": "UTC", "unit": DATETIME_UNIT}
    )

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
    generated_at: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"tz": "UTC", "unit": DATETIME_UNIT}
    )

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


class VorpTableSchema(pa.DataFrameModel):
    """Per-player VORP table. Consumer-facing output of the VORP generator.

    Direct input contract for `AuctionValuesSchema`'s upstream and for the
    snake-draft cheat sheet. One row per player at a position present in the
    caller's `LeagueConfig.roster_slots`; rows at out-of-scope positions are
    dropped by the generator. `vorp` may be negative (sub-replacement players).
    """

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$", unique=True)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    season_mean_fpts: Series[float]
    vorp: Series[float]
    replacement_fpts: Series[float] = pa.Field(ge=0)
    # Optional (not-required): populated only on the consensus-fed path (the raw market ADP
    # the cheat sheet's adp_delta uses). Weekly-path VORP tables omit it and still validate.
    consensus_adp: Series[pd.Float64Dtype] | None = pa.Field(gt=0, nullable=True)
    # Optional (not-required): populated only on the consensus-fed path (the player's
    # display name, incl. placeholder-gsis rookies absent from id_map). Weekly-path VORP
    # tables omit it and still validate. Nullable: a player with no consensus name is NA.
    full_name: Series[str] | None = pa.Field(nullable=True)
    # Optional (not-required): the resolved ESPN human auction value, populated only on the
    # consensus-fed preset path. Weekly-path VORP tables omit it and still validate. Slice 1
    # lands it here; Slice 2 feeds it to generate_auction_values as reference_prices.
    espn_auction_dollars: Series[pd.Int64Dtype] | None = pa.Field(ge=0, nullable=True)

    class Config:
        strict = "filter"
        coerce = True


class SnakeCheatSheetSchema(pa.DataFrameModel):
    """Per-player snake-draft cheat sheet. End-user surface for draft day.

    One row per player in the input VORP table. In-pool players get a numeric
    tier (1..N); out-of-pool players get tier = NA. `display_name` is
    best-effort from id_map.parquet; falls back to '—' for players without
    an id_map row.
    """

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$", unique=True)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    display_name: Series[str]
    positional_rank: Series[pd.Int64Dtype] = pa.Field(ge=1)
    season_mean_fpts: Series[float]
    vorp: Series[float]
    replacement_fpts: Series[float]
    is_in_pool: Series[bool]
    tier: Series[pd.Int64Dtype] = pa.Field(ge=1, nullable=True)
    # Raw consensus ADP (market view) carried from the VORP table; NA on the weekly path.
    consensus_adp: Series[pd.Float64Dtype] = pa.Field(gt=0, nullable=True)
    # Within-position (ADP-rank - VORP-rank): positive = value, negative = reach. NA when
    # consensus_adp is NA (weekly path, or a player no source gave an ADP).
    adp_delta: Series[pd.Int64Dtype] = pa.Field(nullable=True)

    class Config:
        strict = "filter"
        coerce = True


class AuctionValuesSchema(pa.DataFrameModel):
    """Per-player auction $ allocation. Consumer-facing output of the auction-values generator.

    One row per player VORP knows about. `in_pool=False` rows have `auction_dollars=0`
    and `pool_rank=NA`. `reference_dollars` and `value_delta` are present in every
    output frame; both are all-NA when the caller didn't supply a reference-prices CSV.
    """

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$", unique=True)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    season_mean_fpts: Series[float]
    vorp: Series[float]
    in_pool: Series[bool]
    auction_dollars: Series[pd.Int64Dtype] = pa.Field(ge=0)
    pool_rank: Series[pd.Int64Dtype] = pa.Field(ge=1, nullable=True)
    reference_dollars: Series[pd.Int64Dtype] = pa.Field(ge=0, nullable=True)
    value_delta: Series[pd.Int64Dtype] = pa.Field(nullable=True)

    class Config:
        strict = "filter"
        coerce = True


class RecommendationSchema(pa.DataFrameModel):
    """Ranked draft-pick recommendation — the output of a `DraftStrategy`.

    One row per roster-eligible *available* player. `rank` is 1-based and dense
    in the final ordering (`(fills_starting_slot desc, score desc, vorp desc,
    gsis_id asc)`). `p_available_next` is null for null-ADP players and on the
    raw-VORP / last-pick-fallback paths.
    """

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$", unique=True)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    vorp: Series[float]
    consensus_adp: Series[pd.Float64Dtype] = pa.Field(gt=0, nullable=True)
    p_available_next: Series[pd.Float64Dtype] = pa.Field(ge=0, le=1, nullable=True)
    fills_starting_slot: Series[bool]
    score: Series[float]
    rank: Series[pd.Int64Dtype] = pa.Field(ge=1, unique=True)

    class Config:
        strict = "filter"
        coerce = True


class HeroResultSchema(pa.DataFrameModel):
    """Long-format hero-vs-bots eval results — one row per (cell, scoring).

    A cell is one (season, strategy, seat, seed) sole-hero-vs-bots league; each cell
    contributes two rows (`scoring` in {"actual", "projected"}) carrying the hero seat's
    season result. `strategy` is a real strategy key (never "bot" — the bot baseline is
    structural, computed at report time, not stored).
    """

    season: Series[int] = pa.Field(ge=1999, le=2100)
    # Aliased: the column is "strategy"; the attribute can't be (DataFrameModel.strategy
    # is a reserved classmethod).
    strategy_key: Series[str] = pa.Field(alias="strategy")
    seat: Series[int] = pa.Field(ge=1)
    seed: Series[int] = pa.Field(ge=0)
    scoring: Series[str] = pa.Field(isin=("actual", "projected"))
    wins: Series[int] = pa.Field(ge=0)
    losses: Series[int] = pa.Field(ge=0)
    made_playoffs: Series[bool]
    is_champion: Series[bool]
    points_for: Series[float] = pa.Field(ge=0)

    class Config:
        strict = "filter"
        coerce = True


class PreseasonFeaturesSchema(pa.DataFrameModel):
    """One row per (gsis_id, season) for every player on depth_charts_{season}
    with position in {QB, RB, WR, TE}. Inputs to PreseasonModel.predict_season_distribution.

    `prior_{N}_season_per_game_<stat>` columns exist per modeled stat for the
    player's position. They are nullable — a player missing a prior season
    (rookies, injuries) has NaN there. Position-specific stat sets:
        QB: passing_yards, passing_tds, passing_interceptions,
            rushing_yards, rushing_tds.
        RB: rushing_yards, rushing_tds, receptions, receiving_yards,
            receiving_tds.
        WR: receptions, receiving_yards, receiving_tds, rushing_yards,
            rushing_tds.
        TE: receptions, receiving_yards, receiving_tds.
    """

    # Identity
    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=2018, le=2100)
    position: Series[str] = pa.Field(isin=_SKILL_POSITION_VALUES)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    depth_chart_rank: Series[int] = pa.Field(ge=1, le=10)

    # Player profile
    age: Series[float] = pa.Field(ge=18.0, le=50.0, nullable=True)
    years_exp: Series[int] = pa.Field(ge=0, le=30)
    is_rookie: Series[bool]
    draft_round: Series[int] = pa.Field(ge=1, le=7, nullable=True)
    draft_pick_overall: Series[int] = pa.Field(ge=1, le=400, nullable=True)

    # Prior 1/2/3 season per-game aggregates — all nullable + Optional.
    # Pandera schemas can't declare per-position columns cleanly, so we declare
    # the UNION of stats here and rely on `strict="filter"` to drop columns not
    # populated for a given position. `Optional[Series[...]]` marks each prior_*
    # column as not-required at validate time so a QB frame (which has no
    # `prior_*_receiving_*` columns) and a WR frame (which has no
    # `prior_*_passing_*` columns) both validate against the same schema.
    # Population is the builder's job.
    prior_1_season_games_played: Series[int] | None = pa.Field(ge=0, le=17, nullable=True)
    prior_2_season_games_played: Series[int] | None = pa.Field(ge=0, le=17, nullable=True)
    prior_3_season_games_played: Series[int] | None = pa.Field(ge=0, le=17, nullable=True)

    prior_1_season_per_game_passing_yards: Series[float] | None = pa.Field(
        ge=-10, le=500, nullable=True
    )
    prior_2_season_per_game_passing_yards: Series[float] | None = pa.Field(
        ge=-10, le=500, nullable=True
    )
    prior_3_season_per_game_passing_yards: Series[float] | None = pa.Field(
        ge=-10, le=500, nullable=True
    )
    prior_1_season_per_game_passing_tds: Series[float] | None = pa.Field(ge=0, le=10, nullable=True)
    prior_2_season_per_game_passing_tds: Series[float] | None = pa.Field(ge=0, le=10, nullable=True)
    prior_3_season_per_game_passing_tds: Series[float] | None = pa.Field(ge=0, le=10, nullable=True)
    prior_1_season_per_game_passing_interceptions: Series[float] | None = pa.Field(
        ge=0, le=10, nullable=True
    )
    prior_2_season_per_game_passing_interceptions: Series[float] | None = pa.Field(
        ge=0, le=10, nullable=True
    )
    prior_3_season_per_game_passing_interceptions: Series[float] | None = pa.Field(
        ge=0, le=10, nullable=True
    )
    prior_1_season_per_game_rushing_yards: Series[float] | None = pa.Field(
        ge=-5, le=250, nullable=True
    )
    prior_2_season_per_game_rushing_yards: Series[float] | None = pa.Field(
        ge=-5, le=250, nullable=True
    )
    prior_3_season_per_game_rushing_yards: Series[float] | None = pa.Field(
        ge=-5, le=250, nullable=True
    )
    prior_1_season_per_game_rushing_tds: Series[float] | None = pa.Field(ge=0, le=5, nullable=True)
    prior_2_season_per_game_rushing_tds: Series[float] | None = pa.Field(ge=0, le=5, nullable=True)
    prior_3_season_per_game_rushing_tds: Series[float] | None = pa.Field(ge=0, le=5, nullable=True)
    prior_1_season_per_game_receptions: Series[float] | None = pa.Field(ge=0, le=20, nullable=True)
    prior_2_season_per_game_receptions: Series[float] | None = pa.Field(ge=0, le=20, nullable=True)
    prior_3_season_per_game_receptions: Series[float] | None = pa.Field(ge=0, le=20, nullable=True)
    prior_1_season_per_game_receiving_yards: Series[float] | None = pa.Field(
        ge=-5, le=300, nullable=True
    )
    prior_2_season_per_game_receiving_yards: Series[float] | None = pa.Field(
        ge=-5, le=300, nullable=True
    )
    prior_3_season_per_game_receiving_yards: Series[float] | None = pa.Field(
        ge=-5, le=300, nullable=True
    )
    prior_1_season_per_game_receiving_tds: Series[float] | None = pa.Field(
        ge=0, le=5, nullable=True
    )
    prior_2_season_per_game_receiving_tds: Series[float] | None = pa.Field(
        ge=0, le=5, nullable=True
    )
    prior_3_season_per_game_receiving_tds: Series[float] | None = pa.Field(
        ge=0, le=5, nullable=True
    )

    class Config:
        strict = "filter"
        # Coerce so int32/float32 inputs (the builder emits compact dtypes for
        # season + age + per-game floats) are upcast to int64/float64 to match
        # `Series[int]` / `Series[float]` rather than rejected with a dtype
        # error. See WrFeaturesSchema.Config for the same pattern.
        coerce = True


class PreseasonProjectionSchema(pa.DataFrameModel):
    """One row per (gsis_id, season, ruleset) — the v1 preseason output.

    Per-stat season-total quartets `<stat>_season_total_{mean,p10,p50,p90}` are
    populated per the player's position's stat set:
        QB: passing_yards, passing_tds, passing_interceptions,
            rushing_yards, rushing_tds.
        RB: rushing_yards, rushing_tds, receptions, receiving_yards,
            receiving_tds.
        WR: receptions, receiving_yards, receiving_tds, rushing_yards,
            rushing_tds.
        TE: receptions, receiving_yards, receiving_tds.
    Columns not modeled for a position are absent; strict="filter" + Series[T] | None
    on per-stat fields lets per-position frames validate cleanly.
    """

    # Identity
    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=2018, le=2100)
    position: Series[str] = pa.Field(isin=_SKILL_POSITION_VALUES)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    ruleset: Series[str] = pa.Field(isin=_RULESET_NAME_VALUES)
    model_id: Series[str]

    # Scored fpts — required for every row.
    season_total_fpts_mean: Series[float] = pa.Field(ge=0, le=700)
    season_total_fpts_p10: Series[float] = pa.Field(ge=0, le=700)
    season_total_fpts_p50: Series[float] = pa.Field(ge=0, le=700)
    season_total_fpts_p90: Series[float] = pa.Field(ge=0, le=700)

    # Per-stat season totals — UNION of stats across positions; Series[T] | None
    # so per-position frames missing the column validate. strict="filter" drops
    # extras at validate time.
    passing_yards_season_total_mean: Series[float] | None = pa.Field(ge=0, le=7000, nullable=True)
    passing_yards_season_total_p10: Series[float] | None = pa.Field(ge=0, le=7000, nullable=True)
    passing_yards_season_total_p50: Series[float] | None = pa.Field(ge=0, le=7000, nullable=True)
    passing_yards_season_total_p90: Series[float] | None = pa.Field(ge=0, le=7000, nullable=True)
    passing_tds_season_total_mean: Series[float] | None = pa.Field(ge=0, le=80, nullable=True)
    passing_tds_season_total_p10: Series[float] | None = pa.Field(ge=0, le=80, nullable=True)
    passing_tds_season_total_p50: Series[float] | None = pa.Field(ge=0, le=80, nullable=True)
    passing_tds_season_total_p90: Series[float] | None = pa.Field(ge=0, le=80, nullable=True)
    passing_interceptions_season_total_mean: Series[float] | None = pa.Field(
        ge=0, le=40, nullable=True
    )
    passing_interceptions_season_total_p10: Series[float] | None = pa.Field(
        ge=0, le=40, nullable=True
    )
    passing_interceptions_season_total_p50: Series[float] | None = pa.Field(
        ge=0, le=40, nullable=True
    )
    passing_interceptions_season_total_p90: Series[float] | None = pa.Field(
        ge=0, le=40, nullable=True
    )
    rushing_yards_season_total_mean: Series[float] | None = pa.Field(ge=0, le=3000, nullable=True)
    rushing_yards_season_total_p10: Series[float] | None = pa.Field(ge=0, le=3000, nullable=True)
    rushing_yards_season_total_p50: Series[float] | None = pa.Field(ge=0, le=3000, nullable=True)
    rushing_yards_season_total_p90: Series[float] | None = pa.Field(ge=0, le=3000, nullable=True)
    rushing_tds_season_total_mean: Series[float] | None = pa.Field(ge=0, le=40, nullable=True)
    rushing_tds_season_total_p10: Series[float] | None = pa.Field(ge=0, le=40, nullable=True)
    rushing_tds_season_total_p50: Series[float] | None = pa.Field(ge=0, le=40, nullable=True)
    rushing_tds_season_total_p90: Series[float] | None = pa.Field(ge=0, le=40, nullable=True)
    receptions_season_total_mean: Series[float] | None = pa.Field(ge=0, le=200, nullable=True)
    receptions_season_total_p10: Series[float] | None = pa.Field(ge=0, le=200, nullable=True)
    receptions_season_total_p50: Series[float] | None = pa.Field(ge=0, le=200, nullable=True)
    receptions_season_total_p90: Series[float] | None = pa.Field(ge=0, le=200, nullable=True)
    receiving_yards_season_total_mean: Series[float] | None = pa.Field(ge=0, le=3000, nullable=True)
    receiving_yards_season_total_p10: Series[float] | None = pa.Field(ge=0, le=3000, nullable=True)
    receiving_yards_season_total_p50: Series[float] | None = pa.Field(ge=0, le=3000, nullable=True)
    receiving_yards_season_total_p90: Series[float] | None = pa.Field(ge=0, le=3000, nullable=True)
    receiving_tds_season_total_mean: Series[float] | None = pa.Field(ge=0, le=40, nullable=True)
    receiving_tds_season_total_p10: Series[float] | None = pa.Field(ge=0, le=40, nullable=True)
    receiving_tds_season_total_p50: Series[float] | None = pa.Field(ge=0, le=40, nullable=True)
    receiving_tds_season_total_p90: Series[float] | None = pa.Field(ge=0, le=40, nullable=True)

    class Config:
        strict = "filter"
        coerce = True


class PreseasonBacktestSchema(pa.DataFrameModel):
    """One row per (target_season, position, model_class) — output of the v1
    preseason backtest harness. See spec §7."""

    target_season: Series[int] = pa.Field(ge=2018, le=2100)
    position: Series[str] = pa.Field(isin=_SKILL_POSITION_VALUES)
    model_class: Series[str]
    ruleset: Series[str] = pa.Field(isin=_RULESET_NAME_VALUES)
    rmse: Series[float] = pa.Field(ge=0)
    rmse_naive_baseline: Series[float] = pa.Field(ge=0)
    rmse_delta_pct: Series[float]  # signed; can be negative (model beats naive)
    spearman_top50: Series[float] = pa.Field(ge=-1, le=1)
    n_players: Series[int] = pa.Field(ge=0)
    coverage_diff_projected_not_played: Series[int] = pa.Field(ge=0)
    coverage_diff_played_not_projected: Series[int] = pa.Field(ge=0)
    verdict: Series[str] = pa.Field(isin=_BACKTEST_VERDICT_VALUES)

    class Config:
        strict = "filter"
        coerce = True


class WeeklyProjectionSchema(pa.DataFrameModel):
    """Per-(player, week) preseason-source weekly projection, scored to a ruleset.

    `projected_points` is nullable: a player with no ESPN weekly entry that week
    (bye / inactive) carries NULL and cannot be started.
    """

    gsis_id: Series[str]
    season: Series[pd.Int64Dtype] = pa.Field(ge=2000, le=2100)
    week: Series[pd.Int64Dtype] = pa.Field(ge=1, le=17)
    position: Series[str]
    projected_points: Series[pd.Float64Dtype] = pa.Field(nullable=True)

    class Config:
        strict = "filter"
        coerce = True


class WeeklyActualSchema(pa.DataFrameModel):
    """Per-(player, week) realized fantasy points under a ruleset."""

    gsis_id: Series[str]
    season: Series[pd.Int64Dtype] = pa.Field(ge=2000, le=2100)
    week: Series[pd.Int64Dtype] = pa.Field(ge=1, le=17)
    actual_points: Series[pd.Float64Dtype]

    class Config:
        strict = "filter"
        coerce = True
