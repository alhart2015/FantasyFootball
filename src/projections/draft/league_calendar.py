"""LeagueCalendar — how many regular-season weeks a league plays and how its bracket runs.

Split out from `league_projection`, where these lived as module constants (`REG_WEEKS = 1..13`,
`PLAYOFF_SIZE = 6`, a two-week final at weeks 16-17). Constants are fine for a single league and
wrong for any other: ESPN reports `matchupPeriodCount` and `playoffMatchupPeriodLength` per
league, and the Critts 2026 league plays **14** regular weeks with **one-week** playoff rounds
against the module's 13 and two. That mismatch is a rounding error for draft prep — a two-week
final favours the better team, so it nudges championship odds up — but it is disqualifying for
in-season projected standings, which must lock already-played weeks against the league's real
week numbering.

The defaults reproduce the old constants exactly, so existing callers keep their behaviour
until they pass a calendar.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: The historical `league_projection` constants, kept as the default so that every caller
#: that has not yet been given a real calendar behaves exactly as it did before.
DEFAULT_REG_WEEKS = 13
DEFAULT_PLAYOFF_SIZE = 6
DEFAULT_N_BYES = 2
DEFAULT_FINAL_WEEKS = 2


class LeagueCalendar(BaseModel):
    """Regular-season length and playoff bracket shape for one league.

    The bracket this models is the common one and the only one the simulator implements: a
    single-elimination ladder where the top `n_byes` seeds skip the first round. With the
    default `playoff_size=6, n_byes=2` that is wildcard (3v6, 4v5) -> reseeded semis -> final.
    """

    model_config = ConfigDict(frozen=True)

    reg_weeks: int = Field(default=DEFAULT_REG_WEEKS, gt=0)
    playoff_size: int = Field(default=DEFAULT_PLAYOFF_SIZE, gt=1)
    n_byes: int = Field(default=DEFAULT_N_BYES, ge=0)
    #: Weeks the championship match spans. ESPN's `playoffMatchupPeriodLength`. A two-week
    #: final sums both weeks, which lowers variance and so favours the stronger team — this is
    #: a real difference in title odds, not a presentation detail.
    final_weeks: int = Field(default=DEFAULT_FINAL_WEEKS, gt=0)

    @model_validator(mode="after")
    def _bracket_is_coherent(self) -> LeagueCalendar:
        if self.n_byes >= self.playoff_size:
            raise ValueError(
                f"n_byes ({self.n_byes}) must be fewer than playoff_size ({self.playoff_size}); "
                "every playoff team having a bye leaves no first round to play."
            )
        # The simulator pairs the non-bye seeds off against each other in round one, so there
        # has to be an even number of them. An odd count would leave one team unpaired and
        # silently advance or drop it depending on the zip.
        if (self.playoff_size - self.n_byes) % 2 != 0:
            raise ValueError(
                f"playoff_size - n_byes must be even (got {self.playoff_size} - {self.n_byes} "
                f"= {self.playoff_size - self.n_byes}); the first round pairs those seeds off."
            )
        return self

    @property
    def n_playoff_rounds(self) -> int:
        """Rounds to get from `playoff_size` teams to one, byes skipping the first.

        Wildcard halves the non-bye field, then the survivors plus the byes halve each round.
        """
        rounds = 0
        remaining = self.playoff_size
        if self.n_byes:
            remaining = self.n_byes + (self.playoff_size - self.n_byes) // 2
            rounds = 1
        while remaining > 1:
            remaining = (remaining + 1) // 2
            rounds += 1
        return rounds

    @property
    def wildcard_week(self) -> int:
        """First playoff week. Equals `reg_weeks + 1` whether or not byes exist."""
        return self.reg_weeks + 1

    @property
    def championship_weeks(self) -> tuple[int, ...]:
        """The week(s) the final spans, in order."""
        start = self.reg_weeks + self.n_playoff_rounds
        return tuple(range(start, start + self.final_weeks))

    @property
    def total_weeks(self) -> int:
        """Every week the simulator must draw points for, regular season through the final."""
        return self.championship_weeks[-1]

    @property
    def reg_week_numbers(self) -> tuple[int, ...]:
        return tuple(range(1, self.reg_weeks + 1))

    @property
    def all_week_numbers(self) -> tuple[int, ...]:
        return tuple(range(1, self.total_weeks + 1))

    def round_week(self, round_index: int) -> int:
        """Absolute week number of a 0-based playoff round before the final."""
        if not 0 <= round_index < self.n_playoff_rounds - 1:
            raise ValueError(
                f"round_index must be in [0, {self.n_playoff_rounds - 1}); got {round_index}"
            )
        return self.reg_weeks + 1 + round_index

    @classmethod
    def from_espn_settings(cls, schedule_settings: dict[str, object]) -> LeagueCalendar:
        """Build from ESPN's `settings.scheduleSettings`.

        Reads `matchupPeriodCount` (regular-season weeks), `playoffTeamCount`, and
        `playoffMatchupPeriodLength` (weeks per playoff round -> the final's length). ESPN does
        not report a bye count, so it is derived: seeds above the nearest power of two below
        `playoff_size` get one, which is what a 6-team/2-bye or 4-team/0-bye bracket means.
        Absent keys fall back to the defaults rather than raising, so a partial payload still
        yields a usable calendar.
        """

        def _int(key: str, default: int, *, minimum: int = 1) -> int:
            return usable_int(schedule_settings.get(key), minimum=minimum) or default

        # minimum=2: `playoff_size` is `gt=1`, so a reported 1 has to fall back rather than
        # reach the constructor.
        playoff_size = _int("playoffTeamCount", DEFAULT_PLAYOFF_SIZE, minimum=2)
        return cls(
            reg_weeks=_int("matchupPeriodCount", DEFAULT_REG_WEEKS),
            playoff_size=playoff_size,
            n_byes=_byes_for(playoff_size),
            final_weeks=_int("playoffMatchupPeriodLength", DEFAULT_FINAL_WEEKS),
        )


def usable_int(raw: object, *, minimum: int = 1) -> int | None:
    """A positive int from an ESPN settings value, or None if it cannot be one.

    Split out and made public because "did ESPN actually report this?" is a question two
    callers need and a type test cannot answer:

    - **`bool` is an `int` in Python.** `matchupPeriodCount: true` would otherwise become
      `reg_weeks=1` -- a one-week regular season, with every later week discarded from the
      locked record and playoff odds computed off a single game.
    - **Passing the type test is not the same as converting.** `int("full")` raises,
      `int(float("nan"))` raises, `int(float("inf"))` overflows, and `0` or a negative trips
      `LeagueCalendar`'s own `gt=0` -- any of which would abort a caller mid-write rather
      than letting it fall back.
    - **`minimum` because the fields do not share a bound.** `playoff_size` is `gt=1`, not
      `gt=0`, so `playoffTeamCount: 1` would clear a one-size-fits-all check and then raise
      from the constructor -- exactly the abort this function exists to prevent.

    Returning None rather than raising keeps the decision with the caller: `from_espn_settings`
    substitutes its default, while `write_league_snapshot` leaves its record unbounded and says
    so.
    """
    if isinstance(raw, bool) or raw is None:
        return None
    if not isinstance(raw, int | float | str):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if value >= minimum else None


def _byes_for(playoff_size: int) -> int:
    """Byes implied by a playoff field size: enough to make the first round a power of two.

    6 -> 2 (two byes, 3v6 and 4v5), 4 -> 0, 8 -> 0, 12 -> 4. A field that is already a power
    of two plays a full first round and needs none.
    """
    if playoff_size < 2:
        return 0
    largest_pow2 = 1 << (playoff_size.bit_length() - 1)
    if largest_pow2 == playoff_size:
        return 0
    return 2 * largest_pow2 - playoff_size
