"""Two valuations of the same player, and the gap between them.

A trade needs **two** numbers per player, not one:

- `market` — ESPN's own stat-line projection. What the other manager sees when they open the
  app, so it is what a proposal has to look fair against.
- `ours` — the pool's `season_mean_fpts` (the ESPN + Sleeper consensus this repo builds).

`edge = ours - market`. **Positive means we like him more than the market does — a player to
acquire. Negative means a player to send.**

**Both are scored through the same `Ruleset`.** ESPN publishes a stat line, not points, and the
points it shows in its own app are under ESPN's default scoring, not this league's. Comparing a
half-PPR figure to a full-PPR one is a units error that looks exactly like an edge, and it would
be an edge on every pass-catcher at once — plausible, systematic, and wrong. `edge_zero` is
pinned by a test.

**The honest limit, which the CLI prints rather than buries:** our consensus *contains* ESPN, so
the two are correlated and `edge` is really "how far Sleeper pulls the blend away from ESPN",
not an independent second opinion. It is a real signal about where the market disagrees with
itself. It is not evidence that we are right.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from projections.midseason.injuries import season_multiplier
from projections.schemas import InjuryStatus, Ruleset, parse_injury_status
from projections.scoring.score import expected_points

#: ESPN stat-line columns in `external_projections`, mapped to `StatLine` field names. ESPN
#: reports no 2pt conversions or return TDs; absent fields score as 0.0 in `_score_fields`.
_ESPN_STAT_COLUMNS: dict[str, str] = {
    "passing_yards": "passing_yards",
    "passing_tds": "passing_tds",
    "interceptions": "interceptions",
    "rushing_yards": "rushing_yards",
    "rushing_tds": "rushing_tds",
    "receptions": "receptions",
    "receiving_yards": "receiving_yards",
    "receiving_tds": "receiving_tds",
    "fumbles_lost": "fumbles_lost",
}


@dataclass(frozen=True)
class PlayerValue:
    """One player, valued twice, with the injury haircut already applied to both."""

    gsis_id: str
    #: ESPN's own player id. The join key back into the league payload, which is how a trade
    #: moves a roster entry between two teams without synthesising one.
    espn_id: int
    full_name: str
    position: str
    market: float
    ours: float
    injury_status: InjuryStatus
    #: ESPN's own text, kept so an unrecognised designation can be shown rather than silently
    #: treated as healthy -- see `parse_injury_status`.
    injury_raw: str

    @property
    def edge(self) -> float:
        """Ours minus the market. Positive = acquire, negative = send."""
        return self.ours - self.market


def espn_season_points(
    external: pd.DataFrame, ruleset: Ruleset, *, source: str = "ESPN"
) -> dict[str, float]:
    """`gsis_id -> ESPN's projected season points`, scored under `ruleset`.

    `external` is one `asof` partition of `external_projections`. Rows from other sources are
    dropped: Sleeper supplies ADP only and carries a null stat line, which would score as a
    confident 0.0 rather than as "no opinion".
    """
    rows = external[external["source"].astype(str).str.upper() == source.upper()]
    out: dict[str, float] = {}
    for row in rows.itertuples():
        line = {
            field: float(getattr(row, column))
            for field, column in _ESPN_STAT_COLUMNS.items()
            if pd.notna(getattr(row, column, None))
        }
        if not line:
            # A row with no stat line at all is an absence of an opinion, not a projection of
            # nothing. Omitted so callers can tell the two apart.
            continue
        out[str(row.gsis_id)] = expected_points(line, ruleset)
    return out


def build_values(
    rosters: pd.DataFrame,
    gsis_by_row: pd.Series,
    pool: pd.DataFrame,
    external: pd.DataFrame,
    ruleset: Ruleset,
    *,
    games_remaining: int,
) -> dict[str, PlayerValue]:
    """Value every rostered player twice. Keyed by gsis; players we cannot value are omitted.

    `gsis_by_row` is the resolved gsis per roster row (from `espn_to_gsis`), passed in rather
    than recomputed so the trade tool and the standings page cannot resolve a player
    differently.

    **The injury haircut is applied to BOTH valuations, with the same multiplier.** A trade
    proposal is the easiest place in the whole repo to be defrauded by a stale injury tag, in
    both directions: valuing a suspended player at his healthy projection makes him a phantom
    asset, and applying the haircut to only one side manufactures an edge out of arithmetic.
    """
    market = espn_season_points(external, ruleset)
    ours = dict(zip(pool["gsis_id"].astype(str), pool["season_mean_fpts"], strict=True))
    names = dict(zip(pool["gsis_id"].astype(str), pool["full_name"].astype(str), strict=True))

    values: dict[str, PlayerValue] = {}
    for row, gsis in zip(rosters.itertuples(), gsis_by_row, strict=True):
        if pd.isna(gsis):
            continue
        key = str(gsis)
        if key not in ours or key not in market:
            # Kickers, defenses, and anyone only one source has an opinion about. A player
            # valued by one side only cannot be traded on a like-for-like comparison.
            continue
        status, raw = parse_injury_status(getattr(row, "injury_status_raw", None))
        haircut = season_multiplier(status, games_remaining=games_remaining)
        values[key] = PlayerValue(
            gsis_id=key,
            espn_id=int(row.player_id),
            full_name=names.get(key, str(getattr(row, "player", key))),
            position=str(row.pos),
            market=market[key] * haircut,
            ours=ours[key] * haircut,
            injury_status=status,
            injury_raw=raw,
        )
    return values


__all__ = ["PlayerValue", "build_values", "espn_season_points"]
