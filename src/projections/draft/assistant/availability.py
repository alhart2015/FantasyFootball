"""Per-player season availability: injury Bernoulli `p` + bye week (spec §3.2).

`p` is the fraction of its team's games a player plays over their ACTIVE span,
era-normalized over their `weekly_stats` history (16-game 2018-2020 vs 17-game
2021+), with a per-position default for rookies / no-history and a clamp to keep
no player degenerate. Measuring from each season's first appearance keeps a
mid-season debut from looking injury-prone. Byes come from the regular-season
target schedule via the player's `id_map` team.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd

from projections.season_calendar import last_regular_week, regular_season_games


def _team_byes(schedules: pd.DataFrame, season: int) -> dict[str, int]:
    """Map team -> bye week for `season`: the single week the team has no game row.

    Empty (with a warning) if the target-season partition has no rows — graceful
    degradation so the injury model still applies without byes.
    """
    sch = schedules[schedules["season"] == season]
    if len(sch) == 0:
        warnings.warn(f"no schedules for season {season}; byes will be empty", stacklevel=2)
        return {}
    # A bye is a regular-season concept (spec 3.1: weeks 1..18). Schedule partitions
    # carry playoff weeks (19-22); without this filter a team that missed the playoffs
    # "misses" every playoff week too, so the single-gap rule below never fires and no
    # bye is ever detected.
    sch = sch[sch["week"] <= last_regular_week(season)]
    weeks = sorted(int(w) for w in sch["week"].unique())
    teams = pd.unique(pd.concat([sch["home_team"], sch["away_team"]], ignore_index=True))
    byes: dict[str, int] = {}
    for team in teams:
        played = {
            int(w) for w in sch.loc[(sch["home_team"] == team) | (sch["away_team"] == team), "week"]
        }
        missing = [w for w in weeks if w not in played]
        if len(missing) == 1:
            byes[str(team)] = missing[0]
    return byes


@dataclass(frozen=True)
class PlayerAvailability:
    """Resolved availability for the draftable pool. `p` covers every pool player."""

    p: dict[str, float]
    bye: dict[str, int]

    def p_week(self, gsis_id: str) -> float:
        """Per-week probability the player is healthy/active (injury + benching)."""
        return self.p[gsis_id]

    def bye_week(self, gsis_id: str) -> int | None:
        """The week the player is forced out (team bye), or None."""
        return self.bye.get(gsis_id)


def build_availability(
    weekly_stats: pd.DataFrame,
    schedules: pd.DataFrame,
    id_map: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    season: int,
    lo: float = 0.4,
    hi: float = 0.97,
) -> PlayerAvailability:
    """Build per-player availability for every player in `pool` (spec §3.2)."""
    ws = weekly_stats[["gsis_id", "season", "week", "position"]].copy()
    ws["position"] = ws["position"].astype(str)
    ws["gsis_id"] = ws["gsis_id"].astype(str)
    # Drop playoff weeks (same cutoff as the bye filter) so they don't inflate
    # games played or set first_week from a playoff-only season.
    ws = ws[ws["week"] <= ws["season"].map(last_regular_week)]

    games = (
        ws.groupby(["gsis_id", "season"])
        .agg(games=("week", "count"), first_week=("week", "min"))
        .reset_index()
    )
    # Availability is measured over the player's ACTIVE span, not the whole season:
    # weeks before a player's first appearance ("not yet active" -- rookie debut,
    # mid-season signing or trade, call-up) are not missed games, so counting them
    # would conflate roster status with injury risk (spec 3.2: p is "healthy/active").
    # Denominator = scheduled games from the first week through season end
    # (sched - first_week + 1); clip keeps frac in (0, 1] (the span ignores a
    # possible pre-debut bye, which can over-trim).
    sched = games["season"].map(regular_season_games)
    active = (sched - games["first_week"] + 1).clip(lower=1)
    games["frac"] = (games["games"] / active).clip(upper=1.0)
    p_raw = games.groupby("gsis_id")["frac"].mean()
    pos_hist = ws.groupby("gsis_id")["position"].agg(lambda s: str(s.mode().iloc[0]))

    hist = pd.DataFrame({"p": p_raw, "position": pos_hist})
    default_by_pos = {str(k): float(v) for k, v in hist.groupby("position")["p"].mean().items()}
    overall_default = float(hist["p"].mean()) if len(hist) else (lo + hi) / 2

    p: dict[str, float] = {}
    for gid, pos in zip(pool["gsis_id"].astype(str), pool["position"].astype(str), strict=True):
        raw = (
            float(p_raw.loc[gid])
            if gid in p_raw.index
            # position unseen in history falls back to overall_default (the all-players mean)
            else default_by_pos.get(pos, overall_default)
        )
        p[gid] = min(max(raw, lo), hi)

    team_of = dict(zip(id_map["gsis_id"].astype(str), id_map["team"].astype(str), strict=False))
    team_byes = _team_byes(schedules, season)
    bye: dict[str, int] = {}
    for gid in p:
        team = team_of.get(gid)
        if team is not None and team in team_byes:
            bye[gid] = team_byes[team]

    return PlayerAvailability(p=p, bye=bye)
