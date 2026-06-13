"""Round-robin regular-season schedule (circle method) + single-elimination playoff bracket."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

Matchup = tuple[int, int]


def regular_season_schedule(
    *, n_teams: int, n_weeks: int, rng: np.random.Generator
) -> list[list[Matchup]]:
    if n_teams % 2 != 0:
        raise ValueError("n_teams must be even")
    teams = [int(t) for t in rng.permutation(np.arange(1, n_teams + 1))]  # random fixed seating
    fixed, rot = teams[0], teams[1:]
    weeks: list[list[Matchup]] = []
    for _ in range(n_weeks):
        circle = [fixed, *rot]
        half = n_teams // 2
        pairs = [(circle[i], circle[n_teams - 1 - i]) for i in range(half)]
        weeks.append([(int(a), int(b)) for a, b in pairs])
        rot = [rot[-1], *rot[:-1]]  # rotate
    return weeks


def _winner(a: int, b: int, pts: Mapping[int, float]) -> int:
    # higher points wins; deterministic tie-break on lower seat id
    return a if (pts[a], -a) >= (pts[b], -b) else b


def playoff_champion(
    seeds: Sequence[int],
    points: Mapping[int, Mapping[int, float]],
    *,
    playoff_weeks: tuple[int, int, int],
) -> int:
    """seeds: seat ids in seed order (index 0 = #1 seed ...). Top-2 bye, 6-team single-elim."""
    if len(seeds) != 6:
        raise ValueError("v1 playoff bracket expects exactly 6 seeds")
    w15, w16, w17 = playoff_weeks
    a = _winner(seeds[2], seeds[5], points[w15])  # #3 v #6
    b = _winner(seeds[3], seeds[4], points[w15])  # #4 v #5
    survivors = sorted([a, b], key=lambda s: seeds.index(s))  # better seed first
    semi1 = _winner(seeds[0], survivors[1], points[w16])  # #1 v lower survivor
    semi2 = _winner(seeds[1], survivors[0], points[w16])  # #2 v higher survivor
    return _winner(semi1, semi2, points[w17])
