"""Expected season points under per-player availability (spec §3.4).

Monte-Carlo a season: each week, players are available (not on bye, healthy w.p.
`p`), and the best legal lineup is filled from the available roster. Because
`per_game = season_mean_fpts / 17` is a uniform scaling, the weekly optimal lineup
is `optimal_lineup_points(available_subset) / 17`. The per-sim fill is vectorized
across all draws at once (`_vectorized_lineup_points`) — equivalent to looping
`optimal_lineup_points` per sim (pinned by an equivalence test) but orders of
magnitude faster, which is what makes the season-value strategy tractable in a
tournament. Weeks with no roster bye are identical in expectation, so we MC one
generic week and reuse it (the factorization is exact in expectation).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.roster_score import _FLEX_SLOTS
from projections.draft.roster_eligibility import POSITION_SLOTS
from projections.schemas import RosterSlot

# Healthy-season denominator: a full season projection divided into per-game points
# (uniform scaling, spec §3.3). Distinct from availability._sched_games, which is the
# era-correct *historical* schedule length used to estimate injury rates.
_GAMES = 17


@dataclass(frozen=True)
class _FillMeta:
    """Roster fill plan precomputed once, reused across every MC draw of a roster.

    `single`: per single-position starting slot, the roster columns at that position
    sorted by points descending, those points, and the slot count. `flex`: per flex
    tier (FLEX then SUPER_FLEX, narrowest first), the flex-eligible roster columns and
    the slot count. `pts` is every roster player's points in roster-row order.

    The arrays are read-only by contract — `_vectorized_lineup_points` never mutates
    them (it works on `avail.copy()`), so one `_FillMeta` is safely reused across draws.
    """

    pts: np.ndarray
    single: tuple[tuple[np.ndarray, np.ndarray, int], ...]
    flex: tuple[tuple[np.ndarray, int], ...]


def _roster_fill_meta(roster: pd.DataFrame, roster_slots: Mapping[RosterSlot, int]) -> _FillMeta:
    """Precompute the restrictive-first fill plan for `roster` under `roster_slots`."""
    pos = roster["position"].astype(str).to_numpy()
    pts = roster["season_mean_fpts"].to_numpy(dtype=np.float64)
    single: list[tuple[np.ndarray, np.ndarray, int]] = []
    for slot in POSITION_SLOTS:
        count = roster_slots.get(slot, 0)
        if count <= 0:
            continue
        cols = np.flatnonzero(pos == slot.value)
        if cols.size == 0:
            continue
        order = np.argsort(-pts[cols], kind="stable")  # points descending
        sorted_cols = cols[order]
        single.append((sorted_cols, pts[sorted_cols], count))
    flex: list[tuple[np.ndarray, int]] = []
    for slot, eligible in _FLEX_SLOTS:  # same flex tiers/order as optimal_lineup_points
        count = roster_slots.get(slot, 0)
        if count <= 0:
            continue
        elig_values = [p.value for p in eligible]
        cols = np.flatnonzero(np.isin(pos, elig_values))
        if cols.size == 0:
            continue
        flex.append((cols, count))
    return _FillMeta(pts=pts, single=tuple(single), flex=tuple(flex))


def _vectorized_lineup_points(avail: np.ndarray, meta: _FillMeta) -> np.ndarray:
    """Optimal starting-lineup points for each row of `avail` ((n_sims, n) booleans).

    Vectorized equivalent of `optimal_lineup_points` over many availability draws of one
    fixed roster: restrictive-first greedy (single-position slots, then FLEX, then
    SUPER_FLEX — narrowest eligibility first, which is optimal for laminar slots). Equal
    to the per-sim `optimal_lineup_points` sum up to float summation order; pinned by
    `test_vectorized_lineup_matches_optimal_lineup_points`.
    """
    n_sims = avail.shape[0]
    total = np.zeros(n_sims, dtype=np.float64)
    flex_avail = avail.copy()
    for sorted_cols, sorted_pts, count in meta.single:
        a = avail[:, sorted_cols]  # (n_sims, m), points-descending
        rank = np.cumsum(a, axis=1)  # availability rank among players at this position
        used = a & (rank <= count)  # the top-`count` available fill the starting slots
        total += (used * sorted_pts).sum(axis=1)
        flex_avail[:, sorted_cols] &= ~used  # a started player can't also fill a flex slot
    for cols, count in meta.flex:
        pts_cols = meta.pts[cols]
        for _ in range(count):
            cand = np.where(flex_avail[:, cols], pts_cols, -np.inf)  # (n_sims, |cols|)
            best = cand.max(axis=1)
            has = best > -np.inf  # a sim may have no eligible flex player → contributes 0
            total += np.where(has, best, 0.0)
            chosen = cols[cand.argmax(axis=1)]  # roster column per sim (valid where has)
            rows = np.flatnonzero(has)
            flex_avail[rows, chosen[rows]] = False  # consume the chosen player
    return total


def _factorized_season_value(
    roster: pd.DataFrame,
    availability: PlayerAvailability,
    weeks: Iterable[int],
    week_value_fn: Callable[[np.ndarray], float],
) -> float:
    """Sum the season via the single-week factorization (spec §3.4 of PR #60).

    `week_value_fn(forced_out)` takes a boolean mask over roster rows (True where
    the player is on bye that week) and returns E[week points | those players are
    forced out]. Every non-bye week shares one expectation; each distinct roster
    bye week is recomputed with that player forced out. Exact in expectation. Call
    order (clean week, then bye weeks ascending) is fixed so callers that advance a
    shared RNG inside week_value_fn stay reproducible.
    """
    n = len(roster)
    gsis = roster["gsis_id"].astype(str).to_numpy()
    # -1 sentinel = "no bye"; never a real week, so it drops out of roster_bye_weeks below.
    bye_arr = np.array([b if (b := availability.bye_week(g)) is not None else -1 for g in gsis])
    weeks = list(weeks)
    roster_bye_weeks = sorted({w for w in bye_arr.tolist() if w in weeks})

    clean = week_value_fn(np.zeros(n, dtype=bool))
    total = (len(weeks) - len(roster_bye_weeks)) * clean
    for w in roster_bye_weeks:
        total += week_value_fn(bye_arr == w)
    return total


def expected_season_points_crn(
    roster: pd.DataFrame,
    roster_slots: Mapping[RosterSlot, int],
    availability: PlayerAvailability,
    *,
    draws: np.ndarray,
    col_of: Mapping[str, int],
    weeks: Iterable[int] = range(1, 18),
) -> float:
    """Expected season points using a shared pre-drawn availability matrix (CRN).

    `draws` is `(n_sims, universe)` uniforms; `col_of` maps gsis_id -> column.
    Every roster scored against the same `draws` shares per-player draws, so a
    marginal `V(R+c) - V(R)` cancels the common noise (spec §3.3).
    """
    n = len(roster)
    if n == 0:
        return 0.0
    gsis = roster["gsis_id"].astype(str).to_numpy()
    p_arr = np.array([availability.p_week(g) for g in gsis], dtype=np.float64)
    cols = np.array([col_of[g] for g in gsis])
    sub_draws = draws[:, cols]  # (n_sims, n), aligned to roster row order
    meta = _roster_fill_meta(roster, roster_slots)

    def week_value_fn(forced_out: np.ndarray) -> float:
        avail = (sub_draws < p_arr) & ~forced_out  # (n_sims, n)
        return float(_vectorized_lineup_points(avail, meta).mean()) / _GAMES

    return _factorized_season_value(roster, availability, weeks, week_value_fn)


def expected_season_points(
    roster: pd.DataFrame,
    roster_slots: Mapping[RosterSlot, int],
    availability: PlayerAvailability,
    *,
    n_sims: int,
    rng: np.random.Generator,
    weeks: Iterable[int] = range(1, 18),
) -> float:
    """Expected total season points of `roster` under availability risk."""
    n = len(roster)
    if n == 0:
        return 0.0
    gsis = roster["gsis_id"].astype(str).to_numpy()
    p_arr = np.array([availability.p_week(g) for g in gsis], dtype=np.float64)
    meta = _roster_fill_meta(roster, roster_slots)

    def week_value_fn(forced_out: np.ndarray) -> float:
        avail = (rng.random((n_sims, n)) < p_arr) & ~forced_out  # (n_sims, n)
        return float(_vectorized_lineup_points(avail, meta).mean()) / _GAMES

    return _factorized_season_value(roster, availability, weeks, week_value_fn)


def marginal_season_values(
    base_roster: pd.DataFrame,
    candidates: pd.DataFrame,
    roster_slots: Mapping[RosterSlot, int],
    availability: PlayerAvailability,
    *,
    n_sims: int,
    rng: np.random.Generator,
    weeks: Iterable[int] = range(1, 18),
) -> dict[str, float]:
    """CRN marginal expected-season-points of adding each candidate to `base_roster`.

    Returns {candidate gsis_id: V(base + candidate) - V(base)}. All evaluations
    (base and every candidate) share one pre-drawn availability matrix over the
    union of base + candidate ids, so the marginal isolates the candidate's own
    contribution at low variance (spec §3.3). `base_roster` and `candidates` each
    carry `gsis_id`, `position`, `season_mean_fpts`.
    """
    base_ids = [str(g) for g in base_roster["gsis_id"]]
    cand_ids = [str(g) for g in candidates["gsis_id"]]
    universe = sorted(set(base_ids) | set(cand_ids))
    col_of = {g: i for i, g in enumerate(universe)}
    draws = rng.random((n_sims, len(universe)))

    base_val = expected_season_points_crn(
        base_roster, roster_slots, availability, draws=draws, col_of=col_of, weeks=weeks
    )
    out: dict[str, float] = {}
    for i, cand_id in enumerate(cand_ids):
        cand_roster = pd.concat([base_roster, candidates.iloc[[i]]], ignore_index=True)
        val = expected_season_points_crn(
            cand_roster, roster_slots, availability, draws=draws, col_of=col_of, weeks=weeks
        )
        out[cand_id] = val - base_val
    return out
