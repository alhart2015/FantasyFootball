# Risk-Aware Roster Valuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the draft tournament's starters-only roster metric with an expected-season-points valuation that accounts for per-player availability (injury + byes), so bench depth and positional risk finally matter.

**Architecture:** Three new pure modules under `src/projections/draft/assistant/` — `availability` (per-player injury `p` + byes from history), `season_value` (a Monte-Carlo season valuer that reuses `optimal_lineup_points/17`), and `valuer` (a `RosterValuer` Protocol with `StartersValuer` + `SeasonValuer`). The tournament threads a valuer (default `StartersValuer` = today's behavior); the CLI gains `--valuer season`.

**Tech Stack:** Python 3.12, pandas (pyarrow string dtype), numpy (`default_rng` MC), pandera (`IdMapSchema`), pydantic (`LeagueConfig`), the existing `store.read_partition`, pytest, mypy strict, ruff.

**Source spec:** `docs/superpowers/specs/2026-06-11-risk-aware-roster-valuation-design.md`

---

## File Structure

**Create (engine):**
- `src/projections/draft/assistant/availability.py` — `PlayerAvailability` + `build_availability(...)`: per-player per-week injury `p` (era-normalized games-played from `weekly_stats`, position default for rookies, clamped) and bye week (via `id_map` + target-season `schedules`).
- `src/projections/draft/assistant/season_value.py` — `expected_season_points(...)`: the MC season valuer with the single-week factorization, reusing `optimal_lineup_points` (spec §3.4).
- `src/projections/draft/assistant/valuer.py` — `RosterValuer` Protocol + `StartersValuer` (wraps `optimal_lineup_points`) + `SeasonValuer` (wraps `expected_season_points`, deterministic per-roster seed).

**Create (tests):**
- `tests/test_draft/test_assistant_availability.py`
- `tests/test_draft/test_assistant_season_value.py`
- `tests/test_draft/test_assistant_valuer.py`

**Modify:**
- `src/projections/draft/assistant/tournament.py` — thread an optional `RosterValuer` through `_strategy_values` / `run_tournament` / `tune_sigma` (default → `StartersValuer`, no behavior change).
- `src/projections/draft/assistant/tournament_cli.py` — `--valuer {starters,season}` (+ `--season`, `--n-sims`, `--data-root`); build the `SeasonValuer` from `weekly_stats` / `schedules` / `id_map`.
- `tests/test_draft/test_assistant_tournament.py` + `tests/test_draft/test_assistant_tournament_cli.py` — new valuer tests; confirm the default path is unchanged.
- `project_management.md` + `TODO.md` (Task 6).

**Deliberately NOT modified:** `roster_score.py` (`optimal_lineup_points` is reused verbatim — `per_game = season/17` is a uniform scaling, so the optimal weekly lineup is identical), the strategies, the survival model, the bootstrap/winner machinery.

**Read-before-coding facts:**
- `optimal_lineup_points(roster_rows, roster_slots)` reads `gsis_id`, `position`, `season_mean_fpts`; returns the summed optimal starting-lineup `season_mean_fpts`. Empty roster → `0.0`.
- `WeeklyStatsSchema`: one row per `(gsis_id, season, week)` with a `position` column. `SchedulesSchema`: `season`, `week`, `home_team`, `away_team`. `id_map`: `gsis_id`, `team`, `position`, `full_name` (load via `pd.read_parquet` + `IdMapSchema.validate`).
- `store.read_partition(root, table, *, season=...)` reads `data/raw/weekly_stats` / `data/raw/schedules` one season at a time.
- `_PYARROW_STR` is the pyarrow string dtype from `projections.schemas`.

---

## Task 1: Availability model (`availability.py`)

Per-player per-week injury `p` from games-played history (era-normalized, position default for rookies, clamped) + bye week from the target-season schedule.

**Files:**
- Create: `src/projections/draft/assistant/availability.py`
- Test: `tests/test_draft/test_assistant_availability.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_draft/test_assistant_availability.py
"""Tests for the per-player season availability model."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.draft.assistant.availability import build_availability
from projections.schemas import _PYARROW_STR


def _weekly_stats(rows: list[tuple[str, int, int, str]]) -> pd.DataFrame:
    """rows = [(gsis_id, season, week, position), ...]."""
    return pd.DataFrame(
        {
            "gsis_id": pd.array([r[0] for r in rows], dtype=_PYARROW_STR),
            "season": [r[1] for r in rows],
            "week": [r[2] for r in rows],
            "position": pd.array([r[3] for r in rows], dtype=_PYARROW_STR),
        }
    )


def _schedules(season: int, byes: dict[str, int], n_weeks: int = 18) -> pd.DataFrame:
    """Build a schedule where each team in `byes` is missing exactly its bye week.
    Two teams (A1/A2) pair up every week except their byes; enough to exercise the
    'week with no game row for the team' rule."""
    rows = []
    teams = list(byes)
    for w in range(1, n_weeks + 1):
        playing = [t for t in teams if byes[t] != w]
        # pair them arbitrarily; odd one out plays a filler 'ZZ'
        for i in range(0, len(playing) - 1, 2):
            rows.append((season, w, playing[i], playing[i + 1]))
        if len(playing) % 2 == 1:
            rows.append((season, w, playing[-1], "ZZ"))
    return pd.DataFrame(
        {
            "season": [r[0] for r in rows],
            "week": [r[1] for r in rows],
            "home_team": pd.array([r[2] for r in rows], dtype=_PYARROW_STR),
            "away_team": pd.array([r[3] for r in rows], dtype=_PYARROW_STR),
        }
    )


def _id_map(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """rows = [(gsis_id, team), ...]."""
    return pd.DataFrame(
        {
            "gsis_id": pd.array([r[0] for r in rows], dtype=_PYARROW_STR),
            "team": pd.array([r[1] for r in rows], dtype=_PYARROW_STR),
        }
    )


def _pool(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """rows = [(gsis_id, position), ...]."""
    return pd.DataFrame(
        {
            "gsis_id": pd.array([r[0] for r in rows], dtype=_PYARROW_STR),
            "position": pd.array([r[1] for r in rows], dtype=_PYARROW_STR),
        }
    )


def test_workhorse_and_injury_prone_and_rookie() -> None:
    # A: 17/17 in a 17-game season (clamped to hi). B: 9/17 (injury-prone). R: rookie, no history.
    ws = _weekly_stats(
        [("00-0000001", 2022, w, "RB") for w in range(1, 18)]  # A plays all 17
        + [("00-0000002", 2022, w, "RB") for w in range(1, 10)]  # B plays 9
    )
    sched = _schedules(2026, {"AA": 7, "BB": 9, "RR": 5})
    id_map = _id_map([("00-0000001", "AA"), ("00-0000002", "BB"), ("00-0000009", "RR")])
    pool = _pool([("00-0000001", "RB"), ("00-0000002", "RB"), ("00-0000009", "RB")])

    avail = build_availability(ws, sched, id_map, pool, season=2026)

    assert avail.p_week("00-0000001") == pytest.approx(0.97)  # clamped to hi
    assert avail.p_week("00-0000002") == pytest.approx(9 / 17, abs=1e-9)
    # rookie -> RB position default = mean of A(clamped pre-clamp 1.0) and B(0.529)... default
    # uses raw (pre-clamp) means; just assert it lands in a sane band and equals the position mean.
    assert 0.4 <= avail.p_week("00-0000009") <= 0.97
    assert avail.bye_week("00-0000001") == 7
    assert avail.bye_week("00-0000002") == 9
    assert avail.bye_week("00-0000009") == 5


def test_16_game_era_is_normalized() -> None:
    # A plays all 16 of a 2019 (16-game) season -> frac 1.0, not 16/17.
    ws = _weekly_stats([("00-0000001", 2019, w, "WR") for w in range(1, 17)])
    sched = _schedules(2026, {"AA": 6})
    avail = build_availability(
        ws, sched, _id_map([("00-0000001", "AA")]), _pool([("00-0000001", "WR")]), season=2026
    )
    assert avail.p_week("00-0000001") == pytest.approx(0.97)  # 16/16 = 1.0 -> clamp hi


def test_missing_schedule_degrades_to_no_byes() -> None:
    ws = _weekly_stats([("00-0000001", 2022, w, "RB") for w in range(1, 12)])
    sched = _schedules(2025, {"AA": 7})  # wrong season -> no 2026 rows
    with pytest.warns(UserWarning, match="no schedules for season 2026"):
        avail = build_availability(
            ws, sched, _id_map([("00-0000001", "AA")]), _pool([("00-0000001", "RB")]), season=2026
        )
    assert avail.bye_week("00-0000001") is None
    assert avail.p_week("00-0000001") == pytest.approx(11 / 17, abs=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_draft/test_assistant_availability.py -v -n0`
Expected: FAIL — `ModuleNotFoundError: ... 'projections.draft.assistant.availability'`.

- [ ] **Step 3: Write the implementation**

```python
# src/projections/draft/assistant/availability.py
"""Per-player season availability: injury Bernoulli `p` + bye week (spec §3.2).

`p` is the fraction of its team's games a player plays, era-normalized over their
`weekly_stats` history (16-game 2018-2020 vs 17-game 2021+), with a per-position
default for rookies / no-history and a clamp to keep no player degenerate. Byes
come from the target-season schedule via the player's `id_map` team.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd


def _sched_games(season: int) -> int:
    """Regular-season games a team plays that season (excludes the bye)."""
    return 16 if season <= 2020 else 17


def _team_byes(schedules: pd.DataFrame, season: int) -> dict[str, int]:
    """Map team -> bye week for `season`: the single week the team has no game row.

    Empty (with a warning) if the target-season partition has no rows — graceful
    degradation so the injury model still applies without byes.
    """
    sch = schedules[schedules["season"] == season]
    if len(sch) == 0:
        warnings.warn(f"no schedules for season {season}; byes will be empty", stacklevel=2)
        return {}
    weeks = sorted(int(w) for w in sch["week"].unique())
    teams = pd.unique(pd.concat([sch["home_team"], sch["away_team"]], ignore_index=True))
    byes: dict[str, int] = {}
    for team in teams:
        played = {
            int(w)
            for w in sch.loc[
                (sch["home_team"] == team) | (sch["away_team"] == team), "week"
            ]
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
    ws = weekly_stats[["gsis_id", "season", "week", "position"]]
    games = ws.groupby(["gsis_id", "season"]).size().rename("games").reset_index()
    games["sched"] = games["season"].map(_sched_games)
    games["frac"] = games["games"] / games["sched"]
    p_raw = games.groupby("gsis_id")["frac"].mean()
    pos_hist = ws.groupby("gsis_id")["position"].agg(lambda s: str(s.mode().iloc[0]))

    hist = pd.DataFrame({"p": p_raw, "position": pos_hist})
    default_by_pos = {str(k): float(v) for k, v in hist.groupby("position")["p"].mean().items()}
    overall_default = float(hist["p"].mean()) if len(hist) else (lo + hi) / 2

    p: dict[str, float] = {}
    for gid, pos in zip(pool["gsis_id"].astype(str), pool["position"].astype(str), strict=True):
        raw = float(p_raw.loc[gid]) if gid in p_raw.index else default_by_pos.get(pos, overall_default)
        p[gid] = min(max(raw, lo), hi)

    team_of = dict(zip(id_map["gsis_id"].astype(str), id_map["team"].astype(str), strict=False))
    team_byes = _team_byes(schedules, season)
    bye: dict[str, int] = {}
    for gid in p:
        team = team_of.get(gid)
        if team is not None and team in team_byes:
            bye[gid] = team_byes[team]

    return PlayerAvailability(p=p, bye=bye)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_draft/test_assistant_availability.py -v -n0`
Expected: 3 passed.

- [ ] **Step 5: Lint + type-check**

Run: `python -m ruff check src/projections/draft/assistant/availability.py tests/test_draft/test_assistant_availability.py && python -m ruff format src/projections/draft/assistant/availability.py tests/test_draft/test_assistant_availability.py && python -m ruff format --check src/projections/draft/assistant/availability.py tests/test_draft/test_assistant_availability.py && python -m mypy src/projections/draft/assistant/availability.py`
Expected: no violations.

- [ ] **Step 6: Commit**

```bash
git add src/projections/draft/assistant/availability.py tests/test_draft/test_assistant_availability.py
git commit -m "feat(draft): per-player season availability model (risk-aware valuation)"
```

---

## Task 2: Season Monte-Carlo valuer (`season_value.py`)

`expected_season_points`: MC a season, filling the best lineup from the healthy each week, reusing `optimal_lineup_points/17`, with the single-week factorization for speed.

**Files:**
- Create: `src/projections/draft/assistant/season_value.py`
- Test: `tests/test_draft/test_assistant_season_value.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_draft/test_assistant_season_value.py
"""Tests for the Monte-Carlo season valuer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.roster_score import optimal_lineup_points
from projections.draft.assistant.season_value import expected_season_points
from projections.schemas import _PYARROW_STR, RosterSlot


def _roster(players: list[tuple[str, str, float]]) -> pd.DataFrame:
    """players = [(gsis_id, position, season_mean_fpts), ...]."""
    return pd.DataFrame(
        {
            "gsis_id": pd.array([p[0] for p in players], dtype=_PYARROW_STR),
            "position": pd.array([p[1] for p in players], dtype=_PYARROW_STR),
            "season_mean_fpts": [p[2] for p in players],
        }
    )


def _avail(p: dict[str, float], bye: dict[str, int] | None = None) -> PlayerAvailability:
    return PlayerAvailability(p=p, bye=bye or {})


def test_closed_form_single_slot() -> None:
    # 1 RB, p=0.5, no bye, 2 weeks. per_game = 170/17 = 10. E = 2 * 0.5 * 10 = 10.
    roster = _roster([("00-0000001", "RB", 170.0)])
    avail = _avail({"00-0000001": 0.5})
    val = expected_season_points(
        roster, {RosterSlot.RB: 1}, avail, n_sims=20000, rng=np.random.default_rng(0),
        weeks=range(1, 3),
    )
    assert abs(val - 10.0) < 0.3  # MC tolerance


def test_reduces_to_starters_when_always_available() -> None:
    # p=1.0, no byes, 17 weeks -> equals optimal_lineup_points exactly (17 * season/17).
    roster = _roster(
        [("00-0000001", "RB", 200.0), ("00-0000002", "WR", 180.0), ("00-0000003", "RB", 120.0)]
    )
    slots = {RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.FLEX: 1}
    avail = _avail({"00-0000001": 1.0, "00-0000002": 1.0, "00-0000003": 1.0})
    val = expected_season_points(
        roster, slots, avail, n_sims=50, rng=np.random.default_rng(0), weeks=range(1, 18)
    )
    assert val == optimal_lineup_points(roster, slots)


def test_depth_is_rewarded_over_qb_hoarding() -> None:
    # Two rosters, same total projection, same starters. One adds a 3rd RB (real depth),
    # the other a spare QB (useless beyond the 1 QB slot). Depth must score higher.
    slots = {RosterSlot.QB: 1, RosterSlot.RB: 2, RosterSlot.FLEX: 1}
    base = [("00-0000001", "QB", 300.0), ("00-0000002", "RB", 200.0), ("00-0000003", "RB", 190.0)]
    depth = _roster(base + [("00-0000004", "RB", 150.0)])
    hoard = _roster(base + [("00-0000005", "QB", 150.0)])
    p = {f"00-000000{i}": 0.8 for i in range(1, 6)}
    val_depth = expected_season_points(
        depth, slots, _avail(p), n_sims=3000, rng=np.random.default_rng(1), weeks=range(1, 18)
    )
    val_hoard = expected_season_points(
        hoard, slots, _avail(p), n_sims=3000, rng=np.random.default_rng(1), weeks=range(1, 18)
    )
    assert val_depth > val_hoard


def test_determinism() -> None:
    roster = _roster([("00-0000001", "RB", 200.0), ("00-0000002", "RB", 150.0)])
    slots = {RosterSlot.RB: 1, RosterSlot.FLEX: 1}
    avail = _avail({"00-0000001": 0.7, "00-0000002": 0.7})
    a = expected_season_points(roster, slots, avail, n_sims=200, rng=np.random.default_rng(5),
                               weeks=range(1, 18))
    b = expected_season_points(roster, slots, avail, n_sims=200, rng=np.random.default_rng(5),
                               weeks=range(1, 18))
    assert a == b


def test_bye_costs_points_and_factorization_matches_bruteforce() -> None:
    # One RB on bye in week 7; the factorized result must match a brute-force per-week MC.
    roster = _roster([("00-0000001", "RB", 200.0), ("00-0000002", "WR", 150.0)])
    slots = {RosterSlot.RB: 1, RosterSlot.WR: 1}
    avail = _avail({"00-0000001": 0.85, "00-0000002": 0.85}, bye={"00-0000001": 7})

    fact = expected_season_points(
        roster, slots, avail, n_sims=8000, rng=np.random.default_rng(0), weeks=range(1, 18)
    )

    # Brute force: simulate every week independently.
    rng = np.random.default_rng(0)
    gsis = roster["gsis_id"].astype(str).to_numpy()
    p_arr = np.array([avail.p_week(g) for g in gsis])
    weeks = list(range(1, 18))
    acc = 0.0
    n_sims = 8000
    for _ in range(n_sims):
        season_pts = 0.0
        for w in weeks:
            forced = np.array([avail.bye_week(g) == w for g in gsis])
            mask = (rng.random(len(roster)) < p_arr) & ~forced
            sub = roster.iloc[np.flatnonzero(mask)]
            season_pts += optimal_lineup_points(sub, slots) / 17.0
        acc += season_pts
    brute = acc / n_sims

    assert abs(fact - brute) / brute < 0.02  # within 2% (MC noise, same expectation)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_draft/test_assistant_season_value.py -v -n0`
Expected: FAIL — `ModuleNotFoundError: ... 'projections.draft.assistant.season_value'`.

- [ ] **Step 3: Write the implementation**

```python
# src/projections/draft/assistant/season_value.py
"""Expected season points under per-player availability (spec §3.4).

Monte-Carlo a season: each week, players are available (not on bye, healthy w.p.
`p`), and the best legal lineup is filled from the available roster. Because
`per_game = season_mean_fpts / 17` is a uniform scaling, the weekly optimal lineup
is exactly `optimal_lineup_points(available_subset) / 17` — the existing greedy
fill is reused verbatim. Weeks with no roster bye are identical in expectation, so
we MC one generic week and reuse it (the factorization is exact in expectation).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.roster_score import optimal_lineup_points
from projections.schemas import RosterSlot

_GAMES = 17  # season projection -> per-game divisor (uniform scaling, spec §3.3)


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
    bye_arr = np.array([availability.bye_week(g) if availability.bye_week(g) is not None else -1
                        for g in gsis])
    weeks = list(weeks)
    roster_bye_weeks = sorted({w for w in bye_arr.tolist() if w in weeks})

    def week_expectation(forced_out: np.ndarray) -> float:
        acc = 0.0
        for _ in range(n_sims):
            available = (rng.random(n) < p_arr) & ~forced_out
            sub = roster.iloc[np.flatnonzero(available)]
            acc += optimal_lineup_points(sub, roster_slots)
        return acc / n_sims / _GAMES

    no_force = np.zeros(n, dtype=bool)
    clean_week_value = week_expectation(no_force)
    clean_weeks = sum(1 for w in weeks if w not in roster_bye_weeks)
    total = clean_weeks * clean_week_value
    for w in roster_bye_weeks:
        total += week_expectation(bye_arr == w)
    return total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_draft/test_assistant_season_value.py -v -n0`
Expected: 5 passed. (The factorization test is the slow one — a few seconds.)

- [ ] **Step 5: Lint + type-check**

Run: `python -m ruff check src/projections/draft/assistant/season_value.py tests/test_draft/test_assistant_season_value.py && python -m ruff format src/projections/draft/assistant/season_value.py tests/test_draft/test_assistant_season_value.py && python -m ruff format --check src/projections/draft/assistant/season_value.py tests/test_draft/test_assistant_season_value.py && python -m mypy src/projections/draft/assistant/season_value.py`
Expected: no violations.

- [ ] **Step 6: Commit**

```bash
git add src/projections/draft/assistant/season_value.py tests/test_draft/test_assistant_season_value.py
git commit -m "feat(draft): expected-season-points MC valuer (risk-aware valuation)"
```

---

## Task 3: Pluggable `RosterValuer` (`valuer.py`)

The seam: `RosterValuer` Protocol + `StartersValuer` (wraps `optimal_lineup_points`) + `SeasonValuer` (wraps `expected_season_points` with a deterministic per-roster seed).

**Files:**
- Create: `src/projections/draft/assistant/valuer.py`
- Test: `tests/test_draft/test_assistant_valuer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_draft/test_assistant_valuer.py
"""Tests for the RosterValuer seam."""

from __future__ import annotations

import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.roster_score import optimal_lineup_points
from projections.draft.assistant.valuer import RosterValuer, SeasonValuer, StartersValuer
from projections.schemas import _PYARROW_STR, RosterSlot


def _roster(players: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": pd.array([p[0] for p in players], dtype=_PYARROW_STR),
            "position": pd.array([p[1] for p in players], dtype=_PYARROW_STR),
            "season_mean_fpts": [p[2] for p in players],
        }
    )


def test_both_satisfy_protocol() -> None:
    avail = PlayerAvailability(p={}, bye={})
    assert isinstance(StartersValuer(), RosterValuer)
    assert isinstance(SeasonValuer(availability=avail, n_sims=10, base_seed=0), RosterValuer)


def test_starters_valuer_equals_optimal_lineup_points() -> None:
    roster = _roster([("00-0000001", "RB", 200.0), ("00-0000002", "WR", 150.0)])
    slots = {RosterSlot.RB: 1, RosterSlot.WR: 1}
    assert StartersValuer().value(roster, slots) == optimal_lineup_points(roster, slots)


def test_season_valuer_is_deterministic_per_roster() -> None:
    roster = _roster([("00-0000001", "RB", 200.0), ("00-0000002", "RB", 150.0)])
    slots = {RosterSlot.RB: 1, RosterSlot.FLEX: 1}
    avail = PlayerAvailability(p={"00-0000001": 0.7, "00-0000002": 0.7}, bye={})
    v = SeasonValuer(availability=avail, n_sims=100, base_seed=0)
    assert v.value(roster, slots) == v.value(roster, slots)  # same roster -> same value
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_draft/test_assistant_valuer.py -v -n0`
Expected: FAIL — `ModuleNotFoundError: ... 'projections.draft.assistant.valuer'`.

- [ ] **Step 3: Write the implementation**

```python
# src/projections/draft/assistant/valuer.py
"""Pluggable roster valuation (spec §3.5).

`StartersValuer` is the cheap starters-only metric (today's default). `SeasonValuer`
is the risk-aware expected-season-points metric. Both satisfy `RosterValuer`, so the
tournament can A/B them. `SeasonValuer` derives a deterministic per-roster seed (a
sha256 of the sorted gsis_ids, xored with base_seed) so identical rosters score
identically and the tournament stays reproducible.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.roster_score import optimal_lineup_points
from projections.draft.assistant.season_value import expected_season_points
from projections.schemas import RosterSlot


@runtime_checkable
class RosterValuer(Protocol):
    def value(self, roster: pd.DataFrame, roster_slots: Mapping[RosterSlot, int]) -> float:
        """Score a completed roster."""
        ...


@dataclass(frozen=True)
class StartersValuer:
    """Optimal single-week starting lineup (the cheap, deterministic default)."""

    def value(self, roster: pd.DataFrame, roster_slots: Mapping[RosterSlot, int]) -> float:
        return optimal_lineup_points(roster, roster_slots)


def _roster_seed(base_seed: int, roster: pd.DataFrame) -> int:
    """Stable 32-bit seed from base_seed + the roster's sorted gsis_ids."""
    key = ",".join(sorted(str(g) for g in roster["gsis_id"]))
    digest = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
    return (base_seed ^ digest) & 0xFFFFFFFF


@dataclass(frozen=True)
class SeasonValuer:
    """Expected season points under availability risk (spec §3.4)."""

    availability: PlayerAvailability
    n_sims: int
    base_seed: int
    weeks: Iterable[int] = field(default_factory=lambda: range(1, 18))

    def value(self, roster: pd.DataFrame, roster_slots: Mapping[RosterSlot, int]) -> float:
        rng = np.random.default_rng(_roster_seed(self.base_seed, roster))
        return expected_season_points(
            roster, roster_slots, self.availability,
            n_sims=self.n_sims, rng=rng, weeks=self.weeks,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_draft/test_assistant_valuer.py -v -n0`
Expected: 3 passed.

- [ ] **Step 5: Lint + type-check**

Run: `python -m ruff check src/projections/draft/assistant/valuer.py tests/test_draft/test_assistant_valuer.py && python -m ruff format src/projections/draft/assistant/valuer.py tests/test_draft/test_assistant_valuer.py && python -m ruff format --check src/projections/draft/assistant/valuer.py tests/test_draft/test_assistant_valuer.py && python -m mypy src/projections/draft/assistant/valuer.py`
Expected: no violations. (If mypy flags the `range` default on a frozen dataclass `Iterable[int]` field, the `field(default_factory=...)` form shown avoids the mutable-default error.)

- [ ] **Step 6: Commit**

```bash
git add src/projections/draft/assistant/valuer.py tests/test_draft/test_assistant_valuer.py
git commit -m "feat(draft): pluggable RosterValuer (starters + season) (risk-aware valuation)"
```

---

## Task 4: Thread the valuer through the tournament

`_strategy_values` / `run_tournament` / `tune_sigma` take an optional `RosterValuer` (default `StartersValuer`, so existing behavior + tests are unchanged).

**Files:**
- Modify: `src/projections/draft/assistant/tournament.py`
- Test: `tests/test_draft/test_assistant_tournament.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_draft/test_assistant_tournament.py`)

```python
def test_default_valuer_matches_optimal_lineup_points() -> None:
    # The default-valuer tournament must equal the pre-change behavior: StartersValuer
    # is optimal_lineup_points, so run_tournament's numbers are unchanged.
    from projections.draft.assistant.valuer import StartersValuer

    kwargs = dict(pool=_pool(), config=_config(), my_slot=1, n_seeds=20, adp_jitter=2.0, base_seed=0)
    default_run = run_tournament({"best": _BestFpts(), "worst": _WorstFpts()}, **kwargs)
    explicit = run_tournament(
        {"best": _BestFpts(), "worst": _WorstFpts()}, valuer=StartersValuer(), **kwargs
    )
    assert default_run.summaries["best"].point == explicit.summaries["best"].point
    assert default_run.summaries["worst"].point == explicit.summaries["worst"].point


def test_season_valuer_runs_in_tournament() -> None:
    # A tournament scored by the season valuer produces a valid result (smoke + shape).
    from projections.draft.assistant.availability import PlayerAvailability
    from projections.draft.assistant.valuer import SeasonValuer

    pool = _pool()
    avail = PlayerAvailability(
        p={str(g): 0.8 for g in pool["gsis_id"]}, bye={}
    )
    valuer = SeasonValuer(availability=avail, n_sims=50, base_seed=0)
    result = run_tournament(
        {"best": _BestFpts(), "worst": _WorstFpts()},
        pool=pool, config=_config(), my_slot=1, n_seeds=10, adp_jitter=2.0, base_seed=0,
        valuer=valuer,
    )
    assert set(result.summaries) == {"best", "worst"}
    assert all(ci.point > 0 for ci in result.summaries.values())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_draft/test_assistant_tournament.py -k "valuer" -v -n0`
Expected: FAIL — `run_tournament() got an unexpected keyword argument 'valuer'`.

- [ ] **Step 3: Thread the valuer**

In `src/projections/draft/assistant/tournament.py`:

(a) Add the import near the top (with the other `projections.draft.assistant` imports):
```python
from projections.draft.assistant.valuer import RosterValuer, StartersValuer
```

(b) `_strategy_values` — add a `valuer` parameter and use it instead of `optimal_lineup_points`:
```python
def _strategy_values(
    strategy: DraftStrategy,
    pool: pd.DataFrame,
    config: LeagueConfig,
    *,
    my_slot: int,
    n_seeds: int,
    adp_jitter: float,
    base_seed: int,
    valuer: RosterValuer,
) -> np.ndarray:
    """Roster value (per the valuer) of the hero roster for each paired seed."""
    out = np.empty(n_seeds, dtype=np.float64)
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        roster = simulate_draft(strategy, my_slot, pool, config, adp_jitter=adp_jitter, rng=rng)
        out[s] = valuer.value(roster, config.roster_slots)
    return out
```
(Delete the now-unused `from projections.draft.assistant.roster_score import optimal_lineup_points` import if present — `tournament.py` no longer calls it directly; verify with grep.)

(c) `run_tournament` — add `valuer: RosterValuer | None = None`, default it, and pass it to every `_strategy_values` call:
```python
def run_tournament(
    strategies: Mapping[str, DraftStrategy],
    pool: pd.DataFrame,
    config: LeagueConfig,
    *,
    my_slot: int,
    n_seeds: int,
    adp_jitter: float,
    base_seed: int,
    valuer: RosterValuer | None = None,
) -> TournamentResult:
    """Compare `strategies` over `n_seeds` paired drafts; declare a winner."""
    _validate_pool(pool, config)
    _validate_run_params(config, my_slot=my_slot, n_seeds=n_seeds, adp_jitter=adp_jitter)
    valuer = valuer if valuer is not None else StartersValuer()
    values = {
        name: _strategy_values(
            strat, pool, config, my_slot=my_slot, n_seeds=n_seeds,
            adp_jitter=adp_jitter, base_seed=base_seed, valuer=valuer,
        )
        for name, strat in strategies.items()
    }
    # ... rest unchanged ...
```

(d) `tune_sigma` — same treatment: add `valuer: RosterValuer | None = None`, default it after `_validate_run_params`, and pass `valuer=valuer` into its `_strategy_values` call:
```python
def tune_sigma(
    sigma_grid: Sequence[float],
    pool: pd.DataFrame,
    config: LeagueConfig,
    *,
    my_slot: int,
    n_seeds: int,
    adp_jitter: float,
    base_seed: int,
    valuer: RosterValuer | None = None,
) -> SigmaTuningResult:
    ...
    _validate_run_params(config, my_slot=my_slot, n_seeds=n_seeds, adp_jitter=adp_jitter)
    if not sigma_grid:
        raise ValueError("sigma_grid must be non-empty")
    if any(s <= 0 for s in sigma_grid):
        raise ValueError(f"sigma_grid values must all be > 0; got {list(sigma_grid)}")
    valuer = valuer if valuer is not None else StartersValuer()
    grid: list[tuple[float, float]] = []
    for sigma in sigma_grid:
        strat = NowOrNeverStrategy(LogisticSurvival(sigma=float(sigma)))
        vals = _strategy_values(
            strat, pool, config, my_slot=my_slot, n_seeds=n_seeds,
            adp_jitter=adp_jitter, base_seed=base_seed, valuer=valuer,
        )
        grid.append((float(sigma), float(vals.mean())))
    # ... rest unchanged ...
```

- [ ] **Step 4: Run the new + full tournament tests**

Run: `python -m pytest tests/test_draft/test_assistant_tournament.py -v -n0`
Expected: all pass (existing tests unchanged because the default is `StartersValuer`; 2 new pass).

- [ ] **Step 5: Lint + type-check**

Run: `python -m ruff check src/projections/draft/assistant/tournament.py tests/test_draft/test_assistant_tournament.py && python -m ruff format --check src/projections/draft/assistant/tournament.py tests/test_draft/test_assistant_tournament.py && python -m mypy src/projections/draft/assistant/tournament.py`
Expected: no violations.

- [ ] **Step 6: Commit**

```bash
git add src/projections/draft/assistant/tournament.py tests/test_draft/test_assistant_tournament.py
git commit -m "feat(draft): tournament takes a RosterValuer (default unchanged) (risk-aware valuation)"
```

---

## Task 5: CLI `--valuer season`

`--valuer {starters,season}` (default starters). For `season`, build the `SeasonValuer` from `weekly_stats` (2018–2024) + target-season `schedules` + `id_map`.

**Files:**
- Modify: `src/projections/draft/assistant/tournament_cli.py`
- Test: `tests/test_draft/test_assistant_tournament_cli.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_draft/test_assistant_tournament_cli.py`)

```python
def test_compare_with_season_valuer_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from projections.store import write_partition

    vorp_path, cfg_path = _write_inputs(tmp_path)
    data_root = tmp_path / "data"
    raw = data_root / "raw"
    # minimal weekly_stats (one prior season) + target-season schedules + id_map
    n = 24
    gsis = [f"00-00000{i:02d}" for i in range(1, n + 1)]
    ws = pd.DataFrame(
        {
            "gsis_id": pd.array(gsis * 10, dtype=_PYARROW_STR),
            "season": [2022] * (n * 10),
            "week": [w for w in range(1, 11) for _ in range(n)],
            "position": pd.array((["RB" if i % 2 else "WR" for i in range(n)]) * 10, dtype=_PYARROW_STR),
        }
    )
    write_partition(raw, "weekly_stats", ws, season=2022)
    sched = pd.DataFrame(
        {
            "season": [2026] * 2,
            "week": [1, 2],
            "home_team": pd.array(["AA", "AA"], dtype=_PYARROW_STR),
            "away_team": pd.array(["BB", "BB"], dtype=_PYARROW_STR),
        }
    )
    write_partition(raw, "schedules", sched, season=2026)
    id_map = pd.DataFrame(
        {
            "gsis_id": pd.array(gsis, dtype=_PYARROW_STR),
            "team": pd.array(["AA"] * n, dtype=_PYARROW_STR),
        }
    )
    (raw).mkdir(parents=True, exist_ok=True)
    id_map.to_parquet(raw / "id_map.parquet")

    code = run(
        [
            "--vorp-table", str(vorp_path),
            "--league-config", str(cfg_path),
            "--my-slot", "2", "--seeds", "6", "--seed", "0",
            "--valuer", "season", "--season", "2026", "--n-sims", "20",
            "--data-root", str(data_root),
            "compare",
        ]
    )
    assert code == 0
    assert "Winner:" in capsys.readouterr().out
```

(Note: `write_partition`'s exact season/asof signature — confirm with `store/parquet.py`; if it needs an `asof`, pass `asof=None`. The `id_map` lives at `data/raw/id_map.parquet` per the Slice 1 default.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_draft/test_assistant_tournament_cli.py -k season -v -n0`
Expected: FAIL — argparse error on the unknown `--valuer` argument.

- [ ] **Step 3: Implement the CLI valuer**

In `src/projections/draft/assistant/tournament_cli.py`:

(a) Imports:
```python
from projections.draft.assistant.availability import build_availability
from projections.draft.assistant.valuer import RosterValuer, SeasonValuer, StartersValuer
from projections.store import read_partition
```
(`build_availability` reads only `gsis_id` + `team` from the id_map, so no full `IdMapSchema` validation is needed here — `pd.read_parquet` is enough, and it raises `FileNotFoundError` naturally if the file is missing.)

(b) A helper to build the season valuer:
```python
_HISTORY_SEASONS = range(2018, 2025)  # weekly_stats coverage for the availability model


def _build_season_valuer(
    pool: pd.DataFrame, *, season: int, n_sims: int, base_seed: int, data_root: Path
) -> SeasonValuer:
    raw = data_root / "raw"
    frames = []
    for yr in _HISTORY_SEASONS:
        try:
            frames.append(read_partition(raw, "weekly_stats", season=yr))
        except FileNotFoundError:
            continue
    if not frames:
        raise FileNotFoundError(f"no weekly_stats partitions under {raw} for {_HISTORY_SEASONS}")
    weekly_stats = pd.concat(frames, ignore_index=True)
    schedules = read_partition(raw, "schedules", season=season)
    id_map = pd.read_parquet(raw / "id_map.parquet")
    availability = build_availability(weekly_stats, schedules, id_map, pool, season=season)
    return SeasonValuer(availability=availability, n_sims=n_sims, base_seed=base_seed)
```

(c) New global args (alongside `--seed`):
```python
    p.add_argument("--valuer", choices=["starters", "season"], default="starters",
                   help="Roster metric: 'starters' (optimal single-week lineup) or "
                        "'season' (expected points under availability + byes).")
    p.add_argument("--season", type=int, default=2026,
                   help="[--valuer season] target season for byes + availability.")
    p.add_argument("--n-sims", type=int, default=300,
                   help="[--valuer season] Monte-Carlo seasons per roster.")
    p.add_argument("--data-root", type=Path, default=Path("data"),
                   help="[--valuer season] store root for weekly_stats/schedules/id_map.")
```

(d) In `run()`, build the valuer once after loading `pool`/`config` and thread it into both modes:
```python
    valuer: RosterValuer = (
        StartersValuer()
        if args.valuer == "starters"
        else _build_season_valuer(
            pool, season=args.season, n_sims=args.n_sims, base_seed=args.seed,
            data_root=args.data_root,
        )
    )
```
Then add `valuer=valuer` to the `run_tournament(...)` call (compare mode) and the `tune_sigma(...)` call (tune-sigma mode).

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_draft/test_assistant_tournament_cli.py -v -n0`
Expected: all pass (existing CLI tests unchanged; the new season-valuer smoke passes).

- [ ] **Step 5: Lint + type-check**

Run: `python -m ruff check src/projections/draft/assistant/tournament_cli.py tests/test_draft/test_assistant_tournament_cli.py && python -m ruff format --check src/projections/draft/assistant/tournament_cli.py tests/test_draft/test_assistant_tournament_cli.py && python -m mypy src/projections/draft/assistant/tournament_cli.py`
Expected: no violations.

- [ ] **Step 6: Commit**

```bash
git add src/projections/draft/assistant/tournament_cli.py tests/test_draft/test_assistant_tournament_cli.py
git commit -m "feat(draft): CLI --valuer season (risk-aware valuation)"
```

---

## Task 6: Full gate, real-data validation, and docs

Prove the slice green, confirm the metric flips the QB-hoarding story on real 2026 data, and update the running docs.

**Files:**
- Modify: `project_management.md`, `TODO.md`

- [ ] **Step 1: Full project gate**

```bash
python -m pytest -q
python -m mypy src tests
python -m ruff check src tests
python -m ruff format --check src tests
```
Expected: all green except the single pre-existing TODO #40 failure (`tests/backtest/test_backtest_smoke.py::test_backtest_smoke_one_cell`, WR feature build — unrelated; confirm it is the only failure and that `git diff --stat origin/main -- src/projections/features src/projections/backtest` is empty). mypy/ruff/format clean.

- [ ] **Step 2: Real-data validation (the payoff)**

Generate the VORP table (if not present) and compare the two strategies under BOTH metrics:
```bash
python scripts/generate_vorp_table.py --season 2026 --source consensus \
  --league-config configs/league_espn_ppr_12team_skill.json --out data/consensus_vorp_2026.parquet
# starters metric (baseline):
python scripts/draft_tournament.py --vorp-table data/consensus_vorp_2026.parquet \
  --league-config configs/league_espn_ppr_12team_skill.json --my-slot 6 --seeds 60 compare
# season metric (risk-aware):
python scripts/draft_tournament.py --vorp-table data/consensus_vorp_2026.parquet \
  --league-config configs/league_espn_ppr_12team_skill.json --my-slot 6 --seeds 60 \
  --valuer season --season 2026 --n-sims 300 compare
```
Expected: both run and print a winner. Record the season-metric numbers. (If the 2026 `schedules` partition is absent, the run warns and proceeds with no byes — still valid; note it.) The key check: now_or_never's depth advantage should be **at least as large** under the season metric as under starters (the season metric rewards its RB/WR depth and punishes any QB hoarding).

- [ ] **Step 3: Update `project_management.md`**

Add a top entry dated 2026-06-11: the risk-aware roster valuation shipped on `feat/risk-aware-roster-valuation`; what shipped (`availability`, `season_value`, `valuer`, tournament `--valuer`); key decisions (availability-only, `per_game=projection/17`, pluggable valuer default-unchanged, single-week factorization); the validation numbers from Step 2; next directions (depth-aware strategy; weekly performance variance; recency-weighted availability).

- [ ] **Step 4: Update `TODO.md`**

Under the Draft Assistant bullet (TODO #38), add the risk-aware valuation as ✅ DONE with the usage line:
`python scripts/draft_tournament.py ... --valuer season --season 2026 --n-sims 300 {compare|tune-sigma}`, and note the deferred follow-ups (depth-aware strategy, weekly performance variance, recency-weighted availability, playoff weighting).

- [ ] **Step 5: Commit**

```bash
git add project_management.md TODO.md
git commit -m "docs(pm,todo): risk-aware roster valuation shipped (season availability)"
```

---

## Self-Review (completed by plan author)

**Spec coverage:** §2 `availability.py`/`season_value.py`/`RosterValuer`/tournament-integration/validation → Tasks 1/2/3/4+5/6. §3.1 inputs (roster, config, id_map team, weekly_stats, target-season schedules) → Tasks 1, 5. §3.2 availability (era-normalized frac, position default, clamp, byes, missing-schedule degradation, rookie placeholders) → Task 1 (3 tests cover all). §3.3 `per_game=season/17` → Task 2 (constant `_GAMES`, closed-form test). §3.4 MC valuer + uniform-scaling reuse of `optimal_lineup_points/17` + factorization + exact-in-expectation → Task 2 (incl. the factorization-vs-bruteforce test). §3.5 pluggable valuer + deterministic per-roster seed → Task 3. §3.6 touched files → Tasks 4, 5. §4 tests: availability construction, closed-form, reduces-to-starters, factorization correctness, depth-rewarded, determinism, bye handling, valuer seam, CLI → Tasks 1–5. §5 decisions honored by construction. §6 future work untouched.

**Placeholder scan:** no TBD/TODO/"handle edge cases"; every code step is complete. The two prose verification notes (Step 2 real-data, write_partition signature caveat) point at concrete commands/files, not vague work.

**Type consistency:** `PlayerAvailability(p: dict[str,float], bye: dict[str,int])` with `p_week`/`bye_week` used identically in Tasks 1–3; `build_availability(weekly_stats, schedules, id_map, pool, *, season, lo, hi)` signature matches the §3.2 spec and the Task 5 call; `expected_season_points(roster, roster_slots, availability, *, n_sims, rng, weeks)` matches between Task 2 def, Task 3 `SeasonValuer.value`, and the Task 2 tests; `RosterValuer.value(roster, roster_slots)` is the single shared method across `StartersValuer`/`SeasonValuer`/the tournament's `_strategy_values`; `valuer=valuer` threading is identical across `run_tournament`/`tune_sigma`/`_strategy_values`. `_GAMES = 17` (season_value) and the `/17` per_game divisor agree.
