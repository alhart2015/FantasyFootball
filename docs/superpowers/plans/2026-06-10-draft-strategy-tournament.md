# Draft Strategy Comparison Harness (Slice 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a headless CLI tournament that simulates full snake drafts with a `DraftStrategy` in the hero seat against an ADP-bot field, scores the resulting rosters by their optimal starting lineup, and declares an empirical winner — plus a σ-tuning mode.

**Architecture:** Four pure-ish engine modules under `src/projections/draft/assistant/` — `roster_score` (value a roster), `opponent` (ADP-bot pick), `simulation` (one full draft), `tournament` (many seeded drafts → bootstrap CI → winner; plus σ sweep) — composed behind a dedicated CLI module + `scripts/` wrapper. Everything league-shape comes from `LeagueConfig`; nothing is hardcoded. Reuses the Slice 1 engine (`DraftStrategy`, `pick_timing.slot_for`, `DraftState`, `LogisticSurvival`, `roster_eligibility`) unchanged.

**Tech Stack:** Python 3.12, pandas (pyarrow-backed string dtype), numpy (`default_rng` bootstrap), pandera (existing `VorpTableSchema`), pydantic (`LeagueConfig`), pytest, mypy strict, ruff.

**Source spec:** `docs/superpowers/specs/2026-06-10-draft-strategy-tournament-design.md`

---

## File Structure

**Create (engine):**
- `src/projections/draft/assistant/roster_score.py` — `optimal_lineup_points(roster_rows, roster_slots)`; optimal starting-lineup value (spec §3.4).
- `src/projections/draft/assistant/opponent.py` — `bot_pick(available, rng, *, adp_jitter)`; pure noisy-ADP pick (spec §3.2).
- `src/projections/draft/assistant/simulation.py` — `simulate_draft(...)` + internal `_draft_picks(...)`; one full snake draft (spec §3.3).
- `src/projections/draft/assistant/tournament.py` — `_validate_pool`, `_strategy_values`, `_bootstrap_mean_ci`, `_paired_diff_ci`, `Interval`/`TournamentResult`/`SigmaTuningResult`, `run_tournament`, `tune_sigma` (spec §3.5–§3.6).
- `src/projections/draft/assistant/tournament_cli.py` — CLI core: load inputs, run a mode, render (spec §3.7).
- `scripts/draft_tournament.py` — thin `sys.exit(run())` wrapper.

**Create (tests):**
- `tests/test_draft/test_assistant_roster_score.py`
- `tests/test_draft/test_assistant_opponent.py`
- `tests/test_draft/test_assistant_simulation.py`
- `tests/test_draft/test_assistant_tournament.py`
- `tests/test_draft/test_assistant_tournament_cli.py`

**Modify:**
- `project_management.md` + `TODO.md` (Task 6 — status + next-direction update).

**Deliberately NOT modified:** `strategy.py`, `survival.py`, `pick_timing.py`, `state.py`, `roster_eligibility.py`, `schemas.py`, `cli.py` (the single-rec CLI). This slice consumes the Slice 1 surface; it does not change it. (Spec §3.7 named "`assistant/cli.py`" generically; the plan uses a dedicated `tournament_cli.py` so the recommend-CLI and tournament-CLI keep one responsibility each — a cohesion improvement noted here, not a scope change.)

**Note on Position/RosterSlot enums (read before coding):** `Position` ∈ {QB, RB, WR, TE, K, DST}. `RosterSlot` ∈ {QB, RB, WR, TE, FLEX, SUPER_FLEX, K, DST, BENCH, IR}. `roster_eligibility.py` exports `POSITION_SLOTS` (the six single-position slots), `FLEX_ELIGIBLE` (`{RB,WR,TE}`), `SUPER_FLEX_ELIGIBLE` (`{QB,RB,WR,TE}`). Use these — never re-define them.

---

## Task 1: Roster scoring — `optimal_lineup_points`

Value a completed roster by the points it would actually start (spec §3.4): single-position slots first, then `FLEX`, then `SUPER_FLEX` (ascending eligibility breadth), each taking the best unused eligible player; bench scores nothing.

**Files:**
- Create: `src/projections/draft/assistant/roster_score.py`
- Test: `tests/test_draft/test_assistant_roster_score.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_draft/test_assistant_roster_score.py
"""Tests for optimal starting-lineup scoring."""

from __future__ import annotations

import pandas as pd

from projections.draft.assistant.roster_score import optimal_lineup_points
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


def test_strand_case_fills_flex_before_super_flex() -> None:
    # Spec §3.4 counterexample: SUPER_FLEX grabbing RB first would strand the QB.
    roster = _roster([("00-0000001", "RB", 100.0), ("00-0000002", "QB", 90.0)])
    slots = {RosterSlot.FLEX: 1, RosterSlot.SUPER_FLEX: 1}
    assert optimal_lineup_points(roster, slots) == 190.0


def test_single_position_and_flex_pick_best() -> None:
    roster = _roster(
        [
            ("00-0000001", "RB", 100.0),
            ("00-0000002", "RB", 80.0),
            ("00-0000003", "WR", 90.0),
        ]
    )
    # RB slot takes best RB (100); FLEX takes best remaining eligible (WR 90 > RB 80).
    slots = {RosterSlot.RB: 1, RosterSlot.FLEX: 1}
    assert optimal_lineup_points(roster, slots) == 190.0


def test_bench_and_ir_score_nothing() -> None:
    roster = _roster([("00-0000001", "RB", 100.0), ("00-0000002", "RB", 80.0)])
    slots = {RosterSlot.RB: 1, RosterSlot.BENCH: 5, RosterSlot.IR: 1}
    assert optimal_lineup_points(roster, slots) == 100.0


def test_unfillable_slot_scores_partial() -> None:
    # K starting slot on a roster with no kicker (skill-only pool) → that slot = 0.
    roster = _roster([("00-0000001", "RB", 100.0)])
    slots = {RosterSlot.RB: 1, RosterSlot.K: 1}
    assert optimal_lineup_points(roster, slots) == 100.0


def test_tie_break_is_deterministic() -> None:
    # Equal points: which player fills which slot is gsis_id-stable, total invariant.
    roster = _roster(
        [
            ("00-0000002", "RB", 100.0),
            ("00-0000001", "RB", 100.0),
            ("00-0000003", "WR", 50.0),
        ]
    )
    slots = {RosterSlot.RB: 1, RosterSlot.FLEX: 1}
    # RB slot + FLEX both fillable by RB; best two RBs start (100+100); WR benched.
    assert optimal_lineup_points(roster, slots) == 200.0
    # Run twice — identical (no set-iteration nondeterminism).
    assert optimal_lineup_points(roster, slots) == 200.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_draft/test_assistant_roster_score.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'projections.draft.assistant.roster_score'`.

- [ ] **Step 3: Write the implementation**

```python
# src/projections/draft/assistant/roster_score.py
"""Value a completed roster by its optimal starting lineup (spec §3.4).

Fill order is load-bearing: single-position slots, then FLEX, then SUPER_FLEX
(ascending eligibility breadth). The eligibility sets are laminar, so this
restrictive-first greedy is optimal — no assignment solver needed. Filling a
wider slot first can strand a player and undercount.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from projections.draft.roster_eligibility import (
    FLEX_ELIGIBLE,
    POSITION_SLOTS,
    SUPER_FLEX_ELIGIBLE,
)
from projections.schemas import Position, RosterSlot

# Flex-type slots in ascending eligibility breadth (FLEX ⊂ SUPER_FLEX). Order matters.
_FLEX_SLOTS: tuple[tuple[RosterSlot, frozenset[Position]], ...] = (
    (RosterSlot.FLEX, FLEX_ELIGIBLE),
    (RosterSlot.SUPER_FLEX, SUPER_FLEX_ELIGIBLE),
)


def optimal_lineup_points(
    roster_rows: pd.DataFrame, roster_slots: Mapping[RosterSlot, int]
) -> float:
    """Sum the `season_mean_fpts` of the optimal legal starting lineup.

    `roster_rows` needs columns `gsis_id`, `position`, `season_mean_fpts`.
    Bench/IR slots contribute nothing; a starting slot no player can fill scores 0.
    """
    # Per-position points, best-first, deterministic gsis_id tie-break.
    ordered = roster_rows.sort_values(["season_mean_fpts", "gsis_id"], ascending=[False, True])
    by_pos: dict[Position, list[float]] = {}
    for pos in Position:
        by_pos[pos] = [
            float(v) for v in ordered.loc[ordered["position"] == pos.value, "season_mean_fpts"]
        ]
    cursor: dict[Position, int] = {pos: 0 for pos in Position}

    total = 0.0
    # 1) Single-position starting slots.
    for slot in POSITION_SLOTS:
        pos = Position(slot.value)
        for _ in range(roster_slots.get(slot, 0)):
            if cursor[pos] < len(by_pos[pos]):
                total += by_pos[pos][cursor[pos]]
                cursor[pos] += 1
    # 2) Flex tiers, narrowest first; each takes the best remaining eligible player.
    for slot, eligible in _FLEX_SLOTS:
        for _ in range(roster_slots.get(slot, 0)):
            best_pos: Position | None = None
            best_val = float("-inf")
            for pos in sorted(eligible, key=lambda p: p.value):  # sorted → deterministic
                if cursor[pos] < len(by_pos[pos]) and by_pos[pos][cursor[pos]] > best_val:
                    best_pos, best_val = pos, by_pos[pos][cursor[pos]]
            if best_pos is not None:
                total += best_val
                cursor[best_pos] += 1
    return total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_draft/test_assistant_roster_score.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Lint + type-check this file**

Run: `ruff check src/projections/draft/assistant/roster_score.py tests/test_draft/test_assistant_roster_score.py && ruff format --check src/projections/draft/assistant/roster_score.py && mypy src/projections/draft/assistant/roster_score.py`
Expected: no violations.

- [ ] **Step 6: Commit**

```bash
git add src/projections/draft/assistant/roster_score.py tests/test_draft/test_assistant_roster_score.py
git commit -m "feat(draft): optimal starting-lineup scoring (tournament Slice 2)"
```

---

## Task 2: ADP-bot — `bot_pick`

Pure noisy-ADP opponent (spec §3.2): lowest `consensus_adp + N(0, adp_jitter)`, null ADP → `+inf` (left to the hero), deterministic `gsis_id` tie-break. No roster argument.

**Files:**
- Create: `src/projections/draft/assistant/opponent.py`
- Test: `tests/test_draft/test_assistant_opponent.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_draft/test_assistant_opponent.py
"""Tests for the ADP-bot pick policy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.draft.assistant.opponent import bot_pick
from projections.schemas import _PYARROW_STR


def _available(rows: list[tuple[str, float | None]]) -> pd.DataFrame:
    """rows = [(gsis_id, consensus_adp_or_None), ...]."""
    return pd.DataFrame(
        {
            "gsis_id": pd.array([r[0] for r in rows], dtype=_PYARROW_STR),
            "consensus_adp": pd.array(
                [r[1] if r[1] is not None else pd.NA for r in rows], dtype=pd.Float64Dtype()
            ),
        }
    )


def test_zero_jitter_picks_lowest_adp() -> None:
    avail = _available([("00-0000001", 10.0), ("00-0000002", 3.0), ("00-0000003", 7.0)])
    rng = np.random.default_rng(0)
    assert bot_pick(avail, rng, adp_jitter=0.0) == "00-0000002"


def test_null_adp_left_for_hero_until_nothing_else() -> None:
    avail = _available([("00-0000001", None), ("00-0000002", 50.0)])
    # Even with big jitter, a finite ADP always beats +inf.
    for seed in range(20):
        rng = np.random.default_rng(seed)
        assert bot_pick(avail, rng, adp_jitter=10.0) == "00-0000002"


def test_all_null_falls_back_to_gsis_order() -> None:
    avail = _available([("00-0000002", None), ("00-0000001", None)])
    rng = np.random.default_rng(0)
    # All +inf → deterministic gsis_id tie-break (ascending).
    assert bot_pick(avail, rng, adp_jitter=5.0) == "00-0000001"


def test_deterministic_given_seed() -> None:
    avail = _available([("00-0000001", 5.0), ("00-0000002", 5.5), ("00-0000003", 6.0)])
    a = bot_pick(_available_copy(avail), np.random.default_rng(42), adp_jitter=3.0)
    b = bot_pick(_available_copy(avail), np.random.default_rng(42), adp_jitter=3.0)
    assert a == b


def _available_copy(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_draft/test_assistant_opponent.py -v`
Expected: FAIL — `ModuleNotFoundError: ... 'projections.draft.assistant.opponent'`.

- [ ] **Step 3: Write the implementation**

```python
# src/projections/draft/assistant/opponent.py
"""The ADP-bot: a non-hero seat's pick policy (spec §3.2).

Pure noisy-ADP. Realism comes from ADP itself (consensus ADP already spaces
positions like a real room), so the bot takes no roster argument — a
roster-eligibility filter would be a no-op under a shared bench anyway.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.schemas import GsisId, validate_gsis_id


def bot_pick(available: pd.DataFrame, rng: np.random.Generator, *, adp_jitter: float) -> GsisId:
    """Return the lowest noisy-ADP player among `available`.

    `available` needs columns `gsis_id` and `consensus_adp` (nullable Float64).
    Null ADP → treated as `+inf` (no market signal). Ties (incl. all-null) break
    on `gsis_id` ascending. `available` must be non-empty (caller guarantees it).
    """
    adp = available["consensus_adp"].to_numpy(dtype=float, na_value=np.inf)
    noisy = adp + rng.normal(0.0, adp_jitter, size=len(available))
    gsis = available["gsis_id"].to_numpy(dtype=str)
    # lexsort sorts by the LAST key first → primary noisy asc, secondary gsis asc.
    winner = int(np.lexsort((gsis, noisy))[0])
    return validate_gsis_id(str(gsis[winner]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_draft/test_assistant_opponent.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Lint + type-check**

Run: `ruff check src/projections/draft/assistant/opponent.py tests/test_draft/test_assistant_opponent.py && ruff format --check src/projections/draft/assistant/opponent.py && mypy src/projections/draft/assistant/opponent.py`
Expected: no violations.

- [ ] **Step 6: Commit**

```bash
git add src/projections/draft/assistant/opponent.py tests/test_draft/test_assistant_opponent.py
git commit -m "feat(draft): noisy-ADP opponent bot (tournament Slice 2)"
```

---

## Task 3: Draft simulation — `simulate_draft`

One full snake draft (spec §3.3): hero seat runs the `DraftStrategy`, every other seat runs `bot_pick`; deterministic given seed. Internal `_draft_picks` returns the full ordered pick list (so the paired-field property is testable); `simulate_draft` returns the hero's roster rows.

**Files:**
- Create: `src/projections/draft/assistant/simulation.py`
- Test: `tests/test_draft/test_assistant_simulation.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_draft/test_assistant_simulation.py
"""Tests for the full-draft simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.draft.assistant.simulation import _draft_picks, simulate_draft
from projections.draft.assistant.state import DraftState
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, Position, RosterSlot, Ruleset


def _config(n_teams: int = 2) -> LeagueConfig:
    # roster_size = sum of non-IR slots = 3 → 2 teams * 3 = 6 picks.
    return LeagueConfig(
        name="t",
        n_teams=n_teams,
        roster_slots={RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.BENCH: 1},
        ruleset=Ruleset.espn_ppr(),
    )


def _pool(n: int = 12) -> pd.DataFrame:
    # n players, descending fpts/vorp, ascending adp, alternating RB/WR.
    gsis = [f"00-00000{i:02d}" for i in range(1, n + 1)]
    return pd.DataFrame(
        {
            "gsis_id": pd.array(gsis, dtype=_PYARROW_STR),
            "position": pd.array(["RB" if i % 2 else "WR" for i in range(n)], dtype=_PYARROW_STR),
            "season_mean_fpts": [float(200 - i) for i in range(n)],
            "vorp": [float(100 - i) for i in range(n)],
            "replacement_fpts": [100.0] * n,
            "consensus_adp": pd.array([float(i + 1) for i in range(n)], dtype=pd.Float64Dtype()),
        }
    )


class _BestFpts:
    """Hero fake: draft the highest-season_mean_fpts not-yet-drafted player."""

    def recommend(self, state: DraftState, pool: pd.DataFrame, config: LeagueConfig) -> pd.DataFrame:
        avail = pool[~pool["gsis_id"].isin(state.drafted_ids)]
        return avail.sort_values(["season_mean_fpts", "gsis_id"], ascending=[False, True])


class _WorstFpts:
    """Hero fake: draft the lowest-season_mean_fpts not-yet-drafted player."""

    def recommend(self, state: DraftState, pool: pd.DataFrame, config: LeagueConfig) -> pd.DataFrame:
        avail = pool[~pool["gsis_id"].isin(state.drafted_ids)]
        return avail.sort_values(["season_mean_fpts", "gsis_id"], ascending=[True, True])


def test_hero_gets_exactly_roster_size_picks() -> None:
    cfg = _config(n_teams=2)
    rng = np.random.default_rng(0)
    roster = simulate_draft(_BestFpts(), my_slot=1, pool=_pool(), config=cfg, adp_jitter=2.0, rng=rng)
    assert len(roster) == cfg.roster_size  # 3


def test_hero_best_fpts_drafts_the_top_players_slot1() -> None:
    # my_slot=1 picks first each round; _BestFpts always grabs the global best left.
    cfg = _config(n_teams=2)
    rng = np.random.default_rng(0)
    roster = simulate_draft(_BestFpts(), my_slot=1, pool=_pool(), config=cfg, adp_jitter=0.0, rng=rng)
    # Pick 1 = best (00-000001, 200). Bot at pick 2 (jitter 0) takes lowest adp among rest.
    assert "00-0000001" in set(roster["gsis_id"])


def test_determinism_same_seed_same_roster() -> None:
    cfg = _config()
    r1 = simulate_draft(_BestFpts(), my_slot=2, pool=_pool(), config=cfg, adp_jitter=3.0,
                        rng=np.random.default_rng(7))
    r2 = simulate_draft(_BestFpts(), my_slot=2, pool=_pool(), config=cfg, adp_jitter=3.0,
                        rng=np.random.default_rng(7))
    assert list(r1["gsis_id"]) == list(r2["gsis_id"])


def test_different_seed_generally_differs() -> None:
    cfg = _config()
    r1 = simulate_draft(_BestFpts(), my_slot=2, pool=_pool(), config=cfg, adp_jitter=5.0,
                        rng=np.random.default_rng(1))
    r2 = simulate_draft(_BestFpts(), my_slot=2, pool=_pool(), config=cfg, adp_jitter=5.0,
                        rng=np.random.default_rng(2))
    # Not a hard guarantee, but with this pool + jitter the fields diverge.
    assert list(r1["gsis_id"]) != list(r2["gsis_id"])


def test_paired_field_identical_before_hero_diverges() -> None:
    # my_slot=2: pick #1 is a bot; same seed → identical regardless of hero strategy.
    cfg = _config()
    picks_a = _draft_picks(_BestFpts(), my_slot=2, pool=_pool(), config=cfg, adp_jitter=4.0,
                           rng=np.random.default_rng(11))
    picks_b = _draft_picks(_WorstFpts(), my_slot=2, pool=_pool(), config=cfg, adp_jitter=4.0,
                           rng=np.random.default_rng(11))
    assert picks_a[0] == picks_b[0]  # the pre-divergence bot pick
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_draft/test_assistant_simulation.py -v`
Expected: FAIL — `ModuleNotFoundError: ... 'projections.draft.assistant.simulation'`.

- [ ] **Step 3: Write the implementation**

```python
# src/projections/draft/assistant/simulation.py
"""Simulate one full snake draft (spec §3.3).

The hero seat runs a `DraftStrategy` (re-asked from the current `DraftState` at
each of its picks); every other seat runs the noisy-ADP `bot_pick`. One seeded
RNG drives all bot noise, so same seed + same strategy ⇒ identical hero roster.
Assumes a validated pool (size + ADP signal checked at the tournament entry).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.draft.assistant.opponent import bot_pick
from projections.draft.assistant.pick_timing import slot_for
from projections.draft.assistant.state import DraftState
from projections.draft.assistant.strategy import DraftStrategy
from projections.draft.league_config import LeagueConfig
from projections.schemas import GsisId, Position, validate_gsis_id


def _draft_picks(
    strategy: DraftStrategy,
    my_slot: int,
    pool: pd.DataFrame,
    config: LeagueConfig,
    *,
    adp_jitter: float,
    rng: np.random.Generator,
) -> list[GsisId]:
    """Run the draft; return every pick's gsis_id in absolute pick order."""
    pos_by_id = {str(g): str(p) for g, p in zip(pool["gsis_id"], pool["position"], strict=True)}
    n_teams = config.n_teams
    total_picks = n_teams * config.roster_size

    drafted: list[GsisId] = []
    drafted_set: set[GsisId] = set()
    my_roster: list[Position] = []

    for pick_number in range(1, total_picks + 1):
        if slot_for(pick_number, n_teams) == my_slot:
            state = DraftState(
                my_slot=my_slot,
                n_teams=n_teams,
                rounds=config.roster_size,
                picks=tuple(drafted),
                my_roster=tuple(my_roster),
            )
            rec = strategy.recommend(state, pool, config)
            if rec.empty:
                raise ValueError(
                    f"strategy returned no eligible pick at pick {pick_number}; "
                    "pool too small or fully ineligible (should be caught upstream)"
                )
            gid = validate_gsis_id(str(rec.iloc[0]["gsis_id"]))
            my_roster.append(Position(pos_by_id[gid]))
        else:
            available = pool[~pool["gsis_id"].isin(drafted_set)]
            gid = bot_pick(available, rng, adp_jitter=adp_jitter)
        drafted.append(gid)
        drafted_set.add(gid)

    return drafted


def simulate_draft(
    strategy: DraftStrategy,
    my_slot: int,
    pool: pd.DataFrame,
    config: LeagueConfig,
    *,
    adp_jitter: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Run one full draft; return the hero's drafted rows (a sub-frame of `pool`)."""
    all_picks = _draft_picks(
        strategy, my_slot, pool, config, adp_jitter=adp_jitter, rng=rng
    )
    mine = {
        pick for i, pick in enumerate(all_picks) if slot_for(i + 1, config.n_teams) == my_slot
    }
    return pool[pool["gsis_id"].isin(mine)].copy()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_draft/test_assistant_simulation.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Lint + type-check**

Run: `ruff check src/projections/draft/assistant/simulation.py tests/test_draft/test_assistant_simulation.py && ruff format --check src/projections/draft/assistant/simulation.py && mypy src/projections/draft/assistant/simulation.py`
Expected: no violations.

- [ ] **Step 6: Commit**

```bash
git add src/projections/draft/assistant/simulation.py tests/test_draft/test_assistant_simulation.py
git commit -m "feat(draft): full snake-draft simulator (tournament Slice 2)"
```

---

## Task 4: Tournament + σ-tuning + stats

The comparison engine (spec §3.5–§3.6): validate the pool once, run each strategy over `n_seeds` paired drafts, summarize with a percentile bootstrap, declare a winner on the paired difference; `tune_sigma` sweeps σ. The bootstrap mirrors `adoption_gate.py` (`default_rng` + `np.percentile(_, [2.5, 97.5])`).

**Files:**
- Create: `src/projections/draft/assistant/tournament.py`
- Test: `tests/test_draft/test_assistant_tournament.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_draft/test_assistant_tournament.py
"""Tests for the strategy tournament + sigma tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.draft.assistant.state import DraftState
from projections.draft.assistant.tournament import (
    _paired_diff_ci,
    _validate_pool,
    run_tournament,
    tune_sigma,
)
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, RosterSlot, Ruleset


def _config(n_teams: int = 2) -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=n_teams,
        roster_slots={RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.BENCH: 1},
        ruleset=Ruleset.espn_ppr(),
    )


def _pool(n: int = 12) -> pd.DataFrame:
    gsis = [f"00-00000{i:02d}" for i in range(1, n + 1)]
    return pd.DataFrame(
        {
            "gsis_id": pd.array(gsis, dtype=_PYARROW_STR),
            "position": pd.array(["RB" if i % 2 else "WR" for i in range(n)], dtype=_PYARROW_STR),
            "season_mean_fpts": [float(200 - i) for i in range(n)],
            "vorp": [float(100 - i) for i in range(n)],
            "replacement_fpts": [100.0] * n,
            "consensus_adp": pd.array([float(i + 1) for i in range(n)], dtype=pd.Float64Dtype()),
        }
    )


class _BestFpts:
    def recommend(self, state: DraftState, pool: pd.DataFrame, config: LeagueConfig) -> pd.DataFrame:
        avail = pool[~pool["gsis_id"].isin(state.drafted_ids)]
        return avail.sort_values(["season_mean_fpts", "gsis_id"], ascending=[False, True])


class _WorstFpts:
    def recommend(self, state: DraftState, pool: pd.DataFrame, config: LeagueConfig) -> pd.DataFrame:
        avail = pool[~pool["gsis_id"].isin(state.drafted_ids)]
        return avail.sort_values(["season_mean_fpts", "gsis_id"], ascending=[True, True])


def test_validate_pool_rejects_all_null_adp() -> None:
    pool = _pool()
    pool["consensus_adp"] = pd.array([pd.NA] * len(pool), dtype=pd.Float64Dtype())
    with pytest.raises(ValueError, match="consensus_adp"):
        _validate_pool(pool, _config())


def test_validate_pool_rejects_too_small_pool() -> None:
    cfg = _config(n_teams=2)  # needs 6 players
    with pytest.raises(ValueError, match="need >="):
        _validate_pool(_pool(n=4), cfg)


def test_validate_pool_accepts_valid() -> None:
    _validate_pool(_pool(), _config())  # no raise


def test_paired_diff_ci_constant_edge_excludes_zero() -> None:
    a = np.array([10.0, 12.0, 11.0, 9.0, 13.0] * 4)
    b = a - 3.0  # A beats B by a constant 3 every paired seed
    ci = _paired_diff_ci(a, b, n_bootstrap=500, seed=0)
    assert ci.point == pytest.approx(3.0)
    assert ci.lo_95 > 0


def test_paired_diff_ci_zero_edge_brackets_zero() -> None:
    a = np.array([10.0, 12.0, 11.0, 9.0, 13.0] * 4)
    ci = _paired_diff_ci(a, a.copy(), n_bootstrap=500, seed=0)
    assert ci.lo_95 <= 0 <= ci.hi_95


def test_run_tournament_declares_better_strategy() -> None:
    result = run_tournament(
        {"best": _BestFpts(), "worst": _WorstFpts()},
        pool=_pool(),
        config=_config(),
        my_slot=1,
        n_seeds=40,
        adp_jitter=3.0,
        base_seed=0,
    )
    assert result.summaries["best"].point > result.summaries["worst"].point
    assert result.winner == "best"
    assert result.diff is not None and result.diff.lo_95 > 0


def test_tune_sigma_returns_argmax() -> None:
    result = tune_sigma(
        [1.0, 8.0],
        pool=_pool(),
        config=_config(),
        my_slot=1,
        n_seeds=20,
        adp_jitter=3.0,
        base_seed=0,
    )
    assert len(result.grid) == 2
    assert result.best_sigma in (1.0, 8.0)
    assert result.best_sigma == max(result.grid, key=lambda r: r[1])[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_draft/test_assistant_tournament.py -v`
Expected: FAIL — `ModuleNotFoundError: ... 'projections.draft.assistant.tournament'`.

- [ ] **Step 3: Write the implementation**

```python
# src/projections/draft/assistant/tournament.py
"""Compare draft strategies empirically (spec §3.5–§3.6).

Run each strategy over many seeded drafts against an ADP field, score the hero
roster by its optimal starting lineup, and declare a winner on the paired
per-seed difference (percentile bootstrap, mirroring adoption_gate.py). The same
seed index gives every strategy the same bot field — the paired counterfactual.
`tune_sigma` sweeps the survival σ the same way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from projections.draft.assistant.roster_score import optimal_lineup_points
from projections.draft.assistant.simulation import simulate_draft
from projections.draft.assistant.strategy import DraftStrategy, NowOrNeverStrategy
from projections.draft.assistant.survival import LogisticSurvival
from projections.draft.league_config import LeagueConfig

_N_BOOTSTRAP = 1000
_CI_PCTILES = (2.5, 97.5)


@dataclass(frozen=True)
class Interval:
    """A point estimate with a central 95% bootstrap CI."""

    point: float
    lo_95: float
    hi_95: float


@dataclass(frozen=True)
class TournamentResult:
    """Per-strategy mean starting-lineup points, the top-two paired diff, the winner."""

    summaries: dict[str, Interval]
    diff: Interval | None  # top-vs-second paired difference; None if <2 strategies
    winner: str | None  # named iff diff.lo_95 > 0 (CI excludes 0)
    n_seeds: int
    adp_jitter: float
    base_seed: int
    my_slot: int


@dataclass(frozen=True)
class SigmaTuningResult:
    """(sigma, mean hero value) grid + the argmax."""

    grid: list[tuple[float, float]]
    best_sigma: float
    n_seeds: int
    adp_jitter: float
    base_seed: int
    my_slot: int


def _validate_pool(pool: pd.DataFrame, config: LeagueConfig) -> None:
    """Hard preconditions shared by both entry points (spec §3.1, §3.3)."""
    if "consensus_adp" not in pool.columns or bool(pool["consensus_adp"].isna().all()):
        raise ValueError(
            "pool has no consensus_adp signal; the tournament needs market ADP to drive the field"
        )
    need = config.n_teams * config.roster_size
    if len(pool) < need:
        raise ValueError(f"pool has {len(pool)} players; need >= {need} to fill a full draft")


def _bootstrap(values: np.ndarray, *, seed: int) -> Interval:
    """Percentile-bootstrap CI of the mean of `values`."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = v.shape[0]
    boot = np.array([v[rng.integers(0, n, size=n)].mean() for _ in range(_N_BOOTSTRAP)])
    lo, hi = np.percentile(boot, _CI_PCTILES)
    return Interval(point=float(v.mean()), lo_95=float(lo), hi_95=float(hi))


def _paired_diff_ci(a: np.ndarray, b: np.ndarray, *, n_bootstrap: int = _N_BOOTSTRAP, seed: int) -> Interval:
    """Percentile-bootstrap CI of the paired mean difference `a - b`."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = d.shape[0]
    boot = np.array([d[rng.integers(0, n, size=n)].mean() for _ in range(n_bootstrap)])
    lo, hi = np.percentile(boot, _CI_PCTILES)
    return Interval(point=float(d.mean()), lo_95=float(lo), hi_95=float(hi))


def _strategy_values(
    strategy: DraftStrategy,
    pool: pd.DataFrame,
    config: LeagueConfig,
    *,
    my_slot: int,
    n_seeds: int,
    adp_jitter: float,
    base_seed: int,
) -> np.ndarray:
    """Optimal-lineup points of the hero roster for each paired seed."""
    out = np.empty(n_seeds, dtype=np.float64)
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        roster = simulate_draft(strategy, my_slot, pool, config, adp_jitter=adp_jitter, rng=rng)
        out[s] = optimal_lineup_points(roster, config.roster_slots)
    return out


def run_tournament(
    strategies: Mapping[str, DraftStrategy],
    pool: pd.DataFrame,
    config: LeagueConfig,
    *,
    my_slot: int,
    n_seeds: int,
    adp_jitter: float,
    base_seed: int,
) -> TournamentResult:
    """Compare `strategies` over `n_seeds` paired drafts; declare a winner."""
    _validate_pool(pool, config)
    values = {
        name: _strategy_values(
            strat, pool, config, my_slot=my_slot, n_seeds=n_seeds,
            adp_jitter=adp_jitter, base_seed=base_seed,
        )
        for name, strat in strategies.items()
    }
    summaries = {name: _bootstrap(v, seed=base_seed) for name, v in values.items()}
    ranked = sorted(summaries, key=lambda n: summaries[n].point, reverse=True)

    diff: Interval | None = None
    winner: str | None = ranked[0] if ranked else None
    if len(ranked) >= 2:
        top, second = ranked[0], ranked[1]
        diff = _paired_diff_ci(values[top], values[second], seed=base_seed)
        winner = top if diff.lo_95 > 0 else None

    return TournamentResult(
        summaries=summaries, diff=diff, winner=winner, n_seeds=n_seeds,
        adp_jitter=adp_jitter, base_seed=base_seed, my_slot=my_slot,
    )


def tune_sigma(
    sigma_grid: Sequence[float],
    pool: pd.DataFrame,
    config: LeagueConfig,
    *,
    my_slot: int,
    n_seeds: int,
    adp_jitter: float,
    base_seed: int,
) -> SigmaTuningResult:
    """Sweep the survival σ for NowOrNeverStrategy; return the (σ, mean value) grid + argmax."""
    _validate_pool(pool, config)
    grid: list[tuple[float, float]] = []
    for sigma in sigma_grid:
        strat = NowOrNeverStrategy(LogisticSurvival(sigma=float(sigma)))
        vals = _strategy_values(
            strat, pool, config, my_slot=my_slot, n_seeds=n_seeds,
            adp_jitter=adp_jitter, base_seed=base_seed,
        )
        grid.append((float(sigma), float(vals.mean())))
    best_sigma = max(grid, key=lambda r: r[1])[0]
    return SigmaTuningResult(
        grid=grid, best_sigma=best_sigma, n_seeds=n_seeds,
        adp_jitter=adp_jitter, base_seed=base_seed, my_slot=my_slot,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_draft/test_assistant_tournament.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Lint + type-check**

Run: `ruff check src/projections/draft/assistant/tournament.py tests/test_draft/test_assistant_tournament.py && ruff format --check src/projections/draft/assistant/tournament.py && mypy src/projections/draft/assistant/tournament.py`
Expected: no violations. (If mypy flags the list-comprehension `np.array([...])` element type, annotate the comprehension result or use an explicit loop — keep it strict-clean.)

- [ ] **Step 6: Commit**

```bash
git add src/projections/draft/assistant/tournament.py tests/test_draft/test_assistant_tournament.py
git commit -m "feat(draft): strategy tournament + sigma tuning (Slice 2)"
```

---

## Task 5: CLI + script wrapper

The user-facing surface (spec §3.7): a `compare` mode and a `tune-sigma` mode over a consensus VORP parquet + a `LeagueConfig`. Mirrors the Slice 1 `cli.py` conventions (load + validate, build strategy, render fixed-width table).

**Files:**
- Create: `src/projections/draft/assistant/tournament_cli.py`
- Create: `scripts/draft_tournament.py`
- Test: `tests/test_draft/test_assistant_tournament_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_draft/test_assistant_tournament_cli.py
"""Smoke tests for the tournament CLI (both modes, end-to-end)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from projections.draft.assistant.tournament_cli import run
from projections.schemas import _PYARROW_STR


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    n = 24
    gsis = [f"00-00000{i:02d}" for i in range(1, n + 1)]  # 7 digits after dash (\d{2}-\d{7})
    pool = pd.DataFrame(
        {
            "gsis_id": pd.array(gsis, dtype=_PYARROW_STR),
            "position": pd.array(["RB" if i % 2 else "WR" for i in range(n)], dtype=_PYARROW_STR),
            "season_mean_fpts": [float(200 - i) for i in range(n)],
            "vorp": [float(100 - i) for i in range(n)],
            "replacement_fpts": [100.0] * n,
            "consensus_adp": pd.array([float(i + 1) for i in range(n)], dtype=pd.Float64Dtype()),
        }
    )
    vorp_path = tmp_path / "vorp.parquet"
    pool.to_parquet(vorp_path)

    cfg = {
        "name": "test",
        "n_teams": 4,
        "roster_slots": {"RB": 1, "WR": 1, "FLEX": 1, "BENCH": 2},
        "ruleset": "espn_ppr",
    }
    cfg_path = tmp_path / "league.json"
    cfg_path.write_text(json.dumps(cfg))
    return vorp_path, cfg_path


def test_compare_mode_runs(tmp_path: Path, capsys) -> None:
    vorp_path, cfg_path = _write_inputs(tmp_path)
    code = run(
        [
            "--vorp-table", str(vorp_path),
            "--league-config", str(cfg_path),
            "--my-slot", "2",
            "--seeds", "10",
            "--seed", "0",
            "compare",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "now_or_never" in out and "raw_vorp" in out


def test_tune_sigma_mode_runs(tmp_path: Path, capsys) -> None:
    vorp_path, cfg_path = _write_inputs(tmp_path)
    code = run(
        [
            "--vorp-table", str(vorp_path),
            "--league-config", str(cfg_path),
            "--my-slot", "2",
            "--seeds", "8",
            "--seed", "0",
            "tune-sigma",
            "--sigma-grid", "2,4",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "recommended" in out.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_draft/test_assistant_tournament_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: ... 'projections.draft.assistant.tournament_cli'`.

- [ ] **Step 3: Write the CLI core**

```python
# src/projections/draft/assistant/tournament_cli.py
"""CLI core for the strategy comparison harness (spec §3.7). scripts/ wraps this.

Two modes over a consensus VORP parquet + a LeagueConfig:
  compare    — run the registered strategies, print per-strategy CI + winner.
  tune-sigma — sweep the survival σ, print the grid + recommended σ.
The --league-config MUST match the ruleset the VORP table was built under
(the parquet carries no ruleset column to verify it — spec §3.1).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.draft.assistant.strategy import NowOrNeverStrategy, RawVorpStrategy
from projections.draft.assistant.survival import LogisticSurvival, default_sigma
from projections.draft.assistant.tournament import (
    TournamentResult,
    SigmaTuningResult,
    run_tournament,
    tune_sigma,
)
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, VorpTableSchema


def _load_pool(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    return VorpTableSchema.validate(df)


def _load_config(path: Path) -> LeagueConfig:
    return LeagueConfig.model_validate_json(path.read_text())


def _default_sigma_grid(n_teams: int) -> list[float]:
    base = default_sigma(n_teams)
    return [round(f * base, 3) for f in (1 / 3, 1 / 2, 2 / 3, 1.0, 4 / 3)]


def format_compare(result: TournamentResult) -> str:
    lines = [f"Strategy tournament — {result.n_seeds} seeds, my_slot={result.my_slot}, "
             f"adp_jitter={result.adp_jitter:.2f}, base_seed={result.base_seed}",
             f"{'STRATEGY':<16} {'MEAN':>9} {'95% CI':>22}"]
    for name, ci in sorted(result.summaries.items(), key=lambda kv: kv[1].point, reverse=True):
        lines.append(f"{name:<16} {ci.point:>9.2f}  [{ci.lo_95:>8.2f}, {ci.hi_95:>8.2f}]")
    if result.diff is not None:
        lines.append(
            f"\nTop-two paired diff: {result.diff.point:+.2f} "
            f"[{result.diff.lo_95:+.2f}, {result.diff.hi_95:+.2f}]"
        )
    lines.append(f"Winner: {result.winner if result.winner else 'no separation (CI brackets 0)'}")
    return "\n".join(lines)


def format_tune(result: SigmaTuningResult) -> str:
    lines = [f"Sigma tuning — {result.n_seeds} seeds, my_slot={result.my_slot}, "
             f"adp_jitter={result.adp_jitter:.2f}",
             f"{'SIGMA':>8} {'MEAN':>9}"]
    for sigma, mean in result.grid:
        lines.append(f"{sigma:>8.3f} {mean:>9.2f}")
    lines.append(f"\nRecommended sigma: {result.best_sigma:.3f}")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Draft strategy comparison harness.")
    p.add_argument("--vorp-table", type=Path, required=True,
                   help="Consensus VORP parquet (generate_vorp_table.py --source consensus).")
    p.add_argument("--league-config", type=Path, required=True,
                   help="LeagueConfig JSON — must match the table's ruleset (spec §3.1).")
    p.add_argument("--my-slot", type=int, required=True, help="Hero draft slot (1-based).")
    p.add_argument("--seeds", type=int, default=200, help="Paired draft sims per strategy.")
    p.add_argument("--adp-jitter", type=float, default=None,
                   help="Bot ADP noise SD in picks (default ~2/3 of a round).")
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed (reproducibility).")
    sub = p.add_subparsers(dest="mode", required=True)
    cmp = sub.add_parser("compare", help="Compare now_or_never vs raw_vorp.")
    cmp.add_argument("--strategy-sigma", type=float, default=None,
                     help="Survival sigma for now_or_never (default ~2/3 of a round).")
    tune = sub.add_parser("tune-sigma", help="Sweep survival sigma for now_or_never.")
    tune.add_argument("--sigma-grid", type=str, default=None,
                      help="Comma-separated sigmas (default centered on 2/3*n_teams).")
    return p.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    pool = _load_pool(args.vorp_table)
    config = _load_config(args.league_config)
    jitter = default_sigma(config.n_teams) if args.adp_jitter is None else args.adp_jitter

    if args.mode == "compare":
        sigma = default_sigma(config.n_teams) if args.strategy_sigma is None else args.strategy_sigma
        result = run_tournament(
            {"now_or_never": NowOrNeverStrategy(LogisticSurvival(sigma=sigma)),
             "raw_vorp": RawVorpStrategy()},
            pool=pool, config=config, my_slot=args.my_slot,
            n_seeds=args.seeds, adp_jitter=jitter, base_seed=args.seed,
        )
        print(format_compare(result))
        return 0

    grid = (
        _default_sigma_grid(config.n_teams)
        if args.sigma_grid is None
        else [float(x) for x in args.sigma_grid.split(",")]
    )
    tuned = tune_sigma(
        grid, pool=pool, config=config, my_slot=args.my_slot,
        n_seeds=args.seeds, adp_jitter=jitter, base_seed=args.seed,
    )
    print(format_tune(tuned))
    return 0
```

- [ ] **Step 4: Write the script wrapper**

```python
# scripts/draft_tournament.py
"""CLI wrapper for the draft strategy tournament. See projections.draft.assistant.tournament_cli."""

from __future__ import annotations

import sys

from projections.draft.assistant.tournament_cli import run

if __name__ == "__main__":
    sys.exit(run())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_draft/test_assistant_tournament_cli.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Lint + type-check**

Run: `ruff check src/projections/draft/assistant/tournament_cli.py scripts/draft_tournament.py tests/test_draft/test_assistant_tournament_cli.py && ruff format --check src/projections/draft/assistant/tournament_cli.py scripts/draft_tournament.py && mypy src/projections/draft/assistant/tournament_cli.py scripts/draft_tournament.py`
Expected: no violations. (mypy: `capsys` in tests is untyped — that's a test file; if a test signature trips strict mode, annotate `capsys: pytest.CaptureFixture[str]`.)

- [ ] **Step 7: Commit**

```bash
git add src/projections/draft/assistant/tournament_cli.py scripts/draft_tournament.py tests/test_draft/test_assistant_tournament_cli.py
git commit -m "feat(draft): tournament CLI + script wrapper (Slice 2)"
```

---

## Task 6: Full gate run + League-driven test + docs

Prove the whole slice green under the project bar, add the league-driven guard test the spec calls for, and update the running docs.

**Files:**
- Test: `tests/test_draft/test_assistant_tournament.py` (append one test)
- Modify: `project_management.md`, `TODO.md`

- [ ] **Step 1: Add the league-driven guard test**

Append to `tests/test_draft/test_assistant_tournament.py` (spec §4 "League-driven" — same pool, two roster shapes, both score with no code change):

```python
def test_league_driven_two_roster_shapes_same_pool() -> None:
    pool = _pool(n=24)

    skill = LeagueConfig(
        name="skill", n_teams=2,
        roster_slots={RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.FLEX: 1, RosterSlot.BENCH: 1},
        ruleset=Ruleset.espn_ppr(),
    )
    superflex = LeagueConfig(
        name="sf", n_teams=2,
        roster_slots={RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.SUPER_FLEX: 1, RosterSlot.BENCH: 1},
        ruleset=Ruleset.espn_ppr(),
    )

    for cfg in (skill, superflex):
        result = run_tournament(
            {"best": _BestFpts(), "worst": _WorstFpts()},
            pool=pool, config=cfg, my_slot=1, n_seeds=10, adp_jitter=2.0, base_seed=0,
        )
        assert set(result.summaries) == {"best", "worst"}
        assert result.summaries["best"].point >= result.summaries["worst"].point
```

- [ ] **Step 2: Run the new test, then the whole assistant + draft suite**

Run: `pytest tests/test_draft/test_assistant_tournament.py::test_league_driven_two_roster_shapes_same_pool -v`
Expected: PASS.

Run: `pytest tests/test_draft -v`
Expected: all pass (Slice 1 tests + the 5 new test modules).

- [ ] **Step 3: Full project gate**

Run each and fix every failure before proceeding:

```bash
pytest -q
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: `pytest` all green except the single pre-existing TODO #40 failure (`tests/backtest/test_backtest_smoke.py::test_backtest_smoke_one_cell`, WR feature build — unrelated to this branch; confirm it is the only failure and that `git diff --stat origin/main -- src/projections/features src/projections/backtest` shows no changes from this branch). `mypy`/`ruff`/`ruff format` clean.

- [ ] **Step 4: Smoke the real CLI (optional but recommended)**

If a real consensus VORP parquet exists (e.g. `data/.../consensus_vorp.parquet`) and `configs/league_espn_ppr_12team_skill.json` is present:

```bash
python scripts/draft_tournament.py --vorp-table <consensus_vorp.parquet> \
  --league-config configs/league_espn_ppr_12team_skill.json --my-slot 6 --seeds 50 compare
```

Expected: a per-strategy table + winner line. (Skip if no real parquet is checked out — the CLI smoke test already covers the path.)

- [ ] **Step 5: Update `project_management.md`**

Add a top entry dated 2026-06-10 summarizing: Slice 2 (strategy comparison harness) shipped on `feat/draft-strategy-tournament`; what shipped (`roster_score`, `opponent`, `simulation`, `tournament`, `tournament_cli` + `scripts/draft_tournament.py`); key decisions (hero-vs-ADP-field, optimal-lineup metric, paired-seed bootstrap, pure-noisy-ADP bots, league-driven, no new schema, auction seam); gate results; next direction (Slice 3 Streamlit UI; the survival-model conditional refinement is now empirically testable via this harness).

- [ ] **Step 6: Update `TODO.md`**

Under TODO #38's Draft Assistant bullet, mark **Slice 2 — strategy comparison harness — ✅ DONE** (branch `feat/draft-strategy-tournament`; spec/plan paths), with the usage line:
`python scripts/draft_tournament.py --vorp-table <consensus_vorp.parquet> --league-config <league.json> --my-slot N [--seeds K] {compare|tune-sigma}`. Note Slice 3 (Streamlit UI) remains the last Draft Assistant slice, and that σ can now be tuned empirically.

- [ ] **Step 7: Commit**

```bash
git add tests/test_draft/test_assistant_tournament.py project_management.md TODO.md
git commit -m "test(draft): league-driven guard + docs for tournament (Slice 2)"
```

---

## Self-Review (completed by plan author)

**Spec coverage:** §3.1 inputs (pool/config load + `_validate_pool` all-null + sufficiency) → Tasks 4, 5. §3.2 bot → Task 2. §3.3 simulation + paired field + pool-sufficiency-assumed → Task 3 (+ Task 4 validate). §3.4 scoring incl. strand case + tie-break + unfillable slot → Task 1. §3.5 tournament/bootstrap/winner → Task 4. §3.6 σ-tuning → Task 4. §3.7 CLI both modes → Task 5. §4 tests: pick-order, determinism, paired-field, bot-policy, pool-exhaustion (`_validate_pool`) + unfillable-slot (Task 1) + all-null-ADP (Task 4), adp_jitter-vs-σ independence (covered structurally: `RawVorpStrategy` invariant to σ — `run_tournament` only feeds σ into `NowOrNeverStrategy`; `simulate_draft` passes only `adp_jitter` to bots), `optimal_lineup_points`, paired-difference stat, σ-tuning, CLI smoke, league-driven → Tasks 1–6. §5 decisions are honored by construction. §6 future work untouched (correct).

**Note on the one structural test:** the spec's "`adp_jitter` vs σ independence" is enforced by code shape (separate params, separate consumers) rather than a single assertion; the `_BestFpts`/`_WorstFpts` fakes carry no σ and the real `RawVorpStrategy` has none, so no extra test is required — call this out for the executor rather than writing a vacuous test.

**Placeholder scan:** no TBD/TODO/"handle edge cases"; every code step is complete.

**Type consistency:** `Interval(point, lo_95, hi_95)` used uniformly for summaries and diffs; `run_tournament` returns `TournamentResult.summaries[name].point` (matches the CLI's `ci.point` and the test's `result.summaries["best"].point`); `tune_sigma` returns `SigmaTuningResult.grid: list[tuple[float,float]]` + `best_sigma` (matches CLI `format_tune` and tests); `bot_pick(available, rng, *, adp_jitter)`, `simulate_draft(strategy, my_slot, pool, config, *, adp_jitter, rng)`, `_draft_picks` same signature, and `_strategy_values`/`run_tournament`/`tune_sigma` thread `my_slot/n_seeds/adp_jitter/base_seed` identically across Tasks 3–5.
