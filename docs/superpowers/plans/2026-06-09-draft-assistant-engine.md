# Draft Assistant Engine (Slice 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a headless, pure-Python live-draft recommendation engine over the consensus VORP table — a pluggable `DraftStrategy` (now-or-never + raw-VORP control) with snake pick-timing, an ADP survival model, roster-need awareness, and a CLI.

**Architecture:** New subpackage `src/projections/draft/assistant/`. The engine is gsis-id-native and consumes the existing consensus-fed `VorpTableSchema` (change source, not math). Slot↔position eligibility is promoted out of `_pool.py`'s private symbols into a shared `roster_eligibility.py`. Strategies share a `_finalize` step that filters to roster-eligible positions, tags the scale-free starting-need tier, and applies the deterministic final ordering. The CLI core lives in `assistant/cli.py` (testable); `scripts/draft_assistant.py` is a thin wrapper.

**Tech Stack:** Python 3.12, pandas, pandera (`pandera.pandas`), pydantic (`LeagueConfig`), numpy. No new dependencies (logistic survival uses `math`, not scipy).

**Source spec:** `docs/superpowers/specs/2026-06-09-draft-assistant-engine-design.md`

---

## File Structure

**New files:**
- `src/projections/draft/roster_eligibility.py` — slot↔position eligibility sets (promoted from `_pool.py`) + greedy allocation → `eligible_positions(roster_slots, my_roster) -> dict[Position, bool]` (value = `fills_starting_slot`).
- `src/projections/draft/assistant/__init__.py` — subpackage public exports.
- `src/projections/draft/assistant/pick_timing.py` — pure snake-order functions.
- `src/projections/draft/assistant/survival.py` — `SurvivalModel` Protocol + `LogisticSurvival` + `default_sigma`.
- `src/projections/draft/assistant/state.py` — `DraftState` dataclass + `load_draft_state`.
- `src/projections/draft/assistant/strategy.py` — `DraftStrategy` Protocol, `RawVorpStrategy`, `NowOrNeverStrategy`, `_finalize`.
- `src/projections/draft/assistant/cli.py` — `generate_recommendation`, `format_table`, `run`.
- `scripts/draft_assistant.py` — thin CLI wrapper.

**Modified files:**
- `src/projections/schemas.py` — add `RecommendationSchema`.
- `src/projections/draft/_pool.py` — import eligibility sets from `roster_eligibility` (behavior unchanged).
- `src/projections/draft/__init__.py` — re-export the assistant entry points.

**New test files:**
- `tests/test_schemas/test_recommendation_schema.py`
- `tests/test_draft/test_roster_eligibility.py`
- `tests/test_draft/test_assistant_pick_timing.py`
- `tests/test_draft/test_assistant_survival.py`
- `tests/test_draft/test_assistant_state.py`
- `tests/test_draft/test_assistant_strategy.py`
- `tests/test_draft/test_assistant_cli.py`

---

## Task 1: `RecommendationSchema`

**Files:**
- Modify: `src/projections/schemas.py` (add after `AuctionValuesSchema`, ~line 1039)
- Test: `tests/test_schemas/test_recommendation_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schemas/test_recommendation_schema.py
"""Tests for RecommendationSchema."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.schemas import _PYARROW_STR, RecommendationSchema


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000001", "00-0000002"], dtype=_PYARROW_STR),
            "position": pd.array(["RB", "WR"], dtype=_PYARROW_STR),
            "vorp": [50.0, 40.0],
            "consensus_adp": pd.array([3.0, pd.NA], dtype=pd.Float64Dtype()),
            "p_available_next": pd.array([0.1, pd.NA], dtype=pd.Float64Dtype()),
            "fills_starting_slot": [True, True],
            "score": [12.6, 2.6],
            "rank": pd.array([1, 2], dtype=pd.Int64Dtype()),
        }
    )


def test_valid_frame_passes() -> None:
    out = RecommendationSchema.validate(_valid_frame())
    assert list(out["rank"]) == [1, 2]


def test_missing_column_fails() -> None:
    bad = _valid_frame().drop(columns=["score"])
    with pytest.raises(Exception):
        RecommendationSchema.validate(bad)


def test_rank_must_be_unique() -> None:
    bad = _valid_frame()
    bad["rank"] = pd.array([1, 1], dtype=pd.Int64Dtype())
    with pytest.raises(Exception):
        RecommendationSchema.validate(bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas/test_recommendation_schema.py -v`
Expected: FAIL with `ImportError: cannot import name 'RecommendationSchema'`.

- [ ] **Step 3: Add the schema**

Add to `src/projections/schemas.py` immediately after the `AuctionValuesSchema` class:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas/test_recommendation_schema.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/projections/schemas.py tests/test_schemas/test_recommendation_schema.py
git commit -m "feat(schemas): RecommendationSchema for draft-assistant output"
```

---

## Task 2: `roster_eligibility` shared helper

**Files:**
- Create: `src/projections/draft/roster_eligibility.py`
- Modify: `src/projections/draft/_pool.py:14-28` (import the sets instead of redefining)
- Test: `tests/test_draft/test_roster_eligibility.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_draft/test_roster_eligibility.py
"""Tests for projections.draft.roster_eligibility."""

from __future__ import annotations

from projections.draft.roster_eligibility import (
    FLEX_ELIGIBLE,
    SUPER_FLEX_ELIGIBLE,
    eligible_positions,
)
from projections.schemas import Position, RosterSlot

_SLOTS = {
    RosterSlot.QB: 1,
    RosterSlot.RB: 2,
    RosterSlot.WR: 2,
    RosterSlot.TE: 1,
    RosterSlot.FLEX: 1,
    RosterSlot.BENCH: 3,
}


def test_eligibility_sets() -> None:
    assert FLEX_ELIGIBLE == frozenset({Position.RB, Position.WR, Position.TE})
    assert SUPER_FLEX_ELIGIBLE == frozenset(
        {Position.QB, Position.RB, Position.WR, Position.TE}
    )


def test_empty_roster_all_positions_start() -> None:
    elig = eligible_positions(_SLOTS, [])
    # QB/RB/WR/TE all have an open starting (position) slot.
    assert elig[Position.QB] is True
    assert elig[Position.RB] is True
    assert elig[Position.WR] is True
    assert elig[Position.TE] is True
    # K/DST are not rostered by this config → not eligible at all.
    assert Position.K not in elig
    assert Position.DST not in elig


def test_filled_position_slot_falls_to_flex_then_bench() -> None:
    # Two RB position slots filled by two RBs; RB can still start via FLEX.
    elig = eligible_positions(_SLOTS, [Position.RB, Position.RB])
    assert elig[Position.RB] is True  # FLEX is an open *starting* slot

    # A third RB consumes FLEX; now RB is bench-only (not a starting slot).
    elig3 = eligible_positions(_SLOTS, [Position.RB, Position.RB, Position.RB])
    assert Position.RB in elig3
    assert elig3[Position.RB] is False  # only BENCH remains


def test_position_fully_filled_is_dropped() -> None:
    # QB:1 + no flex/superflex eligibility for a 2nd QB starting slot, and the
    # bench is exhausted by three RBs → a 2nd QB can only go to bench.
    slots = {RosterSlot.QB: 1, RosterSlot.RB: 2, RosterSlot.BENCH: 0}
    elig = eligible_positions(slots, [Position.QB])
    # QB position slot filled, no FLEX/SUPER_FLEX/BENCH → QB ineligible.
    assert Position.QB not in elig
    assert elig[Position.RB] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_draft/test_roster_eligibility.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'projections.draft.roster_eligibility'`.

- [ ] **Step 3: Create the helper**

```python
# src/projections/draft/roster_eligibility.py
"""Slot↔position eligibility for draft tooling — one source of truth.

Promoted out of `_pool.py`'s private symbols so both the pool selector and the
draft assistant share the same FLEX/SUPER_FLEX/bench rules. Adds a greedy
allocation that, given my league's roster slots and the positions I've already
drafted, reports which positions I can still roster and whether each still has
an open *starting* (non-bench) slot.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

from projections.schemas import Position, RosterSlot

# Position-specific starting slots (a slot whose label is also a Position).
POSITION_SLOTS: tuple[RosterSlot, ...] = (
    RosterSlot.QB,
    RosterSlot.RB,
    RosterSlot.WR,
    RosterSlot.TE,
    RosterSlot.K,
    RosterSlot.DST,
)

FLEX_ELIGIBLE: frozenset[Position] = frozenset(
    {Position.RB, Position.WR, Position.TE}
)
SUPER_FLEX_ELIGIBLE: frozenset[Position] = frozenset(
    {Position.QB, Position.RB, Position.WR, Position.TE}
)


def bench_eligible_positions(roster_slots: Mapping[RosterSlot, int]) -> frozenset[Position]:
    """Positions the league actually rosters (so the shared bench can hold them).

    Excludes positions with no position slot — e.g. a K-less league never benches
    a kicker. Mirrors `_pool.py`'s bench-eligibility rule.
    """
    return frozenset(
        Position(slot.value)
        for slot in POSITION_SLOTS
        if roster_slots.get(slot, 0) > 0
    )


def _open_slots_after(
    roster_slots: Mapping[RosterSlot, int], my_roster: Iterable[Position]
) -> Counter[RosterSlot]:
    """Per-team open slots remaining after greedily placing my drafted players.

    Fill priority per player: own position slot → FLEX → SUPER_FLEX → BENCH.
    A player with no open slot (roster overflow) is left unplaced (no negatives).
    """
    open_: Counter[RosterSlot] = Counter(
        {
            slot: count
            for slot, count in roster_slots.items()
            if slot != RosterSlot.IR and count > 0
        }
    )
    benchable = bench_eligible_positions(roster_slots)
    for pos in my_roster:
        own = RosterSlot(pos.value)
        candidates = (
            (own, True),
            (RosterSlot.FLEX, pos in FLEX_ELIGIBLE),
            (RosterSlot.SUPER_FLEX, pos in SUPER_FLEX_ELIGIBLE),
            (RosterSlot.BENCH, pos in benchable),
        )
        for slot, eligible in candidates:
            if eligible and open_.get(slot, 0) > 0:
                open_[slot] -= 1
                break
    return open_


def _has_open_starting(pos: Position, open_: Counter[RosterSlot]) -> bool:
    """Is there an open *non-bench* slot this position could occupy?"""
    if open_.get(RosterSlot(pos.value), 0) > 0:
        return True
    if pos in FLEX_ELIGIBLE and open_.get(RosterSlot.FLEX, 0) > 0:
        return True
    if pos in SUPER_FLEX_ELIGIBLE and open_.get(RosterSlot.SUPER_FLEX, 0) > 0:
        return True
    return False


def eligible_positions(
    roster_slots: Mapping[RosterSlot, int], my_roster: Iterable[Position]
) -> dict[Position, bool]:
    """Map every still-rosterable position to its starting-need tier.

    Returns `{position: fills_starting_slot}`:
      - a position is a key iff I can still roster a player there (an open
        position/FLEX/SUPER_FLEX slot, or open BENCH capacity if benchable);
      - the value is True iff it still has an open *starting* (non-bench) slot.
    Positions I can no longer roster are absent (the caller drops them).
    """
    my_roster = list(my_roster)
    open_ = _open_slots_after(roster_slots, my_roster)
    benchable = bench_eligible_positions(roster_slots)
    bench_open = open_.get(RosterSlot.BENCH, 0) > 0
    result: dict[Position, bool] = {}
    for pos in Position:
        starting = _has_open_starting(pos, open_)
        rosterable = starting or (pos in benchable and bench_open)
        if rosterable:
            result[pos] = starting
    return result
```

- [ ] **Step 4: Point `_pool.py` at the shared sets (behavior unchanged)**

In `src/projections/draft/_pool.py`, replace the local definitions (lines ~14-28):

```python
from projections.draft.league_config import LeagueConfig
from projections.draft.roster_eligibility import (
    FLEX_ELIGIBLE as _FLEX_ELIGIBLE,
)
from projections.draft.roster_eligibility import (
    POSITION_SLOTS as _POSITION_SLOTS,
)
from projections.draft.roster_eligibility import (
    SUPER_FLEX_ELIGIBLE as _SUPER_FLEX_ELIGIBLE,
)
from projections.schemas import Position, RosterSlot
```

Delete the now-duplicated `_POSITION_SLOTS`, `_FLEX_ELIGIBLE`, and `_SUPER_FLEX_ELIGIBLE` literals from `_pool.py` (they are imported above). Leave the rest of `_pool.py` unchanged. (`Position` may now be unused in `_pool.py`; if ruff flags F401, drop it from the import.)

- [ ] **Step 5: Run tests to verify pass + no `_pool` regression**

Run: `pytest tests/test_draft/test_roster_eligibility.py tests/test_draft/test_vorp.py tests/test_draft/test_auction.py -v`
Expected: PASS (new eligibility tests + existing pool-driven VORP/auction tests all green).

- [ ] **Step 6: Commit**

```bash
git add src/projections/draft/roster_eligibility.py src/projections/draft/_pool.py tests/test_draft/test_roster_eligibility.py
git commit -m "feat(draft): shared roster_eligibility helper (greedy slot allocation)"
```

---

## Task 3: Pick-timing (pure snake-order functions)

**Files:**
- Create: `src/projections/draft/assistant/__init__.py`
- Create: `src/projections/draft/assistant/pick_timing.py`
- Test: `tests/test_draft/test_assistant_pick_timing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_draft/test_assistant_pick_timing.py
"""Tests for snake-order pick timing."""

from __future__ import annotations

from projections.draft.assistant.pick_timing import (
    my_next_pick,
    my_upcoming_picks,
    picks_until_next,
    slot_for,
)

N = 12
ROUNDS = 4  # picks per team for these tests


def test_slot_for_snake_wrap() -> None:
    assert slot_for(7, N) == 7  # round 1, straight order
    assert slot_for(12, N) == 12  # end of round 1
    assert slot_for(13, N) == 12  # round 2 reverses → slot 12 picks back-to-back
    assert slot_for(18, N) == 7  # round 2, reversed
    assert slot_for(24, N) == 1
    assert slot_for(25, N) == 1  # round 3 straight again


def test_my_upcoming_includes_current_when_mine() -> None:
    # Slot 7 of 12: my picks are 7, 18, 31, 42.
    assert my_upcoming_picks(7, my_slot=7, n_teams=N, rounds=ROUNDS) == [7, 18, 31, 42]
    # Standing at pick 8 (not mine): current pick excluded.
    assert my_upcoming_picks(8, my_slot=7, n_teams=N, rounds=ROUNDS) == [18, 31, 42]


def test_my_next_pick_is_strictly_after_current() -> None:
    assert my_next_pick(7, my_slot=7, n_teams=N, rounds=ROUNDS) == 18
    assert my_next_pick(18, my_slot=7, n_teams=N, rounds=ROUNDS) == 31
    # On my final pick there is no next pick.
    assert my_next_pick(42, my_slot=7, n_teams=N, rounds=ROUNDS) is None


def test_picks_until_next_counts_opponents() -> None:
    assert picks_until_next(7, my_slot=7, n_teams=N, rounds=ROUNDS) == 10  # picks 8..17
    assert picks_until_next(42, my_slot=7, n_teams=N, rounds=ROUNDS) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_draft/test_assistant_pick_timing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'projections.draft.assistant'`.

- [ ] **Step 3: Create the package init + pick_timing**

```python
# src/projections/draft/assistant/__init__.py
"""Live draft-assistant engine (Slice 1): pick timing, survival, strategies."""
```

```python
# src/projections/draft/assistant/pick_timing.py
"""Pure snake-draft pick-timing math. No pandas; hand-computable.

Picks are 1-based absolute pick numbers. Rounds are 0-based internally. Odd
0-based rounds run the slot order in reverse (the "snake").
"""

from __future__ import annotations


def slot_for(pick_number: int, n_teams: int) -> int:
    """Which 1-based slot owns an absolute pick under snake order."""
    if pick_number < 1:
        raise ValueError(f"pick_number must be >= 1; got {pick_number}")
    round_idx = (pick_number - 1) // n_teams
    offset = (pick_number - 1) % n_teams
    if round_idx % 2 == 0:
        return offset + 1
    return n_teams - offset


def _pick_number(round_idx: int, my_slot: int, n_teams: int) -> int:
    """Absolute pick number my slot holds in a given 0-based round."""
    if round_idx % 2 == 0:
        return round_idx * n_teams + my_slot
    return round_idx * n_teams + (n_teams - my_slot + 1)


def my_upcoming_picks(
    current_pick: int, my_slot: int, n_teams: int, rounds: int
) -> list[int]:
    """My absolute pick numbers `>= current_pick` (current included if it's mine)."""
    return [
        p
        for r in range(rounds)
        if (p := _pick_number(r, my_slot, n_teams)) >= current_pick
    ]


def my_next_pick(
    current_pick: int, my_slot: int, n_teams: int, rounds: int
) -> int | None:
    """My first pick strictly after `current_pick`, or None if none remain."""
    later = [
        p
        for p in my_upcoming_picks(current_pick, my_slot, n_teams, rounds)
        if p > current_pick
    ]
    return later[0] if later else None


def picks_until_next(
    current_pick: int, my_slot: int, n_teams: int, rounds: int
) -> int | None:
    """Count of opponent picks strictly between this pick and my next one."""
    nxt = my_next_pick(current_pick, my_slot, n_teams, rounds)
    if nxt is None:
        return None
    return nxt - current_pick - 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_draft/test_assistant_pick_timing.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/__init__.py src/projections/draft/assistant/pick_timing.py tests/test_draft/test_assistant_pick_timing.py
git commit -m "feat(draft): snake pick-timing for the draft assistant"
```

---

## Task 4: Survival model

**Files:**
- Create: `src/projections/draft/assistant/survival.py`
- Test: `tests/test_draft/test_assistant_survival.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_draft/test_assistant_survival.py
"""Tests for the ADP survival model."""

from __future__ import annotations

import math

from projections.draft.assistant.survival import (
    LogisticSurvival,
    SurvivalModel,
    default_sigma,
)


def test_is_survival_model() -> None:
    assert isinstance(LogisticSurvival(sigma=8.0), SurvivalModel)


def test_monotone_in_adp() -> None:
    model = LogisticSurvival(sigma=8.0)
    # Later ADP (drafted later) → more likely to survive to a fixed pick.
    p_early = model.p_available(adp=3.0, at_pick=18)
    p_late = model.p_available(adp=30.0, at_pick=18)
    assert 0.0 <= p_early <= p_late <= 1.0


def test_boundaries() -> None:
    model = LogisticSurvival(sigma=8.0)
    assert model.p_available(adp=1.0, at_pick=60) < 0.05  # long gone
    assert model.p_available(adp=200.0, at_pick=10) > 0.95  # nowhere near taken


def test_null_adp_survives() -> None:
    model = LogisticSurvival(sigma=8.0)
    assert model.p_available(adp=math.nan, at_pick=18) == 1.0


def test_default_sigma_scales_with_teams() -> None:
    assert default_sigma(12) == 8.0  # two-thirds of a 12-team round
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_draft/test_assistant_survival.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the survival model**

```python
# src/projections/draft/assistant/survival.py
"""Probability a player is still available at a future pick, from market ADP.

The default `LogisticSurvival` uses a logistic CDF around ADP with a single
global spread `sigma` (in picks). The exact CDF shape is not load-bearing — it
is monotone and deterministic, and `sigma` is tuned empirically in Slice 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class SurvivalModel(Protocol):
    def p_available(self, adp: float, at_pick: int) -> float:
        """P(player with this ADP is still on the board *at* `at_pick`)."""
        ...


def default_sigma(n_teams: int) -> float:
    """Spread default ≈ two-thirds of one round (picks)."""
    return (2.0 / 3.0) * n_teams


def _sigmoid(x: float) -> float:
    # Numerically stable logistic.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class LogisticSurvival:
    """Logistic survival in ADP space. `sigma` is the spread in picks."""

    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0; got {self.sigma}")

    def p_available(self, adp: float, at_pick: int) -> float:
        # No market signal → treat as "won't be taken soon".
        if adp is None or math.isnan(adp):
            return 1.0
        # Available *at* `at_pick` ⇔ not taken on or before `at_pick - 1`.
        return 1.0 - _sigmoid((at_pick - 1 - adp) / self.sigma)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_draft/test_assistant_survival.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/survival.py tests/test_draft/test_assistant_survival.py
git commit -m "feat(draft): ADP logistic survival model"
```

---

## Task 5: Draft-state model

**Files:**
- Create: `src/projections/draft/assistant/state.py`
- Test: `tests/test_draft/test_assistant_state.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_draft/test_assistant_state.py
"""Tests for DraftState + load_draft_state."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from projections.draft.assistant.state import DraftState, load_draft_state
from projections.schemas import _PYARROW_STR, Position, RosterSlot, Ruleset


def _write_config(tmp_path: Path) -> Path:
    from projections.draft.league_config import LeagueConfig

    cfg = LeagueConfig(
        name="t",
        n_teams=12,
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 5,
        },
        ruleset=Ruleset.espn_ppr(),
    )
    p = tmp_path / "cfg.json"
    p.write_text(cfg.model_dump_json())
    return p


def _id_map() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": pd.array(
                ["00-0000007", "00-0000008", "00-0000018"], dtype=_PYARROW_STR
            ),
            "position": pd.array(["RB", "WR", "QB"], dtype=_PYARROW_STR),
            "full_name": pd.array(["A", "B", "C"], dtype=_PYARROW_STR),
        }
    )


def _state_file(tmp_path: Path, cfg_path: Path, picks: list[str]) -> Path:
    p = tmp_path / "state.json"
    p.write_text(
        json.dumps({"league_config": str(cfg_path), "my_slot": 7, "picks": picks})
    )
    return p


def test_my_roster_from_id_map(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    # Slot 7 of 12 owns pick 7 → "00-0000007" (RB) is mine; pick 8 is an opponent's.
    state_path = _state_file(tmp_path, cfg, ["00-0000018"] * 6 + ["00-0000007", "00-0000008"])
    state, league = load_draft_state(state_path, _id_map())
    assert isinstance(state, DraftState)
    assert state.current_pick == 9
    assert state.drafted_ids == frozenset(
        {"00-0000018", "00-0000007", "00-0000008"}
    )
    assert state.my_roster == (Position.RB,)  # only my pick #7
    assert league.n_teams == 12


def test_missing_id_map_entry_for_my_pick_raises(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    # My pick (#7) references an id not in id_map.
    state_path = _state_file(tmp_path, cfg, ["00-0000018"] * 6 + ["00-0000099"])
    with pytest.raises(ValueError, match="00-0000099"):
        load_draft_state(state_path, _id_map())


def test_duplicate_pick_raises(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    state_path = _state_file(tmp_path, cfg, ["00-0000007", "00-0000007"])
    with pytest.raises(ValueError, match="duplicate"):
        load_draft_state(state_path, _id_map())


def test_bad_slot_raises(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"league_config": str(cfg), "my_slot": 99, "picks": []}))
    with pytest.raises(ValueError, match="my_slot"):
        load_draft_state(p, _id_map())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_draft/test_assistant_state.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the state model**

```python
# src/projections/draft/assistant/state.py
"""DraftState — the live draft as the engine sees it, loaded from a JSON file.

The state file is an ordered list of drafted gsis_ids plus my slot and a path to
the LeagueConfig; a pick's slot is derived from its (1-based) position via snake
order. My roster's positions are resolved through id_map (the universal position
source — the consensus VORP table can't supply a position for off-board picks).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from projections.draft.assistant.pick_timing import slot_for
from projections.draft.league_config import LeagueConfig
from projections.schemas import GsisId, Position, validate_gsis_id


@dataclass(frozen=True)
class DraftState:
    """An immutable snapshot of an in-progress draft."""

    my_slot: int
    n_teams: int
    rounds: int  # picks per team (== LeagueConfig.roster_size)
    picks: tuple[GsisId, ...]  # drafted gsis_ids, in pick order
    my_roster: tuple[Position, ...]  # positions of the picks I made

    @property
    def drafted_ids(self) -> frozenset[GsisId]:
        return frozenset(self.picks)

    @property
    def current_pick(self) -> int:
        return len(self.picks) + 1


def load_draft_state(
    state_path: Path, id_map: pd.DataFrame
) -> tuple[DraftState, LeagueConfig]:
    """Parse a draft-state JSON file into a `DraftState` + its `LeagueConfig`.

    Raises ValueError on: my_slot out of range, a malformed/duplicate gsis_id, or
    one of *my* picks being absent from id_map (unknown position).
    """
    data = json.loads(state_path.read_text())
    league = LeagueConfig.model_validate_json(Path(data["league_config"]).read_text())

    my_slot = int(data["my_slot"])
    if not 1 <= my_slot <= league.n_teams:
        raise ValueError(
            f"my_slot must be in 1..{league.n_teams}; got {my_slot}"
        )

    picks = tuple(validate_gsis_id(str(p)) for p in data["picks"])
    if len(set(picks)) != len(picks):
        raise ValueError("draft state has a duplicate pick (a player drafted twice)")

    pos_by_id = dict(zip(id_map["gsis_id"], id_map["position"], strict=False))
    my_roster: list[Position] = []
    for index, gid in enumerate(picks):
        pick_number = index + 1
        if slot_for(pick_number, league.n_teams) != my_slot:
            continue
        if gid not in pos_by_id:
            raise ValueError(
                f"my pick {gid} (pick #{pick_number}) is absent from id_map; "
                "cannot resolve its position for roster accounting"
            )
        my_roster.append(Position(pos_by_id[gid]))

    state = DraftState(
        my_slot=my_slot,
        n_teams=league.n_teams,
        rounds=league.roster_size,
        picks=picks,
        my_roster=tuple(my_roster),
    )
    return state, league
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_draft/test_assistant_state.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/state.py tests/test_draft/test_assistant_state.py
git commit -m "feat(draft): DraftState + id_map-resolved roster"
```

---

## Task 6: Strategies (`DraftStrategy`, `RawVorpStrategy`, `NowOrNeverStrategy`)

**Files:**
- Create: `src/projections/draft/assistant/strategy.py`
- Test: `tests/test_draft/test_assistant_strategy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_draft/test_assistant_strategy.py
"""Tests for the draft strategies."""

from __future__ import annotations

import pandas as pd

from projections.draft.assistant.state import DraftState
from projections.draft.assistant.strategy import (
    DraftStrategy,
    NowOrNeverStrategy,
    RawVorpStrategy,
)
from projections.schemas import (
    _PYARROW_STR,
    GsisId,
    Position,
    RecommendationSchema,
    RosterSlot,
    Ruleset,
)
from projections.draft.league_config import LeagueConfig


def _config() -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=12,
        roster_slots={
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 5,
        },
        ruleset=Ruleset.espn_ppr(),
    )


def _pool() -> pd.DataFrame:
    # rb1 scarce (low survival), wr1 highest VORP but safe.
    return pd.DataFrame(
        {
            "gsis_id": pd.array(
                ["00-0000010", "00-0000011", "00-0000020", "00-0000021"],
                dtype=_PYARROW_STR,
            ),
            "position": pd.array(["RB", "RB", "WR", "WR"], dtype=_PYARROW_STR),
            "season_mean_fpts": [250.0, 240.0, 252.0, 230.0],
            "vorp": [50.0, 40.0, 52.0, 30.0],
            "replacement_fpts": [200.0, 200.0, 200.0, 200.0],
            "consensus_adp": pd.array([5.0, 6.0, 7.0, 8.0], dtype=pd.Float64Dtype()),
        }
    )


def _state(
    current_pick: int = 7,
    rounds: int = 9,
    my_roster: tuple[Position, ...] = (),
) -> DraftState:
    """Build a state standing at `current_pick` (materialize filler picks so the
    derived current_pick is correct). Fillers use a 9-prefix so they never
    collide with the pool's 00-0000xxx ids. Empty roster → RB and WR both
    eligible & start-needed unless `my_roster` overrides.
    """
    fillers = tuple(GsisId(f"00-9{i:06d}") for i in range(current_pick - 1))
    return DraftState(
        my_slot=7, n_teams=12, rounds=rounds, picks=fillers, my_roster=my_roster
    )


class _FakeSurvival:
    """Deterministic survival lookup keyed by adp, for hand-computed expectations."""

    _P = {5.0: 0.1, 6.0: 0.9, 7.0: 0.95, 8.0: 0.9}

    def p_available(self, adp: float, at_pick: int) -> float:
        return self._P[adp]


def test_both_satisfy_protocol() -> None:
    assert isinstance(RawVorpStrategy(), DraftStrategy)
    assert isinstance(NowOrNeverStrategy(_FakeSurvival()), DraftStrategy)


def test_raw_vorp_orders_by_vorp_and_nulls_p_available() -> None:
    rec = RawVorpStrategy().recommend(_state(), _pool(), _config())
    RecommendationSchema.validate(rec)
    assert list(rec["gsis_id"]) == [
        "00-0000020",  # wr1 52
        "00-0000010",  # rb1 50
        "00-0000011",  # rb2 40
        "00-0000021",  # wr2 30
    ]
    assert rec["p_available_next"].isna().all()


def test_now_or_never_reorders_cross_position() -> None:
    rec = NowOrNeverStrategy(_FakeSurvival()).recommend(_state(), _pool(), _config())
    RecommendationSchema.validate(rec)
    # E[best RB survivor] = 50*.1 + 40*.9*.9 = 37.4 → rb1 score 12.6, rb2 2.6
    # E[best WR survivor] = 52*.95 + 30*.9*.05 = 50.75 → wr1 score 1.25, wr2 -20.75
    assert list(rec["gsis_id"]) == [
        "00-0000010",  # rb1 12.6  (jumps wr1 — the reorder)
        "00-0000011",  # rb2 2.6
        "00-0000020",  # wr1 1.25
        "00-0000021",  # wr2 -20.75
    ]
    assert rec.loc[rec["gsis_id"] == "00-0000010", "score"].iloc[0] == 12.6


def test_within_position_order_is_vorp() -> None:
    rec = NowOrNeverStrategy(_FakeSurvival()).recommend(_state(), _pool(), _config())
    rb = rec[rec["position"] == "RB"]
    assert list(rb["gsis_id"]) == ["00-0000010", "00-0000011"]  # vorp desc


def test_last_pick_fallback_equals_raw_vorp() -> None:
    # rounds=1 → my only pick is pick 7, no next pick.
    last = _state(current_pick=7, rounds=1)
    non = NowOrNeverStrategy(_FakeSurvival()).recommend(last, _pool(), _config())
    raw = RawVorpStrategy().recommend(last, _pool(), _config())
    assert list(non["gsis_id"]) == list(raw["gsis_id"])
    assert non["p_available_next"].isna().all()


def test_roster_eligible_filter_drops_filled_position() -> None:
    # Fill both RB slots + FLEX with RBs → RB only benchable, WR still starts.
    state = DraftState(
        my_slot=7,
        n_teams=12,
        rounds=9,
        picks=(),
        my_roster=(Position.RB, Position.RB, Position.RB),
    )
    rec = RawVorpStrategy().recommend(state, _pool(), _config())
    # RB still rosterable (bench), but WR fills a starting slot → WR tier first.
    assert bool(rec.iloc[0]["fills_starting_slot"]) is True
    assert rec.iloc[0]["position"] == "WR"


def test_equal_score_tie_break_is_gsis_id() -> None:
    # Two WRs, identical vorp → identical raw-vorp score; rank by gsis_id asc.
    pool = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000031", "00-0000030"], dtype=_PYARROW_STR),
            "position": pd.array(["WR", "WR"], dtype=_PYARROW_STR),
            "season_mean_fpts": [240.0, 240.0],
            "vorp": [40.0, 40.0],
            "replacement_fpts": [200.0, 200.0],
            "consensus_adp": pd.array([10.0, 11.0], dtype=pd.Float64Dtype()),
        }
    )
    rec = RawVorpStrategy().recommend(_state(), pool, _config())
    assert list(rec["gsis_id"]) == ["00-0000030", "00-0000031"]  # gsis asc
    assert list(rec["rank"]) == [1, 2]


def test_now_or_never_null_adp_p_available_is_null() -> None:
    # A null-ADP player still ranks, but its displayed p_available_next is null
    # (spec §3.5 output contract). Uses the real survival model (handles NaN).
    from projections.draft.assistant.survival import LogisticSurvival

    pool = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000040", "00-0000041"], dtype=_PYARROW_STR),
            "position": pd.array(["RB", "RB"], dtype=_PYARROW_STR),
            "season_mean_fpts": [240.0, 230.0],
            "vorp": [40.0, 30.0],
            "replacement_fpts": [200.0, 200.0],
            "consensus_adp": pd.array([5.0, pd.NA], dtype=pd.Float64Dtype()),
        }
    )
    rec = NowOrNeverStrategy(LogisticSurvival(sigma=8.0)).recommend(
        _state(), pool, _config()
    )
    RecommendationSchema.validate(rec)
    by_id = rec.set_index("gsis_id")["p_available_next"]
    assert pd.isna(by_id["00-0000041"])  # null ADP → null p_available_next
    assert pd.notna(by_id["00-0000040"])  # has ADP → populated
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_draft/test_assistant_strategy.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the strategies**

```python
# src/projections/draft/assistant/strategy.py
"""Draft strategies: the substitution seam + two concrete implementations.

`RawVorpStrategy` is the best-available control. `NowOrNeverStrategy` is the
analytic opportunity-cost strategy (spec §3.5): rank by value locked in over the
expected best survivor at the same position by my next pick. Both share
`_finalize`, which filters to roster-eligible positions, tags the scale-free
starting-need tier, and applies the deterministic final ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from projections.draft.assistant.pick_timing import my_next_pick
from projections.draft.assistant.state import DraftState
from projections.draft.assistant.survival import SurvivalModel
from projections.draft.league_config import LeagueConfig
from projections.draft.roster_eligibility import eligible_positions
from projections.schemas import _PYARROW_STR, Position, RecommendationSchema


@runtime_checkable
class DraftStrategy(Protocol):
    def recommend(
        self, state: DraftState, pool: pd.DataFrame, config: LeagueConfig
    ) -> pd.DataFrame:
        """Rank the available pool; return a RecommendationSchema frame."""
        ...


def _eligible_subset(
    state: DraftState, pool: pd.DataFrame, config: LeagueConfig
) -> tuple[pd.DataFrame, dict[Position, bool]]:
    """Drop already-drafted + roster-ineligible rows. Returns (subset, eligibility)."""
    elig = eligible_positions(config.roster_slots, list(state.my_roster))
    eligible_values = {pos.value for pos in elig}
    subset = pool[
        ~pool["gsis_id"].isin(state.drafted_ids)
        & pool["position"].isin(eligible_values)
    ].copy()
    # consensus_adp is optional on VorpTableSchema (absent on the weekly path).
    # The assistant is consensus-only, but guard so a missing column degrades to
    # all-null (everything survives) instead of a KeyError in _finalize.
    if "consensus_adp" not in subset.columns:
        subset["consensus_adp"] = pd.array(
            [pd.NA] * len(subset), dtype=pd.Float64Dtype()
        )
    return subset, elig


def _finalize(
    df: pd.DataFrame, elig: dict[Position, bool], p_available: pd.Series
) -> pd.DataFrame:
    """Attach the starting-need tier, order deterministically, validate.

    `df` must already carry `score`. `p_available` is index-aligned (Float64,
    null where unknown).
    """
    out = df.copy()
    out["fills_starting_slot"] = out["position"].map(
        lambda value: elig[Position(value)]
    )
    out["p_available_next"] = p_available.astype(pd.Float64Dtype())
    out["consensus_adp"] = out["consensus_adp"].astype(pd.Float64Dtype())
    out["gsis_id"] = out["gsis_id"].astype(_PYARROW_STR)
    out["position"] = out["position"].astype(_PYARROW_STR)
    out = out.sort_values(
        ["fills_starting_slot", "score", "vorp", "gsis_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    out["rank"] = pd.array(range(1, len(out) + 1), dtype=pd.Int64Dtype())
    cols = list(RecommendationSchema.to_schema().columns)
    return RecommendationSchema.validate(out[cols])


@dataclass(frozen=True)
class RawVorpStrategy:
    """Best available by VORP (roster-eligible), no timing. The control."""

    def recommend(
        self, state: DraftState, pool: pd.DataFrame, config: LeagueConfig
    ) -> pd.DataFrame:
        df, elig = _eligible_subset(state, pool, config)
        df["score"] = df["vorp"].astype(float)
        p_na = pd.Series(pd.NA, index=df.index, dtype=pd.Float64Dtype())
        return _finalize(df, elig, p_na)


@dataclass(frozen=True)
class NowOrNeverStrategy:
    """Opportunity-cost strategy: value over the expected best survivor (spec §3.5)."""

    survival: SurvivalModel

    def recommend(
        self, state: DraftState, pool: pd.DataFrame, config: LeagueConfig
    ) -> pd.DataFrame:
        df, elig = _eligible_subset(state, pool, config)
        next_pick = my_next_pick(
            state.current_pick, state.my_slot, state.n_teams, state.rounds
        )
        if next_pick is None:
            # Last-pick fallback → raw VORP, null p_available.
            df["score"] = df["vorp"].astype(float)
            p_na = pd.Series(pd.NA, index=df.index, dtype=pd.Float64Dtype())
            return _finalize(df, elig, p_na)

        # Internal survival prob per row (1.0 for null ADP); displayed value is
        # null where ADP is null.
        adp = df["consensus_adp"]
        internal_p = adp.map(
            lambda a: self.survival.p_available(float(a) if pd.notna(a) else float("nan"), next_pick)
        ).astype(float)
        display_p = pd.Series(internal_p, index=df.index).where(adp.notna(), other=pd.NA)

        df = df.assign(_p=internal_p)
        e_best: dict[str, float] = {}
        for position, sub in df.groupby("position"):
            sub = sub.sort_values(["vorp", "gsis_id"], ascending=[False, True])
            expected = 0.0
            prob_all_better_gone = 1.0
            for vorp_i, p_i in zip(sub["vorp"], sub["_p"], strict=True):
                expected += float(vorp_i) * p_i * prob_all_better_gone
                prob_all_better_gone *= 1.0 - p_i
            e_best[str(position)] = expected

        df["score"] = df["vorp"].astype(float) - df["position"].map(e_best).astype(float)
        return _finalize(df, elig, display_p)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_draft/test_assistant_strategy.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/strategy.py tests/test_draft/test_assistant_strategy.py
git commit -m "feat(draft): now-or-never + raw-vorp draft strategies"
```

---

## Task 7: CLI (`assistant/cli.py` + `scripts/draft_assistant.py`)

**Files:**
- Create: `src/projections/draft/assistant/cli.py`
- Create: `scripts/draft_assistant.py`
- Modify: `src/projections/draft/assistant/__init__.py` (exports)
- Modify: `src/projections/draft/__init__.py` (re-export)
- Test: `tests/test_draft/test_assistant_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_draft/test_assistant_cli.py
"""End-to-end smoke test for the draft-assistant CLI core."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from projections.draft.assistant.cli import generate_recommendation, run
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, RecommendationSchema, RosterSlot, Ruleset


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    cfg = LeagueConfig(
        name="t",
        n_teams=12,
        roster_slots={
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 5,
        },
        ruleset=Ruleset.espn_ppr(),
    )
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(cfg.model_dump_json())

    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"league_config": str(cfg_path), "my_slot": 7, "picks": []})
    )

    vorp = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000010", "00-0000020"], dtype=_PYARROW_STR),
            "position": pd.array(["RB", "WR"], dtype=_PYARROW_STR),
            "season_mean_fpts": [250.0, 252.0],
            "vorp": [50.0, 52.0],
            "replacement_fpts": [200.0, 200.0],
            "consensus_adp": pd.array([5.0, 7.0], dtype=pd.Float64Dtype()),
        }
    )
    vorp_path = tmp_path / "vorp.parquet"
    vorp.to_parquet(vorp_path, index=False)

    id_map = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000010", "00-0000020"], dtype=_PYARROW_STR),
            "position": pd.array(["RB", "WR"], dtype=_PYARROW_STR),
            "full_name": pd.array(["RB One", "WR One"], dtype=_PYARROW_STR),
        }
    )
    id_path = tmp_path / "id_map.parquet"
    id_map.to_parquet(id_path, index=False)
    return state_path, vorp_path, id_path


def test_generate_recommendation(tmp_path: Path) -> None:
    state_path, vorp_path, id_path = _setup(tmp_path)
    rec = generate_recommendation(
        state_path=state_path,
        vorp_path=vorp_path,
        id_map_path=id_path,
        strategy_name="now_or_never",
        sigma=None,
    )
    RecommendationSchema.validate(rec)
    assert set(rec["gsis_id"]) == {"00-0000010", "00-0000020"}


def test_run_prints_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    state_path, vorp_path, id_path = _setup(tmp_path)
    code = run(
        [
            "--state",
            str(state_path),
            "--vorp-table",
            str(vorp_path),
            "--id-map",
            str(id_path),
            "--strategy",
            "raw_vorp",
            "--top",
            "5",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "WR One" in out and "RB One" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_draft/test_assistant_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'projections.draft.assistant.cli'`.

- [ ] **Step 3: Implement the CLI core**

```python
# src/projections/draft/assistant/cli.py
"""CLI core for the live draft assistant (testable; scripts/ wraps this).

Reads the draft-state file + consensus VORP table + id_map, runs a strategy, and
prints a ranked recommendation with player names attached.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.draft.assistant.state import load_draft_state
from projections.draft.assistant.strategy import (
    DraftStrategy,
    NowOrNeverStrategy,
    RawVorpStrategy,
)
from projections.draft.assistant.survival import LogisticSurvival, default_sigma
from projections.schemas import _PYARROW_STR, IdMapSchema, VorpTableSchema

_DEFAULT_ID_MAP = Path("data/raw/id_map.parquet")


def _load_id_map(path: Path) -> pd.DataFrame:
    """Load + validate id_map. Required — it is the position source (spec §3.2)."""
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"id_map parquet not found at {path}; it is required (position + name source)."
        ) from exc
    return IdMapSchema.validate(df)


def _build_strategy(name: str, n_teams: int, sigma: float | None) -> DraftStrategy:
    if name == "raw_vorp":
        return RawVorpStrategy()
    if name == "now_or_never":
        spread = default_sigma(n_teams) if sigma is None else sigma
        return NowOrNeverStrategy(LogisticSurvival(sigma=spread))
    raise ValueError(f"unknown strategy {name!r}")


def generate_recommendation(
    *,
    state_path: Path,
    vorp_path: Path,
    id_map_path: Path,
    strategy_name: str,
    sigma: float | None,
) -> pd.DataFrame:
    """Load inputs, run the chosen strategy, return a RecommendationSchema frame."""
    id_map = _load_id_map(id_map_path)
    state, league = load_draft_state(state_path, id_map)

    vorp = pd.read_parquet(vorp_path)
    vorp["gsis_id"] = vorp["gsis_id"].astype(_PYARROW_STR)
    vorp = VorpTableSchema.validate(vorp)

    strategy = _build_strategy(strategy_name, league.n_teams, sigma)
    return strategy.recommend(state, vorp, league)


def format_table(rec: pd.DataFrame, id_map: pd.DataFrame, top: int) -> str:
    """Render the top-N recommendation as a fixed-width text table."""
    names = dict(zip(id_map["gsis_id"], id_map["full_name"], strict=False))
    lines = [f"{'#':>3}  {'PLAYER':<24} {'POS':<4} {'VORP':>7} {'ADP':>6} {'P(next)':>8} {'SCORE':>8}"]
    for row in rec.head(top).itertuples(index=False):
        name = str(names.get(row.gsis_id, "—"))
        adp = f"{float(row.consensus_adp):.1f}" if pd.notna(row.consensus_adp) else "—"
        p_next = f"{float(row.p_available_next):.2f}" if pd.notna(row.p_available_next) else "—"
        star = "*" if row.fills_starting_slot else " "
        lines.append(
            f"{int(row.rank):>3}  {name:<24} {row.position:<4} {row.vorp:>7.1f} "
            f"{adp:>6} {p_next:>8} {row.score:>7.2f}{star}"
        )
    lines.append("  (* = fills an open starting slot)")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live snake-draft pick recommender.")
    p.add_argument("--state", type=Path, required=True, help="Draft-state JSON path.")
    p.add_argument(
        "--vorp-table",
        type=Path,
        required=True,
        help="Consensus VORP parquet (generate_vorp_table.py --source consensus).",
    )
    p.add_argument(
        "--id-map", type=Path, default=_DEFAULT_ID_MAP, help="IdMap parquet (position + names)."
    )
    p.add_argument(
        "--strategy",
        choices=["now_or_never", "raw_vorp"],
        default="now_or_never",
        help="Recommendation strategy (default now_or_never).",
    )
    p.add_argument("--top", type=int, default=15, help="Rows to print (default 15).")
    p.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Survival spread in picks (default ≈ 2/3 of a round).",
    )
    return p.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rec = generate_recommendation(
        state_path=args.state,
        vorp_path=args.vorp_table,
        id_map_path=args.id_map,
        strategy_name=args.strategy,
        sigma=args.sigma,
    )
    id_map = _load_id_map(args.id_map)
    print(format_table(rec, id_map, int(args.top)))
    return 0
```

- [ ] **Step 4: Create the thin script wrapper**

```python
# scripts/draft_assistant.py
"""CLI wrapper for the live draft assistant. See projections.draft.assistant.cli."""

from __future__ import annotations

import sys

from projections.draft.assistant.cli import run

if __name__ == "__main__":
    sys.exit(run())
```

- [ ] **Step 5: Wire up the package exports**

Replace `src/projections/draft/assistant/__init__.py` with:

```python
# src/projections/draft/assistant/__init__.py
"""Live draft-assistant engine (Slice 1): pick timing, survival, strategies."""

from projections.draft.assistant.state import DraftState, load_draft_state
from projections.draft.assistant.strategy import (
    DraftStrategy,
    NowOrNeverStrategy,
    RawVorpStrategy,
)
from projections.draft.assistant.survival import (
    LogisticSurvival,
    SurvivalModel,
    default_sigma,
)

__all__ = [
    "DraftState",
    "DraftStrategy",
    "LogisticSurvival",
    "NowOrNeverStrategy",
    "RawVorpStrategy",
    "SurvivalModel",
    "default_sigma",
    "load_draft_state",
]
```

Add to `src/projections/draft/__init__.py` — extend the imports and `__all__`:

```python
from projections.draft.assistant import (
    DraftState,
    NowOrNeverStrategy,
    RawVorpStrategy,
    load_draft_state,
)
```

and add `"DraftState"`, `"NowOrNeverStrategy"`, `"RawVorpStrategy"`, `"load_draft_state"` to the existing `__all__` list (keep it alphabetically sorted to satisfy ruff `RUF022` if configured).

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_draft/test_assistant_cli.py -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add src/projections/draft/assistant/cli.py scripts/draft_assistant.py src/projections/draft/assistant/__init__.py src/projections/draft/__init__.py tests/test_draft/test_assistant_cli.py
git commit -m "feat(draft): draft-assistant CLI"
```

---

## Task 8: Full gate run + project docs

**Files:**
- Modify: `project_management.md` (prepend a status entry)
- Modify: `TODO.md` (mark the overall-board / draft-assistant item)

- [ ] **Step 1: Run the full verification gate**

Run each and fix any failure before proceeding:

```bash
pytest -v -k "draft or assistant or roster_eligibility or recommendation"
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: all green. (The `-k` subset is the changed surface; if anything fails in `mypy`/`ruff`, it scans the whole tree regardless.) If `ruff format --check` reports drift, run `ruff format src tests` and re-stage.

- [ ] **Step 2: Run the ingest/store/schema integration seam (per CLAUDE.md)**

Run: `pytest -v -k "ingest or store or schemas"`
Expected: PASS — confirms the new `RecommendationSchema` didn't perturb the schema module's dtype seams.

- [ ] **Step 3: Update `project_management.md`**

Prepend a new top entry (below the `# Project Management` header block) summarizing: Slice 1 of the Draft Assistant shipped — `DraftState`, pick-timing, ADP survival, `DraftStrategy` protocol + `NowOrNeverStrategy`/`RawVorpStrategy`, roster-need via the new shared `roster_eligibility`, CLI; note the consumer contract (consensus VORP table), the gates that passed, and the next slices (harness, then UI). Mirror the format of the existing PR #56 entry.

- [ ] **Step 4: Update `TODO.md`**

Under TODO #38, record: Draft Assistant Slice 1 (engine core) done on `feat/draft-assistant-engine`; spec/plan at `docs/superpowers/specs|plans/2026-06-09-draft-assistant-engine.*`; remaining slices — Slice 2 strategy comparison harness (CLI tournament, σ tuning), Slice 3 Streamlit UI. Note the survival model's unconditional approximation as a Slice-2 refinement.

- [ ] **Step 5: Commit**

```bash
git add project_management.md TODO.md
git commit -m "docs(pm,todo): draft-assistant engine (Slice 1) shipped"
```

---

## Notes for the executor

- **Determinism is load-bearing** (correctness bar): every sort that feeds `rank` or the survivor sum carries an explicit `gsis_id` tie-break. Don't drop them.
- **dtypes:** nullable floats are `pd.Float64Dtype()`, nullable ints `pd.Int64Dtype()`, string columns `_PYARROW_STR`. Plain `object`+`str` or `int`+`NA` will fail pandera validation (see CLAUDE.md).
- **Enums, never strings:** use `Position.RB` / `RosterSlot.FLEX`. The `position` *column* holds the enum `.value` strings (that's the persisted form); convert with `Position(value)` when you need the enum.
- **`runtime_checkable` is structural-only** — `isinstance(x, DraftStrategy)` checks attribute presence, not signatures. Trust mypy for the real contract.
- **The engine never imports the CLI or any UI.** `cli.py` may import the engine; nothing in the engine imports `cli.py`.
