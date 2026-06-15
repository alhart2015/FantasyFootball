# Live Draft Board Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-06-15-live-draft-board-design.md`

**Goal:** A Streamlit live-draft board (co-pilot + mock simulator) over the existing Draft Assistant engine, driven by a pure, fully-tested `LiveDraftSession` controller.

**Architecture:** All draft logic lives in a testable controller `src/projections/draft/assistant/live.py` (no Streamlit import); `scripts/draft_board.py` is a thin Streamlit view that owns `st.session_state` and delegates every decision to the controller. The controller wraps existing engine functions (`build_draft_state`, `DraftStrategy.recommend`, `bot_pick`, `optimal_lineup_points`, `roster_eligibility`). A shared `build_session_strategy` seam turns a strategy name + params into a `DraftStrategy` for both the dropdown and resume.

**Tech Stack:** Python 3.11, pandas (pyarrow dtypes), pandera schemas, pydantic `LeagueConfig`, numpy RNG, Streamlit (new optional `[ui]` dependency), pytest / mypy --strict / ruff.

**Conventions (follow exactly):**
- pyarrow string dtype for id/position/name columns: `from projections.schemas import _PYARROW_STR`.
- Reference enums, never strings: `Position.RB`, `RosterSlot.FLEX`, etc. Persisted/compared values use `.value`.
- `validate_gsis_id(raw)` is the only sanctioned `GsisId` constructor.
- Run a single test with live output: `pytest -n0 -s <path>::<name> -v` (the suite defaults to `-n auto`).
- Commit messages end with the `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer.

---

## Phase 1 — Controller foundation (pure, tested core)

### Task 1: Extract `build_draft_state` from `load_draft_state`

**Files:**
- Modify: `src/projections/draft/assistant/state.py`
- Test: `tests/test_draft/test_assistant_state.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_draft/test_assistant_state.py`:

```python
def test_build_draft_state_matches_load_draft_state(tmp_path: Path) -> None:
    from projections.draft.assistant.state import build_draft_state
    from projections.draft.league_config import LeagueConfig

    cfg_path = _write_config(tmp_path)
    league = LeagueConfig.model_validate_json(cfg_path.read_text())
    picks = [*_OPP_PICKS, "00-0000007", "00-0000008"]
    state_path = _state_file(tmp_path, cfg_path, picks)

    from_file, _ = load_draft_state(state_path, _id_map())
    in_memory = build_draft_state(picks, my_slot=7, league=league, id_map=_id_map())
    assert in_memory == from_file


def test_build_draft_state_bad_slot_raises() -> None:
    from projections.draft.assistant.state import build_draft_state
    from projections.draft.league_config import LeagueConfig
    from projections.schemas import RosterSlot, Ruleset

    league = LeagueConfig(
        name="t", n_teams=12,
        roster_slots={RosterSlot.QB: 1, RosterSlot.BENCH: 1}, ruleset=Ruleset.espn_ppr(),
    )
    with pytest.raises(ValueError, match="my_slot"):
        build_draft_state([], my_slot=99, league=league, id_map=_id_map())
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest -n0 tests/test_draft/test_assistant_state.py::test_build_draft_state_matches_load_draft_state -v`
Expected: FAIL with `ImportError: cannot import name 'build_draft_state'`.

- [ ] **Step 3: Implement the extraction**

In `src/projections/draft/assistant/state.py`, add `Sequence` to the typing imports at the top:

```python
from collections.abc import Sequence
```

Add this function above `load_draft_state`:

```python
def build_draft_state(
    picks: Sequence[str],
    *,
    my_slot: int,
    league: LeagueConfig,
    id_map: pd.DataFrame,
) -> DraftState:
    """Build a `DraftState` from in-memory picks (the file-free half of load_draft_state).

    Raises ValueError on: my_slot out of range, a malformed/duplicate gsis_id, or
    one of *my* picks being absent from id_map (unknown position).
    """
    if not 1 <= my_slot <= league.n_teams:
        raise ValueError(f"my_slot must be in 1..{league.n_teams}; got {my_slot}")

    parsed = tuple(validate_gsis_id(str(p)) for p in picks)
    if len(set(parsed)) != len(parsed):
        raise ValueError("draft state has a duplicate pick (a player drafted twice)")

    pos_by_id = dict(zip(id_map["gsis_id"], id_map["position"], strict=False))
    my_roster: list[Position] = []
    for index, gid in enumerate(parsed):
        pick_number = index + 1
        if slot_for(pick_number, league.n_teams) != my_slot:
            continue
        if gid not in pos_by_id:
            raise ValueError(
                f"my pick {gid} (pick #{pick_number}) is absent from id_map; "
                "cannot resolve its position for roster accounting"
            )
        my_roster.append(Position(pos_by_id[gid]))

    return DraftState(
        my_slot=my_slot,
        n_teams=league.n_teams,
        rounds=league.roster_size,
        picks=parsed,
        my_roster=tuple(my_roster),
    )
```

Now replace the body of `load_draft_state` after the `league = ...` line so it delegates:

```python
    league = LeagueConfig.model_validate_json(Path(data["league_config"]).read_text())
    state = build_draft_state(
        data["picks"], my_slot=int(data["my_slot"]), league=league, id_map=id_map
    )
    return state, league
```

(Delete the now-duplicated my_slot/picks/dup/roster block that lived between `league = ...` and the old `return`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -n0 tests/test_draft/test_assistant_state.py -v`
Expected: PASS (all existing tests + the two new ones — the existing error-message matches still hold because `build_draft_state` reuses them).

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/state.py tests/test_draft/test_assistant_state.py
git commit -m "refactor(draft): extract build_draft_state from load_draft_state

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `build_session_strategy` seam (+ cli delegation)

**Files:**
- Create: `src/projections/draft/assistant/live.py`
- Modify: `src/projections/draft/assistant/cli.py`
- Test: `tests/test_draft/test_assistant_live.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_draft/test_assistant_live.py`:

```python
"""Tests for the LiveDraftSession controller and its helpers."""

from __future__ import annotations

import pytest

from projections.draft.assistant.live import build_session_strategy
from projections.draft.assistant.strategy import (
    NowOrNeverStrategy,
    RawVorpStrategy,
    SeasonValueStrategy,
    SeasonValueTimingStrategy,
)
from projections.draft.league_config import LeagueConfig
from projections.schemas import RosterSlot, Ruleset


def _league() -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=12,
        roster_slots={
            RosterSlot.QB: 1, RosterSlot.RB: 2, RosterSlot.WR: 2,
            RosterSlot.FLEX: 1, RosterSlot.BENCH: 5,
        },
        ruleset=Ruleset.espn_ppr(),
    )


def test_build_session_strategy_analytic_types() -> None:
    league = _league()
    assert isinstance(
        build_session_strategy("raw_vorp", league=league, sigma=None,
                               availability=None, n_sims=1, base_seed=0),
        RawVorpStrategy,
    )
    assert isinstance(
        build_session_strategy("now_or_never", league=league, sigma=None,
                               availability=None, n_sims=1, base_seed=0),
        NowOrNeverStrategy,
    )


def test_build_session_strategy_mc_requires_availability() -> None:
    league = _league()
    with pytest.raises(ValueError, match="availability"):
        build_session_strategy("season_value", league=league, sigma=None,
                               availability=None, n_sims=1, base_seed=0)


def test_build_session_strategy_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        build_session_strategy("nope", league=_league(), sigma=None,
                               availability=None, n_sims=1, base_seed=0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest -n0 tests/test_draft/test_assistant_live.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'projections.draft.assistant.live'`.

- [ ] **Step 3: Create `live.py` with the seam**

Create `src/projections/draft/assistant/live.py`:

```python
"""LiveDraftSession — the live draft board's controller (testable; Streamlit-free).

Holds the mutable draft truth (ordered picks + league + data) and delegates every
decision to existing engine functions. scripts/draft_board.py is a thin view over it.
"""

from __future__ import annotations

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.strategy import (
    DraftStrategy,
    NowOrNeverStrategy,
    RawVorpStrategy,
    SeasonValueStrategy,
    SeasonValueTimingStrategy,
)
from projections.draft.assistant.survival import LogisticSurvival, default_sigma
from projections.draft.league_config import LeagueConfig

# Strategy names the board's dropdown offers (season_value_var is in STRATEGY_KEYS
# but excluded — its A/B showed no draft benefit; see the spec §2 / memory).
BOARD_STRATEGIES: tuple[str, ...] = (
    "now_or_never",
    "raw_vorp",
    "season_value",
    "season_value_timing",
)


def build_session_strategy(
    name: str,
    *,
    league: LeagueConfig,
    sigma: float | None,
    availability: PlayerAvailability | None,
    n_sims: int,
    base_seed: int,
) -> DraftStrategy:
    """Map a strategy name (+ live params) to a DraftStrategy.

    Shared by the sidebar dropdown and the resume path. MC strategies
    (`season_value*`) require a non-null `availability` and fail loud otherwise.
    """
    if name == "raw_vorp":
        return RawVorpStrategy()
    if name == "now_or_never":
        spread = default_sigma(league.n_teams) if sigma is None else sigma
        return NowOrNeverStrategy(LogisticSurvival(sigma=spread))
    if name in ("season_value", "season_value_var", "season_value_timing"):
        if availability is None:
            raise ValueError(f"strategy {name!r} requires availability data (None given)")
        if name == "season_value":
            return SeasonValueStrategy(availability, n_sims=n_sims, base_seed=base_seed)
        if name == "season_value_var":
            return SeasonValueStrategy(
                availability, n_sims=n_sims, base_seed=base_seed, risk_aware=True
            )
        spread = default_sigma(league.n_teams) if sigma is None else sigma
        return SeasonValueTimingStrategy(
            availability, n_sims=n_sims, base_seed=base_seed,
            survival=LogisticSurvival(sigma=spread),
        )
    raise ValueError(f"unknown strategy {name!r}")
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest -n0 tests/test_draft/test_assistant_live.py -v`
Expected: PASS.

- [ ] **Step 5: Refactor `cli._build_strategy` to delegate**

In `src/projections/draft/assistant/cli.py`, replace the `_build_strategy` function body with a delegation (keeps the analytic-only signature the CLI uses today):

```python
def _build_strategy(name: str, n_teams: int, sigma: float | None) -> DraftStrategy:
    from projections.draft.assistant.live import build_session_strategy
    from projections.draft.league_config import LeagueConfig
    from projections.schemas import RosterSlot, Ruleset

    # _build_strategy is only called for the analytic strategies (raw_vorp/now_or_never),
    # which ignore availability/n_sims/base_seed; a minimal league carries n_teams + sigma.
    league = LeagueConfig(
        name="_", n_teams=n_teams,
        roster_slots={RosterSlot.QB: 1, RosterSlot.BENCH: 1}, ruleset=Ruleset.espn_ppr(),
    )
    return build_session_strategy(
        name, league=league, sigma=sigma, availability=None, n_sims=1, base_seed=0
    )
```

- [ ] **Step 6: Run the CLI tests to verify no behavior change**

Run: `pytest -n0 tests/test_draft/test_assistant_cli.py -v`
Expected: PASS (the constructed `now_or_never`/`raw_vorp` are identical).

- [ ] **Step 7: Commit**

```bash
git add src/projections/draft/assistant/live.py src/projections/draft/assistant/cli.py tests/test_draft/test_assistant_live.py
git commit -m "feat(draft): build_session_strategy seam (cli delegates to it)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `LiveDraftSession` construction + pick-timing properties

**Files:**
- Modify: `src/projections/draft/assistant/live.py`
- Test: `tests/test_draft/test_assistant_live.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_draft/test_assistant_live.py` (top imports + helpers + tests):

```python
import pandas as pd

from projections.draft.assistant.live import LiveDraftSession
from projections.draft.assistant.strategy import DraftStrategy
from projections.draft.assistant.state import DraftState
from projections.schemas import _PYARROW_STR, GsisId


def _id_map() -> pd.DataFrame:
    ids = [f"00-000{i:04d}" for i in range(1, 40)]
    return pd.DataFrame(
        {
            "gsis_id": pd.array(ids, dtype=_PYARROW_STR),
            "position": pd.array(["RB", "WR", "QB", "TE"] * 9 + ["RB", "WR", "QB"], dtype=_PYARROW_STR),
            "full_name": pd.array([f"P{i}" for i in range(1, 40)], dtype=_PYARROW_STR),
            "team": pd.array(["KC"] * 39, dtype=_PYARROW_STR),
        }
    )


def _pool() -> pd.DataFrame:
    ids = [f"00-000{i:04d}" for i in range(1, 40)]
    return pd.DataFrame(
        {
            "gsis_id": pd.array(ids, dtype=_PYARROW_STR),
            "position": pd.array(["RB", "WR", "QB", "TE"] * 9 + ["RB", "WR", "QB"], dtype=_PYARROW_STR),
            "season_mean_fpts": [300.0 - i for i in range(39)],
            "vorp": [150.0 - i for i in range(39)],
            "replacement_fpts": [100.0] * 39,
            "consensus_adp": pd.array([float(i + 1) for i in range(39)], dtype=pd.Float64Dtype()),
        }
    )


class _FakeStrategy:
    """A DraftStrategy that returns the eligible pool sorted by vorp (no MC)."""

    def recommend(self, state: DraftState, pool: pd.DataFrame, config) -> pd.DataFrame:  # type: ignore[no-untyped-def]
        avail = pool[~pool["gsis_id"].isin(state.drafted_ids)].copy()
        avail = avail.sort_values("vorp", ascending=False).reset_index(drop=True)
        avail["p_available_next"] = pd.array([pd.NA] * len(avail), dtype=pd.Float64Dtype())
        avail["fills_starting_slot"] = True
        avail["score"] = avail["vorp"]
        avail["rank"] = pd.array(range(1, len(avail) + 1), dtype=pd.Int64Dtype())
        return avail


def _session(picks: list[str] | None = None, mode: str = "copilot") -> LiveDraftSession:
    return LiveDraftSession(
        league=_league(),
        my_slot=7,
        id_map=_id_map(),
        pool=_pool(),
        strategy=_FakeStrategy(),
        strategy_name="fake",
        mode=mode,
        adp_jitter=8.0,
        base_seed=0,
        picks=list(picks or []),
    )


def test_current_pick_and_my_pick_progression() -> None:
    s = _session()
    assert s.current_pick == 1
    assert not s.is_my_pick  # slot 1 on the clock, I'm slot 7
    s.picks = [f"00-000{i:04d}" for i in range(1, 7)]  # 6 picks made → pick 7 is mine
    assert s.current_pick == 7
    assert s.is_my_pick
    assert s.round_and_slot() == (1, 7)


def test_next_pick_number_snakes() -> None:
    s = _session(picks=[f"00-000{i:04d}" for i in range(1, 7)])  # standing at pick 7 (mine)
    # 12 teams, slot 7: next pick after #7 is #18 (snake).
    assert s.next_pick_number == 18


def test_is_complete_when_roster_full() -> None:
    s = _session()
    total = s.league.n_teams * s.league.roster_size
    s.picks = [f"00-000{i:04d}" for i in range(1, total + 1)]
    assert s.is_complete
```

(Add `_league` import/helper — reuse the `_league()` already defined in Task 2's test block.)

- [ ] **Step 2: Run to verify it fails**

Run: `pytest -n0 tests/test_draft/test_assistant_live.py::test_current_pick_and_my_pick_progression -v`
Expected: FAIL with `ImportError: cannot import name 'LiveDraftSession'`.

- [ ] **Step 3: Implement the dataclass + properties**

Add to `src/projections/draft/assistant/live.py` (imports at top, then the class):

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from projections.draft.assistant.pick_timing import my_next_pick, slot_for
from projections.draft.assistant.state import DraftState, build_draft_state
from projections.schemas import GsisId
```

```python
@dataclass
class LiveDraftSession:
    """Mutable, Streamlit-free controller for one live/mock snake draft."""

    league: LeagueConfig
    my_slot: int
    id_map: pd.DataFrame
    pool: pd.DataFrame
    strategy: DraftStrategy
    strategy_name: str
    mode: Literal["copilot", "mock"] = "copilot"
    adp_jitter: float = 8.0
    base_seed: int = 0
    n_sims: int = 300
    sigma: float | None = None
    season: int = 2026
    picks: list[GsisId] = field(default_factory=list)
    # Persistence-only paths (defaults keep core tests path-free).
    league_config_path: Path = field(default=Path("."))
    vorp_path: Path = field(default=Path("."))
    id_map_path: Path = field(default=Path("."))
    data_root: Path = field(default=Path("data"))

    def state(self) -> DraftState:
        """Rebuild the immutable engine snapshot from current picks (cheap; O(picks))."""
        return build_draft_state(
            self.picks, my_slot=self.my_slot, league=self.league, id_map=self.id_map
        )

    @property
    def current_pick(self) -> int:
        return len(self.picks) + 1

    @property
    def is_complete(self) -> bool:
        return len(self.picks) >= self.league.n_teams * self.league.roster_size

    @property
    def on_clock_slot(self) -> int:
        return slot_for(self.current_pick, self.league.n_teams)

    @property
    def is_my_pick(self) -> bool:
        return not self.is_complete and self.on_clock_slot == self.my_slot

    @property
    def next_pick_number(self) -> int | None:
        return my_next_pick(
            self.current_pick, self.my_slot, self.league.n_teams, self.league.roster_size
        )

    def round_and_slot(self) -> tuple[int, int]:
        rnd = (self.current_pick - 1) // self.league.n_teams + 1
        return rnd, self.on_clock_slot
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest -n0 tests/test_draft/test_assistant_live.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/live.py tests/test_draft/test_assistant_live.py
git commit -m "feat(draft): LiveDraftSession construction + pick-timing properties

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `record_pick` / `undo` / `available_pool`

**Files:**
- Modify: `src/projections/draft/assistant/live.py`
- Test: `tests/test_draft/test_assistant_live.py`

- [ ] **Step 1: Write the failing test**

```python
def test_record_pick_appends_and_rejects_duplicate() -> None:
    s = _session()
    s.record_pick("00-0000001")
    assert s.picks == ["00-0000001"]
    with pytest.raises(ValueError, match="already drafted"):
        s.record_pick("00-0000001")


def test_record_pick_rejects_absent_from_id_map() -> None:
    s = _session()
    with pytest.raises(ValueError, match="id_map"):
        s.record_pick("00-0009999")


def test_undo_pops_last() -> None:
    s = _session()
    s.record_pick("00-0000001")
    s.record_pick("00-0000002")
    assert s.undo() == "00-0000002"
    assert s.picks == ["00-0000001"]
    s.undo()
    assert s.undo() is None  # empty → None


def test_available_pool_excludes_drafted() -> None:
    s = _session(picks=["00-0000001", "00-0000002"])
    avail = s.available_pool()
    assert "00-0000001" not in set(avail["gsis_id"])
    assert len(avail) == len(_pool()) - 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest -n0 tests/test_draft/test_assistant_live.py::test_record_pick_appends_and_rejects_duplicate -v`
Expected: FAIL with `AttributeError: 'LiveDraftSession' object has no attribute 'record_pick'`.

- [ ] **Step 3: Implement**

Add `from projections.schemas import GsisId, validate_gsis_id` (extend the existing schemas import) and these methods to `LiveDraftSession`:

```python
    def available_pool(self) -> pd.DataFrame:
        drafted = self.state().drafted_ids
        return self.pool[~self.pool["gsis_id"].isin(drafted)].reset_index(drop=True)

    def record_pick(self, gsis_id: str) -> None:
        gid = validate_gsis_id(str(gsis_id))
        if gid in self.state().drafted_ids:
            raise ValueError(f"{gid} already drafted")
        if gid not in set(self.id_map["gsis_id"]):
            raise ValueError(f"{gid} absent from id_map (cannot resolve position)")
        self.picks.append(gid)

    def undo(self) -> GsisId | None:
        return self.picks.pop() if self.picks else None
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest -n0 tests/test_draft/test_assistant_live.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/live.py tests/test_draft/test_assistant_live.py
git commit -m "feat(draft): record_pick/undo/available_pool on LiveDraftSession

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: `recommendation` passthrough + `suggested_pick`

**Files:**
- Modify: `src/projections/draft/assistant/live.py`
- Test: `tests/test_draft/test_assistant_live.py`

- [ ] **Step 1: Write the failing test**

```python
def test_recommendation_delegates_to_strategy() -> None:
    s = _session()
    rec = s.recommendation()
    assert list(rec["gsis_id"])[0] == "00-0000001"  # highest vorp, undrafted
    assert "rank" in rec.columns


def test_recommendation_empty_when_complete() -> None:
    s = _session()
    total = s.league.n_teams * s.league.roster_size
    s.picks = [f"00-000{i:04d}" for i in range(1, total + 1)][: len(_pool())]
    # Drafting the whole pool leaves nothing available.
    s.picks = list(_pool()["gsis_id"])
    assert s.recommendation().empty


def test_suggested_pick_is_deterministic_and_low_adp() -> None:
    s = _session()
    first = s.suggested_pick()
    again = s.suggested_pick()
    assert first == again  # stable across reruns for one board state
    # adp_jitter is small relative to ADP spacing → lowest-ADP player wins.
    assert first == "00-0000001"


def test_suggested_pick_none_when_pool_empty() -> None:
    s = _session(picks=list(_pool()["gsis_id"]))
    assert s.suggested_pick() is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest -n0 tests/test_draft/test_assistant_live.py::test_recommendation_delegates_to_strategy -v`
Expected: FAIL with `AttributeError: ... 'recommendation'`.

- [ ] **Step 3: Implement**

Add `import numpy as np` to the top imports, `from projections.draft.assistant.opponent import bot_pick`, and these methods:

```python
    def recommendation(self) -> pd.DataFrame:
        return self.strategy.recommend(self.state(), self.pool, self.league)

    def suggested_pick(self) -> GsisId | None:
        avail = self.available_pool()
        if avail.empty:
            return None
        # Deterministic per board state → stable across Streamlit reruns, reproducible
        # in mock mode. (Re-deriving the seed each call is intentional; no stored RNG.)
        rng = np.random.default_rng([self.base_seed, self.current_pick])
        return bot_pick(avail, rng, adp_jitter=self.adp_jitter)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest -n0 tests/test_draft/test_assistant_live.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/live.py tests/test_draft/test_assistant_live.py
git commit -m "feat(draft): recommendation passthrough + deterministic suggested_pick

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: `my_roster_view` (RosterView) + `best_available_by_position` + `attach_names`

**Files:**
- Modify: `src/projections/draft/assistant/live.py`
- Test: `tests/test_draft/test_assistant_live.py`

- [ ] **Step 1: Write the failing test**

```python
from projections.schemas import Position, RosterSlot


def test_my_roster_view_assigns_slots_and_open_needs() -> None:
    # Make picks 7 (RB) and 18 (mine, snake) land in my roster.
    picks = [f"00-000{i:04d}" for i in range(1, 7)]  # 6 opponent picks
    s = _session(picks=picks)
    s.record_pick("00-0000007")  # pick #7 → mine (RB per id_map pattern)
    view = s.my_roster_view()
    assert len(view.filled) == 1
    assert view.filled.iloc[0]["position"] == "RB"
    assert view.filled.iloc[0]["full_name"] == "P7"
    # An RB slot is now consumed; one RB starter slot remains open (RB:2).
    assert view.open_slots[RosterSlot.RB] == 1


def test_best_available_by_position_top_n() -> None:
    s = _session()
    best = s.best_available_by_position(top=2)
    assert set(best) <= set(Position)
    rb = best[Position.RB]
    assert len(rb) == 2
    assert list(rb["vorp"]) == sorted(rb["vorp"], reverse=True)


def test_attach_names_inserts_full_name() -> None:
    from projections.draft.assistant.live import attach_names

    rec = _session().recommendation()
    named = attach_names(rec, _id_map())
    assert "full_name" in named.columns
    assert named.iloc[0]["full_name"] == "P1"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest -n0 tests/test_draft/test_assistant_live.py::test_my_roster_view_assigns_slots_and_open_needs -v`
Expected: FAIL with `AttributeError: ... 'my_roster_view'`.

- [ ] **Step 3: Implement**

Add to the imports:

```python
from collections import Counter

from projections.draft.roster_eligibility import (
    FLEX_ELIGIBLE,
    SUPER_FLEX_ELIGIBLE,
    bench_eligible_positions,
)
from projections.schemas import Position, RosterSlot
```

Add the `RosterView` dataclass (top-level, above `LiveDraftSession`):

```python
@dataclass
class RosterView:
    """My current roster: filled slots + remaining open starting slots."""

    filled: pd.DataFrame  # columns: slot, gsis_id, full_name, position
    open_slots: dict[RosterSlot, int]
```

Add the module-level `attach_names`:

```python
def attach_names(df: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with a `full_name` column from id_map (— when unknown)."""
    names = dict(zip(id_map["gsis_id"], id_map["full_name"], strict=False))
    out = df.copy()
    out.insert(min(1, len(out.columns)), "full_name", [names.get(g, "—") for g in out["gsis_id"]])
    return out
```

Add the methods to `LiveDraftSession`:

```python
    def my_roster_view(self) -> RosterView:
        state = self.state()
        names = dict(zip(self.id_map["gsis_id"], self.id_map["full_name"], strict=False))
        benchable = bench_eligible_positions(self.league.roster_slots)
        open_: Counter[RosterSlot] = Counter(
            {s: c for s, c in self.league.roster_slots.items() if s != RosterSlot.IR and c > 0}
        )
        rows: list[dict[str, str]] = []
        for gid, pos in zip(state.my_pick_ids, state.my_roster, strict=False):
            own = RosterSlot(pos.value)
            for slot, eligible in (
                (own, True),
                (RosterSlot.FLEX, pos in FLEX_ELIGIBLE),
                (RosterSlot.SUPER_FLEX, pos in SUPER_FLEX_ELIGIBLE),
                (RosterSlot.BENCH, pos in benchable),
            ):
                if eligible and open_.get(slot, 0) > 0:
                    open_[slot] -= 1
                    rows.append(
                        {"slot": slot.value, "gsis_id": gid,
                         "full_name": names.get(gid, "—"), "position": pos.value}
                    )
                    break
        filled = pd.DataFrame(rows, columns=["slot", "gsis_id", "full_name", "position"])
        open_slots = {s: c for s, c in open_.items() if c > 0 and s != RosterSlot.BENCH}
        return RosterView(filled=filled, open_slots=open_slots)

    def best_available_by_position(self, top: int) -> dict[Position, pd.DataFrame]:
        avail = self.available_pool()
        out: dict[Position, pd.DataFrame] = {}
        for pos in Position:
            sub = avail[avail["position"] == pos.value].sort_values("vorp", ascending=False)
            if not sub.empty:
                out[pos] = sub.head(top).reset_index(drop=True)
        return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest -n0 tests/test_draft/test_assistant_live.py -v`
Expected: PASS.

- [ ] **Step 5: Phase-1 gate + commit**

Run all four gates (Phase 1 is the testable core — run them fully here):

```bash
pytest -n0 tests/test_draft/test_assistant_live.py tests/test_draft/test_assistant_state.py tests/test_draft/test_assistant_cli.py -v
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: all PASS / zero violations. Then:

```bash
git add src/projections/draft/assistant/live.py tests/test_draft/test_assistant_live.py
git commit -m "feat(draft): my_roster_view + best_available_by_position + attach_names

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 2 — Streamlit view (co-pilot mode)

### Task 7: Add the `[ui]` optional dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the extra**

In `pyproject.toml`, under `[project.optional-dependencies]`, add a `ui` group after `dev`:

```toml
ui = [
    "streamlit>=1.36",
]
```

- [ ] **Step 2: Install it**

Run: `pip install -e ".[ui]"`
Expected: streamlit installs without disturbing the core deps.

- [ ] **Step 3: Verify import**

Run: `python -c "import streamlit; print(streamlit.__version__)"`
Expected: prints a version `>= 1.36`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build(ui): add streamlit under the [ui] optional-dependencies extra

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Streamlit entry — setup sidebar + session bootstrap + status bar

**Files:**
- Create: `scripts/draft_board.py`

This task is the Streamlit shell. The controller it drives is already fully tested; verification here is **manual** (Streamlit UI), plus the controller tests already guarantee the logic. Keep the file thin — no draft logic, only rendering + `st.session_state` plumbing.

- [ ] **Step 1: Create the file**

Create `scripts/draft_board.py`:

```python
"""Streamlit live draft board (Draft Assistant Slice 3).

Thin view over projections.draft.assistant.live.LiveDraftSession. Run with:
    streamlit run scripts/draft_board.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.assistant.live import (
    BOARD_STRATEGIES,
    LiveDraftSession,
    attach_names,
    build_session_strategy,
)
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, IdMapSchema, VorpTableSchema

_DEFAULT_VORP = "data/consensus_vorp_2026.parquet"
_DEFAULT_ID_MAP = "data/raw/id_map.parquet"
_DEFAULT_LEAGUE = "configs/league_espn_ppr_12team_skill.json"

_MC_STRATEGIES = ("season_value", "season_value_timing")


def _load_inputs(vorp_path: Path, id_map_path: Path, league_path: Path):  # type: ignore[no-untyped-def]
    id_map = IdMapSchema.validate(pd.read_parquet(id_map_path))
    pool = pd.read_parquet(vorp_path)
    pool["gsis_id"] = pool["gsis_id"].astype(_PYARROW_STR)
    pool = VorpTableSchema.validate(pool)
    league = LeagueConfig.model_validate_json(Path(league_path).read_text())
    return id_map, pool, league


def _build_session(
    *, vorp_path: Path, id_map_path: Path, league_path: Path,
    my_slot: int, mode: str, strategy_name: str, n_sims: int,
    adp_jitter: float, season: int, data_root: Path,
) -> LiveDraftSession:
    id_map, pool, league = _load_inputs(vorp_path, id_map_path, league_path)
    availability = None
    if strategy_name in _MC_STRATEGIES:
        availability = load_store_availability(pool, season=season, data_root=data_root)
    strategy = build_session_strategy(
        strategy_name, league=league, sigma=None,
        availability=availability, n_sims=n_sims, base_seed=0,
    )
    return LiveDraftSession(
        league=league, my_slot=my_slot, id_map=id_map, pool=pool,
        strategy=strategy, strategy_name=strategy_name, mode=mode,  # type: ignore[arg-type]
        adp_jitter=adp_jitter, n_sims=n_sims, season=season,
        league_config_path=Path(league_path), vorp_path=vorp_path,
        id_map_path=id_map_path, data_root=data_root,
    )


def _sidebar() -> None:
    st.sidebar.header("⚙ Setup")
    mode = st.sidebar.radio("Mode", ["copilot", "mock"], index=0,
                            format_func=lambda m: "Co-pilot (live)" if m == "copilot" else "Mock")
    vorp_path = st.sidebar.text_input("Consensus VORP parquet", _DEFAULT_VORP)
    id_map_path = st.sidebar.text_input("id_map parquet", _DEFAULT_ID_MAP)
    league_path = st.sidebar.text_input("League config JSON", _DEFAULT_LEAGUE)
    my_slot = st.sidebar.number_input("My draft slot", min_value=1, max_value=32, value=1)
    strategy_name = st.sidebar.selectbox("Strategy", BOARD_STRATEGIES, index=0)
    n_sims = st.sidebar.number_input("n_sims (MC strategies)", min_value=50, max_value=2000,
                                     value=300, step=50)
    adp_jitter = st.sidebar.slider("ADP jitter", 0.0, 20.0, 8.0, 0.5)
    season = st.sidebar.number_input("Season", min_value=2020, max_value=2030, value=2026)

    if st.sidebar.button("Start / restart draft", type="primary"):
        try:
            st.session_state["session"] = _build_session(
                vorp_path=Path(vorp_path), id_map_path=Path(id_map_path),
                league_path=Path(league_path), my_slot=int(my_slot), mode=mode,
                strategy_name=strategy_name, n_sims=int(n_sims),
                adp_jitter=float(adp_jitter), season=int(season), data_root=Path("data"),
            )
        except Exception as exc:  # noqa: BLE001 — surface any setup failure to the user
            st.sidebar.error(f"Setup failed: {exc}")


def _status_bar(s: LiveDraftSession) -> None:
    if s.is_complete:
        st.subheader("✅ Draft complete")
        return
    rnd, slot = s.round_and_slot()
    who = "YOU" if s.is_my_pick else f"Team {slot}"
    nxt = s.next_pick_number
    until = "" if nxt is None else f" · your next pick: #{nxt}"
    st.subheader(f"Pick {rnd}.{s.on_clock_slot:02d} (#{s.current_pick}) · on the clock: {who}{until}")


def main() -> None:
    st.set_page_config(page_title="Draft Board", layout="wide")
    st.title("🏈 Live Draft Board")
    _sidebar()

    s: LiveDraftSession | None = st.session_state.get("session")
    if s is None:
        st.info("Configure the draft in the sidebar and click **Start / restart draft**.")
        return
    _status_bar(s)
    st.caption(f"Mode: {s.mode} · strategy: {s.strategy_name} · {len(s.picks)} picks made")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Type-check + lint the new file**

Run: `mypy src scripts/draft_board.py && ruff check scripts/draft_board.py && ruff format --check scripts/draft_board.py`
Expected: zero violations.

- [ ] **Step 3: Manual smoke**

Run: `streamlit run scripts/draft_board.py`
Expected: the page loads; configuring inputs + "Start / restart draft" shows the status bar with `Pick 1.01 · on the clock: Team 1` (or `YOU` at slot 1). Stop with Ctrl-C.

- [ ] **Step 4: Commit**

```bash
git add scripts/draft_board.py
git commit -m "feat(ui): draft board shell — setup sidebar, session bootstrap, status bar

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: Three-column board + search-and-record + smart-assist confirm

**Files:**
- Modify: `scripts/draft_board.py`

- [ ] **Step 1: Add the render helpers + record/confirm wiring**

In `scripts/draft_board.py`, add these functions above `main`:

```python
@st.cache_data(show_spinner=False)
def _cached_recommendation(
    _s_id: int, picks: tuple[str, ...], strategy_name: str, n_sims: int, sigma: float | None
) -> pd.DataFrame:
    """Cache MC recommendations on every result-affecting param (picks/strategy/n_sims/sigma).

    `_s_id` ties the cache to the current session object (id()); the leading underscore
    tells Streamlit not to hash the (unhashable) session itself.
    """
    s: LiveDraftSession = st.session_state["session"]
    return s.recommendation()


def _record_and_rerun(s: LiveDraftSession, gsis_id: str) -> None:
    try:
        s.record_pick(gsis_id)
    except ValueError as exc:
        st.warning(str(exc))
        return
    _autosave(s)  # defined in Task 13; until then this is a no-op shim
    st.rerun()


def _search_box(s: LiveDraftSession) -> None:
    query = st.text_input("🔍 Record a pick — search a player", key=f"search_{s.current_pick}")
    if not query:
        return
    id_map = s.id_map
    drafted = s.state().drafted_ids
    hits = id_map[
        id_map["full_name"].str.contains(query, case=False, na=False)
        & ~id_map["gsis_id"].isin(drafted)
    ].head(8)
    for row in hits.itertuples(index=False):
        team = "" if pd.isna(row.team) else f" · {row.team}"
        if st.button(f"{row.full_name} ({row.position}{team})", key=f"pick_{row.gsis_id}"):
            _record_and_rerun(s, str(row.gsis_id))


def _board_log_col(s: LiveDraftSession) -> None:
    st.markdown("**Board / pick log**")
    names = dict(zip(s.id_map["gsis_id"], s.id_map["full_name"]))
    rows = []
    for i, gid in enumerate(s.picks):
        pick_no = i + 1
        slot = s.on_clock_slot if pick_no == s.current_pick else None
        from projections.draft.assistant.pick_timing import slot_for
        owner = slot_for(pick_no, s.league.n_teams)
        rows.append({"#": pick_no, "slot": owner, "player": names.get(gid, "—"),
                     "mine": "★" if owner == s.my_slot else ""})
    st.dataframe(pd.DataFrame(rows), height=520, hide_index=True)


def _recommend_col(s: LiveDraftSession) -> None:
    st.markdown("**★ Recommendations**")
    if s.is_complete:
        st.success("Draft complete.")
        return
    if s.mode == "copilot" and not s.is_my_pick:
        sug = s.suggested_pick()
        if sug is not None:
            name = dict(zip(s.id_map["gsis_id"], s.id_map["full_name"])).get(sug, sug)
            st.info(f"Opponent on the clock. ADP suggests: **{name}**")
            if st.button(f"Confirm pick: {name}", type="primary"):
                _record_and_rerun(s, str(sug))
        st.caption("…or search below to record a different pick.")
        return
    with st.spinner("Scoring candidates…"):
        rec = _cached_recommendation(
            id(s), tuple(s.picks), s.strategy_name, s.n_sims, s.sigma
        )
    named = attach_names(rec, s.id_map)
    cols = ["rank", "full_name", "position", "vorp", "consensus_adp",
            "p_available_next", "score", "fills_starting_slot"]
    st.dataframe(named[cols].head(20), height=480, hide_index=True)


def _roster_col(s: LiveDraftSession) -> None:
    st.markdown("**My Roster**")
    view = s.my_roster_view()
    st.dataframe(view.filled[["slot", "full_name", "position"]], hide_index=True)
    if view.open_slots:
        st.caption("Open starting slots: " + ", ".join(
            f"{slot.value}×{n}" for slot, n in view.open_slots.items()))
    st.markdown("**Best available by position**")
    best = s.best_available_by_position(top=3)
    for pos, sub in best.items():
        named = attach_names(sub, s.id_map)
        st.caption(f"{pos.value}: " + ", ".join(
            f"{r.full_name} ({r.vorp:.0f})" for r in named.itertuples(index=False)))
```

- [ ] **Step 2: Wire the columns into `main`**

Replace the `_status_bar(s)` + caption block in `main` with:

```python
    _status_bar(s)
    _search_box(s)
    left, center, right = st.columns([1.1, 2.0, 1.3])
    with left:
        _board_log_col(s)
    with center:
        _recommend_col(s)
    with right:
        _roster_col(s)
    st.caption(f"Mode: {s.mode} · strategy: {s.strategy_name} · {len(s.picks)} picks made")
```

Add a temporary no-op `_autosave` shim near the top (Task 13 replaces it):

```python
def _autosave(s: LiveDraftSession) -> None:
    """No-op until Task 13 wires real autosave."""
```

- [ ] **Step 3: Type-check + lint**

Run: `mypy src scripts/draft_board.py && ruff check scripts/draft_board.py && ruff format --check scripts/draft_board.py`
Expected: zero violations.

- [ ] **Step 4: Manual smoke**

Run: `streamlit run scripts/draft_board.py`
Verify: start a draft at slot 1; the three columns render; searching a name shows clickable buttons; clicking records the pick and the board updates; for an opponent pick (set my_slot to e.g. 6 and make a few picks), the center column shows the ADP suggestion with a Confirm button.

- [ ] **Step 5: Commit**

```bash
git add scripts/draft_board.py
git commit -m "feat(ui): three-column board, search-and-record, smart-assist confirm

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 3 — Mock mode

### Task 10: `mock_advance_to_my_pick` + `roster_scorecard`

**Files:**
- Modify: `src/projections/draft/assistant/live.py`
- Test: `tests/test_draft/test_assistant_live.py`

- [ ] **Step 1: Write the failing test**

```python
def test_mock_advance_stops_at_my_pick() -> None:
    s = _session(mode="mock")  # my_slot=7
    made = s.mock_advance_to_my_pick()
    assert len(made) == 6           # bots take picks 1..6
    assert s.is_my_pick            # standing at pick 7 (mine)
    assert s.current_pick == 7


def test_mock_advance_raises_in_copilot() -> None:
    s = _session(mode="copilot")
    with pytest.raises(RuntimeError, match="mock"):
        s.mock_advance_to_my_pick()


def test_roster_scorecard_matches_optimal_lineup() -> None:
    from projections.draft.assistant.roster_score import optimal_lineup_points

    picks = [f"00-000{i:04d}" for i in range(1, 7)]
    s = _session(picks=picks)
    s.record_pick("00-0000007")  # my RB
    mine = s.pool[s.pool["gsis_id"].isin(s.state().my_pick_ids)]
    expected = optimal_lineup_points(mine, s.league.roster_slots)
    assert s.roster_scorecard() == expected
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest -n0 tests/test_draft/test_assistant_live.py::test_mock_advance_stops_at_my_pick -v`
Expected: FAIL with `AttributeError: ... 'mock_advance_to_my_pick'`.

- [ ] **Step 3: Implement**

Add `from projections.draft.assistant.roster_score import optimal_lineup_points` and these methods:

```python
    def mock_advance_to_my_pick(self) -> list[GsisId]:
        if self.mode != "mock":
            raise RuntimeError("mock_advance_to_my_pick is only valid in mock mode")
        made: list[GsisId] = []
        while not self.is_complete and not self.is_my_pick:
            gid = self.suggested_pick()
            if gid is None:
                break
            self.record_pick(gid)
            made.append(gid)
        return made

    def roster_scorecard(self) -> float:
        mine = self.pool[self.pool["gsis_id"].isin(self.state().my_pick_ids)]
        return optimal_lineup_points(mine, self.league.roster_slots)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest -n0 tests/test_draft/test_assistant_live.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/live.py tests/test_draft/test_assistant_live.py
git commit -m "feat(draft): mock_advance_to_my_pick + roster_scorecard

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 11: Mock-mode view wiring

**Files:**
- Modify: `scripts/draft_board.py`

- [ ] **Step 1: Add the advance button + scorecard**

In `scripts/draft_board.py`, add a mock-controls helper:

```python
def _mock_controls(s: LiveDraftSession) -> None:
    if s.mode != "mock":
        return
    if s.is_complete:
        st.success(f"Mock complete — your optimal-lineup score: **{s.roster_scorecard():.1f}**")
        return
    if not s.is_my_pick and st.button("⏭ Advance to my pick", type="secondary"):
        s.mock_advance_to_my_pick()
        _autosave(s)
        st.rerun()
```

- [ ] **Step 2: Call it in `main`**

Immediately after `_search_box(s)` in `main`, add:

```python
    _mock_controls(s)
```

- [ ] **Step 3: Type-check + lint**

Run: `mypy src scripts/draft_board.py && ruff check scripts/draft_board.py && ruff format --check scripts/draft_board.py`
Expected: zero violations.

- [ ] **Step 4: Manual smoke**

Run: `streamlit run scripts/draft_board.py`
Verify: pick **Mock** mode, slot 6, start; "Advance to my pick" auto-fills opponents up to your turn; drafting through the end shows the scorecard.

- [ ] **Step 5: Commit**

```bash
git add scripts/draft_board.py
git commit -m "feat(ui): mock-mode advance button + end-of-draft scorecard

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 4 — Persistence & polish

### Task 12: `to_state_dict` / `save` / `load`

**Files:**
- Modify: `src/projections/draft/assistant/live.py`
- Test: `tests/test_draft/test_assistant_live.py`

- [ ] **Step 1: Write the failing test**

```python
def test_to_state_dict_is_cli_compatible(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    from projections.draft.assistant.state import load_draft_state

    cfg = _league()
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(cfg.model_dump_json())
    s = _session(picks=["00-0000001", "00-0000002"])
    s.league_config_path = cfg_path
    d = s.to_state_dict()
    assert set(d) >= {"league_config", "my_slot", "picks", "mode", "strategy_name"}

    # load_draft_state must accept the saved superset unchanged.
    state_path = tmp_path / "session.json"
    state_path.write_text(json.dumps(d))
    loaded_state, _ = load_draft_state(state_path, _id_map())
    assert list(loaded_state.picks) == ["00-0000001", "00-0000002"]


def test_save_load_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg = _league()
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(cfg.model_dump_json())
    vorp_path = tmp_path / "vorp.parquet"
    _pool().to_parquet(vorp_path)
    id_map_path = tmp_path / "id_map.parquet"
    _id_map().to_parquet(id_map_path)

    s = _session(picks=["00-0000001"])
    s.league_config_path, s.vorp_path, s.id_map_path = cfg_path, vorp_path, id_map_path
    s.strategy_name = "raw_vorp"  # analytic → load needs no availability
    save_path = tmp_path / "session.json"
    s.save(save_path)

    from projections.draft.assistant.live import LiveDraftSession as L

    loaded = L.load(save_path, id_map=_id_map(), pool=_pool())
    assert loaded.picks == ["00-0000001"]
    assert loaded.strategy_name == "raw_vorp"
    assert loaded.my_slot == 7
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest -n0 tests/test_draft/test_assistant_live.py::test_to_state_dict_is_cli_compatible -v`
Expected: FAIL with `AttributeError: ... 'to_state_dict'`.

- [ ] **Step 3: Implement**

Add `import json` to the top imports, then these methods on `LiveDraftSession`:

```python
    def to_state_dict(self) -> dict[str, object]:
        """CLI-compatible superset: load_draft_state reads the required keys; the rest
        (mode/strategy/data paths) drive one-click resume."""
        return {
            "league_config": str(self.league_config_path),
            "my_slot": self.my_slot,
            "picks": list(self.picks),
            "mode": self.mode,
            "adp_jitter": self.adp_jitter,
            "strategy_name": self.strategy_name,
            "n_sims": self.n_sims,
            "sigma": self.sigma,
            "season": self.season,
            "vorp_table": str(self.vorp_path),
            "id_map": str(self.id_map_path),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_state_dict(), indent=2))

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        id_map: pd.DataFrame,
        pool: pd.DataFrame,
        data_root: Path = Path("data"),
    ) -> LiveDraftSession:
        """Rebuild a session from a saved state dict; strategy via build_session_strategy
        (MC strategies load availability from data_root + saved season)."""
        from projections.draft.assistant.availability_loader import load_store_availability

        data = json.loads(path.read_text())
        league = LeagueConfig.model_validate_json(Path(data["league_config"]).read_text())
        name = str(data["strategy_name"])
        n_sims = int(data.get("n_sims", 300))
        season = int(data.get("season", 2026))
        availability = None
        if name in ("season_value", "season_value_var", "season_value_timing"):
            availability = load_store_availability(pool, season=season, data_root=data_root)
        strategy = build_session_strategy(
            name, league=league, sigma=data.get("sigma"),
            availability=availability, n_sims=n_sims, base_seed=0,
        )
        return cls(
            league=league, my_slot=int(data["my_slot"]), id_map=id_map, pool=pool,
            strategy=strategy, strategy_name=name, mode=data.get("mode", "copilot"),
            adp_jitter=float(data.get("adp_jitter", 8.0)), n_sims=n_sims,
            sigma=data.get("sigma"), season=season,
            picks=[str(p) for p in data["picks"]],
            league_config_path=Path(data["league_config"]),
            vorp_path=Path(data.get("vorp_table", ".")),
            id_map_path=Path(data.get("id_map", ".")),
            data_root=data_root,
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest -n0 tests/test_draft/test_assistant_live.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/live.py tests/test_draft/test_assistant_live.py
git commit -m "feat(draft): session persistence (to_state_dict/save/load, CLI-compatible)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 13: Autosave + resume wiring + gitignore

**Files:**
- Modify: `scripts/draft_board.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add `data/draft_sessions/` to `.gitignore`**

In `.gitignore`, under the `# Generated data ...` block, add:

```
data/draft_sessions/
```

- [ ] **Step 2: Replace the `_autosave` shim with a real implementation**

In `scripts/draft_board.py`, replace the no-op `_autosave` with:

```python
_SESSION_DIR = Path("data/draft_sessions")


def _autosave(s: LiveDraftSession) -> None:
    path = st.session_state.get("autosave_path")
    if path is None:
        # Stable filename per session, derived from the object id (no Date.now needed).
        path = _SESSION_DIR / f"session_{id(s):x}.json"
        st.session_state["autosave_path"] = path
    s.save(Path(path))


def _resume_controls() -> None:
    if not _SESSION_DIR.exists():
        return
    saves = sorted(_SESSION_DIR.glob("session_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not saves:
        return
    newest = saves[0]
    st.sidebar.divider()
    st.sidebar.caption(f"Resume autosave: {newest.name}")
    if st.sidebar.button("↩ Resume last draft"):
        try:
            import pandas as pd

            from projections.schemas import _PYARROW_STR, IdMapSchema, VorpTableSchema

            data = __import__("json").loads(newest.read_text())
            id_map = IdMapSchema.validate(pd.read_parquet(data["id_map"]))
            pool = pd.read_parquet(data["vorp_table"])
            pool["gsis_id"] = pool["gsis_id"].astype(_PYARROW_STR)
            pool = VorpTableSchema.validate(pool)
            st.session_state["session"] = LiveDraftSession.load(
                newest, id_map=id_map, pool=pool
            )
            st.session_state["autosave_path"] = newest
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Resume failed: {exc}")
```

- [ ] **Step 3: Call `_resume_controls` in the sidebar**

At the end of `_sidebar()`, after the Start button block, add:

```python
    _resume_controls()
```

- [ ] **Step 4: Type-check + lint**

Run: `mypy src scripts/draft_board.py && ruff check scripts/draft_board.py && ruff format --check scripts/draft_board.py`
Expected: zero violations.

- [ ] **Step 5: Manual smoke**

Run: `streamlit run scripts/draft_board.py` — make a few picks, confirm `data/draft_sessions/session_*.json` is written; restart the app and use "Resume last draft" to restore the picks.

- [ ] **Step 6: Commit**

```bash
git add scripts/draft_board.py .gitignore
git commit -m "feat(ui): autosave + resume; gitignore data/draft_sessions/

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 14: Optional AppTest smoke + docs + PM/TODO update

**Files:**
- Create: `tests/test_scripts/test_draft_board_smoke.py`
- Modify: `CONTRIBUTING.md`
- Modify: `project_management.md`, `TODO.md`

- [ ] **Step 1: Write the AppTest smoke (guarded)**

Create `tests/test_scripts/test_draft_board_smoke.py`:

```python
"""Headless smoke for the Streamlit board: it imports and runs without raising."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest


def test_draft_board_loads_without_session() -> None:
    at = AppTest.from_file("scripts/draft_board.py").run()
    assert not at.exception
    # Before any draft is started, the info prompt is shown.
    assert any("Start" in str(getattr(el, "value", "")) for el in at.info)
```

- [ ] **Step 2: Run the smoke**

Run: `pytest -n0 tests/test_scripts/test_draft_board_smoke.py -v`
Expected: PASS (or SKIP if streamlit absent).

- [ ] **Step 3: Document the run command in `CONTRIBUTING.md`**

Find the section listing runnable scripts/commands and add a "Live draft board" entry:

```markdown
### Live draft board (Draft Assistant UI)

Install the UI extra once: `pip install -e ".[ui]"`. Then:

    streamlit run scripts/draft_board.py

Co-pilot mode: log every pick (yours + opponents'); opponents get a one-click
ADP suggestion. Mock mode: opponents are auto-drafted; "Advance to my pick"
runs the field to your turn; the draft ends with an optimal-lineup scorecard.
Inputs: a consensus VORP parquet (`generate_vorp_table.py --source consensus`),
`data/raw/id_map.parquet`, and a league-config JSON. Sessions autosave to
`data/draft_sessions/` and can be resumed from the sidebar.
```

- [ ] **Step 4: Update `project_management.md` (top entry) and `TODO.md`**

Add a dated entry to the top of `project_management.md` summarizing the shipped board (controller + view, both modes, persistence), and in `TODO.md` mark the Draft Assistant "Slice 3 — Streamlit live-draft UI" item DONE with the branch name, leaving the documented future seams (ESPN/Sleeper sync, auction, season-value MC scorecard) as open follow-ups.

- [ ] **Step 5: Final full-suite gate**

```bash
pytest -k "draft or ingest or store or schemas"
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: all PASS / zero violations. (Run the broader `pytest` if time permits; the draft/ingest/store/schemas subset is the integration seam this work touches.)

- [ ] **Step 6: Commit**

```bash
git add tests/test_scripts/test_draft_board_smoke.py CONTRIBUTING.md project_management.md TODO.md
git commit -m "test(ui): AppTest smoke; docs + PM/TODO for the live draft board

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review notes (for the executor)

- **Spec coverage:** controller (Tasks 3-6, 10, 12) ↔ spec §4.2; `build_draft_state` §4.1 (Task 1); strategy seam §4.4 (Task 2); view/layout B §4.3 (Tasks 8-9, 11); persistence §4.5 (Tasks 12-13); testing §7 (every task + Task 14); phasing §8 (the four phases here). Non-goals respected: no API sync, no auction, no VORP generation, no new strategy.
- **Type consistency:** `LiveDraftSession`, `RosterView`, `build_session_strategy`, `attach_names`, `BOARD_STRATEGIES`, `next_pick_number`, `on_clock_slot`, `mock_advance_to_my_pick`, `roster_scorecard`, `to_state_dict`/`save`/`load` are used with the same names/signatures across tasks.
- **Known executor watch-outs:** (1) the `_autosave` shim in Task 9 is intentionally a no-op until Task 13 replaces it — keep it so Phase 2 type-checks. (2) `st.cache_data` keys on the `picks` tuple + strategy/n_sims/sigma; do not pass the session object as a hashable arg (the leading-underscore `_s_id` avoids it). (3) MC strategies need ingested `weekly_stats` (and 2026 `schedules` for byes) under `data/`; without them the season-value dropdown options fail loud in the sidebar — expected.
