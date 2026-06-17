# Auction Strategy Tournament Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-06-17-auction-strategy-tournament-design.md`

**Goal:** Build an offline, data-gathering auction-draft harness that races three bid models (static SOS dollar, inflation-adjusted, marginal lineup value) against a shared seeded noisy-WTP bot market, retains every seat's full roster, scores each projected league with the existing `project_draft`, and records per-model metrics (expected season points + H2H win/playoff/bye/champ %) with paired bootstrap CIs — declaring **no winner**.

**Architecture:** A new pure-engine package `src/projections/draft/assistant/auction/` (next to the snake `tournament.py`/`simulation.py` it mirrors and reuses — the existing top-level `projections.draft.auction` pricing module is untouched). The auction sim returns `{seat: [gsis_id]}`, the exact `project_draft` input. Shared bootstrap/CI/pool-size helpers are promoted to `assistant/_compare.py` and imported by both the snake and auction tournaments. Store loading (availability, variance params, `is_rookie`) lives in the CLI; the engine is pure.

**Tech Stack:** Python 3.12+, numpy, pandas, pandera (existing schemas), pydantic (`LeagueConfig`), pytest. No new dependencies.

## Global Constraints

- **`GsisId` is canonical.** Use `validate_gsis_id(raw)` at any untrusted-string boundary; never join on names. (Internally rosters carry `gsis_id` strings.)
- **Reference enums, never strings:** `Position.QB`, `RosterSlot.FLEX`, etc. (`from projections.schemas import Position, RosterSlot`).
- **`pd.StringDtype("pyarrow")`** (alias `_PYARROW_STR` in `schemas.py`) for `gsis_id`/`position` columns; `pd.Float64Dtype()` for nullable floats.
- **Gates (run before declaring any task done):** `pytest -v` (relevant subset OK — state which), `mypy src tests` (strict, zero errors), `ruff check src tests` (zero), `ruff format --check src tests` (no drift). No new pandera schema is introduced, so the ingest/store/schemas suite is unaffected — but `mean_points` touches `league_projection.py`, so run `pytest -v -k "league_projection"` for Task 1.
- **No winner declaration anywhere.** The harness records metrics + paired diffs; it never emits "winner = X" (spec §5.1). The adopt decision is the user's, in September.
- **Determinism:** one seeded `np.random.Generator` per (strategy, seed) for the auction market (`default_rng(base_seed + s)`); a separate, disjoint stream `default_rng(season_base_seed + s)` (default `season_base_seed = base_seed + 1_000_000`) for the season MC, shared across strategies at each seed (CRN).
- **Seat indexing:** `my_seat` / `--my-seat` is **1-based** (`1..n_teams`); internal `AuctionState` lists are 0-based; the sole conversion is `hero0 = my_seat - 1`. `project_draft` keys and the returned league dict are 1-based.

---

## File Structure

**New files:**
- `src/projections/draft/assistant/_compare.py` — `Interval`, `bootstrap_mean`, `validate_pool_size` (promoted from `tournament.py`).
- `src/projections/draft/assistant/auction/__init__.py` — package init.
- `src/projections/draft/assistant/auction/bid_strategy.py` — `AuctionView`, `AuctionBidStrategy` Protocol, `StaticDollarBid`, `InflationBid`, `MarginalValueBid`.
- `src/projections/draft/assistant/auction/market.py` — `SeatView`, `bot_max_bid`, `resolve_bids`.
- `src/projections/draft/assistant/auction/simulation.py` — `AuctionState`, `validate_auction_inputs`, `simulate_auction` (+ internal `_open_slots`, `_feasible_max`, `_build_view`).
- `src/projections/draft/assistant/auction/tournament.py` — `AuctionTournamentResult`, `run_auction_tournament`.
- `src/projections/draft/assistant/auction/tournament_cli.py` — argparse engine + `run()` + formatting.
- `scripts/auction_tournament.py` — thin wrapper (`sys.exit(run())`).

**Modified files:**
- `src/projections/draft/assistant/league_projection.py` — add `mean_points` field to `SeatProjection` and compute it in `project_draft`.
- `src/projections/draft/assistant/tournament.py` — re-import `Interval`/`bootstrap_mean`/size-check from `_compare` (keep `_bootstrap_mean` alias + `_validate_pool` so snake code/tests are unaffected).

**Test files (new unless noted):**
- `tests/test_draft/test_league_projection.py` (modify — add `mean_points` assertions)
- `tests/test_draft/test_assistant_compare.py`
- `tests/test_draft/test_assistant_auction_bid_strategy.py`
- `tests/test_draft/test_assistant_auction_market.py`
- `tests/test_draft/test_assistant_auction_simulation.py`
- `tests/test_draft/test_assistant_auction_tournament.py`
- `tests/test_draft/test_assistant_auction_tournament_cli.py`

---

## Task 1: Add `mean_points` to `SeatProjection`

**Files:**
- Modify: `src/projections/draft/assistant/league_projection.py` (dataclass at lines 96-102; output loop at lines 176-186)
- Test: `tests/test_draft/test_league_projection.py`

**Interfaces:**
- Produces: `SeatProjection` gains `mean_points: float` (per-seat mean over sims of regular-season points-for, weeks 1-13). `project_draft(...)` signature unchanged; its returned `SeatProjection`s now carry `mean_points`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_draft/test_league_projection.py`:

```python
def test_seat_projection_has_mean_points() -> None:
    rosters, pool = _symmetric_league(12)
    params = VarianceParams.load()
    avail = PlayerAvailability(p={g: 0.95 for g in pool["gsis_id"]}, bye={})
    out = project_draft(
        rosters, pool, avail, params, league_config=_config(12), n_sims=200, rng=np.random.default_rng(0)
    )
    sp = out[1]
    assert sp.mean_points > 0.0
    assert np.isfinite(sp.mean_points)


def test_mean_points_rewards_a_stronger_roster() -> None:
    rosters, pool = _symmetric_league(12)
    # Lift seat 1's players' projected points; everyone else unchanged.
    boost = pool["gsis_id"].isin(rosters[1])
    pool.loc[boost, "season_mean_fpts"] = pool.loc[boost, "season_mean_fpts"] * 1.5
    params = VarianceParams.load()
    avail = PlayerAvailability(p={g: 0.95 for g in pool["gsis_id"]}, bye={})
    out = project_draft(
        rosters, pool, avail, params, league_config=_config(12), n_sims=300, rng=np.random.default_rng(0)
    )
    assert out[1].mean_points > out[2].mean_points
```

Confirm the test file already imports `np`, `VarianceParams`, `PlayerAvailability`, `project_draft`, `_symmetric_league`, `_config` (per `tests/test_draft/test_league_projection.py` head); add any missing import.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_draft/test_league_projection.py::test_seat_projection_has_mean_points -v`
Expected: FAIL — `TypeError: SeatProjection.__init__() ... unexpected keyword 'mean_points'` is *not* the failure (we read the attribute); it fails with `AttributeError: 'SeatProjection' object has no attribute 'mean_points'`.

- [ ] **Step 3: Add the field** — edit the dataclass (currently lines 96-102):

```python
@dataclass(frozen=True)
class SeatProjection:
    reg_win_pct: float
    make_playoffs_pct: float
    bye_pct: float
    champ_pct: float
    mean_seed: float
    mean_points: float
```

- [ ] **Step 4: Populate it in `project_draft`** — in the per-seat output loop (currently lines 176-186), add `mean_points` to the `SeatProjection(...)` construction. The per-seat per-sim regular-season points-for already lives in `pf[s]` (shape `(n_sims,)`):

```python
        out[s] = SeatProjection(
            reg_win_pct=float(wins[s].mean() / len(REG_WEEKS)),
            make_playoffs_pct=float(in_playoffs.mean()),
            bye_pct=float(has_bye.mean()),
            champ_pct=float((champ_of == s).mean()),
            mean_seed=float(seed_of_s.mean()),
            mean_points=float(pf[s].mean()),
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_draft/test_league_projection.py -v`
Expected: PASS (new tests + all existing — no existing test constructs `SeatProjection` directly; they read returned fields). If any existing test or the board (`scripts/draft_board.py`) constructs `SeatProjection(...)` positionally, add `mean_points=...`. Grep to be sure: `rg -n "SeatProjection\(" src tests scripts`.

- [ ] **Step 6: Gates + commit**

Run: `mypy src tests && ruff check src tests && ruff format --check src tests`
```bash
git add src/projections/draft/assistant/league_projection.py tests/test_draft/test_league_projection.py
git commit -m "feat(draft): add mean_points (reg-season points-for) to SeatProjection"
```

---

## Task 2: Promote shared stats to `assistant/_compare.py`

**Files:**
- Create: `src/projections/draft/assistant/_compare.py`
- Modify: `src/projections/draft/assistant/tournament.py` (imports + remove local `Interval`/`_bootstrap_mean` bodies; keep `_bootstrap_mean` alias + `_validate_pool`)
- Test: `tests/test_draft/test_assistant_compare.py`

**Interfaces:**
- Produces:
  - `Interval(point: float, lo_95: float, hi_95: float)` (frozen dataclass)
  - `bootstrap_mean(values: np.ndarray, *, n_bootstrap: int = 1000, seed: int) -> Interval`
  - `validate_pool_size(pool: pd.DataFrame, config: LeagueConfig) -> None`
- Consumes (in `tournament.py`): re-exports `Interval`, aliases `bootstrap_mean as _bootstrap_mean`, and calls `validate_pool_size` inside `_validate_pool`.

- [ ] **Step 1: Write the failing test** — `tests/test_draft/test_assistant_compare.py`:

```python
import numpy as np
import pytest

from projections.draft.assistant._compare import Interval, bootstrap_mean, validate_pool_size
from projections.draft.league_config import LeagueConfig
from projections.schemas import RosterSlot, Ruleset


def _config(n_teams: int = 2) -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=n_teams,
        roster_slots={RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.BENCH: 1},
        ruleset=Ruleset.espn_ppr(),
    )


def test_bootstrap_mean_on_constant_array_is_a_point() -> None:
    iv = bootstrap_mean(np.full(50, 7.0), seed=0)
    assert iv == Interval(point=7.0, lo_95=7.0, hi_95=7.0)


def test_bootstrap_mean_ci_brackets_the_mean() -> None:
    iv = bootstrap_mean(np.arange(100, dtype=float), seed=1)
    assert iv.lo_95 < iv.point < iv.hi_95
    assert abs(iv.point - 49.5) < 1e-9


def test_validate_pool_size_raises_when_too_small() -> None:
    import pandas as pd

    pool = pd.DataFrame({"gsis_id": ["00-0000001"]})  # 1 player; need n_teams*roster_size = 2*2 = 4
    with pytest.raises(ValueError, match="need >= 4"):
        validate_pool_size(pool, _config(2))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_draft/test_assistant_compare.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'projections.draft.assistant._compare'`.

- [ ] **Step 3: Create `_compare.py`** with the promoted helpers (verbatim bodies from the current `tournament.py`):

```python
"""Mechanism-agnostic comparison helpers shared by the snake and auction tournaments.

Promoted from tournament.py so both harnesses bootstrap and size-validate identically.
The snake tournament keeps its ADP-specific arms and winner-labeling on top of these;
the auction harness records metrics without declaring a winner (spec §5.1, §5.7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from projections.draft.league_config import LeagueConfig

_N_BOOTSTRAP = 1000
_CI_PCTILES = (2.5, 97.5)


@dataclass(frozen=True)
class Interval:
    """A point estimate with a central 95% bootstrap CI."""

    point: float
    lo_95: float
    hi_95: float


def bootstrap_mean(values: np.ndarray, *, n_bootstrap: int = _N_BOOTSTRAP, seed: int) -> Interval:
    """Percentile-bootstrap CI of the mean of `values` (pass `a - b` for a paired diff)."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = v.shape[0]
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        boot[b] = v[rng.integers(0, n, size=n)].mean()
    lo, hi = np.percentile(boot, _CI_PCTILES)
    return Interval(point=float(v.mean()), lo_95=float(lo), hi_95=float(hi))


def validate_pool_size(pool: pd.DataFrame, config: LeagueConfig) -> None:
    """Mechanism-agnostic pool-size precondition: enough players to fill every roster spot."""
    need = config.n_teams * config.roster_size
    if len(pool) < need:
        raise ValueError(f"pool has {len(pool)} players; need >= {need} to fill a full draft")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_draft/test_assistant_compare.py -v`
Expected: PASS.

- [ ] **Step 5: Rewire `tournament.py` to reuse `_compare` (no behavior change)** — replace the local `Interval` dataclass (lines 29-35), the `_bootstrap_mean` def (lines 94-103), and the size arm inside `_validate_pool` (lines 67-69). Concretely:

In the imports block (after line 19, `from projections.draft.assistant.valuer import ...`):
```python
from projections.draft.assistant._compare import Interval, bootstrap_mean as _bootstrap_mean
from projections.draft.assistant._compare import validate_pool_size
```
Delete the local `@dataclass ... class Interval` block (lines 29-35) and the local `def _bootstrap_mean(...)` block (lines 94-103) — they now come from `_compare`. Keep `_N_BOOTSTRAP`/`_CI_PCTILES` only if still referenced elsewhere; if not, delete them (ruff F811/F401 will flag). Then change `_validate_pool` so its size arm delegates:
```python
def _validate_pool(pool: pd.DataFrame, config: LeagueConfig) -> None:
    """Hard preconditions shared by both entry points (spec §3.1, §3.3)."""
    validate_pool_size(pool, config)
    if "consensus_adp" not in pool.columns or bool(pool["consensus_adp"].isna().all()):
        raise ValueError(
            "pool has no consensus_adp signal; the tournament needs market ADP to drive the field"
        )
```
`_bootstrap_mean` (the alias) and `Interval` remain importable from `tournament.py`, so `tests/test_draft/test_assistant_tournament.py` (which imports `_bootstrap_mean`, `_validate_pool`) is unaffected.

- [ ] **Step 6: Run the snake tournament suite to prove no regression**

Run: `pytest tests/test_draft/test_assistant_tournament.py tests/test_draft/test_assistant_tournament_cli.py tests/test_draft/test_assistant_compare.py -v`
Expected: PASS (all existing snake tests + the new `_compare` tests).

- [ ] **Step 7: Gates + commit**

Run: `mypy src tests && ruff check src tests && ruff format --check src tests`
```bash
git add src/projections/draft/assistant/_compare.py src/projections/draft/assistant/tournament.py tests/test_draft/test_assistant_compare.py
git commit -m "refactor(draft): promote Interval/bootstrap_mean/pool-size to assistant/_compare"
```

---

## Task 3: Bid models (`auction/bid_strategy.py`)

**Files:**
- Create: `src/projections/draft/assistant/auction/__init__.py` (empty)
- Create: `src/projections/draft/assistant/auction/bid_strategy.py`
- Test: `tests/test_draft/test_assistant_auction_bid_strategy.py`

**Interfaces:**
- Produces:
  - `AuctionView` (frozen dataclass): `my_budget: int`, `my_open_slots: int`, `my_positions: Counter[str]`, `my_roster: pd.DataFrame` (pool rows for the hero's drafted gsis_ids — has `gsis_id`/`position`/`season_mean_fpts`), `drafted: frozenset[str]`, `budgets_by_seat: tuple[int, ...]`, `baseline_dollars: pd.DataFrame` (full `AuctionValuesSchema` frame **indexed by `gsis_id`**).
  - `AuctionBidStrategy` Protocol: `max_bid(self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig) -> int`
  - `StaticDollarBid`, `InflationBid`, `MarginalValueBid` (each frozen dataclass, no fields).
- Consumes: `optimal_lineup_points` (`assistant/roster_score.py`), `LeagueConfig`. `player` is a pool row Series (carries `gsis_id`, `position`, `season_mean_fpts`).

- [ ] **Step 1: Write the failing tests** — `tests/test_draft/test_assistant_auction_bid_strategy.py`:

```python
from collections import Counter

import pandas as pd
import pytest

from projections.draft.assistant.auction.bid_strategy import (
    AuctionView,
    InflationBid,
    MarginalValueBid,
    StaticDollarBid,
)
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, RosterSlot, Ruleset


def _config() -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=2,
        budget=100,
        min_bid=1,
        roster_slots={RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.BENCH: 1},
        ruleset=Ruleset.espn_ppr(),
    )


def _pool() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000001", "00-0000002", "00-0000003", "00-0000004"], dtype=_PYARROW_STR),
            "position": pd.array(["RB", "WR", "RB", "WR"], dtype=_PYARROW_STR),
            "season_mean_fpts": [200.0, 180.0, 50.0, 40.0],
        }
    )


def _baseline(in_pool: list[bool], dollars: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "in_pool": in_pool,
            "auction_dollars": dollars,
        },
        index=pd.Index(["00-0000001", "00-0000002", "00-0000003", "00-0000004"], name="gsis_id"),
    )


def _view(my_roster: pd.DataFrame, *, budget: int, drafted: set[str], baseline: pd.DataFrame) -> AuctionView:
    return AuctionView(
        my_budget=budget,
        my_open_slots=3 - len(my_roster),
        my_positions=Counter(my_roster["position"].astype(str)),
        my_roster=my_roster,
        drafted=frozenset(drafted),
        budgets_by_seat=(budget, budget),
        baseline_dollars=baseline,
    )


def test_static_bids_the_baseline_dollar() -> None:
    pool = _pool()
    baseline = _baseline([True, True, False, False], [60, 40, 0, 0])
    view = _view(pool.iloc[:0], budget=100, drafted=set(), baseline=baseline)
    bid = StaticDollarBid().max_bid(view, pool.iloc[0], pool, _config())
    assert bid == 60


def test_inflation_below_one_when_room_overspent() -> None:
    pool = _pool()
    baseline = _baseline([True, True, False, False], [60, 40, 0, 0])
    # Both seats have spent down to $10 each but nothing is drafted yet (contrived overspend):
    view = AuctionView(
        my_budget=10,
        my_open_slots=3,
        my_positions=Counter(),
        my_roster=pool.iloc[:0],
        drafted=frozenset(),
        budgets_by_seat=(10, 10),
        baseline_dollars=baseline,
    )
    bid = InflationBid().max_bid(view, pool.iloc[0], pool, _config())
    assert bid < 60  # inflation < 1 -> below the static dollar


def test_inflation_falls_back_to_one_when_no_surplus_value() -> None:
    pool = _pool()
    # Only out-of-pool players left undrafted -> remaining_surplus_value == 0 -> factor 1.0
    baseline = _baseline([True, True, False, False], [60, 40, 0, 0])
    view = AuctionView(
        my_budget=100,
        my_open_slots=3,
        my_positions=Counter(),
        my_roster=pool.iloc[:0],
        drafted=frozenset({"00-0000001", "00-0000002"}),  # both in-pool already drafted
        budgets_by_seat=(100, 100),
        baseline_dollars=baseline,
    )
    # pricing an out-of-pool player: base==0 -> min_bid + (0-1)*1.0 = 0 -> still returns an int
    bid = InflationBid().max_bid(view, pool.iloc[2], pool, _config())
    assert isinstance(bid, int)


def test_marginal_zero_lift_player_bids_min_bid() -> None:
    pool = _pool()
    baseline = _baseline([True, True, True, True], [60, 40, 5, 5])
    # Hero already holds the best RB and WR (starters full); a worse RB adds 0 lineup lift.
    my_roster = pool.iloc[[0, 1]]
    view = _view(my_roster, budget=50, drafted={"00-0000001", "00-0000002"}, baseline=baseline)
    bid = MarginalValueBid().max_bid(view, pool.iloc[2], pool, _config())
    assert bid == _config().min_bid


def test_marginal_improving_player_bids_above_min_bid() -> None:
    pool = _pool()
    baseline = _baseline([True, True, True, True], [60, 40, 5, 5])
    # Empty roster: the best RB cracks the lineup -> lift > 0 -> bid > min_bid.
    view = _view(pool.iloc[:0], budget=100, drafted=set(), baseline=baseline)
    bid = MarginalValueBid().max_bid(view, pool.iloc[0], pool, _config())
    assert bid > _config().min_bid
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_draft/test_assistant_auction_bid_strategy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'projections.draft.assistant.auction'`.

- [ ] **Step 3: Create the package init**

`src/projections/draft/assistant/auction/__init__.py`:
```python
"""Auction-draft simulation, bid models, bot market, and the data-gathering tournament."""
```

- [ ] **Step 4: Implement `bid_strategy.py`**

```python
"""Auction bid models (the tournament contestants) and the read-only hero view.

Spec docs/superpowers/specs/2026-06-17-auction-strategy-tournament-design.md §3.4.
Each model returns a *desired* max bid (any int); the engine clamps it to
[min_bid, feasible_max] (§3.2), so models never re-implement the reserve or the floor.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from projections.draft.assistant.roster_score import optimal_lineup_points
from projections.draft.league_config import LeagueConfig


@dataclass(frozen=True)
class AuctionView:
    """Read-only snapshot of the auction for the hero seat (built by simulate_auction)."""

    my_budget: int
    my_open_slots: int
    my_positions: Counter[str]
    my_roster: pd.DataFrame  # pool rows for the hero's drafted gsis_ids
    drafted: frozenset[str]
    budgets_by_seat: tuple[int, ...]
    baseline_dollars: pd.DataFrame  # full AuctionValuesSchema frame, indexed by gsis_id


@runtime_checkable
class AuctionBidStrategy(Protocol):
    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int: ...


def _total_open_slots(view: AuctionView, config: LeagueConfig) -> int:
    return config.n_teams * config.roster_size - len(view.drafted)


@dataclass(frozen=True)
class StaticDollarBid:
    """v1 — bid straight to the static SOS dollar."""

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        return int(view.baseline_dollars.loc[player["gsis_id"], "auction_dollars"])


@dataclass(frozen=True)
class InflationBid:
    """v2 — re-price the static dollar by live surplus inflation (spec §3.4)."""

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        min_bid = config.min_bid
        money = sum(view.budgets_by_seat) - min_bid * _total_open_slots(view, config)
        bd = view.baseline_dollars
        undrafted_in_pool = bd[bd["in_pool"] & ~bd.index.isin(view.drafted)]
        value = float((undrafted_in_pool["auction_dollars"] - min_bid).sum())
        inflation = money / value if value > 0 else 1.0
        base = int(bd.loc[player["gsis_id"], "auction_dollars"])
        return int(round(min_bid + (base - min_bid) * inflation))


@dataclass(frozen=True)
class MarginalValueBid:
    """v3 — bid to the player's marginal optimal-lineup lift at the live market rate (spec §3.4)."""

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        min_bid = config.min_bid
        slots = config.roster_slots
        base_pts = optimal_lineup_points(view.my_roster, slots)
        cand = pool[pool["gsis_id"] == player["gsis_id"]]
        with_player = pd.concat([view.my_roster, cand], ignore_index=True)
        lift = optimal_lineup_points(with_player, slots) - base_pts
        if lift <= 0.0:
            return min_bid
        money = sum(view.budgets_by_seat) - min_bid * _total_open_slots(view, config)
        bd = view.baseline_dollars
        undrafted_in_pool_ids = bd[bd["in_pool"] & ~bd.index.isin(view.drafted)].index
        # Lift of a single player to an EMPTY lineup == its season_mean_fpts (every in-pool
        # position has a starting slot), so the board's surplus value in lineup-points is the
        # sum of undrafted in-pool projected points. Cheap, and equal to summing single-player
        # optimal_lineup_points one by one.
        on_board = pool[pool["gsis_id"].isin(undrafted_in_pool_ids)]
        value_points = float(on_board["season_mean_fpts"].sum())
        points_per_dollar = value_points / money if money > 0 else 0.0
        if points_per_dollar <= 0.0:
            return min_bid
        return int(round(min_bid + lift / points_per_dollar))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_draft/test_assistant_auction_bid_strategy.py -v`
Expected: PASS.

- [ ] **Step 6: Gates + commit**

Run: `mypy src tests && ruff check src tests && ruff format --check src tests`
```bash
git add src/projections/draft/assistant/auction/__init__.py src/projections/draft/assistant/auction/bid_strategy.py tests/test_draft/test_assistant_auction_bid_strategy.py
git commit -m "feat(auction): bid models (static, inflation, marginal) + AuctionView"
```

---

## Task 4: Bot market + bid resolution (`auction/market.py`)

**Files:**
- Create: `src/projections/draft/assistant/auction/market.py`
- Test: `tests/test_draft/test_assistant_auction_market.py`

**Interfaces:**
- Produces:
  - `SeatView(open_slots: int)` (frozen dataclass — minimal per-seat view the bot needs).
  - `bot_max_bid(seat_view: SeatView, player: pd.Series, baseline_dollars: pd.DataFrame, config: LeagueConfig, rng: np.random.Generator, *, price_jitter: float) -> int` — desired WTP (floored at `min_bid`; the engine still clamps to `[min_bid, feasible_max]`). Returns `0` when the seat has no open slot (abstain).
  - `resolve_bids(bids: dict[int, int], min_bid: int) -> tuple[int, int]` — `(winner_seat, price)`; second-price + `min_bid` clearing; lone bidder pays `min_bid`; ties break on seat index ascending.
  - `DEFAULT_PRICE_JITTER: float` (named constant).
- Consumes: `LeagueConfig` (for `min_bid`), `baseline_dollars` indexed by `gsis_id`.

- [ ] **Step 1: Write the failing tests** — `tests/test_draft/test_assistant_auction_market.py`:

```python
import numpy as np
import pandas as pd

from projections.draft.assistant.auction.market import SeatView, bot_max_bid, resolve_bids
from projections.draft.league_config import LeagueConfig
from projections.schemas import RosterSlot, Ruleset


def _config(min_bid: int = 1) -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=2,
        budget=100,
        min_bid=min_bid,
        roster_slots={RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.BENCH: 1},
        ruleset=Ruleset.espn_ppr(),
    )


def _baseline() -> pd.DataFrame:
    return pd.DataFrame(
        {"in_pool": [True], "auction_dollars": [40]},
        index=pd.Index(["00-0000001"], name="gsis_id"),
    )


def _player() -> pd.Series:
    return pd.Series({"gsis_id": "00-0000001", "position": "RB", "season_mean_fpts": 200.0})


def test_bot_centers_on_baseline_with_zero_jitter() -> None:
    bid = bot_max_bid(SeatView(open_slots=3), _player(), _baseline(), _config(), np.random.default_rng(0), price_jitter=0.0)
    assert bid == 40


def test_bot_floors_at_min_bid() -> None:
    base = pd.DataFrame({"in_pool": [False], "auction_dollars": [0]}, index=pd.Index(["00-0000001"], name="gsis_id"))
    bid = bot_max_bid(SeatView(open_slots=3), _player(), base, _config(min_bid=2), np.random.default_rng(0), price_jitter=0.0)
    assert bid == 2


def test_full_seat_abstains() -> None:
    bid = bot_max_bid(SeatView(open_slots=0), _player(), _baseline(), _config(), np.random.default_rng(0), price_jitter=0.5)
    assert bid == 0


def test_resolve_second_price_plus_min_bid() -> None:
    winner, price = resolve_bids({0: 40, 1: 25, 2: 10}, min_bid=1)
    assert winner == 0
    assert price == 26  # second-highest (25) + min_bid (1)


def test_resolve_caps_at_winner_max() -> None:
    winner, price = resolve_bids({0: 5, 1: 4}, min_bid=3)
    assert winner == 0
    assert price == min(5, 4 + 3)  # == 5, never above the winner's own ceiling


def test_resolve_lone_bidder_pays_min_bid() -> None:
    assert resolve_bids({2: 80}, min_bid=1) == (2, 1)


def test_resolve_ties_break_on_seat_index() -> None:
    winner, _ = resolve_bids({3: 20, 1: 20}, min_bid=1)
    assert winner == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_draft/test_assistant_auction_market.py -v`
Expected: FAIL — `ModuleNotFoundError: ...auction.market`.

- [ ] **Step 3: Implement `market.py`**

```python
"""Noisy-WTP bot bid policy and second-price clearing (spec §3.5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from projections.draft.league_config import LeagueConfig

DEFAULT_PRICE_JITTER: float = 0.15  # fractional WTP spread; auction analog of adp_jitter


@dataclass(frozen=True)
class SeatView:
    """Minimal per-seat view the bot reads (the engine handles feasible_max)."""

    open_slots: int


def bot_max_bid(
    seat_view: SeatView,
    player: pd.Series,
    baseline_dollars: pd.DataFrame,
    config: LeagueConfig,
    rng: np.random.Generator,
    *,
    price_jitter: float,
) -> int:
    """Value-rational WTP centered on the market dollar, with multiplicative noise."""
    if seat_view.open_slots <= 0:
        return 0
    base = float(baseline_dollars.loc[player["gsis_id"], "auction_dollars"])
    wtp = base * (1.0 + rng.normal(0.0, price_jitter))
    return int(round(max(float(config.min_bid), wtp)))


def resolve_bids(bids: dict[int, int], min_bid: int) -> tuple[int, int]:
    """English (second-price + min_bid) clearing. `bids` maps seat -> clamped max bid.

    Winner is the argmax bid (ties -> lowest seat index). Price is one tick over the
    runner-up's ceiling, never above the winner's own; a lone bidder pays min_bid.
    """
    ordered = sorted(bids.items(), key=lambda kv: (-kv[1], kv[0]))
    winner_seat, winner_max = ordered[0]
    if len(ordered) == 1:
        return winner_seat, min_bid
    second_max = ordered[1][1]
    return winner_seat, min(winner_max, second_max + min_bid)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_draft/test_assistant_auction_market.py -v`
Expected: PASS.

- [ ] **Step 5: Gates + commit**

Run: `mypy src tests && ruff check src tests && ruff format --check src tests`
```bash
git add src/projections/draft/assistant/auction/market.py tests/test_draft/test_assistant_auction_market.py
git commit -m "feat(auction): noisy-WTP bot + second-price clearing"
```

---

## Task 5: Auction simulation (`auction/simulation.py`)

**Files:**
- Create: `src/projections/draft/assistant/auction/simulation.py`
- Test: `tests/test_draft/test_assistant_auction_simulation.py`

**Interfaces:**
- Consumes: `AuctionBidStrategy`/`AuctionView` (Task 3), `SeatView`/`bot_max_bid`/`resolve_bids` (Task 4), `validate_pool_size` (Task 2), `generate_auction_values` (`projections.draft.auction`), `LeagueConfig`.
- Produces:
  - `validate_auction_inputs(pool: pd.DataFrame, config: LeagueConfig) -> None` — pool-size + budget-solvency (`budget >= min_bid * roster_size`) preconditions.
  - `simulate_auction(strategy: AuctionBidStrategy, my_seat: int, pool: pd.DataFrame, config: LeagueConfig, *, baseline_dollars: pd.DataFrame, price_jitter: float, rng: np.random.Generator) -> dict[int, list[str]]` — one full auction; returns every seat's roster `{seat(1-based): [gsis_id, ...]}` (the `project_draft` input).

- [ ] **Step 1: Write the failing tests** — `tests/test_draft/test_assistant_auction_simulation.py`:

```python
import numpy as np
import pandas as pd
import pytest

from projections.draft.auction import generate_auction_values
from projections.draft.assistant.auction.bid_strategy import StaticDollarBid
from projections.draft.assistant.auction.simulation import simulate_auction, validate_auction_inputs
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, RosterSlot, Ruleset


def _config(n_teams: int = 4, budget: int = 100, min_bid: int = 1) -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=n_teams,
        budget=budget,
        min_bid=min_bid,
        roster_slots={RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.BENCH: 1},
        ruleset=Ruleset.espn_ppr(),
    )


def _pool(n: int = 40) -> pd.DataFrame:
    # per-position digit keeps synthetic gsis within the canonical \d{2}-\d{7} regex
    pos = ["RB" if i % 2 else "WR" for i in range(n)]
    prefix = {"RB": 2, "WR": 3}
    gsis = [f"00-{prefix[pos[i]]}{i:06d}" for i in range(n)]
    return pd.DataFrame(
        {
            "gsis_id": pd.array(gsis, dtype=_PYARROW_STR),
            "position": pd.array(pos, dtype=_PYARROW_STR),
            "season_mean_fpts": [float(300 - i) for i in range(n)],
            "vorp": [float(150 - i) for i in range(n)],
            "replacement_fpts": [100.0] * n,
        }
    )


def _baseline(pool: pd.DataFrame, config: LeagueConfig) -> pd.DataFrame:
    return generate_auction_values(pool, config)


def test_validate_rejects_thin_pool() -> None:
    cfg = _config(n_teams=4)  # need 4*3 = 12
    with pytest.raises(ValueError, match="need >= 12"):
        validate_auction_inputs(_pool(8), cfg)


def test_validate_rejects_insolvent_budget() -> None:
    cfg = _config(n_teams=4, budget=2, min_bid=1)  # roster_size 3 > budget 2
    with pytest.raises(ValueError, match="can't afford min_bid"):
        validate_auction_inputs(_pool(40), cfg)


def test_returns_full_league_each_seat_full() -> None:
    cfg = _config(n_teams=4)
    pool = _pool(40)
    league = simulate_auction(
        StaticDollarBid(), 1, pool, cfg,
        baseline_dollars=_baseline(pool, cfg), price_jitter=0.1, rng=np.random.default_rng(0),
    )
    assert set(league) == {1, 2, 3, 4}
    assert all(len(r) == cfg.roster_size for r in league.values())
    all_ids = [g for r in league.values() for g in r]
    assert len(all_ids) == len(set(all_ids))  # no player drafted twice


def test_determinism_same_seed_same_league() -> None:
    cfg = _config(n_teams=4)
    pool = _pool(40)
    bd = _baseline(pool, cfg)
    a = simulate_auction(StaticDollarBid(), 2, pool, cfg, baseline_dollars=bd, price_jitter=0.2, rng=np.random.default_rng(7))
    b = simulate_auction(StaticDollarBid(), 2, pool, cfg, baseline_dollars=bd, price_jitter=0.2, rng=np.random.default_rng(7))
    assert a == b


def test_seat_index_is_one_based() -> None:
    # my_seat=3 must address the 3rd seat (1-based); the returned dict is 1-based.
    cfg = _config(n_teams=4)
    pool = _pool(40)
    league = simulate_auction(
        StaticDollarBid(), 3, pool, cfg,
        baseline_dollars=_baseline(pool, cfg), price_jitter=0.0, rng=np.random.default_rng(1),
    )
    assert 3 in league and len(league[3]) == cfg.roster_size


def test_solvency_holds_no_negative_budget_path() -> None:
    # A tight budget where every seat must reserve min_bid for remaining slots still completes.
    cfg = _config(n_teams=4, budget=3, min_bid=1)  # budget == min_bid*roster_size, endgame all $1
    pool = _pool(40)
    league = simulate_auction(
        StaticDollarBid(), 1, pool, cfg,
        baseline_dollars=_baseline(pool, cfg), price_jitter=0.0, rng=np.random.default_rng(0),
    )
    assert all(len(r) == cfg.roster_size for r in league.values())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_draft/test_assistant_auction_simulation.py -v`
Expected: FAIL — `ModuleNotFoundError: ...auction.simulation`.

- [ ] **Step 3: Implement `simulation.py`**

```python
"""One full auction: nominate -> bid -> award, hero via a bid model, rest via bots (spec §3.6).

Returns every seat's roster as {seat(1-based): [gsis_id, ...]} — the project_draft input.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from projections.draft.assistant._compare import validate_pool_size
from projections.draft.assistant.auction.bid_strategy import AuctionBidStrategy, AuctionView
from projections.draft.assistant.auction.market import SeatView, bot_max_bid, resolve_bids
from projections.draft.league_config import LeagueConfig


def validate_auction_inputs(pool: pd.DataFrame, config: LeagueConfig) -> None:
    """Pool-size and budget-solvency preconditions (spec §3.1)."""
    validate_pool_size(pool, config)
    if config.budget < config.min_bid * config.roster_size:
        raise ValueError(
            f"budget {config.budget} < min_bid*roster_size "
            f"({config.min_bid}*{config.roster_size}); a seat can't afford min_bid for every slot"
        )


@dataclass
class AuctionState:
    budgets: list[int]  # 0-based per seat
    rosters: list[list[tuple[str, str, int]]]  # (gsis_id, position, price) per seat, 0-based
    drafted: set[str]
    nominator: int  # 0-based

    @classmethod
    def initial(cls, config: LeagueConfig) -> AuctionState:
        n = config.n_teams
        return cls([config.budget] * n, [[] for _ in range(n)], set(), 0)


def _open_slots(state: AuctionState, seat: int, roster_size: int) -> int:
    return roster_size - len(state.rosters[seat])


def _feasible_max(state: AuctionState, seat: int, roster_size: int, min_bid: int) -> int:
    return state.budgets[seat] - min_bid * (_open_slots(state, seat, roster_size) - 1)


def _build_view(
    state: AuctionState, hero0: int, pool: pd.DataFrame, bd: pd.DataFrame, config: LeagueConfig
) -> AuctionView:
    my_ids = [g for (g, _p, _pr) in state.rosters[hero0]]
    return AuctionView(
        my_budget=state.budgets[hero0],
        my_open_slots=_open_slots(state, hero0, config.roster_size),
        my_positions=Counter(p for (_g, p, _pr) in state.rosters[hero0]),
        my_roster=pool[pool["gsis_id"].isin(my_ids)],
        drafted=frozenset(state.drafted),
        budgets_by_seat=tuple(state.budgets),
        baseline_dollars=bd,
    )


def simulate_auction(
    strategy: AuctionBidStrategy,
    my_seat: int,
    pool: pd.DataFrame,
    config: LeagueConfig,
    *,
    baseline_dollars: pd.DataFrame,
    price_jitter: float,
    rng: np.random.Generator,
) -> dict[int, list[str]]:
    validate_auction_inputs(pool, config)
    n = config.n_teams
    rs = config.roster_size
    min_bid = config.min_bid
    hero0 = my_seat - 1  # the single 1-based -> 0-based conversion (spec §3.2)

    bd = baseline_dollars.set_index("gsis_id")
    # nomination order: highest baseline dollar first (rebuilt-free; we skip drafted on the fly)
    nominate_order = bd.sort_values("auction_dollars", ascending=False).index.tolist()
    state = AuctionState.initial(config)

    while any(_open_slots(state, s, rs) > 0 for s in range(n)):
        # 1. advance nominator to a seat that still has an open slot
        while _open_slots(state, state.nominator, rs) == 0:
            state.nominator = (state.nominator + 1) % n
        nominee_id = next(g for g in nominate_order if g not in state.drafted)
        player = pool.loc[pool["gsis_id"] == nominee_id].iloc[0]

        # 2. collect clamped bids from every eligible seat
        bids: dict[int, int] = {}
        for seat in range(n):
            if _open_slots(state, seat, rs) <= 0:
                continue
            fmax = _feasible_max(state, seat, rs, min_bid)
            if seat == hero0:
                desired = strategy.max_bid(_build_view(state, hero0, pool, bd, config), player, pool, config)
            else:
                desired = bot_max_bid(
                    SeatView(open_slots=_open_slots(state, seat, rs)),
                    player, bd, config, rng, price_jitter=price_jitter,
                )
            bids[seat] = max(min_bid, min(int(desired), fmax))  # clamp to [min_bid, feasible_max]

        # 3. resolve + award
        winner, price = resolve_bids(bids, min_bid)
        state.budgets[winner] -= price
        state.rosters[winner].append((nominee_id, str(player["position"]), price))
        state.drafted.add(nominee_id)
        state.nominator = (state.nominator + 1) % n

    return {seat + 1: [g for (g, _p, _pr) in state.rosters[seat]] for seat in range(n)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_draft/test_assistant_auction_simulation.py -v`
Expected: PASS.

- [ ] **Step 5: Gates + commit**

Run: `mypy src tests && ruff check src tests && ruff format --check src tests`
```bash
git add src/projections/draft/assistant/auction/simulation.py tests/test_draft/test_assistant_auction_simulation.py
git commit -m "feat(auction): simulate_auction (full-league nominate/bid/award loop)"
```

---

## Task 6: Auction tournament engine (`auction/tournament.py`)

**Files:**
- Create: `src/projections/draft/assistant/auction/tournament.py`
- Test: `tests/test_draft/test_assistant_auction_tournament.py`

**Interfaces:**
- Consumes: `simulate_auction`/`validate_auction_inputs` (Task 5), `Interval`/`bootstrap_mean` (Task 2), `project_draft`/`SeatProjection` (Task 1), `generate_auction_values` (`projections.draft.auction`), `PlayerAvailability`, `VarianceParams`, `LeagueConfig`.
- Produces:
  - `METRICS: tuple[str, ...] = ("mean_points", "reg_win_pct", "make_playoffs_pct", "bye_pct", "champ_pct")`
  - `AuctionTournamentResult` (frozen dataclass): `summaries: dict[str, dict[str, Interval]]` (strategy -> metric -> Interval), `paired_diffs: dict[str, dict[str, Interval]]` (`"a_vs_b"` -> metric -> Interval), `n_seeds`, `price_jitter`, `base_seed`, `season_base_seed`, `n_sims`, `my_seat`, `budget`, `min_bid` (all echoed for reproducibility).
  - `run_auction_tournament(strategies: Mapping[str, AuctionBidStrategy], pool: pd.DataFrame, config: LeagueConfig, *, my_seat: int, n_seeds: int, price_jitter: float, base_seed: int, n_sims: int, availability: PlayerAvailability, params: VarianceParams, season_base_seed: int | None = None) -> AuctionTournamentResult`. **Precondition:** `pool` already carries `is_rookie` (the CLI attaches it). No winner is computed.

- [ ] **Step 1: Write the failing tests** — `tests/test_draft/test_assistant_auction_tournament.py`:

```python
import numpy as np
import pandas as pd

from projections.draft.assistant.auction.bid_strategy import StaticDollarBid
from projections.draft.assistant.auction.tournament import (
    METRICS,
    AuctionTournamentResult,
    run_auction_tournament,
)
from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, RosterSlot, Ruleset


def _config(n_teams: int = 4, budget: int = 100, roster_slots=None) -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=n_teams,
        budget=budget,
        min_bid=1,
        roster_slots=roster_slots or {RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.BENCH: 1},
        ruleset=Ruleset.espn_ppr(),
    )


def _pool(n: int = 40) -> pd.DataFrame:
    pos = ["RB" if i % 2 else "WR" for i in range(n)]
    prefix = {"RB": 2, "WR": 3}
    gsis = [f"00-{prefix[pos[i]]}{i:06d}" for i in range(n)]
    return pd.DataFrame(
        {
            "gsis_id": pd.array(gsis, dtype=_PYARROW_STR),
            "position": pd.array(pos, dtype=_PYARROW_STR),
            "season_mean_fpts": [float(300 - i) for i in range(n)],
            "vorp": [float(150 - i) for i in range(n)],
            "replacement_fpts": [100.0] * n,
            "is_rookie": [False] * n,
        }
    )


def _avail(pool: pd.DataFrame) -> PlayerAvailability:
    return PlayerAvailability(p={g: 0.95 for g in pool["gsis_id"].astype(str)}, bye={})


def test_result_has_per_model_per_metric_intervals_and_no_winner() -> None:
    pool = _pool(40)
    cfg = _config(4)
    result = run_auction_tournament(
        {"static": StaticDollarBid()},
        pool, cfg,
        my_seat=1, n_seeds=4, price_jitter=0.1, base_seed=0, n_sims=50,
        availability=_avail(pool), params=VarianceParams.load(),
    )
    assert isinstance(result, AuctionTournamentResult)
    assert set(result.summaries["static"]) == set(METRICS)
    assert not hasattr(result, "winner")  # data-gathering: no winner field exists
    assert result.season_base_seed == 0 + 1_000_000


def test_paired_diffs_recorded_for_each_pair() -> None:
    pool = _pool(40)
    cfg = _config(4)
    result = run_auction_tournament(
        {"a": StaticDollarBid(), "b": StaticDollarBid()},
        pool, cfg,
        my_seat=1, n_seeds=3, price_jitter=0.1, base_seed=0, n_sims=40,
        availability=_avail(pool), params=VarianceParams.load(),
    )
    assert "a_vs_b" in result.paired_diffs
    assert set(result.paired_diffs["a_vs_b"]) == set(METRICS)
    # identical models, paired: every metric diff is ~0
    assert abs(result.paired_diffs["a_vs_b"]["mean_points"].point) < 1e-9


def test_league_driven_runs_under_two_configs() -> None:
    pool = _pool(60)
    avail, params = _avail(pool), VarianceParams.load()
    for cfg in (
        _config(4, budget=100),
        _config(6, budget=50, roster_slots={RosterSlot.QB: 1, RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.BENCH: 2}),
    ):
        # ensure the QB config has QBs in the pool
        p = pool.copy()
        if RosterSlot.QB in cfg.roster_slots:
            p.loc[p.index[:10], "position"] = "QB"
        res = run_auction_tournament(
            {"static": StaticDollarBid()},
            p, cfg, my_seat=1, n_seeds=2, price_jitter=0.1, base_seed=0, n_sims=30,
            availability=avail, params=params,
        )
        assert set(res.summaries["static"]) == set(METRICS)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_draft/test_assistant_auction_tournament.py -v`
Expected: FAIL — `ModuleNotFoundError: ...auction.tournament`.

- [ ] **Step 3: Implement `tournament.py`**

```python
"""Data-gathering auction tournament: race bid models, score each league with project_draft.

Records per-model per-metric means + bootstrap CIs and paired per-seed diffs. Declares NO
winner (spec §5.1) — the adopt decision is the user's, in September.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from projections.draft.assistant._compare import Interval, bootstrap_mean
from projections.draft.assistant.auction.bid_strategy import AuctionBidStrategy
from projections.draft.assistant.auction.simulation import simulate_auction, validate_auction_inputs
from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.league_projection import project_draft
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.auction import generate_auction_values
from projections.draft.league_config import LeagueConfig

METRICS: tuple[str, ...] = ("mean_points", "reg_win_pct", "make_playoffs_pct", "bye_pct", "champ_pct")


@dataclass(frozen=True)
class AuctionTournamentResult:
    """Per-model per-metric means/CIs + paired diffs. No winner (data-gathering)."""

    summaries: dict[str, dict[str, Interval]]
    paired_diffs: dict[str, dict[str, Interval]]
    n_seeds: int
    price_jitter: float
    base_seed: int
    season_base_seed: int
    n_sims: int
    my_seat: int
    budget: int
    min_bid: int


def _validate(config: LeagueConfig, *, my_seat: int, n_seeds: int, price_jitter: float, n_sims: int) -> None:
    if not 1 <= my_seat <= config.n_teams:
        raise ValueError(f"my_seat must be in 1..{config.n_teams}; got {my_seat}")
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1; got {n_seeds}")
    if price_jitter < 0:
        raise ValueError(f"price_jitter must be >= 0; got {price_jitter}")
    if n_sims < 1:
        raise ValueError(f"n_sims must be >= 1; got {n_sims}")


def run_auction_tournament(
    strategies: Mapping[str, AuctionBidStrategy],
    pool: pd.DataFrame,
    config: LeagueConfig,
    *,
    my_seat: int,
    n_seeds: int,
    price_jitter: float,
    base_seed: int,
    n_sims: int,
    availability: PlayerAvailability,
    params: VarianceParams,
    season_base_seed: int | None = None,
) -> AuctionTournamentResult:
    if season_base_seed is None:
        season_base_seed = base_seed + 1_000_000
    validate_auction_inputs(pool, config)
    _validate(config, my_seat=my_seat, n_seeds=n_seeds, price_jitter=price_jitter, n_sims=n_sims)
    baseline_dollars = generate_auction_values(pool, config)  # config-determined; computed once

    per: dict[str, dict[str, np.ndarray]] = {
        name: {m: np.empty(n_seeds, dtype=np.float64) for m in METRICS} for name in strategies
    }
    for name, strat in strategies.items():
        for s in range(n_seeds):
            league = simulate_auction(
                strat, my_seat, pool, config,
                baseline_dollars=baseline_dollars, price_jitter=price_jitter,
                rng=np.random.default_rng(base_seed + s),
            )
            proj = project_draft(
                league, pool, availability, params,
                league_config=config, n_sims=n_sims,
                rng=np.random.default_rng(season_base_seed + s),  # CRN: shared across strategies
            )
            sp = proj[my_seat]
            for m in METRICS:
                per[name][m][s] = float(getattr(sp, m))

    summaries = {
        name: {m: bootstrap_mean(per[name][m], seed=base_seed) for m in METRICS} for name in strategies
    }
    names = list(strategies)
    paired: dict[str, dict[str, Interval]] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            paired[f"{a}_vs_{b}"] = {
                m: bootstrap_mean(per[a][m] - per[b][m], seed=base_seed) for m in METRICS
            }

    return AuctionTournamentResult(
        summaries=summaries,
        paired_diffs=paired,
        n_seeds=n_seeds,
        price_jitter=price_jitter,
        base_seed=base_seed,
        season_base_seed=season_base_seed,
        n_sims=n_sims,
        my_seat=my_seat,
        budget=config.budget,
        min_bid=config.min_bid,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_draft/test_assistant_auction_tournament.py -v`
Expected: PASS.

- [ ] **Step 5: Gates + commit**

Run: `mypy src tests && ruff check src tests && ruff format --check src tests`
```bash
git add src/projections/draft/assistant/auction/tournament.py tests/test_draft/test_assistant_auction_tournament.py
git commit -m "feat(auction): run_auction_tournament (league-sim scoring, paired CIs, no winner)"
```

---

## Task 7: CLI (`auction/tournament_cli.py` + `scripts/auction_tournament.py`)

**Files:**
- Create: `src/projections/draft/assistant/auction/tournament_cli.py`
- Create: `scripts/auction_tournament.py`
- Test: `tests/test_draft/test_assistant_auction_tournament_cli.py`

**Interfaces:**
- Consumes: `run_auction_tournament`/`AuctionTournamentResult`/`METRICS` (Task 6), the three bid models (Task 3), `VorpTableSchema`/`_PYARROW_STR` (`schemas.py`), `LeagueConfig`, `load_store_availability` (`assistant/availability_loader.py`), `VarianceParams` (`assistant/performance_variance.py`), `attach_is_rookie` (`assistant/rookies.py`), `DEFAULT_PRICE_JITTER` (Task 4).
- Produces: `run(argv: list[str] | None = None) -> int`; `format_compare(result: AuctionTournamentResult) -> str` (per-model per-metric table + paired diffs, **no winner line**).

- [ ] **Step 1: Write the failing test** — `tests/test_draft/test_assistant_auction_tournament_cli.py`:

```python
import json

import numpy as np
import pandas as pd

from projections.draft.assistant.auction.tournament import AuctionTournamentResult
from projections.draft.assistant.auction.tournament_cli import format_compare, run
from projections.draft.assistant._compare import Interval
from projections.schemas import _PYARROW_STR, RosterSlot, Ruleset


def _write_pool(path) -> None:
    n = 40
    pos = ["RB" if i % 2 else "WR" for i in range(n)]
    prefix = {"RB": 2, "WR": 3}
    df = pd.DataFrame(
        {
            "gsis_id": pd.array([f"00-{prefix[pos[i]]}{i:06d}" for i in range(n)], dtype=_PYARROW_STR),
            "position": pd.array(pos, dtype=_PYARROW_STR),
            "season_mean_fpts": [float(300 - i) for i in range(n)],
            "vorp": [float(150 - i) for i in range(n)],
            "replacement_fpts": [100.0] * n,
        }
    )
    df.to_parquet(path)


def _write_config(path) -> None:
    cfg = {
        "name": "t",
        "n_teams": 4,
        "budget": 100,
        "min_bid": 1,
        "roster_slots": {"RB": 1, "WR": 1, "BENCH": 1},
        "ruleset": "espn_ppr",
    }
    path.write_text(json.dumps(cfg))


def test_format_compare_has_no_winner_line() -> None:
    iv = Interval(1.0, 0.5, 1.5)
    metrics = {m: iv for m in ("mean_points", "reg_win_pct", "make_playoffs_pct", "bye_pct", "champ_pct")}
    result = AuctionTournamentResult(
        summaries={"static": metrics, "inflation": metrics, "marginal": metrics},
        paired_diffs={"static_vs_inflation": metrics},
        n_seeds=10, price_jitter=0.15, base_seed=0, season_base_seed=1_000_000,
        n_sims=500, my_seat=1, budget=100, min_bid=1,
    )
    text = format_compare(result)
    assert "winner" not in text.lower()
    assert "champ" in text.lower()
    assert "static" in text and "inflation" in text and "marginal" in text


def test_cli_compare_smoke(tmp_path, monkeypatch) -> None:
    pool_path = tmp_path / "vorp.parquet"
    cfg_path = tmp_path / "league.json"
    _write_pool(pool_path)
    _write_config(cfg_path)

    # Stub the store-backed loaders so the smoke test needs no data store.
    import projections.draft.assistant.auction.tournament_cli as cli
    from projections.draft.assistant.availability import PlayerAvailability

    monkeypatch.setattr(cli, "load_store_availability", lambda pool, **kw: PlayerAvailability(
        p={g: 0.95 for g in pool["gsis_id"].astype(str)}, bye={}
    ))
    monkeypatch.setattr(cli, "attach_is_rookie", lambda pool, **kw: pool.assign(is_rookie=False))

    rc = run([
        "--vorp-table", str(pool_path),
        "--league-config", str(cfg_path),
        "--my-seat", "1",
        "--season", "2026",
        "--seeds", "2",
        "--n-sims", "20",
        "--price-jitter", "0.1",
        "compare",
    ])
    assert rc == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_draft/test_assistant_auction_tournament_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: ...auction.tournament_cli`.

- [ ] **Step 3: Implement `tournament_cli.py`**

```python
"""CLI engine for the auction bid-model tournament (spec §3.7). Mirrors tournament_cli.py.

`run([...])` loads the VORP pool + LeagueConfig, attaches is_rookie, loads availability +
variance params (store-backed), then races static/inflation/marginal and prints per-metric
means + CIs and paired diffs. No winner is printed (data-gathering, spec §5.1).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.draft.assistant.auction.bid_strategy import (
    AuctionBidStrategy,
    InflationBid,
    MarginalValueBid,
    StaticDollarBid,
)
from projections.draft.assistant.auction.market import DEFAULT_PRICE_JITTER
from projections.draft.assistant.auction.tournament import (
    METRICS,
    AuctionTournamentResult,
    run_auction_tournament,
)
from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.assistant.rookies import attach_is_rookie
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, VorpTableSchema

_MODELS: dict[str, AuctionBidStrategy] = {
    "static": StaticDollarBid(),
    "inflation": InflationBid(),
    "marginal": MarginalValueBid(),
}


def _load_pool(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    return VorpTableSchema.validate(df)


def _load_config(path: Path) -> LeagueConfig:
    return LeagueConfig.model_validate_json(path.read_text())


def format_compare(result: AuctionTournamentResult) -> str:
    lines: list[str] = []
    lines.append(
        f"Auction bid-model data (seat {result.my_seat}, {result.n_seeds} seeds, "
        f"n_sims={result.n_sims}, price_jitter={result.price_jitter}, "
        f"budget={result.budget}, min_bid={result.min_bid}) — no winner declared (data-gathering)."
    )
    header = f"{'model':<12}" + "".join(f"{m:>22}" for m in METRICS)
    lines.append(header)
    for name, metrics in result.summaries.items():
        cells = "".join(f"{iv.point:>10.2f} [{iv.lo_95:.1f},{iv.hi_95:.1f}]".rjust(22) for iv in (metrics[m] for m in METRICS))
        lines.append(f"{name:<12}{cells}")
    lines.append("")
    lines.append("paired per-seed differences (point [95% CI]):")
    for pair, metrics in result.paired_diffs.items():
        lines.append(f"  {pair}")
        for m in METRICS:
            iv = metrics[m]
            lines.append(f"    {m:<20} {iv.point:+.3f} [{iv.lo_95:+.3f}, {iv.hi_95:+.3f}]")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auction bid-model data-gathering harness.")
    p.add_argument("--vorp-table", type=Path, required=True, help="Consensus VORP parquet.")
    p.add_argument("--league-config", type=Path, required=True, help="LeagueConfig JSON (matches the table).")
    p.add_argument("--my-seat", type=int, required=True, help="Hero seat (1-based).")
    p.add_argument("--season", type=int, required=True, help="Season for availability/byes + is_rookie.")
    p.add_argument("--seeds", type=int, default=200, help="Paired auction sims per model.")
    p.add_argument("--price-jitter", type=float, default=DEFAULT_PRICE_JITTER, help="Bot WTP noise (fractional).")
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    p.add_argument("--n-sims", type=int, default=500, help="Monte-Carlo seasons per league (CRN).")
    p.add_argument("--data-root", type=Path, default=Path("data"), help="Store root for availability/rookies.")
    sub = p.add_subparsers(dest="mode", required=True)
    sub.add_parser("compare", help="Race static/inflation/marginal; record per-metric data.")
    return p.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    pool = _load_pool(args.vorp_table)
    config = _load_config(args.league_config)
    pool = attach_is_rookie(pool, season=args.season, data_root=args.data_root)
    availability = load_store_availability(pool, season=args.season, data_root=args.data_root)
    params = VarianceParams.load()
    result = run_auction_tournament(
        _MODELS, pool, config,
        my_seat=args.my_seat, n_seeds=args.seeds, price_jitter=args.price_jitter,
        base_seed=args.seed, n_sims=args.n_sims, availability=availability, params=params,
    )
    print(format_compare(result))
    return 0
```

- [ ] **Step 4: Implement the thin wrapper** `scripts/auction_tournament.py`:

```python
"""CLI wrapper for the auction bid-model tournament. See auction.tournament_cli."""

from __future__ import annotations

import sys

from projections.draft.assistant.auction.tournament_cli import run

if __name__ == "__main__":
    sys.exit(run())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_draft/test_assistant_auction_tournament_cli.py -v`
Expected: PASS.

- [ ] **Step 6: Full gates + commit**

Run: `pytest -v -k "auction or league_projection or compare or tournament" && mypy src tests && ruff check src tests && ruff format --check src tests`
Expected: PASS / zero violations.
```bash
git add src/projections/draft/assistant/auction/tournament_cli.py scripts/auction_tournament.py tests/test_draft/test_assistant_auction_tournament_cli.py
git commit -m "feat(auction): compare CLI (per-metric data table + paired diffs, no winner)"
```

---

## Task 8: Wire-up verification + docs sync

**Files:**
- Modify: `project_management.md`, `TODO.md` (status + decision-log entry)
- Reference: `reports/auction_tournament_validation_2026.md` (the tracking doc — fill the reproduce recipe + the fixed-setup defaults now that the CLI exists; leave the experiment log empty)

- [ ] **Step 1: Full suite green**

Run: `pytest -v`
Expected: PASS (entire suite — confirms the `_compare` promotion didn't regress the snake harness and `mean_points` didn't break the board/projection tests).

- [ ] **Step 2: All gates**

Run: `mypy src tests && ruff check src tests && ruff format --check src tests`
Expected: zero violations.

- [ ] **Step 3: Real-data smoke (manual, optional but recommended)**

Run (if `data/consensus_vorp_2026.parquet` + a matching league JSON + the 2026 store partitions are present):
```bash
python scripts/auction_tournament.py --vorp-table data/consensus_vorp_2026.parquet \
    --league-config configs/league_espn_half_16team.json --my-seat 1 --season 2026 \
    --seeds 20 --n-sims 200 compare
```
Expected: a per-model table for static/inflation/marginal + paired diffs, no winner line, in a few seconds. Paste the row(s) into `reports/auction_tournament_validation_2026.md`'s experiment log.

- [ ] **Step 4: Fill the tracking doc's reproduce recipe + defaults**

In `reports/auction_tournament_validation_2026.md`, replace the `_TBD_` defaults in "Fixed setup" with the CLI defaults (`price_jitter` = `DEFAULT_PRICE_JITTER` = 0.15, `n_sims` = 500) and the reproduce recipe stub with the real command from Step 3. Leave the experiment-log table empty (data-gathering starts after this lands).

- [ ] **Step 5: Update PM + TODO + commit**

Add a `project_management.md` status line ("Auction Slice 1 harness shipped; data-gathering, decision Sept") and a decision-log entry (league-sim scoring, StartersValuer dropped, no winner). Add/clear the relevant `TODO.md` item.
```bash
git add project_management.md TODO.md reports/auction_tournament_validation_2026.md
git commit -m "docs(auction): record Slice 1 harness; fill tracking-doc recipe/defaults"
```

---

## Self-Review

**1. Spec coverage** (each spec section → task):
- §2 `bid_strategy.py` (3 models + Protocol) → Task 3. `market.py` (bot + clearing) → Task 4. `simulation.py` (full-league return) → Task 5. `tournament.py` (per-metric CIs + paired diffs, no winner) → Task 6. CLI → Task 7. `mean_points` on `SeatProjection` → Task 1. `StartersValuer` dropped → not used anywhere (no task adds it). ✓
- §3.1 inputs: pool (no `consensus_adp` requirement — the auction sim never calls `_validate_pool`, only `validate_pool_size`) ✓; `baseline_dollars` = full frame ✓ (Task 3/5 index it by gsis_id); availability/params/`--season` ✓ (Task 7); both preconditions (size + solvency) → Task 5 `validate_auction_inputs`; `is_rookie` attached by CLI → Task 7. ✓
- §3.2 `AuctionState` (0-based) + `feasible_max` + `[min_bid, feasible_max]` clamp + seat-index conversion → Task 5. ✓
- §3.4 v1/v2/v3 + `AuctionView` fields → Task 3. ✓
- §3.5 `bot_max_bid` + second-price clearing → Task 4. ✓
- §3.6 `simulate_auction` returns full league → Task 5; `project_draft` scoring + CRN season_rng + `season_base_seed` default + per-metric means/CIs + paired diffs (no winner) → Task 6; `_compare.py` promotion → Task 2. ✓
- §3.7 CLI surface (`--season` required, `--n-sims`, no `--valuer`, no winner line) → Task 7. ✓
- §4 tests: preconditions (T5), feasible-max/reserve (T5), bid resolution (T4), conservation/full-league (T5), determinism/paired (T5), bot policy (T4), v1/v2/v3 (T3), scoring wiring + `mean_points` (T1/T6), seat-index guard (T5), recorded-comparison-no-verdict (T6), league-driven (T6), CLI smoke (T7). ✓
- §5.1 no winner → enforced in T6 (`AuctionTournamentResult` has no `winner` field; test asserts `not hasattr`) + T7 (`format_compare` has no winner line; test asserts). ✓
- §5.8 league-sim scoring, no StartersValuer → T1/T6. ✓

**2. Placeholder scan:** No `TODO`/`TBD`/"handle edge cases"/"similar to Task N" in code steps; every code step shows full code. The only `_TBD_` remaining is in the *tracking doc* (data not yet gathered), which Task 8 fills for the recipe/defaults and intentionally leaves the experiment log empty. ✓

**3. Type consistency:** `Interval`/`bootstrap_mean` names match between `_compare.py` (Task 2) and consumers (Tasks 6); `AuctionView` field names match between `bid_strategy.py` (Task 3), `_build_view` (Task 5), and the bid models' reads; `simulate_auction` return type `dict[int, list[str]]` matches `project_draft`'s `rosters: Mapping[int, list[str]]` (Task 6 call); `METRICS` tuple matches `SeatProjection` attribute names (`mean_points` added in Task 1; `reg_win_pct`/`make_playoffs_pct`/`bye_pct`/`champ_pct` existing) read via `getattr(sp, m)` in Task 6. `bot_max_bid`/`resolve_bids`/`SeatView` names match between Task 4 and Task 5. ✓

---

## Execution Handoff

Plan complete. Default execution mode: **subagent-driven-development** (fresh subagent per task, review between tasks) — tasks here are cleanly separable with explicit Consumes/Produces interfaces, so per-task isolation fits well.
