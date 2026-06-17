# Auction Sane Bots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-06-17-auction-sane-bots-design.md`

**Goal:** Give the auction-harness bots league-driven positional discipline (never leave a starting slot unfilled, never hoard a position) using one shared rule that the snake hero-vs-bots field also adopts — without changing the snake field's behavior — then re-run the bake-off to see whether the hero can compete.

**Architecture:** Promote `draft_field.py::_bot_eligible`'s deficit-reservation + per-position-cap logic into a shared, `Position`-keyed `roster_eligibility.bot_eligible(counts, picks_left, *, minimums, maximums)`, plus a league-driven `bot_position_bounds(roster_slots)`. The snake field passes its existing hand-tuned values (byte-identical); the auction engine derives bounds from `roster_slots`, gates each bot's bid (abstain on a position it can't/shouldn't roster), and nominates "a player the room can roster," with a forced-pick path that un-gates bidders in lockstep so the auction always completes. The hero stays ungated.

**Tech Stack:** Python 3.12+, numpy, pandas, pytest. No new dependencies, no schema changes.

## Global Constraints

- **`GsisId` canonical; reference enums not strings:** use `Position.QB`, `RosterSlot.FLEX` (`from projections.schemas import Position, RosterSlot`); positions stored in rosters are strings, convert with `Position(s)`.
- **League-driven:** nothing hardcodes a roster shape — auction bounds derive from `config.roster_slots` via `bot_position_bounds`.
- **Snake no drift:** `backtest/draft_field.py` keeps its exact `_MINP`/`_MAXP` *values*; the refactor only changes *how* the selection is computed (shared helper). Proven by the existing `test_draft_field.py` + backtest tests passing unchanged.
- **`bot_eligible` iteration domain:** the eligible set is drawn strictly from the `minimums`/`maximums` keysets — never `counts` or `Position` — so positions absent from the bound maps (K/DST on the snake side) are never returned.
- **Abstention handling:** a bot's `bot_max_bid` returns `0` to abstain (full seat *or* ineligible position); the engine **drops** a `0` bid before the `[min_bid, feasible_max]` clamp (else the clamp floors it to `min_bid` and defeats the gate). The hero's bid is always collected and clamped.
- **Hero ungated:** the hero seat has no eligibility gate.
- **Gates (run before declaring any task done):** `pytest -v` (named subset OK — state which), `mypy src tests` (strict, 0), `ruff check src tests` (0), `ruff format --check src tests` (clean). `round(float)` returns `int` in Py3 — don't wrap in `int(...)` (ruff RUF046). No schema/store path is touched.

## File Structure

- `src/projections/draft/roster_eligibility.py` — **modify**: add `bot_eligible` (Task 1) and `bot_position_bounds` (Task 2). Single source of truth for both callers.
- `src/projections/draft/backtest/draft_field.py` — **modify** (Task 3): `_MINP`/`_MAXP` → `Position`-keyed; delete local `_bot_eligible`; call the shared helper.
- `src/projections/draft/assistant/auction/market.py` — **modify** (Task 4): `SeatView` gains `eligible_positions`; `bot_max_bid` abstains on ineligible position.
- `src/projections/draft/assistant/auction/simulation.py` — **modify** (Task 5): compute bounds once; per-bot eligibility; nomination union + forced-pick; drop abstentions.
- `reports/auction_tournament_validation_2026.md`, `project_management.md`, `TODO.md` — **modify** (Task 6): record Run B, sync.
- Tests: `tests/test_draft/test_roster_eligibility.py` (T1, T2), `tests/test_draft/test_backtest/test_draft_field.py` (T3), `tests/test_draft/test_assistant_auction_market.py` (T4), `tests/test_draft/test_assistant_auction_simulation.py` (T5).

---

## Task 1: `bot_eligible` (shared selection algorithm)

**Files:**
- Modify: `src/projections/draft/roster_eligibility.py` (append after the existing functions)
- Test: `tests/test_draft/test_roster_eligibility.py`

**Interfaces:**
- Produces: `bot_eligible(counts: Mapping[Position, int], picks_left: int, *, minimums: Mapping[Position, int], maximums: Mapping[Position, int]) -> frozenset[Position]`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_draft/test_roster_eligibility.py` (the file already imports `Position`, `RosterSlot`; add `bot_eligible` to the `roster_eligibility` import):

```python
from projections.draft.roster_eligibility import bot_eligible  # add to existing import

_MN = {Position.QB: 1, Position.RB: 3, Position.WR: 3, Position.TE: 1}
_MX = {Position.QB: 3, Position.RB: 6, Position.WR: 6, Position.TE: 3}


def test_bot_eligible_reserves_final_picks_for_deficits() -> None:
    counts = {Position.QB: 1, Position.RB: 1, Position.WR: 3, Position.TE: 1}  # RB deficit = 2
    # picks_left == Σdeficit (2) -> forced: only positions still below minimum
    assert bot_eligible(counts, 2, minimums=_MN, maximums=_MX) == frozenset({Position.RB})


def test_bot_eligible_cap_branch_above_the_deficit_boundary() -> None:
    counts = {Position.QB: 3, Position.RB: 1, Position.WR: 1, Position.TE: 0}  # QB at max (3)
    # picks_left (10) > Σdeficit -> cap branch: every position still under its max, QB excluded
    assert bot_eligible(counts, 10, minimums=_MN, maximums=_MX) == frozenset(
        {Position.RB, Position.WR, Position.TE}
    )


def test_bot_eligible_ignores_positions_absent_from_bounds() -> None:
    counts = {Position.QB: 1, Position.K: 2}  # K not in the bound maps
    assert Position.K not in bot_eligible(counts, 10, minimums={Position.QB: 1}, maximums={Position.QB: 3})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_draft/test_roster_eligibility.py -k bot_eligible -v`
Expected: FAIL — `ImportError: cannot import name 'bot_eligible'`.

- [ ] **Step 3: Implement `bot_eligible`** — append to `roster_eligibility.py`:

```python
def bot_eligible(
    counts: Mapping[Position, int],
    picks_left: int,
    *,
    minimums: Mapping[Position, int],
    maximums: Mapping[Position, int],
) -> frozenset[Position]:
    """Positions a roster-disciplined bot may take now (the snake draft_field rule, generalized).

    Reserve the final picks for unmet minimums; otherwise allow any position still under its cap.
    The eligible set is drawn strictly from the `minimums`/`maximums` keysets, so a position present
    in `counts` but absent from the bound maps (e.g. K/DST when the bounds omit them) is never returned.
    """
    deficit = {p: max(0, minimums.get(p, 0) - counts.get(p, 0)) for p in minimums}
    if picks_left <= sum(deficit.values()):
        return frozenset(p for p, d in deficit.items() if d > 0)
    return frozenset(p for p in maximums if counts.get(p, 0) < maximums[p])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_draft/test_roster_eligibility.py -k bot_eligible -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Gates + commit**

Run: `mypy src tests && ruff check src tests && ruff format --check src tests`
```bash
git add src/projections/draft/roster_eligibility.py tests/test_draft/test_roster_eligibility.py
git commit -m "feat(draft): shared bot_eligible (deficit-reservation + cap) in roster_eligibility"
```

---

## Task 2: `bot_position_bounds` (league-driven derivation)

**Files:**
- Modify: `src/projections/draft/roster_eligibility.py` (append; add `from math import ceil` to the imports)
- Test: `tests/test_draft/test_roster_eligibility.py`

**Interfaces:**
- Produces: `bot_position_bounds(roster_slots: Mapping[RosterSlot, int]) -> tuple[dict[Position, int], dict[Position, int]]` — `(minimums, maximums)`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_draft/test_roster_eligibility.py`:

```python
from projections.draft.roster_eligibility import bot_position_bounds  # add to existing import

_SKILL = {
    RosterSlot.QB: 1, RosterSlot.RB: 2, RosterSlot.WR: 3, RosterSlot.TE: 1,
    RosterSlot.FLEX: 1, RosterSlot.BENCH: 9,
}


def test_bounds_skill_roster_min_and_max() -> None:
    mn, mx = bot_position_bounds(_SKILL)
    assert mn == {Position.QB: 1, Position.RB: 3, Position.WR: 3, Position.TE: 1}  # FLEX anchored to RB
    assert mx == {Position.QB: 3, Position.RB: 7, Position.WR: 7, Position.TE: 3}  # min + ceil bench share


def test_bounds_superflex_anchors_to_qb() -> None:
    slots = dict(_SKILL)
    del slots[RosterSlot.FLEX]
    slots[RosterSlot.SUPER_FLEX] = 1
    mn, _ = bot_position_bounds(slots)
    assert mn[Position.QB] == 2  # 1 strict + 1 super-flex


def test_bounds_sigma_max_at_least_roster_size() -> None:
    _, mx = bot_position_bounds(_SKILL)
    roster_size = sum(c for s, c in _SKILL.items() if s != RosterSlot.IR)
    assert sum(mx.values()) >= roster_size  # caps always permit a full roster
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_draft/test_roster_eligibility.py -k bounds -v`
Expected: FAIL — `ImportError: cannot import name 'bot_position_bounds'`.

- [ ] **Step 3: Implement `bot_position_bounds`** — add `from math import ceil` to the top-of-file imports, then append:

```python
def bot_position_bounds(
    roster_slots: Mapping[RosterSlot, int],
) -> tuple[dict[Position, int], dict[Position, int]]:
    """League-driven per-position (minimums, maximums) for a roster-disciplined bot.

    min = strict starting slots + flex anchored (FLEX -> RB, SUPER_FLEX -> QB); the anchor add is
    unconditional. max = min + bench distributed proportionally to min, rounded up so every cap
    leaves room for a full roster (Σmax >= roster_size).
    """
    minimums: dict[Position, int] = {}
    for slot in POSITION_SLOTS:
        count = roster_slots.get(slot, 0)
        if count > 0:
            minimums[Position(slot.value)] = count
    flex = roster_slots.get(RosterSlot.FLEX, 0)
    if flex:
        minimums[Position.RB] = minimums.get(Position.RB, 0) + flex
    superflex = roster_slots.get(RosterSlot.SUPER_FLEX, 0)
    if superflex:
        minimums[Position.QB] = minimums.get(Position.QB, 0) + superflex

    sum_min = sum(minimums.values())
    bench = roster_slots.get(RosterSlot.BENCH, 0)
    maximums = {
        pos: m + (ceil(bench * m / sum_min) if sum_min > 0 else 0) for pos, m in minimums.items()
    }
    return minimums, maximums
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_draft/test_roster_eligibility.py -k bounds -v`
Expected: PASS (3 tests). (Skill check: `sum_min=8`, `bench=9`; RB `3+ceil(27/8)=3+4=7`, QB `1+ceil(9/8)=1+2=3`.)

- [ ] **Step 5: Gates + commit**

Run: `mypy src tests && ruff check src tests && ruff format --check src tests`
```bash
git add src/projections/draft/roster_eligibility.py tests/test_draft/test_roster_eligibility.py
git commit -m "feat(draft): bot_position_bounds — league-driven bot min/max from roster_slots"
```

---

## Task 3: Snake field adopts the shared helper (no drift)

**Files:**
- Modify: `src/projections/draft/backtest/draft_field.py` (imports ~14-19; `_MINP`/`_MAXP` lines 21-22; delete `_bot_eligible` lines 53-58; call site lines 99-100)
- Test: `tests/test_draft/test_backtest/test_draft_field.py`

**Interfaces:**
- Consumes: `bot_eligible` (Task 1).

- [ ] **Step 1: Write the failing test** — append to `tests/test_draft/test_backtest/test_draft_field.py` (it already imports from the module; import `_MINP`, `_MAXP` too):

```python
from projections.draft.backtest.draft_field import _MINP, _MAXP
from projections.draft.roster_eligibility import bot_eligible
from projections.schemas import Position


def test_snake_bounds_are_position_keyed_and_exclude_k_dst() -> None:
    # The snake field's tuned values, now Position-keyed; K/DST a bot holds are never returned.
    assert _MINP == {Position.QB: 1, Position.RB: 3, Position.WR: 3, Position.TE: 1}
    assert _MAXP == {Position.QB: 3, Position.RB: 6, Position.WR: 6, Position.TE: 3}
    counts = {Position.QB: 1, Position.RB: 3, Position.WR: 3, Position.TE: 1, Position.K: 1}
    elig = bot_eligible(counts, 5, minimums=_MINP, maximums=_MAXP)
    assert Position.K not in elig  # iteration domain = bound keysets, not counts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_draft/test_backtest/test_draft_field.py::test_snake_bounds_are_position_keyed_and_exclude_k_dst -v`
Expected: FAIL — `_MINP` is still string-keyed (`{"QB":1,...}`), so `_MINP == {Position.QB:1,...}` is False (and `Position` may be unimported in the test).

- [ ] **Step 3: Convert `_MINP`/`_MAXP` to `Position`-keyed** — `draft_field.py` already imports `Position` (line 19). Replace lines 21-22:

```python
_MINP = {Position.QB: 1, Position.RB: 3, Position.WR: 3, Position.TE: 1}
_MAXP = {Position.QB: 3, Position.RB: 6, Position.WR: 6, Position.TE: 3}
```

- [ ] **Step 4: Delete the local `_bot_eligible` and import the shared one** — remove the whole `def _bot_eligible(...)` block (lines 53-58) and add to the imports (near line 14):

```python
from projections.draft.roster_eligibility import bot_eligible
```

- [ ] **Step 5: Update the bot call site** — in `draft_mixed_field`, replace the eligibility lines (currently 99-100) so `counts` is converted to `Position`-keyed and the shared helper is called, then filter `pool` by the eligible position *values*:

```python
            counts_pos = {Position(p): c for p, c in counts[seat].items()}
            elig = bot_eligible(
                counts_pos, rs - len(rosters[seat]), minimums=_MINP, maximums=_MAXP
            )
            elig_values = {p.value for p in elig}
            sub = pool[avail & pos_str.isin(elig_values)]
```
(The `counts[seat][pos_by_id[gid]] = ...` update at line 111 stays unchanged — `counts[seat]` stays string-keyed. The `if sub.empty:` fallback at lines 101-109 stays, **except** its warning interpolates `sorted(elig)`: `elig` is now a `frozenset[Position]` (was `set[str]`), so change that one interpolation to `sorted(p.value for p in elig)` to keep the warning text byte-identical to before.)

- [ ] **Step 6: Run the no-drift regression + the new test**

Run: `pytest tests/test_draft/test_backtest/test_draft_field.py tests/test_draft/test_backtest/test_harness.py tests/test_draft/test_backtest/test_hero_harness.py -v`
Expected: PASS — all existing draft-field / harness tests unchanged (proves the bot field is byte-identical) **plus** the new K/DST test. If any existing test changed output, STOP: the values or logic drifted — re-check that `_MINP`/`_MAXP` values are identical and the deficit/cap logic matches.

- [ ] **Step 7: Gates + commit**

Run: `mypy src tests && ruff check src tests && ruff format --check src tests`
```bash
git add src/projections/draft/backtest/draft_field.py tests/test_draft/test_backtest/test_draft_field.py
git commit -m "refactor(draft): snake field uses shared bot_eligible (Position-keyed, no drift)"
```

---

## Task 4: Auction bot gate (`market.py`)

**Files:**
- Modify: `src/projections/draft/assistant/auction/market.py` (`SeatView`, `bot_max_bid`; add `Position` import)
- Test: `tests/test_draft/test_assistant_auction_market.py`

**Interfaces:**
- Produces: `SeatView(open_slots: int, eligible_positions: frozenset[Position])`; `bot_max_bid(...)` returns `0` when the player's position is not in `seat_view.eligible_positions`.
- Consumes: `Position` (`projections.schemas`).

- [ ] **Step 1: Add the gate tests** — in `tests/test_draft/test_assistant_auction_market.py`, add `from projections.schemas import Position`. The existing `SeatView(open_slots=...)` constructions need **no change** — Step 3 gives `eligible_positions` a default of all-positions (ungated), so they keep working. Add the two new gate tests (which pass `eligible_positions` explicitly):

```python
def test_bot_abstains_when_position_not_eligible() -> None:
    bid = bot_max_bid(
        SeatView(open_slots=3, eligible_positions=frozenset({Position.WR})),  # RB not eligible
        _player(),  # position "RB"
        _baseline(),
        _config(),
        np.random.default_rng(0),
        price_jitter=0.0,
    )
    assert bid == 0


def test_bot_bids_when_position_eligible() -> None:
    bid = bot_max_bid(
        SeatView(open_slots=3, eligible_positions=frozenset({Position.RB})),
        _player(),  # position "RB"
        _baseline(),
        _config(),
        np.random.default_rng(0),
        price_jitter=0.0,
    )
    assert bid == 40  # baseline, unchanged when eligible
```
(`_player()` returns a Series with `position="RB"`, `gsis_id="00-0000001"`; `_baseline()` maps that id to `auction_dollars=40` — both already in the fixture file.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_draft/test_assistant_auction_market.py -v`
Expected: FAIL — the two new tests raise `TypeError: SeatView.__init__() got an unexpected keyword argument 'eligible_positions'` (the field doesn't exist yet). The existing market tests still pass (they pass only `open_slots`).

- [ ] **Step 3: Add the field + the gate** — edit `market.py`: add `from projections.schemas import Position` to the imports, then:

```python
@dataclass(frozen=True)
class SeatView:
    """Minimal per-seat view the bot reads (the engine handles feasible_max + eligibility)."""

    open_slots: int
    eligible_positions: frozenset[Position] = frozenset(Position)  # default: all positions (ungated)


def bot_max_bid(
    seat_view: SeatView,
    player: pd.Series,
    baseline_dollars: pd.DataFrame,
    config: LeagueConfig,
    rng: np.random.Generator,
    *,
    price_jitter: float,
) -> int:
    """Value-rational WTP centered on the market dollar; abstain (0) if full or position-ineligible."""
    if seat_view.open_slots <= 0:
        return 0
    if Position(player["position"]) not in seat_view.eligible_positions:
        return 0
    base = float(baseline_dollars.loc[player["gsis_id"], "auction_dollars"])
    wtp = base * (1.0 + rng.normal(0.0, price_jitter))
    return round(max(float(config.min_bid), wtp))
```
The `frozenset(Position)` default (every position = ungated = today's behavior) means existing `SeatView(open_slots=...)` constructions — including the engine's current call in `simulation.py` (line 107) — keep working unchanged, so the **full `pytest -v` stays green after Task 4** (auction sim + tournament tests still pass with ungated bots). Task 5 then passes the real per-bot eligible set. (`frozenset` is immutable, so it is a valid frozen-dataclass default; `open_slots` has no default and precedes it, so field ordering is legal.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_draft/test_assistant_auction_market.py -v`  then  `pytest -v -k "auction"`
Expected: PASS — the market file (existing tests unchanged + the two new gate tests) AND the auction simulation/tournament suites (still green because the default keeps bots ungated until Task 5).

- [ ] **Step 5: Gates + commit**

Run: `mypy src tests && ruff check src tests && ruff format --check src tests`
```bash
git add src/projections/draft/assistant/auction/market.py tests/test_draft/test_assistant_auction_market.py
git commit -m "feat(auction): bot abstains on ineligible positions (SeatView.eligible_positions)"
```

---

## Task 5: Auction engine — eligibility + nomination union + forced-pick

**Files:**
- Modify: `src/projections/draft/assistant/auction/simulation.py` (`_simulate_to_state` loop; imports)
- Test: `tests/test_draft/test_assistant_auction_simulation.py`

**Interfaces:**
- Consumes: `bot_eligible`, `bot_position_bounds` (Tasks 1-2); `SeatView(open_slots, eligible_positions)` (Task 4).
- Produces: unchanged public contracts (`_simulate_to_state -> AuctionState`, `simulate_auction -> dict[int, list[str]]`); bot rosters are now startable.

- [ ] **Step 1a: Widen the `_config` helper** — the new tests use non-default roster shapes. In `tests/test_draft/test_assistant_auction_simulation.py`, change `_config` to accept an optional `roster_slots` (the default is preserved verbatim, so every existing test in the file is unaffected):

```python
def _config(n_teams: int = 4, budget: int = 100, min_bid: int = 1, roster_slots=None) -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=n_teams,
        budget=budget,
        min_bid=min_bid,
        roster_slots=roster_slots or {RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.BENCH: 1},
        ruleset=Ruleset.espn_ppr(),
    )
```

- [ ] **Step 1b: Write the failing tests** — append to the same file (reuses `_pool`, `_baseline`, `StaticDollarBid`, the file's `_MaxBidStub` — bids `10**9`; if it is not present, add `class _MaxBidStub:` with `def max_bid(self, view, player, pool, config) -> int: return 10**9` — plus `_simulate_to_state`, `simulate_auction`, `_PYARROW_STR`, `RosterSlot`, `Ruleset`):

```python
import warnings

import pytest

from projections.schemas import Position


def test_every_bot_roster_is_startable() -> None:
    # n_teams=4, roster {RB:1, WR:1, BENCH:1} -> bounds min {RB:1,WR:1}; every bot must end
    # with >=1 RB and >=1 WR (a fillable starting lineup). The hero (seat 1) is NOT checked.
    cfg = _config(n_teams=4)
    pool = _pool(40)
    state = _simulate_to_state(
        StaticDollarBid(), 1, pool, cfg,
        baseline_dollars=_baseline(pool, cfg), price_jitter=0.1, rng=np.random.default_rng(0),
    )
    for seat in range(1, 4):  # bot seats are indices 1,2,3 (hero is index 0)
        positions = [p for (_g, p, _pr) in state.rosters[seat]]
        assert positions.count("RB") >= 1 and positions.count("WR") >= 1


def test_forced_pick_completes_and_warns_when_pool_thin() -> None:
    # 2 teams, roster {RB:1, BENCH:1}: bench-eligible is RB only (no QB slot), so a QB is never
    # rosterable. With exactly 2 RB + 2 QB, after both RBs are drafted each seat still needs a 2nd
    # player but only (un-rosterable) QBs remain -> the forced-pick path fires (ungated) and the
    # auction completes. Deterministic: no eligible position has an available player.
    cfg = _config(n_teams=2, roster_slots={RosterSlot.RB: 1, RosterSlot.BENCH: 1})
    rows = [
        {"gsis_id": f"00-2{i:06d}", "position": "RB", "season_mean_fpts": float(150 - i), "vorp": float(80 - i), "replacement_fpts": 100.0}
        for i in range(2)
    ] + [
        {"gsis_id": f"00-1{i:06d}", "position": "QB", "season_mean_fpts": float(50 - i), "vorp": float(-10 - i), "replacement_fpts": 100.0}
        for i in range(2)
    ]
    pool = pd.DataFrame(rows)
    pool["gsis_id"] = pool["gsis_id"].astype(_PYARROW_STR)
    pool["position"] = pool["position"].astype(_PYARROW_STR)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        league = simulate_auction(
            StaticDollarBid(), 1, pool, cfg,
            baseline_dollars=_baseline(pool, cfg), price_jitter=0.0, rng=np.random.default_rng(0),
        )
    assert all(len(r) == cfg.roster_size for r in league.values())  # every seat filled (2 players)
    assert any("pool thin" in str(w.message) for w in caught)  # the forced-pick path warned


def test_hero_is_not_gated() -> None:
    # RB-heavy pool: RBs carry the top baseline, so they're nominated first. A max-bidding hero
    # (high budget so its feasible_max never runs low) wins the top 3 nominations -> 3 RB, 0 WR.
    # That exceeds the bot RB cap (max RB = 2 for {RB:1,WR:1,BENCH:1}) AND strands the WR starter,
    # proving the hero has NO eligibility gate and NO starter reservation.
    cfg = _config(n_teams=4, budget=1000)  # high budget -> hero wins the first 3 picks outright
    n = 40
    pool = pd.DataFrame({
        "gsis_id": pd.array(
            [f"00-2{i:06d}" if i < n - 4 else f"00-3{i:06d}" for i in range(n)], dtype=_PYARROW_STR
        ),
        "position": pd.array(["RB"] * (n - 4) + ["WR"] * 4, dtype=_PYARROW_STR),
        "season_mean_fpts": [float(300 - i) for i in range(n)],  # RBs (first) score highest
        "vorp": [float(150 - i) for i in range(n)],
        "replacement_fpts": [100.0] * n,
    })
    state = _simulate_to_state(
        _MaxBidStub(), 1, pool, cfg,
        baseline_dollars=_baseline(pool, cfg), price_jitter=0.0, rng=np.random.default_rng(0),
    )
    hero = [p for (_g, p, _pr) in state.rosters[0]]
    assert hero.count("RB") == 3 and hero.count("WR") == 0  # > bot RB cap (2), starter stranded
```
(The existing determinism / conservation / full-league / feasible_max tests in this file must still pass after the change — run the whole file in Step 4.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_draft/test_assistant_auction_simulation.py -k "startable or forced or not_gated" -v`
Expected: FAIL — with Task 4's ungated-default `SeatView`, the engine runs (no TypeError) but has no positional gate, nomination-union, or forced-pick yet: `test_forced_pick...` fails (no `"pool thin"` warning is emitted) and `test_every_bot_roster_is_startable` fails (ungated bots aren't guaranteed an RB + WR). `test_hero_is_not_gated` may already pass (the hero is ungated before and after) — it's a guard that must stay green post-Task-5.

- [ ] **Step 3: Rewrite the `_simulate_to_state` loop** — update `simulation.py` imports and replace the body of `_simulate_to_state` (everything from `validate_auction_inputs(...)` through `return state`). Add to the imports at the top of the file:

```python
import warnings

from projections.draft.assistant.auction.market import SeatView, bot_max_bid, resolve_bids
from projections.draft.roster_eligibility import bot_eligible, bot_position_bounds
from projections.schemas import Position
```
(Keep the existing `_compare`, `bid_strategy`, `LeagueConfig`, numpy/pandas imports; `Counter` is already imported.)

Replace the `_simulate_to_state` body with:

```python
    validate_auction_inputs(pool, config)
    n = config.n_teams
    rs = config.roster_size
    min_bid = config.min_bid
    hero0 = my_seat - 1  # the single 1-based -> 0-based conversion (spec §3.2)

    minimums, maximums = bot_position_bounds(config.roster_slots)
    pos_by_id = {str(g): Position(str(p)) for g, p in zip(pool["gsis_id"], pool["position"], strict=True)}
    bd = baseline_dollars.set_index("gsis_id")
    nominate_order = bd.sort_values("auction_dollars", ascending=False).index.tolist()
    all_positions = frozenset(Position)
    state = AuctionState.initial(config)

    while any(_open_slots(state, s, rs) > 0 for s in range(n)):
        # advance the nominator pointer to a seat that still has an open slot
        while _open_slots(state, state.nominator, rs) == 0:
            state.nominator = (state.nominator + 1) % n

        # eligible positions per open seat (hero is ungated -> all positions); union over the room
        seat_eligible: dict[int, frozenset[Position]] = {}
        union: set[Position] = set()
        for seat in range(n):
            if _open_slots(state, seat, rs) <= 0:
                continue
            if seat == hero0:
                seat_eligible[seat] = all_positions
                union |= all_positions
            else:
                counts = {Position(p): c for p, c in Counter(
                    p for (_g, p, _pr) in state.rosters[seat]
                ).items()}
                elig = bot_eligible(
                    counts, _open_slots(state, seat, rs), minimums=minimums, maximums=maximums
                )
                seat_eligible[seat] = elig
                union |= elig

        # nominate the highest-baseline undrafted player the room can roster; else forced (un-gated)
        nominee_id = next(
            (g for g in nominate_order if g not in state.drafted and pos_by_id[str(g)] in union),
            None,
        )
        forced = nominee_id is None
        if forced:
            nominee_id = next(g for g in nominate_order if g not in state.drafted)
            warnings.warn(
                f"auction: no open seat can roster a remaining position; forcing nominee "
                f"{nominee_id} ({pos_by_id[str(nominee_id)].value}) ungated (pool thin).",
                stacklevel=2,
            )
        player = pool.loc[pool["gsis_id"] == nominee_id].iloc[0]

        # collect clamped bids: hero always bids; bots abstain off-eligibility (dropped) unless forced
        bids: dict[int, int] = {}
        for seat in range(n):
            if _open_slots(state, seat, rs) <= 0:
                continue
            fmax = _feasible_max(state, seat, rs, min_bid)
            if seat == hero0:
                desired = strategy.max_bid(
                    _build_view(state, hero0, pool, bd, config), player, pool, config
                )
                bids[seat] = max(min_bid, min(int(desired), fmax))
            else:
                elig = all_positions if forced else seat_eligible[seat]
                desired = bot_max_bid(
                    SeatView(open_slots=_open_slots(state, seat, rs), eligible_positions=elig),
                    player, bd, config, rng, price_jitter=price_jitter,
                )
                if desired <= 0:  # abstain -> dropped before the clamp
                    continue
                bids[seat] = max(min_bid, min(int(desired), fmax))

        assert bids, "resolve_bids requires >=1 bid; forced-pick path guarantees it"
        winner, price = resolve_bids(bids, min_bid)
        state.budgets[winner] -= price
        state.rosters[winner].append((nominee_id, str(player["position"]), price))
        state.drafted.add(nominee_id)
        state.nominator = (state.nominator + 1) % n

    return state
```

- [ ] **Step 4: Run the new tests + the full sim file**

Run: `pytest tests/test_draft/test_assistant_auction_simulation.py -v`
Expected: PASS — the new startable/forced/hero tests **and** all pre-existing tests (determinism, conservation, full-league return, feasible-max, seat-index guards) green. If a determinism test changed value, that's expected only if it asserted a *specific roster* (it should assert invariants); if it asserts an invariant it must still hold — investigate any failure as a real regression.

- [ ] **Step 5: Run the auction tournament file too** (its tests build leagues via this engine)

Run: `pytest tests/test_draft/test_assistant_auction_tournament.py -v`
Expected: PASS (the tournament still scores; sane bots don't break its structural assertions).

- [ ] **Step 6: Gates + commit**

Run: `mypy src tests && ruff check src tests && ruff format --check src tests`
```bash
git add src/projections/draft/assistant/auction/simulation.py tests/test_draft/test_assistant_auction_simulation.py
git commit -m "feat(auction): positional gating + nomination union + forced-pick (sane bots)"
```

---

## Task 6: Re-run the bake-off (Run B) + sync

**Files:**
- Modify: `reports/auction_tournament_validation_2026.md`, `project_management.md`, `TODO.md`

- [ ] **Step 1: Full suite + gates**

Run: `pytest -v` then `mypy src tests && ruff check src tests && ruff format --check src tests`
Expected: the auction + roster-eligibility + draft-field/backtest suites green; gates clean. (The pre-existing TODO #40 `test_backtest_smoke_one_cell` failure — stale WR feature cache, unrelated — may still fail; confirm it's the same one and out of scope, do not fix it here.)

- [ ] **Step 2: Re-run the bake-off (Run B — same knobs as Run A)**

Run:
```bash
python scripts/auction_tournament.py --vorp-table data/vorp_2026/half_16team.parquet \
    --league-config configs/league_espn_half_16team.json --my-seat 1 --season 2026 \
    --seeds 150 --n-sims 500 --seed 0 compare
```
These are the **identical knobs to Run A** (recorded in `reports/auction_tournament_validation_2026.md` under "Run A" / the Reproduce recipe: `half_16team`, seat 1, 150 seeds, n_sims 500, seed 0, `price_jitter` default 0.15). Before running, confirm the `--vorp-table` / `--league-config` paths match what Run A used (per the tracking doc), so Run B differs from Run A *only* in the sane-bots code change.
Expected: a per-model table + paired diffs, no winner line, in a few minutes. Capture the output.

- [ ] **Step 3: Record Run B (data point, no verdict)** — in `reports/auction_tournament_validation_2026.md`, add Run B rows to the experiment-log table and a "Run B" note beneath Run A: the per-model means, the paired diffs, and **the comparison to Run A** — specifically whether the hero's `playoff %` / `champ %` moved back toward (or above) the uniform baseline (0.375 / 0.0625) now that bots are positionally disciplined. Frame strictly as data: "in isolation, with sane bots, the hero ..."; **no winner declared** (September decision). Note byes are still off (2026 schedule not ingested).

- [ ] **Step 4: Update PM + TODO** — add a `project_management.md` line ("Auction sane-bots slice shipped: league-driven `bot_position_bounds` + shared `bot_eligible`; snake field unchanged; Run B recorded") and a decision-log note (positional discipline added to mitigate the bot-field handicap; real auction values + any WTP/clearing change remain deferred). Update/close the relevant `TODO.md` item.

- [ ] **Step 5: Commit**

```bash
git add reports/auction_tournament_validation_2026.md project_management.md TODO.md
git commit -m "data(auction): record Run B (sane bots) vs Run A; sync PM/TODO"
```

---

## Self-Review

**1. Spec coverage:**
- §4.1 shared `bot_eligible` (iteration domain pinned) → Task 1. §4.2 `bot_position_bounds` (FLEX→RB/SUPER_FLEX→QB, ceil bench, Σmax≥roster_size) → Task 2. §4.3 snake adoption, no drift → Task 3 (existing backtest tests as the regression). §4.4 `SeatView.eligible_positions` + abstain gate + drop-before-clamp → Tasks 4 (gate) & 5 (drop in engine). §4.5 nomination union + forced-pick lockstep + completion → Task 5. §3 hero-ungated → Task 5 (`test_hero_is_not_gated`). §6 tests → Tasks 1-5; bake-off → Task 6. §7 phasing maps 1:1 (Phase 1→T1+T2, Phase 2→T3, Phase 3→T4+T5, Phase 4→T6). ✓
- Edge cases (§5): K/DST absent from bounds → T1/T3 tests; pool-thin forced-pick → T5 `test_forced_pick...`; Σmin guard → `bot_position_bounds` `if sum_min > 0` branch; abstain-vs-clamp → T5 engine drops `0`. ✓

**2. Placeholder scan:** every code step has complete code; no TODO/TBD/"similar to". The only "describe" steps (T6 doc edits) are documentation, not code. ✓

**3. Type consistency:** `bot_eligible(counts, picks_left, *, minimums, maximums) -> frozenset[Position]` identical in T1 (def), T3 (snake call with `_MINP`/`_MAXP`), T5 (engine call). `bot_position_bounds(roster_slots) -> (dict, dict)` consistent T2→T5. `SeatView(open_slots, eligible_positions)` consistent T4→T5. Positions stored as strings in `AuctionState.rosters`, converted via `Position(...)` everywhere they meet the enum-keyed helpers. `round()` (no `int()` wrapper) per the gate rule. ✓

---

## Execution Handoff

Plan complete. Default mode: **subagent-driven-development** — six tasks with explicit Consumes/Produces interfaces and independent test cycles; Task 3's no-drift regression and Task 5's engine rewrite each warrant a fresh reviewer's gate.
