# Auction Nomination Poisoning (Feasibility Probe) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-07-15-auction-nomination-poisoning-design.md`

**Goal:** Give the auction hero opt-in control of its own nominations, then probe whether two "poison" heuristics (drain-max, drain-off-position) lift seat-averaged `reg_win_pct` above the ~0.59 `balanced` p0.0 baseline in both markets — a go/no-go before building a full nomination abstraction.

**Architecture:** A behavior-preserving `hero_nominator` hook on the auction engine (`_simulate_to_state`/`simulate_auction`) is consulted only on the hero seat's non-forced nomination turns; `None` is byte-identical to today. The two poison heuristics live in a new `nomination.py` module and consume a small `NominationContext`. A scratch crash-safe runner races `control`/`drain_max`/`drain_off_position` (all bidding `balanced` p0.0) with Common Random Numbers, and a CRN-paired analysis produces the verdict.

**Tech Stack:** Python 3.12, numpy, pandas, pytest (+ xdist), mypy strict, ruff. Auction engine under `src/projections/draft/assistant/auction/`.

## Global Constraints

- Reference enums, never the strings they wrap: `Position.RB`, not `"RB"` (copy verbatim from CLAUDE.md).
- `mypy src tests` strict, `ruff check`, `ruff format --check` must all be clean (zero violations); no broad `# type: ignore`/`# noqa`.
- The retuned `balanced` default is `BalancedValueBid(premium=0.0)` — every probe contestant uses it as the fixed bid strategy; only the nominator varies (spec R5).
- `hero_nominator=None` must be byte-identical to the current engine (spec R1).
- The go/no-go is computed from the **CRN-paired** `poison − control` per-`(seed, seat)` lift, not a comparison of independent levels (spec R7). Go = `min(Δ_model, Δ_espn) ≥ +0.02` and seat-stable (paired lift positive at a majority of the 12 seats) in both markets.
- Bounded/chunked runs only (dev-box Raptor Lake fault): one `(seat, market)` per process.

---

### Task 1: Nomination module — `NominationContext` + the two poison heuristics

Build the pure, sim-independent pieces first (no engine wiring yet), fully unit-tested.

**Files:**
- Create: `src/projections/draft/assistant/auction/nomination.py`
- Test: `tests/test_draft/test_assistant_auction_nomination.py`

**Interfaces:**
- Produces:
  - `NominationContext` — frozen dataclass with fields `hero_positions: Counter[Position]`, `value_by_id: Mapping[str, float]`, `position_by_id: Mapping[str, Position]`, `position_minimums: Mapping[Position, int]`.
  - `HeroNominator = Callable[[list[str], NominationContext], str]`.
  - `drain_max(candidates: list[str], ctx: NominationContext) -> str`
  - `drain_off_position(candidates: list[str], ctx: NominationContext) -> str`

> **Spec-refinement note:** the spec's `NominationContext` listed `config`. Planning found the heuristics actually need (a) each candidate's position and (b) the per-position starter minimum. So the context carries the two precomputed lookups `position_by_id` + `position_minimums` (both derived from `config.roster_slots` via `bot_position_bounds`, computed once in the engine) instead of raw `config`. This is strictly what the heuristics consume; nothing else changes.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_draft/test_assistant_auction_nomination.py
from collections import Counter
from collections.abc import Mapping

from projections.draft.assistant.auction.nomination import (
    NominationContext,
    drain_max,
    drain_off_position,
)
from projections.schemas import Position


def _ctx(
    hero_positions: dict[Position, int],
    value_by_id: Mapping[str, float],
    position_by_id: Mapping[str, Position],
    position_minimums: Mapping[Position, int],
) -> NominationContext:
    return NominationContext(
        hero_positions=Counter(hero_positions),
        value_by_id=value_by_id,
        position_by_id=position_by_id,
        position_minimums=position_minimums,
    )


def test_drain_max_returns_the_priciest_candidate() -> None:
    ctx = _ctx(
        {},
        {"a": 10.0, "b": 30.0, "c": 20.0},
        {"a": Position.RB, "b": Position.WR, "c": Position.QB},
        {Position.RB: 2, Position.WR: 3, Position.QB: 1},
    )
    assert drain_max(["a", "b", "c"], ctx) == "b"  # 30.0 is the max value


def test_drain_off_position_drains_a_filled_position_not_the_priciest() -> None:
    # Hero has filled RB (2 >= min 2); WR is unfilled (0 < 3). 'b' (WR, $30) is priciest overall,
    # but the off-position pick is 'c' (RB, $20) — drain the slot the hero is done with.
    ctx = _ctx(
        {Position.RB: 2},
        {"a": 10.0, "b": 30.0, "c": 20.0},
        {"a": Position.WR, "b": Position.WR, "c": Position.RB},
        {Position.RB: 2, Position.WR: 3},
    )
    assert drain_off_position(["a", "b", "c"], ctx) == "c"


def test_drain_off_position_falls_back_to_drain_max_when_nothing_filled() -> None:
    # Hero has filled no position -> no off-position candidate -> fall back to the priciest overall.
    ctx = _ctx(
        {},
        {"a": 10.0, "b": 30.0, "c": 20.0},
        {"a": Position.RB, "b": Position.WR, "c": Position.QB},
        {Position.RB: 2, Position.WR: 3, Position.QB: 1},
    )
    assert drain_off_position(["a", "b", "c"], ctx) == "b"  # == drain_max
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_draft/test_assistant_auction_nomination.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'projections.draft.assistant.auction.nomination'`.

- [ ] **Step 3: Write the module**

```python
# src/projections/draft/assistant/auction/nomination.py
"""Hero nomination strategies for the auction (Slice 2 feasibility probe).

A `HeroNominator` picks the hero's nominee from the room-rosterable `candidates`. The poison
heuristics aim to drain opponents' budgets: `drain_max` nominates the priciest player (forcing the
room to spend on a stud the capped hero would lose anyway); `drain_off_position` nominates the
priciest player at a position the hero has already filled, so the drain lands on opponents who still
need that slot. See docs/superpowers/specs/2026-07-15-auction-nomination-poisoning-design.md.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from projections.schemas import Position


@dataclass(frozen=True)
class NominationContext:
    """What a HeroNominator reads to choose the hero's nominee.

    `value_by_id` is the market value the room bids on (`bot_dollars`), so "priciest" means "biggest
    drain" in both the model and ESPN markets (they differ under ESPN anchoring). `hero_positions`
    is the hero's drafted position counts; `position_minimums` the per-position starter requirement
    (`bot_position_bounds`); `position_by_id` each candidate's position.
    """

    hero_positions: Counter[Position]
    value_by_id: Mapping[str, float]
    position_by_id: Mapping[str, Position]
    position_minimums: Mapping[Position, int]


HeroNominator = Callable[[list[str], NominationContext], str]


def drain_max(candidates: list[str], ctx: NominationContext) -> str:
    """Nominate the priciest room-rosterable player (max value the room bids on)."""
    return max(candidates, key=lambda g: ctx.value_by_id[str(g)])


def drain_off_position(candidates: list[str], ctx: NominationContext) -> str:
    """Nominate the priciest candidate at a position the hero has already filled to its starter
    requirement; fall back to `drain_max` (priciest overall) when none qualifies."""
    off = [
        g
        for g in candidates
        if ctx.hero_positions[ctx.position_by_id[str(g)]]
        >= ctx.position_minimums.get(ctx.position_by_id[str(g)], 0)
    ]
    return max(off or candidates, key=lambda g: ctx.value_by_id[str(g)])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_draft/test_assistant_auction_nomination.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Lint + type-check the new files**

Run: `python -m ruff check src/projections/draft/assistant/auction/nomination.py tests/test_draft/test_assistant_auction_nomination.py && python -m ruff format --check src/projections/draft/assistant/auction/nomination.py tests/test_draft/test_assistant_auction_nomination.py && python -m mypy src tests`
Expected: `All checks passed!`, formatted, `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add src/projections/draft/assistant/auction/nomination.py tests/test_draft/test_assistant_auction_nomination.py
git commit -m "feat(auction): nomination poison heuristics (drain-max, drain-off-position)"
```

---

### Task 2: Wire the `hero_nominator` hook into the engine

Add the opt-in hook to `_simulate_to_state` and thread it through `simulate_auction`. `None` stays byte-identical.

**Files:**
- Modify: `src/projections/draft/assistant/auction/simulation.py`
- Test: `tests/test_draft/test_assistant_auction_simulation.py`

**Interfaces:**
- Consumes: `NominationContext`, `HeroNominator`, `drain_max` from Task 1.
- Produces: `_simulate_to_state(..., hero_nominator: HeroNominator | None = None)` and `simulate_auction(..., hero_nominator: HeroNominator | None = None)`. When `state.nominator == hero0`, `hero_nominator is not None`, and the nomination is non-forced, the hero's nominee is `hero_nominator(candidates, ctx)`; the engine asserts the result is in `candidates`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_draft/test_assistant_auction_simulation.py`)

```python
def test_hero_nominator_none_matches_default() -> None:
    # None hook is byte-identical to no hook: same rosters and budgets (spec R1).
    cfg = _config(n_teams=4)
    pool = _pool(40)
    bd = _baseline(pool, cfg)
    kw = dict(baseline_dollars=bd, price_jitter=0.1)
    a = _simulate_to_state(StaticDollarBid(), 1, pool, cfg, rng=np.random.default_rng(0), **kw)
    b = _simulate_to_state(
        StaticDollarBid(), 1, pool, cfg, rng=np.random.default_rng(0), hero_nominator=None, **kw
    )
    assert a.rosters == b.rosters
    assert a.budgets == b.budgets


def test_hero_nominator_choice_changes_the_draft() -> None:
    # Two different hero nominators (priciest vs cheapest) diverge deterministically -> the hook
    # fires and the returned id is actually used (spec R2 behavioral evidence).
    cfg = _config(n_teams=4)
    pool = _pool(40)
    bd = _baseline(pool, cfg)
    kw = dict(baseline_dollars=bd, price_jitter=0.1, nomination_temp=0.0)
    priciest = _simulate_to_state(
        StaticDollarBid(), 1, pool, cfg, rng=np.random.default_rng(0),
        hero_nominator=lambda c, ctx: max(c, key=lambda g: ctx.value_by_id[str(g)]), **kw,
    )
    cheapest = _simulate_to_state(
        StaticDollarBid(), 1, pool, cfg, rng=np.random.default_rng(0),
        hero_nominator=lambda c, ctx: min(c, key=lambda g: ctx.value_by_id[str(g)]), **kw,
    )
    assert priciest.rosters != cheapest.rosters
    # both are still valid full drafts
    for state in (priciest, cheapest):
        assert all(len(r) == cfg.roster_size for r in state.rosters)
        ids = [g for r in state.rosters for (g, _p, _pr) in r]
        assert len(ids) == len(set(ids))


def test_hero_nominator_receives_only_valid_candidate_lists() -> None:
    # Every candidate list the hero sees is non-empty and dup-free (spec R3, checked live).
    cfg = _config(n_teams=4)
    pool = _pool(40)
    bd = _baseline(pool, cfg)
    calls = 0

    def spy(candidates: list[str], ctx: object) -> str:
        nonlocal calls
        calls += 1
        assert candidates
        assert len({str(g) for g in candidates}) == len(candidates)
        return candidates[0]

    _simulate_to_state(
        StaticDollarBid(), 1, pool, cfg, baseline_dollars=bd, price_jitter=0.1,
        rng=np.random.default_rng(0), hero_nominator=spy,
    )
    assert calls > 0
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `python -m pytest tests/test_draft/test_assistant_auction_simulation.py -k hero_nominator -q`
Expected: FAIL — `_simulate_to_state() got an unexpected keyword argument 'hero_nominator'`.

- [ ] **Step 3: Add the imports and the `bot_by_id` lookup to `simulation.py`**

Add to the imports block (near the other `auction` imports):

```python
from projections.draft.assistant.auction.nomination import HeroNominator, NominationContext
```

In `_simulate_to_state`, immediately AFTER the block that sets `bd["bot_dollars"]` (the `if bot_dollars is None: ... else: ...`), add a once-built lookup:

```python
    bot_by_id = {str(g): float(v) for g, v in bd["bot_dollars"].items()}  # value the room bids on
```

(`minimums` and `pos_by_id` already exist above; `bot_by_id` is the only new precompute.)

- [ ] **Step 4: Add the `hero_nominator` parameter and the hook**

Add the parameter to BOTH signatures (keyword-only, default `None`):

`_simulate_to_state(...)`:
```python
    bot_dollars: pd.Series | None = None,
    trace: list[PickRecord] | None = None,
    hero_nominator: HeroNominator | None = None,
) -> AuctionState:
```

`simulate_auction(...)` — add the same parameter and pass it through:
```python
    bot_dollars: pd.Series | None = None,
    hero_nominator: HeroNominator | None = None,
) -> dict[int, list[str]]:
    ...
    state = _simulate_to_state(
        strategy,
        my_seat,
        pool,
        config,
        baseline_dollars=baseline_dollars,
        price_jitter=price_jitter,
        rng=rng,
        snake_rng=snake_rng,
        nomination_temp=nomination_temp,
        bot_archetypes=bot_archetypes,
        bot_dollars=bot_dollars,
        hero_nominator=hero_nominator,
    )
```

In `_simulate_to_state`, replace the final fallback in the non-forced nomination branch. Find:

```python
            if nominee_id is None:
                nominee_id = _sample_nominee(candidates, val_by_id, nomination_temp, rng)
```

Replace with:

```python
            if nominee_id is None:
                if nom == hero0 and hero_nominator is not None:
                    ctx = NominationContext(
                        hero_positions=Counter(
                            Position(p) for (_g, p, _pr) in state.rosters[hero0]
                        ),
                        value_by_id=bot_by_id,
                        position_by_id=pos_by_id,
                        position_minimums=minimums,
                    )
                    nominee_id = hero_nominator(candidates, ctx)
                    assert nominee_id in candidates, (
                        "hero_nominator must return a member of candidates"
                    )
                else:
                    nominee_id = _sample_nominee(candidates, val_by_id, nomination_temp, rng)
```

(`Counter` and `Position` are already imported in `simulation.py`.)

- [ ] **Step 5: Run the hook tests + the full simulation suite**

Run: `python -m pytest tests/test_draft/test_assistant_auction_simulation.py -q`
Expected: PASS (all prior tests + the 3 new hook tests).

- [ ] **Step 6: Gates**

Run: `python -m mypy src tests && python -m ruff check src tests && python -m ruff format --check src tests`
Expected: `Success: no issues found`, `All checks passed!`, all formatted.

- [ ] **Step 7: Commit**

```bash
git add src/projections/draft/assistant/auction/simulation.py tests/test_draft/test_assistant_auction_simulation.py
git commit -m "feat(auction): opt-in hero_nominator hook on the auction loop (Slice 2)"
```

---

### Task 3: Regression guard — the whole auction suite is green under the new default

The retuned bid default (`premium=0.0`) plus the new hook are both live; confirm nothing else regressed before the compute-heavy probe.

**Files:**
- Test: (existing) all `tests/test_draft/test_assistant_auction_*.py`

- [ ] **Step 1: Run the full auction test suite**

Run: `python -m pytest tests/test_draft/test_assistant_auction_bid_strategy.py tests/test_draft/test_assistant_auction_market.py tests/test_draft/test_assistant_auction_simulation.py tests/test_draft/test_assistant_auction_tournament.py tests/test_draft/test_assistant_auction_tournament_cli.py tests/test_draft/test_assistant_auction_snake_bot.py tests/test_draft/test_auction.py tests/test_draft/test_assistant_auction_nomination.py -q`
Expected: all pass.

- [ ] **Step 2: Full gates**

Run: `python -m mypy src tests && python -m ruff check src tests && python -m ruff format --check src tests`
Expected: clean. (No commit — this task is a gate, not a code change.)

---

### Task 4: The probe — CRN-paired both-market sweep + Run O verdict

Race `control`/`drain_max`/`drain_off_position` (all `balanced` p0.0) with CRN, compute the paired lift, apply the R8 sanity gate and the go/no-go, and write it up. The runner is a **scratch** experiment harness (committed only if the probe graduates), mirroring the Run N `premium_sweep.py` structure.

**Files:**
- Create (scratch): `<scratchpad>/nomination_probe.py`
- Create (scratch): `<scratchpad>/nomination_probe_driver.sh`
- Modify: `reports/auction_tournament_validation_2026.md` (add "Run O")
- Modify: memory `auction-bid-model-investigation-status.md` + `MEMORY.md`

**Interfaces:**
- Consumes: `simulate_auction(..., hero_nominator=...)`, `drain_max`, `drain_off_position`, `BalancedValueBid(premium=0.0)`, `_load_tournament_inputs`, `_REALISTIC_FIELD`, `generate_auction_values`, `espn_anchored_bot_prices`, `_SNAKE_SUBSTREAM`, `project_draft`.

- [ ] **Step 1: Write the scratch probe runner**

Write `<scratchpad>/nomination_probe.py` (adapts Run N's `premium_sweep.py`; the crucial addition is recording **per-`(seed, seat)`** `reg_win_pct` per contestant so the CRN-paired lift is computable, per spec R7). Contestants all bid `BalancedValueBid(premium=0.0)`; only `hero_nominator` varies: `control`→`None`, `drain_max`→`drain_max`, `drain_off_position`→`drain_off_position`.

```python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from projections.draft.assistant.auction.bid_strategy import BalancedValueBid
from projections.draft.assistant.auction.nomination import drain_max, drain_off_position
from projections.draft.assistant.auction.simulation import simulate_auction
from projections.draft.assistant.auction.tournament import _SNAKE_SUBSTREAM
from projections.draft.assistant.auction.tournament_cli import (
    _REALISTIC_FIELD,
    _load_tournament_inputs,
)
from projections.draft.assistant.league_projection import project_draft
from projections.draft.auction import espn_anchored_bot_prices, generate_auction_values

POOL = Path("data/vorp_2026/half_12team.parquet")
CFG = Path("data/vorp_2026/half_12team.league.json")
SEASON = 2026
BID = BalancedValueBid(premium=0.0)
NOMINATORS = {"control": None, "drain_max": drain_max, "drain_off_position": drain_off_position}


def _run_chunk(a: argparse.Namespace) -> int:
    pool, config, availability, params = _load_tournament_inputs(
        POOL, CFG, season=SEASON, data_root=Path("data")
    )
    bd = generate_auction_values(pool, config)
    bot_dollars = (
        espn_anchored_bot_prices(pool, config, model_values=bd, unranked_discount=None)
        if a.market == "espn"
        else None
    )
    season_base = a.seed_base + 1_000_000
    # name -> list of per-seed reg_win_pct at THIS seat (CRN paired across names)
    per_seed: dict[str, list[float]] = {n: [] for n in NOMINATORS}
    extra: dict[str, dict[str, list[float]]] = {
        n: {"make_playoffs_pct": [], "champ_pct": []} for n in NOMINATORS
    }
    for s in range(a.seeds):
        base = a.seed_base + s
        for name, nominator in NOMINATORS.items():
            league = simulate_auction(
                BID,
                a.seat,
                pool,
                config,
                baseline_dollars=bd,
                price_jitter=0.15,
                rng=np.random.default_rng(base),
                snake_rng=np.random.default_rng([base, _SNAKE_SUBSTREAM]),
                nomination_temp=1.0,
                bot_archetypes=_REALISTIC_FIELD,
                bot_dollars=bot_dollars,
                hero_nominator=nominator,
            )
            proj = project_draft(
                league, pool, availability, params, league_config=config,
                n_sims=a.n_sims, rng=np.random.default_rng(season_base + s),
            )[a.seat]
            per_seed[name].append(float(proj.reg_win_pct))
            extra[name]["make_playoffs_pct"].append(float(proj.make_playoffs_pct))
            extra[name]["champ_pct"].append(float(proj.champ_pct))
    payload = {
        "market": a.market,
        "seat": a.seat,
        "seeds": a.seeds,
        "n_sims": a.n_sims,
        "reg_win_pct_per_seed": per_seed,  # name -> [per-seed]; CRN aligned by index
        "extra_mean": {
            n: {m: float(np.mean(extra[n][m])) for m in extra[n]} for n in NOMINATORS
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {a.out} (market={a.market}, seat={a.seat})")
    return 0


def _aggregate(a: argparse.Namespace) -> int:
    chunks = []
    for p in sorted(a.chunk_dir.glob("*.json")):
        data = json.loads(p.read_text())
        if isinstance(data, dict) and "market" in data and "reg_win_pct_per_seed" in data:
            chunks.append(data)
    markets = sorted({c["market"] for c in chunks})
    names = list(NOMINATORS)
    # seat-avg level per (market, name), and CRN-paired lift vs control per (market, name)
    level: dict[tuple[str, str], list[float]] = defaultdict(list)  # -> per-seat means
    paired: dict[tuple[str, str], list[float]] = defaultdict(list)  # -> per-seat paired lift
    for c in chunks:
        m = c["market"]
        ctrl = c["reg_win_pct_per_seed"]["control"]
        for name in names:
            vals = c["reg_win_pct_per_seed"][name]
            level[(m, name)].append(float(np.mean(vals)))
            # paired lift at this seat = mean over seeds of (name - control), CRN aligned by index
            paired[(m, name)].append(float(np.mean(np.array(vals) - np.array(ctrl))))

    def savg(d: dict[tuple[str, str], list[float]], m: str, n: str) -> float:
        return float(np.mean(d[(m, n)])) if d[(m, n)] else float("nan")

    print(f"{'contestant':<20}" + "".join(f"{m + ' lvl':>14}{m + ' Δ':>12}" for m in markets))
    for name in names:
        cells = ""
        for m in markets:
            cells += f"{savg(level, m, name):>14.3f}{savg(paired, m, name):>12.3f}"
        print(f"{name:<20}{cells}")
    print("\ncontext (seat-avg make_playoffs_pct / champ_pct):")
    for name in names:
        cells = ""
        for m in markets:
            pos = [c["extra_mean"][name]["make_playoffs_pct"] for c in chunks if c["market"] == m]
            ch = [c["extra_mean"][name]["champ_pct"] for c in chunks if c["market"] == m]
            cells += f"  {m}: {np.mean(pos):.3f}/{np.mean(ch):.3f}"
        print(f"  {name:<20}{cells}")
    print("\ngo/no-go (paired lift vs control; go = min market Δ >= +0.02 AND seat-stable):")
    for name in names:
        if name == "control":
            continue
        market_deltas = {m: savg(paired, m, name) for m in markets}
        # seat-stable = paired lift positive at a majority of the 12 seats, in each market
        stable = {
            m: sum(1 for v in paired[(m, name)] if v > 0) > len(paired[(m, name)]) / 2
            for m in markets
        }
        worst = min(market_deltas.values())
        verdict = "GO" if worst >= 0.02 and all(stable.values()) else "no-go"
        print(f"  {name:<20} min Δ={worst:+.3f}  seat-stable={stable}  -> {verdict}")
    return 0


def _args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seat", type=int, required=True)
    r.add_argument("--market", choices=("espn", "model"), required=True)
    r.add_argument("--seeds", type=int, default=20)
    r.add_argument("--n-sims", type=int, default=300)
    r.add_argument("--seed-base", type=int, default=0)
    r.add_argument("--out", type=Path, required=True)
    r.set_defaults(func=_run_chunk)
    g = sub.add_parser("aggregate")
    g.add_argument("--chunk-dir", type=Path, required=True)
    g.set_defaults(func=_aggregate)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-test one chunk (tiny) to catch errors before the long run**

Run: `python <scratchpad>/nomination_probe.py run --seat 1 --market model --seeds 2 --n-sims 20 --out reports/_nom_probe/smoke.json` then inspect and delete: `cat reports/_nom_probe/smoke.json && rm reports/_nom_probe/smoke.json`
Expected: writes a JSON with `reg_win_pct_per_seed` holding 3 keys (`control`, `drain_max`, `drain_off_position`), each a 2-element list. No traceback.

- [ ] **Step 3: Write the crash-safe driver**

Write `<scratchpad>/nomination_probe_driver.sh` (resumable, one bounded process per `(seat, market)`, 12 seats × 2 markets = 24 chunks, 20 seeds × 300 sims), modeled exactly on the Run N `prem_sweep_driver.sh`, writing to `reports/_nom_probe/2026/`.

- [ ] **Step 4: Launch the driver in the background and wait for completion**

Run the driver with `run_in_background: true`; it emits `reports/_nom_probe/2026/*.json` and a `driver.log`. Expected ~30 min, all 24 chunks `rc=0`.

- [ ] **Step 5: Aggregate + apply the R8 sanity gate**

Run: `python <scratchpad>/nomination_probe.py aggregate --chunk-dir reports/_nom_probe/2026`
**R8 gate:** confirm `control`'s seat-avg reproduces Run N `balanced` p0.0 (~0.592 model / ~0.621 espn) within seed noise. If not, STOP — the harness is wrong; do not read poison lift.

- [ ] **Step 6: Record the verdict — Run O + memory**

Add a "Run O" section to `reports/auction_tournament_validation_2026.md`: the level + paired-lift table, the R8 sanity check, and the **go/no-go** per the spec bar (`min market Δ ≥ +0.02` AND seat-stable in both markets). Update the memory file with the outcome. State the decision explicitly:
- **Go** → next slice: full `NominationStrategy` abstraction. Keep the hook.
- **No-go** → the hook is dead code; either delete it in a follow-up commit or leave it behind a clearly-labeled "probe, not adopted" note per the user's call. `balanced` p0.0 remains the hero.

- [ ] **Step 7: Commit the writeup**

```bash
git add reports/auction_tournament_validation_2026.md
git commit -m "docs(auction): Run O — nomination-poisoning probe verdict (Slice 2)"
```

---

## Self-Review

**Spec coverage:**
- Seam (opt-in `hero_nominator`, hero-only, non-forced, None=identity) → Task 2 (+ R1/R2/R3 tests).
- Two heuristics (`drain_max`, `drain_off_position` with fallback) → Task 1 (+ R4 tests).
- Bid fixed at `balanced` p0.0 (R5) → Task 4 `BID` constant.
- CRN-paired decision (R7) → Task 4 records per-`(seed, seat)`, aggregates paired lift.
- Control sanity gate (R8) → Task 4 Step 5.
- Go/no-go (min market Δ ≥ +0.02, seat-stable) → Task 4 Step 6 + aggregate verdict.
- Gates (R6) → Steps in every code task + Task 3.
- Edge cases (backfire/forced/early-draft/value-source) → covered by design: forced path untouched (hook only on non-forced), early-draft fallback tested (Task 1), value = `bot_dollars` (Task 2 `bot_by_id`).

**Placeholder scan:** none — every code step has complete code; `<scratchpad>` is the session scratch dir path (a real location, not a placeholder).

**Type consistency:** `NominationContext` fields, `HeroNominator` alias, `drain_max`/`drain_off_position` signatures, and the `hero_nominator` parameter name are used identically across Tasks 1, 2, and 4.
