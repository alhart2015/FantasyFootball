# Big-Stack Overspend Hero Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-07-16-auction-bigstack-overspend-design.md`

**Goal:** Add a `BigStackBid` hero that overpays (above fair value, lifting the low pace cap) in proportion to the hero's remaining-budget advantage over the field, and is identical to `balanced` p0.0 when not the big stack — then A/B two "big stack?" variants under the realistic ADP-nomination market.

**Architecture:** A new frozen dataclass `BigStackBid` in `bid_strategy.py` computes the normal balanced bid plus an `overpay` term driven by a stack-advantage signal (two variants: vs the richest opponent, or vs the league-average per-slot budget). It returns a *desired* bid; the engine already clamps to `[min_bid, feasible_max]`, so it can never overspend into insolvency. A scratch sweep runner races the variants × an `overpay_gain` grid vs the `balanced` control, reusing `run_auction_tournament` (which threads `market_adp_jitter`) and `auction_seat_sweep.aggregate_seat_sweep`.

**Tech Stack:** Python 3.12, numpy/pandas, pytest, mypy strict, ruff. Auction code under `src/projections/draft/assistant/auction/`.

## Global Constraints

- `reference` is a `Literal["max_opp", "field_avg"]`; invalid values raise at construction (mirror `BalancedValueBid`'s `__post_init__`) — spec R7.
- At `advantage ≤ 1` (or `overpay_gain=0`), `BigStackBid(premium=p, pace=q)` is **exactly** `BalancedValueBid(premium=p, pace=q)` (no `non_increasing_cap`) — spec R1.
- Strategies return a *desired* int bid; the engine clamps to `[min_bid, feasible_max]` — never re-implement the reserve. Solvency is the engine's invariant (spec R4).
- The A/B changes ONLY the bid; nomination (`market_adp_jitter=12`), bot field, markets, seeds (20), sims (300) match Run P (spec R5).
- Gates: `pytest`, `mypy src tests` (strict), `ruff check`, `ruff format --check` clean (spec R6).
- Data-gathering: no adopt/reject bar; the Phase-3 deliverable is the characterized lift vs `balanced` p0.0, noise-flagged (spec "Interpretation").
- Merge order: this branch is stacked on PR #99 (`market_adp_jitter`); it rebases onto / lands after #99.

---

### Task 1: `BigStackBid` strategy + unit tests

**Files:**
- Modify: `src/projections/draft/assistant/auction/bid_strategy.py` (add `Literal` import + `BigStackBid` after `BalancedValueBid`)
- Test: `tests/test_draft/test_assistant_auction_bid_strategy.py`

**Interfaces:**
- Consumes: `AuctionView` (fields `my_budget`, `my_open_slots`, `budgets_by_seat`, `baseline_dollars`), `_total_open_slots(view, config)`, `BalancedValueBid` — all already in `bid_strategy.py`.
- Produces: `BigStackBid(overpay_gain: float = 1.0, reference: Literal["max_opp","field_avg"] = "field_avg", premium: float = 0.0, pace: float = 2.0)` with `max_bid(view, player, pool, config) -> int`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_draft/test_assistant_auction_bid_strategy.py`)

```python
def test_bigstack_falls_back_to_balanced_when_not_big_stack() -> None:
    # Hero is the short stack ($50 vs $100) -> advantage <= 1 for BOTH references -> no overpay ->
    # identical to BalancedValueBid (defaults, and a non-default premium/pace).
    pool = _pool()
    baseline = _baseline([True, True, False, False], [60, 40, 0, 0])
    view = AuctionView(
        my_budget=50, my_open_slots=3, my_positions=Counter(), my_roster=pool.iloc[:0],
        drafted=frozenset(), budgets_by_seat=(50, 100), baseline_dollars=baseline,
    )
    for ref in ("max_opp", "field_avg"):
        bs = BigStackBid(reference=ref, overpay_gain=1.0).max_bid(view, pool.iloc[0], pool, _config())
        assert bs == BalancedValueBid(premium=0.0).max_bid(view, pool.iloc[0], pool, _config())
    bs = BigStackBid(reference="field_avg", overpay_gain=1.0, premium=0.15, pace=1.5).max_bid(
        view, pool.iloc[0], pool, _config()
    )
    assert bs == BalancedValueBid(premium=0.15, pace=1.5).max_bid(view, pool.iloc[0], pool, _config())


def test_bigstack_overpays_above_balanced_when_big_stack() -> None:
    # Hero is the big stack ($60 vs $10). fair=60, per_slot=20 -> balanced cap binds at 40. The
    # overpay lifts both target and cap -> bid strictly above the balanced 40.
    pool = _pool()
    baseline = _baseline([True, True, False, False], [60, 40, 0, 0])
    view = AuctionView(
        my_budget=60, my_open_slots=3, my_positions=Counter(), my_roster=pool.iloc[:0],
        drafted=frozenset(), budgets_by_seat=(60, 10), baseline_dollars=baseline,
    )
    bal = BalancedValueBid(premium=0.0).max_bid(view, pool.iloc[0], pool, _config())
    for ref in ("max_opp", "field_avg"):
        bs = BigStackBid(reference=ref, overpay_gain=1.0).max_bid(view, pool.iloc[0], pool, _config())
        assert bs > bal


def test_bigstack_gain_zero_is_balanced_even_when_big_stack() -> None:
    # overpay_gain=0 -> overpay is always 0, even as the big stack -> exactly balanced.
    pool = _pool()
    baseline = _baseline([True, True, False, False], [60, 40, 0, 0])
    view = AuctionView(
        my_budget=100, my_open_slots=3, my_positions=Counter(), my_roster=pool.iloc[:0],
        drafted=frozenset(), budgets_by_seat=(100, 20), baseline_dollars=baseline,
    )
    for ref in ("max_opp", "field_avg"):
        bs = BigStackBid(reference=ref, overpay_gain=0.0).max_bid(view, pool.iloc[0], pool, _config())
        assert bs == BalancedValueBid(premium=0.0).max_bid(view, pool.iloc[0], pool, _config())


def test_field_avg_survives_one_hoarder_but_max_opp_does_not() -> None:
    # 4 teams: hero $100, one hoarder $100, two poor $20. The richest opponent ties the hero, so
    # max_opp sees advantage 1 (no overpay); field_avg compares to the LEAGUE AVERAGE per slot
    # (240/12 = $20 vs the hero's 100/3 = $33) -> still big stack -> overpays. Spec R3.
    cfg = LeagueConfig(
        name="t", n_teams=4, budget=100, min_bid=1,
        roster_slots={RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.BENCH: 1},
        ruleset=Ruleset.espn_ppr(),
    )
    pool = _pool()
    baseline = _baseline([True, True, False, False], [60, 40, 0, 0])
    view = AuctionView(
        my_budget=100, my_open_slots=3, my_positions=Counter(), my_roster=pool.iloc[:0],
        drafted=frozenset(), budgets_by_seat=(100, 100, 20, 20), baseline_dollars=baseline,
    )
    bal = BalancedValueBid(premium=0.0).max_bid(view, pool.iloc[0], pool, cfg)
    maxopp = BigStackBid(reference="max_opp", overpay_gain=1.0).max_bid(view, pool.iloc[0], pool, cfg)
    field = BigStackBid(reference="field_avg", overpay_gain=1.0).max_bid(view, pool.iloc[0], pool, cfg)
    assert maxopp == bal  # richest opponent ties the hero -> no overpay
    assert field > bal  # league average is low -> big stack -> overpay


def test_bigstack_rejects_bad_params() -> None:
    with pytest.raises(ValueError, match="reference"):
        BigStackBid(reference="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="overpay_gain"):
        BigStackBid(overpay_gain=-1.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_draft/test_assistant_auction_bid_strategy.py -q -k bigstack -o addopts=""`
Expected: FAIL — `ImportError` / `NameError: name 'BigStackBid' is not defined`. (Also add `BigStackBid` to the file's `from ...bid_strategy import (...)` block; it will fail until Step 3 defines it.)

- [ ] **Step 3: Add the `Literal` import and `BigStackBid`**

In `src/projections/draft/assistant/auction/bid_strategy.py`, change the typing import:

```python
from typing import Literal, Protocol, runtime_checkable
```

Add this class immediately after `BalancedValueBid` (it reuses `AuctionView` and the module-level `_total_open_slots`):

```python
@dataclass(frozen=True)
class BigStackBid:
    """Balanced-breadth hero that OVERPAYS in proportion to its remaining-budget lead over the field.

    When the hero is the "big stack" (more remaining budget than the table), unused budget is pure
    waste — so it bids above fair value and lifts the low pace cap, letting it win/pay-up on
    contested players and deploy the lead (second-price still clears uncontested players at min_bid,
    so cash only moves when an opponent is pushing the price). When NOT the big stack it is exactly
    `BalancedValueBid(premium, pace)`. Two `advantage` signals:

    - "max_opp": my_budget / richest opponent's budget (one hoarding opponent flattens it).
    - "field_avg": my per-slot budget / the league-average per-slot budget (robust to one hoarder).

    See docs/superpowers/specs/2026-07-16-auction-bigstack-overspend-design.md.
    """

    overpay_gain: float = 1.0
    reference: Literal["max_opp", "field_avg"] = "field_avg"
    premium: float = 0.0
    pace: float = 2.0

    def __post_init__(self) -> None:
        if self.reference not in ("max_opp", "field_avg"):
            raise ValueError(f"reference must be 'max_opp' or 'field_avg'; got {self.reference!r}")
        if not (self.overpay_gain >= 0.0 and math.isfinite(self.overpay_gain)):
            raise ValueError(f"overpay_gain must be finite and >= 0; got {self.overpay_gain}")
        if not (self.premium >= 0.0 and math.isfinite(self.premium)):
            raise ValueError(f"premium must be finite and >= 0; got {self.premium}")
        if not (self.pace > 0.0 and math.isfinite(self.pace)):
            raise ValueError(f"pace must be finite and > 0; got {self.pace}")

    def _advantage(self, view: AuctionView, config: LeagueConfig) -> float:
        """>1 iff the hero is the big stack (more remaining budget than the reference field)."""
        if self.reference == "max_opp":
            opp = list(view.budgets_by_seat)
            opp.remove(view.my_budget)  # drop one instance of the hero's own budget
            return view.my_budget / max(max(opp), config.min_bid)
        total_open = _total_open_slots(view, config)
        league_per_slot = sum(view.budgets_by_seat) / max(1, total_open)
        my_per_slot = view.my_budget / max(1, view.my_open_slots)
        return my_per_slot / max(league_per_slot, config.min_bid)

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        fair = float(view.baseline_dollars.loc[player["gsis_id"], "auction_dollars"])
        per_slot = view.my_budget / max(1, view.my_open_slots)
        overpay = self.overpay_gain * max(0.0, self._advantage(view, config) - 1.0)
        target = fair * (1.0 + self.premium + overpay)
        cap = self.pace * per_slot * (1.0 + overpay)
        return round(min(target, cap))
```

Add `BigStackBid` to the test file's import block `from projections.draft.assistant.auction.bid_strategy import (...)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_draft/test_assistant_auction_bid_strategy.py -q -k bigstack -o addopts=""`
Expected: PASS (5 passed).

- [ ] **Step 5: Gates**

Run: `python -m mypy src tests && python -m ruff check src tests && python -m ruff format --check src tests`
Expected: `Success: no issues found`, `All checks passed!`, formatted. (If `ruff format` reflows the new test calls, run `python -m ruff format src tests` first.)

- [ ] **Step 6: Commit**

```bash
git add src/projections/draft/assistant/auction/bid_strategy.py tests/test_draft/test_assistant_auction_bid_strategy.py
git commit -m "feat(auction): BigStackBid overspend hero (stack-advantage overpay, two variants)"
```

---

### Task 2: Engine smoke — legal roster (R4)

**Files:**
- Test: `tests/test_draft/test_assistant_auction_simulation.py` (append)

**Interfaces:**
- Consumes: `BigStackBid` (Task 1), `simulate_auction`, existing sim-test fixtures `_config(n_teams=4)`, `_pool(40)`, `_baseline`.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_bigstack_produces_a_legal_full_roster() -> None:
    # The engine clamps BigStackBid's (possibly huge) desired bids to feasible_max, so the hero
    # always fills a legal, full roster with no duplicate players. Spec R4.
    from projections.draft.assistant.auction.bid_strategy import BigStackBid

    cfg = _config(n_teams=4)
    pool = _pool(40)
    league = simulate_auction(
        BigStackBid(reference="field_avg", overpay_gain=2.0), 1, pool, cfg,
        baseline_dollars=_baseline(pool, cfg), price_jitter=0.1, rng=np.random.default_rng(0),
    )
    assert all(len(r) == cfg.roster_size for r in league.values())
    ids = [g for r in league.values() for g in r]
    assert len(ids) == len(set(ids))
```

- [ ] **Step 2: Run to verify it fails, then passes**

Run: `python -m pytest tests/test_draft/test_assistant_auction_simulation.py -q -k bigstack -o addopts=""`
Expected: initially FAIL only if import wrong; since `BigStackBid` exists after Task 1, this should PASS immediately (it is a smoke, not red-green — the engine invariant already holds). If it fails, the engine is NOT clamping BigStackBid's bid — investigate `simulation._simulate_to_state` clamping before proceeding.

- [ ] **Step 3: Gates + commit**

Run: `python -m pytest tests/test_draft/test_assistant_auction_simulation.py -q && python -m mypy src tests && python -m ruff check src tests`
Expected: all pass / clean.

```bash
git add tests/test_draft/test_assistant_auction_simulation.py
git commit -m "test(auction): BigStackBid yields a legal full roster (engine clamps overpay)"
```

---

### Task 3: The A/B sweep + Run writeup

Race `balanced` p0.0 (control) + `BigStackBid` × {max_opp, field_avg} × `overpay_gain` {0.5, 1.0, 2.0} under ADP nomination, seat-averaged both markets, and record the characterized lift. Scratch runner (committed only if it graduates).

**Files:**
- Create (scratch): `<scratchpad>/bigstack_sweep.py`
- Create (scratch): `<scratchpad>/bigstack_driver.sh`
- Modify: `reports/auction_tournament_validation_2026.md` (add a Run)
- Modify: memory `auction-bid-model-investigation-status.md` + `MEMORY.md`

**Interfaces:**
- Consumes: `run_auction_tournament` (threads `market_adp_jitter`, from PR #99), `auction_seat_sweep.aggregate_seat_sweep` + `_load_chunks`, `BalancedValueBid`, `BigStackBid`, `_load_tournament_inputs`, `_REALISTIC_FIELD`, `DEFAULT_PRICE_JITTER`.

- [ ] **Step 1: Write the scratch runner**

Write `<scratchpad>/bigstack_sweep.py`. It mirrors `scripts/auction_seat_sweep.py`'s chunk format (so `aggregate_seat_sweep` consumes it), but races the dedicated `CONTESTANTS` dict with `market_adp_jitter=12`.

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from projections.draft.assistant.auction.bid_strategy import BalancedValueBid, BigStackBid
from projections.draft.assistant.auction.market import DEFAULT_PRICE_JITTER
from projections.draft.assistant.auction.tournament import run_auction_tournament
from projections.draft.assistant.auction.tournament_cli import (
    _REALISTIC_FIELD,
    _load_tournament_inputs,
)
from projections.draft.auction import has_usable_espn_prices

# Import the aggregation/loader from the committed seat-sweep runner (scripts/ is on sys.path).
from auction_seat_sweep import _load_chunks, aggregate_seat_sweep  # type: ignore[import-not-found]

POOL = Path("data/vorp_2026/half_12team.parquet")
CFG = Path("data/vorp_2026/half_12team.league.json")
SEASON, MARKET_ADP_JITTER = 2026, 12.0

CONTESTANTS = {"balanced": BalancedValueBid(premium=0.0)}
for _ref in ("max_opp", "field_avg"):
    for _g in (0.5, 1.0, 2.0):
        CONTESTANTS[f"bigstack_{_ref}_g{_g}"] = BigStackBid(reference=_ref, overpay_gain=_g)


def _run_chunk(a: argparse.Namespace) -> int:
    pool, config, availability, params = _load_tournament_inputs(
        POOL, CFG, season=SEASON, data_root=Path("data")
    )
    if a.market == "espn" and not has_usable_espn_prices(pool):
        raise SystemExit("espn requested but pool has no usable espn prices")
    result = run_auction_tournament(
        CONTESTANTS, pool, config, my_seat=a.seat, n_seeds=a.seeds,
        price_jitter=DEFAULT_PRICE_JITTER, base_seed=0, n_sims=a.n_sims,
        availability=availability, params=params, nomination_temp=1.0,
        bot_archetypes=_REALISTIC_FIELD, bot_prices=a.market, market_adp_jitter=MARKET_ADP_JITTER,
    )
    payload = {
        "market": a.market, "seat": a.seat, "n_seeds": a.seeds, "n_sims": a.n_sims,
        "reg_win_pct": {n: result.summaries[n]["reg_win_pct"].point for n in result.summaries},
        "all_metrics": {
            n: {m: result.summaries[n][m].point for m in result.summaries[n]}
            for n in result.summaries
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {a.out} (market={a.market}, seat={a.seat})")
    return 0


def _aggregate(a: argparse.Namespace) -> int:
    chunks, skipped = _load_chunks(a.chunk_dir)
    markets, seats, rows, best = aggregate_seat_sweep(chunks)
    print(f"seats: {seats} | skipped: {skipped}")
    print(f"{'contestant':<22}" + "".join(f"{m:>10}" for m in markets) + f"{'worst':>10}")
    ctrl = next((r for r in rows if r.name == "balanced"), None)
    for row in rows:
        cells = "".join(f"{c:>10.3f}" if c is not None else f"{'-':>10}" for c in row.seat_avg)
        print(f"{row.name:<22}{cells}{row.worst:>10.3f}")
    if ctrl is not None:
        print("\ndelta vs balanced (per market):")
        for row in rows:
            if row.name == "balanced":
                continue
            d = [
                (a_ - b_) if (a_ is not None and b_ is not None) else None
                for a_, b_ in zip(row.seat_avg, ctrl.seat_avg)
            ]
            cells = "".join(f"{x:>+10.3f}" if x is not None else f"{'-':>10}" for x in d)
            print(f"  {row.name:<20}{cells}")
    return 0


def _args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seat", type=int, required=True)
    r.add_argument("--market", choices=("espn", "model"), required=True)
    r.add_argument("--seeds", type=int, default=20)
    r.add_argument("--n-sims", type=int, default=300)
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

- [ ] **Step 2: Smoke one tiny chunk**

Run: `python <scratchpad>/bigstack_sweep.py run --seat 1 --market model --seeds 2 --n-sims 20 --out reports/_bigstack/smoke.json` then `python -c "import json;d=json.load(open('reports/_bigstack/smoke.json'));print(sorted(d['reg_win_pct']))"` and `rm reports/_bigstack/smoke.json`
Expected: 7 contestant keys (`balanced` + 6 `bigstack_*`); no traceback.

- [ ] **Step 3: Crash-safe driver + launch**

Write `<scratchpad>/bigstack_driver.sh` (resumable, one bounded process per `(seat, market)`, 12 seats × 2 markets = 24 chunks, `--seeds 20 --n-sims 300`, output `reports/_bigstack/2026/`), modeled on `reports/_seat_sweep_adp/adp_bakeoff_driver.sh`. Launch with `run_in_background: true`. Expected ~40 min (7 contestants), all chunks `rc=0`.

- [ ] **Step 4: Aggregate + interpret**

Run: `python <scratchpad>/bigstack_sweep.py aggregate --chunk-dir reports/_bigstack/2026`
- **Sanity:** `balanced` reproduces its Run-P ADP figure (~espn 0.684 / model 0.593). If not, stop.
- Read the seat-avg table + the per-market delta-vs-balanced for each variant×gain.

- [ ] **Step 5: Write the Run + memory**

Add a Run section to `reports/auction_tournament_validation_2026.md`: the seat-avg table + delta-vs-`balanced`, per the spec's Interpretation standard (delta in both markets, flag whether it clears the ~±0.03 seed-noise band; **no adopt bar** — data-gathering). Note which variant/gain (if any) deploys budget into a real reg-win% lift, whether the effect is ESPN-only or both-market, and the residual unspent-budget if measurable. Update the memory files with the outcome. State plainly: no strategy adopted.

- [ ] **Step 6: Commit the writeup**

```bash
git add reports/auction_tournament_validation_2026.md
git commit -m "docs(auction): Run — big-stack overspend A/B (data, no adoption)"
```

---

## Self-Review

**Spec coverage:**
- `BigStackBid` formula (lift target + cap by `1+overpay`) → Task 1 Step 3.
- Both advantage variants (`max_opp`, `field_avg`) → Task 1 `_advantage`.
- R1 fallback identity → `test_bigstack_falls_back_to_balanced...` + gain=0 test.
- R2 overpay direction → `test_bigstack_overpays_above_balanced...`.
- R3 variant correctness (one hoarder) → `test_field_avg_survives_one_hoarder...`.
- R4 solvency → Task 2 legal-roster smoke.
- R5 bid-fixed A/B → Task 3 `run_auction_tournament(..., market_adp_jitter=12)`, same seeds/sims/field.
- R6 gates → every code task.
- R7 literal/param validation → `test_bigstack_rejects_bad_params`.
- Interpretation (delta vs balanced, noise-flag, no adopt bar) → Task 3 Steps 4–5.
- 7-contestant dedicated dict, gain grid {0.5,1.0,2.0} → Task 3 `CONTESTANTS`.

**Placeholder scan:** none — every code step has complete code; `<scratchpad>` is the session scratch dir (a real path).

**Type consistency:** `BigStackBid(overpay_gain, reference, premium, pace)`, `_advantage`, and the `bigstack_{ref}_g{g}` contestant keys are used identically across Tasks 1 and 3. `reference` literal values `"max_opp"`/`"field_avg"` match between the class, tests, and `CONTESTANTS`.
