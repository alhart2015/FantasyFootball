# Auction realistic market — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-06-18-auction-realistic-market-design.md`

**Goal:** Make mid-round auction prices realistic (no more $1 Marvin Harrisons) via value-weighted-random nomination + a heterogeneous bot field, add a patient hero contestant, and re-run the bake-off.

**Architecture:** Three additive engine/market changes, each defaulting to *current behavior* so existing tests stay green: a `nomination_temp` knob on the nominee selection; bot *archetypes* (Aggressive = today's bot, Patient, Balanced) dispatched per seat; a `PatientValueBid` hero model. The bake-off CLI opts into the realistic market (random nomination + mixed field, 7 contestants).

**Tech Stack:** Python 3, pandas (pyarrow strings), numpy RNG, pytest, mypy strict, ruff.

## Global Constraints

- Backward-compat: every new knob defaults to current behavior. `nomination_temp=0.0` ⇒ today's argmax nomination, consuming **no** nomination RNG (bot-bid stream byte-identical). `bot_archetypes=None` ⇒ all `AggressiveBot`. Existing engine/market/CLI tests pass unchanged.
- `AggressiveBot` and `resolve_bids` are byte-identical to today; the snake field (`backtest/draft_field.py`) and scorer (`project_draft`) are untouched.
- Determinism: nomination draws + bot bids are a pure function of the seeded `rng`; the hero bid is a pure function of state. Same `(seed, temp, mix, strategy)` ⇒ identical rosters.
- `GsisId` canonical; `Position`/`RosterSlot` enums, never raw strings; `_PYARROW_STR` for gsis columns. **No new pandera schema.**
- Bid models keep `max_bid(view, player, pool, config) -> int`; bot archetypes share `max_bid(seat_view, player, baseline_dollars, config, rng, *, price_jitter) -> int`.
- Gates per task: `python -m pytest <touched> -n0 -q`, `python -m mypy <touched>`, `python -m ruff check <touched>`, `python -m ruff format --check <touched>`. (`-n0` = serial; the box's Raptor Lake fault makes xdist flaky.)

---

### Task 1: Value-weighted-random nomination

**Files:**
- Modify: `src/projections/draft/assistant/auction/simulation.py` (add `_sample_nominee`; thread `nomination_temp` through `_simulate_to_state` + `simulate_auction`; replace the nomination block ~lines 118–130)
- Test: `tests/test_draft/test_assistant_auction_simulation.py`

**Interfaces:**
- Consumes: `_simulate_to_state(strategy, my_seat, pool, config, *, baseline_dollars, price_jitter, rng)` and `simulate_auction(...)` (same kw-only tail) as they exist today.
- Produces: `_sample_nominee(candidates: list[str], val_by_id: dict[str, float], temp: float, rng) -> str`; both engine fns gain a kw-only `nomination_temp: float = 0.0`.

- [ ] **Step 1: Write failing tests.**

Add to `tests/test_draft/test_assistant_auction_simulation.py` (imports at top: add `_sample_nominee` and `simulate_auction` is already imported):

```python
from projections.draft.assistant.auction.simulation import _sample_nominee


def test_sample_nominee_temp_zero_is_argmax() -> None:
    # candidates are pre-sorted value-desc; temp=0 must return the first (no RNG draw).
    cands = ["A", "B", "C"]
    val = {"A": 50.0, "B": 20.0, "C": 1.0}
    assert _sample_nominee(cands, val, 0.0, np.random.default_rng(0)) == "A"


def test_sample_nominee_single_candidate() -> None:
    assert _sample_nominee(["X"], {"X": 0.0}, 1.0, np.random.default_rng(0)) == "X"


def test_sample_nominee_temp_one_favors_value_but_samples_tail() -> None:
    cands = ["hi", "lo1", "lo2"]
    val = {"hi": 100.0, "lo1": 1.0, "lo2": 1.0}
    rng = np.random.default_rng(0)
    picks = [_sample_nominee(cands, val, 1.0, rng) for _ in range(500)]
    assert picks.count("hi") > picks.count("lo1") + picks.count("lo2")  # value-weighted
    assert (picks.count("lo1") + picks.count("lo2")) > 0  # but the tail does come up


def test_nomination_temp_zero_is_deterministic() -> None:
    # temp=0 consumes no nomination RNG, so two runs at the same seed are identical. (Backward-compat
    # to pre-change behavior is guarded by the rest of the engine suite, which runs at default temp=0.)
    cfg = _config(n_teams=4, roster_slots={RosterSlot.RB: 2, RosterSlot.WR: 2, RosterSlot.BENCH: 2})
    pool = _pool(40)
    baseline = _baseline(pool, cfg)
    kw = dict(baseline_dollars=baseline, price_jitter=0.15)
    legacy = _simulate_to_state(StaticDollarBid(), 1, pool, cfg, rng=np.random.default_rng(3), **kw)
    temp0 = _simulate_to_state(
        StaticDollarBid(), 1, pool, cfg, rng=np.random.default_rng(3), nomination_temp=0.0, **kw
    )
    assert legacy.rosters == temp0.rosters
```

- [ ] **Step 2: Run to verify they fail.**

Run: `python -m pytest tests/test_draft/test_assistant_auction_simulation.py -k "sample_nominee or temp_zero_is_deterministic" -n0 -q`
Expected: FAIL — `cannot import name '_sample_nominee'` / `nomination_temp` is an unexpected kwarg.

- [ ] **Step 3: Add `_sample_nominee` and thread `nomination_temp`.**

In `simulation.py`, add the helper near the top of the module (after imports):

```python
def _sample_nominee(
    candidates: list[str], val_by_id: dict[str, float], temp: float, rng: np.random.Generator
) -> str:
    """Pick the next nominee. temp<=0 -> the highest-value candidate (candidates are pre-sorted
    value-desc), consuming no RNG. temp>0 -> sample with weight max(value, 0.5)**(1/temp)."""
    if temp <= 0.0:
        return candidates[0]
    weights = np.array([max(val_by_id[str(g)], 0.5) ** (1.0 / temp) for g in candidates], dtype=float)
    return candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]
```

Add `nomination_temp: float = 0.0` to the kw-only params of BOTH `_simulate_to_state` and `simulate_auction` (after `rng`), and pass it through in `simulate_auction`'s call to `_simulate_to_state` (`nomination_temp=nomination_temp`).

In `_simulate_to_state`, after `pos_by_id` is built, add the value map:

```python
    val_by_id = {
        str(g): float(v)
        for g, v in zip(baseline_dollars["gsis_id"], baseline_dollars["auction_dollars"], strict=True)
    }
```

Replace the nomination block (the `nominee_id = next(...)` / `forced = nominee_id is None` / `if forced:` lines) with:

```python
        candidates = [
            g for g in nominate_order if g not in state.drafted and pos_by_id[str(g)] in union
        ]
        forced = not candidates
        if forced:
            nominee_id = next(g for g in nominate_order if g not in state.drafted)
            warnings.warn(
                f"auction: no open seat can roster a remaining position; forcing nominee "
                f"{nominee_id} ({pos_by_id[str(nominee_id)].value}) ungated (pool thin).",
                stacklevel=2,
            )
        else:
            nominee_id = _sample_nominee(candidates, val_by_id, nomination_temp, rng)
        assert nominee_id is not None  # guaranteed: pool is non-empty while any seat has open slots
```

- [ ] **Step 4: Run to verify they pass.**

Run: `python -m pytest tests/test_draft/test_assistant_auction_simulation.py -n0 -q`
Expected: PASS (new + all existing; `temp=0` reproduces legacy).

- [ ] **Step 5: Lint/type, commit.**

Run: `python -m ruff check src/projections/draft/assistant/auction/simulation.py && python -m ruff format --check src/projections/draft/assistant/auction/simulation.py && python -m mypy src/projections/draft/assistant/auction/simulation.py`

```bash
git add src/projections/draft/assistant/auction/simulation.py tests/test_draft/test_assistant_auction_simulation.py
git commit -m "feat(auction): value-weighted-random nomination (nomination_temp; temp=0 = legacy)"
```

---

### Task 2: Bot archetypes in `market.py`

**Files:**
- Modify: `src/projections/draft/assistant/auction/market.py` (add `budget` to `SeatView`; add `BotArchetype`, `AggressiveBot`, `PatientValueBot`, `BalancedBot`, `assign_bot_archetypes`)
- Test: `tests/test_draft/test_assistant_auction_market.py`

**Interfaces:**
- Consumes: existing `bot_max_bid`, `SeatView`, `LeagueConfig`.
- Produces: `SeatView(open_slots, eligible_positions=frozenset(Position), budget=0)`; `BotArchetype` protocol (`max_bid(seat_view, player, baseline_dollars, config, rng, *, price_jitter) -> int`); frozen dataclasses `AggressiveBot()`, `PatientValueBot(understud=0.5, midtier_premium=0.35, stud_frac=0.10, scrub_frac=0.50)`, `BalancedBot(pace=2.0)`; `assign_bot_archetypes(n_bots, mix) -> list[BotArchetype]`.

- [ ] **Step 1: Write failing tests.**

Add to `tests/test_draft/test_assistant_auction_market.py`:

```python
from projections.draft.assistant.auction.market import (
    AggressiveBot,
    BalancedBot,
    PatientValueBot,
    assign_bot_archetypes,
)


def _tiered_baseline() -> pd.DataFrame:
    # 10 in-pool players, values descending: ranks 0=stud(>=0.10*10=1 -> rank 0 only),
    # ranks 5..9 = scrub (>= (1-0.50)*10 = 5), ranks 1..4 = mid.
    ids = [f"00-000000{i}" for i in range(10)]
    return pd.DataFrame(
        {"in_pool": [True] * 10, "auction_dollars": [60, 50, 40, 30, 25, 20, 15, 10, 5, 2]},
        index=pd.Index(ids, name="gsis_id"),
    )


def _p(gid: str, pos: str = "RB") -> pd.Series:
    return pd.Series({"gsis_id": gid, "position": pos, "season_mean_fpts": 150.0})


def test_aggressive_matches_legacy_bot_max_bid() -> None:
    bl, cfg = _tiered_baseline(), _config()
    agg = AggressiveBot().max_bid(
        SeatView(open_slots=3, budget=100), _p("00-0000000"), bl, cfg, np.random.default_rng(0),
        price_jitter=0.0,
    )
    legacy = bot_max_bid(
        SeatView(open_slots=3), _p("00-0000000"), bl, cfg, np.random.default_rng(0), price_jitter=0.0
    )
    assert agg == legacy == 60


def test_patient_underbids_a_stud() -> None:
    bid = PatientValueBot().max_bid(
        SeatView(open_slots=3, budget=100), _p("00-0000000"), _tiered_baseline(), _config(),
        np.random.default_rng(0), price_jitter=0.0,
    )
    assert bid == 30  # value 60 (stud) * understud 0.5 == 30, below market


def test_patient_pays_premium_for_midtier_with_reserve() -> None:
    bid = PatientValueBot().max_bid(
        SeatView(open_slots=3, budget=100), _p("00-0000002"), _tiered_baseline(), _config(),
        np.random.default_rng(0), price_jitter=0.0,
    )
    assert bid == round(40 * 1.35)  # value 40 (mid) * (1+0.35) == 54, above market


def test_patient_midtier_without_reserve_bids_min() -> None:
    bid = PatientValueBot().max_bid(
        SeatView(open_slots=3, budget=3), _p("00-0000002"), _tiered_baseline(), _config(),
        np.random.default_rng(0), price_jitter=0.0,
    )
    assert bid == _config().min_bid  # budget (3) not > min_bid*open_slots (3) -> no reserve


def test_patient_scrub_and_ineligible() -> None:
    cfg = _config()
    assert PatientValueBot().max_bid(
        SeatView(open_slots=3, budget=100), _p("00-0000008"), _tiered_baseline(), cfg,
        np.random.default_rng(0), price_jitter=0.0,
    ) == cfg.min_bid  # value 5 -> scrub -> min_bid
    assert PatientValueBot().max_bid(
        SeatView(open_slots=3, budget=100, eligible_positions=frozenset({Position.WR})),
        _p("00-0000000", "RB"), _tiered_baseline(), cfg, np.random.default_rng(0), price_jitter=0.0,
    ) == 0  # RB not eligible -> abstain


def test_balanced_caps_at_pace_ceiling() -> None:
    bid = BalancedBot(pace=2.0).max_bid(
        SeatView(open_slots=3, budget=20), _p("00-0000000"), _tiered_baseline(), _config(),
        np.random.default_rng(0), price_jitter=0.0,
    )
    assert bid == 13  # min(value 60, 2*(20/3)=13.33) -> 13, paced (won't blow the bank)


def test_assign_bot_archetypes_round_robins() -> None:
    mix = [AggressiveBot(), PatientValueBot(), BalancedBot()]
    out = assign_bot_archetypes(5, mix)
    assert [type(a).__name__ for a in out] == [
        "AggressiveBot", "PatientValueBot", "BalancedBot", "AggressiveBot", "PatientValueBot"
    ]
```

- [ ] **Step 2: Run to verify they fail.**

Run: `python -m pytest tests/test_draft/test_assistant_auction_market.py -k "aggressive_matches or patient or balanced or assign" -n0 -q`
Expected: FAIL — names not importable.

- [ ] **Step 3: Implement in `market.py`.**

Add `budget: int = 0` to `SeatView` (after `eligible_positions`). Then append:

```python
from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class BotArchetype(Protocol):
    def max_bid(
        self, seat_view: SeatView, player: pd.Series, baseline_dollars: pd.DataFrame,
        config: LeagueConfig, rng: np.random.Generator, *, price_jitter: float,
    ) -> int: ...


def _value_tier(value: float, baseline_dollars: pd.DataFrame, stud_frac: float, scrub_frac: float) -> str:
    """'stud' | 'mid' | 'scrub' by rank of `value` among in-pool auction_dollars (desc)."""
    inpool = baseline_dollars.loc[baseline_dollars["in_pool"], "auction_dollars"]
    n = len(inpool)
    rank = int((inpool > value).sum())  # 0-based rank, higher value -> lower rank
    if rank < stud_frac * n:
        return "stud"
    if rank >= (1.0 - scrub_frac) * n:
        return "scrub"
    return "mid"


@dataclass(frozen=True)
class AggressiveBot:
    """Today's bot: value*(1+noise), blows budget early. Delegates to bot_max_bid (byte-identical)."""

    def max_bid(self, seat_view, player, baseline_dollars, config, rng, *, price_jitter) -> int:
        return bot_max_bid(seat_view, player, baseline_dollars, config, rng, price_jitter=price_jitter)


@dataclass(frozen=True)
class PatientValueBot:
    """Underbids studs (reserves budget), pays a premium for mid-tier value when it has reserve."""

    understud: float = 0.5
    midtier_premium: float = 0.35
    stud_frac: float = 0.10
    scrub_frac: float = 0.50

    def max_bid(self, seat_view, player, baseline_dollars, config, rng, *, price_jitter) -> int:
        if seat_view.open_slots <= 0 or Position(player["position"]) not in seat_view.eligible_positions:
            return 0
        value = float(baseline_dollars.loc[player["gsis_id"], "auction_dollars"])
        tier = _value_tier(value, baseline_dollars, self.stud_frac, self.scrub_frac)
        noise = 1.0 + rng.normal(0.0, price_jitter)
        if tier == "stud":
            return round(max(float(config.min_bid), value * self.understud * noise))
        reserve = seat_view.budget - config.min_bid * (seat_view.open_slots - 1)
        if tier == "mid" and reserve > value:  # value-aware reserve (spec §Part 2)
            return round(max(float(config.min_bid), value * (1.0 + self.midtier_premium) * noise))
        return config.min_bid


@dataclass(frozen=True)
class BalancedBot:
    """Aggressive WTP, but paced: never spends more than `pace` x its even per-slot share."""

    pace: float = 2.0

    def max_bid(self, seat_view, player, baseline_dollars, config, rng, *, price_jitter) -> int:
        if seat_view.open_slots <= 0 or Position(player["position"]) not in seat_view.eligible_positions:
            return 0
        value = float(baseline_dollars.loc[player["gsis_id"], "auction_dollars"])
        wtp = value * (1.0 + rng.normal(0.0, price_jitter))
        cap = self.pace * (seat_view.budget / seat_view.open_slots)
        return round(max(float(config.min_bid), min(wtp, cap)))


def assign_bot_archetypes(n_bots: int, mix: Sequence[BotArchetype]) -> list[BotArchetype]:
    """Round-robin `mix` across `n_bots` seats — exact, reproducible composition."""
    return [mix[i % len(mix)] for i in range(n_bots)]
```

(Add `Position` is already imported. Move the `from collections.abc import Sequence` / `from typing import ...` lines to the import block at the top per ruff.)

- [ ] **Step 4: Run to verify they pass.**

Run: `python -m pytest tests/test_draft/test_assistant_auction_market.py -n0 -q`
Expected: PASS (new + existing).

- [ ] **Step 5: Lint/type, commit.**

Run: `python -m ruff check src/projections/draft/assistant/auction/market.py && python -m ruff format --check src/projections/draft/assistant/auction/market.py && python -m mypy src/projections/draft/assistant/auction/market.py`

```bash
git add src/projections/draft/assistant/auction/market.py tests/test_draft/test_assistant_auction_market.py
git commit -m "feat(auction): bot archetypes (Aggressive/Patient/Balanced) + assign_bot_archetypes"
```

---

### Task 3: Engine dispatch of per-seat archetypes

**Files:**
- Modify: `src/projections/draft/assistant/auction/simulation.py` (thread `bot_archetypes`; dispatch per bot seat in the bid loop)
- Test: `tests/test_draft/test_assistant_auction_simulation.py`

**Interfaces:**
- Consumes: `AggressiveBot`, `PatientValueBot`, `BalancedBot`, `assign_bot_archetypes`, `BotArchetype` (Task 2); `SeatView` now has `budget`.
- Produces: `_simulate_to_state` + `simulate_auction` gain kw-only `bot_archetypes: Sequence[BotArchetype] | None = None` (None ⇒ all `AggressiveBot`).

- [ ] **Step 1: Write the failing integration test.**

Add to `tests/test_draft/test_assistant_auction_simulation.py` (add to the market import: `AggressiveBot, BalancedBot, PatientValueBot`; and `from projections.draft.assistant.auction.market import _value_tier`):

```python
def test_mixed_field_bids_midtier_off_the_dollar_floor() -> None:
    # THE CORE FIX: same seed, legacy (all-aggressive, temp=0) vs realistic (mixed field, temp=1).
    # Mid-tier players should clear ABOVE min_bid far more often under the mixed field.
    cfg = _config(n_teams=8, roster_slots={RosterSlot.RB: 2, RosterSlot.WR: 2, RosterSlot.BENCH: 3})
    pool = _pool(80)
    baseline = _baseline(pool, cfg)
    bd_idx = baseline.set_index("gsis_id")

    def midtier_above_floor(state) -> int:
        n = 0
        for seat in range(cfg.n_teams):
            for gsis, _pos, price in state.rosters[seat]:
                val = float(bd_idx.loc[gsis, "auction_dollars"])
                if _value_tier(val, baseline, 0.10, 0.50) == "mid" and price > cfg.min_bid:
                    n += 1
        return n

    legacy = _simulate_to_state(
        StaticDollarBid(), 1, pool, cfg, baseline_dollars=baseline, price_jitter=0.15,
        rng=np.random.default_rng(11),
    )
    mixed = _simulate_to_state(
        StaticDollarBid(), 1, pool, cfg, baseline_dollars=baseline, price_jitter=0.15,
        rng=np.random.default_rng(11), nomination_temp=1.0,
        bot_archetypes=[AggressiveBot(), PatientValueBot(), BalancedBot()],
    )
    assert midtier_above_floor(mixed) > midtier_above_floor(legacy)
    assert midtier_above_floor(mixed) >= 1


def test_mixed_field_is_deterministic() -> None:
    cfg = _config(n_teams=8, roster_slots={RosterSlot.RB: 2, RosterSlot.WR: 2, RosterSlot.BENCH: 3})
    pool = _pool(80)
    bl = _baseline(pool, cfg)
    kw = dict(baseline_dollars=bl, price_jitter=0.15, nomination_temp=1.0,
              bot_archetypes=[AggressiveBot(), PatientValueBot(), BalancedBot()])
    a = _simulate_to_state(StaticDollarBid(), 1, pool, cfg, rng=np.random.default_rng(5), **kw)
    b = _simulate_to_state(StaticDollarBid(), 1, pool, cfg, rng=np.random.default_rng(5), **kw)
    assert a.rosters == b.rosters
```

- [ ] **Step 2: Run to verify they fail.**

Run: `python -m pytest tests/test_draft/test_assistant_auction_simulation.py -k "mixed_field" -n0 -q`
Expected: FAIL — `bot_archetypes` is an unexpected kwarg.

- [ ] **Step 3: Thread `bot_archetypes` and dispatch per seat.**

In `simulation.py`, add `bot_archetypes: Sequence[BotArchetype] | None = None` to the kw-only params of `_simulate_to_state` and `simulate_auction` (pass through in `simulate_auction`). Add the imports: `from projections.draft.assistant.auction.market import AggressiveBot, BotArchetype, assign_bot_archetypes` (and `from collections.abc import Sequence` if not present).

In `_simulate_to_state`, after `hero0` is set, build the per-seat archetype map:

```python
    bot_seats = [s for s in range(n) if s != hero0]
    if bot_archetypes is None:
        seat_arch: dict[int, BotArchetype] = {s: AggressiveBot() for s in bot_seats}
    else:
        _assigned = assign_bot_archetypes(len(bot_seats), bot_archetypes)
        seat_arch = {s: _assigned[i] for i, s in enumerate(bot_seats)}
```

In the bid loop, replace the bot branch's `bot_max_bid(SeatView(open_slots=..., eligible_positions=elig), player, bd, config, rng, price_jitter=price_jitter)` call with the dispatched archetype (note the added `budget`):

```python
            else:
                desired = seat_arch[seat].max_bid(
                    SeatView(
                        open_slots=_open_slots(state, seat, rs),
                        eligible_positions=elig,
                        budget=state.budgets[seat],
                    ),
                    player,
                    bd,
                    config,
                    rng,
                    price_jitter=price_jitter,
                )
                if desired <= 0:  # abstain -> dropped before the clamp
                    continue
                bids[seat] = max(min_bid, min(int(desired), fmax))
```

(`AggressiveBot().max_bid` delegates to `bot_max_bid`, which ignores `budget` and consumes the same `rng.normal` — so the `bot_archetypes=None` default is byte-identical to today, including the RNG stream.)

- [ ] **Step 4: Run to verify they pass.**

Run: `python -m pytest tests/test_draft/test_assistant_auction_simulation.py -n0 -q`
Expected: PASS — the mixed field lifts mid-tier off the floor; determinism holds; all existing engine tests still green (None default unchanged).

- [ ] **Step 5: Lint/type, commit.**

Run: `python -m ruff check src/projections/draft/assistant/auction/simulation.py && python -m ruff format --check src/projections/draft/assistant/auction/simulation.py && python -m mypy src/projections/draft/assistant/auction/simulation.py`

```bash
git add src/projections/draft/assistant/auction/simulation.py tests/test_draft/test_assistant_auction_simulation.py
git commit -m "feat(auction): per-seat bot-archetype dispatch (default all-aggressive; mixed field lifts mid-tier)"
```

---

### Task 4: `PatientValueBid` hero contestant

**Files:**
- Modify: `src/projections/draft/assistant/auction/bid_strategy.py` (add `PatientValueBid`)
- Test: `tests/test_draft/test_assistant_auction_bid_strategy.py`

**Interfaces:**
- Consumes: `AuctionView`, `_vorp_threshold` (existing); pool with `vorp`; `view.baseline_dollars` (auction_dollars); fixtures `_vpool`/`_vbaseline`/`_aview`/`_vconfig` (already in the test file).
- Produces: `PatientValueBid(midtier_premium=0.35, stud_frac=0.10, scrub_frac=0.50)` satisfying `AuctionBidStrategy`.

- [ ] **Step 1: Write failing tests.**

Add to `tests/test_draft/test_assistant_auction_bid_strategy.py` (add `PatientValueBid` to the `bid_strategy` import):

```python
def test_patient_hero_holds_on_a_stud() -> None:
    # _vpool vorps: 120,110,90,20,10,5 (6 players). stud_frac 0.10 -> round(0.6)=1 -> top-1 cutoff
    # = _vorp_threshold(pool,1)=120; vorp 120 is a stud -> min_bid.
    pool = _vpool()
    view = _aview(pool, budget=100, open_slots=8)
    assert PatientValueBid().max_bid(view, pool.iloc[0], pool, _vconfig()) == _vconfig().min_bid


def test_patient_hero_pays_premium_for_midtier_with_reserve() -> None:
    # scrub_frac 0.50 -> (1-0.50)*6=3 -> scrub cutoff = _vorp_threshold(pool,3)=90; stud cutoff 120.
    # vorp 110 (player 2) is in (90,120) -> mid. auction_dollars for it = 28 -> round(28*1.35)=38.
    pool = _vpool()
    view = _aview(pool, budget=100, open_slots=8)
    assert PatientValueBid().max_bid(view, pool.iloc[1], pool, _vconfig()) == round(28 * 1.35)


def test_patient_hero_midtier_without_reserve_bids_min() -> None:
    pool = _vpool()
    view = _aview(pool, budget=8, open_slots=8)  # reserve = 8 - 1*7 = 1 < the premium bid
    assert PatientValueBid().max_bid(view, pool.iloc[1], pool, _vconfig()) == _vconfig().min_bid


def test_patient_hero_scrub_bids_min() -> None:
    pool = _vpool()
    view = _aview(pool, budget=100, open_slots=8)
    # vorp 5 (player 6) is below the scrub cutoff (90) -> scrub -> min_bid.
    assert PatientValueBid().max_bid(view, pool.iloc[5], pool, _vconfig()) == _vconfig().min_bid
```

- [ ] **Step 2: Run to verify they fail.**

Run: `python -m pytest tests/test_draft/test_assistant_auction_bid_strategy.py -k "patient_hero" -n0 -q`
Expected: FAIL — `cannot import name 'PatientValueBid'`.

- [ ] **Step 3: Implement `PatientValueBid` in `bid_strategy.py`.**

Append after `VorpShareBid`:

```python
@dataclass(frozen=True)
class PatientValueBid:
    """Holds budget through the stud frenzy; pays up for mid-tier value when reserve remains."""

    midtier_premium: float = 0.35
    stud_frac: float = 0.10
    scrub_frac: float = 0.50

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        min_bid = config.min_bid
        n = len(pool)
        stud_cut = _vorp_threshold(pool, round(self.stud_frac * n))
        scrub_cut = _vorp_threshold(pool, round((1.0 - self.scrub_frac) * n))
        v = float(player["vorp"])
        if v >= stud_cut or v < scrub_cut:  # stud (let it go) or scrub
            return min_bid
        value = int(view.baseline_dollars.loc[player["gsis_id"], "auction_dollars"])
        bid = round(value * (1.0 + self.midtier_premium))
        reserve = view.my_budget - min_bid * (view.my_open_slots - 1)
        return bid if reserve >= bid else min_bid
```

- [ ] **Step 4: Run to verify they pass.**

Run: `python -m pytest tests/test_draft/test_assistant_auction_bid_strategy.py -n0 -q`
Expected: PASS.

- [ ] **Step 5: Lint/type, commit.**

Run: `python -m ruff check src/projections/draft/assistant/auction/bid_strategy.py && python -m ruff format --check src/projections/draft/assistant/auction/bid_strategy.py && python -m mypy src/projections/draft/assistant/auction/bid_strategy.py`

```bash
git add src/projections/draft/assistant/auction/bid_strategy.py tests/test_draft/test_assistant_auction_bid_strategy.py
git commit -m "feat(auction): PatientValueBid hero contestant (holds studs, pays up mid-tier)"
```

---

### Task 5: CLI wiring — seven models + realistic-market defaults

**Files:**
- Modify: `src/projections/draft/assistant/auction/tournament.py` (thread `nomination_temp` + `bot_archetypes` through `run_auction_tournament`)
- Modify: `src/projections/draft/assistant/auction/tournament_cli.py` (add `patient` model; `--nomination-temp`; default realistic market)
- Test: `tests/test_draft/test_assistant_auction_tournament_cli.py`

**Interfaces:**
- Consumes: `PatientValueBid` (Task 4); `AggressiveBot`/`PatientValueBot`/`BalancedBot` (Task 2); `simulate_auction(..., nomination_temp=, bot_archetypes=)` (Tasks 1+3).
- Produces: `run_auction_tournament(..., nomination_temp=0.0, bot_archetypes=None)`; `_MODELS` has 7 entries; CLI `--nomination-temp` (default 1.0) + a default mixed field.

- [ ] **Step 1: Write failing tests.**

Add to / update `tests/test_draft/test_assistant_auction_tournament_cli.py` (rename the existing six-contestant test):

```python
from projections.draft.assistant.auction.bid_strategy import AuctionBidStrategy
from projections.draft.assistant.auction.tournament_cli import _MODELS, _parse_args


def test_default_models_are_the_seven_contestants() -> None:
    assert set(_MODELS) == {
        "static", "inflation", "marginal", "anchors", "overbid", "vorpshare", "patient"
    }


def test_every_default_model_satisfies_the_protocol() -> None:
    assert all(isinstance(m, AuctionBidStrategy) for m in _MODELS.values())


def test_nomination_temp_defaults_to_one() -> None:
    args = _parse_args(
        ["--vorp-table", "x", "--league-config", "y", "--my-seat", "1", "--season", "2026", "compare"]
    )
    assert args.nomination_temp == 1.0
```

(Delete the old `test_default_models_are_the_six_contestants`.)

- [ ] **Step 2: Run to verify they fail.**

Run: `python -m pytest tests/test_draft/test_assistant_auction_tournament_cli.py -k "seven or nomination_temp" -n0 -q`
Expected: FAIL — `patient` missing from `_MODELS`; `args.nomination_temp` AttributeError.

- [ ] **Step 3a: Thread the two params through `run_auction_tournament` (`tournament.py`).**

Read `run_auction_tournament` (around line 68) and its `simulate_auction(...)` call (around line 93). Add kw-only params `nomination_temp: float = 0.0` and `bot_archetypes: Sequence[BotArchetype] | None = None` to the signature (import `Sequence` from `collections.abc` and `BotArchetype` from `...auction.market`), and pass them into the `simulate_auction(...)` call:

```python
            league = simulate_auction(
                strat,
                my_seat,
                pool,
                config,
                baseline_dollars=baseline_dollars,
                price_jitter=price_jitter,
                rng=np.random.default_rng(base_seed + s),
                nomination_temp=nomination_temp,
                bot_archetypes=bot_archetypes,
            )
```

- [ ] **Step 3b: Add the model + flag + realistic default (`tournament_cli.py`).**

Extend the `bid_strategy` import with `PatientValueBid`; add to `_MODELS`:

```python
    "patient": PatientValueBid(),
```

Import the archetypes and define the default field:

```python
from projections.draft.assistant.auction.market import (
    AggressiveBot,
    BalancedBot,
    DEFAULT_PRICE_JITTER,
    PatientValueBot,
)

_REALISTIC_FIELD = [AggressiveBot(), PatientValueBot(), BalancedBot()]
```

Add the CLI flag in `_parse_args` (next to `--price-jitter`):

```python
    p.add_argument(
        "--nomination-temp", type=float, default=1.0,
        help="Nomination randomness (0=value-first; 1=value-weighted random).",
    )
```

In `run`, pass the realistic market into the tournament call:

```python
    result = run_auction_tournament(
        _MODELS,
        pool,
        config,
        my_seat=args.my_seat,
        n_seeds=args.seeds,
        price_jitter=args.price_jitter,
        base_seed=args.seed,
        n_sims=args.n_sims,
        availability=availability,
        params=params,
        nomination_temp=args.nomination_temp,
        bot_archetypes=_REALISTIC_FIELD,
    )
```

Update the module docstring "races the six bid models" → "races the seven bid models against a mixed bot field with randomized nomination."

- [ ] **Step 4: Run to verify they pass.**

Run: `python -m pytest tests/test_draft/test_assistant_auction_tournament_cli.py tests/test_draft/test_assistant_auction_tournament.py -n0 -q`
Expected: PASS.

- [ ] **Step 5: Lint/type, commit.**

Run: `python -m ruff check src/projections/draft/assistant/auction/ && python -m ruff format --check src/projections/draft/assistant/auction/ && python -m mypy src/projections/draft/assistant/auction/`

```bash
git add src/projections/draft/assistant/auction/tournament.py src/projections/draft/assistant/auction/tournament_cli.py tests/test_draft/test_assistant_auction_tournament_cli.py
git commit -m "feat(auction): CLI races seven models vs realistic market (mixed field + random nomination)"
```

---

### Task 6: Run E — realistic-market bake-off + tracking doc

**Files:**
- Modify: `reports/auction_tournament_validation_2026.md`

Data/verification task — no TDD. **Run chunked / small** (Raptor Lake fault segfaults large 1-process runs — memory `h2h-backtest-native-crash`).

- [ ] **Step 1: Full gates before the run.**

Run: `python -m pytest tests/test_draft -n0 -q && python -m mypy src tests && python -m ruff check src tests && python -m ruff format --check src tests`
Expected: green except the known pre-existing `test_backtest_smoke_one_cell` (TODO #40) — note, don't fix.

- [ ] **Step 2: Run the realistic-market bake-off (half/16, seat 1).**

Run (start small to confirm it completes; the realistic market is the default now):
```bash
python scripts/auction_tournament.py \
  --vorp-table data/vorp_2026/half_16team.parquet \
  --league-config configs/league_espn_half_16team.json \
  --my-seat 1 --season 2026 --seeds 60 --n-sims 200 --seed 0 \
  compare > "$LOCALAPPDATA/Temp/auction_runE.txt" 2>&1
```
Read the file; if it segfaults, drop `--seeds`/`--n-sims` and re-run. Record the exact flags used.

- [ ] **Step 3: Record Run E in the tracking doc.**

Append a "Run E — realistic market" section to `reports/auction_tournament_validation_2026.md`: the seven per-model rows (exp pts + CI, win/playoff/bye/champ%), paired diffs, exact flags, and the caveat that the market changed (random nomination + mixed field), so Run E is **not** level-comparable to Runs A–D — and note whether the ranking compressed or re-ordered vs Run C (does `overbid`/`anchors` still lead once mid-tier studs cost real money?). State plainly what it favored **in isolation**; **no winner** (September).

- [ ] **Step 4: Commit.**

```bash
git add reports/auction_tournament_validation_2026.md
git commit -m "data(auction): Run E — seven-model bake-off vs realistic market (no winner)"
```

---

## Self-Review

**Spec coverage:**
- R1 (`nomination_temp` threaded; temp=0 = legacy) → Task 1. ✅
- R2 (temp>0 weighted sample, EPS floor, seeded rng) → Task 1 `_sample_nominee`. ✅
- R3 (archetypes + `SeatView.budget` + abstain gate + tiers) → Task 2. ✅
- R4 (`assign_bot_archetypes` deterministic; per-seat dispatch; None=all-aggressive) → Tasks 2+3. ✅
- R5 (`PatientValueBid`) → Task 4. ✅
- R6 (CLI 7 models + realistic default + `--nomination-temp`) → Task 5. ✅
- R7 (determinism) → Task 1 `temp_zero_matches_legacy`, Task 3 `mixed_field_is_deterministic`. ✅
- R8 (backward-compat; AggressiveBot byte-identical; snake untouched) → Tasks 2+3 (None default, delegate to `bot_max_bid`); no snake edits. ✅
- R9 (Run E) → Task 6. ✅
- Core fix (mid-tier off the $1 floor) → Task 3 `test_mixed_field_bids_midtier_off_the_dollar_floor`. ✅
- CLI test rename (six→seven) → Task 5. ✅

**Placeholder scan:** none — every code step has complete code; commands have expected outcomes.

**Type consistency:** `nomination_temp: float`, `bot_archetypes: Sequence[BotArchetype] | None`, `_sample_nominee(candidates, val_by_id, temp, rng) -> str`, `SeatView(open_slots, eligible_positions, budget=0)`, archetype `max_bid(seat_view, player, baseline_dollars, config, rng, *, price_jitter) -> int`, `assign_bot_archetypes(n_bots, mix)`, `PatientValueBid(midtier_premium, stud_frac, scrub_frac)` — used identically across Tasks 1–5 and the tests. `_value_tier` is shared between Task 2 (impl) and Task 3 (test import).
