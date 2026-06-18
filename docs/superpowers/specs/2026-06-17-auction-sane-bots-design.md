# Auction Sane Bots — league-driven positional discipline (shared with the snake field)

**Status:** Design (pre-plan), 2026-06-17. Approved via brainstorming.
**Branch:** `feat/auction-sane-bots`, stacked on `feat/auction-tournament` (the auction harness it refines; PR #76).
**Source:** brainstorming session 2026-06-17 (this conversation).

## 1. Problem

The auction harness's noisy-WTP bots (`auction/market.py::bot_max_bid`) bid on **every** nominated player with no positional awareness. Two consequences:

1. **Degenerate bot rosters are possible** — a bot can win a 3rd/4th QB while leaving an RB starter unfillable. The league sim (`project_draft`) then scores a nonsense opponent, and the field as a whole is unrealistic.
2. **The hero is buried.** Run A (half-PPR/16, seat 1, 150 seeds) showed the hero *below* the uniform baseline on every metric (playoff 0.15 vs 0.375 expected). With all 15 bots bidding on every player, the order-statistic of their noisy WTPs clears studs ~20–26% above `auction_dollars`, so a value-bidding hero is outbid on contested players (recorded in `reports/auction_tournament_validation_2026.md`, Run A).

Critically, this is **inconsistent with the snake side.** The snake *hero-vs-bots / mixed-field* path (`backtest/draft_field.py`) already enforces positional discipline: every bot pick is gated by `_bot_eligible(counts, picks_left)` against hand-tuned `_MINP`/`_MAXP`, reserving final picks for unmet minimums and capping each position. The auction bots should be just as sane.

**Hypothesis to test:** positional discipline both (a) guarantees realistic, startable bot rosters and (b) **thins the per-player bidder pool** — a bot that has filled a position stops bidding it — which should shrink the order-statistic and pull the hero back toward baseline. We measure this by re-running the bake-off (Phase 4); we do **not** pre-emptively change the bidding/clearing model.

## 2. Goals

- **Promote the bot-eligibility selection algorithm** out of `backtest/draft_field.py` into a shared, **`Position`-enum-keyed** helper in `src/projections/draft/roster_eligibility.py`, parameterized by `minimums`/`maximums` maps, used by **both** the snake field and the auction bots.
- **League-driven bounds** — a `bot_position_bounds(roster_slots)` helper derives per-position min/max from the league's `roster_slots` (no hardcoded roster shape), honoring the auction harness's league-driven hard constraint.
- **Gate auction bot bids** — a bot abstains (bids 0) on a nominated player whose position it cannot/should-not roster (at max, or not in its reserved deficit set late), on top of the existing `[min_bid, feasible_max]` budget clamp. Bots end with a **full *and* startable** roster.
- **Zero drift on the snake side** — `backtest/draft_field.py` keeps its exact `_MINP`/`_MAXP` values (passed into the shared helper), so its validated backtests are byte-identical; proven by re-running them.
- **Measure** — re-run the half-PPR/16 bake-off and compare the hero's metrics to Run A.

## 3. Non-goals

- **No change to the WTP noise model or the second-price clearing rule.** Positional discipline is the hypothesized fix for the hero handicap; we measure before touching the bidding/clearing math.
- **No real published auction values.** Anchoring bots on ESPN/Sleeper/FantasyPros consensus auction dollars is a separate, future slice; this slice keeps the existing `generate_auction_values` (SOS) anchor.
- **The hero is NOT gated.** The hero bids however its `AuctionBidStrategy` says — the whole point is to test the bid model; a naive hero that over-rosters a position is a *measured outcome*, not a bug. (It still reserves budget via `feasible_max`.)
- **No strategic nomination.** The nominator stays the shared default (§5.4). The one refinement here is the minimal "nominate a player the room can actually roster" needed to guarantee a bidder once bots gate — not dump-the-budget / price-enforcement strategy.
- **No new pandera schema.** Pure logic + plumbing.

## 4. Chosen approach / architecture

**Paths (shorthand used below → real location):** `roster_eligibility.py` = `src/projections/draft/roster_eligibility.py`; `draft_field.py` = `src/projections/draft/backtest/draft_field.py`; `auction/market.py`, `auction/simulation.py` = `src/projections/draft/assistant/auction/{market,simulation}.py`. There is no top-level `auction/` package.

### 4.1 Shared selection algorithm (`roster_eligibility.py`)

```python
def bot_eligible(
    counts: Mapping[Position, int],
    picks_left: int,
    *,
    minimums: Mapping[Position, int],
    maximums: Mapping[Position, int],
) -> frozenset[Position]:
    """Positions a roster-disciplined bot may take now (the snake draft_field rule, generalized)."""
    deficit = {p: max(0, minimums.get(p, 0) - counts.get(p, 0)) for p in minimums}
    if picks_left <= sum(deficit.values()):
        return frozenset(p for p, d in deficit.items() if d > 0)   # reserve final picks for unmet minimums
    return frozenset(p for p in maximums if counts.get(p, 0) < maximums[p])  # else any position under its cap
```

This is `_bot_eligible`'s exact logic, `Position`-keyed and parameterized. `backtest/draft_field.py` converts its `_MINP`/`_MAXP` string dicts to `dict[Position, int]` (same values) and calls this — **values unchanged ⇒ byte-identical field** (Phase 2 proves it).

**Iteration domain (load-bearing).** The eligible set is drawn **strictly from the `minimums`/`maximums` keysets** — both branches iterate `minimums`/`maximums`, never `counts` or `Position`. So a position present in `counts` but absent from the bound maps (e.g. K/DST a snake bot drafted late, which aren't in `_MINP`/`_MAXP`) is **never returned**. This is exactly what preserves the snake field's behavior; an implementation that iterated `counts` or `Position` in the cap branch would make K/DST appear under-max and silently drift. (Phase 2's equivalence test pins this with a K/DST-holding bot.)

### 4.2 League-driven bounds (`roster_eligibility.py`)

```python
def bot_position_bounds(
    roster_slots: Mapping[RosterSlot, int],
) -> tuple[dict[Position, int], dict[Position, int]]:
    """Derive per-position (minimums, maximums) for a roster-disciplined bot from the league shape."""
```

- **Minimums** — strict starting slots per position, with flex slots anchored to a canonical filler:
  - `min[pos] = roster_slots.get(RosterSlot(pos.value), 0)` for each `pos` whose `RosterSlot` is a position slot (`POSITION_SLOTS`);
  - add `roster_slots.get(RosterSlot.FLEX, 0)` to **RB** (the canonical flex), and `roster_slots.get(RosterSlot.SUPER_FLEX, 0)` to **QB** (the canonical super-flex);
  - **the anchor add is unconditional and intentional:** RB's (or QB's) minimum may be nonzero *purely* from flex anchoring even in a league with **zero strict RB (or QB) starts** — RB/QB are flex/super-flex-eligible, so requiring depth there to cover the flex is the desired bound. A planner should not special-case "no strict starts at the anchor position."
  - positions with no starting slot get min 0 (and are not in the pool).
  - `Σ min == number of starting slots` (each strict + flex slot counted once).
- **Maximums** — min plus the bench, distributed proportionally and rounded **up** so the caps always permit a full roster:
  - `max[pos] = min[pos] + ceil(bench_slots * min[pos] / Σmin)`, where `bench_slots = roster_slots.get(RosterSlot.BENCH, 0)`.
  - Because `ceil` rounds each share up, `Σ extra ≥ bench_slots`, so `Σ max ≥ Σmin + bench_slots == roster_size` — **the caps always allow a full, legal roster, with slack** (so bots retain variety rather than being forced into one fixed distribution).
  - A position with `min == 0` gets `max == 0` (never biddable).

**Worked example — skill roster `{QB:1, RB:2, WR:3, TE:1, FLEX:1, BENCH:9}`** (`roster_size = 17`, `Σmin = 8`):
- min: QB 1, RB `2+1(FLEX)=3`, WR 3, TE 1 → `{QB:1, RB:3, WR:3, TE:1}` (matches the snake `_MINP`).
- max: QB `1+ceil(9·1/8)=1+2=3`, RB `3+ceil(9·3/8)=3+4=7`, WR `3+4=7`, TE `1+2=3` → `{QB:3, RB:7, WR:7, TE:3}` (snake-like; `Σmax=20 ≥ 17`).

**SUPER_FLEX example** `{QB:1, RB:2, WR:3, TE:1, SUPER_FLEX:1, BENCH:9}`: the SUPER_FLEX anchors to QB → `min QB = 1+1 = 2`.

### 4.3 Snake field (`backtest/draft_field.py`) — share, don't drift

Replace the local `_bot_eligible` body with a call to the shared `bot_eligible`, and convert the module's `_MINP`/`_MAXP` to `dict[Position, int]` with the **same numeric values** (`{Position.QB:1, Position.RB:3, ...}` / `{...:3,6,6,3}`). `counts` is converted to `Position`-keyed at the call site. No value changes ⇒ the bot field is byte-identical (Phase 2 regression).

The snake field continues to use its own hand-tuned maps (it does **not** adopt `bot_position_bounds`) — preserving its validated behavior. `bot_position_bounds` is the auction side's derivation.

### 4.4 Auction engine (`auction/simulation.py`, `auction/market.py`)

- At entry, `_simulate_to_state` computes `minimums, maximums = bot_position_bounds(config.roster_slots)` **once**.
- Per nomination, for each open bot seat it computes
  `eligible = bot_eligible(counts, open_slots, minimums=minimums, maximums=maximums)`
  where `counts` is the `Counter[Position]` of that seat's roster and `open_slots` is its `picks_left`.
- **`SeatView` gains `eligible_positions: frozenset[Position]`.** `bot_max_bid` returns `0` (abstain) when `Position(player["position"]) not in seat_view.eligible_positions`; otherwise it bids exactly as today (`value × (1+noise)`).
- **An abstention (`0`) drops the seat from the bid set entirely — it is *not* passed through the `[min_bid, feasible_max]` clamp.** The clamp (`max(min_bid, min(desired, feasible_max))`) would otherwise floor the `0` up to `min_bid` and defeat the gate. So the engine collects bids only from non-abstaining eligible seats, and applies the clamp only to those.
- The **hero** seat is unchanged — no eligibility gate; its `AuctionBidStrategy` bids freely (and its bid *is* clamped as before).

### 4.5 Nomination + the forced-pick path (completion guarantee)

With bots abstaining, naive "nominate the highest baseline" could nominate a player **no open seat will bid on**. The fix is a single **forced-pick** path that relaxes nomination *and* bidding **in lockstep**. Each round the engine works from the **union of the open seats' eligible positions that still have an undrafted player** (the hero, being ungated, contributes *all* positions while it has open slots):

- **Normal round** — nominate the highest-`auction_dollars` undrafted player whose position is in that union. By construction ≥1 open seat is eligible for it, so it draws ≥1 bid. (While the hero has open slots the union is all positions, so this reduces to "highest baseline undrafted.") The eligibility gate (§4.4) is honored.
- **Forced pick** — if that union is empty (every open seat's eligible positions are exhausted from the pool — a thin-pool degeneracy), the engine sets a `forced` flag for this nomination, nominates the plain highest-`auction_dollars` undrafted player, and **the bid step bypasses the eligibility gate for every open seat** so they all bid regardless of position. A `warnings.warn(...)` names the thin position (mirroring `draft_field.py`). Because nomination and bidding relax together, a forced nominee still draws ≥1 bid.

This is still **not** strategic nomination — it only ensures the round resolves. The hero is never gated either way.

**Why completion is guaranteed (both gates move together):** any seat with `open_slots > 0` has `Σ counts < roster_size ≤ Σ max`, so some position is under its max ⇒ its `bot_eligible` set is non-empty (in forced-deficit mode the deficit set is non-empty). On a **normal** round the union of those positions that *also have an undrafted player* is non-empty ⇒ ≥1 bidder. When that union is empty, the **forced** path un-gates every open seat ⇒ ≥1 bidder. So `resolve_bids` is never handed an empty bid set on either path, and every roster spot fills.

## 5. Edge cases / failure modes

- **Zero-bid nominee** — impossible: a normal round nominates only a position ≥1 open seat is eligible for, and the forced-pick path (§4.5) un-gates **all** open seats in lockstep with the relaxed nomination. A defensive assertion guards `resolve_bids` against an empty bid set — unreachable on either path, so it only catches a future logic bug.
- **Pool thin at a required position** — the forced-pick path (§4.5) fires: nominate ungated + every open seat bids + `warn`. The bot may miss a positional minimum; the auction still completes.
- **Position absent from `roster_slots`** (e.g. K/DST in a skill league) — min/max 0 ⇒ never eligible (and absent from the pool anyway).
- **`Σmin == 0`** — impossible: `LeagueConfig` guarantees `roster_size ≥ 1` (≥1 non-IR slot); a config with bench-only/no-starting-slots is out of scope (and would make `project_draft` meaningless).
- **Determinism** — gating is a deterministic function of `counts`/`picks_left`/bounds; same seed ⇒ same league (the existing determinism test still holds).
- **Multiple FLEX / SUPER_FLEX** — all FLEX count anchors to RB, all SUPER_FLEX to QB (a documented convention; adjustable later).
- **Hero unbalanced roster** — allowed by design (§3); the league sim scores it (a starting slot no player fills scores 0).

## 6. Testing (TDD)

- **`bot_eligible`** — deficit reservation (`picks_left ≤ Σdeficit` ⇒ only deficit positions), cap branch (`picks_left > Σdeficit` ⇒ positions under max), the boundary at `picks_left == Σdeficit`, and "a position at its max is excluded."
- **`bot_position_bounds`** — skill roster ⇒ `min {QB:1,RB:3,WR:3,TE:1}`, `max {QB:3,RB:7,WR:7,TE:3}`; a SUPER_FLEX roster ⇒ QB min bumped; the invariant `Σmax ≥ roster_size` over a couple of shapes; a position absent from `roster_slots` ⇒ min/max 0.
- **Snake no-drift** — re-run `backtest/draft_field.py`'s tests and the H2H backtest tests; add an equivalence assertion that `draft_mixed_field` produces the **identical** bot field for a fixed seed before/after the refactor. Include a case where a bot holds **K/DST** (positions absent from the bound maps) and assert the eligible set is identical old-vs-new — pinning the minimums/maximums-keyset iteration domain (§4.1).
- **Auction bots** — a bot at its QB max abstains on a nominated QB (`bot_max_bid → 0`, and that `0` is dropped, not clamped to `min_bid`); a bot with `open_slots == unmet-minimum-count` only bids deficit positions; a **full auction yields a startable roster for every bot** (each bot can fill all its starting slots); the **hero is not gated** (a static hero can over-roster a position).
- **Nomination / completion** — normal rounds nominate a position some open seat can roster; the **forced-pick path**: a fixture where the hero is full and a bot's only eligible position has no undrafted players ⇒ `forced` fires, every open seat bids the ungated nominee, a warning is emitted, and the auction completes with no empty bid set; the auction still fills every seat (existing conservation/solvency tests remain green).
- **Bake-off re-run** (Phase 4, verify-level) — half-PPR/16, seat 1, same seeds as Run A; record Run B and compare the hero's metrics to Run A (does the hero return toward/above baseline?).

All project gates: `pytest -v`, `mypy src tests` (strict), `ruff check src tests`, `ruff format --check src tests`. No schema/store path is touched.

## 7. Phasing

- **Phase 1 — shared core:** add `bot_eligible` + `bot_position_bounds` to `roster_eligibility.py` + their tests.
- **Phase 2 — snake adoption (no drift):** `backtest/draft_field.py` calls the shared `bot_eligible` with its existing values (`Position`-keyed); prove the bot field is byte-identical.
- **Phase 3 — auction gating:** `SeatView.eligible_positions`; `bot_max_bid` abstain gate; engine computes bounds once + per-bot eligibility; nomination refinement + pool-thin fallback; auction bot tests.
- **Phase 4 — measure + sync:** re-run the bake-off, record Run B in `reports/auction_tournament_validation_2026.md` (data point, no verdict), update PM/TODO.

## 8. Open questions

None blocking. Deferred (named): real published auction values as the bot anchor (separate slice); whether the clearing/WTP model needs adjustment (decide *after* measuring Phase 4); making the FLEX/SUPER_FLEX anchor configurable rather than RB/QB-by-convention.
