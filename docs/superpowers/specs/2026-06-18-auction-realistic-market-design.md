# Auction realistic market — randomized nomination + mixed bot field + patient hero

**Status:** approved (brainstormed 2026-06-18), pending spec review.
**Owner:** draft-hub / auction.
**Branch:** `feat/auction-realistic-market` (stacked on `feat/auction-stars-and-scrubs` / PR #78).
**Source context:** `reports/auction_tournament_validation_2026.md` (Runs A–D), prior specs `2026-06-17-auction-strategy-tournament-design.md`, `…-stars-and-scrubs-design.md`.

## Problem

The roster eye-test (half/12, seed 4) exposed a simulation artifact, not a strategy insight: mid-tier starters cleared at **$1** — Marvin Harrison Jr. (159), DK Metcalf (162), Stafford (292), LaPorta (149) all went for a dollar. In a real auction those never reach $1; someone bids them up. Two coupled causes:

1. **Nomination is strict value-descending** (`nominate_order = bd.sort_values("auction_dollars", ascending=False)`): every stud is auctioned first, draining budgets at the top, so mid-tier players only come up after the room is broke.
2. **Every bot is the same all-in bidder** (`bot_max_bid` = `value×(1+noise)` until broke): the order-statistic overpay on studs pins ~all 15 bots at the $1 floor by mid-draft. Nobody *holds* budget to contest mid-round value.

This inflates exactly the strategies that cash $1 studs (`overbid`/`anchors`), so the current ranking is suspect. Fixing both makes mid-round prices realistic and turns the bake-off into an honest test.

## Goals

1. **Value-weighted-random nomination** — a tunable knob so worse players sometimes surface early (and bait an overpay), instead of always auctioning studs first.
2. **Heterogeneous bot field** — a mix of bidding archetypes so mid-round value is contested: some bots hold budget and pay up for mid-tier.
3. **A patient hero contestant** — `PatientValueBid`, a manager who deliberately holds budget for mid-round value; the 7th bake-off contestant.
4. **Re-run (Run E)** under the realistic market and record whether the ranking compresses or re-orders. **No winner** — data-gathering; September.

## Non-goals

- **Not** changing the scorer (`project_draft`), `resolve_bids`, the snake field, or the six existing hero models' bid magnitudes.
- **Not** strategic *nomination by seat* (each seat choosing whom to nominate to drain rivals) — that remains the deferred "nominators as contestants" axis. This slice randomizes the *order*, it does not give seats nomination agency.
- **Not** real published auction values (still its own future slice).
- **No** new pandera schema; reference enums not strings; `GsisId` canonical; `_PYARROW_STR` for gsis.
- **Backward-compatible defaults:** every new knob defaults to *current behavior* so existing engine/market tests stay green; the realistic settings are opted into at the bake-off/CLI layer.

## Chosen approach

### Part 1 — Value-weighted-random nomination (`simulation.py`)

Add `nomination_temp: float = 0.0`, threaded through `_simulate_to_state` → `simulate_auction` → `run_auction_tournament` → CLI.

- **`temp == 0.0` (default):** unchanged — the deterministic argmax over the value-sorted order (`next(g for g in nominate_order if undrafted and pos in union)`). Byte-identical to today; all existing tests pass.
- **`temp > 0.0`:** each round, build the candidate set = undrafted players whose position ∈ `union`; sample **one** with weight `w(g) = max(value(g), EPS) ** (1.0 / temp)` where `value(g) = auction_dollars(g)` and `EPS = 0.5` (so out-of-pool $0 players are rare-but-possible, not impossible). Normalize to probabilities; draw with the engine's seeded `rng`. `temp == 1.0` ⇒ probability ∝ value (studs likely early, scrubs sprinkled in); larger temp ⇒ flatter (toward uniform).
- The forced-pick fallback (when the candidate set is empty) is unchanged and still un-gates every seat.
- **Bake-off default:** the CLI passes `nomination_temp = 1.0` (value-weighted). Exposed as `--nomination-temp`.

RNG note: nomination sampling consumes from the same seeded `rng` as the bot bids; this is deterministic given the seed (R-determinism) but means realistic-market runs are a different stream than Runs A–D (expected; not comparable at absolute levels).

### Part 2 — Mixed bot field (`market.py` + `simulation.py`)

Generalize the single `bot_max_bid` into archetype objects sharing one signature. `SeatView` gains `budget: int` (remaining $) so reserve-aware archetypes can pace; default keeps the existing fields.

Define a `BotArchetype` protocol: `max_bid(seat_view, player, baseline_dollars, config, rng, *, price_jitter) -> int`. Each abstains (returns 0) when `open_slots <= 0` or the player's position ∉ `eligible_positions` (the existing gate, unchanged). Value tier is computed from `baseline_dollars` (the archetype ranks the nominee's `auction_dollars` against the in-pool field):

- **`AggressiveBot`** — *exactly today's* `bot_max_bid`: `round(max(min_bid, value*(1+N(0,jitter))))`. Blows budget early. (The current function is retained/wrapped so its behavior is byte-identical.)
- **`PatientValueBot(understud=0.5, midtier_premium=0.35, stud_frac=0.10, scrub_frac=0.50)`** — tier the nominee by value percentile among in-pool players:
  - **stud** (top `stud_frac`): bid `round(value * understud * (1+N(0,jitter)))` — well below market, won't chase (reserves budget).
  - **scrub** (bottom `scrub_frac`): `min_bid`.
  - **mid-tier** (the middle band): if it still has reserve (`budget - min_bid*(open_slots-1) > value`), bid `round(value * (1+midtier_premium) * (1+N(0,jitter)))` — *pays a premium to win value*; else `min_bid`. This is what bids Harrison up.
- **`BalancedBot(pace=2.0)`** — `AggressiveBot`'s WTP but capped to a pace ceiling so it can't blow the bank on one player: `min(value*(1+noise), pace * (budget / open_slots))`. Holds money across the draft.

`assign_bot_archetypes(n_bots: int, *, mix) -> list[BotArchetype]` deterministically assigns an archetype to each bot seat by index (round-robin over the mix so the composition is exact and reproducible). Default engine arg `bot_archetypes: Sequence[BotArchetype] | None = None` ⇒ **all `AggressiveBot`** (current behavior; existing tests unchanged). The bake-off passes a mixed field; default mix ≈ ⅓ aggressive / ⅓ patient / ⅓ balanced (exact split pinned in the plan; tunable).

The bid loop dispatches each bot seat through its assigned archetype instead of the lone `bot_max_bid`; the hero path is unchanged.

### Part 3 — Patient value-hunter hero contestant (`bid_strategy.py`)

`PatientValueBid()` — the AuctionBidStrategy analog of `PatientValueBot`, but reading VORP from the pool (consistent with the other hero models) and `AuctionView`:
- tier the nominee by VORP rank (reuse `_vorp_threshold`): **stud** (top tier) → `min_bid` (let it go; hold budget); **mid-tier** (next band) → bid up to a VORP-share of remaining budget while reserve remains (pays for value); **scrub** → `min_bid`.
- Spends late, on mid-round value — the human who waits. Shares the tier/reserve *concept* with `PatientValueBot`; a small shared helper (tier boundaries / reserve test) is factored if it stays clean across the two interfaces, else implemented in parallel (noted, not forced).

### Part 4 — Wiring + re-run

- `run_auction_tournament` + `simulate_auction` thread `nomination_temp` and `bot_archetypes`; `_MODELS` in the CLI grows to **seven** (adds `patient` → `PatientValueBid()`).
- CLI `compare` **defaults to the realistic market**: `nomination_temp = 1.0` and the mixed bot field. New flags `--nomination-temp` (default 1.0) and, if cheap, `--bot-mix`; otherwise the mix is a sensible fixed default. (Engine/function defaults remain the legacy market so library callers and tests are unaffected.)
- `reports/auction_tournament_validation_2026.md` gains **Run E** (seven models, realistic market). No winner.

## Requirements

R1. `nomination_temp` threaded through `_simulate_to_state`/`simulate_auction`/`run_auction_tournament`/CLI; default `0.0` reproduces today's deterministic argmax nomination exactly.
R2. `temp > 0`: nominee sampled from undrafted players with position ∈ `union`, weight `max(value, 0.5)**(1/temp)`, via the seeded `rng`; forced-pick + union constraints unchanged; the `assert bids` invariant still holds.
R3. `market.py` exposes `AggressiveBot` (byte-identical to current `bot_max_bid`), `PatientValueBot`, `BalancedBot` under a common `BotArchetype` signature; each respects the open-slots + eligibility abstain gate; `SeatView` gains `budget`.
R4. `assign_bot_archetypes(n_bots, mix)` is deterministic by seat index. `_simulate_to_state`'s `bot_archetypes` arg defaults to all-aggressive (current behavior); the bid loop dispatches per-seat archetypes.
R5. `PatientValueBid` exists in `bid_strategy.py`, satisfies `AuctionBidStrategy`, is a frozen dataclass, holds budget on studs and pays up for mid-tier value per §Part 3.
R6. CLI `compare` races seven models and defaults to the realistic market (`nomination_temp=1.0` + mixed field); `--nomination-temp` exposed.
R7. Determinism: nomination draws + bot bids are a pure function of the seeded `rng`; the hero bid is a pure function of state. Same seed ⇒ identical draft.
R8. Backward-compat: all existing engine/market/CLI tests pass unchanged (legacy defaults); the `AggressiveBot` path and `resolve_bids` are byte-identical; the snake field is untouched.
R9. Run E recorded; no winner declared.

## Edge cases / failure modes

- **Nomination candidate set all value-0** (only out-of-pool players left in union): the `EPS` floor makes weights equal ⇒ uniform sample among them. Never empty while a seat has an open slot (else forced path).
- **`temp` huge** → weights → equal → ~uniform; **`temp`→0⁺** is *not* used (temp=0 is the special-cased argmax); the plan picks the realistic default (1.0).
- **PatientValueBot reserve exhausted** → mid-tier bid falls to `min_bid` (won't overspend); still a valid bid.
- **All patient/balanced bots underbid a stud** → they still return ≥ `min_bid` (they bid *low*, not abstain), so `resolve_bids` always has ≥1 bid; completeness preserved. Studs still clear high because aggressive bots (and the hero) bid up.
- **Patient bots make studs *too cheap*** for the hero to be tested against — mitigated because the field is mixed (aggressive bots still chase studs) and the hero competes; verified by the integration test below.
- **`BalancedBot` pace cap < min_bid** (tiny budget late) → clamp so it never returns below `min_bid` when eligible.
- **Determinism across archetypes**: archetype assignment is seat-index deterministic; RNG consumption order is fixed by seat iteration.

## Testing expectations

Unit (hand-built small pool + config; no I/O):
- **Nomination:** `temp=0` selects the argmax (current); `temp>0` with a fixed seed can select a non-argmax, and over many draws a scrub is sometimes first while a stud is most often first (distribution sanity, seeded).
- **Archetypes:** `AggressiveBot` equals the old `bot_max_bid` on the same inputs; `PatientValueBot` underbids a stud (< market), pays a premium on a mid-tier with reserve, `min_bid`s a scrub and a mid-tier with no reserve; `BalancedBot` never exceeds its pace ceiling and never blows the budget on one stud.
- **`assign_bot_archetypes`** yields the exact requested composition, deterministically.
- **`PatientValueBid`:** `min_bid` on a stud, pays up a mid-tier with reserve, `min_bid` scrub.

Integration (engine):
- **The core fix:** in a draft with a mixed field + `temp>0`, a mid-tier player does **not** clear at `min_bid` (the Harrison-at-$1 artifact is gone) — assert a representative mid-tier nominee clears above `min_bid`.
- **Completeness:** `assert bids` never fires; the forced-pick "pool thin" path still works.
- **Determinism:** same `(strategy, seed, nomination_temp, mix)` ⇒ identical rosters.
- **Backward-compat:** with legacy defaults (`temp=0`, all-aggressive) the engine reproduces the current tests; `AggressiveBot` output matches the retained `bot_max_bid`.

Gates: `pytest -v` (the touched test modules; plus `-k "auction or simulation or market"`), `mypy src tests`, `ruff check src tests`, `ruff format --check src tests`.

## Phasing

One cohesive slice; the plan decomposes into ~6 tasks, each ≤5 files:
1. **Value-weighted-random nomination** in `_simulate_to_state` + param threading + tests (temp=0 identical; temp>0 sampling).
2. **Bot archetypes** in `market.py` (`AggressiveBot`/`PatientValueBot`/`BalancedBot` + `BotArchetype` + `SeatView.budget` + `assign_bot_archetypes`) + unit tests.
3. **Engine dispatch** — bid loop uses per-seat archetypes (default all-aggressive) + integration test (mid-tier clears above $1; completeness; determinism).
4. **`PatientValueBid`** hero contestant + tests.
5. **CLI wiring** — seven models, realistic-market defaults, `--nomination-temp` + tests.
6. **Run E** bake-off (chunked per the Raptor Lake fault) + tracking doc; no winner.
