# Broke-bot snake-draft behavior — design

**Status:** approved (brainstormed 2026-06-19). Proceeding to spec-review → plan → execute.
**Owner:** draft-hub / auction.
**Depends on:** the auction tournament (`market.py` / `simulation.py` / `tournament.py`),
the ESPN-anchored bot pricing (`espn_anchored_bot_prices`, PR #83), and the snake-draft
noisy-ADP picker `bot_pick` (`assistant/opponent.py`).

## Problem

The auction bake-off exists to measure whether a hero strategy beats a **realistic field of
league mates**. Beating bots is only meaningful insofar as the bots proxy real human managers.
Two facts make the current bot field a poor proxy at the bottom of the budget:

1. **The bots' unranked-player value leaks our private VORP.** When a bot is asked to value an
   ESPN-*unranked* player, `espn_anchored_bot_prices` falls back to `_UNRANKED_MODEL_DISCOUNT`
   (0.4) × our own VORP-based model value. A real opponent never has our VORP. The one signal a
   human *does* have for these players — their **draft rank (consensus ADP)** and positional
   context ("$1 on ESPN, but he's the 3rd-best RB left") — is exactly what the bots don't use.

2. **A broke bot bids `$1` on whatever is nominated.** Once a bot is down to `min_bid` per open
   slot (`feasible_max == min_bid`), it can no longer outbid anyone, yet the current archetypes
   still bid `min_bid` on *every* eligible nominee. So a broke bot grabs whatever cheap player
   happens to be nominated next — including a 3rd-string QB to fill a QB slot — instead of
   holding out for the best player still available at a position it needs. (This is the
   "bot literally started a 3rd-string QB" bug the discount was a stopgap for; it treated a
   broke-bot *behavior* problem by bending *flush-bot pricing*.)

A real manager who is out of money behaves like a **snake drafter**: with $1 to spend, take the
best player still on the board that fills a need, by draft rank, not whatever is in front of you.

## Goals

1. Give every **bot** a snake-draft behavior in the **broke** regime (`feasible_max == min_bid`):
   it acts only on its single best-available-by-noisy-ADP target among the positions it can still
   roster — as **nominator** it nominates that target; as **responder** it bids `min_bid` only on
   that target and otherwise abstains.
2. Add the standard auction rule that makes abstention safe: **the nominating seat takes its
   nominee at `min_bid` when no one bids.** (Today every eligible seat always bids, so the case
   never arises; abstaining broke bots make it reachable.)
3. Reuse the existing snake-draft noisy-ADP selection (`bot_pick`) so an out-of-money auction bot
   and a snake-draft bot pick by identical semantics.
4. Leave **flush** bots, the **hero**, and **flush/hero nomination** byte-for-byte unchanged.

## Non-goals

- **Not** changing flush-bot bidding or pricing. Flush bots keep centering WTP on `bot_dollars`
  (the ESPN-anchored vector, discount and all). The `_UNRANKED_MODEL_DISCOUNT` knob and its sweep
  are **shelved, not removed** — the discount still prices unranked players for *flush* bots.
- **Not** changing the hero. The hero is the contestant under test; its broke behavior is part of
  the bid model being evaluated. The snake regime applies to **bot seats only**.
- **Not** changing flush/hero nomination. Central value-nomination (`_sample_nominee` over
  `auction_dollars`) is unchanged for every seat that is not a broke bot.
- **Not** a soft/parameterized "almost broke" threshold. The regime is the crisp binary
  `feasible_max == min_bid`.
- **Not** re-running the multi-year bake-off as part of this code change (that is the validation
  follow-up, tracked under #49a — see Rollout).

## Chosen approach

### Regime: a per-bot, per-nomination binary

A bot seat is **broke** at a nomination iff `feasible_max == min_bid`, where
`feasible_max = budget − min_bid × (open_slots − 1)` (the engine's existing `_feasible_max`).
Feasibility guarantees `feasible_max ≥ min_bid` always, so the test is equivalently "the bot
cannot bid above `min_bid` on any single player." Flush = not broke.

The **engine** owns regime detection (it already computes `feasible_max` and each open seat's
cap-legal eligible positions `seat_eligible[seat]`). The engine routes:
- **flush bot** → `archetype.max_bid(...)` exactly as today (unchanged);
- **broke bot** → the snake policy below;
- **hero** → its strategy exactly as today (unchanged), regardless of the hero's own budget.

### The snake board: a per-bot, fixed-per-draft noisy-ADP ranking

Each **bot seat** gets one `SnakeBoard` created at auction start. It holds a fixed perturbation
of consensus ADP — every player's `consensus_adp` plus a single `N(0, adp_jitter)` draw, sampled
**once per draft per bot** and reused for the rest of that draft. This mirrors a real manager:
the board is set on draft day (random, different every draft/seed) but does not reshuffle on
every nomination.

`SnakeBoard.best_available(undrafted_ids, eligible_positions) -> GsisId | None` returns the
lowest fixed-noisy-ADP player among `undrafted_ids` whose position is in `eligible_positions`
(`None` if that set is empty). Selection semantics — null ADP → `+inf`, ties broken on `gsis_id`
ascending, result independent of input row order — are **identical to `bot_pick`** because both
share the same extracted selection core (see Reuse).

**Why fixed-per-draft (not redraw-per-nomination):** a per-nomination redraw makes the bot's
target flip-flop round to round (player A is the target when A is nominated, then B looks better a
round later), so the bot passes inconsistently; it also consumes a variable amount of RNG, muddying
the paired/CRN comparison across models. Fixed-per-draft is stable and CRN-clean.

**CRN / shared-field seeding.** The bot field must be identical across the models a bake-off
compares (paired design). The per-bot ADP noise is therefore drawn at auction init from a
**dedicated RNG seeded off the auction seed alone** — independent of the bidding `rng` whose
stream the hero strategy perturbs differently per model. This keeps every model's bot snake boards
byte-identical for a given seed.

### Broke-bot behavior, end to end

For a broke bot seat with eligible positions `E = seat_eligible[seat]` and undrafted set `U`,
let `target = SnakeBoard.best_available(U, E)`:

- **As nominator:** nominate `target` (instead of central `_sample_nominee`). Because `target`'s
  position is in `E`, the nominee is always one this seat can roster — so the "take it for $1"
  backstop is always valid for a broke nominator.
- **As responder** (someone else's nominee `g`): bid `min_bid` iff `g == target`; else abstain (0).

Round-robin nomination guarantees each open seat nominates periodically, so a broke bot reliably
secures its target (either by nominating it and winning the backstop, or by sniping it when
another seat nominates it) and **never grabs an off-target scrub**.

### The nominator backstop

In the engine's bid-collection step, if `bids` is empty after polling every open seat, **award the
nominee at `min_bid`** rather than asserting. The awardee is:
- the **nominator**, if it is eligible to roster the nominee (the common, intended case — and the
  *only* case in the all-broke endgame, since a broke nominator nominates its own eligible target
  and bids `min_bid` on it, so `bids` is in fact non-empty there);
- otherwise the **lowest-index open seat eligible to roster the nominee**, which is guaranteed to
  exist because central nomination only ever nominates a player some open seat can roster (the
  room-union rule). This covers the sole residual empty-bids edge: a *flush* nominator nominating a
  room-union player it personally cannot roster (its own position capped) while every responding
  open seat is a broke bot not targeting it.

This is a faithful, last-resort generalization of "the nominator drafts the player for $1," and it
never reintroduces broke-bot scrub-grabbing (broke bots still only ever *bid* on their target).

### Reuse, don't duplicate

`bot_pick(available, rng, *, adp_jitter)` (`assistant/opponent.py`) currently (a) draws fresh
`N(0, adp_jitter)` noise and (b) selects the argmin noisy-ADP with a `gsis_id` tiebreak. Extract
(b) into a shared, noise-injected core so the snake board and `bot_pick` cannot drift apart:

```
_best_by_noisy_adp(available: pd.DataFrame, noisy_adp: np.ndarray) -> GsisId
```

`bot_pick` becomes "draw noise, call the core"; `SnakeBoard` stores its fixed noisy ADP and calls
the core with a slice for the eligible/undrafted subset. The tiebreak (`gsis` ascending) and
null→`+inf` handling live in the core, asserted once.

## Components

- **`SnakeBoard`** (new) — per-bot fixed-noisy-ADP ranking + `best_available`. Lives in the auction
  package (`market.py`, or a small `snake_bot.py` if `market.py` grows — plan decides). Pure and
  unit-tested. Depends on: pool `gsis_id`/`consensus_adp`/`position`, an init-time noise draw, and
  the shared `_best_by_noisy_adp` core.
- **`_best_by_noisy_adp`** (new, extracted from `bot_pick`) — shared selection core in
  `assistant/opponent.py`. `bot_pick` refactored to call it; behavior byte-identical (equivalence
  test).
- **`_simulate_to_state`** (`simulation.py`) — (1) build a `SnakeBoard` per bot seat at init from
  the dedicated snake RNG; (2) in the nomination step, if the nominator is a broke bot, nominate
  its `target` instead of `_sample_nominee`; (3) in the bid step, route broke bots to the snake
  responder rule; (4) replace the `assert bids` with the nominator backstop.
- **Bot archetypes** (`market.py`) — **unchanged**. The regime switch is engine-side; flush
  bidding is untouched.

## Data flow

Auction init → draw per-bot ADP noise (dedicated snake RNG) → `SnakeBoard` per bot seat.
Each round → engine computes `seat_eligible` + `feasible_max` per open seat → nominator step
(broke bot ⇒ its target; else `_sample_nominee`) → bid step (per open seat: hero ⇒ strategy;
flush bot ⇒ archetype; broke bot ⇒ `min_bid` iff nominee is its target else abstain) →
`resolve_bids`, or the backstop award if no bids → update budgets/rosters → next round.

## Edge cases

- **All open seats broke (endgame):** every nominator nominates its own eligible target and bids
  `min_bid` on it; `bids` is non-empty; `resolve_bids` gives a lone bidder the player at `min_bid`.
- **Flush nominator, nominee it can't roster, all responders broke non-targeters:** the residual
  empty-bids case → backstop awards to the lowest-index open seat that can roster the nominee.
- **`best_available` returns `None`** (no undrafted player in the seat's eligible positions): the
  broke bot abstains as responder and, as nominator, cannot have been routed here (it has an open
  slot ⇒ `bot_eligible` yields a non-empty set, falling back to the engine's existing `forced`
  thin-pool path). Spec the `None` guard explicitly.
- **Ties when two broke bots target the same player:** both bid `min_bid`; `resolve_bids` breaks on
  lowest seat. Per-bot noise makes exact ties rare; the deterministic tiebreak is acceptable.
- **Null `consensus_adp`:** treated as `+inf` (no market signal) — identical to `bot_pick`. Per the
  coverage check, every in-pool ESPN-unranked player has a non-null ADP, so this only affects deep
  out-of-pool rows that never get nominated.

## Testing

- **`_best_by_noisy_adp` equivalence:** `bot_pick` before/after the refactor returns the identical
  pick for fixed seeds across several pools (pins byte-identity).
- **`SnakeBoard.best_available`:** respects eligibility filtering; honors the fixed noise (same board
  → same pick across calls within a draft); null-ADP → `+inf`; `None` on an empty eligible set;
  order-independent.
- **Regime routing:** a flush bot's bid is unchanged vs. `archetype.max_bid` (the snake path is not
  taken); a broke bot abstains on a non-target nominee and bids `min_bid` on its target.
- **Backstop:** an engineered no-bid nomination awards the nominee at `min_bid` to an eligible seat
  (nominator when eligible; else lowest-index eligible open seat); rosters still fill; no assertion.
- **No-scrub property:** in a seeded endgame, broke bots never roster a player that was not their
  best-available-for-needs at award time (the behavioral guarantee the discount could only
  approximate).
- **Full-auction invariants:** every seat fills its roster; budgets never go negative; pool-thin
  `forced` path still fires when intended.
- **Flush-unchanged regression:** an all-flush auction (large budgets) produces byte-identical
  rosters to `main` for fixed seeds (the snake path is never entered).

## Open questions / deferred

- **Broke-bot nomination vs. budget-drain nomination.** Real managers sometimes nominate players
  they *don't* want to drain rivals' budgets. Out of scope; broke bots nominate their own target.
- **Should flush bots also use ADP rather than `discount × VORP` for unranked players?** The user's
  "don't touch flush (yet)" defers this; if adopted later it would retire `_UNRANKED_MODEL_DISCOUNT`
  entirely.
- **Re-tuning `adp_jitter` for the auction broke regime.** Reuse the snake-draft jitter default to
  start; a sweep is a later axis, not part of this slice.

## Rollout / validation

Code-only behavior change to the bot field; it shifts every ESPN-anchored bake-off number again
(expected). After merge, the **multi-year bake-off (#49a) is re-run** with the snake-draft broke
bots as the new methodology baseline and recorded in
`reports/auction_tournament_validation_2026.md`, replacing the discount-era numbers. No winner is
declared (the September strategy decision is unchanged).
