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
4. Leave the **flush**-bot bid logic, the **hero**, and **flush/hero nomination** unchanged at the
   *code* level. (Realized flush *bids* in a mixed auction will differ from `main` because the bot
   field's RNG consumption changes — broke bots that abstain draw no `price_jitter` noise where they
   used to. That is expected and acceptable: we are deliberately changing the bot field, the auction's
   paired comparison is **seed**-paired not draw-aligned, and the load-bearing CRN for the headline
   metrics is the *season* sim's separate `season_base_seed`-keyed RNG (tournament.py:144), which this
   change does not touch. Byte-identity to `main` is asserted only for the **all-flush** case, where the
   snake path is provably never entered — see Testing.)

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
Feasibility guarantees `feasible_max ≥ min_bid` always (`validate_auction_inputs` rejects
`budget < min_bid × roster_size`, and the bid clamp preserves the invariant inductively), so the
test is equivalently "the bot cannot bid above `min_bid` on any single player." Flush = not broke.

**Accepted bound (the crisp binary is deliberate, per the user's "binary — has money or doesn't").**
`feasible_max == min_bid` ⟺ `budget == min_bid × open_slots` (exactly $1 per remaining slot). A bot
one dollar above the line (`budget == min_bid × open_slots + 1`) is *flush* by this test and still
bids via its archetype, so it can still grab an off-target $1–$2 player. This residual is **bounded
and self-correcting**: such a bot can win at most ≈1–2 players above `min_bid` before its budget
drops to the floor and it becomes broke, after which the snake rule governs. We accept it rather than
introduce a tunable `feasible_max ≤ min_bid + k` margin (no free parameter to sweep, keeps the CRN
comparison clean, and matches the exact point at which the engine's own clamp pins the bid to
`min_bid`). The "no-scrub" guarantee (Testing) is therefore stated for genuinely-broke bots only.

The **engine** owns regime detection (it already computes `feasible_max` and each open seat's
cap-legal eligible positions `seat_eligible[seat]`). On the **non-`forced`** path (see "Interaction
with the `forced` thin-pool path" below) the engine routes each open seat:
- **flush bot** → `archetype.max_bid(...)` exactly as today (unchanged);
- **broke bot** → the snake policy below;
- **hero** → its strategy exactly as today (unchanged), regardless of the hero's own budget.

### The snake board: a per-bot, fixed-per-draft noisy-ADP ranking

Each **bot seat** gets one `SnakeBoard` created at auction start. It holds a fixed perturbation
of consensus ADP — every player's `consensus_adp` plus a single `N(0, adp_jitter)` draw, sampled
**once per draft per bot** and reused for the rest of that draft. This mirrors a real manager:
the board is set on draft day (random, different every draft/seed) but does not reshuffle on
every nomination.

**`consensus_adp` is optional — guard it (the snake regime degrades gracefully).** `VorpTableSchema`
declares `consensus_adp` as an **optional** column (`schemas.py`: present only on the consensus-fed
path; weekly-path tables omit it and still validate). The existing auction engine never reads it. So
the snake regime is **enabled only when the pool carries a usable `consensus_adp`** (column present
and not all-null). When it is absent or all-null, broke bots fall back to **today's behavior**
(`archetype.max_bid`, central nomination, no abstention) — i.e. the snake regime is silently
disabled and the auction runs exactly as on `main`. This is decided **once per auction at init**
(a single boolean), so all-flush auctions on a no-ADP pool are byte-identical to `main`. The
multi-year bake-off pools all carry `consensus_adp` (verified: 100% in-pool coverage, every season),
so the regime is always active there; the guard exists for robustness on other pool sources.

`SnakeBoard.best_available(undrafted_ids, eligible_positions) -> GsisId | None` returns the
lowest fixed-noisy-ADP player among `undrafted_ids` whose position is in `eligible_positions`
(`None` if that set is empty). Selection semantics — null ADP → `+inf`, ties broken on `gsis_id`
ascending, result independent of input row order — are **identical to `bot_pick`** because both
share the same extracted selection core (see Reuse).

**Why fixed-per-draft (not redraw-per-nomination):** a per-nomination redraw makes the bot's
target flip-flop round to round (player A is the target when A is nominated, then B looks better a
round later), so the bot passes inconsistently; it also consumes a variable amount of RNG, muddying
the paired/CRN comparison across models. Fixed-per-draft is stable and CRN-clean.

**CRN / shared-field seeding (requires a signature change — the engine has no seed today).** The
bot field must be identical across the models a bake-off compares (paired design), so the per-bot
ADP noise must come from an RNG **independent of the bidding `rng`** (whose stream the hero strategy
perturbs differently per model). But the engine currently receives only the *constructed* bidding
`Generator` (`tournament.py:132` builds `np.random.default_rng(base_seed + s)` and passes it as
`rng=`), never the integer seed — so "seed a dedicated RNG off the auction seed" is not implementable
without threading. The chosen mechanism:

- Add an optional `snake_rng: np.random.Generator | None = None` parameter to `simulate_auction` and
  `_simulate_to_state` (the only production caller is `tournament.py`; the rest are tests — small
  blast radius).
- **The tournament constructs a dedicated substream off the auction seed alone:**
  `snake_rng = np.random.default_rng([base_seed + s, _SNAKE_SUBSTREAM])` (a list-seed sub-key,
  verified deterministic and distinct from the bidding stream). Identical across every model at seed
  `s`; independent of the hero's bidding-stream consumption ⇒ snake boards byte-identical across
  models.
- **Default when `snake_rng is None`:** derive `snake_rng = rng.spawn(1)[0]`. Verified: numpy's
  `Generator.spawn` advances the parent's *seed sequence spawn-key*, **not** its output stream — so
  spawning does **not** perturb the bidding `rng`'s subsequent draws (an all-flush auction stays
  byte-identical to `main`), and it is deterministic for a given parent seed. This keeps existing
  test callers working without passing a seed while remaining CRN-clean.

The per-nomination decisions (`best_available` lookups) consume **no** RNG — all randomness is the
one-shot init draw from `snake_rng` — so the bidding stream's draw count is unaffected by the snake
machinery itself (only by broke-bot *abstention*, which is the accepted, documented drift in Goal 4).

### Broke-bot behavior, end to end

For a broke bot seat with eligible positions `E = seat_eligible[seat]` and undrafted set `U`,
let `target = SnakeBoard.best_available(U, E)`:

- **As nominator:** nominate `target` (instead of central `_sample_nominee`). Because `target`'s
  position is in `E`, the nominee is always one this seat can roster — so the "take it for $1"
  backstop is always valid for a broke nominator. **If `target is None`** (no undrafted player in the
  seat's eligible positions — possible when `E` is non-empty but its positions are pool-exhausted,
  while the room-`union` is still satisfied by *other* seats so the engine's `forced` flag is False),
  the broke bot falls back to **central `_sample_nominee`** for that nomination (it nominates like a
  flush seat would). It does **not** rely on the engine `forced` path, which is a per-round *union*
  property and will not fire just because this one seat's positions are exhausted.
- **As responder** (someone else's nominee `g`): bid `min_bid` iff `g == target`; else abstain (0).
  **If `target is None`,** abstain.

Round-robin nomination guarantees each open seat nominates periodically, so a broke bot keeps acting
on its **current** best-available-for-needs target — securing it when it nominates (winning the
backstop if unbid) or sniping it when another seat nominates it. It may be **out-bid by flush bots**
on its top targets (a flush bot bids above `min_bid`; the broke bot bids exactly `min_bid`), in which
case it simply re-targets the new best-available next round — so it is **not** guaranteed any
particular player, but it **never rosters an off-target scrub**: every player it wins was its
best-available-for-needs at award time.

### Interaction with the `forced` thin-pool path

The engine has a `forced` branch (`simulation.py:164–171`): when no undrafted player's position is in
the room-`union`, it nominates an **ungated** player and widens every seat's eligibility to
`all_positions` for that round. **The snake regime and the backstop apply only when `not forced`.**
Under `forced`, every open seat — broke bots included — uses **today's** ungated behavior
(`archetype.max_bid` for bots, floored to `min_bid` by the clamp), so `bids` is non-empty exactly as
on `main` and neither the abstention nor the backstop is reached. This is deliberate: `forced` is a
rare pool-exhaustion path where `seat_eligible` is not the eligibility the engine actually applies
(it uses `all_positions`), so routing broke bots through the snake board there would consult the wrong
eligibility set. Suppressing the regime under `forced` keeps that path byte-identical to `main`.

### The nominator backstop (non-`forced` path only)

On the **non-`forced`** path, if `bids` is empty after polling every open seat, **award the nominee
at `min_bid`** rather than asserting. The awardee is:
- the **nominator**, if it is eligible to roster the nominee (the common, intended case — and the
  *only* case in the all-broke endgame, since a broke nominator nominates its own eligible target and
  bids `min_bid` on it, so `bids` is in fact non-empty there);
- otherwise the **lowest-index open seat eligible to roster the nominee** (consulting
  `seat_eligible[seat]`, well-defined because `not forced`), which is **guaranteed to exist** because
  on the non-`forced` path central nomination only ever nominates a player some open seat can roster
  (the room-`union` rule: `candidates` is filtered to `pos_by_id[g] in union`, and `union` is the
  join of `seat_eligible` over open seats). This covers the sole residual empty-bids edge: a *flush*
  nominator nominating a room-union player it personally cannot roster (its own position capped) while
  every responding open seat is a broke bot not targeting it.

This is a faithful, last-resort generalization of "the nominator drafts the player for $1," and it
never reintroduces broke-bot scrub-grabbing (broke bots still only ever *bid* on their target).

**Broke-hero endgame note.** The all-broke-endgame "`bids` is non-empty" claim above holds even when
a **broke hero** is the nominator: the hero nominates centrally (it is not a broke *bot*) and bids via
`strategy.max_bid`, which the engine clamps to `≥ min_bid` (`simulation.py:190`) — so the hero
self-bids on its own nominee and `bids` is non-empty. The endgame invariant therefore does not depend
on all open seats being bots.

### Reuse, don't duplicate

`bot_pick(available, rng, *, adp_jitter)` (`assistant/opponent.py`) currently (a) draws fresh
`N(0, adp_jitter)` noise and (b) selects the argmin noisy-ADP with a `gsis_id` tiebreak. Extract
(b) into a shared, noise-injected core so the snake board and `bot_pick` cannot drift apart:

```
_best_by_noisy_adp(gsis: np.ndarray, noisy_adp: np.ndarray) -> GsisId
```

`bot_pick` becomes "draw noise, call the core"; `SnakeBoard` stores its fixed noisy ADP and calls
the core with a slice for the eligible/undrafted subset. The tiebreak (`gsis` ascending) and
null→`+inf` handling live in the core, asserted once.

## Components

- **`SnakeBoard`** (new) — per-bot fixed-noisy-ADP ranking + `best_available`. Lives in the auction
  package (`market.py`, or a small `snake_bot.py` if `market.py` grows — plan decides). Pure and
  unit-tested. Depends on: pool `gsis_id`/`consensus_adp`/`position`, an init-time noise draw, and
  the shared `_best_by_noisy_adp` core. **Enum hygiene:** `eligible_positions` arrives as
  `frozenset[Position]` (`seat_eligible` values) while `pool["position"]` is a raw string column —
  the board must normalize via `Position(str(p))` for the filter join (CLAUDE.md "reference enums,
  never the strings"). Reuse the engine's existing `pos_by_id` (`simulation.py:117–119`) and the pool
  frame already in scope rather than re-deriving.
- **`_best_by_noisy_adp`** (new, extracted from `bot_pick`) — shared selection core in
  `assistant/opponent.py`. `bot_pick` refactored to "sort-by-gsis → draw noise → call the core"
  (the noise draw and the sort-before-draw ordering stay **inside** `bot_pick` to preserve its
  existing order-independence guarantee); behavior byte-identical (equivalence test).
- **`_simulate_to_state` / `simulate_auction`** (`simulation.py`) — (0) add the
  `snake_rng: np.random.Generator | None = None` parameter (default `rng.spawn(1)[0]`); (1) decide
  the `consensus_adp`-usable boolean once at init and, if usable, build a `SnakeBoard` per bot seat
  from `snake_rng`; (2) in the nomination step, **when `not forced`** and the nominator is a broke
  bot with a usable board, nominate its `target` (else `_sample_nominee`); (3) in the bid step,
  **when `not forced`**, route broke bots to the snake responder rule; (4) replace the `assert bids`
  with the non-`forced` nominator backstop. Under `forced`, or when the snake regime is disabled
  (no usable ADP), the loop behaves exactly as on `main`.
- **`run_auction_tournament`** (`tournament.py`) — construct and pass
  `snake_rng = np.random.default_rng([base_seed + s, _SNAKE_SUBSTREAM])` alongside the existing
  bidding `rng` (`tournament.py:125–132`).
- **Bot archetypes** (`market.py`) — **unchanged**. The regime switch is engine-side; flush
  bidding is untouched.

## Data flow

Auction init → decide `adp_usable` (consensus_adp present + not all-null) → if usable, draw per-bot
ADP noise from `snake_rng` → `SnakeBoard` per bot seat.
Each round → engine computes `seat_eligible` + `feasible_max` per open seat, and `forced`.
**If `forced` or not `adp_usable`:** behave exactly as `main` (central nomination, archetype bids,
`assert bids` holds). **Else:** nominator step (broke bot with target ⇒ its target; broke bot whose
target is `None` ⇒ `_sample_nominee`; flush/hero ⇒ `_sample_nominee`) → bid step (per open seat:
hero ⇒ strategy; flush bot ⇒ archetype; broke bot ⇒ `min_bid` iff nominee is its target else
abstain) → `resolve_bids`, or the backstop award if `bids` empty → update budgets/rosters → next
round.

## Edge cases

- **All open seats broke (endgame):** every nominator nominates its own eligible target and bids
  `min_bid` on it; `bids` is non-empty; `resolve_bids` gives a lone bidder the player at `min_bid`.
- **Flush nominator, nominee it can't roster, all responders broke non-targeters:** the residual
  empty-bids case → backstop awards to the lowest-index open seat that can roster the nominee.
- **`best_available` returns `None`** (the seat's eligible positions hold no undrafted player, even
  though `not forced` because the room-`union` is still satisfied by *other* seats): the broke bot
  **abstains** as a responder; as a **nominator** it falls back to **central `_sample_nominee`** for
  that round (it does *not* depend on the engine `forced` flag, which is a per-round union property
  and won't fire for one seat's exhausted positions).
- **`forced` thin-pool round:** the snake regime is suppressed; all seats use today's ungated
  behavior; `bids` non-empty as on `main` (see "Interaction with the `forced` thin-pool path").
- **No usable `consensus_adp` on the pool:** the snake regime is disabled for the whole auction;
  identical to `main`.
- **Ties when two broke bots target the same player:** both bid `min_bid`; `resolve_bids` breaks on
  lowest seat. Per-bot noise makes exact ties rare; the deterministic tiebreak is acceptable.
- **Null `consensus_adp` on individual rows** (pool *has* the column but some rows are null): treated
  as `+inf` (no market signal) — identical to `bot_pick`. Per the coverage check, every in-pool
  ESPN-unranked player has a non-null ADP, so this only affects deep out-of-pool rows never nominated.

## Testing

- **`_best_by_noisy_adp` equivalence:** `bot_pick` before/after the refactor returns the identical
  pick for fixed seeds across several pools (pins byte-identity).
- **`SnakeBoard.best_available`:** respects eligibility filtering; honors the fixed noise (same board
  → same pick across calls within a draft); null-ADP → `+inf`; `None` on an empty eligible set;
  order-independent.
- **Regime routing:** a flush bot's bid is unchanged vs. `archetype.max_bid` (the snake path is not
  taken); a broke bot abstains on a non-target nominee and bids `min_bid` on its target.
- **Backstop:** an engineered non-`forced` no-bid nomination awards the nominee at `min_bid` to an
  eligible seat (nominator when eligible; else lowest-index eligible open seat); rosters still fill;
  no assertion fires.
- **No-scrub property:** in a seeded endgame, every player a **genuinely-broke** bot
  (`feasible_max == min_bid`) wins was its best-available-for-needs **at award time** (not necessarily
  its initial target — it may have been out-sniped earlier). The behavioral guarantee the discount
  could only approximate; asserted for broke bots only (the accepted marginal-surplus residual, H1,
  is out of scope for this assertion).
- **Full-auction invariants:** every seat fills its roster; budgets never go negative; pool-thin
  `forced` path still fires when intended **and runs as on `main`** (snake regime suppressed).
- **No-usable-ADP fallback:** an auction on a pool lacking `consensus_adp` (or all-null) produces
  byte-identical rosters to `main` for fixed seeds (snake regime disabled at init).
- **Snake-RNG isolation:** (a) `rng.spawn(1)` does not perturb the parent bidding stream, so an
  **all-flush** auction (large budgets, snake path never entered) is byte-identical to `main`;
  (b) the tournament's `default_rng([base_seed+s, _SNAKE_SUBSTREAM])` yields identical snake boards
  across two different hero strategies at the same seed (CRN: shared bot field).

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
