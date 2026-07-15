# Auction Nomination Poisoning — Feasibility Probe (Slice 2)

**Status:** design approved (brainstorming, 2026-07-15).
**Branch:** `feat/auction-nomination-poison-probe`.

## Problem

The auction hero (`BalancedValueBid`, `premium=0.0` — the retuned default from Run N) has **zero control over nominations**. `simulation._sample_nominee` picks a nominee value-weighted-at-random for *every* seat, the hero included; there is no hero-specific branch.

The hypothesis: the hero could use its own nominations to **poison** the market — decoy-nominate expensive players to drain opponents' budgets (especially the uncapped `AggressiveBot`s, which already blow their budget early) *before* the hero's targets clear. The Run M microstructure diagnostic showed the raw material: early studs clear at ~1.34× fair value and aggressive bots end the draft broke.

The post-fix baseline is **seat-averaged `reg_win_pct` ≈ 0.59** in both markets (`balanced` p0.0, Run N). The question this slice answers: **does giving the hero nomination control lift `reg_win_pct` above ~0.59 in both markets?**

Nomination is a **bounded lever** — the hero controls only its own turns in the round-robin: ~17 of the 204 total nominations (~8%). So before investing in a full `NominationStrategy` abstraction, we run a cheap **feasibility probe** to get a go/no-go.

## Goals

- Add a minimal, **opt-in** `hero_nominator` seam to the auction sim so the hero's nomination turns can use a poison heuristic instead of `_sample_nominee`.
- Probe **two** poison heuristics (`drain-max`, `drain-off-position`) against a no-poison **control**, seat-averaged across **both** bot markets.
- Produce a clean **go/no-go** on whether nomination poisoning moves `reg_win_pct` above the ~0.59 baseline.

## Non-goals

- The full `NominationStrategy` protocol / heuristic family / nomination tournament — built **only if** the probe says "go" (its own future slice).
- Re-optimizing the **bid** strategy alongside nomination. The bid is held fixed at `balanced` `premium=0.0` for every contestant so the probe isolates the *nomination* lift.
- The `shadow-target` / price-enforcement heuristics (deferred; `shadow-target` needs a notion of the hero's "next target" that `BalancedValueBid` does not have).
- Any live-draft strategy decision — deferred to September per project policy. This is data-gathering.

## Chosen approach

### The seam (minimal, reusable)

Add an opt-in parameter `hero_nominator` to `_simulate_to_state` and `simulate_auction`:

- When `state.nominator == hero0` **and** `hero_nominator is not None` **and** the nomination is on the **non-forced** path, the hero's nominee is chosen by the callable from the already-computed room-rosterable `candidates` list; the returned id replaces the `_sample_nominee` result for that turn only.
- Otherwise (`None`, a bot's turn, or the forced pool-thin fallback) behavior is **unchanged**, byte-for-byte.

Signature: `hero_nominator(candidates: list[str], ctx: NominationContext) -> str`, where `candidates` is the existing room-rosterable undrafted list (already sorted value-descending via `nominate_order`) and `NominationContext` is a small frozen dataclass exposing what the heuristics need:

- `hero_positions: Counter[str]` — the hero's drafted position counts (for `drain-off-position`).
- `value_by_id: Mapping[str, float]` — the market value the room prices on (`bd["bot_dollars"]`), so "priciest" means "what the room will spend most on."
- `config: LeagueConfig` — for the position starter requirements.

The hook does **not** replace the snake-bot broke-nominator path (that path is bots-only). It is the seed of the future `NominationStrategy` protocol: grow it if the probe wins, delete one parameter if it loses.

### The two poison heuristics (bid fixed = `balanced` `premium=0.0`)

- **`drain_max`** → return the highest-`value_by_id` candidate. `candidates` is already value-descending, so this is `candidates[0]` under the model market; under ESPN the ordering is by our SOS `auction_dollars` while the drain intent wants the `bot_dollars` max, so the heuristic takes an explicit `argmax` over `value_by_id` rather than trusting the list order. Forces the room to spend on a stud the hero would lose anyway (clears ~1.34× fair > its cap) → pure drain, near-zero backfire (high-value players always draw a bid).
- **`drain_off_position`** → return the highest-`value_by_id` candidate whose position the hero has **already filled to its starter requirement** (from `config.roster_slots`); if no candidate qualifies (early draft, before the hero has filled any position), **fall back to `drain_max`**. Targets the drain at opponents who still need that slot, with near-zero chance the hero is left holding the decoy.

### Measurement

Identical methodology to the Run N premium sweep, so results are directly comparable:

- Seat-averaged `reg_win_pct` (+ `make_playoffs_pct`, `champ_pct` for context) over **12 seats × both markets × 20 seeds × 300 sims**, crash-safe chunked (reuse the `scripts/auction_seat_sweep.py` seat-average + aggregation shape).
- Contestants, all bidding `balanced` `premium=0.0`: **`control`** (`hero_nominator=None`), **`drain_max`**, **`drain_off_position`**.
- **Sanity check:** `control` must reproduce the Run N `balanced` figure (~0.59 model / ~0.62 espn seat-avg). If it does not, the probe harness is wrong — stop and fix before reading poison lift.

## Requirements

- **R1 — None is identity.** `hero_nominator=None` leaves every seat's roster and budget byte-identical to the current engine (regression test, same shape as the `PickRecord` trace test).
- **R2 — hero-only, non-forced-only.** The hook affects only the hero seat's non-forced nominations. Bot nominations, the snake-bot broke path, and the forced pool-thin fallback are unchanged.
- **R3 — validity by construction.** The heuristic picks from the passed `candidates` list; the engine asserts the returned id is in `candidates` (a heuristic returning an out-of-set or drafted id is a bug, not silently accepted).
- **R4 — heuristic correctness.** `drain_max` returns the max-`value_by_id` candidate; `drain_off_position` returns the max-value candidate at an over-filled position, or falls back to `drain_max` when none qualifies.
- **R5 — bid held fixed.** Every probe contestant uses `BalancedValueBid(premium=0.0)` as the bid strategy; only the nominator varies.
- **R6 — gates.** `pytest`, `mypy src tests` (strict), `ruff check`, `ruff format --check` all clean.

## Edge cases / failure modes

- **Backfire (decoy nobody bids on).** If the hero nominates a player and no one bids, `resolve_unbid` awards it to the nominator (the hero) at `min_bid` if the hero can roster the position. Mitigation: both heuristics nominate **high-value** players, which always draw an aggressive-bot bid; `drain_off_position` additionally picks a position the hero has filled, so `resolve_unbid` would pass it to another open seat rather than the hero. This is a *mitigation, not a hard guarantee* — documented, and the win% measurement captures any residual harm.
- **Early draft, no filled position.** `drain_off_position` falls back to `drain_max`.
- **Value source.** Heuristics rank by `bd["bot_dollars"]` (the value the room actually bids on), not our `auction_dollars`, so "priciest" means "biggest drain" in *both* markets (they differ under ESPN anchoring).
- **Forced nomination.** On the pool-thin forced path the hook is not consulted; the existing forced nominee is used.
- **Hero nominates a player it wants.** Allowed — `drain_max` may nominate a stud the hero would like; the hero bids its normal capped bid and typically loses, and the drain still lands. No special-casing.
- **Candidate list empty.** Cannot occur on the non-forced path (the loop only reaches nomination while a rosterable candidate exists); the hook is only called there.

## Testing expectations

- `hero_nominator=None` → rosters + budgets identical to a no-hook run (regression).
- Hook invoked only on the hero's turn and only non-forced; returns a member of `candidates`.
- `drain_max` unit test: returns the max-`value_by_id` candidate on a small synthetic pool.
- `drain_off_position` unit tests: returns the off-position pick when the hero has over-filled a position; falls back to `drain_max` when none qualifies.
- If the probe runner is committed (only on graduation), a light aggregation test parallel to `test_auction_seat_sweep.py`.

## Phasing

- **Phase 1 — the seam.** `hero_nominator` hook on `_simulate_to_state`/`simulate_auction` + `NominationContext` + wiring; tests R1/R2/R3. No heuristics yet.
- **Phase 2 — the heuristics.** `drain_max` and `drain_off_position` + unit tests R4.
- **Phase 3 — the probe + verdict.** Crash-safe both-market seat sweep (control + 2 poisons), `Run O` writeup in `reports/auction_tournament_validation_2026.md`, memory update, and the **go/no-go** call.

## Go / no-go

- **Go** → a poison heuristic beats `control` by **≥ +0.02 worst-case `reg_win_pct`** (min over the two markets), seat-stable, in **both** markets. Then design the full `NominationStrategy` abstraction + heuristic family as a new slice.
- **No-go** → neither poison beats `control` in both markets → record as data, the ~0.59 `balanced` p0.0 hero stands, **delete the probe hook**, and close nomination poisoning. Either outcome is a clean, publishable result.
