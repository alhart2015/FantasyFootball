# Auction Value-Gap Nomination — Will-league probe (Slice 2b)

**Status:** design approved (2026-08-12).
**Branch:** `feat/auction-value-gap-nomination`.

## Problem

Run O probed nomination poisoning and returned NO-GO. But it tested the wrong lever, on the wrong
board, for the wrong league:

1. **Both heuristics ranked by price, not by disagreement.** `drain_max` picks the priciest
   candidate overall; `drain_off_position` picks the priciest candidate at a position the hero has
   already filled. Neither reads the *gap* between what the room will pay (`bot_dollars`) and what
   we think the player is worth (`auction_dollars`) — which is the entire premise of "nominate the
   players the room overvalues so their money leaves the room on someone else's roster."
2. **`drain_max` was near-redundant with the room's own nomination order.** The room nominates
   value-weighted (Run O) or ADP-weighted (Run P onward), so "nominate the priciest player" mostly
   surfaces the player who was coming up next anyway — measured lift +0.001 in the model market.
   The only information the hero added that the room did not already have was *which* expensive
   player is off-position **for us**, and that is exactly the variant that showed a real (if small)
   effect: +0.010 model, not separable from zero in ESPN.
3. **Run O predates the realism fix.** It ran on value-weighted nomination
   (`nomination_temp=1.0`, no `market_adp_jitter`). Run P (2026-07-16, one day later) showed the
   nomination model is not a detail — it *scrambled the entire hero ranking* (`anchors` last -> #3).
   The nomination probe was never re-run on the ADP board.
4. **Run O ran the generic `_REALISTIC_FIELD` on `half_12team`.** The question in front of us is
   Will's league: `overbidder` field, `will_half12` table, ESPN market.

## Hypothesis

In the ESPN market our board and the room's board genuinely disagree, and that disagreement is the
hero's whole edge. Nominating **where the room's number most exceeds ours** should drain opponent
budget at maximum efficiency per nomination: the room pays a price we consider an overpay, and the
hero was never going to buy that player anyway.

`drain_max` drains indiscriminately (including on players *we* rate, i.e. our own targets).
`drain_value_gap` drains only where the room is wrong in our favor.

## Goals

- Add a `drain_value_gap` heuristic family that ranks candidates by `bot_dollars - auction_dollars`
  (absolute dollars, not a ratio — the drain we care about is dollars out the door, and a ratio
  would favour $3 players who are "200% overpriced" and drain nothing).
- Probe it against the `control` (no hook) and the Run-O incumbent `drain_off_position`, under
  **Will's league settings**, CRN-paired, on the ADP nomination board.
- Produce a go/no-go on whether value-gap nomination lifts `reg_win_pct` for the shipped
  `overbid_noramp` plan.

## Non-goals

- Re-tuning the **bid** strategy. `overbid_noramp` is held fixed for every contestant so the probe
  isolates the *nomination* lift. (This is Run O's R5, re-applied.)
- The full `NominationStrategy` abstraction — built only on a "go".
- A live-draft strategy change. This is data-gathering; the guide's plan does not move on this run.

## Chosen approach

### The heuristics (new, in `nomination.py`)

Both read a new `NominationContext.hero_value_by_id` field (our `auction_dollars`, already built as
`val_by_id` in `_simulate_to_state` — it is passed, not recomputed).

- **`drain_value_gap`** -> `argmax(value_by_id[g] - hero_value_by_id[g])` over all candidates. The
  purest form of the hypothesis: nominate whoever the room most overvalues relative to us.
- **`drain_value_gap_off_position`** -> the same argmax restricted to candidates at a position the
  hero has already filled to its starter requirement; falls back to the unrestricted `drain_value_gap`
  when none qualifies. This composes the one thing Run O found that worked (off-position targeting)
  with the disagreement signal, and is the variant the hypothesis actually predicts should win.

`drain_max` and `drain_off_position` are retained unchanged as the Run-O incumbents.

### The seam

`NominationContext` gains `hero_value_by_id: Mapping[str, float]`. The existing `bot_by_id` guard in
`_simulate_to_state` (skip the O(pool) build when the hook is off) is mirrored: `val_by_id` is
already built unconditionally, so this costs nothing on the default path.

`run_auction_tournament` gains an optional `hero_nominators: Mapping[str, HeroNominator | None]`,
keyed by the same contestant names as `strategies`. This is what makes the probe CRN-paired **for
free**: every contestant already plays the identical auction draw (`default_rng(base_seed + s)`) and
identical season draw, and `run_auction_tournament` already returns `paired_diffs`. Racing four
*nominators* at one fixed bid is then the same shape as racing four bids — no bespoke pairing
arithmetic, and the Run-T/`auction_field_bakeoff.stratified_paired` aggregation applies verbatim.
The CRN-desync fix (`f9ccb0e` — always draw the central nominee, then override) is already in the
engine and is what makes this sound.

### The runner

`scripts/auction_nomination_probe.py`, deliberately a thin wrapper: it reuses
`auction_field_bakeoff.build_field` (the calibrated `overbidder` room) and its
`stratified_paired` / `_oriented` aggregation rather than reimplementing either. One process per
seat (crash-safe; the dev box's Raptor Lake fault wants bounded processes).

## Measurement

Will's league settings, taken from `reports/_will_bakeoff/jitter_2026/*.json` (the run behind the
committed guide), not re-derived:

| knob | value |
|---|---|
| league config | `configs/will_half12_pass5.league.json` |
| vorp table | `data/vorp_2026/will_half12.parquet` |
| field | `overbidder` (`overbid=0.2`, `pace=4.5`, basis `opening`, `pace_jitter=0.35`) |
| market | `espn` |
| nomination | `market_adp_jitter=12.0`, `nomination_temp=1.0` |
| seeds / sims | 20 seeds x 300 sims, `base_seed=0` |
| seats | all 12 |
| hero bid | `overbid_noramp` (`OverbidValueBid(use_urgency=False)`), fixed |

Contestants (all bidding `overbid_noramp`): `control` (hook None), `off_pos`
(`drain_off_position`, the Run-O incumbent), `gap` (`drain_value_gap`), `gap_off`
(`drain_value_gap_off_position`).

## Requirements

- **R1 — None is identity.** `hero_nominators=None`, and any contestant mapped to `None`, leaves
  every seat's roster and budget byte-identical to a no-hook run.
- **R2 — hero-only, non-forced-only.** Unchanged from Run O; the new heuristics ride the existing
  hook and inherit its scope.
- **R3 — validity by construction.** The engine's existing `override not in candidates` hard check
  covers the new heuristics.
- **R4 — heuristic correctness.** `drain_value_gap` returns the max-`(bot - ours)` candidate, which
  is **not** in general the max-`bot` candidate (unit-tested with a pool where those differ, so the
  test would fail if the heuristic silently degraded to `drain_max`).
  `drain_value_gap_off_position` returns the max-gap candidate at an over-filled position, and falls
  back to the unrestricted max-gap when none qualifies.
- **R5 — bid held fixed.** Every contestant uses `overbid_noramp`.
- **R6 — CRN keys must line up.** `hero_nominators` keys must be a subset of `strategies` keys; an
  unknown key raises rather than being silently ignored (a typo'd contestant name would otherwise
  produce a "control vs control" null result that reads as a clean no-effect finding).
- **R7 — sanity gate.** `control` must reproduce the committed `overbid_noramp` Will-league figure
  (`reports/_noramp_ab/espn.json`, seat-avg `reg_win_pct`) within seed noise. A mismatch means the
  harness is wrong; stop and fix before reading any lift.
- **R8 — gates.** `pytest`, `mypy src tests` (strict), `ruff check`, `ruff format --check` clean.

## Edge cases / failure modes

- **Backfire.** A max-gap player is by construction one the *room* prices highly, so it draws a bid;
  and it is by construction one *we* price low, so `resolve_unbid` handing it to us at `min_bid`
  costs a roster spot we did not want. `gap_off` additionally targets a filled position so
  `resolve_unbid` passes it on. Mitigation, not a guarantee — the win% measurement absorbs the
  residual.
- **Unpriced players.** In the ESPN market, players without an ESPN value are floored (`$1`
  unranked discount), which makes their gap strongly *negative* (our value minus ~nothing). Negative
  gaps sort to the bottom, so they are never nominated by these heuristics. Correct by construction,
  but worth stating: the heuristic will never nominate an ESPN-unpriced player.
- **Early draft.** `gap_off` falls back to `gap` before any position is filled — same shape as
  Run O's `drain_off_position` -> `drain_max` fallback.
- **Model market.** With `bot_prices="model"` the room prices off our own numbers, so the gap is
  identically zero (`bot_by_id is val_by_id`) and the heuristic degenerates to an arbitrary argmax
  tie-break. **This probe is therefore ESPN-only, and that is a property of the hypothesis, not a
  scoping shortcut** — "the room disagrees with us" has no meaning in a market defined to agree with
  us. Documented in the module and asserted in the runner.

## Go / no-go

Paired lift `Δ = mean over seats of (contestant − control)` on `reg_win_pct`, seat-stratified, with
a 95% CI (the `auction_field_bakeoff.stratified_paired` combination).

- **Go** -> some gap heuristic has a 95% CI on `Δ reg_win_pct` that **excludes zero**, positive, and
  is positive at a **majority of the 12 seats**. Then design the full nomination strategy slice.
- **No-go** -> record as data; `overbid_noramp` with no nomination hook stands as the guide's plan.

Single-market, so the pre-registered bar is a single-market bar — deliberately weaker than Run O's
both-markets criterion, and any adopted result inherits that weaker footing. Stated up front so the
verdict is not read as stronger than it is.
