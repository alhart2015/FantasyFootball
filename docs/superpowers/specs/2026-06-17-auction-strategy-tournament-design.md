# Auction Strategy Tournament — bid-model data-gathering harness (Auction Slice 1)

**Status:** Design (pre-plan), revised 2026-06-17.
**Decision posture — data-gathering only.** This harness exists to **gather reproducible data** on how the
candidate bid models compare. It does **not** declare a winning bid model. The single adopt decision is
made by the user **in ~September 2026**, close to the real draft, weighing the accumulated rows. Every run
records *what it favors in isolation* into `reports/auction_tournament_validation_2026.md`; no run is a
verdict. (Standing rule across the draft investigation — same as the snake side.)

**Companion tracking doc:** `reports/auction_tournament_validation_2026.md`.

## 0. Umbrella: the auction draft, the deferred seam now coming due

The snake **Draft Strategy Comparison Harness**
(`docs/superpowers/specs/2026-06-10-draft-strategy-tournament-design.md`) shipped a deliberate
*simulate → score → compare* split (§5.7) so that an auction equivalent could reuse the scoring and
bootstrap-statistics layers untouched and swap **only the draft-mechanism module** — snake pick order
becomes a nominate/bid loop. `LeagueConfig` already carries `budget` and `min_bid`, so no config work was
deferred onto this effort. This spec cashes in that seam.

It cashes in a **second** seam at the same time: the **projected-draft-eval**
(`docs/superpowers/specs/2026-06-16-projected-draft-eval-design.md`,
`src/projections/draft/assistant/league_projection.py`) shipped a *projected-vs-projected* full-league
simulation — `project_draft(rosters, …)` scores an entire league of rosters by Monte-Carlo season (variance
model + availability + optimal weekly lineups + a gauntlet schedule + a playoff bracket) and returns per-seat
**regular-season win %, make-playoffs %, bye %, championship %** (plus mean seed). That is exactly how we
now evaluate snake drafts, and it is the scoring layer this harness uses — **not** a single-roster scalar.

On the value side, `generate_auction_values` (`src/projections/draft/auction.py`) already converts the
consensus VORP table into one Surplus-of-Surplus (SOS) **dollar value per player**
(`AuctionValuesSchema.auction_dollars`). What we have *never* had is any way to tell whether bidding to
those static dollars produces a better *roster* in a live room than an inflation-aware or marginal-value
bid. This slice builds that measurement: the candidate **bid models become the contestants**, raced against
a shared seeded market, each producing a full projected league that is scored by the same projected-draft
evaluator the snake side uses.

## 1. Purpose

Gather reproducible data on **how three auction bid models compare** — (v1) bid to the static SOS dollar,
(v2) bid to an inflation-adjusted dollar, (v3) bid to the player's marginal lineup value — by simulating
realistic auctions (the hero seat runs the bid model under test against a field of value-rational
noisy-WTP bots), retaining **every seat's full roster**, and scoring the resulting projected league with
`project_draft`. For each model we record the hero seat's **expected season points** and **H2H win %,
playoff %, bye %, championship %**, with percentile-bootstrap confidence intervals over **paired** auction
realizations (same seed ⇒ same bot market). The output is rows in the tracking doc; **no winner is
declared here** (§5.1).

Same hard constraint as the snake harness, restated: the harness is **league-driven, not hardcoded**.
Team count, roster shape, scoring ruleset, **budget, and min-bid** all come from the `LeagueConfig` the
user already passes. Nothing assumes a 12-team, $200, PPR room.

## 2. Scope

### In scope (this slice)

- **`auction/bid_strategy.py`** — an `AuctionBidStrategy` `runtime_checkable` Protocol (mirroring
  `DraftStrategy`) and the three concrete bid models: `StaticDollarBid` (v1), `InflationBid` (v2),
  `MarginalValueBid` (v3, tournament surrogate — see §3.4).
- **`auction/market.py`** — the noisy-WTP **bot** bid policy (`bot_max_bid`), the auction analog of
  `opponent.bot_pick`; the field the hero bids against.
- **`auction/simulation.py`** — `simulate_auction`: one full nominate → bid → award auction, hero seat via
  an `AuctionBidStrategy`, all other seats via the bot, shared default nominator. **Returns the full league
  — every seat's roster** as `Mapping[int, list[GsisId]]` (the exact `rosters` input `project_draft`
  consumes), not just the hero's rows.
- **`auction/tournament.py`** — `run_auction_tournament`: run each bid model over many paired seeds,
  scoring every realization with `project_draft`, and record **per-model mean + bootstrap CI for each
  metric** plus the **paired per-seed differences** (also with CIs) as data. **Reuses** the
  bootstrap/Interval machinery from the snake tournament (promoted to a shared helper — §3.6) and
  **reuses `project_draft` / `team_weekly_points` / `gauntlet_schedule` verbatim** for scoring.
- **A `mean_points` field added to `SeatProjection`** (`league_projection.py`) so "expected season points"
  is a first-class recorded metric. Small, backward-compatible addition (the board may surface it too).
- **CLI** — `scripts/auction_tournament.py` → `assistant/cli.py` (or a sibling `auction/cli.py`): one
  `compare` mode over the registered bid models, printing the per-model per-metric table + paired diffs.
- Tests following the project's TDD + synthetic-fixture norms.

### Explicitly out of scope (later slices / other work)

- **Declaring a winning bid model.** This harness gathers data; the adopt decision is the user's, in
  September (§5.1). No code path emits "the winner is X."
- **Single-roster scalar scoring (`StartersValuer` / `RosterValuer`).** Dropped — the projected-draft-eval
  league sim is the only scorer (§5.7, §5.8). No `--valuer` switch.
- **Strategic nomination.** v1 fixes a single shared nominator ("nominate the highest remaining baseline
  dollar") for every seat, so the *only* variable across contestants is the bid model — the same way the
  snake harness holds the bot field fixed and varies only the hero. Nomination strategy is a real second
  axis and a deliberate future slice, not a confound smuggled into v1. §5.2.
- **Live auction UI / assistant.** This is the offline data-gathering harness. A live "what's my max bid
  right now" board over a chosen model is the analog of the snake Slice 3 and comes later.
- **Full-fidelity v3 in the harness.** v3's *principled* form prices a player by the marginal
  **expected-season** points it adds (a `SeasonValuer`-style MC per candidate per nomination). That is far
  too expensive inside an `n_seeds × ~n_slots·n_teams nominations × n_sims` loop. v3 ships with a **cheap
  optimal-lineup-marginal surrogate** (§3.4); the full-season-MC bid is reserved for the eventual
  single-decision *live* assistant. Load-bearing scope cut, called out so the harness is runnable — §5.4.
- **Behavioral / adversarial bots.** Bots are value-rational noisy-WTP (the market proxy), not
  bluffing/sniping/budget-baiting agents — the same honest-but-simple stance as the snake harness's pure
  noisy-ADP bots (the human-opponent model is the biggest realism lever, logged for the snake side in
  TODO #46). §5.5.
- **Keeper / dynasty auctions, in-auction trades, FAAB.** None modeled.
- **New pandera schema.** The scorer returns `SeatProjection` dataclasses; the harness records floats
  rendered by the CLI / pasted into the tracking doc (§5.6); `generate_auction_values` already owns
  `AuctionValuesSchema`.

## 3. Design

### 3.1 Inputs

The same consensus VORP pool the snake harness and the auction-value generator already consume, plus the
availability/variance inputs the projected-draft-eval already loads. The harness adds no ingest or
schema-producing path.

- **Pool** — a consensus VORP table (`VorpTableSchema`) parquet: `gsis_id`, `position`,
  `season_mean_fpts` (the roster-scoring currency), `vorp`. **`consensus_adp` is *not* required** here —
  an auction has no draft order, so the all-null-ADP hard error from the snake harness does **not** carry
  over. The market signal is **dollars**, derived below. The pool must carry **`is_rookie`** for the
  variance sampler (the controller attaches it via the projected-draft-eval's `_attach_is_rookie`; the
  harness reuses that path).
- **Baseline dollars** — `generate_auction_values(pool, config)` produces the **full `AuctionValuesSchema`
  frame** (one `auction_dollars` per player under the league's budget/min-bid/roster shape, plus `in_pool`,
  `season_mean_fpts`, `vorp`). **This whole frame — not a bare gsis→dollar mapping — is the shared
  currency** threaded as `baseline_dollars`: v1 bids straight to `auction_dollars`, v2 re-prices it by
  inflation (and needs `in_pool` for its surplus-value sum), v3 uses it for the market exchange rate, and
  the bots center their WTP on it. Computed **once** at the harness entry and threaded in (it is
  config-determined and seed-independent — recomputing per seed would be wasted work and a determinism
  footgun).
- **Availability + variance params** — `availability` (`load_store_availability(pool, season, data_root)`)
  and `VarianceParams`, the same inputs `project_league_outcomes` loads; required because scoring is the
  league sim. A **`--season`** is therefore required (it was optional in the pre-revision draft).
- **`LeagueConfig`** — `n_teams`, `roster_slots`, `ruleset`, **`budget` (per-team), `min_bid`**. Same
  documented precondition as the snake harness: it must be the config the VORP table was generated under
  (the parquet can't self-verify the ruleset). `total_budget = n_teams · budget` and `roster_size`
  (drafted slots, IR-excluded) come straight off the model.
- **Preconditions (both validated up front):**
  - **Pool sufficiency.** A full auction fills `n_teams · roster_size` roster spots; the pool must hold at
    least that many players. Re-use the snake harness's `_validate_pool` **size arm** (promoted alongside
    the stats helpers in §3.6); the all-null-ADP arm is snake-only and stays there. The size check counts
    **all** VORP-table rows, and the nominator/bid range is the **whole** baseline-dollars frame (`in_pool`
    *and* not — out-of-pool rows are `min_bid` bench fillers); so every drafted `gsis_id` is guaranteed
    present in the scoring `pool`, and `project_draft` never sees an id absent from the table.
  - **Budget solvency.** **`config.budget ≥ config.min_bid · config.roster_size`** — otherwise a seat
    cannot afford `min_bid` for every slot and the auction cannot complete (the `feasible_max` endgame
    invariant in §3.2 is preserved inductively but only holds if it holds at kickoff). This is the same
    condition as `generate_auction_values`' surplus ≥ 0. Validate it explicitly and error clearly.

### 3.2 The auction state (`auction/simulation.py`, internal)

A small **mutable** dataclass, internal to the simulation (not a schema, not persisted — the analog of the
snake sim's local `drafted` / `my_roster` bookkeeping):

```
AuctionState:
    budgets:   list[int]                      # remaining $ per seat, indexed 0..n_teams-1
    rosters:   list[list[(GsisId, Position, int)]]   # (player, position, price paid) per seat
    drafted:   set[GsisId]
    nominator: int                            # whose turn to nominate
```

Two derived quantities the whole engine leans on, both read off `roster_slots`:

- **`open_slots(seat) = roster_size − len(rosters[seat])`** — spots left to fill.
- **`feasible_max(seat) = budgets[seat] − min_bid · (open_slots(seat) − 1)`** — the most a seat can bid on
  the current player while still affording `min_bid` for every *remaining* spot. This single invariant
  makes the endgame fall out for free: a seat with `budget == min_bid · open_slots` has `feasible_max ==
  min_bid` and can only take $1 players the rest of the way. **The engine enforces `feasible_max` as a
  hard clamp on every bid** (hero and bot alike), so no bid model has to re-implement the reserve — it
  returns its *desired* max and the engine clamps. A seat with `open_slots == 0` is out of the auction.

At auction end, the state's `rosters` are projected to the `project_draft` input shape:
`{seat_index + 1: [gsis_id, …]}` for **every** seat (§3.6).

**Seat indexing — one convention, one conversion point.** `my_seat` — and the CLI `--my-seat N` — is
**1-based** (`1..n_teams`), matching the snake `my_slot`, `project_draft`'s 1-based keys, and the
run-param validation. The internal `AuctionState` lists (`budgets`, `rosters`) are **0-based**
(`0..n_teams-1`). The **sole** conversion point: building the hero's `AuctionView` reads
`rosters[my_seat - 1]` / `budgets[my_seat - 1]`, while the returned league dict and the scored projection
stay 1-based and are read directly (`proj[my_seat]`, no offset). A guard test pins this (§4): the hero's
roster in the returned league at key `my_seat` must be exactly the roster scored at `proj[my_seat]`.

### 3.3 Legality and positional need

Legality is **"the seat has an open slot,"** nothing more. Under a shared bench, `bench_eligible_positions`
holds every rostered position until a roster is *full*, so any player a seat can afford fits *some* slot
until the seat is full. **Positional need is a matter of valuation, not legality** — a seat that already
has its starters will value a third RB low (v3's marginal is ~0; v1/v2's static dollar is unchanged but the
bot's interest noise can drop it), and that low valuation, not a hard rule, is what stops it from
overpaying. This keeps the mechanism simple and puts all the positional intelligence in the contestant.
Slot eligibility, when needed (e.g. for the marginal surrogate's lineup fill), reuses `roster_eligibility`'s
`FLEX_SLOTS` / `bench_eligible_positions` — one source of truth.

### 3.4 The three contestants (`auction/bid_strategy.py`)

```
@runtime_checkable
class AuctionBidStrategy(Protocol):
    def max_bid(self, view: AuctionView, player: pd.Series,
                pool: pd.DataFrame, config: LeagueConfig) -> int: ...
```

`AuctionView` is a **read-only** projection of `AuctionState` for the hero seat: `my_budget`,
`my_open_slots`, `my_positions` (Counter), `my_roster` (rows, for v3's marginal), `drafted` (set),
`budgets_by_seat` (for inflation), `baseline_dollars` (**the full §3.1 `AuctionValuesSchema` frame**). The
engine clamps the returned value to `feasible_max` (§3.2), so a model may return an "I'd go to $X" number
without tracking the reserve itself. All three vary **only the valuation** — same nominator, same bot field,
same clamp — so a measured difference is attributable to the bid model alone.

- **v1 `StaticDollarBid`** — `max_bid = baseline_dollars[player]`. Bid to the static SOS value, full stop.
  The control: exactly "trust `generate_auction_values`." Cheap, deterministic, no state.
- **v2 `InflationBid`** — re-price the static dollar by live **surplus inflation**, mirroring the SOS
  structure (`auction.py` reserves `min_bid` then distributes surplus):
  - `remaining_surplus_money = Σ_seats budgets[seat] − min_bid · Σ_seats open_slots(seat)`, where
    `Σ_seats open_slots(seat) = n_teams · roster_size − |drafted|` (derivable from the view + config; no
    per-seat open-slots field needed).
  - `remaining_surplus_value = Σ_{undrafted in-pool p} (baseline_dollars[p] − min_bid)` (uses the frame's
    `in_pool` flag).
  - `inflation = remaining_surplus_money / remaining_surplus_value` (guarded: if the denominator ≤ 0 —
    only the dregs left — fall back to `1.0`).
  - `max_bid = min_bid + (baseline_dollars[player] − min_bid) · inflation`.
  Recomputed each nomination. Overpay early ⇒ money shrinks faster than value ⇒ inflation < 1 (discipline
  late); underspend ⇒ inflation > 1 (the remaining studs carry a premium). One scalar re-pricing the whole
  board — the textbook inflation adjustment in the same surplus units the generator uses.
- **v3 `MarginalValueBid`** — bid to the player's **marginal lineup value**, converted to dollars at the
  live market exchange rate:
  - `lift = optimal_lineup_points(my_roster + player) − optimal_lineup_points(my_roster)` — the
    optimal-starting-lineup points the player adds to *my* current roster (reuses `optimal_lineup_points`;
    restrictive-slot-first fill, FLEX before SUPER_FLEX). **This is the tournament surrogate** (§2): cheap,
    deterministic, no MC.
  - `points_per_dollar = remaining_surplus_value_points / remaining_surplus_money`, where
    `remaining_surplus_value_points = Σ_{undrafted in-pool p}` marginal-lineup-lift of `p` to an empty
    slot — how many lineup points a surplus dollar buys at the current board (computed once per nomination
    over the board, not per bidder).
  - `max_bid = min_bid + lift / points_per_dollar` (guarded for `points_per_dollar ≤ 0`).
  A player who doesn't crack my starting lineup has `lift == 0` and so is bid at `min_bid`. **Known
  limitation, stated plainly:** this surrogate is single-week-optimal and availability-blind, so it
  under-values *depth* (a backup that only matters via injury weeks scores 0 marginal). The depth-aware
  version uses a `SeasonValuer` marginal (`marginal_season_values`, which credits depth via availability)
  and is exactly the full-fidelity v3 deferred to the live assistant (§2). **Note the bid/score mismatch is
  intentional, not circular:** v3 *bids* on the cheap starters surrogate but is *graded* on the
  full-season, availability-aware, H2H league sim (§3.6) — so v3 is not in-sample-optimizing its own grade
  (the circularity the snake-side projected-eval flagged in TODO #48 does not arise here).

### 3.5 The market — noisy-WTP bots (`auction/market.py`)

A bot occupies every non-hero seat. Its willingness-to-pay is value-rational with multiplicative noise —
the auction analog of `opponent.bot_pick`'s noisy ADP:

```
bot_max_bid(seat_view, player, baseline_dollars, config, rng, *, price_jitter) -> int
```

1. If the seat has no open slot, it does not bid (returns 0 / abstains).
2. Center on the market value: `wtp = baseline_dollars[player] · (1 + rng.normal(0, price_jitter))`,
   floored at `min_bid` (read from `config`), rounded to an int.
3. The engine clamps `wtp` to the seat's `feasible_max` (§3.2), so a bot can never bid itself out of a
   fillable roster.

`config` supplies `min_bid` (and `roster_slots`/`n_teams` for the seat-view derivations), mirroring the
hero's `max_bid(view, player, pool, config)`; the bot carries no model state of its own.

`price_jitter` (fractional WTP spread) is the auction analog of the snake harness's `adp_jitter` — the
single market-noise knob — and is **distinct from any model parameter** the contestants carry. Realism, as
on the snake side, comes from the market values themselves (SOS dollars already encode positional scarcity),
not from a roster-construction rule. Default `price_jitter` lives in a named constant; bots take no
inflation/marginal logic — they *are* the static-value market the contestants are measured against.

**Bid resolution — second-price + min_bid (English clearing).** Given the nominee, collect every eligible
seat's clamped max bid (hero via its `AuctionBidStrategy`, bots via `bot_max_bid`). The winner is the
**argmax** max-bid; the **price paid** is `min(winner_max, second_highest_max + min_bid)`, i.e. one tick
over the runner-up's ceiling, never more than its own — the standard ascending-auction clearing price,
deterministic given the max-bids (no live increment loop). With a lone eligible bidder, the price is
`min_bid`. Ties in max-bid break deterministically (seat index, then `gsis_id`), so same seed ⇒ same
outcome.

### 3.6 Simulation, scoring, and the paired comparison

```
simulate_auction(strategy: AuctionBidStrategy, my_seat: int, pool, config,
                 *, baseline_dollars, price_jitter, rng) -> dict[int, list[GsisId]]   # ALL seats
```

The loop, until every seat's roster is full:

1. **Nominate.** Advance `nominator` in seat order, skipping full seats; the active nominator nominates the
   **highest baseline-dollar undrafted player** (shared default nominator — §5.2). The nominee is
   nominator-independent under this default, but the pointer still governs *whose* turn it is, so strategic
   nomination drops in later without reshaping the loop.
2. **Bid.** Collect clamped max-bids from every eligible seat (hero via `strategy`, others via
   `bot_max_bid`); resolve the winner and price per §3.5.
3. **Award.** Subtract the price from the winner's budget, append `(gsis_id, position, price)` to its
   roster, add to `drafted`.

One seeded `numpy.random.Generator` drives **all** bot WTP noise for the whole auction, so — exactly as in
the snake harness — same seed + same hero strategy ⇒ byte-identical league (a tested invariant), and two
strategies at the same seed face the **same market realization** up to where the hero's own diverging wins
perturb budgets/board downstream (the paired "same room, different me" counterfactual; the pairing is clean
pre-divergence and weakens gracefully after — same property the snake harness has). The function returns
**every seat's roster** as `{seat: [gsis_id, …]}` — the exact `rosters` input `project_draft` consumes, so
scoring is reused with **zero** changes.

**Scoring — the projected-draft-eval league sim.** Each realization's full league is scored by
`project_draft(rosters, pool, availability, params, *, league_config, n_sims, rng=season_rng)` →
`{seat: SeatProjection}`; the harness reads the **hero seat** and records its metrics: **expected season
points** (`mean_points`, the new field), **win %**, **playoff %**, **bye %**, **champ %**.

- **Common random numbers on the season MC.** Define `season_rng(s) = default_rng(season_base_seed + s)`,
  keyed off the *auction* seed `s` and **shared across all strategies at seed `s`**, so the season-sim
  noise largely cancels in the paired differences (most of the league overlaps between strategies; only the
  hero's diverging wins differ). This makes a modest `n_sims` cheap. `season_base_seed` is a
  `run_auction_tournament` parameter defaulting to **`base_seed + 1_000_000`** — a stream disjoint from the
  auction market RNG (`default_rng(base_seed + s)`): the two sequences `{base_seed + s}` and
  `{season_base_seed + s}` cannot overlap for any realistic `n_seeds`, so market noise and season noise stay
  independent. (Reusing `base_seed` for both would couple the market draw and the season draw at each seed —
  the footgun this default avoids.)
- **`n_sims` budget.** Default **~500** (tunable). 2000 — the board default — is reserved for a post-hoc
  deep-dive on a roster of interest, not for the in-loop sweep. At `n_sims≈500`, `3 models × ~200 seeds` is
  a few minutes.

```
run_auction_tournament(strategies: Mapping[str, AuctionBidStrategy], pool, config, *, my_seat, n_seeds,
price_jitter, base_seed, n_sims, availability, params,
season_base_seed=base_seed + 1_000_000) -> AuctionTournamentResult
```

For each strategy and seed `s`: `rng = default_rng(base_seed + s)` (same `s` ⇒ same market across
strategies), `league = simulate_auction(..., rng=rng)`, `proj = project_draft(league, …,
rng=season_rng(s))` with `season_rng(s) = default_rng(season_base_seed + s)` (shared across strategies),
`metrics[strategy][s] = proj[my_seat]` (`my_seat` 1-based, read directly — §3.2). Then:

- **Per-strategy, per-metric mean + percentile-bootstrap CI** (reusing the promoted `_bootstrap_mean` /
  `Interval`).
- **Paired per-seed differences** for the top-of-interest pairs, **per metric**, also as `Interval`s —
  **recorded as data, not used to crown a winner.** (`AuctionTournamentResult` carries the full per-metric
  means/CIs and paired diffs; the CLI prints them; the rows go in the tracking doc. No "winner = …"
  label is emitted — §5.1.)

**Bot-field caveat (load-bearing here).** Win/playoff/bye/champ % are **relative to the noisy-WTP bot
field** — unlike a single-roster scalar, which scored the hero in a vacuum. The paired design keeps the
*ranking across models* robust to bot mediocrity (same bots per seed), but the **absolute** numbers are
only as meaningful as the bots. Bot realism remains the biggest realism lever (shared with the snake-side
follow-up, TODO #46). Champ % is also the **noisiest** metric (≈`1/n` base rate); expected points and
playoff % separate more readily.

**Reuse, not re-implement (the project bar).** `_bootstrap_mean`, `Interval`, and the size arm of
`_validate_pool` already live in `assistant/tournament.py`. Promote the strategy-agnostic pieces —
`Interval`, `_bootstrap_mean`, and the pool **size** check — into a shared `assistant/_compare.py` that
**both** the snake and auction harnesses import. The snake `run_tournament` keeps its ADP-specific arms
(`adp_jitter`, all-null-ADP check) **and its winner-labeling** (the snake harness declares a winner; this
one does not). The auction harness gets a sibling `AuctionTournamentResult` (per-metric `Interval` fields;
`my_slot → my_seat`, `adp_jitter → price_jitter`, plus `budget`/`min_bid`/`n_sims` echoed for
reproducibility). Scoring reuses `project_draft` / `team_weekly_points` / `gauntlet_schedule` unchanged
(aside from the `mean_points` field addition).

### 3.7 CLI surface

`scripts/auction_tournament.py` → engine:

```
python scripts/auction_tournament.py \
    --vorp-table <consensus_vorp.parquet> \
    --league-config <league.json> \
    --my-seat N \
    --season YYYY \
    [--seeds K] [--price-jitter F] [--seed BASE] [--n-sims M] \
    compare
```

`compare` runs the three registered bid models (`static`, `inflation`, `marginal`) and prints, **per
model**, the mean + 95% CI for each metric (exp pts / win % / playoff % / bye % / champ %) and the **paired
per-seed differences** — formatted to paste straight into `reports/auction_tournament_validation_2026.md`.
**No winner line.** `--league-config` is the single source of roster shape + team count + ruleset + budget
+ min-bid (matching `generate_vorp_table`). `--season` selects the availability/bye data for the league
sim. `--seed` makes any run reproducible. Defaults cover the tuning knobs (`--seeds`, `--price-jitter`,
`--seed`, `--n-sims`); `--vorp-table`, `--league-config`, `--my-seat`, `--season` are required. CLI imports
the engine, never the reverse.

## 4. Testing

Synthetic fixtures only (project norm — no network, no real parquet in unit tests):

- **Preconditions** — pool-size check fires when the pool is too small; **budget-solvency check fires when
  `budget < min_bid · roster_size`** with a clear message; both pass on a valid config.
- **Reserve / feasible-max invariant** — a seat with `budget == min_bid · open_slots` can only bid
  `min_bid`; a seat with one slot left can bid its whole budget; the engine clamps an over-budget desired
  bid down to `feasible_max`. Pin the endgame-$1 path directly.
- **Bid resolution** — winner is the argmax max-bid and pays `second_highest + min_bid` (not its own
  ceiling); a lone bidder pays `min_bid`; a max-bid tie breaks deterministically (seat, then `gsis_id`).
- **Full-auction conservation** — every seat ends with exactly `roster_size` players; `Σ prices ≤
  total_budget`; no player drafted twice; no seat ends with `open_slots > 0` while it still had
  `feasible_max ≥ min_bid` and an affordable player existed (the auction completes); `simulate_auction`
  returns a roster list for **every** seat (the full-league contract `project_draft` needs).
- **Determinism & paired market** — same `base_seed` + same strategy ⇒ identical league *and* identical
  recorded metrics (the season MC is seeded by `season_rng(s)`); different seed ⇒ generally different; two
  strategies at the same seed see identical bot WTPs until the hero's first diverging win (the paired
  counterfactual holds).
- **Bot policy** — `bot_max_bid` centers on `baseline_dollars` and with `price_jitter → 0` equals the
  baseline (clamped to `feasible_max`); a full seat abstains.
- **v1 `StaticDollarBid`** — returns the player's baseline dollar (pre-clamp); after clamp never exceeds
  `feasible_max`.
- **v2 `InflationBid`** — on a hand-built state where the room has overspent, inflation < 1 and the bid
  drops below baseline; underspent ⇒ inflation > 1 and it rises; the degenerate
  `remaining_surplus_value ≤ 0` falls back to factor 1.0 (no div-by-zero).
- **v3 `MarginalValueBid`** — a player who improves the optimal lineup bids above `min_bid`; a player who
  doesn't crack it (`lift == 0`) bids exactly `min_bid`; `points_per_dollar ≤ 0` is guarded. Reuses
  `optimal_lineup_points` (FLEX-before-SUPER_FLEX covered transitively; re-assert one case to pin wiring).
- **Scoring wiring** — a completed `simulate_auction` league feeds `project_draft` unchanged and yields a
  `SeatProjection` for `my_seat`; `SeatProjection.mean_points` is present and finite; a stronger hero
  roster scores higher expected points / champ % than a deliberately weak one (sanity of the seam).
- **Seat-index convention (off-by-one guard)** — for a `my_seat` that is *not* the first seat, the hero
  roster returned at `league[my_seat]` (1-based) equals the roster the engine built for the hero, and is
  the same one scored at `proj[my_seat]`; assert the hero's `AuctionView` was built from `rosters[my_seat-1]`
  (0-based state) — i.e. the 1-based/0-based conversion (§3.2) is applied at exactly one point. (Pins the
  latent off-by-one between 0-based `AuctionState` and 1-based `project_draft` keys.)
- **Recorded comparison (no verdict)** — a synthetic fixture where one model holds a constant per-seed edge
  ⇒ the paired-diff CI excludes 0 **and the result records that diff** (the harness reports it; it does
  **not** label a "winner"); a zero-edge fixture ⇒ a diff CI bracketing 0. (Reuses the promoted `_compare`
  helper; the snake harness's existing stat tests cover the shared core; add one auction smoke.)
- **League-driven** — run the harness under two different `roster_slots` **and** two different
  `budget`/`min_bid` configs on the same pool; both complete and score with no code change (the guard
  against any hardcoded $200 / 12-team / position-count assumption).
- **CLI smoke** — `compare` runs end-to-end on a fixture parquet + config and prints the per-model
  per-metric table with no winner line.

All gates per the project bar: `pytest -v`, `mypy src tests` (strict), `ruff check src tests`,
`ruff format --check src tests`; plus `pytest -v -k "ingest or store or schemas"` if any schema/store path
is touched (it should not be — no new schema; `mean_points` is a dataclass field, not a pandera change).

## 5. Key decisions

- **5.1 Data-gathering, not a winner declaration.** The user builds all three models and gathers data on
  how they compare; the single adopt decision is the user's, in **September 2026**, weighing the
  accumulated rows in `reports/auction_tournament_validation_2026.md`. The harness therefore **records**
  per-model metrics + CIs + paired diffs and emits **no "winner"**. This mirrors the standing rule on the
  snake side (rankings swing by season and methodology; each run is one data point).
- **5.2 Isolate the variable: one shared nominator, vary only the bid.** All three contestants share the
  "nominate highest remaining dollar" nominator and the same bot field, so a measured difference is
  attributable to the bid model alone. Strategic nomination is a real but *separate* axis, deferred (§2).
- **5.3 Second-price + min_bid clearing, not a live increment loop.** Given each seat's max-bid, the
  ascending-auction outcome is determined: the top bidder wins at one tick over the runner-up. Simulating
  per-dollar increments would add loop cost and RNG surface for an identical result. Deterministic given
  seed → reproducible.
- **5.4 v3 ships as a cheap starters-marginal surrogate; full-season-MC v3 is the live model.** A
  per-candidate `SeasonValuer` MC inside the `seeds × nominations × sims` loop is computationally infeasible
  at scale (the snake side already learned the per-pick-MC cost lesson — a naive per-pick MC was ≈9 hrs/slot
  before the numpy fast-path). The optimal-lineup-marginal surrogate makes the harness runnable now; the
  richer availability-aware marginal is reserved for the once-per-real-pick live assistant. Because grading
  is the full-season league sim (5.8), the surrogate-bids/season-scored mismatch is an honest cross-check,
  not circularity.
- **5.5 Bots are value-rational noisy-WTP, not behavioral.** Realism comes from the SOS dollars (which
  already encode scarcity) plus one multiplicative noise knob — the exact stance (and honest caveat) as the
  snake harness's noisy-ADP bots. With league-sim scoring this caveat is **load-bearing**: H2H metrics are
  relative to the bot field, so the *ranking* is paired-robust but *absolute* numbers are bot-relative
  (§3.6). A better human-opponent model is the single biggest realism lever (TODO #46); the auction harness
  inherits that backlog item rather than pretending to solve it.
- **5.6 No new schema.** `generate_auction_values` owns `AuctionValuesSchema`; `project_draft` returns
  `SeatProjection` dataclasses; the harness records floats. `mean_points` is a new dataclass field on
  `SeatProjection`, not a pandera change. A schema would be ceremony — the call the snake harness made.
- **5.7 Reuse the scoring and stats layers verbatim; promote the shared core.** Both seams pay off:
  `project_draft` / `team_weekly_points` / `gauntlet_schedule` (scoring) and the bootstrap/`Interval` stats
  are mechanism-agnostic by design. This slice promotes `Interval` / `_bootstrap_mean` / the pool-size
  check into a shared `_compare.py` and reuses the league sim — "obviously a later effort," not a rewrite.
- **5.8 Score by the projected-vs-projected league sim; drop single-roster scalar scoring.** Per user
  decision, the harness scores rosters the same way we evaluate snake drafts — full-league `project_draft`
  (exp pts + H2H win/playoff/bye/champ %) — and **does not** use `StartersValuer` / a `RosterValuer` scalar
  (no added value over the league sim, and it would re-introduce v3 circularity). The H2H / championship
  metrics are what the September decision will weigh.

## 6. Open questions / future slices

- **Strategic nomination axis** — dump-the-enemy's-budget, price-enforcement, nominate-your-target-late;
  the natural Slice 2 once the bid-model data matures (the bid model becomes the fixed baseline, and
  nominators become the contestants).
- **Full-fidelity v3 in a live auction assistant** — the availability-aware `SeasonValuer` marginal as a
  "max bid right now" recommender, evaluated once per real pick; the auction analog of the snake Slice 3
  board.
- **Better bot market** — a learned or behavior-calibrated WTP model (shared with the snake-side TODO #46
  human-opponent effort) so the absolute metrics reflect a real room, not a value-rational one.
- **Seat sweep** — average the hero across all seats for a seat-robust read (the auction analog of the
  snake hero-slot sweep), if a model proves seat-sensitive.
- **Align the snake tournament's scoring** — the snake `run_tournament` still scores with a scalar valuer;
  moving it onto `project_draft` (to match this harness and the projected-draft-eval) is a possible future
  consistency pass, out of scope here.
- **Budget-curve diagnostics** — persist per-seed spend-by-round / stars-and-scrubs vs balanced shape of
  each model; earns a stored schema only if a consumer needs the history (per §5.6).
