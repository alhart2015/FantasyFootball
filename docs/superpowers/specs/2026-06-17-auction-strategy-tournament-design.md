# Auction Strategy Tournament — which bid model wins (Auction Slice 1)

## 0. Umbrella: the auction draft, the deferred seam now coming due

The snake **Draft Strategy Comparison Harness**
(`docs/superpowers/specs/2026-06-10-draft-strategy-tournament-design.md`) shipped a deliberate
*simulate → score → compare* split (§5.7) so that an auction equivalent could reuse the roster-scoring
and bootstrap-statistics layers untouched and swap **only the draft-mechanism module** — snake pick order
becomes a nominate/bid loop. `LeagueConfig` already carries `budget` and `min_bid`, so no config work was
deferred onto this effort. This spec cashes in that seam.

On the value side, `generate_auction_values` (`src/projections/draft/auction.py`) already converts the
consensus VORP table into one Surplus-of-Surplus (SOS) **dollar value per player**
(`AuctionValuesSchema.auction_dollars`). What we have *never* had is any way to tell whether bidding to
those static dollars is the right *behavior* in a live room, or whether an inflation-aware or
marginal-value bid does better. This slice builds that test: the candidate **bid models become the
contestants**, raced against a shared seeded market and scored by the same optimal-lineup valuer the snake
harness uses.

## 1. Purpose

Answer, with a reproducible number: **which auction bid model maximizes the hero's roster value** —
(v1) bid to the static SOS dollar, (v2) bid to an inflation-adjusted dollar, or (v3) bid to the player's
marginal lineup value? Do it by simulating realistic auctions — the hero seat runs the bid model under
test against a field of value-rational noisy-WTP bots — scoring each completed roster by the points it
would actually start, and comparing models on **paired auction realizations** (same seed ⇒ same bot
market) with a percentile-bootstrap confidence interval, exactly as the snake harness compares snake
strategies. The harness is the empirical backbone the (future) live auction assistant needs before any of
the three bid models is trusted.

Same hard constraint as the snake harness, restated: the harness is **league-driven, not hardcoded**.
Team count, roster shape, scoring ruleset, **budget, and min-bid** all come from the `LeagueConfig` the
user already passes. Nothing assumes a 12-team, $200, PPR room.

## 2. Scope

### In scope (this slice)

- **`auction/bid_strategy.py`** — an `AuctionBidStrategy` `runtime_checkable` Protocol (mirroring
  `DraftStrategy` / `RosterValuer`) and the three concrete bid models: `StaticDollarBid` (v1),
  `InflationBid` (v2), `MarginalValueBid` (v3, tournament surrogate — see §3.4).
- **`auction/market.py`** — the noisy-WTP **bot** bid policy (`bot_max_bid`), the auction analog of
  `opponent.bot_pick`; the field the hero bids against.
- **`auction/simulation.py`** — `simulate_auction`: one full nominate → bid → award auction, hero seat via
  an `AuctionBidStrategy`, all other seats via the bot, shared default nominator; returns the hero's final
  roster (a sub-frame of the pool — identical contract to `simulate_draft`).
- **`auction/tournament.py`** — `run_auction_tournament`: compare bid models over many paired seeds, mean
  + bootstrap CI, paired-difference winner test. **Reuses** the bootstrap/winner machinery from the snake
  tournament (promoted to a shared helper — §3.6), and **reuses `RosterValuer` / `StartersValuer`
  verbatim** for scoring.
- **CLI** — `scripts/auction_tournament.py` → `assistant/cli.py` (or a sibling `auction/cli.py`): one
  `compare` mode over the registered bid models.
- Tests following the project's TDD + synthetic-fixture norms.

### Explicitly out of scope (later slices / other work)

- **Strategic nomination.** v1 fixes a single shared nominator ("nominate the highest remaining baseline
  dollar") for every seat, so the *only* variable across contestants is the bid model — the same way the
  snake harness holds the bot field fixed and varies only the hero. Nomination strategy (dump-the-enemy's
  budget, price-enforce, nominate-your-target-when-others-are-broke) is a real second axis and a
  deliberate future slice, not a confound smuggled into v1. §5.2.
- **Live auction UI / assistant.** This is the offline bake-off. A live "what's my max bid right now"
  board over the winning model is the analog of the snake Slice 3 and comes later.
- **Full-fidelity v3 in the tournament.** v3's *principled* form prices a player by the marginal
  **expected-season** points it adds (a `SeasonValuer`-style MC per candidate per nomination). That is far
  too expensive inside an `n_seeds × ~n_slots·n_teams nominations × n_sims` loop. v3 ships in the
  tournament with a **cheap optimal-lineup-marginal surrogate** (§3.4); the full-season-MC bid is reserved
  for the eventual single-decision *live* assistant, where it is evaluated once per real pick. This is a
  load-bearing scope cut, called out so the bake-off is actually runnable — §5.4.
- **Behavioral / adversarial bots.** Bots are value-rational noisy-WTP (the market proxy), not
  bluffing/sniping/budget-baiting agents — the same honest-but-simple stance as the snake harness's pure
  noisy-ADP bots, and the same caveat (the human-opponent model is the biggest realism lever, already
  logged for the snake side in TODO #46). §5.5.
- **Keeper / dynasty auctions, in-auction trades, FAAB.** None modeled.
- **New pandera schema.** As in the snake harness, the result is a handful of floats rendered by the CLI
  (§5.6); `generate_auction_values` already owns `AuctionValuesSchema`.

## 3. Design

### 3.1 Inputs

The same consensus VORP pool the snake harness and the auction-value generator already consume. The
harness adds no ingest or schema-producing path.

- **Pool** — a consensus VORP table (`VorpTableSchema`) parquet: `gsis_id`, `position`,
  `season_mean_fpts` (the roster-scoring currency), `vorp`. **`consensus_adp` is *not* required** here —
  an auction has no draft order, so the all-null-ADP hard error from the snake harness does **not** carry
  over. The market signal is **dollars**, derived below.
- **Baseline auction dollars** — `generate_auction_values(pool, config)` produces one
  `auction_dollars` per player under the league's budget/min-bid/roster shape. This single table is the
  shared currency: v1 bids straight to it, v2 re-prices it by inflation, v3 uses it only for the
  market exchange rate, and the bots center their WTP on it. Computed **once** at the tournament entry and
  threaded in (it is config-determined and seed-independent — recomputing per seed would be wasted work
  and a determinism footgun).
- **`LeagueConfig`** — `n_teams`, `roster_slots`, `ruleset`, **`budget`, `min_bid`**. Same documented
  precondition as the snake harness: it must be the config the VORP table was generated under (the parquet
  can't self-verify the ruleset). `total_budget = n_teams · budget` and `roster_size` (drafted slots,
  IR-excluded) come straight off the model.
- **Pool sufficiency.** A full auction fills `n_teams · roster_size` roster spots; the pool must hold at
  least that many *rosterable* players. Re-use the snake harness's `_validate_pool` size check (promoted
  alongside the stats helpers in §3.6); the all-null-ADP arm is snake-only and stays there.

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

### 3.3 Legality and positional need

Legality is **"the seat has an open slot,"** nothing more. Under a shared bench, `bench_eligible_positions`
holds every rostered position until a roster is *full* (the same fact the snake harness leans on, §5.5
there), so any player a seat can afford fits *some* slot until the seat is full. **Positional need is a
matter of valuation, not legality** — a seat that already has its starters will value a third RB low
(v3's marginal is ~0; v2/v1's static dollar is unchanged but the bot's interest noise can drop it), and
that low valuation, not a hard rule, is what stops it from overpaying. This keeps the mechanism simple and
puts all the positional intelligence where it belongs: in the contestant. Slot eligibility, when needed
(e.g. for the marginal surrogate's lineup fill), reuses `roster_eligibility`'s `FLEX_SLOTS` /
`bench_eligible_positions` — one source of truth, no second copy of "what can play FLEX."

### 3.4 The three contestants (`auction/bid_strategy.py`)

```
@runtime_checkable
class AuctionBidStrategy(Protocol):
    def max_bid(self, view: AuctionView, player: pd.Series,
                pool: pd.DataFrame, config: LeagueConfig) -> int: ...
```

`AuctionView` is a **read-only** projection of `AuctionState` for the hero seat: `my_budget`,
`my_open_slots`, `my_positions` (Counter), `drafted` (set), `budgets_by_seat` (for inflation),
`baseline_dollars` (the §3.1 table). The engine clamps the returned value to `feasible_max` (§3.2), so a
model may return an "I'd go to $X" number without tracking the reserve itself. All three vary **only the
valuation** — same nominator, same bot field, same clamp — so the comparison isolates the bid model.

- **v1 `StaticDollarBid`** — `max_bid = baseline_dollars[player]`. Bid to the static SOS value, full stop.
  The control: it is exactly "trust `generate_auction_values`." Cheap, deterministic, no state.
- **v2 `InflationBid`** — re-price the static dollar by live **surplus inflation**, mirroring the SOS
  structure (`auction.py` reserves `min_bid` then distributes surplus):
  - `remaining_surplus_money = Σ_seats budgets[seat] − min_bid · Σ_seats open_slots(seat)`
  - `remaining_surplus_value = Σ_{undrafted in-pool p} (baseline_dollars[p] − min_bid)`
  - `inflation = remaining_surplus_money / remaining_surplus_value` (guarded: if the denominator ≤ 0 —
    only the dregs left — fall back to `1.0`)
  - `max_bid = min_bid + (baseline_dollars[player] − min_bid) · inflation`
  Recomputed each nomination. When the room overpays early, `remaining_surplus_money` shrinks faster than
  value and inflation drops below 1 (be disciplined late); when the room underspends, inflation rises
  (the remaining studs are worth a premium). One scalar re-pricing the whole board — the textbook
  inflation adjustment, expressed in the same surplus units the generator uses.
- **v3 `MarginalValueBid`** — bid to the player's **marginal lineup value**, converted to dollars at the
  live market exchange rate:
  - `lift = StartersValuer().value(my_roster + player) − StartersValuer().value(my_roster)` — the
    optimal-starting-lineup points the player adds to *my* current roster (reuses `optimal_lineup_points`
    via the existing valuer; restrictive-slot-first fill, FLEX before SUPER_FLEX — §3.4 of the snake
    spec). **This is the tournament surrogate** (§2): cheap, deterministic, no MC.
  - `points_per_dollar = remaining_surplus_value_points / remaining_surplus_money`, where
    `remaining_surplus_value_points = Σ_{undrafted in-pool p} marginal-lineup-lift(p) to an empty slot`
    — i.e. how many lineup points a surplus dollar buys at the current board. (Computed once per
    nomination over the board, not per bidder.)
  - `max_bid = min_bid + lift / points_per_dollar` (guarded for `points_per_dollar ≤ 0`).
  A player who doesn't crack my starting lineup has `lift == 0` and so is bid at `min_bid` — the honest
  behavior of a win-now, starters-only objective. **Known limitation, stated plainly:** this surrogate
  under-values *depth* (a backup that only matters via injury weeks scores 0 marginal), because
  `StartersValuer` is single-week-optimal and availability-blind. The depth-aware version uses a
  `SeasonValuer` marginal (`marginal_season_values`, which already credits depth via availability) and is
  exactly the full-fidelity v3 deferred to the live assistant (§2). The tournament can still *score* with
  a `SeasonValuer` (§3.6) while v3 *bids* on the starters surrogate — that mismatch is the honest
  cross-check, not a bug (and it avoids v3 trivially in-sample-optimizing the exact metric it's graded on
  — the circularity the snake-side projected-eval flagged in TODO #48).

### 3.5 The market — noisy-WTP bots (`auction/market.py`)

A bot occupies every non-hero seat. Its willingness-to-pay is value-rational with multiplicative noise —
the auction analog of `opponent.bot_pick`'s noisy ADP:

```
bot_max_bid(seat_view, player, baseline_dollars, rng, *, price_jitter) -> int
```

1. If the seat has no open slot, it does not bid (returns 0 / abstains).
2. Center on the market value: `wtp = baseline_dollars[player] · (1 + rng.normal(0, price_jitter))`,
   floored at `min_bid`, rounded to an int.
3. The engine clamps `wtp` to the seat's `feasible_max` (§3.2), so a bot can never bid itself out of a
   fillable roster.

`price_jitter` (fractional WTP spread) is the auction analog of the snake harness's `adp_jitter` — the
single market-noise knob — and is **distinct from any model parameter** the contestants carry. Realism,
as on the snake side, comes from the market values themselves (SOS dollars already encode positional
scarcity), not from a roster-construction rule. Default `price_jitter` lives in a named constant; bots
take no inflation/marginal logic — they *are* the static-value market the contestants are measured
against.

**Bid resolution — second-price + min_bid (English clearing).** Given the nominee, collect every eligible
seat's clamped max bid (hero via its `AuctionBidStrategy`, bots via `bot_max_bid`). The winner is the
**argmax** max-bid; the **price paid** is `min(winner_max, second_highest_max + min_bid)`, i.e. the winner
pays one tick over the runner-up's ceiling, never more than its own — the standard ascending-auction
clearing price, deterministic given the max-bids (no live increment loop needed). With a lone eligible
bidder, the price is `min_bid`. Ties in max-bid break deterministically (seat index, then `gsis_id`), so
same seed ⇒ same outcome.

### 3.6 Simulation, scoring, and the paired tournament (`auction/simulation.py`, `auction/tournament.py`)

```
simulate_auction(strategy: AuctionBidStrategy, my_seat: int, pool, config,
                 *, baseline_dollars, price_jitter, rng) -> DataFrame   # hero's roster rows
```

The loop, until every seat's roster is full:

1. **Nominate.** Advance `nominator` in seat order, skipping full seats; the active nominator nominates the
   **highest baseline-dollar undrafted player** (shared default nominator — §5.2). (The nominee is
   nominator-independent under this default, but the nominator pointer still governs *whose* turn it is,
   so strategic nomination drops in later without reshaping the loop.)
2. **Bid.** Collect clamped max-bids from every eligible seat (hero via `strategy`, others via
   `bot_max_bid`), resolve the winner and price per §3.5.
3. **Award.** Subtract the price from the winner's budget, append `(gsis_id, position, price)` to its
   roster, add to `drafted`.

One seeded `numpy.random.Generator` drives **all** bot WTP noise for the whole auction, so — exactly as in
the snake harness — same seed + same hero strategy ⇒ byte-identical hero roster (a tested invariant), and
two strategies at the same seed face the **same market realization** up to where the hero's own diverging
wins perturb budgets/board downstream. That is the paired "same room, different me" counterfactual. The
function returns the hero's drafted rows (a sub-frame of `pool`) — the identical contract to
`simulate_draft`, so scoring is reused with **zero** changes:

`run_auction_tournament(strategies: Mapping[str, AuctionBidStrategy], pool, config, *, my_seat, n_seeds,
price_jitter, base_seed, valuer=StartersValuer())` mirrors `run_tournament` exactly:

- For each strategy and seed `s`: `rng = default_rng(base_seed + s)` (same `s` ⇒ same market across
  strategies — the paired design), `roster = simulate_auction(...)`,
  `value[strategy][s] = valuer.value(roster, config.roster_slots)`.
- Per-strategy mean + percentile-bootstrap CI; **paired** winner test on the top-two per-seed difference,
  declaring a winner only when the diff CI excludes 0; with >2 strategies, report every mean+CI and the
  top-two paired diff.

**Reuse, not re-implement (the project bar).** `_bootstrap_mean`, `Interval`, the paired-diff/winner
logic, and `_validate_run_params` already live in `assistant/tournament.py`. Promote the strategy-agnostic
pieces — `Interval`, `_bootstrap_mean`, the size arm of `_validate_pool`, and the ranked-top-two-diff
winner computation — into a shared `assistant/_compare.py` that **both** the snake and auction tournaments
import. The snake `run_tournament` keeps its ADP-specific arms (`adp_jitter`, all-null-ADP check); the
auction harness gets a sibling `AuctionTournamentResult` (same `Interval` fields; `my_slot → my_seat`,
`adp_jitter → price_jitter`, plus `budget`/`min_bid` echoed for reproducibility). `StartersValuer` is the
default; `--valuer season` swaps in `SeasonValuer` unchanged (the snake harness already wired the
risk-aware valuer — the auction harness inherits it for free).

### 3.7 CLI surface

`scripts/auction_tournament.py` → engine:

```
python scripts/auction_tournament.py \
    --vorp-table <consensus_vorp.parquet> \
    --league-config <league.json> \
    --my-seat N \
    [--seeds K] [--price-jitter F] [--seed BASE] \
    [--valuer starters|season --season YYYY --n-sims M] \
    compare
```

`compare` runs the three registered bid models (`static`, `inflation`, `marginal`) and prints the
per-model mean+CI table and the paired winner line. `--league-config` is the single source of roster
shape + team count + ruleset + budget + min-bid (matching `generate_vorp_table`). `--seed` makes any run
reproducible. Defaults are sensible so a bare invocation works. CLI imports the engine, never the reverse.

## 4. Testing

Synthetic fixtures only (project norm — no network, no real parquet in unit tests):

- **Reserve / feasible-max invariant** — a seat with `budget == min_bid · open_slots` can only bid
  `min_bid`; a seat with one slot left can bid its whole budget; the engine clamps an over-budget desired
  bid down to `feasible_max`. Pin the endgame-$1 path directly.
- **Bid resolution** — winner is the argmax max-bid and pays `second_highest + min_bid` (not its own
  ceiling); a lone bidder pays `min_bid`; a max-bid tie breaks deterministically (seat, then `gsis_id`).
- **Full-auction conservation** — every seat ends with exactly `roster_size` players; `Σ prices ≤
  total_budget`; no player drafted twice; no seat ends with `open_slots > 0` while it still had
  `feasible_max ≥ min_bid` and an affordable player existed (the auction actually completes).
- **Determinism & paired market** — same `base_seed` + same strategy ⇒ identical hero roster; different
  seed ⇒ generally different; two strategies at the same seed see identical bot WTPs until the hero's
  first diverging win (proves the paired counterfactual holds).
- **Bot policy** — `bot_max_bid` centers on `baseline_dollars` and with `price_jitter → 0` equals the
  baseline (clamped to `feasible_max`); a full seat abstains.
- **v1 `StaticDollarBid`** — returns the player's baseline dollar (pre-clamp); after clamp never exceeds
  `feasible_max`.
- **v2 `InflationBid`** — on a hand-built state where the room has overspent, inflation < 1 and the bid
  drops below baseline; where the room has underspent, inflation > 1 and it rises; the degenerate
  `remaining_surplus_value ≤ 0` falls back to factor 1.0 (no div-by-zero).
- **v3 `MarginalValueBid`** — a player who improves the optimal lineup bids above `min_bid`; a player who
  doesn't crack it (`lift == 0`) bids exactly `min_bid`; `points_per_dollar ≤ 0` is guarded. Reuses
  `optimal_lineup_points`, so the FLEX-before-SUPER_FLEX strand case is covered transitively (re-assert
  one case to pin the wiring).
- **Paired-difference stat** — a synthetic fixture where one model holds a constant per-seed edge ⇒ the
  paired-diff CI excludes 0 and names it; a zero-edge fixture ⇒ "no separation" (reuses the promoted
  `_compare` helper, so the snake harness's existing stat tests cover the shared core; add one auction
  smoke).
- **League-driven** — run the harness under two different `roster_slots` **and** two different
  `budget`/`min_bid` configs on the same pool; both complete and score with no code change (the guard
  against any hardcoded $200 / 12-team / position-count assumption).
- **CLI smoke** — `compare` runs end-to-end on a fixture parquet + config and prints a result.

All gates per the project bar: `pytest -v`, `mypy src tests` (strict), `ruff check src tests`,
`ruff format --check src tests`; plus `pytest -v -k "ingest or store or schemas"` if any schema/store path
is touched (it should not be — no new schema).

## 5. Key decisions

- **5.1 The bid models *are* the contestants; the tournament is the bake-off.** The user asked to build
  all three (static, inflation-aware, marginal) and test which is best. The snake tournament's
  simulate→score→compare seam is precisely the apparatus to settle it — race them against one seeded
  market, score by optimal lineup, decide on the paired bootstrap. No separate "evaluation" rig is needed.
- **5.2 Isolate the variable: one shared nominator, vary only the bid.** All three contestants share the
  "nominate highest remaining dollar" nominator and the same bot field, so a measured difference is
  attributable to the bid model alone — mirroring how the snake harness holds the bot field fixed and
  varies only the hero. Strategic nomination is a real but *separate* axis, deferred (§2), not blended in.
- **5.3 Second-price + min_bid clearing, not a live increment loop.** Given each seat's max-bid, the
  ascending-auction outcome is determined: the top bidder wins at one tick over the runner-up. Simulating
  per-dollar increments would add loop cost and RNG surface for an identical result. Deterministic given
  seed → reproducible.
- **5.4 v3 ships as a cheap starters-marginal surrogate in the tournament; full-season-MC v3 is the live
  model.** A per-candidate `SeasonValuer` MC inside the `seeds × nominations × sims` loop is
  computationally infeasible at tournament scale (the snake side already learned the per-pick-MC cost
  lesson — TODO's depth-aware-strategy entry: a naive per-pick MC was ≈9 hrs/slot before the numpy
  fast-path). The optimal-lineup-marginal surrogate makes the bake-off runnable now; the richer
  availability-aware marginal is reserved for the once-per-real-pick live assistant. Stated as scope, not
  hidden.
- **5.5 Bots are value-rational noisy-WTP, not behavioral.** Realism comes from the SOS dollars (which
  already encode scarcity), plus one multiplicative noise knob — the exact stance (and the exact honest
  caveat) as the snake harness's noisy-ADP bots. A better human-opponent model is the single biggest
  realism lever and is already an open follow-up on the snake side (TODO #46); the auction harness inherits
  that backlog item rather than pretending to solve it.
- **5.6 No new schema.** `generate_auction_values` owns `AuctionValuesSchema`; the tournament result is a
  handful of floats the CLI renders. A schema would be ceremony — the same call the snake harness made.
- **5.7 Reuse the scoring and stats layers verbatim; promote the shared core.** `StartersValuer` /
  `SeasonValuer` and the bootstrap/winner logic are mechanism-agnostic by the snake spec's design (§5.7
  there). This slice promotes `Interval` / `_bootstrap_mean` / the paired-winner computation into a shared
  `_compare.py` and reuses them — the seam paying off exactly as intended, "obviously a later effort"
  rather than a rewrite.

## 6. Open questions / future slices

- **Strategic nomination axis** — dump-the-enemy's-budget, price-enforcement, nominate-your-target-late;
  the natural Slice 2 once the bid bake-off has a winner (the bid model becomes the fixed baseline, and
  nominators become the contestants).
- **Full-fidelity v3 in a live auction assistant** — the availability-aware `SeasonValuer` marginal as a
  "max bid right now" recommender, evaluated once per real pick; the auction analog of the snake Slice 3
  board.
- **Better bot market** — a learned or behavior-calibrated WTP model (shared with the snake-side TODO #46
  human-opponent effort) so the verdict reflects a real room, not a value-rational one.
- **Seat sweep** — average the hero across all seats for a seat-robust verdict (the auction analog of the
  snake hero-slot sweep), if the winner proves seat-sensitive.
- **Budget-curve diagnostics** — persist per-seed spend-by-round / stars-and-scrubs vs balanced shape of
  the winning model; earns a stored schema only if a consumer needs the history (per §5.6).
