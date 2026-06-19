# ESPN-anchored auction bots (Slice 2) — design

**Status:** approved (brainstormed 2026-06-18). Proceeding to plan + execute.
**Owner:** draft-hub / auction.
**Depends on:** Slice 1 (`espn_auction_dollars` on `VorpTableSchema`, shipped PR #82), `generate_auction_values` (the SOS allocator), and the auction tournament (`market.py` / `simulation.py` / `tournament.py` / `tournament_cli.py`).

## Problem

The auction bake-off has a **shared-value problem**: the bot field (`market.py`) centers its willingness-to-pay on the *same* `auction_dollars` the hero uses — our own SOS-computed value — plus mean-zero noise. Because opponents price every player off the hero's own numbers, the hero has **no informational edge**; the market leaves no systematically exploitable bargain, only random noise. This is the leading suspect for why every hero strategy sits at or below the field's fair-share baseline (Runs E–G).

Slice 1 landed the fix's raw material: a real, human-facing **`espn_auction_dollars`** column on the preset VORP/pool table (crowd `auctionValueAverage` now, expert `draftRanksByRankType` fallback). **This slice wires it in**: the **bots** price off the real ESPN values (rescaled to the league budget; unpriced players go for $1), while the **hero keeps pricing off our SOS model**. The hero's edge becomes the gap between the two — the players ESPN under-prices or ignores that our projection values.

## Goals

1. Build a per-player **bot reference-price vector** from `espn_auction_dollars`, rescaled to the league budget, with a $1 floor for players ESPN didn't price.
2. Point the **bot** bid WTP (all three archetypes) at that vector; leave the **hero** strategies and `generate_auction_values` on `auction_dollars` untouched.
3. Expose it behind a CLI flag `--bot-prices {espn,model}` (**default `espn`**), with graceful degradation to `model` when the pool carries no ESPN values.
4. Carry an ESPN-vs-ours diagnostic (`reference_dollars`/`value_delta`) for the report.
5. Run the validation bake-off (half-PPR 12-team + 16-team) and record it as **Run H** alongside Runs G/F. No winner declared (the September decision is unchanged).

## Non-goals

- **Not** changing the hero. Hero strategies and `generate_auction_values` keep reading `auction_dollars`.
- **Not** SOS-backfilling unpriced players — the chosen model is **ESPN-only with a $1 floor** for players ESPN didn't rank (the deliberate "max divergence" variant; see Chosen approach).
- **Not** per-position rescaling — a **global** budget normalization that *preserves* ESPN's cross-position allocation (that bias is the exploitable lever).
- **Not** changing nomination order. Nomination stays on the shared SOS `auction_dollars`, so the *only* variable that changes between `model` and `espn` modes is the bot bid price (clean A/B). Room-driven nomination is a possible later refinement (Open Questions).
- **Not** the RL bid policy (#49b) or multi-year averaging (#49a).

## Chosen approach

### The bot price vector: SOS-style allocation over ESPN dollars

The bot reference price is computed by the **same Surplus-Of-Surplus machinery** `generate_auction_values` already uses, with **ESPN dollars as the value signal instead of VORP**:

- Same in-pool set as the hero (`_select_pool(pool, config)`).
- Every in-pool player floored at `min_bid`. The surplus (`total_budget − pool_size × min_bid`) is distributed **proportionally to each player's `espn_auction_dollars`** among priced players; **unpriced players (NA ESPN value) contribute zero and therefore sit at exactly `min_bid` ($1)**.
- Result: the vector sums to exactly `total_budget` (the bot market clears to the same budget the hero plays under), ESPN's relative and cross-position pricing is preserved, and the $1 floor is honored.

This is the **choice-B "max divergence" model**: bots pay real ESPN prices for ESPN-ranked players and ignore the deep pool (it goes for $1). That deep pool is exactly where the SOS-valued hero finds bargains the bots leave on the table.

**Reuse, don't duplicate (senior-dev override).** Extract the surplus→round→drift-correct→floor-protect core out of `generate_auction_values` (currently inline at `auction.py:57–90`) into a shared private helper:

```
_allocate_surplus(value_signal: pd.Series, config: LeagueConfig) -> pd.Series  # Int64 dollars
```

where `value_signal` is a non-negative per-player weight indexed over the in-pool players, and the return sums to `total_budget` with every entry ≥ `min_bid`. Two callers:
- `generate_auction_values` (hero side, unchanged behavior): `value_signal = pool_df["vorp"].clip(lower=0)`.
- `espn_anchored_bot_prices` (new, bot side): `value_signal = pool_df["espn_auction_dollars"].astype("Float64").fillna(0).clip(lower=0)`.

`generate_auction_values` must produce **byte-identical** output after the extraction (an equivalence test pins this — see Testing). The all-zero degenerate branch (uniform split) is preserved in the helper.

```
espn_anchored_bot_prices(pool: pd.DataFrame, config: LeagueConfig) -> pd.Series
```
returns a `gsis_id`-indexed `Int64` Series covering **every** row of `pool`: in-pool players get the allocation above; out-of-pool players get `0` (matching `auction_dollars=0` for out-of-pool, so bots reading it bid `min_bid` exactly as today). Lives in `src/projections/draft/auction.py` (it needs `_select_pool` + `_allocate_surplus` + `LeagueConfig`; same home as `generate_auction_values`). Pure, unit-tested.

**Known property (deliberately accepted, choice B).** In deep leagues (16-team) where ESPN prices only ~150 of ~256 pooled players, the surplus concentrates on the priced studs and inflates them above ESPN's nominal dollars. This is a direct consequence of "$1 floor for unpriced + normalize to budget," and the bake-off (Run H, 16-team) is where we observe whether it's too extreme. Flagged, not guarded.

### The seam: a `bot_dollars` column the bots read

`simulate_auction` / `_simulate_to_state` gain a parameter:

```
bot_dollars: pd.Series | None = None   # gsis_id-indexed Int64; None => bots use auction_dollars
```

Immediately after `bd = baseline_dollars.set_index("gsis_id")` (`simulation.py:127`), attach a `bot_dollars` column:
- `bot_dollars is None` → `bd["bot_dollars"] = bd["auction_dollars"]` (back-compat; reproduces today's behavior byte-for-byte).
- provided → `bd["bot_dollars"] = bot_dollars.reindex(bd.index)` (the producer covers every `gsis_id`, so the reindex is exact; defensively `.fillna(bd["auction_dollars"])`).

Every place a **bot** reads value switches from `"auction_dollars"` to `"bot_dollars"` (4 sites in `market.py`): `bot_max_bid` (line 42), `_value_tier`'s in-pool ranking (line 87), `PatientValueBot.max_bid` (line 138), `BalancedBot.max_bid` (line 168). A bot's *entire* value view becomes `bot_dollars` (so `PatientValueBot`'s stud/mid/scrub tiers rank by the bot's own ESPN-anchored view, consistently).

**Untouched:** the hero (`AuctionView.baseline_dollars` → `auction_dollars`), `generate_auction_values`, and nomination (`val_by_id` / `nominate_order` at `simulation.py:119–128` keep using `auction_dollars`).

### The flag: `--bot-prices {espn,model}`, default `espn`

`run_auction_tournament` gains `bot_prices: str = "espn"` (the CLI default; `tournament_cli.py` adds `--bot-prices`):
- `espn` and the pool carries usable ESPN values → compute `bot_dollars = espn_anchored_bot_prices(pool, config)`, pass it to every `simulate_auction` call.
- `model`, or `espn` but the pool has **no `espn_auction_dollars` column or it is entirely NA** → `warnings.warn(...)` and pass `bot_dollars=None` (the shared-value baseline). This graceful degradation is what makes default-on safe on a weekly-path table (otherwise default-on would price every player at $1).

"Usable" = the column is present **and** has ≥1 non-null value. The warning text names the fallback so a silent all-$1 market is impossible.

### Diagnostics

In `espn` mode, call `generate_auction_values(pool, config, reference_prices=<espn_auction_dollars renamed to reference_dollars>)` so the hero's `baseline_dollars` frame carries `reference_dollars` (ESPN) and `value_delta` (`auction_dollars − reference_dollars`, our$ − ESPN$) for the Run-H write-up. This is additive — it does not change `auction_dollars` and the hero is unaffected.

## Requirements

R1. `_allocate_surplus(value_signal, config)` is extracted from `generate_auction_values` (the surplus split, rounding, drift correction, floor protection, and the all-zero uniform fallback). `generate_auction_values` calls it with `clip(lower=0)`'d VORP and produces **byte-identical** output to before (pinned by an equivalence test).

R2. `espn_anchored_bot_prices(pool, config)` returns a `gsis_id`-indexed `Int64` Series over **every** pool row: in-pool = SOS allocation over `espn_auction_dollars` (NA→0 weight, so unpriced = exactly `min_bid`); out-of-pool = 0. The in-pool entries sum to `total_budget`. Pure, unit-tested. Treats an absent `espn_auction_dollars` column as all-NA.

R3. `simulate_auction`/`_simulate_to_state` accept `bot_dollars: pd.Series | None = None`; attach a `bot_dollars` column to `bd` (defaulting to `auction_dollars` when None). The 4 bot read-sites in `market.py` read `"bot_dollars"`. The hero, `generate_auction_values`, and nomination order are untouched.

R4. `run_auction_tournament` accepts `bot_prices: str = "espn"`; computes the bot vector in `espn` mode (when usable) and threads `bot_dollars` into every `simulate_auction` call; warns and falls back to `model` (`bot_dollars=None`) when ESPN data is absent/all-NA. In `espn` mode it also passes ESPN as `reference_prices` to `generate_auction_values` for the diagnostic columns.

R5. `tournament_cli.py` adds `--bot-prices {espn,model}` (default `espn`), passed through to `run_auction_tournament`.

R6. **Back-compat / clean A/B**: with `bot_dollars=None` (and `--bot-prices model`), the tournament reproduces today's behavior **byte-for-byte** — `bot_dollars == auction_dollars`, nomination unchanged. The only difference in `espn` mode is the bot bid price. Existing auction tests pass unchanged (the new param defaults preserve every current call site).

R7. Conventions: `GsisId` canonical; `Position` referenced as the enum; `pd.Int64Dtype`/`pd.Float64Dtype` for nullable numerics; `SCHEMA.validate(df)` with reassignment at boundaries; no new direct parquet I/O. `bot_dollars` is an **engine-internal** column on the indexed `bd` frame — it is **not** added to `AuctionValuesSchema` (which `generate_auction_values` returns and validates); it is attached after validation inside the simulation layer.

R8. **Data prerequisite for the bake-off**: populating `espn_auction_dollars` needs a fresh `refresh_external_projections` re-ingest + `generate_preset_vorp_tables.py` regen. The *code* does not require it (graceful degradation), but Run H does.

## Edge cases / failure modes

- **No ESPN data (weekly-path table, or stale snapshot pre-reingest):** column absent or all-NA → warn + `model` fallback. Default-on never produces an all-$1 market.
- **Sparse ESPN coverage (deep 16-team pool):** unpriced players = $1; surplus concentrates on priced studs (the accepted deep-league inflation property). Realistic-ish (studs cost more in deep leagues) but observed in Run H.
- **A priced player is out-of-pool:** `espn_anchored_bot_prices` returns 0 for out-of-pool rows regardless of ESPN value (out-of-pool players aren't rosterable; bots bid `min_bid`, same as today).
- **All in-pool players unpriced but column present (degenerate):** the `_allocate_surplus` all-zero branch splits the surplus uniformly. Upstream this is caught by the "≥1 non-null" usability gate (→ `model` fallback), so it shouldn't arise in `espn` mode; the uniform branch is a safe backstop.
- **Negative drift with few priced players:** the existing floor-protection in `_allocate_surplus` excludes rows already at `min_bid`; with many unpriced players parked at the floor, the adjustable set is the priced players (drift ≤ pool_size, absorbable). The pre-existing `ValueError` guards the pathological case.
- **Scoring-basis mixing (carried from Slice 1):** the crowd `auctionValueAverage` is on ESPN's default-league basis while the expert fallback is ruleset-specific — two bases mixed in one column. Accepted per Slice 1; whether it distorts the bot field (esp. STANDARD) is observed, not corrected, here.

## Testing expectations

- **`_allocate_surplus` extraction (equivalence):** `generate_auction_values` output is byte-identical before/after the refactor on a fixture VORP table (gold-frame or recompute-and-`assert_frame_equal`).
- **`espn_anchored_bot_prices` (new unit test):** (a) in-pool entries sum to `total_budget`; (b) unpriced players get exactly `min_bid`; (c) priced players' dollars are monotonic in `espn_auction_dollars` and split the surplus proportionally; (d) out-of-pool rows = 0; (e) every entry ≥ `min_bid`; (f) absent/all-NA column → uniform fallback (sum still `total_budget`); result dtype `Int64`.
- **Seam (`simulation.py`):** with `bot_dollars=None`, `bd["bot_dollars"] == bd["auction_dollars"]` and a full `simulate_auction` run is identical to baseline (same RNG → same rosters). With a provided `bot_dollars`, the bots' winning prices track the ESPN vector (a stud ESPN over-prices clears higher; an unpriced startable player clears near `min_bid`).
- **Bot read-sites (`market.py`):** each archetype's `max_bid` centers on `bot_dollars` (not `auction_dollars`) when the two differ; `_value_tier` ranks by `bot_dollars`.
- **Flag (`tournament.py`/CLI):** `--bot-prices espn` on a pool **without** ESPN values warns and produces the `model`-mode result (equality test vs `--bot-prices model`); `--bot-prices espn` on a pool **with** ESPN values differs from `model`. `reference_dollars`/`value_delta` populated in `espn` mode.
- **No-regression:** the full auction suite (`test_market`, `test_simulation`, `test_tournament`, bid-strategy, VORP/auction-values tests) passes unchanged (R6).
- Per `CLAUDE.md`, run `pytest -k "auction or schemas"` (touches the auction engine; no schema change, but the auction-values seam is adjacent) plus the targeted new tests.

## Phasing

~4 tasks, each ≤5 files (edit-site count called out where it exceeds file count):
1. **Allocator extraction + bot price vector** — extract `_allocate_surplus`, add `espn_anchored_bot_prices` (`auction.py`) + equivalence test + `espn_anchored_bot_prices` unit tests. (No behavior change yet.)
2. **Engine seam** — `bot_dollars` param + column in `simulation.py`; switch the 4 bot read-sites in `market.py` to `bot_dollars` + seam/equality tests.
3. **Flag plumbing + diagnostics** — `bot_prices` in `run_auction_tournament` (compute vector, graceful fallback, `reference_prices` diagnostic) + `--bot-prices` in `tournament_cli.py` + flag tests.
4. **Validation bake-off** — re-ingest `external_projections` + regen preset tables (R8), run the 8-model field vs the ESPN-anchored bot market at half-PPR **12-team and 16-team**, write **Run H** into `reports/auction_tournament_validation_2026.md` (vs Runs G/F), update TODO #49c + `project_management.md`.

## Open questions for later (Slice 3+)

- **Room-driven nomination:** should nomination order follow the bots' (ESPN) view in `espn` mode rather than the shared SOS order? Kept on SOS here for a clean A/B; revisit if nomination sequence proves load-bearing.
- **SOS-backfill variant:** an A/B of choice-B (ESPN-only, $1 floor) vs the ESPN+SOS-backfill variant, to measure how much the deep-pool $1 floor drives the hero's edge.
- **Deep-league inflation guard:** if Run H shows 16-team stud inflation is unrealistic, cap the per-player bot dollar at (e.g.) a multiple of its ESPN value.
- **Multi-year (#49a):** once Run H is read, fold ESPN-anchored bots into the multi-year averaging before the September decision.
