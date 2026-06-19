# ESPN-anchored auction bots (Slice 2) — design

**Status:** approved (brainstormed 2026-06-18; hardened via spec-review 2026-06-18). Proceeding to plan + execute.
**Owner:** draft-hub / auction.
**Depends on:** Slice 1 (`espn_auction_dollars` on `VorpTableSchema`, shipped PR #82), `generate_auction_values` (the SOS allocator, which already exposes `reference_prices → reference_dollars/value_delta`), and the auction tournament (`market.py` / `simulation.py` / `tournament.py` / `tournament_cli.py`).

## Problem

The auction bake-off has a **shared-value problem**: the bot field (`market.py`) centers its willingness-to-pay on the *same* `auction_dollars` the hero uses — our own SOS-computed value — plus mean-zero noise. Because opponents price every player off the hero's own numbers, the hero has **no informational edge**; the market leaves no systematically exploitable bargain, only random noise. This is the leading suspect for why every hero strategy sits at or below the field's fair-share baseline (Runs E–G).

Slice 1 landed the fix's raw material: a real, human-facing **`espn_auction_dollars`** column on the preset VORP/pool table (crowd `auctionValueAverage` now, expert `draftRanksByRankType` fallback). **This slice wires it in**: the **bots** price off the real ESPN values (rescaled to the league budget; unpriced players go for $1), while the **hero keeps pricing off our SOS model**. The hero's edge becomes the gap between the two — the players ESPN under-prices or ignores that our projection values.

## Goals

1. Build a per-player **bot reference-price vector** from `espn_auction_dollars`, rescaled to the league budget, with a $1 floor for players ESPN didn't price.
2. Point the **bot** bid WTP (all three archetypes) at that vector; leave the **hero** strategies and `generate_auction_values` on `auction_dollars` untouched.
3. Expose it behind a CLI flag `--bot-prices {espn,model}` (**default `espn`**), with graceful degradation to `model` when the pool carries no ESPN values.
4. Emit an ESPN-vs-ours diagnostic (`reference_dollars`/`value_delta`) **in the CLI readout** for the report (the columns + delta math already exist on `generate_auction_values`; only the populated call + the readout are new).
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
- Every in-pool player floored at `min_bid`. The surplus (`total_budget − pool_size × min_bid`) is distributed **proportionally to each player's `espn_auction_dollars`** among priced players; **unpriced players (NA ESPN value) contribute zero surplus weight and therefore park at `min_bid` ($1)** (see the drift caveat below).
- Result: the vector sums to exactly `total_budget` (the bot market clears to the same budget the hero plays under), ESPN's relative and cross-position pricing is preserved, and the $1 floor is honored.

This is the **choice-B "max divergence" model**: bots pay real ESPN prices for ESPN-ranked players and ignore the deep pool (it goes for $1). That deep pool is exactly where the SOS-valued hero finds bargains the bots leave on the table.

**Reuse, don't duplicate (senior-dev override).** Factor the **surplus-split → round → drift-correct → floor-protect core** of `generate_auction_values` into a shared private helper:

```
_allocate_surplus(value_signal: pd.Series, config: LeagueConfig) -> pd.Series  # Int64 dollars
```

**What moves into the helper vs. what stays in the caller:**
- *Into the helper:* the logic of `auction.py:61–89` — the `if … > 0 / else (uniform)` branch, `min_bid + extra`, `.round().astype("int64")`, and the drift-correction loop with floor protection. The helper **returns** `rounded.astype(pd.Int64Dtype())` as its value (the RHS of line 90).
- *Stays in the caller:* lines **57–59** (the `pool_df` selection + value-signal construction) **and line 90's column assignment** — the caller writes `pool_df["auction_dollars"] = _allocate_surplus(value_signal, config)`. The helper never assigns to `pool_df`.

**Name substitution (helper is self-contained):** every caller-local reference in 61–89 becomes the parameter — `positive_vorp_sum` → `float(value_signal.sum())`, `positive_vorp` → `value_signal`, **and line 66's `index=pool_df.index` → `index=value_signal.index`** (the uniform-fallback branch; equivalent because the caller's `value_signal` preserves `pool_df.index`). So the branch is `if float(value_signal.sum()) > 0:` and the proportional term `(value_signal / float(value_signal.sum())) * surplus`. The scalars `total_budget`/`reserve`/`surplus`/`min_bid`/`total_pool_size` are config-only (lines 53–55) and are recomputed inside the helper from `config`.

**Index + dtype contract (load-bearing for byte-identity):** the helper **preserves `value_signal`'s index** (reindexes nothing, resets nothing) — the drift loop's `fractional.sort_values(...).index` and `rounded.loc[idx]` are index-dependent, so the caller passes a Series indexed exactly as `pool_df.index`. The divisor is `float(value_signal.sum())` (matching the original `float(positive_vorp.sum())`), so the VORP path is **numerically identical** — pinned by the equivalence test, not asserted as textually identical. **Callers must pass a non-null plain `float64` Series** (no nullable `Float64`, no NA), so the helper runs the single, already-tested float64 path on both sides; `.astype("int64")` in the drift step would otherwise raise on NA. The helper returns `Int64` on **every** branch, including the all-zero uniform fallback.

Two callers:
- `generate_auction_values` (hero side, unchanged behavior): `value_signal = pool_df["vorp"].clip(lower=0.0)` (already float64), then `pool_df["auction_dollars"] = _allocate_surplus(value_signal, config)`. Output must be **byte-identical** to before (an equivalence test pins this — see Testing).
- `espn_anchored_bot_prices` (new, bot side): `value_signal = pool_df["espn_auction_dollars"].astype("Float64").fillna(0).clip(lower=0.0).astype("float64")` (NA → 0 weight → parks at `min_bid`; final `.astype("float64")` makes it the same plain-float64 input the helper expects), then calls the helper.

```
espn_anchored_bot_prices(pool: pd.DataFrame, config: LeagueConfig) -> pd.Series
```
returns a `gsis_id`-indexed `Int64` Series covering **every** row of `pool`: in-pool players get the allocation above; out-of-pool players get `0` (matching `auction_dollars=0` for out-of-pool, so bots reading it bid `min_bid` exactly as today). It must be called on the **same `pool` frame** later passed to `generate_auction_values` (the post-`attach_is_rookie` pool in the tournament), so the two share an identical `gsis_id` set and the downstream `reindex` is exact. An absent `espn_auction_dollars` column is guarded (`"espn_auction_dollars" in pool.columns`) and treated as all-NA. Lives in `src/projections/draft/auction.py` (it needs `_select_pool` + `_allocate_surplus` + `LeagueConfig`; same home as `generate_auction_values`). Pure, unit-tested.

**Drift caveat (unpriced ≈ $1, not a hard invariant).** Unpriced rows compute `dollars_float = min_bid + 0 = 1.0` (fractional `0.0`). The drift loop, when `drift > 0`, sorts by fractional *descending*, so zero-fractional (unpriced) rows are bumped **last** — only if positive drift exceeds the count of priced rows with nonzero fractional parts. With ~150 priced players in a realistic pool and drift of a few dollars, this never happens, so unpriced players sit at exactly `min_bid`. But it is an **expectation, not a guarantee**; a pathological fixture could push a small handful of unpriced players to `min_bid + 1`. Immaterial to bot bidding ($1 vs $2 on a deep player), and the helper is shared with the VORP path so we cannot special-case it without breaking byte-identity. Tests assert the property on a controlled (non-positive-drift) fixture and document the bound.

**Known property (deliberately accepted, choice B).** In deep leagues (16-team) where ESPN prices only ~150 of ~256 pooled players, the entire surplus piles onto the priced players (unpriced contribute zero weight), inflating the studs well above ESPN's nominal dollars — potentially a large multiplier, and in the degenerate single-priced-player case that one player absorbs nearly the whole budget. This is a direct consequence of "$1 floor for unpriced + normalize to budget." A unit test pins the inflation factor on a representative fixture (so it is bounded by more than "Run H will notice"), and Run H (16-team) observes whether it is too extreme. A per-player cap is parked in Open Questions, not built here.

### The seam: a `bot_dollars` column the bots read

`simulate_auction` **and** `_simulate_to_state` (both signatures) gain a parameter:

```
bot_dollars: pd.Series | None = None   # gsis_id-indexed Int64; None => bots use auction_dollars
```

In `_simulate_to_state`, **after `nominate_order` is built (after `simulation.py:128`)** — so the nomination machinery at 119–128 is untouched — attach a `bot_dollars` column to `bd`:
- `bot_dollars is None` → `bd["bot_dollars"] = bd["auction_dollars"]` (back-compat; reproduces today's behavior byte-for-byte).
- provided → `bd["bot_dollars"] = bot_dollars.reindex(bd.index)` (the producer covers every `gsis_id`, so the reindex is exact; defensively `.fillna(bd["auction_dollars"])`). Both sides are `Int64`, so the reindex/fillna is dtype-safe.

Every place a **bot** reads value switches from `"auction_dollars"` to `"bot_dollars"` (4 sites in `market.py`): `bot_max_bid` (line 42), `_value_tier`'s in-pool value lookup (line 87), `PatientValueBot.max_bid` (line 138), `BalancedBot.max_bid` (line 168). A bot's *entire value view* becomes `bot_dollars` (so `PatientValueBot`'s stud/mid/scrub tiers rank by the bot's own ESPN-anchored view, consistently). **Note:** `_value_tier`'s `in_pool` *mask* (line 87) stays as-is — the bot still ranks within the shared VORP-selected pool; only the value column it ranks *by* switches to `bot_dollars`. The `in_pool` column survives `set_index` onto `bd`.

Because `bd` now carries an engine-internal `bot_dollars` column **in addition to** the `AuctionValuesSchema` columns, update the `AuctionView.baseline_dollars` docstring (`bid_strategy.py:30`) to describe it as the indexed engine frame (`AuctionValuesSchema` columns + `bot_dollars`), not a pristine `AuctionValuesSchema`. Runtime is unaffected — there is no re-validation in the sim loop and the hero reads only `auction_dollars`.

**Untouched:** the hero (`AuctionView.baseline_dollars` → `auction_dollars`), `generate_auction_values`, and nomination (`val_by_id` / `nominate_order` at `simulation.py:119–128` keep using `auction_dollars`).

### The flag: `--bot-prices {espn,model}`, default `espn`

`run_auction_tournament` gains `bot_prices: Literal["espn", "model"] = "espn"` (the CLI default; `tournament_cli.py` adds `--bot-prices` with `choices=("espn","model")`). An unknown value raises `ValueError` (no silent fallthrough). **mypy-strict bridge:** argparse `Namespace` attributes are typed `str`/`Any`, so the CLI must explicitly narrow `args.bot_prices` to the `Literal` before passing it — e.g. a local `bot_prices: Literal["espn","model"] = "espn" if args.bot_prices == "espn" else "model"`, not a raw pass-through. (Analogous to the existing `--valuer` local-branch narrowing in `assistant/tournament_cli.py`, which narrows str→instance; there is no prior `Literal`-narrowing precedent, so this is the first.) **Parser placement:** the auction CLI uses a required `compare` subparser, so add `--bot-prices` to the **top-level** parser (before `add_subparsers`) to keep it mode-agnostic. There is a **single** `simulate_auction` call site (`tournament.py:96`, inside the `for name / for s` loops); the computed `bot_dollars` Series is threaded into it.

- `espn` **and** the pool carries usable ESPN values → compute `bot_dollars = espn_anchored_bot_prices(pool, config)` once before the loops, pass it to the `simulate_auction` call.
- `model`, **or** `espn` but the pool has no usable ESPN values → pass `bot_dollars=None` (the shared-value baseline). In the `model`-fallback case under `espn` mode, `warnings.warn(...)` names the fallback so a silent all-$1 market is impossible.

**Usability gate (exact):** `usable = ("espn_auction_dollars" in pool.columns) and bool(pool["espn_auction_dollars"].notna().any())`. The column is Optional on `VorpTableSchema` (`Series[pd.Int64Dtype] | None`) and `strict="filter"`, so on a weekly-path table the column is simply **absent** (the common R8 case) — the `in pool.columns` check is required, not just a null check.

**Degenerate-drift safety:** computing `espn_anchored_bot_prices` could in principle raise the pre-existing `ValueError` from `_allocate_surplus`'s floor-protection (if a sparse ESPN distribution leaves too few adjustable rows for negative drift). In `espn` mode, wrap the producer call; on `ValueError`, `warnings.warn(...)` and fall back to `model` (`bot_dollars=None`) rather than crashing the bake-off.

### Diagnostics (in the CLI, not the tournament)

The ESPN-vs-ours readout is computed **in `tournament_cli.py`**, not inside `run_auction_tournament` — which returns only `AuctionTournamentResult` (summaries/paired-diffs/scalars) and would discard any baseline frame. The CLI already holds `pool` + `config`. It must reuse the **same usability gate** as the tournament (`"espn_auction_dollars" in pool.columns and pool["espn_auction_dollars"].notna().any()`) and **skip the diagnostic when the gate fails** — otherwise `pool["espn_auction_dollars"]` raises `KeyError` in exactly the weekly-path/absent-column fallback case. When the gate passes it computes:

```
ref = pool.loc[pool["espn_auction_dollars"].notna(), ["gsis_id", "espn_auction_dollars"]] \
          .rename(columns={"espn_auction_dollars": "reference_dollars"})
diag = generate_auction_values(pool, config, reference_prices=ref)   # adds reference_dollars + value_delta
```

`reference_prices` must be this `{gsis_id, reference_dollars}` **DataFrame** (not a Series) — `generate_auction_values` selects `reference_prices[["gsis_id", "reference_dollars"]]` and runs `_reject_duplicate_gsis_ids` on it. NA ESPN rows are dropped before the rename so the diagnostic compares only priced players. The CLI prints/records a short our$-vs-ESPN$ `value_delta` summary (e.g. the largest positive/negative deltas) for the Run-H write-up. This is purely additive: it does not touch `auction_dollars`, the hero, or the sim — and `run_auction_tournament`'s signature/return are unchanged by it.

## Requirements

R1. `_allocate_surplus(value_signal, config)` is factored from `generate_auction_values`: it owns the logic of **lines 61–89** (surplus split, rounding, drift correction, floor protection, all-zero uniform fallback) and **returns** `rounded.astype(pd.Int64Dtype())`; the caller keeps the value-specific selection (lines 57–59) **and** the column assignment `pool_df["auction_dollars"] = _allocate_surplus(...)` (line 90). Inside, caller-local names are substituted for the parameter (`positive_vorp_sum` → `float(value_signal.sum())`, `positive_vorp` → `value_signal`); config scalars are recomputed from `config`. It **preserves `value_signal`'s index**, requires a non-null `float64` input, and returns `Int64` on every branch. `generate_auction_values` builds `value_signal = pool_df["vorp"].clip(lower=0.0)` and produces output **numerically identical** to before (pinned by an equivalence test).

R2. `espn_anchored_bot_prices(pool, config)` returns a `gsis_id`-indexed `Int64` Series over **every** pool row: in-pool = `_allocate_surplus` over `espn_auction_dollars` (NA→0 weight, so unpriced ≈ `min_bid`); out-of-pool = 0. The in-pool entries sum to `total_budget`. Guards an absent column (`in pool.columns`) as all-NA. Called on the same `pool` frame passed to `generate_auction_values`. Pure, unit-tested.

R3. `simulate_auction` **and** `_simulate_to_state` accept `bot_dollars: pd.Series | None = None`; `_simulate_to_state` attaches a `bot_dollars` column to `bd` after `nominate_order` is built (defaulting to `auction_dollars` when None; `Int64`-safe reindex/fillna). The 4 bot read-sites in `market.py` read `"bot_dollars"`; the `_value_tier` `in_pool` mask is unchanged. The hero, `generate_auction_values`, and nomination order are untouched.

R4. `run_auction_tournament` accepts `bot_prices: Literal["espn","model"] = "espn"`; in `espn` mode, when the usability gate passes, it computes `bot_dollars` once and threads it into the single `simulate_auction` call; otherwise (`model`, or no usable ESPN data, or a `ValueError` from the producer) it warns and passes `bot_dollars=None`. An unknown `bot_prices` value raises `ValueError`. The function's return type is unchanged (no diagnostic plumbing in it). (`tournament.py` does not currently `import warnings` — add it.)

R5. `tournament_cli.py` adds `--bot-prices {espn,model}` (default `espn`) passed through to `run_auction_tournament`, **and** (in `espn` mode) computes + prints the `reference_dollars`/`value_delta` diagnostic via `generate_auction_values(pool, config, reference_prices=<{gsis_id, reference_dollars} DataFrame>)`.

R6. **Back-compat / clean A/B**: with `bot_dollars=None` (and `--bot-prices model`), the tournament reproduces today's behavior **byte-for-byte** — `bot_dollars == auction_dollars`, nomination unchanged. The only difference in `espn` mode is the bot bid price. Existing auction tests pass unchanged (the new param defaults preserve every current call site).

R7. Conventions: `GsisId` canonical; `Position` referenced as the enum; `pd.Int64Dtype`/`pd.Float64Dtype` for nullable numerics; `bot_prices` typed `Literal[...]` (mypy-strict); `SCHEMA.validate(df)` with reassignment at boundaries; no new direct parquet I/O. `bot_dollars` is an **engine-internal** column on the indexed `bd` frame — it is **not** added to `AuctionValuesSchema` (`strict="filter"`, which `generate_auction_values` returns/validates); it is attached after that validation inside the simulation layer.

R8. **Data prerequisite for the bake-off**: populating `espn_auction_dollars` needs a fresh `refresh_external_projections` re-ingest + `generate_preset_vorp_tables.py` regen. The *code* does not require it (graceful degradation), but Run H does. **Confirm the regenerated preset table supports 16-team pool selection** (enough players per position for `_select_pool`, which raises `ValueError` if a position can't fill `n_teams × wanted`) before the 16-team Run H.

## Edge cases / failure modes

- **No ESPN data (weekly-path table, or stale snapshot pre-reingest):** column absent (`in pool.columns` false) or all-NA → warn + `model` fallback. Default-on never produces an all-$1 market.
- **Sparse / single-priced ESPN coverage (deep 16-team pool):** unpriced players ≈ $1; surplus concentrates on priced studs (the accepted deep-league inflation property; degenerate single-priced player absorbs ~whole budget). Pinned by a unit test; observed in Run H.
- **Degenerate negative drift (sparse priced set):** `_allocate_surplus`'s floor-protection (only `rounded > min_bid` rows are adjustable) can raise `ValueError` if too few priced rows exist to absorb negative drift. The `espn`-mode producer call is wrapped → warn + `model` fallback.
- **A priced player is out-of-pool:** `espn_anchored_bot_prices` returns 0 for out-of-pool rows regardless of ESPN value (out-of-pool players aren't rosterable; bots bid `min_bid`, same as today).
- **All in-pool players unpriced but column present (degenerate):** caught by the "≥1 non-null" usability gate → `model` fallback. (The helper's all-zero uniform branch is a safe backstop if ever reached.)
- **Scoring-basis mixing (carried from Slice 1):** the crowd `auctionValueAverage` is on ESPN's default-league basis while the expert fallback is ruleset-specific — two bases mixed in one column. Accepted per Slice 1; whether it distorts the bot field (esp. STANDARD) is observed, not corrected, here.

## Testing expectations

- **`_allocate_surplus` extraction (equivalence):** `generate_auction_values` output is byte-identical before/after the refactor on a fixture VORP table (`assert_frame_equal` against the pre-refactor output, or a committed gold frame). Include a fixture exercising the all-zero uniform branch and one exercising drift correction.
- **`espn_anchored_bot_prices` (new unit test):** (a) in-pool entries sum to `total_budget`; (b) on a **non-positive-drift fixture**, unpriced players get exactly `min_bid`; (c) priced players' dollars split the surplus proportionally — assert on a **non-drift fixture** (so strict monotonicity in `espn_auction_dollars` holds), since the drift loop's ±1 bumps can otherwise break strict ordering between near-tied players (assert weak monotonicity ±1 if a drift fixture is used); (d) out-of-pool rows = 0; (e) every entry ≥ `min_bid`; (f) absent/all-NA column → uniform fallback (sum still `total_budget`); (g) result dtype `Int64` on every branch; (h) **deep-league inflation bound**: on a fixture with many unpriced players, the top priced player's bot dollar is inflated above its ESPN value by a factor pinned to a concrete number (guards degenerate blow-up).
- **Seam (`simulation.py`):** with `bot_dollars=None`, `bd["bot_dollars"] == bd["auction_dollars"]` and a full `simulate_auction` run is identical to baseline (same RNG → same rosters). With a provided `bot_dollars`, the bots' winning prices track the ESPN vector (a stud ESPN over-prices clears higher; an unpriced startable player clears near `min_bid`). Nomination order is unchanged between the two.
- **Bot read-sites (`market.py`):** each archetype's `max_bid` centers on `bot_dollars` (not `auction_dollars`) when the two differ; `_value_tier` ranks by `bot_dollars` over the unchanged `in_pool` mask.
- **Flag (`tournament.py`/CLI):** `--bot-prices espn` on a pool **without** ESPN values warns and produces the `model`-mode result (equality test vs `--bot-prices model`); `--bot-prices espn` on a pool **with** ESPN values differs from `model`; an unknown `bot_prices` value raises `ValueError`. The CLI diagnostic populates `reference_dollars`/`value_delta` (DataFrame-shaped `reference_prices`).
- **No-regression:** the full auction suite (`test_market`, `test_simulation`, `test_tournament`, bid-strategy, VORP/auction-values tests) passes unchanged (R6).
- Per `CLAUDE.md`, run `pytest -k "auction or schemas"` (touches the auction engine; no schema change, but the auction-values seam is adjacent) plus the targeted new tests.

## Phasing

~4 tasks, each ≤5 files (edit-site count called out where it exceeds file count):
1. **Allocator extraction + bot price vector** — extract `_allocate_surplus` (preserving the index/op-order contract), add `espn_anchored_bot_prices` (`auction.py`) + equivalence test + `espn_anchored_bot_prices` unit tests. (No behavior change yet.)
2. **Engine seam** — `bot_dollars` param + column in `simulation.py` (both `simulate_auction` and `_simulate_to_state`); switch the 4 bot read-sites in `market.py` to `bot_dollars` + seam/equality tests.
3. **Flag plumbing + diagnostics** — `bot_prices: Literal[...]` in `run_auction_tournament` (compute vector, usability gate, `ValueError`-fallback) + `--bot-prices` and the CLI diagnostic readout in `tournament_cli.py` + flag tests.
4. **Validation bake-off** — re-ingest `external_projections` + regen preset tables (R8), run the 8-model field vs the ESPN-anchored bot market at half-PPR **12-team and 16-team**, write **Run H** into `reports/auction_tournament_validation_2026.md`, update TODO #49c + `project_management.md`. **Run-H artifact (acceptance):** a Run-H section with the per-model table (all 8 models × the standard metric set × {12,16}-team) and the ESPN-vs-ours `value_delta` summary, framed as data-gathering (no winner declared).

## Open questions for later (Slice 3+)

- **Room-driven nomination:** should nomination order follow the bots' (ESPN) view in `espn` mode rather than the shared SOS order? Kept on SOS here for a clean A/B; revisit if nomination sequence proves load-bearing.
- **SOS-backfill variant:** an A/B of choice-B (ESPN-only, $1 floor) vs the ESPN+SOS-backfill variant, to measure how much the deep-pool $1 floor drives the hero's edge.
- **Deep-league inflation guard:** if Run H shows 16-team stud inflation is unrealistic, cap the per-player bot dollar at (e.g.) a multiple of its ESPN value.
- **Multi-year (#49a):** once Run H is read, fold ESPN-anchored bots into the multi-year averaging before the September decision.
