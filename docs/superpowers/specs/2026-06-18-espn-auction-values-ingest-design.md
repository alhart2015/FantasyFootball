# ESPN auction-value ingest (Slice 1) — design

**Status:** approved (brainstormed 2026-06-18). Proceeding to plan + execute.
**Owner:** draft-hub / auction + ingest.
**Depends on:** the existing `external_projections` ingest, the consensus blend (`build_consensus`), and the preset VORP-table generator (`scripts/generate_preset_vorp_tables.py`).

## Problem

The auction bid-model tournament has a **shared-value problem**: the bot field (`market.py`) centers its willingness-to-pay on the *same* `auction_dollars` the hero uses, our own SOS-computed value, plus mean-zero noise. Because opponents price every player off the hero's own numbers, the hero has **no informational edge** — it can never be "more right than the room," and the market leaves no systematically exploitable bargain (only random noise). This is the leading suspect for why every hero strategy sits at or below the field's fair-share baseline (Runs E–G). Real opponents are beatable because they're *predictably* biased (name premiums, position runs); these bots are only randomly wrong.

The fix is to anchor the **bots** on **real, human-facing auction values** (with real human bias) while the **hero** keeps our projection-based value — so the hero's edge becomes the gap between the two. This spec is **Slice 1**: get those real values ingested, plumbed, and validated onto the pool. Slice 2 (wire the bot WTP + run the bake-off) is a separate spec.

### Data availability (verified 2026-06-18, live API probe)

- **ESPN** `kona_player_info` (the payload we *already* fetch for ADP) carries two human-facing auction values per player, both currently discarded:
  - `ownership.auctionValueAverage` — the **crowd/behavioral** average (what real ESPN drafters paid; embeds human bias). 2026: Gibbs $58.67. **Current-season-only** — 2025 had already reset to $0.
  - `draftRanksByRankType.{PPR,STANDARD}.auctionValue` — ESPN's **expert/editorial** value, split by scoring. **Persists historically** (2025 still returned $54/$53).
- **Sleeper** `projections/nfl/{season}` exposes a dozen ADP flavors but **no auction value field at all** (verified by recursive key scan). The auction-value anchor is **ESPN-only**.

## Goals

1. Capture ESPN's auction-value fields in the `external_projections` ingest (no new endpoint — same payload).
2. Carry them through `build_consensus` into the consensus snapshot.
3. Resolve a single per-player **`espn_auction_dollars`** column onto the pool/VORP table, ruleset-aware, using the **crowd-now / expert-fallback** rule.
4. No behavior change: the tournament, bots, and hero are untouched. Slice 1 only makes the column exist and validate.

## Non-goals

- **Not** wiring the bot WTP to the new column, **not** running a bake-off, **not** touching `market.py` / the engine / `bid_strategy.py` — that is Slice 2.
- **Not** rescaling ESPN dollars to the target league's budget (raw ESPN $ is stored; rescaling is a Slice-2 consumption concern).
- **Not** a multi-source "market value" blend (Sleeper has no auction value); the column is ESPN-only, hence the name `espn_auction_dollars`.
- **Not** backfilling historical crowd values (impossible — ESPN resets them); historical runs fall back to the expert value.

## Chosen approach (A — extend the existing pipeline)

The ESPN auction value rides the established `external_projections → consensus → vorp-table` flow.

### Canonical column names (single source of truth — use verbatim across parsers, all three schemas, the resolver, and tests to avoid name drift)

| Column | Where | Dtype | Notes |
|---|---|---|---|
| `espn_auction_value_avg` | external + consensus | `Float64` (Optional) | crowd `ownership.auctionValueAverage` (fractional) |
| `espn_auction_value_ppr` | external + consensus | `Float64` (Optional) | expert `draftRanksByRankType.PPR.auctionValue` |
| `espn_auction_value_std` | external + consensus | `Float64` (Optional) | expert `draftRanksByRankType.STANDARD.auctionValue` |
| `espn_auction_dollars` | vorp/pool table | `Int64` (Optional) | resolved, rounded; the Slice-1 deliverable |

### Backward-compatibility — all four columns are **Optional**, not required

This is load-bearing for "no behavior change." Existing on-disk partitions (`external_projections`, the consensus snapshot) and the weekly `generate_vorp_table` path were written **without** these columns, and `VorpTableSchema`/`ExternalProjectionSchema` validate those frames on read. A **required** new column would raise `SchemaError: column not in dataframe` on every stale partition and every weekly-path table — a real regression. So each new column is declared `Optional` (pandera `Series[...] | None`, exactly as `VorpTableSchema.consensus_adp`/`full_name` already are), every stage treats absence as NA, and **populating the columns requires a fresh `refresh_external_projections` re-ingest** (the stale snapshot won't carry them). Note: `data/raw/external_projections` may be absent on a given machine — re-ingest is a prerequisite for `build_preset_table` to produce a populated column.

### The resolved value: `espn_auction_dollars`

Per player, resolved where the league ruleset is known (the pool/VORP stage). `ruleset` is a `Ruleset` pydantic model (there is **no Ruleset enum** — see R7); match on `ruleset.name` against the sanctioned constants `"ESPN_PPR"` / `"ESPN_HALF"` / `"STANDARD"`:

```
crowd = frame["espn_auction_value_avg"]   # may be absent (Optional) -> treat as NA
if crowd is present and > 0:                       # crowd: what humans actually paid
    value = crowd
elif ruleset.name in ("ESPN_PPR", "ESPN_HALF"):    # half-PPR uses the PPR expert (closest)
    value = frame["espn_auction_value_ppr"]
elif ruleset.name == "STANDARD":
    value = frame["espn_auction_value_std"]
else:
    value = NA
espn_auction_dollars = round(value) as pd.Int64Dtype()   # NA-safe; NA stays NA
```

The raw external/consensus columns stay `Float64` (preserve the fractional 58.67); only the **resolved** column rounds to `Int64`. Rounding is round-half-to-even (pandas `Series.round`); the sub-dollar tie rule is immaterial for auction dollars. **Use `pd.Int64Dtype()` for the cast — NOT `.astype("int64")`** (the non-nullable cast that `generate_auction_values` uses crashes on NA; we have NA).

Implement this **vectorized over the Series** — a `crowd.notna() & (crowd > 0)` mask plus `.where`/positional fill — not scalar `if`/`.apply` (`if crowd:` on a Series raises "truth value is ambiguous"). If an input column is entirely absent from `frame`, treat it as all-NA.

### A — Ingest (`ingest/external_projections.py`, `schemas.ExternalProjectionSchema`)

- `parse_espn_players` extracts three more fields per player via **defensive `.get`** (these are net-new payload keys — the parser currently reads only `draftRanksByRankType.PPR.rank`, never `auctionValue` and never `STANDARD`): `ownership.auctionValueAverage`, and `(draftRanksByRankType.get("PPR") or {}).get("auctionValue")` / `... .get("STANDARD") ...`. A live probe (2026-06-18) confirmed all three keys present (Gibbs crowd $58.67, PPR/STD expert $57/$56; 2025 expert $54/$53, crowd $0). Normalize ≤0 → `None` — the `≤0 → None` precedent is the **parser-local `espn_adp` variable** at `parse_espn_players`, canonicalized to the schema column `adp` (the schema has no `espn_adp` column).
- **Edit `_to_canonical` too** (not free): emit the three columns null-filled when absent. Note the `espn_draft_rank` precedent resolves absence via the `rank_col` *parameter* threaded through `source_specs`; these columns have no such parameter — they are simply **present in the ESPN parsed frame and absent in the Sleeper one** (`parse_sleeper_projections` returns a fixed column list without them). So guard on presence: `keyed[col] if col in keyed.columns else null_col` for each, cast to `Float64` alongside the existing `astype("Float64")` loop so all-NA Sleeper columns survive `pd.concat`. (Threading them through `source_specs` like `rank_col` is an equivalent alternative; the presence check is simpler.)
- `ExternalProjectionSchema` gains the three `Float64` columns as **Optional** (`Series[...] | None`) so stale partitions still validate. Note: these are the **first** Optional columns on `ExternalProjectionSchema` — the `| None` precedent lives on `VorpTableSchema.consensus_adp`/`full_name`, not within this schema; the mechanism is identical.

### B — Consensus pass-through (`consensus/blend.py`, `schemas.ConsensusProjectionSchema`)

`build_consensus` has **no generic pass-through** — it returns `df[list(_OUTPUT_COLUMNS)]` (a hard allowlist) and builds each per-group `rec` field-by-field. So carrying the value is a coordinated edit, not free:

- Add the three columns to `_OUTPUT_COLUMNS` **and** to `_empty_output()` (which builds columns from that tuple — the empty path must match or diverge).
- **Guard for the columns being absent from `external` entirely.** They are Optional, so a stale (pre-reingest) snapshot or any existing caller's frame won't carry them — and `build_consensus(external, ...)` is called directly in `build_preset_table` (R8). Without a guard, `grp[col]` raises `KeyError`, breaking the ~39 existing blend tests (R6) and the stale-partition path (R8). Seed any missing column to all-NA before the group loop (or use `col in grp.columns`).
- When present, set each `rec` field to the **first non-null value across the group** (`grp[col].dropna().iloc[0]` only `if not grp[col].dropna().empty else pd.NA` — the non-empty guard avoids `IndexError` on an all-NA group). This is deliberately **NOT** the `identity_row` pick used for `full_name`: a player whose identity row is the Sleeper row (ESPN present but not stat-bearing) would lose the ESPN auction value under an identity-row pick. First-non-null is correct because the columns are ESPN-only (only ESPN rows ever carry a value).
- Cast the columns to `Float64` explicitly in the dtype block (pd.NA does not survive otherwise — mirrors the existing `astype("Int64")` casts).
- `ConsensusProjectionSchema` gains the three columns as **Optional**.

### C — Resolve + land on the pool (`scripts/generate_preset_vorp_tables.py`, `schemas.VorpTableSchema`)

- A new pure helper `resolve_espn_auction_dollars(frame, ruleset) -> pd.Series` implements the rule above (NA-safe `pd.Int64Dtype()`; absent input column → all-NA). **`frame` is the `consensus` frame**, the only frame that still carries the three columns: `consensus_to_season_projections` builds a fresh column dict and **drops** them (and filters to `has_points`), so they are **not** on the VORP `table`. Reading from `table` would silently produce all-NA.
- `build_preset_table` resolves from `consensus` (ruleset = `preset.league_config.ruleset`) into a `gsis_id`-keyed `espn_auction_dollars`, then **merges** it onto the table at the same post-`generate_vorp_table` seam where `consensus_adp`/`full_name` are attached (`cols = consensus[[...]]; table = table.merge(cols, on="gsis_id", how="left")`) — **not** inside `generate_vorp_table` (whose own `_OUTPUT_COLUMNS` is fixed and feeds the weekly path).
- `VorpTableSchema` gains `espn_auction_dollars` as an **Optional** `Int64` column (exactly like `consensus_adp`/`full_name`), so the weekly `generate_vorp_table` output — which never has it — still validates. **This Optionality is what preserves R6.**

### Slice-2 seam (informational, not built here)

`generate_auction_values(vorp_table, league_config, reference_prices=None)` already exposes a `reference_prices → reference_dollars` seam and computes `value_delta = auction_dollars − reference_dollars` (NA reference → NA delta). Slice 2 will pass `espn_auction_dollars` in as `reference_prices` and point the bot WTP at `reference_dollars`, leaving the hero on `auction_dollars`. **A large fraction of in-pool players will have NA ESPN value** (deep players, rookies, Sleeper-only) — how the bots handle a missing reference is the central unresolved Slice-2 question (Open Question #2), not something Slice 1 solves. Slice 1 leaves `reference_prices=None` (unchanged); the column lands on the **pool/VORP table only** and is silently dropped at the `generate_auction_values` boundary (its `_OUTPUT_COLUMNS` excludes it) — it does **not** appear on `AuctionValuesSchema` in Slice 1.

## Requirements

R1. `parse_espn_players` extracts all three ESPN auction fields via defensive `.get` (net-new keys), normalizing ≤0 → `None` on each. A player missing the `ownership`/`draftRanksByRankType` block (or a specific sub-key) gets `None` for that field, no crash. `_to_canonical` is edited to emit the three columns, null-filled when absent via a `col in keyed.columns` presence guard (Sleeper's parsed frame lacks them).
R2. `ExternalProjectionSchema` gains the three `Float64` columns as **Optional** (stale partitions written without them still validate). Fresh ESPN rows populate them; Sleeper rows are NA (`Float64`/`pd.NA`, not `float64`/NaN).
R3. `build_consensus` carries the ESPN values via a **first-non-null-across-group** reduction (added to the per-group record, `_OUTPUT_COLUMNS`, and `_empty_output`, with explicit `Float64` casts), **guarding for the columns being absent from `external`** (seed-to-NA so existing blend tests and stale partitions don't `KeyError`); `ConsensusProjectionSchema` gains the three columns as **Optional**. A Sleeper-only player has NA.
R4. `resolve_espn_auction_dollars(frame, ruleset)` operates **vectorized** on the `consensus` frame (the columns don't survive `consensus_to_season_projections`); implements crowd-now/expert-fallback by `ruleset.name`; rounds NA-safely via `pd.Int64Dtype()`; returns NA when no ESPN value exists (incl. when the input column is absent). Pure, unit-tested.
R5. `build_preset_table` **merges** a `espn_auction_dollars` column at the post-`generate_vorp_table` seam (with `consensus_adp`/`full_name`); `VorpTableSchema` gains it as an **Optional** `Int64` column so the weekly `generate_vorp_table` path (which lacks it) still validates.
R6. **No behavior change**: `generate_auction_values`, `market.py`, `bid_strategy.py`, the engine, and the tournament are untouched; the tournament still passes `reference_prices=None`. All four new columns are Optional, so every existing partition and the weekly VORP path validate unchanged. The existing auction/VORP tests pass unchanged.
R7. Conventions: `GsisId` canonical; `Position` referenced as the enum. **`Ruleset` is a pydantic model with a free-form `name: str` (no enum)** — match on `ruleset.name` against the sanctioned constants `"ESPN_PPR"`/`"ESPN_HALF"`/`"STANDARD"` (this is the sanctioned form; the "reference enums not strings" rule applies to `Position`/`Team`/etc., not to ruleset). `pd.Float64Dtype`/`pd.Int64Dtype` for nullable numerics; `SCHEMA.validate(df)` with reassignment at every boundary; no new direct parquet I/O outside the sanctioned paths.
R8. **Re-ingest prerequisite**: populating the columns requires a fresh `refresh_external_projections` run; stale on-disk snapshots validate (Optional) but carry NA. `build_preset_table` needs a current `external_projections` partition present (it may be absent on a given machine).

## Edge cases / failure modes

- **Crowd value = 0 (past season or no live data):** treated as absent → expert fallback. This is the intended historical path.
- **Expert value = 0 (undraftable/deep player):** normalized to `None`; if crowd also absent → `espn_auction_dollars` NA.
- **ESPN row present but no auction value (ranked-but-unpriced deep player; crowd=0, expert=0):** → NA. Realistic and must be tested (see Testing).
- **Player only in Sleeper (ESPN has no row):** all three ESPN columns NA → `espn_auction_dollars` NA.
- **ESPN player with no projection block:** `parse_espn_players` `continue`s before building the row (existing behavior), so such a player contributes no auction value. Not a loss for Slice 1 — a player with no projection has no consensus points and is therefore not in the VORP pool anyway, so the missing value is moot for the pool column.
- **Placeholder-gsis rookie / hash collision:** value rides the existing id_map join. If two distinct rookies collide on one placeholder gsis (rare, already warned, not deduped), the first-non-null reduction picks one arbitrarily — **no worse than the existing name/stat mis-attribution under the same collision**, which the codebase already accepts and warns about. Pre-existing, bounded; not addressed here.
- **Scoring-basis mismatch (all rulesets, not just half-PPR):** the crowd `auctionValueAverage` is ESPN's fixed default-league average (`leaguedefaults/3`), whose scoring does **not** vary with the target ruleset. So within one table, crowd-present players carry a crowd dollar on ESPN's default basis while crowd-absent players carry the ruleset-specific expert — two bases mixed in one column, and the divergence is largest for STANDARD. Accepted: the crowd value is the *behavioral* signal we want, and Slice 1 only lands it; whether the mixing matters is a Slice-2 modeling question. Half-PPR additionally has no exact expert (uses PPR). Documented; revisit if a basis-matched source appears.
- **Fetch window:** `fetch_espn` pulls the top ~800 by percent-owned; players outside that window get no ESPN row → NA regardless of whether ESPN has a value. Minor coverage caveat.

## Testing expectations

- **Ingest parsers** (`tests/test_ingest/test_external_projections.py`): synthetic ESPN payloads — (a) crowd present, (b) crowd=0 + expert present, (c) ESPN present but crowd=0 **and** expert=0 → NA (the ranked-unpriced case), (d) missing `ownership`/`draftRanksByRankType` sub-keys → NA no-crash. Assert ≤0 → NA. **Assert the column dtype is `Float64` (not `float64`/NaN)** for both ESPN and Sleeper rows — guards the CLAUDE.md dtype-regression trap. `ExternalProjectionSchema.validate` round-trips, **and a frame lacking the columns entirely still validates** (Optional).
- **Consensus** (blend tests): a two-source player keeps the ESPN value; a player whose **identity row is the Sleeper row but whose ESPN row carries a value** still gets the value (the first-non-null vs identity-row distinction); a Sleeper-only player gets NA; `_empty_output()` carries the columns; `ConsensusProjectionSchema.validate` round-trips; columns are `Float64`.
- **Resolve** (new unit test): each branch (crowd>0 → crowd; crowd=0 → expert by `ruleset.name` for ESPN_PPR / ESPN_HALF → ppr, STANDARD → std; both absent → NA; **input column entirely absent → all-NA, no crash**); result dtype `Int64`; a fractional crowd value rounds (e.g. 58.67 → 59).
- **Pool** (`tests/test_scripts/test_generate_preset_vorp_tables*`): a generated table built from a fixture carrying ESPN values has `espn_auction_dollars`; `VorpTableSchema.validate` passes; **a weekly-path `generate_vorp_table` output (no column) still validates** (Optional guard for R6).
- **No-regression**: the auction tournament/bid-strategy/simulation/VORP tests pass unchanged (R6).
- Per `CLAUDE.md`, run `pytest -k "ingest or store or schemas"` (this touches pandera schemas + an ingest path).

## Phasing

~4 tasks, each ≤5 files. (Edit-site count is higher than file count — call out the hidden sites so they aren't missed.)
1. **Ingest** — `parse_espn_players` **+ `_to_canonical`** (two sites in `external_projections.py`) + `ExternalProjectionSchema` (Optional) + parser unit tests.
2. **Consensus pass-through** — `build_consensus` loop **+ `_OUTPUT_COLUMNS` + `_empty_output` + dtype casts** (`blend.py`) + `ConsensusProjectionSchema` (Optional) + blend tests.
3. **Resolve + pool** — `resolve_espn_auction_dollars` + `build_preset_table` merge + `VorpTableSchema` (Optional) + tests.
4. **Integration + no-regression sweep** — re-ingest a fresh `external_projections` snapshot (R8 prerequisite), regenerate a sample preset table, run the schema-seam + weekly-path validation tests, confirm the auction suite is unchanged.

## Open questions for later (Slice 2+)

- Rescaling raw ESPN dollars to the target league's budget × teams (ESPN's default league differs from our 12/16-team $200 configs).
- **(central Slice-2 question)** How bots handle the **large NA fraction** — many in-pool players (deep/rookie/Sleeper-only) have no ESPN value; fall back to our `auction_dollars`? min_bid? — and whether bots read `espn_auction_dollars` directly or via the `reference_dollars` seam.
- Whether the crowd/expert scoring-basis mixing (Edge cases) materially distorts the bot field, esp. for STANDARD.
- Whether to snapshot the crowd value live each preseason so future multi-year runs can use behavioral (not just expert) values going forward.
