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

### The resolved value: `espn_auction_dollars`

Per player, resolved where the league ruleset is known (the pool/VORP stage):

```
if espn_auction_value_avg is present and > 0:   # crowd: what humans actually paid
    espn_auction_dollars = espn_auction_value_avg
elif ruleset is PPR or HALF:                     # half-PPR uses the PPR expert (closest)
    espn_auction_dollars = espn_auction_value_ppr
elif ruleset is STANDARD:
    espn_auction_dollars = espn_auction_value_std
# else (no ESPN value at all, e.g. rookie / Sleeper-only / deep player):
    espn_auction_dollars = NA
```

Stored as raw ESPN dollars (a nullable integer-dollar figure). Nullable end-to-end.

### A — Ingest (`ingest/external_projections.py`, `schemas.ExternalProjectionSchema`)

- `parse_espn_players` extracts three more fields per player: `ownership.auctionValueAverage`, `draftRanksByRankType.PPR.auctionValue`, `draftRanksByRankType.STANDARD.auctionValue`. Normalize ≤0 → `None` (mirrors the existing `espn_adp <= 0 → None` rule — ESPN encodes "no data" as 0).
- `ExternalProjectionSchema` gains three nullable `Float64` columns: `espn_auction_value_avg`, `espn_auction_value_ppr`, `espn_auction_value_std`. ESPN-source rows populate them; **Sleeper rows are NA** (`_to_canonical` null-fills, exactly as it already does for `espn_draft_rank`).

### B — Consensus pass-through (`consensus/blend.py`, `schemas.ConsensusProjectionSchema`)

- The three auction columns are **ESPN-only and not blended** across sources. `build_consensus` carries the non-null ESPN value per `gsis_id` through to the consensus record (take the first non-null across the group, same shape as the identity/`full_name` pick).
- `ConsensusProjectionSchema` gains the three nullable columns.

### C — Resolve + land on the pool (`scripts/generate_preset_vorp_tables.py`, `schemas.VorpTableSchema`)

- A new pure helper `resolve_espn_auction_dollars(frame, ruleset) -> pd.Series` implements the resolution rule above. The raw external/consensus columns are `Float64` (the crowd average is fractional, e.g. 58.67); the resolver **rounds to the nearest whole dollar** so the landed column is integer dollars, matching the auction-dollar convention and `reference_dollars` (which it feeds in Slice 2). NA stays NA.
- `build_preset_table` calls it (the ruleset is `preset.league_config.ruleset`) and attaches one nullable `espn_auction_dollars` column to the table it writes.
- `VorpTableSchema` gains a nullable `espn_auction_dollars` column (`Int64` dollars, like `auction_dollars`/`reference_dollars` in `AuctionValuesSchema`).

### Slice-2 seam (informational, not built here)

`generate_auction_values(vorp_table, league_config, reference_prices=None)` already exposes a `reference_prices → reference_dollars` seam and computes `value_delta = auction_dollars − reference_dollars`. Slice 2 will pass `espn_auction_dollars` in as `reference_prices` and point the bot WTP at `reference_dollars`, leaving the hero on `auction_dollars`. Slice 1 leaves `reference_prices=None` (unchanged); it only makes the data available.

## Requirements

R1. `parse_espn_players` extracts the three ESPN auction fields, normalizing ≤0 → `None`. Players with no `ownership`/`draftRanksByRankType` block get `None` for the missing field (no crash).
R2. `ExternalProjectionSchema` validates with three nullable `Float64` auction columns; Sleeper rows carry NA for all three.
R3. `build_consensus` carries the ESPN auction values through per `gsis_id`; `ConsensusProjectionSchema` validates with the three nullable columns. A player present only in Sleeper has NA.
R4. `resolve_espn_auction_dollars(frame, ruleset)` implements crowd-now/expert-fallback-by-ruleset exactly; returns NA when no ESPN value exists. Pure, unit-tested.
R5. `build_preset_table` attaches a nullable `espn_auction_dollars` column; `VorpTableSchema` validates it (nullable Int64 dollars).
R6. **No behavior change**: `generate_auction_values`, `market.py`, `bid_strategy.py`, the engine, and the tournament are untouched; the tournament still passes `reference_prices=None`. The existing auction tests pass unchanged.
R7. Conventions: `GsisId` canonical; reference enums (`Position`, ruleset by name) not string literals; `pd.Float64Dtype`/`pd.Int64Dtype` for nullable numerics; `SCHEMA.validate(df)` with reassignment at every boundary; no new direct parquet I/O outside the sanctioned paths.

## Edge cases / failure modes

- **Crowd value = 0 (past season or no live data):** treated as absent → expert fallback. This is the intended historical path.
- **Expert value = 0 (undraftable/deep player):** normalized to `None`; if crowd also absent → `espn_auction_dollars` NA.
- **Player only in Sleeper (ESPN has no row):** all three ESPN columns NA → `espn_auction_dollars` NA.
- **Placeholder-gsis rookie:** ESPN may still carry a value keyed to its real id; if the rookie is a placeholder in our id_map it joins as today, value rides along if present, else NA. No special-casing.
- **Half-PPR ruleset:** uses the **PPR** expert value as the closest available (ESPN exposes only PPR/STANDARD experts). Documented; revisit if a half-specific source appears.
- **`ownership.auctionValueAverage` scoring basis:** it is ESPN's default-league average (leaguedefaults/3), whose scoring may not exactly match the target ruleset. Accepted as a known caveat — it is the *behavioral* signal we want; the expert fallback is scoring-split.

## Testing expectations

- **Ingest parsers** (`tests/test_ingest/test_external_projections.py`): synthetic ESPN payloads — (a) crowd present, (b) crowd=0 + expert present, (c) all absent; assert the three columns parse and ≤0 → NA. Sleeper rows NA. `ExternalProjectionSchema.validate` round-trips.
- **Consensus** (`tests/test_consensus` / blend tests): a two-source player keeps the ESPN value; a Sleeper-only player gets NA; `ConsensusProjectionSchema.validate` round-trips.
- **Resolve** (new unit test): each branch (crowd>0 → crowd; crowd=0 → expert-by-ruleset PPR/HALF vs STANDARD; both absent → NA).
- **Pool** (`tests/test_scripts/test_generate_preset_vorp_tables*`): the generated table carries `espn_auction_dollars`; `VorpTableSchema.validate` passes.
- **No-regression**: the auction tournament/bid-strategy/simulation tests pass unchanged (R6).
- Per `CLAUDE.md`, run `pytest -k "ingest or store or schemas"` (this touches pandera schemas + an ingest path).

## Phasing

~4 tasks, each ≤5 files:
1. **Ingest** — `parse_espn_players` + `ExternalProjectionSchema` + parser unit tests.
2. **Consensus pass-through** — `build_consensus` + `ConsensusProjectionSchema` + blend tests.
3. **Resolve + pool** — `resolve_espn_auction_dollars` + `build_preset_table` + `VorpTableSchema` + tests.
4. **Integration + no-regression sweep** — schema-seam tests, regenerate a sample preset table, confirm the auction suite is unchanged.

## Open questions for later (Slice 2+)

- Rescaling raw ESPN dollars to the target league's budget × teams (ESPN's default league differs from our 12/16-team $200 configs).
- Whether bots read `espn_auction_dollars` directly or via the `reference_dollars` seam, and how to handle players with NA ESPN value (fall back to our `auction_dollars`? min_bid?).
- Whether to snapshot the crowd value live each preseason so future multi-year runs can use behavioral (not just expert) values going forward.
