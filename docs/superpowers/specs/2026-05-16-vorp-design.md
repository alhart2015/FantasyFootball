# VORP — Value Over Replacement Player — Design

**Status:** draft (brainstorming, 2026-05-16). Ready for user review.
**Date:** 2026-05-16
**Author:** alden + claude
**Sub-project of:** FantasyFootball / Draft Hub
**Branch:** `feat/vorp` cut from `feat/auction-values` at `37693ae` (PR #40 not yet merged; this spec ships against an unmerged dependency by necessity — see §6).

**Depends on:** `LeagueConfig` from `src/projections/draft/league_config.py` (shipped on `feat/auction-values`). `aggregate_to_season` and `ProjectionSeasonSchema` from `src/projections/aggregation/season.py` + `src/projections/schemas.py` (already on `main`).

**Consumed by (contract is fixed):** `generate_auction_values(vorp_table, ...)` in `src/projections/draft/auction.py` (shipped on `feat/auction-values`). The auction spec pins the input contract: a parquet with columns `(gsis_id, position, season_mean_fpts, vorp)`, no duplicate `gsis_id`, covering every `Position` referenced in `LeagueConfig.roster_slots`. This spec MUST produce that contract exactly.

**Related specs / docs:**
- `docs/superpowers/specs/2026-05-16-auction-values-design.md` — downstream consumer; §3 step 1 (pool selection) and §6 (open items) describe what VORP must satisfy.
- `draft_ready_checklist.md` §2a "Foundational valuation" — names VORP as the dollar-zero gating item for both snake-draft and auction surfaces. §2b.1 (snake-draft cheat sheet) and §2c.1 (auction $ generator, shipped) both consume this spec's output.
- `TODO.md` #10 — K/DST positions still unbuilt; informs §6 position-scope decision below.

---

## 1. Overview

The Draft Hub needs a way to convert per-player season-mean fantasy points into a single position-aware comparable scalar — VORP. Two downstream tools consume it: the auction $ generator (shipped) and the snake-draft cheat sheet (next-up spec). Both need to compare a QB1 against a RB12 against a WR40 on one axis.

VORP is `season_mean_fpts(player) − replacement_fpts(position, league_config)`. This spec defines `replacement_fpts` precisely, builds the conversion as a pure function, and exposes a CLI that turns weekly projections into a VORP parquet that the auction script (and the future snake script) can consume.

### 1.1 Goals (in scope)

- **New module `src/projections/draft/vorp.py`** with one public function `generate_vorp_table(season_projections, league_config) -> pd.DataFrame`. Pure transform; no I/O.
- **New schema `VorpTableSchema` in `src/projections/schemas.py`** validating the output. Must conform exactly to the input contract auction expects (column names, dtypes, uniqueness).
- **New CLI `scripts/generate_vorp_table.py`** that reads a weekly-projections parquet partition, aggregates to season, runs `generate_vorp_table`, writes the output. CSV + parquet output both supported (sniff by extension), matching the auction CLI's pattern.
- **Tests in `tests/test_draft/test_vorp.py`** + a CLI integration test in `tests/test_scripts/test_generate_vorp_table_cli.py` + a schema round-trip test appended to `tests/test_schemas/test_dataframe_schemas.py`. Coverage in §5.
- **No new ingest, no new model, no new feature builder, no new store partition.** Pure transform over already-published `ProjectionSeasonSchema` rows.
- **Recommendation: pool-boundary replacement-level** (§3 below) as the default, with the strict-positional alternative documented but not implemented in v1.

### 1.2 Non-goals (deferred)

- **No upside-sensitive VORP** (using `season_p90` instead of `season_mean`). Add as a `--variant` flag in a follow-up spec only if the snake-draft cheat sheet needs it; auction is mean-only by design.
- **No tiering.** Spec §2a "tier breaks" in the checklist is a separate concern — happens on top of VORP, not inside it.
- **No confidence bands per ranking** ("floor rank" / "ceiling rank" from p10/p90). Also a separate concern; lives in whichever cheat-sheet / UI surface wants it.
- **No K/DST VORP generation.** Codebase produces no K/DST projections today (TODO #10). VORP emits rows only for positions present in the projection input; the CLI warns loudly if `LeagueConfig.roster_slots` requires K/DST. Auction's existing contract errors clearly downstream.
- **No rookie projection handling.** Same gap as K/DST — rookies have no trailing-window history; not in projection input → not in VORP output. Tracked in `draft_ready_checklist.md` §1a.
- **No multi-ruleset output in one run.** VORP is ruleset-scoped because replacement-level depends on scoring. One `LeagueConfig` → one ruleset → one run.
- **No persistence to the store layer.** Output is a local file the user keeps, same convention as auction values.
- **No `predict_season.py` generalization.** Out of scope; tracked in `draft_ready_checklist.md` §1b. This spec's CLI accepts any weekly-projections partition path — works against `predict_2024.py`'s output today and against a future `predict_season.py` without changes.

---

## 2. Architecture

```
src/projections/draft/
├── __init__.py                                  (edited — re-export generate_vorp_table)
├── league_config.py                             (unchanged)
├── auction.py                                   (unchanged)
└── vorp.py                                      (new — generate_vorp_table + replacement-level math)

src/projections/schemas.py                       (edited — append VorpTableSchema after ProjectionSeasonSchema)

scripts/
├── generate_auction_values.py                   (unchanged)
└── generate_vorp_table.py                       (new — CLI)

tests/test_draft/
├── test_league_config.py                        (unchanged)
├── test_auction.py                              (unchanged)
└── test_vorp.py                                 (new)

tests/test_scripts/
├── test_generate_auction_values_cli.py          (unchanged)
└── test_generate_vorp_table_cli.py              (new)

tests/test_schemas/test_dataframe_schemas.py     (edited — append VorpTableSchema round-trip)
```

### 2.1 Public function

```python
def generate_vorp_table(
    season_projections: pd.DataFrame,        # validated against ProjectionSeasonSchema
    league_config: LeagueConfig,
) -> pd.DataFrame:                           # validated against VorpTableSchema
    ...
```

Pure function. No I/O, no side effects, no caching. Algorithm in §3. Input contract:

- `season_projections` is `ProjectionSeasonSchema`-validated on entry (we re-validate to be defensive). Columns of interest: `gsis_id`, `position`, `ruleset`, `season_mean`.
- `season_projections` must have exactly one ruleset, and it must match `league_config.ruleset.name`. Mixed-ruleset input raises `ValueError` with a clear message. Wrong-ruleset input raises `ValueError` naming both the expected and observed ruleset.
- Multiple seasons in the input raises `ValueError` — VORP is computed for one season at a time (use season-scoped filtering upstream).
- Duplicate `gsis_id` in the input raises `ValueError` before any computation.

Returns a new DataFrame validated against `VorpTableSchema`. One row per `(gsis_id)` in the input. The rename `season_mean → season_mean_fpts` happens here (the only place in the codebase that bridges the projection layer's `season_mean` to the auction layer's `season_mean_fpts` — see §6 for the cross-spec naming note).

### 2.2 `VorpTableSchema`

Lives in `src/projections/schemas.py` per the "single source of truth" rule, appended after `ProjectionSeasonSchema` and before `AuctionValuesSchema`.

```python
class VorpTableSchema(pa.DataFrameModel):
    """Per-player VORP table. Consumer-facing output of the VORP generator.

    Direct input contract for AuctionValuesSchema's upstream and for the
    snake-draft cheat sheet. One row per player in the input season-projection
    set. `vorp` may be negative (sub-replacement players).
    """

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$", unique=True)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    season_mean_fpts: Series[float]
    vorp: Series[float]
    replacement_fpts: Series[float]              # surfaced for eyeball mitigation

    class Config:
        strict = "filter"
        coerce = True
```

`replacement_fpts` is added as a per-row column (the per-position replacement-fpts value, broadcast across all players at that position). It's redundant given `vorp = season_mean_fpts − replacement_fpts`, but it makes the table self-explanatory and supports the eyeball-mitigation summary the CLI prints (per §4). Auction validates against its own schema which ignores extra columns via `strict="filter"`, so this doesn't break the contract.

Note: `season` is NOT a column. VORP is scoped to one season per file; the season lives in the filename and in the CLI's `--season` arg, not in the data. If we ever need multi-season VORP storage, add it then.

### 2.3 Why pure-function-plus-CLI, not store partition

Same call as auction: output is a local user artifact, not a partitioned dataset under `store.write_partition`. No backtest-style temporal indexing, no historical retention requirement. One file per (season, ruleset, league_config). User keeps it next to their draft notes.

If the live snake-draft tool later needs structured VORP storage, add it then. YAGNI.

---

## 3. Algorithm

### 3.1 Replacement-level: the pool-boundary method (recommended, v1)

First, filter `season_projections` to only positions present in `league_config.roster_slots` (so positions like K/DST in the projection input but absent from the config are dropped immediately; see §3.6). Then for each position `pos` present in the filtered set:

1. Run the auction-values pool selection algorithm (`_select_pool` from `src/projections/draft/auction.py`) against `season_projections`. This produces a list of `gsis_id`s of length `total_pool_size = n_teams × roster_size`, accounting for position-specific slots → FLEX → SUPER_FLEX → BENCH.
2. `replacement_fpts(pos) = max(season_mean_fpts)` over players at position `pos` who are NOT in the pool.
3. If every player at position `pos` is in the pool (the pool exhausts the position's available depth — possible for shallow positions like TE in some configs), `replacement_fpts(pos) = min(season_mean_fpts)` over players at position `pos`. This makes the bottom player's VORP exactly 0 and is the conservative interpretation.

Then:

```
vorp_i = season_mean_fpts_i − replacement_fpts(position_i)
```

VORP is signed: top players have positive VORP, fringe players near the pool boundary have small positive VORP, and players outside the pool have ≤ 0 VORP (often 0 exactly if they're the replacement, slightly negative if worse).

**Why pool-boundary, not strict positional.** Strict positional replacement (`replacement(pos) = (n_teams × roster_slots[pos] + 1)`th best at `pos` by mean) is simpler and standard in most cheat sheets — but it ignores FLEX, SUPER_FLEX, and BENCH. In a 12-team league with 1QB / 2RB / 3WR / 1TE / 1FLEX / 7BENCH, strict positional puts WR replacement at WR37; reality is closer to WR60 because the FLEX slot and bench depth drain another two dozen WRs from the available pool. Pool-boundary captures this naturally because the pool selection algorithm already accounts for slot composition. It also has internal consistency: the pool of "who actually gets drafted" is the same pool the auction script uses to compute $ values, and the replacement-level is the first player off that pool — i.e., the waiver-wire equivalent.

The cost is one extra dependency on `_select_pool` (which is internal to `auction.py` today). The cleanup: lift `_select_pool` out of `auction.py` into a new module-private helper that both `auction.py` and `vorp.py` import. See §2 architecture and §7 acceptance for the refactor specifics.

**Implementation refactor.** `_select_pool` currently lives at module scope in `src/projections/draft/auction.py` (lines 81-147). Move it to `src/projections/draft/_pool.py` (new module-private file, leading underscore to mark non-public) and import it from both `auction.py` and `vorp.py`. Pure mechanical move, no signature change, no behavior change — but Phase 1 of the implementation plan, with the auction test suite as the verification gate. The auction spec's §3.1 description remains accurate after the move (the algorithm doesn't change, only the home file).

### 3.2 Alternative considered: strict positional (NOT shipping in v1)

```
n_starters(pos)    = n_teams × roster_slots.get(pos, 0)
replacement_fpts(pos) = season_mean_fpts of the (n_starters(pos) + 1)-th best player at pos
```

Strictly simpler — no FLEX/SUPER_FLEX/BENCH coupling, no shared `_pool` module dependency. Common in published cheat sheets.

Why we're not shipping it: ignoring FLEX systematically overstates replacement-level for FLEX-eligible positions, which systematically understates VORP for top RB/WR/TE relative to QB/K/DST. In a typical 12-team league this is ~15-25% VORP compression at the top of the RB/WR boards. The pool-boundary method costs one helper-module move and produces materially better numbers.

Documented here so the next reader knows it was considered and rejected with reason. If a future contributor wants this as an opt-in (`--method strict` or similar), they can add it in a small follow-up.

### 3.3 Alternative considered: bench-buffer positional (NOT shipping in v1)

```
n_drafted(pos)    = n_teams × (roster_slots[pos] + bench_buffer[pos])
replacement_fpts(pos) = season_mean_fpts of the (n_drafted(pos) + 1)-th best player at pos
```

A middle ground between strict positional and pool-boundary, where `bench_buffer[pos]` is a hand-picked per-position estimate of "how deep does the bench typically go for this position in this league" (e.g., {QB: 1, RB: 4, WR: 4, TE: 1}). FantasyCalc and similar published sheets use this.

Why we're not shipping it: the buffers are league-and-ruleset-specific magic numbers that drift over time and need re-tuning. Pool-boundary derives equivalent information from `LeagueConfig` directly, without the magic. Documented here for completeness.

### 3.4 Worked example

12-team standard PPR, roster slots `{QB: 1, RB: 2, WR: 3, TE: 1, FLEX: 1, K: 1, DST: 1, BENCH: 7}`, `roster_size = 17`, `total_pool_size = 204`. Assume `season_projections` is generously sized (e.g. 60 QBs, 100 RBs, 120 WRs, 40 TEs, 32 Ks, 32 DSTs — 384 rows total).

`_select_pool` runs in slot order and returns the 204 in-pool `gsis_id`s: position-specific slots first (12 QBs + 24 RBs + 36 WRs + 12 TEs + 12 Ks + 12 DSTs = 108 picks), then 12 FLEX (filled from the top remaining RB/WR/TE by `season_mean_fpts`), then 0 SUPER_FLEX, then 84 BENCH filled from the top remaining at any rostered position. Exact bench composition depends on the input projection ranks; deeper positions (RB, WR) consume more bench slots than thin ones (TE, K, DST).

For each position, `replacement_fpts(pos) = max(season_mean_fpts)` over players at that position who are NOT in the 204-player pool. So if the pool contains the top 14 QBs (12 starters + 2 bench), `replacement_fpts(QB)` is the 15th-ranked QB's `season_mean_fpts`. If the pool contains the top 50 RBs (24 RB-slot + ~10 FLEX + ~16 bench), `replacement_fpts(RB)` is the 51st-ranked RB's `season_mean_fpts`. And so on per position.

Then `vorp_i = season_mean_fpts_i − replacement_fpts(position_i)` for every input row, signed. The output table has `replacement_fpts` as a per-row column (broadcast within position) so a reader can verify the math by hand.

### 3.5 Tie-breaks and determinism

`_select_pool` ranks by `(season_mean_fpts desc, vorp desc, gsis_id asc)`. But VORP isn't known yet at pool-selection time inside the VORP module — that ordering depends on a quantity we're about to compute. **Resolution:** when `_select_pool` is called from `vorp.py`, pass a `vorp_table` whose `vorp` column is all zeros (or absent — see refactor below). Pool ranking falls back to `(season_mean_fpts desc, gsis_id asc)`, which is deterministic.

When the same `_select_pool` is later called from `auction.py` against the published VORP table, `vorp` is populated and acts as a tie-break. The pool composition is unchanged for any non-tied (`season_mean_fpts`) input; for tied inputs, the auction pool may differ from the VORP pool by one player at the tie boundary. This is a documented harmless inconsistency: tie-breaks only matter for synthetic test inputs, real projections almost never tie on `season_mean_fpts`.

**Refactor note.** `_select_pool`'s current signature is `(vorp_table: pd.DataFrame, league_config: LeagueConfig) -> list[str]`. Generalize to accept a DataFrame with at minimum `(gsis_id, position, season_mean_fpts)` columns and an optional `vorp` column used only for tie-breaking. The auction code path passes both columns; the VORP code path passes the first three. Update the function's docstring and signature accordingly during the refactor.

### 3.6 Edge cases

- **Empty input.** `season_projections.empty` → return an empty `VorpTableSchema`-validated frame.
- **One position only.** League with only QB (`{QB: 1, BENCH: 0}`) — `replacement_fpts(QB) = best non-pool QB` works fine; if every QB fits in the pool (`n_teams × 1 = 12`), use `min(season_mean_fpts)` over QBs.
- **All players at one position fit in the pool.** Per §3.1 step 3: replacement = min. The bottom QB has VORP 0; everyone else has VORP > 0. This is the right behavior — there's no "waiver wire" QB if the league rosters every projectable QB.
- **Position in projection input but not in `LeagueConfig.roster_slots`.** VORP drops those rows from the output entirely. (Alternative considered: keep them with NaN `vorp`. Rejected — the auction-values consumer's schema is non-nullable on `vorp`, and downstream tools shouldn't have to defend against undraftable rows. The CLI logs the dropped count by position for transparency.)
- **Position in `LeagueConfig.roster_slots` but not in projection input.** VORP raises `ValueError` from `_select_pool`'s `_take_top_n` helper — the message names the missing slot and the (zero) count of eligible players. Pool composition is undefined when a required position has no input rows, so silent truncation would produce a meaningless output. The CLI surfaces the raise to the user; the caller fixes `LeagueConfig` (drop the position) OR adds projections for that position. (Earlier draft of this spec said "silent + CLI-warns + auction errors downstream" — that design was incoherent because the pool can't be computed; reconsidered.)

---

## 4. CLI surface

`scripts/generate_vorp_table.py`:

```
python scripts/generate_vorp_table.py \
    --season 2026 \
    --league-config configs/league_espn_ppr_12team.json \
    --weekly-projections data/projections/weekly/ruleset=espn_ppr \
    --out reports/vorp_2026.parquet
```

**Flags:**

| Flag | Required | Description |
|---|---|---|
| `--season` | yes | Integer. Filters `--weekly-projections` to that season's `season=YYYY/week=WW/` partitions before aggregation. Errors clearly if no rows match. |
| `--league-config` | yes | Path to `LeagueConfig` JSON (same format as auction's `--league-config`). |
| `--weekly-projections` | yes | Path to the weekly-projections partition root (e.g., `data/projections/weekly/ruleset=espn_ppr`). Reads all `season=YYYY/week=WW/part.parquet` files matching `--season`. |
| `--out` | yes | Output destination. `.csv` and `.parquet` both supported (sniff by extension). |

**Script flow:**

1. Parse args.
2. Load `LeagueConfig` from JSON.
3. Read weekly-projections parquet files matching `--season`, concat into a single `ProjectionWeeklySchema` DataFrame.
4. Filter to `ruleset == league_config.ruleset.name`. Error if zero rows match.
5. Call `aggregate_to_season(weekly, ruleset=league_config.ruleset)` → `ProjectionSeasonSchema` DataFrame.
6. Call `generate_vorp_table(season_projections, league_config)`.
7. Log a per-position summary to stdout — the eyeball-mitigation surface (analogous to auction's per-position summary, see §6 below).
8. Write output. CSV writes are sorted by `vorp` descending for readability; parquet writes preserve the schema's column order without re-sorting.

**Per-position stdout summary (eyeball mitigation):**

```
VORP table written: 300 players, ruleset=ESPN_PPR

Position summary (replacement_fpts | in-scope row count | top-3 by VORP):
  QB  replacement= 242.10  rows= 14  top: 00-0033873(VORP+91.3), 00-0034796(VORP+78.0), 00-0034857(VORP+72.5)
  RB  replacement=  98.40  rows= 50  top: 00-0034681(VORP+181.2), 00-0036244(VORP+109.0), 00-0036414(VORP+95.4)
  WR  replacement= 130.90  rows= 62  top: 00-0036442(VORP+115.7), 00-0033921(VORP+88.2), 00-0036414(VORP+79.9)
  TE  replacement=  87.20  rows= 19  top: 00-0030506(VORP+62.1), 00-0032764(VORP+44.5), 00-0035245(VORP+31.2)
```

The user can sanity-check that replacement-level looks reasonable (~RB30-ish region projecting to ~100 fpts is plausible) and that the top of each position list makes sense. Cheap, no extra schema.

Note: positions that are absent from the projection input (today: K and DST) simply don't appear in the summary — `_log_per_position_summary` skips empty position-row sets. If `LeagueConfig.roster_slots` requires a position with zero input rows, the function raises from `_select_pool` before reaching this summary; the CLI surfaces the raise as a non-zero exit with a "cannot fill N {slot} slots" message. See §3.6 / §5.1 #17 and §5.4 #26.

---

## 5. Testing

All tests under `tests/test_draft/test_vorp.py` unless noted. Standard pandera-validated DataFrame contract plus algorithmic invariants. Naming and pattern mirror `tests/test_draft/test_auction.py`.

### 5.1 Algorithmic tests in `test_vorp.py`

1. **Schema round-trip.** Output validates against `VorpTableSchema` without filter losses; column order matches schema.
2. **Row count for in-scope positions preserved.** Output row count == count of input rows whose `position` is in `LeagueConfig.roster_slots`. Rows at out-of-scope positions are dropped (test #18); no duplicates among the kept rows.
3. **Rename invariant.** Output's `season_mean_fpts` column has values identical to input's `season_mean` column (row-aligned by `gsis_id`).
4. **VORP equation.** `vorp == season_mean_fpts − replacement_fpts` exactly (no rounding) for every row.
5. **Replacement-level pinned per position.** Synthetic input where the pool boundary at each position is known by construction; assert `replacement_fpts(QB)` equals the projection of the expected boundary QB, same for RB/WR/TE.
6. **Top-of-position players have non-negative VORP.** Every player ranked ≤ `n_teams × roster_slots[pos]` at their position by `season_mean_fpts` has `vorp ≥ 0`. (Strict positive doesn't hold in the corner case where the boundary player ties the replacement player — pin to `≥ 0`.)
7. **Replacement player has VORP ≈ 0.** The player at each position whose `season_mean_fpts == replacement_fpts(pos)` has `vorp == 0` (or near-zero if there are ties at the boundary).
8. **Sub-replacement players have negative VORP.** Players outside the pool with lower `season_mean_fpts` than their position's `replacement_fpts` get negative VORP.
9. **Pool-boundary equivalence with auction.** Run `_select_pool` against the input directly and against the VORP-augmented output (passing the new VORP table back through); pool composition matches for inputs with unique `season_mean_fpts` values per position (the documented tie-break consistency from §3.5).
10. **FLEX deepens RB/WR/TE replacement.** Compare two `LeagueConfig`s differing only in `FLEX: 0` vs `FLEX: 1`; assert that `replacement_fpts(RB)`, `replacement_fpts(WR)`, `replacement_fpts(TE)` are all ≤ in the FLEX=1 config (deeper replacement = lower value).
11. **SUPER_FLEX deepens QB replacement.** Same construction, `SUPER_FLEX: 0` vs `SUPER_FLEX: 1`; `replacement_fpts(QB)` is ≤ in the SUPER_FLEX=1 config.
12. **Ruleset mismatch raises.** Input with `ruleset == "espn_half"` and `LeagueConfig.ruleset == Ruleset.espn_ppr()` raises `ValueError` naming both.
13. **Mixed ruleset raises.** Input with rows from two rulesets raises `ValueError`.
14. **Mixed season raises.** Input with rows from two seasons raises `ValueError`.
15. **Duplicate `gsis_id` raises.** Same as auction's check; raises before any computation.
16. **Empty input returns empty.** Empty input → empty output, schema-validated.
17. **Position in config but not in input.** `LeagueConfig` requires K but `season_projections` has no K rows → `generate_vorp_table` raises `ValueError` matching `r"cannot fill \d+ K slots"` (raised from `_select_pool`'s `_take_top_n` helper). Pool composition is undefined in this case; the function makes the failure explicit rather than silently truncating.
18. **Position in input but not in config.** Input has K rows, `LeagueConfig` has no K slot; function drops K rows from the output. Output row count == count of input rows whose `position` is in `LeagueConfig.roster_slots`.
19. **All players at one position fit in pool.** Synthetic input with 12 QBs in a 12-team 1QB league → `replacement_fpts(QB) = min(season_mean_fpts of QBs)`; the bottom QB has VORP 0.
20. **Underfilled pool raises.** `season_projections` smaller than `total_pool_size` → `_select_pool` raises `ValueError` with the existing message naming the under-filled slot. Pin this as the documented contract — VORP does not silently emit a degraded output on insufficient input.
21. **Determinism.** Calling `generate_vorp_table` twice with the same input produces byte-identical output (DataFrames compared via `assert_frame_equal`).

### 5.2 `_select_pool` refactor regression test

22. **Auction pool equivalence post-refactor.** After moving `_select_pool` to `_pool.py`, the entire existing auction test suite must pass with zero modifications. This is the verification gate for the refactor; pinned as a regression contract.
23. **`_select_pool` accepts vorp-less input.** Direct call to `_select_pool` with a DataFrame missing the `vorp` column produces the same pool as a call with `vorp` all-zero (tie-break degrades to `(season_mean_fpts, gsis_id)`).

### 5.3 Schema round-trip test

Appended to `tests/test_schemas/test_dataframe_schemas.py`:

24. **`VorpTableSchema` round-trip.** Build a minimal valid frame; validate; re-validate; assert idempotent. Matches the pattern of the existing `test_auction_values_schema_round_trip`.

### 5.4 CLI integration test in `test_generate_vorp_table_cli.py`

25. **End-to-end with a known fixture.** Synthetic weekly-projections parquet partition + known `LeagueConfig` JSON → script produces an output parquet with expected per-position VORP values. Asserts the pin on a small canonical output snippet and the per-position replacement-fpts values. Uses the same `env={..., "PYTHONPATH": str(repo_root / "src")}` subprocess workaround as the auction CLI test (see PM doc deviation #5 from auction PR).
26. **CLI errors when config requires a position missing from input.** Run with a `LeagueConfig` requiring K but with no K rows in the input; assert non-zero exit code and that combined stderr+stdout contains "cannot fill" and the slot label "K" (the raise text from `_select_pool`'s `_take_top_n`). Per §3.6 / §5.1 #17, the failure is explicit, not silent — the function raises before the per-position summary can print, so there is no "warning" stdout to assert against.
27. **Ruleset mismatch errors cleanly.** Run with mismatched LeagueConfig ruleset vs input ruleset; assert non-zero exit and the error message names both rulesets.

### 5.5 What's deliberately not tested

- **VORP magnitudes vs published ADP-derived rankings.** Calibration is downstream; pool-boundary VORP is a model-internal quantity, not a market signal.
- **Stability across `predict_*.py` revisions.** The projection input is the ProjectionSeasonSchema-validated contract; if upstream projections shift, VORP shifts with them. That's correct behavior, not a bug to test.
- **K/DST math.** No projections, no test fixtures; deferred with TODO #10.

---

## 6. Open items, risks, and explicit out-of-scope

### Open items

**Cross-spec naming inconsistency (`season_mean` vs `season_mean_fpts`).** `ProjectionSeasonSchema.season_mean` (aggregation layer, shipped on `main`) and `AuctionValuesSchema.season_mean_fpts` (auction layer, shipped on `feat/auction-values`) are the same quantity under different names. VORP bridges the rename. Future cleanup: rename `AuctionValuesSchema.season_mean_fpts → season_mean` and remove the rename from VORP, OR rename `ProjectionSeasonSchema.season_mean → season_mean_fpts` (forces upstream changes — bigger blast radius). Not load-bearing for this spec; flagged for a future consistency pass. Pick the smaller-blast-radius option when it comes up.

**`predict_season.py` not yet built.** This spec's CLI consumes weekly-projection partitions, which today are only produced by `scripts/predict_2024.py` (hardcoded to 2024). For 2026 draft season, either generalize `predict_2024.py` first (tracked in `draft_ready_checklist.md` §1b) or manually hand-edit the script. The VORP CLI is agnostic to which one ran — it just reads the partition path it's given.

**Pre-season roster source.** The full draft-day flow requires 2026 rosters, which aren't ingested yet (`draft_ready_checklist.md` §1a). VORP itself doesn't depend on rosters — it operates on whatever players are in the projection input. But the projection input is currently built from `depth_charts`, which don't exist pre-season. So VORP runs end-to-end today only against past seasons (2024). Pre-season 2026 VORP is gated on `predict_season.py` + a 2026 roster source.

**Rookie projections.** Same gap. Tracked in `draft_ready_checklist.md` §1a. Rookies absent from projection input → absent from VORP output → not draftable by auction script. The CLI can warn about specific draft-relevant rookies if a list is provided; out of scope for v1.

### Risks

**Pool-boundary replacement-level is internally consistent but is one specific definition.** Many published cheat sheets use strict positional or bench-buffer methods. If the user compares this spec's VORP numbers against a public ESPN/Yahoo cheat sheet, the absolute magnitudes will differ even though the rank ordering will largely match. **Mitigation:** the CLI emits `replacement_fpts` per position to stdout (§4), and the parquet output has `replacement_fpts` as a per-row column. A reader can sanity-check both the boundary player choice and the resulting numbers. If the user wants strict-positional for comparison purposes, a `--method strict` flag is a small follow-up.

**Refactoring `_select_pool` out of `auction.py` could break auction tests.** Mechanical move + import change, but the auction test suite is the verification gate. Implementation plan must run the full auction test suite as Phase 1's verification step before proceeding to VORP code. (See `superpowers:verification-before-completion`.)

**K/DST silently producing empty output.** If a user runs VORP against a config requiring K/DST and ignores the warning, the auction script will error downstream, but the user might assume "auction is broken" rather than "VORP missing K/DST." **Mitigation:** the warning text in §4 explicitly names the downstream consequence and points at TODO #10. Don't bury the warning.

**Tie-break inconsistency between VORP and auction pool selection.** Documented in §3.5. Only affects ties in `season_mean_fpts`, which are rare in real data. Tests pin the documented behavior so future readers don't trip on it.

### Explicit out-of-scope (so they don't get smuggled in)

- Strategy-aware VORP (stars-and-scrubs adjusters). → consumer-layer concern, lives in live-draft tooling.
- Tier breaks. → separate concern, computed on top of VORP.
- Confidence bands per ranking from p10/p90. → separate concern.
- Upside-sensitive VORP (using p90 instead of mean). → follow-up spec if needed.
- Market calibration (ADP-anchored VORP). → separate spec, blocked on ADP ingest.
- K/DST projection generation. → TODO #10, separate work.
- Rookie projection generation. → draft-readiness §1a, separate work.
- `predict_season.py` generalization. → draft-readiness §1b, separate work.
- Multi-ruleset or multi-season VORP in one run. → run the script multiple times.
- Persistence to the store layer. → output stays as a local file.

---

## 7. Acceptance

This spec is complete when:

- `_select_pool` has been moved from `src/projections/draft/auction.py` to `src/projections/draft/_pool.py` and its signature accepts inputs without a `vorp` column. Auction test suite passes unchanged.
- `generate_vorp_table` is implemented per §2-3 in `src/projections/draft/vorp.py` and re-exported from `src/projections/draft/__init__.py`.
- `VorpTableSchema` is in `src/projections/schemas.py`.
- `scripts/generate_vorp_table.py` runs end-to-end against a synthetic weekly-projections partition fixture and a sample `LeagueConfig` JSON.
- All §5 tests pass (≥ 27 new tests).
- `mypy src tests` and `ruff check src tests` clean.
- `ruff format --check src tests` clean.
- `pytest -v -k "ingest or store or schemas"` passes (schema-seam guard per CLAUDE.md §4).

No adoption gate, no §1.3.5-style contingency matrix — this is feature work, not a model-change probe. The shipping decision is binary: does the feature work, do the per-position summary numbers look sane, and do the tests pass.

---

## 8. Follow-ups (sequenced)

In rough priority order, each its own spec:

1. **Snake-draft cheat sheet** (`draft_ready_checklist.md` §2b.1). Sister to auction-values: same VORP input, different output surface. Per-position ordered list with VORP, ADP delta (when ADP is available), tier, confidence band. Can ship in parallel with this spec since they're independent surfaces over the same VORP output.
2. **`predict_season.py` generalization** (`draft_ready_checklist.md` §1b). Required for any post-2024 VORP run. Small refactor; tracked separately.
3. **K/DST projection** (TODO #10). Unblocks K/DST VORP. Two options sketched in TODO #10: v0 from `implied_team_total` only (fast), or ingest the missing data (correct). Decide there.
4. **Upside-sensitive VORP variant.** Add a `--method` or `--variant` flag accepting `mean` (default) or `p90`. Useful for DFS GPP and tournament-format snake leagues. Trigger: snake-draft cheat sheet design surfaces a real need.
5. **Strict-positional VORP method.** As a `--method strict` opt-in, for users who want apples-to-apples comparison against published ESPN/Yahoo cheat sheets. Trigger: user feedback after first draft season.
6. **Cross-spec column rename.** `season_mean ↔ season_mean_fpts` consistency pass. Trigger: next time either `ProjectionSeasonSchema` or `AuctionValuesSchema` needs a non-trivial edit.

---

## 9. Implementation phases (for the plan)

The implementation plan should phase as:

**Phase 1 — Refactor.** Move `_select_pool` from `auction.py` to `_pool.py`. Update `auction.py` import. Generalize signature to accept inputs without a `vorp` column. Run the full auction test suite as the verification gate. No new features — pure mechanical move. Commit as its own commit.

**Phase 2 — Schema.** Append `VorpTableSchema` to `schemas.py`. Add a round-trip test in `test_dataframe_schemas.py`. Run the schema-seam guard.

**Phase 3 — Function.** Implement `generate_vorp_table` in `vorp.py`. Implement §5.1 algorithmic tests (test #1-21).

**Phase 4 — CLI.** Implement `scripts/generate_vorp_table.py`. Implement §5.4 CLI tests (test #25-27).

**Phase 5 — Refactor regression + integration.** Run §5.2 regression tests (test #22-23). Run `pytest -v` over the entire test suite. Run `mypy src tests`, `ruff check src tests`, `ruff format --check src tests`. Run `pytest -v -k "ingest or store or schemas"` per CLAUDE.md §4. Fix anything that breaks.

Each phase touches ≤ 5 files (CLAUDE.md §2). Phase 1 is the most architecturally risky (touches shipped auction code); the auction test suite is the gate.
