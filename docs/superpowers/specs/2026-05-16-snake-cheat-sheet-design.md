# Snake-Draft Cheat Sheet — Design

**Status:** draft (brainstorming, 2026-05-16). Ready for user review.
**Date:** 2026-05-16
**Author:** alden + claude
**Sub-project of:** FantasyFootball / Draft Hub
**Branch:** `feat/snake-cheat-sheet` cut from `origin/main` at `c8da85b` (post-merge of PR #40 auction-values and PR #43 VORP).

**Depends on:**
- `VorpTableSchema` and the parquet output of `scripts/generate_vorp_table.py` (`src/projections/draft/vorp.py`, shipped on PR #43).
- `LeagueConfig` from `src/projections/draft/league_config.py` (shipped on PR #40).
- `_select_pool` from `src/projections/draft/_pool.py` (shipped on PR #43 as part of the auction/VORP refactor).
- `id_map.parquet` at `data/raw/id_map.parquet` (existing — populated by `src/projections/ingest/id_map.py`). `IdMapSchema.full_name` is the canonical display-name source in this codebase. (Note: `depth_charts` does **not** carry player names — its schema is `gsis_id / season / week / team / position / depth_team / depth_rank`. id_map is the right source.)

**Consumed by:** end user on draft day (the cheat sheet is itself the user-facing surface). No downstream code consumer in this PR. A future live-snake recommender (`draft_ready_checklist.md` §2b.2) is the natural next consumer and will read this output's parquet form for fast per-pick lookup, but spec'd separately.

**Related specs / docs:**
- `docs/superpowers/specs/2026-05-16-vorp-design.md` — upstream input contract producer.
- `docs/superpowers/specs/2026-05-16-auction-values-design.md` — sister VORP consumer (auction $ values). Same `_select_pool` helper; mirror CLI and test patterns.
- `draft_ready_checklist.md` §2b "Snake draft" — names the cheat sheet (§2b.1) as the v1 snake surface. ADP (§2b.3) and live recommender (§2b.2) deferred.
- `TODO.md` #10 — K/DST positions still unbuilt; informs §6 position-scope decision.

---

## 1. Overview

After VORP exists, the Draft Hub needs a human-facing surface that takes a VORP parquet and turns it into a printable / spreadsheet-able per-position ranking with tier breaks — the artifact a drafter actually looks at during a snake draft. This spec defines that transform, its schema, and its CLI.

Scope is deliberately narrow. The `draft_ready_checklist.md` §2b.1 wishlist names four columns (VORP, ADP delta, tier, confidence band). v1 ships **only VORP + tier breaks** plus the necessary supporting columns (display name, positional rank, in-pool flag). ADP and confidence band are deferred — each blocks on infrastructure that doesn't exist (no ADP ingest; no p10/p90 plumbed through to VORP).

### 1.1 Goals (in scope)

- **New module `src/projections/draft/snake_cheat_sheet.py`** with one public function `generate_snake_cheat_sheet(vorp_table, league_config, display_names, tiers_per_position) -> pd.DataFrame`. Pure transform; no I/O.
- **New schema `SnakeCheatSheetSchema` in `src/projections/schemas.py`** validating the output, appended after `VorpTableSchema`.
- **New CLI `scripts/generate_snake_cheat_sheet.py`** that reads a VORP parquet + `id_map.parquet` + league config, runs the transform, writes CSV or parquet (sniffed by `--out` extension). Mirrors the flag conventions established by the auction and VORP CLIs.
- **Tests in `tests/test_draft/test_snake_cheat_sheet.py`** plus a CLI integration test in `tests/test_scripts/test_generate_snake_cheat_sheet_cli.py` plus a schema round-trip test in `tests/test_schemas/test_dataframe_schemas.py`. Coverage in §5.
- **Tier algorithm: gap-based, fixed N tiers per position.** N defaults to 8, configurable via `--tiers-per-position`. See §3.2 for the algorithm and ties / fallbacks.
- **Pool treatment: show all players, tier the in-pool subset, mark the rest.** `is_in_pool` boolean column on every row; `tier` is nullable integer (1..N for in-pool, NA otherwise). `positional_rank` is computed across all players (in-pool and out) so the cheat sheet can serve double-duty for first-week-of-season waiver claims.
- **Player names: auto-join from `id_map.parquet`.** Read `data/raw/id_map.parquet`; project `(gsis_id, full_name)`; build a `gsis_id → display_name` map; fall back to `"—"` for any player without an id_map row. If the file itself is missing, log a warning and use `"—"` for everyone (the cheat sheet still emits, with all names as `"—"` — same defensive UX the auction CLI uses for its missing-VORP case).

### 1.2 Non-goals (deferred)

- **No ADP / ADP delta.** Blocks on `draft_ready_checklist.md` §2b.3 — no ADP ingest exists today. The manual `--adp-csv` escape hatch was considered and rejected for v1 (adds an id-resolver surface that grows quickly). Future follow-up spec.
- **No confidence band (floor / ceiling rank from p10 / p90).** `VorpTableSchema` carries `season_mean_fpts` only; p10/p90 live one layer up in `ProjectionSeasonSchema`. Plumbing them through is a meaningful schema decision (either expand `VorpTableSchema` or have this CLI re-aggregate). Defer to its own follow-up.
- **No live snake-draft recommender.** `draft_ready_checklist.md` §2b.2 — separate spec. The recommender will consume this output but the algorithmic shape (greedy vs. lookahead, opponent-pick simulation) is a different design problem.
- **No markdown output.** CSV is enough for spreadsheet-on-draft-day use; downstream tooling can read parquet. Markdown formatting belongs to a UI-layer if and when one lands.
- **No cross-position overall board.** The cheat sheet groups by position; an "ADP-style overall ranking" needs an external pick-cost signal we don't have without ADP.
- **No K / DST.** Same gap as VORP (TODO #10). If `LeagueConfig.roster_slots` requires a missing position, `_select_pool` raises before reaching tier code; this spec inherits VORP's behavior verbatim.
- **No rookie handling.** Same as VORP — rookies absent from projections → absent from VORP → absent from cheat sheet.
- **No persistence to the store layer.** Output is a user artifact, not a partitioned dataset, same convention as auction and VORP.
- **No multi-season or multi-ruleset output in one run.** One run produces one season × one ruleset × one league config × one tiers_per_position.

---

## 2. Architecture

```
src/projections/draft/
├── __init__.py                          (edited — re-export generate_snake_cheat_sheet)
├── _pool.py                             (unchanged — _select_pool reused)
├── league_config.py                     (unchanged)
├── auction.py                           (unchanged)
├── vorp.py                              (unchanged)
└── snake_cheat_sheet.py                 (NEW — generate_snake_cheat_sheet + tier algorithm)

src/projections/schemas.py               (edited — append SnakeCheatSheetSchema after VorpTableSchema)

scripts/
├── generate_auction_values.py           (unchanged)
├── generate_vorp_table.py               (unchanged)
└── generate_snake_cheat_sheet.py        (NEW — CLI)

tests/test_draft/
├── test_auction.py                      (unchanged)
├── test_league_config.py                (unchanged)
├── test_pool.py                         (unchanged)
├── test_vorp.py                         (unchanged)
└── test_snake_cheat_sheet.py            (NEW)

tests/test_scripts/
├── test_generate_auction_values_cli.py  (unchanged)
├── test_generate_vorp_table_cli.py      (unchanged)
└── test_generate_snake_cheat_sheet_cli.py (NEW)

tests/test_schemas/test_dataframe_schemas.py (edited — append SnakeCheatSheetSchema round-trip)
```

### 2.1 Public function

```python
def generate_snake_cheat_sheet(
    vorp_table: pd.DataFrame,                # validated against VorpTableSchema
    league_config: LeagueConfig,
    display_names: pd.DataFrame | None = None,
    tiers_per_position: int = 8,
) -> pd.DataFrame:                           # validated against SnakeCheatSheetSchema
    ...
```

Pure function. No I/O, no side effects, no caching. Algorithm in §3.

**Input contract:**

- `vorp_table` validates cleanly against `VorpTableSchema` (re-validated defensively on entry). Columns of interest: `gsis_id`, `position`, `season_mean_fpts`, `vorp`, `replacement_fpts`.
- `league_config` must satisfy the same VORP / auction precondition: every position in `roster_slots` must appear in `vorp_table` (otherwise `_select_pool` raises with the existing "cannot fill N {slot} slots" message).
- `display_names`, if provided, is a two-column DataFrame `(gsis_id, display_name)` with unique `gsis_id`. Bare strings, no schema validation — internal usage. If `None`, all `display_name` values in the output become `"—"`.
- `tiers_per_position` must be a positive int. Validated; non-positive raises `ValueError`.

**Output:** a new DataFrame validated against `SnakeCheatSheetSchema`. One row per input `gsis_id`. The output's row ordering is `(position canonical order, positional_rank ascending)` for CSV-readability — i.e. all QBs first (QB1, QB2, …), then all RBs (RB1, RB2, …), etc. Position canonical order: `QB, RB, WR, TE, K, DST` (driven by the `Position` enum, only including positions present in the input).

### 2.2 `SnakeCheatSheetSchema`

Lives in `src/projections/schemas.py`, appended after `VorpTableSchema` and before `AuctionValuesSchema` to keep draft-related schemas grouped.

```python
class SnakeCheatSheetSchema(pa.DataFrameModel):
    """Per-player snake-draft cheat sheet. End-user surface for draft day.

    One row per player in the input VORP table. In-pool players get a numeric
    tier (1..N); out-of-pool players get tier = NA. `display_name` is
    best-effort from id_map.parquet; falls back to '—' for players without
    an id_map row.
    """

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$", unique=True)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    display_name: Series[str]
    positional_rank: Series[pd.Int64Dtype] = pa.Field(ge=1)
    season_mean_fpts: Series[float]
    vorp: Series[float]
    replacement_fpts: Series[float]
    is_in_pool: Series[bool]
    tier: Series[pd.Int64Dtype] = pa.Field(ge=1, nullable=True)

    class Config:
        strict = "filter"
        coerce = True
```

`positional_rank` is 1-indexed within position, sorted by `vorp` descending. Computed across both in-pool and out-of-pool rows (so the RB30 line that's out of pool still has a clear rank label).

`tier` is nullable `Int64` — pandas dtype `pd.Int64Dtype()`. Out-of-pool rows have `pd.NA`. In-pool rows have an integer in `[1, min(N, n_in_pool_at_position)]`.

`replacement_fpts` is carried through from `VorpTableSchema` unchanged (broadcast within position). Keeping it in the output makes the cheat sheet self-explanatory and supports the stdout summary (§4).

### 2.3 Why pure-function-plus-CLI, not store partition

Same answer as auction and VORP. Output is a per-draft local artifact, not a partitioned dataset with retention semantics. One file per `(season, ruleset, league_config, tiers_per_position)`. If a future live snake recommender wants structured storage, add it then.

---

## 3. Algorithm

### 3.1 Stages

`generate_snake_cheat_sheet` runs three pure stages:

1. **In-pool flag.** Call `_select_pool(vorp_table, league_config)` (the same helper auction and VORP use, lifted into `src/projections/draft/_pool.py` on PR #43). Returns a list of in-pool `gsis_id`s. Tag the input rows: `is_in_pool = gsis_id ∈ pool`.
2. **Positional rank.** Within each position, sort by `vorp` descending and assign `positional_rank = 1, 2, 3, …`. Tie-break for equal VORP: `gsis_id` ascending (matches `_select_pool`'s tie-break convention; deterministic).
3. **Tier breaks.** Within each position, restricted to `is_in_pool=True` rows, run the gap-based tier algorithm (§3.2). Out-of-pool rows get `tier = pd.NA`.

After stage 3, attach `display_name` via the optional `display_names` map (left-join on `gsis_id`; missing → `"—"`), reorder columns to schema order, sort by `(position canonical, positional_rank)`, and validate.

### 3.2 Tier-break algorithm: gap-based, fixed N

For each position `pos` with `n_in_pool = len(in_pool_rows_at_pos)`:

1. **n_in_pool == 0:** skip (no tiers at that position — only reachable if a position is in the projection input but not in `LeagueConfig.roster_slots`, in which case the VORP table doesn't contain those rows in the first place; defensive).
2. **n_in_pool ≤ N:** each in-pool player gets their own tier. `tier = 1..n_in_pool` by positional rank ascending. This is the corner case for thin positions (K, DST with 1 starter, or TE in shallow configs).
3. **n_in_pool > N:** compute `gaps[i] = vorp[i] − vorp[i+1]` for `i ∈ [0, n_in_pool − 2]` (where `i=0` is the top of the position by VORP). Take the `N-1` largest gaps; sort those gap-indices ascending. The gap-indices partition the in-pool rows into `N` contiguous buckets: rows `[0, g_1]` → tier 1, `(g_1, g_2]` → tier 2, …, `(g_{N-1}, n_in_pool−1]` → tier N.

**Tie-break for ties in `gaps`:** if multiple gaps have the same value and they're competing for the "N-1th largest gap" slot, prefer the higher-rank cut (lower gap-index). This is deterministic and matches the snake-draft intuition that the cliff happens "as early as possible" when the data is ambiguous.

**Algorithmically:**

```python
def _assign_tiers(vorp_desc: NDArray[np.float64], n_tiers: int) -> NDArray[np.int64]:
    n = len(vorp_desc)
    if n == 0:
        return np.array([], dtype=np.int64)
    if n <= n_tiers:
        return np.arange(1, n + 1, dtype=np.int64)

    gaps = vorp_desc[:-1] - vorp_desc[1:]                # shape (n-1,)
    # argsort ascending by (-gap, gap_index) so larger gaps come first
    # and ties broken by earlier gap-position
    order = np.lexsort((np.arange(n - 1), -gaps))
    cut_indices = np.sort(order[: n_tiers - 1])           # shape (n_tiers - 1,)

    tier = np.empty(n, dtype=np.int64)
    start = 0
    for t, cut in enumerate(cut_indices, start=1):
        tier[start : cut + 1] = t
        start = cut + 1
    tier[start:] = n_tiers
    return tier
```

`cut_indices[k]` is the index of the LAST row in tier `k+1`; the row at `cut_indices[k] + 1` starts tier `k+2`.

### 3.3 Worked example

12-team standard PPR, `tiers_per_position = 8`. WR position has 62 in-pool players. Computed gaps yield (illustratively) the seven largest gaps at gap-indices `[4, 9, 18, 27, 35, 44, 53]`. Per the §3.2 pseudocode, `cut_indices[k]` is the LAST index in tier `k+1` (0-indexed), so the player at gap-index 4 is the last player in tier 1, the player at gap-index 9 is the last player in tier 2, etc.

Tier composition for WRs (1-indexed positional ranks):
- Tier 1: WR1–WR5  (5 players — the "elite" cliff at gap-index 4)
- Tier 2: WR6–WR10 (5 players)
- Tier 3: WR11–WR19 (9 players)
- Tier 4: WR20–WR28 (9 players)
- Tier 5: WR29–WR36 (8 players)
- Tier 6: WR37–WR45 (9 players)
- Tier 7: WR46–WR54 (9 players)
- Tier 8: WR55–WR62 (8 players)

Total: 62 players. WRs ranked WR63 and below have `tier = NA, is_in_pool = False`.

Edge case at TE: pool has 19 TEs; `tiers_per_position = 8` so all 19 fit into 8 tiers via the gap-based partition. If a config had `roster_slots = {TE: 1, BENCH: 0}` in a 6-team league (i.e. 6 TEs in-pool), `n_in_pool = 6 < N = 8`, so each TE is its own tier (1–6).

### 3.4 Determinism

`_select_pool` is deterministic (the auction/VORP test suite pins this). `positional_rank` is deterministic given the `vorp / gsis_id` tie-break. The tier algorithm uses `np.lexsort` which is stable and deterministic. Final sort `(position, positional_rank)` is deterministic. The cheat sheet is byte-reproducible for byte-identical inputs.

### 3.5 Tier-instability across runs

Because tier breaks are picked from the top `N-1` gaps, a small VORP shift in any player can move which gap is "Nth largest." If two gaps are nearly tied at the N-1 / Nth-largest boundary, a 0.1-fpt change in any player at either side can move the cliff one row in either direction. This is documented expected behavior, not a bug. Mitigation: the stdout summary surfaces the size of tier 1 for each position (§4), so a user can sanity-check that the tier-1 cliff looks stable across runs.

### 3.6 Edge cases

- **Empty VORP input + non-empty `LeagueConfig.roster_slots`.** `_select_pool` raises `ValueError` ("cannot fill N {slot} slots") because no players are available to fill any required slot. The function fails loudly rather than emitting a zero-row cheat sheet for a non-empty league config. Tests pin this contract (§5.1 #18). If a future caller genuinely wants empty-in / empty-out, add an explicit `empty_ok=True` branch in a follow-up.
- **`tiers_per_position = 1`.** Every in-pool player at every position is tier 1. Algorithm degrades cleanly (no gap to compute; tier array is all 1s).
- **`tiers_per_position` larger than any position's `n_in_pool`.** Falls into the §3.2 step 2 branch for every position — each player is their own tier per position. Stays valid.
- **All VORPs equal at a position.** All gaps are 0; the lexsort tie-break picks the lowest-index gaps, so tier 1 = rank 1, tier 2 = rank 2, …, tier N = ranks N..end. Reasonable behavior (no real cliff exists, so the algorithm produces an even split).
- **Multiple id_map rows per gsis_id.** `IdMapSchema` declares `gsis_id` as unique, so duplicates would already be a schema violation. The CLI re-validates via `IdMapSchema.validate(id_map_df)` before building the name map; if validation fails (uniqueness or other), surface the validation error rather than silently picking one row.
- **Out-of-config position rows.** VORP already drops those rows on the upstream side (per VORP spec §3.6). They never appear in the cheat sheet input.

---

## 4. CLI surface

`scripts/generate_snake_cheat_sheet.py`:

```
python scripts/generate_snake_cheat_sheet.py \
    --season 2026 \
    --league-config configs/league_espn_ppr_12team.json \
    --vorp-input reports/vorp_2026.parquet \
    --id-map data/raw/id_map.parquet \
    --tiers-per-position 8 \
    --out reports/snake_cheat_sheet_2026.csv
```

**Flags:**

| Flag | Required | Default | Description |
|---|---|---|---|
| `--season` | yes | — | Integer. Appears in the stdout banner; not used to filter id_map (id_map is roster-wide, not per-season). |
| `--league-config` | yes | — | Path to `LeagueConfig` JSON (same format as auction and VORP CLIs). |
| `--vorp-input` | yes | — | Path to a `VorpTableSchema`-validated parquet (output of `scripts/generate_vorp_table.py`). |
| `--id-map` | no | `data/raw/id_map.parquet` | Path to the `IdMapSchema`-validated parquet (output of `build_id_map`). Provides `gsis_id → full_name`. |
| `--tiers-per-position` | no | `8` | Default N for tier breaks. Positive integer. |
| `--out` | yes | — | Output destination. `.csv` and `.parquet` both supported (sniffed by extension). |

**Script flow:**

1. Parse args; validate `--tiers-per-position > 0`.
2. Load `LeagueConfig` from JSON.
3. Read VORP parquet; re-validate against `VorpTableSchema`.
4. Read `--id-map`:
   - If the file is missing, log a warning ("id_map parquet not found at {path}; display names will be '—'") and pass `display_names = None`.
   - Otherwise, read the parquet; re-validate against `IdMapSchema`; project to a `(gsis_id, full_name)` frame renamed to `(gsis_id, display_name)`. Log the number of unique players mapped.
5. Call `generate_snake_cheat_sheet(vorp_table, league_config, display_names, tiers_per_position)`.
6. Emit the per-position stdout summary (eyeball mitigation, §4 example below).
7. Write output. `.csv` writes preserve the `(position canonical, positional_rank)` sort order with a header row. `.parquet` writes preserve schema column order without re-sorting.

**Per-position stdout summary (eyeball mitigation):**

```
Snake cheat sheet written: 300 players, ruleset=ESPN_PPR, tiers_per_position=8

Position summary (n_in_pool | tier-1 size | top-3):
  QB  in_pool= 14  tier1= 2  top: Patrick Mahomes (QB1, T1, VORP+91.3), Josh Allen (QB2, T1, VORP+78.0), Jalen Hurts (QB3, T2, VORP+72.5)
  RB  in_pool= 50  tier1= 3  top: Christian McCaffrey (RB1, T1, VORP+181.2), Bijan Robinson (RB2, T1, VORP+109.0), Saquon Barkley (RB3, T1, VORP+95.4)
  WR  in_pool= 62  tier1= 4  top: Justin Jefferson (WR1, T1, VORP+115.7), CeeDee Lamb (WR2, T1, VORP+88.2), Tyreek Hill (WR3, T1, VORP+79.9)
  TE  in_pool= 19  tier1= 1  top: Travis Kelce (TE1, T1, VORP+62.1), Mark Andrews (TE2, T2, VORP+44.5), Sam LaPorta (TE3, T2, VORP+31.2)
```

`tier1` size is the column that surfaces tier-cliff stability — if it shifts week-to-week between runs, the user knows the binding cliff is sensitive. Cheap, no extra schema.

Positions absent from the projection input (today: K and DST) don't appear in the summary — same behavior as VORP. If `LeagueConfig.roster_slots` requires a position with zero input rows, the function raises from `_select_pool`'s `_take_top_n` helper before reaching this summary; the CLI surfaces the raise as a non-zero exit with the existing "cannot fill N {slot} slots" message. See VORP spec §3.6 / §5.1 #17 for the upstream contract.

---

## 5. Testing

All tests in `tests/test_draft/test_snake_cheat_sheet.py` unless noted. Standard pandera-validated DataFrame contract plus algorithmic invariants. Pattern mirrors `tests/test_draft/test_vorp.py` and `test_auction.py`.

### 5.1 Algorithmic tests in `test_snake_cheat_sheet.py`

1. **Schema round-trip.** Output validates against `SnakeCheatSheetSchema` without filter losses; column order matches schema.
2. **Row count preserved.** Output row count == input row count (all positions; in-pool and out alike).
3. **`positional_rank` strictly monotonic.** Within each position, `positional_rank` is `[1, 2, 3, …]` ordered by `vorp` descending.
4. **`positional_rank` tie-break.** Equal-VORP rows are tie-broken by `gsis_id` ascending (matches `_select_pool` tie-break).
5. **`is_in_pool` matches `_select_pool`.** Set of `gsis_id` with `is_in_pool=True` equals the set returned by `_select_pool(vorp_table, league_config)`.
6. **Tier dtype is nullable Int64.** Out-of-pool rows have `pd.NA`, in-pool rows have integers in `[1, N]`.
7. **Tier monotonic with VORP.** Within each position, mean `vorp` of tier `t` ≥ mean `vorp` of tier `t+1`. (Stricter version: tier `t`'s minimum `vorp` ≥ tier `t+1`'s maximum `vorp` — gap-based partitioning is contiguous.)
8. **Gap-based correctness.** Synthetic input with hand-built VORP gaps `[100, 99, 98, 50, 49, 48, 10, 9, 8]` at a single position, `tiers_per_position = 3` → tiers `[1, 1, 1, 2, 2, 2, 3, 3, 3]`. Pin the exact partition.
9. **`n_in_pool < N` fallback.** Synthetic input with 5 in-pool players at TE, `tiers_per_position = 8` → tiers `[1, 2, 3, 4, 5]`. Each player is own tier.
10. **`n_in_pool == N` exact.** With exactly N in-pool players, tiers are `[1..N]` (one player per tier).
11. **Position with no in-pool rows.** If a position is in the input but `_select_pool` puts none of its players in pool (corner case: large pool exhausted by other positions), no tier rows are emitted for that position (but `positional_rank` rows still exist; `is_in_pool=False`, `tier=NA`).
12. **Display name auto-join — happy path.** Pass a `display_names` map covering all input gsis_ids → output `display_name` matches the map for every row.
13. **Display name auto-join — missing rows fall back to `"—"`.** Pass a partial map covering half the players → covered rows get the mapped name; uncovered rows get `"—"`.
14. **Display name auto-join — `display_names=None`.** All rows get `display_name = "—"`.
15. **Sort order.** Output sorted by `(position canonical order: QB, RB, WR, TE, K, DST), positional_rank ascending`. Pin the row sequence on a synthetic 4-position input.
16. **Determinism.** Calling `generate_snake_cheat_sheet` twice with identical inputs produces byte-identical output (compared via `assert_frame_equal`).
17. **Ruleset / roster_slots missing-position raises.** `LeagueConfig` requires K but VORP table has no K rows → raises `ValueError` matching `r"cannot fill \d+ K slots"` (delegated to `_select_pool`).
18. **Empty input raises.** Empty `vorp_table` with non-empty `LeagueConfig.roster_slots` → `ValueError` matching `r"cannot fill"` raised from `_select_pool`. Stricter than an "empty-in, empty-out" contract: failing loudly is the correct behavior when the caller asked for rankings at positions with no input data.
19. **`tiers_per_position = 1` produces all-tier-1.** Every in-pool player at every position has `tier = 1`.
20. **`tiers_per_position ≤ 0` raises.** Validation guard.
21. **Tier algorithm tie-break for equal-magnitude gaps.** Synthetic input with two identically-sized gaps competing for the N-1th-largest slot → the earlier (higher-rank) gap wins. Pin the exact partition.

### 5.2 Schema round-trip test

Appended to `tests/test_schemas/test_dataframe_schemas.py`:

22. **`SnakeCheatSheetSchema` round-trip.** Build a minimal valid frame; validate; re-validate; assert idempotent. Matches the pattern of the existing `test_vorp_table_schema_round_trip` and `test_auction_values_schema_round_trip`.

### 5.3 CLI integration tests in `test_generate_snake_cheat_sheet_cli.py`

Use the same `env={..., "PYTHONPATH": str(repo_root / "src")}` subprocess workaround the auction and VORP CLI tests use (PR #40 / #43 deviation note about the editable install pointing at other worktrees).

23. **End-to-end with a synthetic fixture.** Synthetic VORP parquet + synthetic `id_map.parquet` (built via `IdMapSchema`-valid rows) under `tmp_path` + synthetic `LeagueConfig` JSON → script produces an output CSV. Assert the per-position row count, the `tier` column dtype, the `display_name` column populated for known players, and exit code 0.
24. **Missing id_map file logs a warning and falls back to `"—"`.** Run the CLI with `--id-map` pointing at a path that doesn't exist. Assert exit 0, warning text in stderr, every output row has `display_name = "—"`.
25. **`--tiers-per-position 3` flag propagates.** Run with N=3; assert the output's `tier` column max value at every position is ≤ 3.

### 5.4 What's deliberately not tested

- **Cross-run tier stability.** Mentioned in §3.5 as expected behavior, not a contract. No test pins it.
- **Subjective "this looks right" cheat-sheet quality.** Pinned only via the stdout-summary eyeball pattern that auction and VORP already use. Numerical correctness is what gets tested.
- **id_map schema details.** This spec consumes the existing parquet via `IdMapSchema.validate`; if `IdMapSchema` changes shape, the consumer code updates. Not in scope here.
- **K / DST math.** Same answer as VORP — out of scope until TODO #10.

---

## 6. Open items, risks, and explicit out-of-scope

### Open items

**Cross-run tier instability.** §3.5. Documented; mitigated by stdout summary. If user feedback after a real draft is "the tiers wobble too much," add a follow-up spec for stability heuristics (e.g., minimum-tier-size constraint, or seeded random hysteresis).

**Display name source quality.** `id_map.parquet` is built from `nflreadpy.load_ff_playerids()` — a roster-wide source that includes pre-season players (rookies, free agents, current roster) without needing in-season game data. So name coverage for 2026 should be good even pre-season, contingent on `nflreadpy` having ingested the 2026 cohort. If a 2026 rookie or new signing is missing from id_map, their `display_name` falls back to `"—"`. Workaround for v1: re-run `build_id_map` to pick up new players, or the user manually edits the CSV. Long-term name coverage tracks `nflreadpy` upstream.

**Tier algorithm choice is one of several.** §3.2 ships gap-based, fixed N. Alternatives considered:
- Variable tier count from gap-size threshold (rejected — less predictable output structure; harder to reason about).
- 1D k-means with fixed k (rejected — less "natural cliff" interpretation; smooths over what should be sharp jumps).
- Fixed buckets by rank (rejected — ignores VORP gap signal entirely).
Each is a small follow-up flag if the gap-based default proves unsatisfying.

### Risks

**No ADP signal means the cheat sheet shows your model's view, not the room's view.** A drafter who only consults this sheet may reach for "value" players the room is happy to let them have, while missing players the room overvalues. Mitigation: the user knows this; ADP delta is the explicit §1.2 deferral. Manual ADP cross-reference during draft is workable for v1.

**`_select_pool` is now called from three places** (`auction.py`, `vorp.py`, `snake_cheat_sheet.py`). Any future refactor of pool selection must consider all three consumers. The auction test suite is the verification gate (PR #43 acceptance), and VORP / snake tests piggyback on the same pool contract.

**Tier algorithm produces nullable Int64 (`pd.Int64Dtype()`) for `tier`.** This is the correct pandas dtype for "integer with NaN support" — Object + plain ints + None or float + NaN would fail validation or quietly lose data (per CLAUDE.md "pyarrow / Int64Dtype" rule). The schema declares `nullable=True` for `tier`; tests pin the dtype.

**Depth_charts may contain a `gsis_id` mapped to multiple display names** (e.g., a player's name being updated mid-season, or two rows for different teams). The CLI picks the first; warns. Edge case rare in practice; not load-bearing for the spec.

### Explicit out-of-scope

- ADP / ADP delta (§1.2). Future follow-up.
- Confidence band (p10 / p90 floor / ceiling rank) (§1.2). Future follow-up.
- Markdown output (§1.2).
- Cross-position overall ranking (§1.2).
- Live snake-draft recommender (§1.2; `draft_ready_checklist.md` §2b.2).
- K / DST projections (TODO #10).
- Rookie projection handling (§1.2; `draft_ready_checklist.md` §1a).
- `predict_season.py` generalization (`draft_ready_checklist.md` §1b — inherited gap, not solved here).
- Multi-season / multi-ruleset / multi-league-config output in one run.
- Persistence to the store layer.

---

## 7. Acceptance

This spec is complete when:

- `generate_snake_cheat_sheet` is implemented per §2-3 in `src/projections/draft/snake_cheat_sheet.py` and re-exported from `src/projections/draft/__init__.py`.
- `SnakeCheatSheetSchema` is in `src/projections/schemas.py`.
- `scripts/generate_snake_cheat_sheet.py` runs end-to-end against a synthetic VORP parquet + synthetic `id_map.parquet` fixture and a sample `LeagueConfig` JSON.
- All §5 tests pass (≥ 25 new tests).
- `pytest -v` over the whole suite is clean.
- `mypy src tests` and `ruff check src tests` clean.
- `ruff format --check src tests` clean.
- `pytest -v -k "ingest or store or schemas"` passes (schema-seam guard per CLAUDE.md §4).
- `draft_ready_checklist.md` §2b.1 flipped from `[ ]` to `[x]`.
- `project_management.md` has a top entry describing the shipped surface and decision log.

No adoption gate. This is feature work, not a model-change probe.

---

## 8. Follow-ups (sequenced)

Each its own spec.

1. **ADP ingest + ADP-delta column** (`draft_ready_checklist.md` §2b.3). The biggest decision-relevance lift to the cheat sheet. FantasyPros has a free CSV export; Sleeper API exposes it. Either lands as an ingest spec + a small schema extension on `SnakeCheatSheetSchema`.
2. **Confidence band — p10 / p90 floor / ceiling rank.** Plumb `season_p10` and `season_p90` through to the cheat sheet. Cleanest path: extend `VorpTableSchema` to carry them (forces upstream / downstream changes; bigger blast radius) OR have the cheat-sheet CLI re-aggregate from `weekly_projections` directly (small blast radius; duplicates aggregation work). Decide in the follow-up spec.
3. **Live snake-draft recommender** (`draft_ready_checklist.md` §2b.2). The other §2b consumer. Two viable approaches sketched: greedy (highest-VORP available at position of need) and lookahead (ADP-simulated opponents; optimize expected total VORP across remaining rounds). The latter is ADP-blocked.
4. **Tier-stability variants.** If users find the default gap-based tiers too wobbly across runs, add `--tier-algorithm` flag accepting `gap` (default), `kmeans`, `fixed-buckets`. Trigger: user feedback after first real draft.
5. **Markdown export.** If draft-day UX favors phone / tablet over spreadsheet. `--markdown-out` as a second optional output flag.
6. **Cross-position overall board.** ADP-style overall ranking. Needs ADP first.

---

## 9. Implementation phases (for the plan)

**Phase 1 — Schema.** Append `SnakeCheatSheetSchema` to `schemas.py`. Add a round-trip test in `test_dataframe_schemas.py`. Run the schema-seam guard. Single commit.

**Phase 2 — Tier algorithm helper.** Implement `_assign_tiers` (§3.2) as a private helper in `snake_cheat_sheet.py`. Implement the unit tests pinning the algorithm (§5.1 #8, #9, #10, #19, #21). Pure numpy; no dependencies on the rest of the module yet. Single commit.

**Phase 3 — Public function.** Implement `generate_snake_cheat_sheet` in `snake_cheat_sheet.py`. Implement the algorithmic tests (§5.1 #1–7, #11–18, #20). Re-export from `src/projections/draft/__init__.py`.

**Phase 4 — CLI.** Implement `scripts/generate_snake_cheat_sheet.py`. Implement the CLI integration tests (§5.3 #23–25). Implement the id_map auto-join logic (re-validate via `IdMapSchema`, project to `(gsis_id, display_name)`) with the warn-on-missing fallback.

**Phase 5 — Integration + verification.** Run `pytest -v` over the entire test suite. Run `mypy src tests`, `ruff check src tests`, `ruff format --check src tests`. Run `pytest -v -k "ingest or store or schemas"` per CLAUDE.md §4. Flip `draft_ready_checklist.md` §2b.1 to `[x]`. Add the `project_management.md` top entry. Fix anything that breaks.

Each phase touches ≤ 5 files per CLAUDE.md §2. Phase 2 is the algorithmically loadbearing piece; the rest is plumbing.
