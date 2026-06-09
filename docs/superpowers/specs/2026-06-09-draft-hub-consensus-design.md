# Draft Hub on Consensus — repoint VORP + ADP-delta cheat sheet

**Date:** 2026-06-09
**Sub-project:** #2 (external consensus projection layer → Draft Hub), TODO #38
**Branch:** `feat/draft-hub-consensus`
**Predecessors:**
- #2b slice 1 consensus blend (PR #55) — `ConsensusProjectionSchema`, `build_consensus`, `refresh_consensus`.
- Draft Hub surfaces (already shipped, May 2026): VORP (`2026-05-16-vorp-design.md`), auction $ (`2026-05-16-auction-values-design.md`), snake cheat sheet (`2026-05-16-snake-cheat-sheet-design.md`).
- Benchmark spike (PR #52) — verdict: the home-grown in-season model **cannot** produce a preseason projection; pivot to external for draft.

---

## 1. Purpose

The Draft Hub surfaces (VORP → auction $ → snake cheat sheet) are **already built, tested, and
CLI-wrapped**. They are fed the wrong projection: the VORP CLI reads `ProjectionWeeklySchema` (the
home-grown **in-season** model) and aggregates it to a season total — exactly the projection the
PR #52 spike proved is invalid for preseason draft (e.g. it "projected" the injured CMC for 4 weeks
at 63 pts vs ESPN's honest preseason 335).

This slice makes the Draft Hub **draft-valid** by re-pointing it at PR #55's published
`ConsensusProjectionSchema`, and uses the consensus's `consensus_adp` to add the **ADP-delta**
column the snake cheat sheet deferred at ship time (its #1 logged limitation: "no ADP signal means
the cheat sheet reflects model view, not room view").

We change projection *source*, not draft *math*. `generate_vorp_table`, `generate_auction_values`,
`_select_pool`, and the weekly→season path are untouched.

---

## 2. Scope

### In scope (this slice)
1. A pure **consensus → season-projection adapter** that converts `ConsensusProjectionSchema` rows
   into the `ProjectionSeasonSchema`-shaped frame `generate_vorp_table` already accepts.
2. A **consensus source-mode** on the VORP CLI (`--source consensus`), reading the published
   consensus partition instead of the weekly model, and writing the same VORP parquet — now also
   carrying `consensus_adp`.
3. An **optional, nullable `consensus_adp` column** on `VorpTableSchema` (the channel that carries
   ADP from the consensus snapshot through to the cheat sheet).
4. An **ADP-delta** addition to the snake cheat sheet: `consensus_adp` (raw market signal) +
   `adp_delta` (within-position market-rank − value-rank; positive = value), with two new columns on
   `SnakeCheatSheetSchema`.
5. A **skill-only example league config** for the consensus path (consensus has no K/DST).

### Explicitly out of scope (later slices / other TODOs)
- **K/DST** in the draft tooling — consensus has no kicker/defense rows; stays TODO #10.
- **Confidence band / floor-ceiling** on the cheat sheet — consensus is point-only (single
  stat-line source); real spread arrives with the scraping slice (TODO #38 #2b+).
- **Auction $ changes** — the auction generator already consumes the VORP parquet and needs no
  change; it ignores the new `consensus_adp` column.
- **The weekly→season path** — left intact for the separate start/sit question (TODO #39); it is
  simply no longer the *draft* basis. `--source weekly` remains and keeps its current behavior.
- **Pure-ADP-only players** (Sleeper ADP, no ESPN points) — they are the long tail beyond the
  draftable universe and have no points to rank on; the draft surfaces cover the `has_points`
  population (see §3.1).

---

## 3. Design

### 3.1 Consensus → season-projection adapter — `src/projections/draft/consensus_source.py`

New module in the `draft` package. Pure, no I/O.

```python
def consensus_to_season_projections(consensus: pd.DataFrame) -> pd.DataFrame: ...
```

Input: a validated `ConsensusProjectionSchema` frame (one snapshot). Output: a
`ProjectionSeasonSchema`-validated frame (the contract `generate_vorp_table` accepts).

**Single-snapshot precondition:** the adapter **asserts a single `asof` and a single `season`** in
the input (mirroring `build_consensus`'s single-`ruleset`/`season` guards); it raises a clear error
if the frame mixes snapshots. `asof` is read from the frame (not a parameter), and is used for
`model_id`. A multi-`asof` frame is a caller bug (the CLI reads exactly one partition).

Transform:
- **Filter to `has_points == True`.** VORP needs a non-null season-points total; `projected_points_ppr`
  is null for players no stat-line source covers. ADP-only players are dropped here (consistent with
  the "consensus-only points" decision). The `has_points` population (~458 in the 2026 snapshot) is
  far larger than any single-league draftable universe. **The set of players dropped here that have a
  draftable ADP is not silent — the CLI surfaces it (§3.3).**
- `season_mean` ← `projected_points_ppr` (float64).
- **Degenerate distribution:** `season_p10 = season_p50 = season_p90 = season_mean`. This is the
  honest representation of a point estimate (no spread to claim), and matches `NaivePreseasonModel`'s
  own degenerate point-mass convention. A real band waits for ≥2 stat-line sources.
- `n_weeks = 17` (sentinel = "full-season total, not a per-week aggregate"; `17` reads as a complete
  season rather than the misleading `1` = "one week of data". Schema requires `1..22`.). **Documented
  contract:** consumers must not treat a consensus-derived `ProjectionSeasonSchema` row's `n_weeks` as
  a real per-week count; VORP does not read `n_weeks`, and no consumer should filter consensus rows on
  it. (Noted in the adapter docstring.)
- `position` ← consensus `position`; `season` ← consensus `season` (coerced to `int`).
- `ruleset` ← consensus `ruleset` (carried through, so the existing VORP ruleset-match guard fires
  if the caller's `LeagueConfig.ruleset` disagrees).
- `model_id = f"consensus:{asof}"` (provenance — the snapshot the table was derived from).
- `generated_at = pd.Timestamp.now(tz="UTC")` (tz-aware; `ProjectionSeasonSchema` rejects naive).

**`consensus_adp` is intentionally *not* on this frame** — `ProjectionSeasonSchema` has
`strict="filter"` and would drop it anyway. ADP is carried separately by the CLI (§3.3), because the
VORP *math* does not use it.

Returns `ProjectionSeasonSchema.validate(frame)`. Empty input (no `has_points` rows) → a valid empty
frame (downstream `generate_vorp_table` already handles empty).

### 3.2 `VorpTableSchema` — add optional nullable `consensus_adp` — `schemas.py`

Append one column, marked **Optional** (not-required) so legacy weekly-path parquets that lack it
still validate (same pattern as `PreseasonFeaturesSchema`'s per-position `Optional[Series[...]]`
columns):

```python
consensus_adp: Series[pd.Float64Dtype] | None = pa.Field(gt=0, nullable=True)
```

- Weekly path: column absent → frame still validates (Optional).
- Consensus path: column present, populated from the snapshot; nullable per-row (a `has_points`
  player could in principle lack an ADP).
- `generate_vorp_table` is **unchanged** — it selects its explicit `_OUTPUT_COLUMNS` and never emits
  `consensus_adp`; the CLI joins ADP onto the output afterward (§3.3). The auction generator reads
  its own column subset and is unaffected.

### 3.3 VORP CLI — consensus source-mode — `scripts/generate_vorp_table.py`

Add a `--source {weekly,consensus}` flag (**default `weekly`**, preserving current behavior and all
existing CLI tests). New flags used only in consensus mode:
- `--data-root` (default `data`) — the store root; consensus lives at
  `<data-root>/processed/consensus_projections/season=YYYY/asof=YYYY-MM-DD/`.
- `--asof YYYY-MM-DD` (optional) — specific snapshot; default = the **latest** available
  (`read_latest_partition`).

Consensus-mode flow:
1. Read the consensus partition (`read_partition(..., asof=...)` for a named snapshot, else
   `read_latest_partition(root, "consensus_projections", season=…)`), `ConsensusProjectionSchema.validate(...)`.
2. `season_proj = consensus_to_season_projections(consensus)` (§3.1).
3. **Dropped-but-draftable warning (required, the §3.1 #2 mitigation).** Before/after the adapter,
   compute the players the `has_points` filter dropped whose `consensus_adp` falls inside the
   draftable range (`consensus_adp <= league_config.total_pool_size`). If any, emit a stderr warning
   listing count + `full_name`/`consensus_adp`/`consensus_rank` for each (capped, e.g. top 25). This
   converts the silent coverage gap into an explicit eyeball check at draft-prep time. Zero such
   players → a one-line "0 draftable players dropped for missing points" confirmation.
4. `vorp = generate_vorp_table(season_proj, league_config)` (unchanged function).
5. **Left-join `consensus_adp`** from the consensus snapshot onto `vorp` by `gsis_id`, then
   `VorpTableSchema.validate(...)` (now includes the optional column).
6. Write CSV/parquet (existing `_write_output` logic) + the existing per-position stdout summary.

`--weekly-projections` becomes required only for `--source weekly` (argparse validation: error
clearly if the flag needed for the chosen source is missing). Mode dispatch is a small branch at the
top of `main()`; the summary/writing tail is shared.

### 3.4 Snake cheat sheet — ADP-delta — `src/projections/draft/snake_cheat_sheet.py` + `schemas.py`

`generate_snake_cheat_sheet` stays a pure transform over the (now ADP-bearing) VORP frame. Two new
`SnakeCheatSheetSchema` columns:

| Column | Type | Notes |
|---|---|---|
| `consensus_adp` | Float64 | nullable; raw market ADP (the room's view). NA when the input has no ADP (weekly path). |
| `adp_delta` | Int64 | nullable; within-position *(ADP-rank − VORP-rank)*. Positive = value (market drafts them later than their VORP warrants); negative = reach. NA when `consensus_adp` is NA. |

Computation (added after the existing `positional_rank` / tier logic):
- If the input frame has no `consensus_adp` column (weekly path) → both new columns are all-NA;
  output is otherwise identical to today (backward compatible).
- Else, **within each position**, over the subset of rows with non-null `consensus_adp`, compute two
  **deterministic, gap-free integer ranks** using the same `sort + cumcount` idiom the existing
  `positional_rank` uses (never `Series.rank()`, whose default `method="average"` yields fractional
  ranks that violate the `Int64` contract):
  - `adp_rank`: sort the subset by `(consensus_adp ascending, gsis_id ascending)`, then `cumcount()+1`
    (lower ADP = earlier pick = rank 1; `gsis_id` is the deterministic tie-break).
  - `vorp_rank`: sort the subset by `(vorp descending, gsis_id ascending)`, then `cumcount()+1`
    (same tie-break key, so ties resolve identically to `adp_rank`).
  - `adp_delta = adp_rank − vorp_rank` (Int64). Positive = value (market drafts later than VORP rank);
    negative = reach.
  Rows with null `consensus_adp` get null `adp_delta`. `consensus_adp` is passed through as-is.
  Both ranks share one population (the position's non-null-`consensus_adp` rows) so the delta is
  self-consistent; the existing all-rows `positional_rank` is untouched.
- `positional_rank` (the existing all-rows VORP rank) is **unchanged** — `adp_delta` is a separate,
  self-consistent within-subset comparison so a player missing ADP doesn't shift everyone's value rank.

The CLI (`generate_snake_cheat_sheet.py`) needs no new flags — it already reads the VORP parquet,
which now carries `consensus_adp`. The per-position stdout summary gains the top-3 `adp_delta` for an
eyeball check (biggest values / reaches).

### 3.5 Skill-only example league config — `configs/league_espn_ppr_12team_skill.json`

The shipped configs (`league_espn_ppr_12team.json`, `league_espn_half_10team.json`) include `K: 1`
and `DST: 1`. Fed consensus (no K/DST rows), `_select_pool` raises "cannot fill N K slots". Add a
skill-only config — same scoring/teams/budget, `roster_slots` limited to QB/RB/WR/TE/FLEX/BENCH — for
the consensus path. The existing configs stay for the (K/DST-capable, but draft-invalid) weekly path
and future TODO #10 work.

**Ruleset coupling (v1 = `espn_ppr` only):** the consensus snapshot's `ruleset` (hence its
`projected_points_ppr`) is fixed at `refresh_consensus` time. **As built, `refresh_consensus`
hardcodes `Ruleset()` = `ESPN_PPR` (`refresh.py:59`) — there is no `--ruleset` flag.** Because the
adapter carries the snapshot's ruleset through and `generate_vorp_table` guards
`LeagueConfig.ruleset` against it, a half-PPR or standard skill config would trip that guard against
an `ESPN_PPR` snapshot. So **half/standard drafts are blocked until a separate consensus-layer change
adds ruleset selection to `refresh_consensus`** (re-scoring the stored stat line under the chosen
ruleset) — that is *not* in this slice's scope. v1 ships the `espn_ppr` skill config only; this is a
known limitation, logged in §7, not a config the user can simply author.

**Pool-size note (decision §5):** dropping K/DST shrinks `total_pool_size`, which nudges
replacement levels for the skill positions slightly upward vs. a full 16-man roster. This is an
accepted v1 approximation — K/DST are drafted off-tool (late, off ADP) until TODO #10. Documented in
the config and the CLI summary.

---

## 4. Data flow

```
data/processed/consensus_projections/season=2026/asof=YYYY-MM-DD/part.parquet   (PR #55 output)
        │  read_partition(asof) | read_latest_partition   +  ConsensusProjectionSchema.validate
        ▼
consensus_to_season_projections(consensus)        # filter has_points; points→season_mean; degenerate band
        │  → ProjectionSeasonSchema frame                 (consensus_adp dropped here, carried by CLI)
        ▼
generate_vorp_table(season_proj, league_config)   # UNCHANGED
        │  + left-join consensus_adp by gsis_id  →  VorpTableSchema.validate (incl. optional col)
        ▼
VORP parquet  (gsis_id, position, season_mean_fpts, vorp, replacement_fpts, consensus_adp)
        │
        ├─► generate_auction_values  (unchanged; ignores consensus_adp)        → auction $ table
        └─► generate_snake_cheat_sheet  (+ consensus_adp, adp_delta)           → cheat sheet
```

---

## 5. Decisions log (from brainstorming)

- **Consensus-only points source** (user-confirmed). The cleanest realization of the spike's
  "pivot to external for draft" verdict; defers multi-source points quality to the scraping slice.
  Rejected: blending in the home-grown `PreseasonModel` to fill ADP-only players — it would re-mix
  an unvalidated home-grown source into the basis the spike moved away from, and muddy provenance.
- **Repoint + ADP-delta scope** (user-confirmed). Closes the cheat sheet's #1 logged gap now that
  `consensus_adp` exists. Confidence-band explicitly excluded (point-only source).
- **Change source, not math.** `generate_vorp_table` / auction / `_select_pool` untouched; the seam
  is a pure adapter + a CLI mode. Minimal blast radius into tested code.
- **`consensus_adp` rides the VORP parquet** as the single channel to the cheat sheet (the cheat
  sheet already reads only that parquet). Added to `VorpTableSchema` as an **Optional** column for
  backward compatibility with the weekly path.
- **Degenerate distribution in the adapter** (`p10=p50=p90=mean`) — honest for a point estimate;
  matches `NaivePreseasonModel`. No synthetic spread.
- **`adp_delta` = within-position (ADP-rank − VORP-rank), positive = value** — self-consistent over
  the position's ADP-bearing subset; leaves the existing `positional_rank` untouched.
- **Skill-positions-only for v1** — consensus has no K/DST; ship a skill-only config and document
  the pool-size approximation. K/DST stays TODO #10.
- **`has_points` is the draft surface** — pure-ADP-only players (no ESPN points) are the long tail
  and drop out; the ~458 has_points players dwarf any single league's draftable set.

---

## 6. Testing

All tests network-free (synthetic frames / fixture parquet).

- **`consensus_to_season_projections`** (`tests/test_draft/test_consensus_source.py`):
  - `has_points` filtering (ADP-only rows dropped; points rows kept).
  - `projected_points_ppr` → `season_mean`; degenerate `p10=p50=p90=mean`; `n_weeks=17`.
  - `ruleset` / `season` / `position` carried through; `model_id == f"consensus:{asof}"`;
    `generated_at` tz-aware UTC (passes `ProjectionSeasonSchema`'s naive-rejection parser).
  - Empty / all-ADP-only input → valid empty `ProjectionSeasonSchema` frame.
  - Output validates `ProjectionSeasonSchema`.
- **`VorpTableSchema`** (`tests/test_schemas/`): round-trip with `consensus_adp` present; a frame
  **without** `consensus_adp` still validates (Optional column); `gt=0` + nullable enforced.
- **`generate_vorp_table` unchanged** — existing `tests/test_draft/test_vorp.py` stays green
  (regression gate that the math didn't move).
- **`generate_snake_cheat_sheet` ADP-delta** (`tests/test_draft/test_snake_cheat_sheet.py`):
  - Input without `consensus_adp` → both new columns all-NA, all other output byte-identical to
    today (backward-compat).
  - A clear **value** (high VORP-rank, late ADP) → positive `adp_delta`; a clear **reach** → negative.
  - Null-`consensus_adp` row → null `adp_delta`, but `consensus_adp` passed through where present.
  - Within-position population isolation (a missing-ADP row doesn't shift others' `adp_delta`).
  - `SnakeCheatSheetSchema` round-trip with the two new columns.
- **VORP CLI consensus mode** (`tests/test_scripts/test_generate_vorp_table_cli.py`): write a fixture
  consensus partition to a tmp store → `--source consensus` → read back VORP parquet → assert rows,
  `consensus_adp` populated, schema valid; `--asof` selects the named snapshot; default picks latest;
  ruleset mismatch (config vs snapshot) raises clearly; `--source weekly` path unchanged.
- **Cheat-sheet CLI** (`tests/test_scripts/test_generate_snake_cheat_sheet_cli.py`): a consensus-fed
  VORP parquet flows through to a cheat sheet carrying `consensus_adp` / `adp_delta`.
- **Auction CLI regression guard** (`tests/test_scripts/test_generate_auction_values_cli.py`): run
  the auction CLI on a consensus-fed VORP parquet (one that **includes** the `consensus_adp` column)
  and assert the output equals the output for the same parquet without that column. Guards the
  "auction unaffected" claim (`_read_vorp` is subset-based and should stay blind to the extra column).
- **Dropped-but-draftable warning** (VORP CLI consensus mode test): a fixture consensus snapshot with
  an ADP-only player inside `total_pool_size` → assert the stderr warning names it; a snapshot with
  none → assert the "0 draftable players dropped" confirmation.

### Verification gates (CLAUDE.md end-of-effort checklist)
`pytest -v` (or the stated `draft or scripts or schemas` subset), `mypy src tests`,
`ruff check src tests`, `ruff format --check src tests`. Touches a pandera schema + a store-read
path, so also run `pytest -v -k "ingest or store or schemas"`.

### Live 2026 smoke (manual, post-merge)
`refresh_external_projections` → `refresh_consensus` (PR #54/#55) →
`generate_vorp_table --source consensus --season 2026 --league-config configs/league_espn_ppr_12team_skill.json`
→ `generate_snake_cheat_sheet`. Eyeball: Bijan Robinson near the top of RB by VORP; `adp_delta`
signs sane (consensus #1 ADP players show small deltas; known late-ADP producers show positive).

---

## 7. Open follow-ups (NOT this slice)

- **K/DST in the Draft Hub** (TODO #10) — needs K/DST projections + a pool-fill that tolerates them;
  unblocks the full ESPN/Yahoo roster shape.
- **Confidence band on the cheat sheet** — unblocked once the scraping slice gives ≥2 stat-line
  sources (real cross-source spread → p10/p90 → floor/ceiling rank).
- **Overall (cross-position) draft board** — the current cheat sheet is per-position; a VORP-ranked
  overall board with ADP-delta is a natural follow-up surface.
- **ADP staleness / riser-faller** — the consensus `asof` series (multiple snapshots) can power an
  ADP-trend column; out of scope here.
- **Non-`espn_ppr` rulesets** — half-PPR / standard drafts need `refresh_consensus` to accept a
  ruleset (re-scoring the stored stat line); a consensus-layer change, separate from this draft slice
  (see §3.5). v1 is `espn_ppr`-only.
