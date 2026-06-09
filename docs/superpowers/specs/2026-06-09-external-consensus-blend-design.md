# External Consensus Projection Layer — v1 blend (#2b slice 1)

**Date:** 2026-06-09
**Sub-project:** #2 (external consensus projection layer for draft), TODO #38, slice #2b-1
**Branch:** `feat/external-consensus-blend`
**Predecessor:** #2a ingest mechanism (PR #54) — `src/projections/ingest/external_projections.py`, `ExternalProjectionSchema`, store `asof` partition.

---

## 1. Purpose

Derive a single per-player **published preseason projection** from the already-ingested
`external_projections` snapshots, so downstream tools (Draft Hub first) consume one
consensus contract instead of raw multi-source rows.

v1 scope: a consensus ADP/ranking over the two sources we already ingest (ESPN + Sleeper)
plus ESPN-derived point estimates. This pins the consumer-facing contract over data we
already trust, **before** taking on web-scraping fragility (the next slice) and **before**
distribution-wrapping (also a later slice, when ≥2 stat-line sources give a real
cross-source spread to build floor/ceiling from).

### Why this ordering

ESPN is currently the **only** stat-line source; Sleeper is ADP-only. So:
- A consensus of **ADP/ranking** is buildable now (both sources provide ADP).
- A consensus of **projected points** requires ≥2 stat-line sources → needs scraping → next slice.
- A real probability **distribution** needs cross-source spread → needs the 2nd points source → later slice.

Building the blend module + published schema now, over data we already have, decouples the
consumer contract from the scraping work and gives the Draft Hub something to consume
(consensus ADP is the single most load-bearing draft input).

---

## 2. Scope

### In scope (this slice)
1. A fractional-aware scorer in the **scoring layer** (`expected_points`), because the
   existing `score()` requires integer realized counts and preseason projections are fractional.
2. A pure **consensus blend** function over one `external_projections` snapshot.
3. A published **`ConsensusProjectionSchema`** (consumer-facing contract).
4. An **orchestrator + CLI** (`refresh_consensus`) that reads a raw snapshot, builds, validates,
   and writes a derived snapshot via sanctioned store I/O.
5. Promote **`_placeholder_name_key`** out of `external_projections.py` into a shared identity util
   (TODO #38 directive; pure move, forward-looking for the next slice).

### Explicitly out of scope (later slices / TODO #38)
- Scraped sources (FantasyPros / CBS / NumberFire).
- A ≥2-source **points** consensus (we have one stat-line source today).
- **Distribution-wrapping** into the `Distribution` types.
- **Accuracy-weighting** of sources (simple mean only, per the TODO directive).
- Draft Hub consumption of this table.
- The cross-source rookie *matching refinement* (handling nickname/hyphen divergence, collision
  prevention). Only the `_placeholder_name_key` **move** lands here; the refinement stays queued.

---

## 3. Design

### 3.1 Scoring-layer addition — `src/projections/scoring/score.py`

The scoring layer is the only place that knows what counts as a fantasy point (CLAUDE.md), so
the fractional scorer lives here, not in the consensus module.

- Factor the per-field arithmetic in `score()` into a shared helper keyed on a mapping of
  stat-field → value. `score(StatLine, ruleset)` keeps its integer contract and delegates to it.
- Add:
  ```python
  def expected_points(line: Mapping[str, float], ruleset: Ruleset) -> float: ...
  ```
  Accepts fractional values (e.g. `receiving_tds=8.4`), uses the same ruleset coefficients,
  ignores absent keys (treated as 0.0). Unknown keys are ignored (the consensus passes only the
  9 canonical `STAT_FIELDS`).
- Invariant (tested): `expected_points(line, r) == score(StatLine(**line), r)` for any line whose
  count fields are whole numbers.

### 3.2 Consensus blend (pure) — `src/projections/consensus/blend.py`

New package `src/projections/consensus/` (sibling to `features/`, `scoring/`).

```python
def build_consensus(external: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame: ...
```

Input: one validated `external_projections` snapshot (all rows share one `season`, one `asof`;
one row per `(source, gsis_id)`). Output: one row per `gsis_id` conforming to
`ConsensusProjectionSchema` (pre-validation shape).

Per-`gsis_id` aggregation:
- `consensus_adp` = mean of non-null `adp` across the group's source rows; **null** if the group
  has no non-null `adp` (raw `adp` is `nullable=True`, so a stat-line-only / unranked player is
  possible in principle — in the 2026 snapshot every row has an ADP, but the contract must not
  assume it).
- `n_adp_sources` = count of non-null `adp` in the group (≥0).
- **Stat line**: for each of the 9 `STAT_FIELDS`, mean of the non-null values across the group's
  source rows (only ESPN carries them today, so the mean is just ESPN's value; the mean is the
  future-proof aggregation for when slice 2 adds a 2nd stat-line source). If **no** source in the
  group has any stat line, all 9 stat cols are null for that player.
- `has_points` = True iff the group has ≥1 stat-line source.
- `projected_points_ppr` = `expected_points(consensus_stat_line, ruleset)` if `has_points` else null.
- `full_name`, `position`, `is_placeholder_gsis`: carried through. Prefer the value from a
  stat-line-bearing source (ESPN) when present, else the first source's value. `position` must be
  a valid `Position` value.
- `consensus_rank`: ordinal rank (1-based, ascending) over `(consensus_adp, gsis_id)` computed
  **over the non-null-`consensus_adp` subset only**. The `gsis_id` secondary key makes ties
  deterministic. Lower `consensus_adp` → earlier pick → rank 1. A player with null `consensus_adp`
  gets a **null** `consensus_rank` (still appears in the table — union coverage — just unranked).

**Coverage = union of all sources** (decision): every player ranked by ≥1 source appears.
ADP-only players (e.g. Sleeper-only) get `consensus_adp` + `has_points=False` + null points/stats.

The function is pure (no I/O), so it is unit-testable with synthetic frames.

### 3.3 Published schema — `ConsensusProjectionSchema` in `schemas.py`

One row per `(gsis_id, season, asof)`:

| Column | Type | Notes |
|---|---|---|
| `gsis_id` | str (pyarrow) | real id or `99-XXXXXXX` placeholder; matches `GSIS_ID_PATTERN` |
| `season` | Int64 | `ge=1999, le=2100` |
| `asof` | str | ISO `YYYY-MM-DD`; mirrors the raw snapshot it was built from |
| `full_name` | str (pyarrow) | display only |
| `position` | str | `isin=_POSITION_VALUES` |
| `consensus_adp` | float | nullable; mean of available ADPs (100% populated in the 2026 snapshot, but not guaranteed) |
| `consensus_rank` | Int64 | nullable; ordinal over non-null `consensus_adp` by `(consensus_adp, gsis_id)`; null iff `consensus_adp` null |
| `n_adp_sources` | Int64 | ≥0 |
| `has_points` | bool | True iff a stat line was available |
| `projected_points_ppr` | float | nullable; null when `has_points=False` |
| `passing_yards` … `fumbles_lost` (9 cols) | float | nullable; the consensus stat line |
| `is_placeholder_gsis` | bool | carried from ingest |
| `ruleset` | str (pyarrow) | the ruleset name used for `projected_points_ppr` (default `ESPN_PPR`) |

Nullable dtypes per CLAUDE.md: `pd.Int64Dtype()` for count/rank/season cols, `pd.StringDtype("pyarrow")`
for nullable string cols. `Config.strict = "filter"`, `coerce = True`.

### 3.4 Orchestrator + CLI — `src/projections/consensus/refresh.py`

```python
def refresh_consensus(data_root: Path, *, season: int, asof: date | None = None) -> Path: ...
```

- Reads the raw `external_projections` snapshot: specific `asof` if given, else the **latest**
  available snapshot via the store's `read_latest_partition`.
- Calls `build_consensus(external, ruleset=Ruleset())` (default `ESPN_PPR`).
- `ConsensusProjectionSchema.validate(frame)` (with reassignment).
- Writes to a **new derived table** via sanctioned `write_partition`:
  `data/processed/consensus_projections/season=YYYY/asof=YYYY-MM-DD/part.parquet`.
  The consensus `asof` **mirrors** the raw snapshot's `asof` (the snapshot it was derived from),
  not "today" — so the derived table is reproducible from its raw input.
- `record_manifest(data_root, table="consensus_projections", season=season, df=frame)`.
- CLI: `python -m projections.consensus.refresh --season 2026 [--asof YYYY-MM-DD] [--data-root PATH]`.
- Error handling: if the raw snapshot is missing for `(season, asof)`, raise a clear error (no
  silent empty write). Mirror `ExternalProjectionError`'s shape with a `ConsensusError`.

**Output location decision:** `data/processed/` root marks the table as derived (vs. `data/raw/`
for ingested sources). Reuses the existing store `write_partition(root, table, …)` signature with
`root = data_root / "processed"`.

### 3.5 Identity util — `src/projections/ingest/identity.py`

Move `_placeholder_name_key` (currently private in `external_projections.py`) into a shared module
and re-import it there (`from projections.ingest.identity import placeholder_name_key`). Pure
refactor: ingest behavior and the existing placeholder ids are unchanged (verified by the existing
`external_projections` tests still passing). This satisfies the TODO #38 directive that the blend
and ingest "agree by construction"; the blend relies on the `gsis_id` ingest already assigned (it
does **not** re-match in this slice), so the util is forward-looking for slice 2's refinement.

---

## 4. Data flow

```
data/raw/external_projections/season=2026/asof=2026-06-09/part.parquet
        │  read_partition(asof=2026-06-09)   (or read_latest_partition)
        ▼
build_consensus(external, ruleset=ESPN_PPR)        # pure: group by gsis_id, mean ADP + stat line, score, rank
        │
        ▼
ConsensusProjectionSchema.validate(frame)
        │  write_partition(data/processed, "consensus_projections", asof=2026-06-09)
        ▼
data/processed/consensus_projections/season=2026/asof=2026-06-09/part.parquet
        + record_manifest(table="consensus_projections", season=2026)
```

---

## 5. Testing

All test files are network-free (operate on synthetic frames / fixture parquet).

- **`expected_points`** (`tests/test_scoring/`): fractional cases under `ESPN_PPR` / `ESPN_HALF`;
  the int-equivalence invariant vs. `score()`; absent-key handling.
- **`build_consensus`** (`tests/test_consensus/`):
  - 2-source veteran → mean ADP, `n_adp_sources=2`, stat line passes through, points scored.
  - Single-source ADP-only player (Sleeper-only) → `has_points=False`, null points/stats,
    `n_adp_sources=1`, still appears (union coverage).
  - ESPN-only player → points present, `n_adp_sources=1`.
  - Player with a stat line but no ADP in any source → `consensus_adp`/`consensus_rank` null,
    `n_adp_sources=0`, `has_points=True`, still appears (union coverage).
  - Placeholder rookie present in both sources → one row, `is_placeholder_gsis=True`, carried through.
  - Deterministic `consensus_rank` tie-break on equal `consensus_adp`.
  - `position` is a valid `Position` value; ESPN-preferred when sources disagree.
- **Schema** (`tests/test_schemas/`): `ConsensusProjectionSchema` accepts a well-formed frame;
  rejects wrong dtypes; nullable cols accept `pd.NA`.
- **`refresh_consensus`** (`tests/test_consensus/`): write a fixture raw `external_projections`
  snapshot to a tmp store → `refresh_consensus` → read back the derived partition → assert rows +
  schema + that consensus `asof` mirrors the raw `asof`; missing-snapshot raises `ConsensusError`.

### Verification gates (CLAUDE.md end-of-effort checklist)
`pytest -v` (or the consensus/scoring/schemas/store/ingest subset, stated), `mypy src tests`,
`ruff check src tests`, `ruff format --check src tests`. Because this touches a pandera schema and
a store path, also run `pytest -v -k "ingest or store or schemas"`.

---

## 6. Decisions log (from brainstorming)

- **First slice = blend over existing 2 sources** (not scrape-first), de-risking the consumer
  contract before scraping fragility.
- **Output = point estimates now, distributions next slice** — no synthetic spread from a single
  points source; `n_adp_sources` / `has_points` make coverage honest.
- **Coverage = union of all sources** — broadest draft coverage; ADP-only players carried with null points.
- **Simple mean** for ADP and stat line (TODO directive; accuracy-weighting deferred).
- **`expected_points` in the scoring layer** — surfaced during brainstorming: `score()`'s integer
  `StatLine` can't score fractional preseason projections; reuse ruleset coefficients via a shared helper.
- **Derived table under `data/processed/`**; consensus `asof` mirrors the raw snapshot's `asof`.
- **Default ruleset `ESPN_PPR`**; the stat line is stored so consumers can re-score under any ruleset.

---

## 7. Open follow-ups (NOT this slice)

- Slice 2: scraped stat-line source(s) → real ≥2-source points consensus → Distribution-wrapping
  (real cross-source spread for floor/ceiling).
- Cross-source rookie matching refinement (TODO #38): nickname/hyphen divergence, collision prevention.
- Accuracy-weighted blend (needs realized-outcome history per source).
- The `external_projections.py:398` pandas `FutureWarning` cleanup (logged under TODO #38).
