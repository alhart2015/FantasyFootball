# External Projection Ingest Mechanism (v1) — Design

**Date:** 2026-06-08
**Branch:** `feat/external-projection-ingest`
**Status:** spec
**Sub-project:** #2a (first slice of TODO #38, "external consensus projection layer for draft")
**Predecessor:** External Projection Benchmark Spike (PR #52). The spike proved our home-grown model cannot produce a preseason/draft projection (it's a weekly, in-season model) and that ESPN + Sleeper are genuine, free, no-auth preseason sources, pullable for any season. The spike's `scripts/pull_external_projections.py` (a throwaway script) is the seed promoted here into a real ingest source.

---

## 1. Goal & scope

### 1.1 Goal

A **repeatable, dated-snapshot ingest** of ESPN + Sleeper preseason projections into the sanctioned store, canonically `gsis_id`-keyed (with placeholder ids for pre-camp rookies), re-runnable to refresh weekly from now through draft day (early August 2026). The mechanism is the foundation the consensus blend and Draft Hub sit on; the data itself is expected to be re-pulled and improved as draft day approaches — the deliverable is the **mechanism**, not a one-time data drop.

### 1.2 In scope (v1)

1. New ingest source `src/projections/ingest/external_projections.py` following the canonical `refresh_<source>` template (promotes the spike's parsers into `src/`, properly tested).
2. ESPN (preseason projected stat line + ADP + PPR draft rank) and Sleeper (ADP only) — the two verified no-auth sources.
3. Dated-snapshot storage: each refresh writes a point-in-time snapshot keyed by pull date (`asof`), accumulating a time series.
4. A small, reusable extension to the store (`write_partition` / `read_partition` gain an optional `asof` date partition).
5. `ExternalProjectionSchema` in `schemas.py`.
6. Rookie placeholder `gsis_id`s — **scoped to this table only** (§5), deterministic + auto-reconciling.
7. Fix the `id_map` float-stringified-id defect (`'4374302.0'`) at ingest so the crosswalk is clean.
8. A thin CLI to run a refresh: `python -m projections.ingest.external_projections --season 2026`.

### 1.3 Out of scope (named follow-ups, #2b+)

- **Consensus blend** across sources (a true multi-source average needs ≥ 2 stat-line sources; only ESPN provides a stat line today). The schema (one row per source) makes the blend a pure read-time/aggregation add.
- **Scraped sources** (FantasyPros / CBS / NumberFire) for real consensus.
- **Distribution-wrapping** (external point estimates → `Distribution` types).
- **The Draft Hub** (rankings, ADP, VORP, tiers, riser/faller views) that consumes this data.
- **Pipeline-wide placeholder gsis_ids** (TODO #37 Path 2 across all 8 ingest seams) — explicitly NOT this; see §5.
- The other TODO #38 graduation cleanups (shared report-path constant, score-actuals consolidation) — unrelated to ingest; not touched here.

### 1.4 Success criteria

Ships when: `refresh_external_projections(data_root, season=2026)` runs end-to-end against the live APIs and writes a validated `asof`-dated snapshot containing both veterans (real `gsis_id`) and rookies (placeholder `gsis_id`); a second run on a later (simulated) `asof` produces a second snapshot without clobbering the first; pure transforms are unit-tested; mypy strict + ruff + ruff format clean; relevant `pytest -k "ingest or store or schemas"` green.

---

## 2. Architecture & components

New module `src/projections/ingest/external_projections.py`, mirroring `src/projections/ingest/schedules.py`'s shape:

| Unit | Responsibility | Network? |
|---|---|---|
| `fetch_espn(season, limit=800)` | GET ESPN `kona_player_info` for `season`; return raw JSON. | yes |
| `fetch_sleeper_adp(season)` | GET Sleeper season projections; return raw list. | yes |
| `parse_espn_players(payload, season)` | Tidy → one row per QB/RB/WR/TE with preseason projected stat line + `espn_id` + ADP + PPR draft rank. (Promoted + reused from the spike, already verified.) | no |
| `parse_sleeper_adp(payload)` | Tidy → `sleeper_id` + ADP. | no |
| `_to_canonical(parsed, source, season, asof, id_map)` | Normalize each source's tidy frame into `ExternalProjectionSchema` rows: rename to the common stat fields, attach `gsis_id` via crosswalk + placeholder (§5), stamp `source`/`season`/`asof`, validate. | no |
| `refresh_external_projections(data_root, *, season, asof=None)` | Orchestrator: for each source fetch → parse → `_to_canonical` → concat → `ExternalProjectionSchema.validate` → `write_partition(..., asof=asof)`. `asof` defaults to today (UTC date). Returns the written path(s). | yes |
| `main()` (CLI) | `--season` (required), `--asof` (default today), `--data-root` (default `data`). | yes |

Pure parsers + `_to_canonical` + the placeholder generator are unit-tested with captured/synthetic fixtures; `fetch_*` and `main` are verified by running.

ESPN stat-id → field decode and the endpoint URLs are carried over verbatim from the spike (`ESPN_STAT_IDS`, verified end-to-end against real 2024 + 2026 data). The 9 common scoring fields are the canonical set: `passing_yards, passing_tds, interceptions, rushing_yards, rushing_tds, receptions, receiving_yards, receiving_tds, fumbles_lost`.

---

## 3. Storage — dated snapshots

### 3.1 Layout

```
data/raw/external_projections/season=2026/asof=2026-07-15/part.parquet
```

`asof` is the pull date (ISO `YYYY-MM-DD`). Each refresh writes one snapshot. Re-running on the same `asof` overwrites that snapshot (idempotent per day). "Latest" = the max `asof` present. The accumulated snapshots are the time series that later powers ADP / projection movement in the Draft Hub.

### 3.2 Store extension

The store currently partitions by `season` / `week` only (`{table}/season=YYYY/week=WW/part.parquet`). Extend the sanctioned helpers with an **optional `asof` partition** so external-projection I/O stays in the store layer (per the "`store.write_partition`/`read_partition` are the only sanctioned parquet I/O" convention):

- `write_partition(root, table, df, *, season=None, week=None, asof=None) -> Path` — when `asof` is set, the path gains `/asof=YYYY-MM-DD`. `asof` is a `datetime.date`; rendered ISO.
- `read_partition(root, table, *, season=None, week=None, asof=None) -> pd.DataFrame` — `asof=None` with a season set reads **all** `asof` snapshots under that season and concatenates (each row already carries its `asof` column, so they remain distinguishable); a specific `asof` reads that one snapshot.
- Add `read_latest_partition(root, table, *, season) -> pd.DataFrame` (or a documented `asof="latest"` sentinel) returning only the max-`asof` snapshot — the common "give me current projections" read.

`asof` composes with `season` (both can be set); `week` and `asof` are not used together for this table. Existing `season`/`week` callers are unaffected (new keyword defaults to `None`).

---

## 4. `ExternalProjectionSchema`

One row per (source, player, season, asof). New pandera schema in `schemas.py`, `strict="filter"`, following the existing schema conventions (`pd.StringDtype("pyarrow")` for nullable strings, `pd.Int64Dtype()` for nullable ints).

| Column | Type | Notes |
|---|---|---|
| `source` | str (isin) | `"ESPN"` / `"SLEEPER"` (new `ProjectionSource` enum; reference the enum, never the string). |
| `source_player_id` | str (pyarrow) | Stable native id (ESPN/Sleeper player id). The cross-snapshot join key. |
| `gsis_id` | str (pyarrow), matches `GSIS_ID_PATTERN` | Real where the crosswalk matched, else a placeholder (§5). |
| `is_placeholder_gsis` | bool | `True` for rookies/unmatched (placeholder id). |
| `full_name` | str (pyarrow) | Display only. |
| `position` | str (isin `_POSITION_VALUES`) | QB/RB/WR/TE. |
| `season` | int | e.g. 2026. |
| `asof` | date / str | Pull date (also encoded in the partition path; kept in-row so concatenated reads stay distinguishable). |
| `adp` | float, nullable | Average draft position (both sources). |
| `espn_draft_rank` | Int64, nullable | ESPN's `draftRanksByRankType.PPR.rank` (an **overall** PPR draft rank, per the spike's observation — verify positional-vs-overall against real data at implementation time); null for Sleeper. |
| 9 stat fields | float, **all nullable** | `passing_yards … fumbles_lost`. Populated for ESPN; all-null for Sleeper (ADP-only). |

**We store the stat line, not fantasy points** — the scoring layer (`projections.scoring`) converts to points under any ruleset downstream, per the repo convention ("scoring is the only place that knows what a fantasy point is"). Sources that give only ADP (Sleeper) carry a null stat line but a non-null `adp`; that is a valid, expected row shape.

---

## 5. Rookie placeholder `gsis_id`s (narrow scope)

**Problem (per TODO #37):** 2026 rookies have no real `gsis_id` until ~late July (nflverse uses PFR-style placeholders; `id_map` omits them). ESPN already projects them (they have an `espn_id`). A strict crosswalk-and-drop would make the entire 2026 rookie class — the #1 overall pick included — invisible in the draft data through early July.

**Approach:** `_attach_gsis_id` crosswalks `source_player_id → id_map` (ESPN via `espn_id`, Sleeper via `sleeper_id`). For matched players, use the real `gsis_id`. For unmatched players, generate a **deterministic placeholder**:

```
placeholder = f"99-{int(hashlib.sha1(f'{source}:{source_player_id}'.encode()).hexdigest(), 16) % 10_000_000:07d}"
```

This matches `GSIS_ID_PATTERN` (`\d{2}-\d{7}`), uses the otherwise-unused `99-` prefix to mark it as synthetic, and sets `is_placeholder_gsis=True`.

**Why this stays cheap (NOT TODO #37's 2-3 day estimate):** that estimate was for propagating placeholders through **all 8 ingest seams** (weekly_stats, depth_charts, draft_picks, …) so the *home-grown weekly model* could project rookies. We retired that model for draft. Here the placeholder lives **only in `external_projections`** — no other ingest seam, join path, or `validate_gsis_id` call site is touched.

**Auto-reconciliation:** because (a) the placeholder is deterministic and (b) every row also carries `source_player_id`, no separate reconciliation job is needed. Once `id_map` propagates the real `gsis_id` (~late July), the *next* refresh's snapshot carries the real id for that player. Old snapshots keep their placeholder (correct — they were point-in-time). **Cross-snapshot tracking joins on `source_player_id`** (stable across the placeholder→real flip), so a rookie's June→August ADP movement remains followable.

**Convention note:** the repo says "`GsisId` is canonical; all internal storage and joins use it." This design keeps `gsis_id` populated on every row (real or placeholder) and uses it for internal joins; the deliberate, documented nuance is that for external preseason data, some `gsis_id`s are synthetic placeholders flagged by `is_placeholder_gsis`. Consumers that must exclude synthetic ids filter on that flag.

---

## 6. In-scope `id_map` fix

The spike found `id_map.parquet` stores `espn_id`/`sleeper_id` as float-stringified values (`'4374302.0'`), which forced a consumer-side `.str.replace(r'\.0+$','')` normalize. Fix at the source in `src/projections/ingest/id_map.py`: coerce these id columns to a clean integer-string (or nullable `Int64` → string) before persisting, so the crosswalk join in §5 matches without a workaround. Add a regression test pinning that a float-valued upstream id persists as `'4374302'`, not `'4374302.0'`. (This also retroactively lets the spike's `benchmark_projections._normalize_join_id` workaround be simplified later — out of scope here, just noted.)

---

## 7. Error handling

- `fetch_*`: catch `urllib.error.URLError` (covers HTTPError + DNS/timeout) → clean `SystemExit` with season + source context (carried from the spike fixes).
- **Refuse to write an empty/partial snapshot:** if a source parses to 0 rows, raise rather than persist an empty partition (an empty `asof` snapshot would silently corrupt the "latest" read). A source returning 0 rows is a hard error for that run.
- `id_map` crosswalk: dedup `id_map` on the join id before merging (it has duplicate rows) so a duplicate mapping can't multiply rows.
- Validate with `ExternalProjectionSchema.validate(df)` (reassigned) at the module boundary.

---

## 8. Testing

- **Unit (pure, fixture-driven):** `parse_espn_players` / `parse_sleeper_adp` (reuse the spike's fixtures); `_attach_gsis_id` — matched veteran → real id, unmatched → deterministic placeholder + `is_placeholder_gsis=True`, same input → same placeholder (determinism), placeholder matches `GSIS_ID_PATTERN`; `_to_canonical` → schema-valid rows incl. null stat line for Sleeper; the store `asof` round-trip (`write_partition(asof=…)` then `read_partition(asof=…)` and all-asof concat and `read_latest_partition`).
- **Integration seam:** `pytest -k "ingest or store or schemas"` green (schema/store change).
- **Network (manual, verified by running):** `python -m projections.ingest.external_projections --season 2026` writes a real snapshot; spot-check a known veteran (real `gsis_id`) and a known 2026 rookie (placeholder `gsis_id`, `is_placeholder_gsis=True`).

---

## 9. Definition of done

`refresh_external_projections(data_root, season=2026)` writes `data/raw/external_projections/season=2026/asof=<today>/part.parquet`, validated, containing ESPN + Sleeper rows with real `gsis_id`s for veterans and placeholders for 2026 rookies; a second run with a later `asof` adds a second snapshot without clobbering the first; `read_latest_partition` returns only the newest; the `id_map` float-id fix has a regression test; pure transforms unit-tested; mypy strict + ruff + ruff format clean; `pytest -k "ingest or store or schemas"` green. PM + TODO updated (TODO #38 advanced; this slice closed; consensus/scraping/draft-hub remain).

---

## 10. Risks

1. **ESPN stat-id drift.** ESPN's numeric stat ids could change across seasons. *Mitigation:* the decode is pinned + verified against real 2024 and 2026 data; the empty-snapshot guard (§7) turns a silent decode failure into a loud one (0 stat-bearing rows).
2. **Placeholder collision.** A 7-digit hash mod could collide for two unmatched players. *Mitigation:* 10M space vs ~hundreds of unmatched players → negligible; the synthetic ids are transient (reconcile to real within weeks) and flagged; if ever a concern, widen to the hash's full range within the 7-digit field or add a collision check in `_attach_gsis_id`.
3. **`asof` partition correctness across snapshots.** Concatenated multi-`asof` reads must keep `asof` in-row (they do — it's a schema column) so rows from different pulls stay distinguishable. *Mitigation:* covered by the round-trip test.
4. **Same-day re-run semantics.** Overwriting the same `asof` is intended (idempotent per day); a user expecting per-run history within a day would be surprised. *Mitigation:* documented; daily granularity is the right cadence for a months-long preseason ramp.
