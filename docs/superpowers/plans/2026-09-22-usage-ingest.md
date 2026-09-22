# Usage & Availability Ingest — Implementation Plan

Spec: `docs/superpowers/specs/2026-09-22-usage-ingest-design.md`
Branch: `feat/usage-ingest`

Five phases. Each phase touches ≤5 files, ends with the verification gate green, and is committed
before the next begins. Phases 2–4 are independent of each other and both depend on Phase 1.

---

## Phase 1 — Extract the pfr→gsis crosswalk

Phase 4 needs the `pfr_player_id` → `gsis_id` join that currently lives as a private function in
`snap_counts.py`. Extract it first, with `snap_counts`' existing tests as the regression guard, so
Phase 4 adds a caller rather than a copy.

**Files**
1. `src/projections/ingest/identity.py` — add `resolve_gsis_via_id_map(df, data_root, *, pfr_col)`.
   Same inner-join semantics as today; `pfr_col` parameterized because `snap_counts` and
   `pfr_advstats` both use `pfr_player_id` but the caller shouldn't have to assume that.
2. `src/projections/ingest/snap_counts.py` — delete `_resolve_gsis_via_id_map`, import the shared
   one. No behaviour change.
3. `tests/test_ingest_identity.py` — direct tests for the helper: match, no-match-dropped,
   null-pfr-id, missing id_map raises `FileNotFoundError`.

**Gate:** `pytest -v -k "snap_counts or identity"`, then the full gate.

**Done when:** snap_counts tests pass unchanged and the helper has its own coverage.

---

## Phase 2 — Injury report ingest (closes #119)

Cleanest of the three (real `gsis_id`, int season/week). Doing it first proves the end-to-end
shape — schema + module + registry + tests — on the source with the fewest traps.

**Files**
1. `src/projections/schemas.py` — add `InjuryStatus` and `PracticeStatus` enums and
   `InjuryReportSchema`. `report_status`, `practice_status`, and the four injury-description
   columns are `nullable=True`; docstring states that null is meaningful here (on the report, no
   designation) and not a gap.
2. `src/projections/ingest/injuries.py` — new module on the `weekly_stats.py` template.
   `_fetch_raw_injuries` seam; filter `season_type == "REG"`; map verbose practice labels to
   `PracticeStatus` values; `drop_placeholder_gsis_rows`; `normalize_team_code`; validate.
3. `src/projections/ingest/sources.py` — register `injuries`, `needs_games_played=True`,
   `heavy=False`.
4. `tests/test_ingest_injuries.py` — POST filtered; null `report_status` survives as null;
   verbose practice labels → enum; placeholder ids dropped.

**Gate:** `pytest -v -k "ingest or store or schemas"`, then the full gate.

---

## Phase 3 — Expected fantasy points (`ff_opportunity`)

The highest-value source and the one with the most upstream traps. Each trap from the spec gets a
named test.

**Files**
1. `src/projections/schemas.py` — add `FfOpportunitySchema` over the 28 kept columns from the
   spec. Docstring must say that `total_fantasy_points*` is the *upstream model's* scoring, not
   ours, and is diagnostic-only.
2. `src/projections/ingest/ff_opportunity.py` — new module. Order matters and is the whole
   difficulty: drop null `player_id` → coerce `season` (String→int64) and `week` (Float64→int64)
   → rename `posteam`→`team` → `normalize_team_code` → filter to `Position` enum →
   `drop_placeholder_gsis_rows` → select `_KEEP` → validate.
3. `src/projections/ingest/sources.py` — register `ff_opportunity`.
4. `tests/test_ingest_ff_opportunity.py` — one test per spec trap: string season, float week, null
   `player_id`, `DB`/`OL`/`None` positions filtered, `posteam` normalized, week 22 accepted.

**Gate:** `pytest -v -k "ingest or store or schemas"`, then the full gate.

---

## Phase 4 — PFR advanced stats (pass / rush / rec)

Mirrors the `ngs` multi-stat-type pattern: three schemas, one module, one registry entry per type.
`def` is deferred to its own spec (spec → "Follow-ups").

**Files**
1. `src/projections/schemas.py` — `PfrPassingSchema`, `PfrRushingSchema`, `PfrReceivingSchema`.
2. `src/projections/ingest/pfr_advstats.py` — new module with a `PfrStatType` literal and
   `STAT_TYPES` tuple, following `ngs.py`'s structure. Uses the Phase 1 crosswalk.
3. `src/projections/ingest/sources.py` — register `pfr_pass`, `pfr_rush`, `pfr_rec` via a
   comprehension over `STAT_TYPES`, as `ngs_*` already does.
4. `tests/test_ingest_pfr_advstats.py` — unmatched `pfr_player_id` dropped; each stat type
   normalizes and validates; missing id_map raises.

**Gate:** `pytest -v -k "ingest or store or schemas"`, then the full gate.

---

## Phase 5 — Real-data smoke run, docs, PR

Synthetic fixtures cannot catch an upstream shape change; this phase is the only one that touches
the network.

1. Run each new source against 2024 and 2025 for real. Record row counts and any dropped-row
   warnings.
2. Confirm partitions land under `data/raw/<source>/season=YYYY/` and read back through
   `store.read_partition`.
3. `CONTRIBUTING.md` — add the three sources to whatever inventory of ingest sources exists there;
   note the `ff_opportunity` scoring caveat next to the scoring-layer rule.
4. Open the deferred follow-up issues from the spec (`pfr def` → D/ST; feature-builder consumption
   under #121; injury consumption under #119).
5. Open the PR. Body carries the smoke-run row counts, the full verification gate output, and a
   link to the spec. Closes #119.

---

## Risks

- **`ff_opportunity` upstream dtypes are unusually loose** (String season, Float week). If nflverse
  tightens them, our coercion still works — `astype("int64")` on an already-int column is a no-op.
  The reverse (them loosening further) would surface as a pandera failure, which is the desired
  loud failure.
- **`id_map` coverage bounds Phase 4.** Rows whose `pfr_player_id` has no crosswalk entry are
  dropped silently today in `snap_counts`. Phase 1 keeps that behaviour rather than changing it
  mid-refactor, but the Phase 5 smoke run should record the drop rate so we know if it is material.
- **No consumer yet.** This plan lands data that nothing reads. That is deliberate (spec → Scope),
  but it means the real proof of value is the follow-up feature work, not this PR.
