# Pool gsis reconciliation — fix availability-blind projected-H2H evals

**Status:** design
**Branch:** `feat/snake-strategy-search`
**Date:** 2026-06-20

## Problem

While searching for a snake-draft strategy to beat every baseline in projected-H2H
win% (16-team half-PPR, all seats, 6-year mean), the search surfaced a data-integrity
bug that invalidates part of the existing evidence base.

**Every per-season preset VORP pool (`data/vorp_{2021..2026}/*.parquet`) carries 100 %
deterministic `99-` placeholder gsis_ids** — even established veterans (Christian
McCaffrey, Davante Adams). Real gsis are `00-0…`. Consequence: the pools join to
**neither** `weekly_stats` (the injury-availability history) **nor** `id_map`
(team → bye week). Measured on 2024: **0 of 630** pool ids overlap `weekly_stats`.

`build_availability` (`availability.py`) degrades silently on this:
- injury `p` falls back to a **position-average** for every player (2024: 4 distinct
  values instead of ~375; range collapses to ~[0.57, 0.62] instead of [0.40, 0.97]);
- the bye map is **empty** (0 byes) because no player resolves to a team.

The degradation warns (`"no schedules … byes will be empty"` / position-default `p`)
but the warning was filtered out in the bake-off runners, so it went unnoticed.

**Impact — the projected-H2H metric was blind to player-specific availability** in
every snake eval that used these pools (Tests 12 and 13, and any per-season run of the
hero harness). The metric differentiated rosters only by projected points + a
position-level variance/injury constant. This **inverts a headline conclusion**: Tests
12/13 reported "season_value family underperforms at 16-team; the now_or_never family is
best." That ranking is an artifact — the season_value strategies' entire mechanism is a
per-player availability Monte-Carlo, which is pointless when `p` is position-constant.

### Root cause

The preset tables are built by `generate_preset_vorp_tables.py`, which calls
`build_consensus(raw external_projections snapshot)`. Those raw per-season snapshots
assigned placeholder gsis to (nearly) all players and have since been deleted
(untracked/regenerable). The *processed* 2026 consensus partition, by contrast, carries
real gsis for 1741/3042 players — so the placeholder problem is specific to the raw
snapshots the preset generator consumed, not the consensus layer in general. Because the
raw snapshots are gone, the per-season tables cannot be regenerated through the normal
pipeline without re-pulling external data (heavy, network-dependent, and the historical
ESPN/Sleeper→id_map crosswalk may still miss).

## Goal

Make player-specific availability (injury `p` + byes) flow into the projected-H2H metric
for all consumers, correct the affected evidence (Tests 12/13), and ship the strategy that
meets the original goal. **Outcome:** no *fixed* strategy beats every baseline at every seat
(a per-seat Pareto frontier), but the seat-aware router **`seat_aware`** strictly beats every
baseline on the pooled 6-year average win% (CI-separated) while matching the per-seat best —
it is shipped to `STRATEGY_KEYS` + the board. See "Findings".

## Design

### 1. `reconcile_pool_gsis(pool, id_map)` — shared helper

A pure function that relabels placeholder (`99-`) gsis to the real (`00-`) gsis via a
**name+position** match, reusing `ingest.identity.placeholder_name_key` (the canonical
cross-source key) so the match rule agrees with ingest by construction.

- Build `key → gsis` from id_map rows with a real (`00-`) gsis; **drop ambiguous keys**
  (a key mapping to >1 distinct real gsis is left unreconciled).
- For each pool row with a placeholder gsis, take the unique real match **only if it does
  not collide** with another kept pool id (preserve gsis uniqueness — `project_draft`
  indexes the pool by gsis).
- Rows already carrying a real gsis, or with no/ambiguous/colliding match, are unchanged.
- Returns a new frame (no mutation); `gsis_id` stays `pd.StringDtype("pyarrow")`.

Measured recovery (2024 half_16team): **617/630 reconciled, 0 collisions**; availability
`p` goes 4→375 distinct values, byes 0→479.

**Home:** `src/projections/draft/assistant/pool_identity.py` (new small module; depends
only on `ingest.identity` + schemas). Unit-tested.

### 2. Apply it at pool construction

- **`generate_preset_vorp_tables.py`** — reconcile each preset table before writing, so
  future regenerations carry real gsis (defends against the bug recurring if external is
  re-pulled).
- **One-time maintenance** (`scripts/reconcile_vorp_gsis.py`, scratch/ops) — reconcile
  the existing `data/vorp_{2021..2026}/*.parquet` **in place** (the only way to fix them
  without the deleted external snapshots). Idempotent.
- **`build_draft_basis`** (hero-harness path) — **wired** (Test 15 re-ran Test 11 on it).
  Its VORP table lacks `full_name` (the reconcile key), so the function attaches it from the
  consensus frame, then reconciles when `load_inputs` passes an `id_map`; `load_inputs` warns
  loudly if `id_map.parquet` is missing (no silent availability-blind degradation).
- **Not reconciled (acknowledged):** `consensus/refresh.py` writes the *processed* consensus
  partition directly, but the spec's measurement shows that layer already carries real gsis for
  most players (1741/3042 for 2026), so it is a lower-risk source and is intentionally left to
  the source-side ingest fix (Out of scope) rather than the consumer-side guard.

### 3. Correct the evidence

- Re-run the 16-team multi-year bake-off (all 6 baselines, seats 1/8/16, 2021–2026) on
  the reconciled tables. Log as **Test 14** in `reports/draft_strategy_tests.md`.
- Add **correction notes** to Tests 12 and 13: their availability signal was absent
  (placeholder-gsis bug); post-fix, `season_value_timing` / `season_value_var` are the
  top tier and the now_or_never family is the bottom — the opposite of what those tests
  reported.
- Record the practical recommendation: **`season_value_timing`** (pooled #1; wins
  wing/mid) with `season_value_var` strongest at the turn.

## Findings from the strategy search (recorded, not shipped)

The search that surfaced the bug is documented for the record (validated on the disjoint
holdout 2022/24/26, seeds 100–139, n_sims 250):

- **σ, F, λ knobs are a plateau** — can't CI-beat `now_or_never_floored` (the tuned
  config *is* nn_floored).
- A **durability tilt** `score = nn_floored − μ·(1−p_week)·vorp` (μ≈1.0) **beats the
  entire nn-family and raw_vorp at every seat** (+0.03…+0.06, CI-separated) once
  availability is real — but **loses to the sv-family**.
- **`sv_var_timing`** (variance-aware MC + timing; an unwired `risk_aware=True` path of
  `SeasonValueTimingStrategy`) ties `season_value_timing` at the wing and loses at the
  turn — does not dominate.
- **The per-seat Pareto frontier + the router that clears it.** Post-fix the baselines form
  a per-seat frontier — timing helps at the wings (`season_value_timing` best at s1/s8) and
  *hurts* at the turn (`season_value_var` best at s16). A *fixed* blend only interpolates.
  **`SeatAwareStrategy`** routes by the hero's known slot (`season_value_timing` for slots
  ≤ n−2, `season_value_var` for the last two) → strictly beats every baseline on the pooled
  6-year win% (vs `season_value_timing` +0.0038 [+0.0011,+0.0065]; vs `season_value_var`
  +0.0047 [+0.0002,+0.0091]) while matching the per-seat best. **Shipped** (`STRATEGY_KEYS`,
  `_build_strategy`, `build_session_strategy`, board `BOARD_STRATEGIES`; tests in
  `test_assistant_seat_aware.py`). The strict "beat every baseline at *every* seat" bar is
  unmet only because it *ties* the strategy it delegates to at each seat (can't beat itself).

## Testing

- Unit tests for `reconcile_pool_gsis`: placeholder→real on a unique match; ambiguous key
  left unreconciled; collision left unreconciled; already-real unchanged; dtype preserved.
- An availability-flow assertion: after reconciliation a known veteran resolves to a real
  gsis and gets a non-default `p` + a bye week.
- Full gate: `pytest -v`, `mypy src tests`, `ruff check`, `ruff format --check`.

## Out of scope / follow-ups

- Fixing the **external-ingest crosswalk** so raw snapshots carry real gsis at the source
  (needs re-pull; the reconciliation helper is the durable consumer-side guard regardless).
- Re-pulling external for 2021–2025 to regenerate tables from scratch.
- The seat-Pareto strategy problem (a strategy that strictly beats the sv-family at every
  seat) — deferred.
