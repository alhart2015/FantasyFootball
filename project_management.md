# Project Management

Running log of project status, decisions, and next steps. Append new entries at the top; keep the bottom as the long-tail backlog. Specs live in `docs/superpowers/specs/`, plans in `docs/superpowers/plans/`, single-task TODOs in `TODO.md`.

---

## Current status (as of 2026-04-24)

**Projections Core — Plan 1 (Foundations) merged to `main` at commit `8f02a6c`.**

89 passing tests, mypy strict clean, ruff clean. In place:

- Project bootstrap (`pyproject.toml`, mypy strict, ruff with E/F/W/I/B/UP/N/RUF)
- `src/projections/schemas.py` — single source of truth for canonical types: `Position`/`Team`/`RosterSlot`/`DistributionFamily`/`Stat` enums, `GsisId`/`EspnId`/`SleeperId`/`PfrId` NewTypes, `Ruleset` pydantic model with ESPN_PPR/ESPN_HALF/STANDARD presets, pandera schemas (`WeeklyStatsSchema`, `IdMapSchema`, `ProjectionWeeklySchema`)
- `src/projections/distributions/` — `Distribution` Protocol + `ParametricNormal` + `ParametricGamma`
- `src/projections/scoring/` — `StatLine` + `score()` + `score_distribution()` (Monte Carlo) + `SampledDistribution`
- `src/projections/store/` — partitioned parquet read/write (idempotent) + DuckDB view layer
- `src/projections/ingest/` — `build_id_map()`, `refresh_weekly_stats()` with team-code normalization and position filtering, manifest with SHA-256 checksum and idempotent upsert

---

## Next action

**Recommended: TODO #0 first, then Plan 2.**

The natural next implementation step is **Plan 2 — Ingest expansion + per-position features** (schedules, snap_counts, depth_charts, NGS ingest paths plus the per-position feature builders the spec called out). It follows the foundations patterns and should move quickly.

Before that, a 30-min detour for **TODO #0** (pre-commit hooks + `CONTRIBUTING.md`) is worth doing — it locks the conventions in CI before Plan 2 expands the codebase, so any drift is caught automatically rather than relying on per-PR discipline.

### Three options considered, in order of recommendation:

1. **TODO #0 (~30 min)** — pre-commit hooks (mypy + ruff + pytest) and CONTRIBUTING.md. Locks conventions in CI before Plan 2 grows the codebase. Picked.
2. **Plan 2 (large)** — Ingest expansion + features. Biggest momentum gain, follows foundations patterns. Next after TODO #0.
3. **Drive-by minor cleanups (~15 min)** — `_PYARROW_STR` consolidation into `schemas.py`, programmatic `_INTEGER_STATS` from `StatLine` annotations, drop helpers from ingest `__all__`. Not blocking; can fold into Plan 2 or land separately.

---

## Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-24 | Decompose project into 4 sub-projects (Projections Core, Draft Hub, Mid-season Manager, DFS Engine) | Each subsystem has different consumer logic; shared dependency is a probabilistic projection engine. Keeps any single design doc executable. |
| 2026-04-24 | Build Projections Core first | Earliest dependency for everything else. |
| 2026-04-24 | `nfl_data_py` as primary data source | Free, comprehensive, modern; Python-native. Paid feeds (PFF, FantasyPros API) deferred until we've validated need. |
| 2026-04-24 | Full per-player distributions (option C from brainstorming), not point estimates | Subsumes point estimates for free; required for DFS GPP work later. Joint correlations (option D) deferred to TODO #1 — schema designed so D is additive. |
| 2026-04-24 | Weekly model as foundation; season aggregates as derived layer | Weekly is where play-by-play signal lives; season is Monte Carlo aggregation with bye + availability. |
| 2026-04-24 | A → C → D modeling roadmap | Baseline regression first (Model A) to establish data pipeline + backtest harness; gradient boosted (Model C) only if it beats baseline; ensemble (Model D) reserved for last. |
| 2026-04-24 | Strong typing posture: pandera schemas at module boundaries, pydantic models for configs/records, NewType per ID flavor, mypy strict, enums for every reused string-keyed concept | User had prior pain with stringly-typed/dict-laden code. Catch errors at boundaries, not three modules deep. |
| 2026-04-24 | Parquet + DuckDB storage | Friendly to free-tier hosting (Streamlit Community Cloud, HF Spaces, DuckDB-WASM in browser). |
| 2026-04-24 | Subagent-driven execution for foundations plan | Faster iteration, fresh context per task, two-stage review (spec then code quality) at higher-risk tasks. |

---

## Backlog (longer-term)

Roughly in order. Each is its own brainstorm → spec → plan cycle.

### Projections Core (remaining)

- **Plan 2** — Ingest expansion (schedules, snap_counts, depth_charts, NGS) + per-position feature builders.
- **Plan 3** — Model A baseline (per-position regressions) + season aggregation (Monte Carlo with bye + availability) + first-class backtest harness.
- **Plan 4** — Public Python API + CLI verbs (`refresh`, `project`, `backtest`, `query`) + free-tier web hosting setup (likely Streamlit on Community Cloud).
- **Plan 5** — Model C (LightGBM with quantile regression). Adopt only if it beats Model A on the backtest harness.

### Subsequent sub-projects

- **Draft Hub** — pre-draft rankings, ADP, tier breaks, VORP, mock-draft sim, live draft assistant (consumes Projections Core + ESPN league API).
- **Mid-season Manager** — weekly start/sit, waiver-wire valuator, trade analyzer, schedule strength.
- **DFS Engine** — slate projections, ownership, salary-constrained lineup optimizer, multi-lineup portfolio. Triggers TODO #1 (joint correlations) work.

### Cross-cutting

- **TODO #0** — pre-commit hooks + `CONTRIBUTING.md` (conventions in CI). Recommended next.
- **TODO #1** — option D exploration: joint-correlation projections (covariance / scenario sim / factor / copula). Decide before DFS Engine.
- **`score_distribution` vectorization** — TODO marker in code; needed before backtest scale (~85M Pydantic instantiations otherwise).
- Minor cleanups from foundations review: `_PYARROW_STR` to `schemas.py`, programmatic `_INTEGER_STATS`, drop ingest helpers from `__all__`.
- ESPN league API integration (year-long league sync). Belongs in Draft Hub / Mid-season Manager sub-projects.
- Pyarrow strings everywhere story: pandera 0.31 enforces `string[pyarrow]` for `Series[str]`. Consider whether a future schema or storage shift makes this implicit rather than per-module.
