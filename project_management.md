# Project Management

Running log of project status, decisions, and next steps. Append new entries at the top; keep the bottom as the long-tail backlog. Specs live in `docs/superpowers/specs/`, plans in `docs/superpowers/plans/`, single-task TODOs in `TODO.md`.

---

## Current status (as of 2026-04-24)

**Projections Core — Plan 2a (Ingest expansion + WR feature builder) merged to `main` at commit `7926090`.**

**Predecessors:**
- Plan 1 (Foundations) merged at `8f02a6c`.
- Dev tooling merged via `feat/dev-tooling`.

**Plan 2a delivered:**
- Four new ingest modules (`schedules`, `snap_counts`, `depth_charts`, `ngs` parameterized over passing/rushing/receiving) + 6 new partition tables, all idempotent.
- `WeeklyStatsSchema` extended with `targets`, `receiving_air_yards`, `carries` (foundations-era omission).
- `src/projections/features/` package with shared `_rolling.py`, `_opponent.py` helpers and a fully-tested `build_wr_features` (pure-function, no parquet storage).
- 158 total tests (was 89 baseline; +69 new). 5 leakage tests for the WR builder, one per input source.
- Drive-by cleanups: `_PYARROW_STR` to `schemas.py` (covering `weekly_stats.py` AND `id_map.py`), programmatic `_INTEGER_STATS`, trimmed ingest `__all__`.

---

## Next action

**Recommended: Plan 2b — per-position feature builders for QB / RB / TE / K / DST.**

2a validated the feature-builder pattern (signatures, leakage prevention, schemas, tests) end-to-end on WR. 2b copy-pastes the pattern across the remaining five positions, reusing `_rolling.py` and `_opponent.py` from 2a. Each position gets its own pandera schema (`QbFeaturesSchema`, etc.) following `WrFeaturesSchema` as the template. One PR per position or one bundled — TBD when 2b is brainstormed.

After 2b, Plan 3 (Model A baseline + season aggregation + backtest harness) becomes unblocked.

---

## Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-24 | `nfl_data_py.import_snap_counts` returns `pfr_player_id` not `gsis_id`; ingest joins on id_map | Discovered during fixture-construction (Task 8). Snap_counts ingest now reads id_map.parquet and inner-joins pfr_id → gsis_id; bench/practice players with no id_map match are dropped silently |
| 2026-04-24 | `spread_line` from `nfl_data_py` is positive when home favored (inverts standard sportsbook) | Discovered during code review of Task 15. Empirically verified against import_schedules([2023]). `_build_game_environment` in features/wr.py uses the empirically-correct convention; team-perspective `spread` follows standard "favorite is negative" |
| 2026-04-24 | Split Plan 2 into 2a (ingest expansion + WR feature builder) and 2b (QB / RB / TE / K / DST feature builders) | Validate the feature-builder pattern end-to-end on one position before copy-pasting across five files; isolate ingest (mechanical) from features (greenfield design) |
| 2026-04-24 | WR is the first end-to-end position | Exercises every new ingest source (snap_counts, depth_charts, NGS receiving) in one builder; surfaces design issues before propagating to other positions |
| 2026-04-24 | Feature builders are pure functions in 2a — no parquet storage | Output is small (~1.8K rows/season for WR) and computes in milliseconds; defer caching until backtest performance demands it (Plan 3+) |
| 2026-04-24 | Ingest all three NGS stat types (passing, rushing, receiving) in 2a, even though only NGS receiving is consumed by WR | The hard part of NGS ingest is the snapshot/partition decision; make it once across all three rather than three times |
| 2026-04-24 | Opponent strength via `opp_allowed_fppg_l4` proxy in 2a, not play-by-play EPA | True EPA needs play-by-play ingest (separate concern, deferred); the FPPG-allowed proxy is sufficient for v1 baseline |
| 2026-04-24 | Shared `_rolling.py` and `_opponent.py` helpers built and tested in 2a | Pin helper API on the first builder so 2b's five other builders consume a stable contract |
| 2026-04-24 | Schedule ingest captures Vegas lines (spread, total, moneyline) | "Implied team total" is a load-bearing feature for every offensive position |
| 2026-04-24 | Drive-by cleanups (`_PYARROW_STR` to `schemas.py`, programmatic `_INTEGER_STATS`, ingest `__all__`) folded into 2a | We're touching every ingest module anyway; cheaper to clean up once than across two PRs |
| 2026-04-24 | Extend `WeeklyStatsSchema` with `targets`, `receiving_air_yards`, `carries` | Discovered during plan-writing: WR feature builder needs these source columns and the foundations-era schema didn't include them. All three are present in raw `nfl_data_py.import_weekly_data` output |
| 2026-04-24 | Test fixtures are synthetic in-memory `pd.DataFrame`s, not real-data parquet snapshots | Matches existing convention from foundations (`fake_weekly_df` etc.); simpler maintenance; `nfl_data_py` API drift is handled separately by opt-in network smoke tests (TODO #8) |
| 2026-04-24 | Decompose project into 4 sub-projects (Projections Core, Draft Hub, Mid-season Manager, DFS Engine) | Each subsystem has different consumer logic; shared dependency is a probabilistic projection engine. Keeps any single design doc executable. |
| 2026-04-24 | Build Projections Core first | Earliest dependency for everything else. |
| 2026-04-24 | `nfl_data_py` as primary data source | Free, comprehensive, modern; Python-native. Paid feeds (PFF, FantasyPros API) deferred until we've validated need. |
| 2026-04-24 | Full per-player distributions (option C from brainstorming), not point estimates | Subsumes point estimates for free; required for DFS GPP work later. Joint correlations (option D) deferred to TODO #1 — schema designed so D is additive. |
| 2026-04-24 | Weekly model as foundation; season aggregates as derived layer | Weekly is where play-by-play signal lives; season is Monte Carlo aggregation with bye + availability. |
| 2026-04-24 | A → C → D modeling roadmap | Baseline regression first (Model A) to establish data pipeline + backtest harness; gradient boosted (Model C) only if it beats baseline; ensemble (Model D) reserved for last. |
| 2026-04-24 | Strong typing posture: pandera schemas at module boundaries, pydantic models for configs/records, NewType per ID flavor, mypy strict, enums for every reused string-keyed concept | User had prior pain with stringly-typed/dict-laden code. Catch errors at boundaries, not three modules deep. |
| 2026-04-24 | Parquet + DuckDB storage | Friendly to free-tier hosting (Streamlit Community Cloud, HF Spaces, DuckDB-WASM in browser). |
| 2026-04-24 | Subagent-driven execution for foundations plan | Faster iteration, fresh context per task, two-stage review (spec then code quality) at higher-risk tasks. |
| 2026-04-24 | Pre-commit hooks (ruff lint+format, mypy, housekeeping); no GitHub Actions CI; pytest manual before PR | Catches the regressions that matter at commit time without slowing commits with full pytest. CI deferred indefinitely per user direction. |
| 2026-04-24 | No direct commits to `main` — specs, plans, and implementation all on feature branch via PR | User correction after I committed a spec to main. Conventions encoded in CONTRIBUTING.md and CLAUDE.md. |
| 2026-04-24 | `CLAUDE.md` trimmed; `CONTRIBUTING.md` is the deep contributor doc | CLAUDE.md auto-loads into Claude's context every interaction; every line costs context budget. Detail moves to CONTRIBUTING.md. |

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

- **TODO #1** — option D exploration: joint-correlation projections (covariance / scenario sim / factor / copula). Decide before DFS Engine.
- **`score_distribution` vectorization** — TODO marker in code; needed before backtest scale (~85M Pydantic instantiations otherwise).
- Minor cleanups from foundations review: `_PYARROW_STR` to `schemas.py`, programmatic `_INTEGER_STATS`, drop ingest helpers from `__all__`.
- ESPN league API integration (year-long league sync). Belongs in Draft Hub / Mid-season Manager sub-projects.
- Pyarrow strings everywhere story: pandera 0.31 enforces `string[pyarrow]` for `Series[str]`. Consider whether a future schema or storage shift makes this implicit rather than per-module.
