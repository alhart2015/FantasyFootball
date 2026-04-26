# TODO

Running project management list. Add items as they come up; remove or check off when resolved.

## Open

### 1. Explore option D: joint-correlation projections

**Context.** During Projections Core brainstorming we picked option C (full per-player distributions, marginal only). Option D would extend C to model how player outcomes *co-move* — same-game stacks, opponent dependencies, game-script effects. We deferred D because it adds storage and modeling complexity we may not need until DFS tournament work; we want C's schema to make D an additive upgrade rather than a rewrite.

**Why it matters.**
- DFS GPPs (top-heavy tournaments) live and die on correlated ceilings; an uncorrelated "stack" model dramatically underestimates QB+WR1 joint upside.
- Cash-game DFS and start/sit decisions can survive on marginal distributions alone.
- Season-long draft and waiver tools mostly want means and ranks; correlations are nice-to-have, not load-bearing.

**Questions to answer when we explore.**
- *Scope:* which correlations actually move the needle? Likely candidates, in priority order:
  - Same-team QB ↔ pass-catchers (typically ρ ≈ 0.4–0.6 for QB↔WR1)
  - Same-game opposing players (shootouts lift everyone)
  - RB ↔ team defense (negative; if you allowed the opposing RB to score, your D suffers)
  - Weather and pace shared across a game
- *Modeling approaches:*
  - Empirical covariance matrix from historical weekly fantasy points (simple, but noisy and assumes stationarity)
  - Scenario / Monte Carlo from simulated game states (richer, much more code — could lean on `nflfastR` win-probability and play-type models)
  - Factor model: shared "game environment" latent variable (pace, total) plus player-specific noise (compromise)
  - Gaussian copula on marginals from C (clean separation: marginals stay as in C, dependence lives in the copula)
- *Storage:* covariance matrices per slate/week are O(N²); scenario tables are O(N · S) for S draws. Need to pick one before DFS optimizer work.
- *Optimizer interface:* most ILP optimizers accept point projections + ownership; correlated upside requires either a sim-based optimizer or a stacking-rule heuristic on top of ILP. Decide which path.
- *Validation:* how do we measure that correlated projections beat uncorrelated? Backtest against historical DK/FD GPP results — compare uncorrelated lineups vs correlated lineups by realized payout percentile, not just RMSE.

**Inputs / references to gather.**
- `nfl_data_py` play-by-play features useful for game-script modeling (EPA, pace, success rate, win prob).
- Historical DK/FD slate salaries + ownership (for backtest target).
- Existing OSS work: `pydfs-lineup-optimizer`, `pulp`/`cvxpy`-based optimizers, any public correlation matrices.
- Blog/academic refs: RotoGrinders/Fantasy Labs on stacking; any published papers on DFS lineup construction under correlation.

**Definition of done for this exploration.**
A short written recommendation: pick one modeling approach (covariance / scenario sim / factor / copula), one storage format, and a concrete API addition to the C-era projections schema. Include a backtest plan so we know whether D is actually paying off before we commit to building it.

### 2. Plan 2b — remaining position feature builders

**Status:** QB/RB/TE complete in Plan 2b (merged). K and DST split out into TODO #10.

### 3. Play-by-play ingest (`nfl_data_py.import_pbp_data`)

Required for true opponent-adjusted EPA features. Defer until Plan 3 backtest reveals whether the `opp_allowed_fppg_l4` proxy is good enough. If not, ingest PBP and add EPA-derived opponent features in a focused plan.

### 4. Feature parquet storage — closed in Plan 3c

Closed 2026-04-26. `data/features/{position}/season=YYYY/week=WW/part.parquet`
populated by `scripts/refresh_features.py`; read by `src/projections/features/cache.py`
and consumed by the backtest harness. Manual invalidation only — see TODO #21
below for code-hash auto-invalidation.

### 5. NGS missing-data forward-fill policy

v1 leaves NaN. Revisit after a notebook investigation against a recent season quantifying how often qualifying-threshold misses happen and whether forward-fill changes feature distributions materially.

### 6. Opening / week-of Vegas line source

`import_schedules` returns *closing* lines. Closing is fine for backtest. Only worth pursuing if Plan 5 ever projects pre-week selections (e.g., DFS workflow uses lines that change through the week).

### 7. Depth chart slot-label parser refinement

v1 extracts the trailing digit from labels like `WR1`, falling back to `1` for unrankable labels (`LWR`/`RWR`/`SWR`) with a warning. If Plan 3 model fitting shows `depth_rank` is noisy or wrong, build a richer parser using alignment + rank.

### 9. WR feature builder edge cases for production data

Issues flagged during Task 15 / final code review that don't manifest on the synthetic fixtures but could surface in real `nfl_data_py` data:

a) `is_home` and `roof_dome` are non-nullable in `WrFeaturesSchema` but the schedule join is a left-merge — if a depth-chart team has no schedule row in the target week (bye week, missing future game), validation fails. Fix: filter rostered teams to only those with schedule rows, OR mark the columns nullable. Revisit when Plan 3 wires real data.

b) `IdMapSchema.pfr_id` is not marked `unique=True`. The snap_counts ingest does an inner-join on pfr_id; duplicate pfr_ids in id_map would multiply rows. Add `unique=True` to `pfr_id` (and to `espn_id`/`sleeper_id` for symmetry) as defense-in-depth, or add `.drop_duplicates(subset=["pfr_id"])` in the snap_counts join helper.

c) `_trailing_4_share_per_team` in `features/wr.py` groups by `(gsis_id, team)`, which produces two rows for any player traded mid-season. The downstream merge in `build_wr_features` joins on `gsis_id` only, so duplicates would propagate into the output. Not exercised by the synthetic fixtures (no traded players). Fix: filter `last4_player` to only the player's *current* team before computing the share, or merge on `(gsis_id, team)` so only the matching team's row survives.

### 10. Plan 2c — K and DST feature builders

Both positions need data we don't currently ingest:

- **K**: spec calls for "recent FG distance distribution" and "opp redzone TD allowed %." Neither is in `WeeklyStatsSchema`. Need to ingest a new source covering FG attempts by distance and accuracy by range. `nfl_data_py.import_weekly_pfr_data` may have this — verify before designing.
- **DST**: team-level not player-level. Schema's primary key is `Team`, not `GsisId` — fundamentally different from the per-player pattern Plan 2a/2b established. Intended features (opp pass-block win rate, sack rate allowed, turnover-worthy throw rate) all need play-by-play (TODO #3).

Decision before brainstorming Plan 2c: do we ingest the missing data first (extending the ingest layer), or build degraded v0 K/DST features from `implied_team_total` alone? The latter is fast but creates a future rewrite; the former takes longer but yields the right shape.

Plan 3 (Model A baseline) doesn't depend on K/DST, so this can run in parallel.

### 11. Bound `percent_attempts_gte_eight_defenders_std`

Surfaced in PR #4 review (latent issue #5). The field on `RbFeaturesSchema` currently has only `nullable=True` — no `ge`/`le`. NGS reports the metric as a 0–100 percentage, so a `ge=0, le=100` bound would catch a unit-mismatch upstream (e.g., if a future ingest change accidentally emits the same value as a 0–1 fraction). One-line fix; defer until any other RB schema work to keep the diff scoped.

### 12. Lift `rushing_qb` / `passing_down_back` to consumption time

Surfaced in PR #4 review (latent issue #6). Both are thresholded booleans baked into the persisted feature schema (`rushing_qb` = `rushing_attempts_per_game_l4 >= 5.0`, `passing_down_back` = `targets_per_game_l4 >= 4.0`). Decision logged in the Plan 2b plan as "rough heuristics from feel," but the threshold is now fixed at the producer side, so a downstream consumer (a model wanting a different cut, or a categorical instead of a boolean) has to recompute from the underlying `*_l4` column anyway. Two options:

- Drop the boolean from the schema and compute at use time (cleanest; consumer holds the policy).
- Keep but document explicitly that this threshold is the canonical league-wide convention and shouldn't be re-derived elsewhere.

Revisit before Plan 3 model fitting — if the model never uses these booleans, just remove them.

### 13. Per-row seed derivation in BaselineModel.predict_distribution

Surfaced in PR for Plan 3a (Task 9 review). `BaselineModel.predict_distribution` calls `score_distribution(..., seed=42)` for every row, so the underlying Monte Carlo sample arrays are correlated across rows. Per-row stats (mean/p10/p50/p90) are unaffected, but downstream callers that combine multiple rows' samples (DFS lineup variance, roster-total simulation, joint correlations once TODO #1 lands) MUST NOT assume cross-row independence.

Fix: derive per-row seeds from `(gsis_id, season, week, ruleset.name)` (e.g., `hash(...) & 0xFFFFFFFF`) so each row gets an independent draw. Cheap, no perf cost. Defer until Plan 3c's backtest harness or DFS work needs cross-row independence.

### 14. ProjectionWeeklySchema params blob carries summary, not samples

Surfaced in PR for Plan 3a (Task 9 review). `BaselineModel.predict_distribution` sets `family="SAMPLED"` but the `params` bytes encode only `{"samples_summary": {"n", "mean"}}`, not the full sample array. The persisted distributional info lives in `mean`/`p10`/`p50`/`p90`; `params` is a breadcrumb.

Two fix options:
- (a) Add `DistributionFamily.SAMPLED_SUMMARY` enum value; rename the family on these rows. Backward-compatible.
- (b) Pack the full `points.samples` array (np.float64 × 10000 = 80KB per row, msgpack-compressible). Material storage cost; right answer once parquet partitioning is stable, but premature for v1.

Decide before Plan 3c's backtest output consumes ProjectionWeeklySchema rows.

### 16. Real-data drifts not caught by synthetic-fixture tests

Surfaced during Plan 3a Tasks 14-17. The synthetic fixtures used by 2a/2b/3a's CI tests don't exercise the real `nfl_data_py` API surface. Eight ingest/feature drifts had to be patched live during Plan 3a's first real-data pull:

1. `weekly_stats`: `fumbles_lost` had to be derived from three source-specific columns (no aggregated column upstream).
2. `weekly_stats`/`depth_charts`/`ngs`/`snap_counts`: int32 vs int64 dtype mismatch on `season`/`week`.
3. `depth_charts`/`ngs`/`snap_counts`: NaN season/week rows had to be filtered before int coercion.
4. `ngs`: pro-bowl/all-star weeks (>22) and season-summary rows (week=0) had to be filtered.
5. `id_map`: 16+ pro-football-reference 3-letter team aliases (GBP, KAN, NWE, NOR, SDG, TAM, etc.).
6. `id_map`: "FA"/"FA*" (free agent) team codes had to be handled as None.
7. `id_map`: malformed legacy gsis_ids (PFR-style strings) had to be filtered.
8. `wr.py`: bye-week WRs without schedule rows; duplicate depth-chart entries; negative trailing-mean yardage; share calc going negative.

The opt-in `pytest -m network --run-network` smokes (`tests/test_ingest/test_api_drift.py`, formerly TODO #8 — closed) now guard against the same class of column-rename / column-removal drift after a `nfl_data_py` version bump. They do NOT replace this drift list as historical context, and they will NOT catch every real-data edge case (some — like the `id_map` malformed-legacy-gsis-id rows — are data-quality issues per row, not column-level drift), so keep this entry as a record of what the synthetic fixtures missed and audit it after each `nfl_data_py` upgrade.

**Additional Plan 3b real-data drifts (Phase 6):**

9. `WrFeaturesSchema` / `QbFeaturesSchema` / `RbFeaturesSchema` / `TeFeaturesSchema`: `*_yards_per_game_l4` and `passing_yards_per_game_std` had `ge=0` bounds inconsistent with the underlying weekly_stats schema (which allows negative yards from sacks / TFL / kneels). Trailing-4 means can therefore be negative; bounds dropped in commits `fa864ac` + `e25eb57`.
10. QB/RB/TE feature builders missing the bye-week filter on rostered teams (analogous to WR's TODO #9a). Players on a team with no schedule row in `as_of_week` produced NaN `opponent`/`is_home`/`roof_dome` and failed schema validation. Filter ported from `wr.py` in commit `f79806a`.
11. QB/RB/TE feature builders missing the depth-chart dedupe (analogous to WR's TODO #9c). Players listed under multiple slots or traded mid-week produced duplicate `(gsis_id, season, week)` rows and failed `BaselineModel.fit`'s one-to-one merge. Dedupe ported from `wr.py` in commit `54b6d95`.

The TODO #8 opt-in network smokes confirmed no upstream column-rename drift in this run; the new entries are all schema-bound or builder-edge-case mismatches between WR's already-hardened path and QB/RB/TE's pre-3b path. The four fixes above bring QB/RB/TE feature builders to parity with WR.

### 18. Add `python -m projections.ingest.refresh` CLI entry point

Surfaced during Plan 3b Phase 6 ingest. `src/projections/ingest/refresh.py`
exports a `refresh()` function but has no `if __name__ == "__main__":` block,
so `python -m projections.ingest.refresh ...` doesn't work and ingest must
be invoked via `python -c "from projections.ingest.refresh import refresh;
refresh(data_root=Path('data'), seasons=range(2018, 2025))"`. A small
argparse `main()` (with `--seasons RANGE` / `--data-root PATH` and a
sensible default for `data_root=Path('data')`) would make the canonical
ingest invocation a one-liner. Defer until next ingest-touching plan
(Plan 4's CLI verbs are the natural home).

### 19. Walk-forward gate non-determinism check

Phase 6 of Plan 3c may surface tiny RMSE jitter on the 2021 cells
where RidgeCV trains on only 3 prior seasons. If empirically observed
on a re-run with unchanged data, add explicit `random_state` propagation
through `BaselineModel.fit` and re-snapshot. Otherwise close.

### 20. Naive-baseline parquet output for trend tracking

Plan 3c writes naive metrics into the in-memory `BacktestRun` and prints
them in `--report` mode but does not persist them. If we ever want to
track "how much value is Model A adding over naive *over time*", persist
naive metrics to a parquet table at `data/backtest/naive_history/...`
keyed by run timestamp. Not load-bearing for v1.

### 21. Feature cache code-hash auto-invalidation

Plan 3c's feature cache is invalidated manually — the user must re-run
`scripts/refresh_features.py` after touching any feature builder.
Auto-invalidation reads the source files for the feature builder (the
same set `BaselineModel.code_hash_files` already tracks) and refuses
to read stale cache. Deferred until manual invalidation produces a
real-world bug.
