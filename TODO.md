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

### 4. Decide feature parquet storage during Plan 3

Gated on backtest performance: if a single training pass takes >~30s recomputing features, add `data/features/{position}/...` storage and a `refresh_features` CLI verb; otherwise stay pure-function.

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

### 17. Retrain WR 2018-2023 artifact after Plan 3b

Plan 3b adds two required fields (`feature_schema`, `code_hash_files`)
to the `BaselineModel` dataclass. The existing artifact at
`models/artifacts/wr-baseline-2018-2023-925f492b.joblib` (gitignored)
becomes unloadable through `BaselineModel.load()` — joblib pickle
reconstruction will raise `TypeError` on the missing required args.

Mitigation: run `python scripts/train_baseline.py wr` (the new
generalized script from Plan 3b Phase 5) once 3b is merged. Closes
this entry.
