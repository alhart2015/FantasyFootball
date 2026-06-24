# DFS Projection Edge Study — Layer 1 status (2026-06-23)

**Verdict: DEFERRED** (not ADOPT / STOP / INCONCLUSIVE). The kill-test harness is built, unit-tested, and the Sleeper-ingest half is validated on live data; the model-projection half is blocked on stale feature data (TODO #40), so no real verdict was computed.

## What this slice delivers

The DFS Engine's first slice — a retrospective kill-test that asks whether our home-grown weekly model (alone or blended with Sleeper) beats Sleeper's own weekly projections under DraftKings base scoring, with a single pre-registered statistical gate. Spec: `docs/superpowers/specs/2026-06-23-dfs-projection-edge-study-design.md`. Plan: `docs/superpowers/plans/2026-06-23-dfs-projection-edge-study.md`.

**Shipped + unit-tested (12 tasks, each TDD + dual review):**
- `season_calendar` (era-aware regular-season week, TODO #41 hoist).
- `normalize_join_id` lifted into `ingest/identity.py` (the `.0`-stripping `sleeper_id` join, TODO #38).
- `Ruleset.draftkings()` base preset + allowlist; standalone `dk_actuals_bonus` helper.
- `ExternalProjectionWeeklySchema` + Sleeper **weekly** projection ingest.
- `dfs/actuals.py` (era-aware DK-base weekly actuals, base + bonus columns).
- `dfs/projections.py` (walk-forward emitter over the production model per position + a **non-vacuous** trajectory/vegas leakage guard — verified to fail under a deliberately leaky implementation).
- `dfs/blend.py` (stat-line-space blend + `sleeper_weekly_points`).
- `dfs/edge_study.py` (comparable universe, disagreement head-to-head, **player-season clustered bootstrap** + by-week robustness, ranking skill, inclusion-disagreement, coverage, pre-registered single-primary gate, ADOPT/STOP/INCONCLUSIVE verdict + anti-masking + bonus sensitivity).
- `dfs/run.py` + `scripts/dfs_edge_study.py` (orchestrator + CLI: `ingest-sleeper` / `calibrate` / `study`).

## What was validated on real data (2026-06-23)

A live probe against the real 2023 data:
- **Sleeper weekly ingest works end-to-end.** `refresh_sleeper_weekly(season=2023, week=5)` wrote 303 rows in ~0.5s; **placeholder-gsis fraction = 0.00** (every Sleeper id resolved to a real `gsis_id` through the float-stringified `id_map.sleeper_id` join — the TODO #38 seam holds), and `sleeper_weekly_points` scored them under DK base.

## Why the verdict is deferred (the blocker)

`emit_weekly_projections` reads the stored feature partitions via `read_features`, which validates against the **current** `WrFeaturesSchema`. The on-disk partitions (`data/features/{qb,rb,wr,te}/season=2018..2024`) are **stale**: they predate PR #51's Vegas team-context features and are missing `preseason_implied_team_total` (and `preseason_spread` / `season_avg_*`). Validation fails:

```
SchemaError: column 'preseason_implied_team_total' not in dataframe.
```

This is **TODO #40** (the stale-snapshot / Vegas-feature-parity follow-up), not a defect in this slice. The features cannot be rebuilt in this environment because the raw inputs (`pbp`, `ngs`, `snap_counts`, `depth_charts`) are absent from `data/raw/` (only `weekly_stats` + `schedules` remain). Rebuilding requires re-ingesting those sources from `nflreadpy`.

## To produce the real verdict

1. Re-ingest the raw feature sources for 2018–2024 (`scripts/refresh_features.py` + the underlying `pbp`/`ngs`/`snap_counts`/`depth_charts` ingest) and rebuild features (resolves TODO #40).
2. Ingest Sleeper weekly data: `python scripts/dfs_edge_study.py ingest-sleeper --seasons 2021-2024 --data-root C:/Users/alden/FantasyFootball/data`
3. Calibrate the pre-registered constants from a prior season and confirm/adjust `src/projections/dfs/config.py` **before** running the study: `python scripts/dfs_edge_study.py calibrate --prior-season 2020 --data-root C:/Users/alden/FantasyFootball/data`
4. Run the study: `python scripts/dfs_edge_study.py study --seasons 2021-2024 --data-root C:/Users/alden/FantasyFootball/data --out reports/dfs_projection_edge_<date>.md`

(Note: on Windows pass `--data-root` a drive-letter path like `C:/Users/...`, not a Git-Bash mount path like `/c/Users/...`.)

## Limitations (carry into the verdict when it runs)

- **Sleeper-alone is a softer proxy than the true DFS field** (which aggregates many sharp sources). Beating Sleeper is necessary but not sufficient for real DFS profitability — it is a valid *kill-test*, not a profit proof (spec §4.3 / §6.1).
- **Bonuses are excluded from the projection comparison** (a point projection cannot express `E[bonus]`); this is conservative (may understate our edge), and the report's bonus-sensitivity check scores actuals with full bonuses to test whether the verdict flips (spec §6.2).
- Skill positions only (QB/RB/WR/TE). Pre-registered constants (`δ`, usage floor, anti-masking margin, `N_min`) in `dfs/config.py` are placeholders pending the calibration step above.

## Disposition of TODO #39

The fair-weekly-benchmark question (TODO #39) is **superseded** by this slice's method (re-pointed at Sleeper + DraftKings scoring with a pre-registered gate). The method is built and unit-tested; #39 stays open only until the real verdict above is computed.
