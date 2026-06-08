# Project Management

Running log of project status, decisions, and next steps. Append new entries at the top; keep the bottom as the long-tail backlog. Specs live in `docs/superpowers/specs/`, plans in `docs/superpowers/plans/`, single-task TODOs in `TODO.md`.

---

## Test suite 10× speedup — 28 min → 2m48s (2026-06-08, on branch `chore/test-suite-speedup`)

**Status:** Shipped. The default `pytest` run went from **28 min → 2m48s** on this 32-core box, same pass set (1220 passed; the 15 pre-existing `main` failures unchanged — see TODO #40). Three levers:

1. **Parallelism (the big one):** added `pytest-xdist` + `-n auto` to `addopts`. The suite is fully xdist-safe (verified: parallel run produces the same failures, no new ones). 28 min → ~9 min alone. `-n0` disables it for single-test debugging / live output.
2. **Test-only lightgbm `n_estimators` cap:** the production default `LGBM_DEFAULTS["n_estimators"]=2000` made each unit model-fit take 30–90s. A session-autouse fixture in `tests/conftest.py` caps it to 30 for the test run (single funnel — base/tuned/nb/ensemble all start from `LGBM_DEFAULTS`; tuned JSON doesn't set it; the only hyperparam assertion is `_best_iters` membership, unaffected). `PYTEST_FULL_LIGHTGBM=1` restores the real config. Cut the lightgbm-family tests from ~70–90s to a few seconds each.
3. **Ensemble smoke fixtures shrunk:** the remaining floor was the ensemble tests' `scipy.minimize_scalar` weight-fit evaluating the slow `MixtureDistribution.quantile()` over a 1360-row fixture. Shrank `test_ensemble_model_smoke` + `test_ensemble_decomposed` fixtures to 4 seasons × 6 weeks × 12 players (288 rows); each ensemble test dropped 91s → ~17s, still passing.

**Not done (diminishing returns):** `test_ensemble_weight_fit`'s n=500/200 optimum-recovery fixtures are correctness-sensitive; left as-is. Further fixture surgery risks degenerate fits for little wall-clock gain at 2m48s.

**Also fixed here:** 15 tests were broken on `main` itself (confirmed on a clean `origin/main` checkout) — PR #51 half-completed its Vegas integration: it added the 4 Vegas cols to `Qb/WrFeaturesSchema` but left `_WR/_QB_FEATURE_COLUMNS` and two hardcoded WR fixtures inconsistent. Brought the baseline feature lists to schema parity + emitted the cols in the `test_decomposed_baseline` / `test_tune_lightgbm` fixtures → **full suite now green** (1235 passed). The one residual (the `--run-backtest` snapshot is now stale for the changed baseline WR/QB features) is TODO #40.

---

## External Projection Benchmark Spike — verdict **our model cannot do preseason; pivot to external for draft** (2026-06-08, on branch `feat/external-projection-benchmark`)

**Status:** Spike concluded. The planned preseason ESPN-vs-ours RMSE benchmark was **not run** — it is invalid on our side and would falsely flatter our model. The verdict stands on architecture, not a metric. Spec `docs/superpowers/specs/2026-06-08-external-projection-benchmark-design.md`, plan `docs/superpowers/plans/2026-06-08-external-projection-benchmark.md`, verdict `reports/external_projection_benchmark_2024.md`.

**Strategic context.** First time the home-grown projection model was held up against freely available projections. Motivation: ~2 months of recent work (PRs ~20–35) were feature-family probes measuring 0.004–0.04 fpts/week effects (below decision-relevant magnitude) while zero user-facing tools shipped and the model had never been benchmarked against free sources. Question: keep grinding the model, or use free consensus and spend effort on tools (draft, start/sit, DFS)?

**Disqualifying finding (the spike's real result).** `scripts/project_season.py` is **not a preseason projection** and our model **cannot produce one**. It is a weekly, in-season model: per-week features read trailing windows over the current season, and only players active in a given week get a projection. So its aggregated season totals secretly use the 2024 outcomes we wanted to predict. Smoking gun (2024, PPR): **Christian McCaffrey — our model 63.4 pts over n_weeks=4** (it tracked his early-season injury) **vs ESPN preseason 335.5 vs actual 47.8.** A preseason forecast cannot know an injury in advance; ESPN's 335 is the honest healthy-season forecast. Lamar 419-vs-430 actual is the same effect inverted (tracking an MVP season, not forecasting it). Benchmarking an in-season-informed projection against an honest preseason one is meaningless and biased toward us.

**Decision (user-confirmed).** **For draft (priority #1): pivot to external sources — our model is not in the running, not because it loses but because it structurally cannot play.** ESPN (preseason stat lines + ADP + draft ranks, no auth) and Sleeper (ADP) are genuine free preseason sources, verified pullable for historical seasons. There is nothing on our side to benchmark for draft.

**Reusable assets built (kept):**
- `scripts/pull_external_projections.py` — working ESPN + Sleeper preseason puller (no auth). ESPN stat-id decode verified end-to-end on real 2024 data (Ja'Marr Chase 105 rec / 1335 yds / 8 TD). Pure parsers unit-tested. **This is the seed of sub-project #2's external-ingest layer.** Pulled parquet is gitignored (`data/external_projections/`); regenerable from the script.
- `scripts/benchmark_projections.py` — correct join + PPR-scoring + RMSE/MAE/Spearman/cohort machinery, with a prominent docstring warning that it must NOT be run against `project_season.py` output as a preseason verdict. Reusable as-is for the fair weekly start/sit benchmark.
- Verified: ESPN's `statSourceId=1, statSplitTypeId=0` season projection is genuine preseason (rookie/breakout/injury misses, not contaminated); historical projections retrievable from ESPN + Sleeper.

**Recommended next direction:**
1. **Sub-project #2 — external consensus projection layer for draft** (TODO #38): build on `pull_external_projections.py`; add 1–2 scraped sources (FantasyPros/CBS preseason) for a real consensus average; this becomes the projection basis downstream tools consume.
2. **Draft Hub** on top of the consensus (the actual goal — spend effort on *how we use* projections, not on the projections themselves).
3. **Optional, separate:** the fair weekly start/sit benchmark (our weekly model vs ESPN weekly vs weekly actuals) to decide whether to keep our model for in-season start/sit or retire it. Capability ≠ accuracy — this spike only proved our model can't do preseason, not that it's bad weekly.

**What this does NOT close:** whether our weekly model adds value for start/sit (use-case #2) — unanswered; needs the weekly benchmark. The Track 2 feature-probe treadmill should stop until a downstream consumer exists that the projection quality actually moves.

---

## Vegas Team-Context Integration — DO_NOT_ADOPT across 3 gates (2026-05-18, on branch `feat/qb-wr-vegas-team-context-integration`)

**Status:** Spec + plan + 8 implementation commits + 4 backtest runs + 3 gate runs + verdict report on `feat/qb-wr-vegas-team-context-integration`. Phase 2 of TODO #33c (production integration of the lgb-nb × swap winners from the predecessor probe). Spec at `docs/superpowers/specs/2026-05-17-qb-wr-vegas-team-context-integration-design.md`; plan at `docs/superpowers/plans/2026-05-17-qb-wr-vegas-team-context-integration.md`; summary at `reports/qb_wr_vegas_team_context_integration_summary.md`.

**Verdict: DO_NOT_ADOPT across all 3 gates.**

| Gate | Probe ΔRMSE | Observed ΔRMSE | CI | Verdict |
|---|---|---|---|---|
| (lgb-nb, QB) | −0.0587 | **+0.1112** | [+0.0735, +0.1482] | REGRESSION (sign-flipped, 290% miss) |
| (lgb-nb, WR) | −0.0130 | +0.0068 | [−0.0031, +0.0170] | null/inconclusive |
| (ensemble-decomposed, WR) | n/a | +0.0004 | [−0.0060, +0.0073] | null |

**Builder correctness verified before reporting the regression:** integration's QB feature parquet matched the probe's override parquet byte-identically on all 4 Vegas cols across 9,379 QB rows (max abs delta = 0.0). Not a builder bug.

**Root cause traced:** harness-pairing divergence between `probe_composite` and `run_backtest`. Initial PR #50 data-drift hypothesis was wrong — PR #50's full diff is purely additive logging (`import logging` + `logger.warning(...)`); the placeholder-gsis filter logic is identical to pre-#50. Pre and post backtest runs have **byte-identical row coverage** (2676 QB rows each, perfectly aligned on (gsis_id, season, week)). Re-ran the probe on current data state with `--force-composite --drop implied_team_total spread --model lightgbm-nb --position QB`: **probe reproduces −0.0587 ADOPT exactly** on n=2692. The 16-row delta between probe and gate is all **Taysom Hill** (`gsis_id 00-0033357`) in 2023 — listed as QB on the depth chart (so the QB feature builder includes him) but recorded as `position == "TE"` in weekly_stats. `probe_composite` merges predictions with weekly_stats on `(gsis_id, season, week)` only — pairs Taysom Hill. `run_backtest` filters `holdout_pos[position == "QB"]` first — drops him. Math: those 16 rows alone account for ~6,624 SSE swing (~20 fpts residual diff per row, consistent with Taysom Hill's high-variance utility profile) — enough to flip the QB verdict from gate's +0.111 REGRESSION to probe's −0.0587 ADOPT. **The gate's +0.111 is the production-truth signal** because production never pairs Taysom Hill's QB-depth-chart predictions with his TE weekly_stats row. **First observed case** of a `--force-composite` probe Phase-2 ADOPT failing to replicate in the production gate (prior cases — PR #21 RB PBP, Plan 9 negatives — all replicated cleanly). **Framework follow-up to consider:** align `probe_composite`'s truth-merge to use a position filter matching `run_backtest`'s, so future probes' Phase-2 verdicts are more faithful predictors of the production gate.

**Shipped surface (10 commits on branch):**
- Spec + plan + summary report (3 docs).
- `src/projections/schemas.py` — 4 nullable Float cols on each of `QbFeaturesSchema` + `WrFeaturesSchema`.
- `src/projections/features/qb.py` + `wr.py` — `attach_vegas_team_context_features(out, schedules)` wired into builders.
- `src/projections/models/lightgbm_nb.py` — `_VEGAS_SWAP_REPLACE`, `_VEGAS_SWAP_ADD`, `_swap_for` helper, `_QB_FEATURE_COLUMNS_NB`, `_WR_FEATURE_COLUMNS_NB` constants; `qb_lightgbm_nb` + `wr_lightgbm_nb` factories rewired; `_code_hash_files_nb` tracks `vegas_team_context_features.py`.
- ~20 new tests across schemas, features, models (4 schema files + 1 builder test per position + 6 model tests + 1 rewritten broken-premise test for `test_yards_stat_predictions_match_tuned_baseline`).

**Branch disposition (pending user decision):** Phase 0 (schemas + builder wire-up) is harmless and may help future re-investigation. Phase 1 (lgb-nb factory swap) is what the gate rejects. Three options documented in summary: (1) close PR without merging, (2) merge Phase 0 only + revert Phase 1 factory swap before merge — **recommended**, (3) merge as-is + flip production routing back to baseline — not recommended.

**Decision log:**
- Subagent-driven execution went smoothly for Phase 0+1 (8 implementer subagents + 6 review passes; one subagent suspended mid-test-run on Task 6 and was completed directly in main session).
- Rebased integration branch onto current main HEAD (`f961ab6`) after first gate-1 run flagged the PR #50 asymmetry concern; re-ran post-integration backtests; result was unchanged (same +0.1112 ΔRMSE on QB) confirming PR #50 alone is not the cause but the data-state drift is.
- All 363 schema + feature + lgb-nb tests pass on the integration branch.

**Plan-vs-execution deviations:**
- Task 1+2 implementers needed to also update 2 pre-existing fixture tests in `test_dataframe_schemas.py` to populate the new required cols. Necessary scope-creep; same shape Task 4 had with `test_cache.py::_minimal_wr_features_row`.
- Task 6 subagent suspended waiting for a long-running pytest (~12 min); main session completed it directly. All Task 6 changes match plan template byte-for-byte.
- Phase 2 (backtests + gates) executed directly in main session rather than subagents — appropriate for operational work with branch-switching + long-running shell commands.

**Next direction (concrete):**
1. **External preseason Vegas data spec** — genuine May win totals, OC/HC tenure, FA-acquisition flag, projected pace, projected pass rate. Different mechanism axis from re-deriving `spread_line` / `total_line`; not affected by the probe → gate generalization gap encountered here. Load-bearing for TODO #31 Draft Hub preseason projections.
2. **RB `preseason_*`-only follow-up probe** still queued, but with a weakened prior — this branch's gate reversal suggests the RB probe's verdict may also fail to generalize. Run with the dual-run gate as the load-bearing decision criterion, not the probe.
3. **Probe framework follow-up:** add a position filter to `probe_composite`'s truth merge in `src/projections/backtest/feature_probe.py:574-577` so the probe's Phase-2 pairing matches `run_backtest`'s. Without it, candidates that differentially help/hurt cross-position-mislabeled rows (Taysom Hill, multi-position utility players) will be artifactually advantaged by the probe.

See `reports/qb_wr_vegas_team_context_integration_summary.md` for the full gate-by-gate verdict tables, mechanism analysis, and three branch-disposition options.

---

## Vegas Team-Context Feature Family Probe — SIGNAL at lgb-nb swap QB+WR (2026-05-17, on branch `feat/probe-vegas-team-context`)

**Status:** Spec + plan + impl + 4 probe runs + audit + summary on `feat/probe-vegas-team-context`. Phase 1 of TODO #33c. New compute module `src/projections/features/vegas_team_context_features.py` produces 4 candidate cols (`preseason_implied_team_total`, `preseason_spread`, `season_avg_implied_team_total`, `season_avg_spread`) from `SchedulesSchema`'s already-ingested `spread_line` / `total_line`. Override generator CLI `scripts/build_vegas_team_context_override.py`. Probe runs via existing `scripts/probe_feature_signal.py` (no changes). Spec at `docs/superpowers/specs/2026-05-17-vegas-team-context-probe-design.md`; plan at `docs/superpowers/plans/2026-05-17-vegas-team-context-probe.md`.

**Verdict: SIGNAL.** lgb-nb × swap composite returns 2/4 ADOPT — **QB −0.0587 fpts (CI [−0.092, −0.028]) + WR −0.0130 fpts (CI [−0.022, −0.003])**. RB just misses ADOPT (−0.0113, CI [−0.023, +0.001]); TE NULL. Phase 2 fires via `--force-composite` on lgb-nb because Phase 1 is RidgeCV (which actually *REGRESSES* on QB pooled `passing_yards` at +0.34 fpts augment / +0.22 fpts swap — the recurring "QB augment regression" pattern from PRs #23/#24/#25/#28). The lgb-nb augment is 0/4 ADOPT — the SIGNAL only emerges when the 4 new cols *replace* per-game `implied_team_total` + `spread`, not when they augment them. **Mechanism: trees overfit per-game line noise; smoother preseason + season-to-date signals generalize better at the lgb-nb swap level.**

**Pre-registered prediction (spec §1.2) vs observed:**
- Predicted RB swap as most likely SIGNAL → WRONG; RB just misses, QB + WR carry the signal instead.
- Predicted Phase 1 SIGNAL at QB/WR `passing_yards` / `receiving_yards` → WRONG; Phase 1 REGRESSES on those cells.
- Predicted TE NULL → CORRECT.
- Predicted lgb-nb augment tautological → CORRECT (0/4 ADOPT, all CIs bracket zero).

**Shipped surface:**
- `src/projections/features/vegas_team_context_features.py` — `compute_vegas_team_context_features` (wraps `_shared.build_game_environment` + preseason broadcast + expanding-mean-shift-1) + `attach_vegas_team_context_features` + `build_vegas_team_context_overrides` (with input validation: required cols, gsis_id format, dup-key rejection, row-count invariant).
- `scripts/build_vegas_team_context_override.py` — override generator CLI mirroring `scripts/build_weather_override.py`.
- 20 new tests (15 feature + 5 CLI); all gates clean (mypy + ruff + ruff format).
- 7 report artifacts: override audit + 4 probe outputs (.md + .csv each) + summary.
- `data/features_probe/vegas_team_context.parquet` generated (56,652 rows, 100% preseason coverage / 94.13% season_avg coverage); NOT committed (regenerable).

**Decision log:**
- Hybrid 4-col bundle (preseason × 2 + season-to-date × 2) per spec §3.
- Coverage threshold relaxed to 0.90 for week-1 cold-start NaN on `season_avg_*`; actual coverage 100% / 94.13% so the relaxation was unused in practice.
- Eval window 2021–2024 holdout; 2025 deferred (would require `refresh_features --seasons 2025`).
- Schema integration deferred — probe-only.
- `--force-composite` on lgb-nb runs was load-bearing: without it Phase 2 wouldn't have fired (Phase 1 returned no pooled SIGNAL), and the lgb-nb swap SIGNAL would have been missed.
- Task 1 implementation got a dtype-contract fix in code review (`season_avg_*` cols were dropping from Float64 to float64 through `expanding().mean()`; cast back at commit `715509b`).
- Plan estimated 1–2 hr per lgb-nb run; actual was ~8 min each. Plan estimate was conservative.

**Recommended next direction (concrete):**

Greenlight a **per-position integration plan for QB + WR only** (lgb-nb / ensemble-decomposed routes). Specs / decisions for the integration plan:
1. **Schema-swap on lgb-nb only:** Keep `implied_team_total` / `spread` in `{Qb,Wr}FeaturesSchema` (BaselineModel + ensemble Ridge children still need them); rewire lgb-nb's `_X_FEATURE_COLUMNS` to read the 4 new cols instead of the per-game cols. More surgical than schema-augment; preserves Ridge children's signal.
2. **Extend `_shared.build_game_environment` (or sibling)** to emit the 4 new cols per team-game row.
3. **Refresh feature caches** under `data/features/{qb,wr}/`.
4. **Run dual-run adoption gate** on QB + WR with production routing (QB → lgb-nb directly; WR → ensemble-decomposed). Confirms probe Phase 2 matches gate verdict.
5. **TE: do NOT integrate** (no SIGNAL).
6. **RB: defer.** Phase 2 just misses; a follow-up probe with `preseason_*`-only (drop `season_avg_*`) may cross the threshold.

**Caveats on impact magnitude:**
- The SIGNAL is at *composite ΔRMSE*, not at elite-season magnitude. A QB Phase 2 ΔRMSE of −0.06 fpts is a ~1–2% per-week RMSE reduction; the 33d Chase 250 → 403 gap (a +150 fpts season miss) won't be closed by this integration alone.
- The integration is **necessary but not sufficient** for the elite-season problem. The right framing: this is one step along the 33c axis, not the closing step.
- If integration confirms ADOPT but elite-magnitude persists, the next step is **external preseason Vegas data** (genuine May win totals, OC/HC tenure, FA flags) — a richer encoding of the same mechanism the pbp-derivable proxy partially captures.

**Plan-vs-execution deviations:**
- Task 1 dtype contract fix (caught in code review, fixed at `715509b`).
- Task 4 fixture extension: `_make_schedule_rows([])` early-return branch added for empty-input rejection tests.
- Task 8 audit "unique preseason_spread values per season": shows 30–32 per season rather than always 32 due to coincidental ties (e.g., two teams favored by identical margins in week 1). Not missing data — `preseason_*` coverage is 100%.
- Task 10 stderr handling: first lgb-nb run polluted .md output with sklearn `UserWarning`s; killed and re-ran with `-W ignore::UserWarning` + stderr to separate log. No data loss.
- Verdict interpretation flipped during execution: pre-registered prediction was NULL-leaning with RB as most-likely SIGNAL; lgb-nb swap result flipped it to SIGNAL on QB + WR. Summary captures this honestly.

See `reports/feature_probe_vegas_team_context_summary.md` for the full per-(model, mode) verdict + mechanism annotation, and `reports/feature_probe_vegas_team_context_override_audit.md` for the override audit.

---

## Upside-Sensitive Ranking Diagnostic — Phase 1 verdict NO GREENLIGHT (2026-05-17, on branch `feat/upside-ranking-diagnostic`)

**Status:** Spec + plan + diagnostic CLI on `feat/upside-ranking-diagnostic`. Phase 1 of TODO #33d. New module `scripts/diagnose_upside_ranking.py` consumes the new `season_projection_weekly_<season>.parquet` + `season_projection_distributions_<season>.csv` artifacts (now emitted by extended `scripts/project_season.py` alongside the unchanged naive CSV) and produces a markdown verdict report comparing per-player ranks under `mean / p90 / blend_70_30 / p_elite`. Spec at `docs/superpowers/specs/2026-05-16-upside-sensitive-ranking-diagnostic-design.md`; plan at `docs/superpowers/plans/2026-05-16-upside-sensitive-ranking-diagnostic.md`.

**Verdict: NO GREENLIGHT (durable).** Zero SIGNAL cells across all 24 (position × season × non-mean-metric) cells; 2 MARGINAL cells (WR p_elite 2024, QB p_elite 2024) but neither survives to 2025. Decision gate returns `No greenlight`. **The hypothesis behind TODO #33d is falsified:** the elite-season under-projection problem does NOT live in the upper tail of the existing distribution — it lives in feature signal coverage. Chase 2024 mean=250.74, **p90=283.80**, actual=403. Gibbs 2024 mean=258.75, p90=295.29, actual=390.40. The whole distribution is shifted-down for elites.

**Mechanism finding (load-bearing):** `p90` and `blend_70_30` reshuffle the middle/bottom of each positional cohort (only ~54% / ~77% of all player ranks match `mean` byte-for-byte), but produce **identical top-K sets** at K=5 across all 8 cells and at K=12 across 5 of 8 cells. None of the actual elite finishers move into the predicted top tier under any of these candidates. The mechanism: `blend = 0.7·μ + 0.3·(μ + k·σ) = μ + 0.3·k·σ` is monotonic in μ whenever σ is monotonic in μ, which is empirically what lgb-nb / ensemble-decomposed / baseline produce here. `p_elite` (the only metric that's NOT a monotonic transform of mean) does change rank order and shows MARGINAL on WR/QB 2024 — but its Kendall tau is LOWER than mean across all 8 cells (RB drops 0.77→0.61, WR drops 0.70→0.42), so the top-of-cohort gains come at the cost of overall ordering quality. Not a free win.

**Shipped surface:**
- `src/projections/aggregation/season.py` — `aggregate_to_season(return_samples=True)` overload returning `(summary_df, dict[(gsis_id, season), np.ndarray])`. Backward-compatible default. (T1, commit `2b909f5`.)
- `src/projections/aggregation/season.py` — guard widened to accept `{SAMPLED_SUMMARY, QUANTILE, MIXED}` row-level families (was SAMPLED_SUMMARY-only). Required because QB lgb-nb + WR ensemble-decomposed both emit MIXED rows. **Closes TODO #28.** (commit `ffdd334`.)
- `src/projections/scoring/actuals.py` — new `actual_season_total(weekly_stats, ruleset)` helper extracted from `scripts/compare_predictions_to_actuals.py`. Re-exported from `projections.scoring`. (T2, commits `2e2e31d` + `1b7f9cd`.)
- `scripts/project_season.py` — `_write_season_artifacts(weekly, ruleset, out_dir, season, id_map)` helper; emits 3 artifacts per run (unchanged naïve CSV + new weekly parquet + new distributions CSV). (T3-T4, commits `0fc92c6` + `10460e9`.)
- `scripts/diagnose_upside_ranking.py` — new CLI. Helpers: `_compute_elite_thresholds`, `top_k_overlap`, `top5_rank_err`, `kendall_tau_filtered`, `cell_verdict`, `decision_gate`, `assemble_season_diagnostic`. Markdown report + per-player CSV output. (T5-T10, commits `aedd6f1` → `e13eea5`.)
- 25 new tests across `tests/test_aggregation/test_season_return_samples.py`, `tests/test_aggregation/test_aggregate_mixed_family.py`, `tests/test_scoring/test_actuals.py`, `tests/test_scripts/test_project_season_artifacts.py`, `tests/test_scripts/test_upside_ranking_metrics.py`, `tests/test_scripts/test_diagnose_upside_ranking_cli.py`.
- `pyproject.toml` — added `tabulate>=0.9` (required by `DataFrame.to_markdown` in the diagnostic; was a missing transitive dep).
- 2 report artifacts: `reports/upside_ranking_diagnostic.md`, `reports/upside_ranking_diagnostic_table.csv`. Committed at `34a1685`.

**Decision log:**
- **Diagnostic-first scope** (per spec §1.1). Phase 2 production ranking surface was conditional on this verdict — NO GREENLIGHT means no Phase 2 work.
- **4 metrics committed before run** to avoid data-snooping: `mean` (baseline), `season_p90`, `blend_70_30 = 0.7·mean + 0.3·p90`, `p_elite = P(season ≥ elite_threshold)`. Blend coefficient (0.7/0.3) fixed pre-run.
- **Elite threshold:** 5-year (2019-2023) mean of the 5th-highest season fpts at position, ≥8 games. Computed at run time. Observed: QB=354.3, RB=290.0, WR=316.6, TE=203.5.
- **Decision gate (spec §1.3 #3):** Greenlight requires a single metric SIGNAL at ≥3/4 positions in BOTH 2024 AND 2025. Strict-on-greenlight, lenient-on-marginal — the bar for committing weeks of Phase 2 work was the strong signal.
- **Production routing used:** QB lightgbm-nb, RB baseline, WR ensemble-decomposed, TE baseline (per Plan 8 + PR #41).

**Risks logged (spec §6) — observed outcomes:**
- **Risk #1 (independent weekly draws understate true season variance):** Real. The composite [p10, p90] under-coverage on RB/TE/WR (Plan 5c/6) propagates into season_p90 being biased toward the center. This is plausibly part of why p90 ≈ mean for these positions — the distribution-tail compression compounds the mean-regression compression.
- **Risk #2 (elite threshold sensitivity):** Per-position thresholds are reasonable (top-5 = elite tier). Multiple top-K's reported (5/12/24) bracket the choice. Not a blocker on the negative result.
- **Risk #3 (multiple-comparison effect):** Decision gate's "≥3/4 in BOTH years" requirement protected against false positives. With zero SIGNAL cells observed, this is moot.

**Recommended next direction (highest-leverage):**
1. **33c — forward-looking Vegas team-context features family probe.** Genuinely unexplored feature class. Candidates: as-of-time season win total, season O/U, projected pace, projected pass rate, OC/HC tenure, FA-acquisition flag. Mechanism prediction: most lift on RB + WR (the exact positions where the elite-season miss is worst). Also load-bearing for TODO #31 (Draft Hub preseason projections). Cheapest probe entry: bundle 3-4 Vegas signals → override parquet → `scripts/probe_feature_signal.py`.
2. **33b — `td_rate_per_target` factor-appropriate sub-model probe** (TODO #23 continuation). Named follow-up after PR #38/#41 shipped WR ensemble-decomposed-child for receptions. Lower prior given PR #39 (logit catch_rate) and PR #44 (Tweedie yards_per_target) were both NULL.
3. **Pivot toward Draft Hub surfaces** (TODO #31, K/DST per TODO #10, live snake recommender per draft_ready_checklist §2b.2). The projection model is already good enough for draft-day decisions; what's missing is the surfaces that consume it.

**Plan-vs-execution deviations:**
- **TODO #28 surfaced as a real blocker in T11.** Plan assumed `aggregate_to_season` already accepted all per-row family tags; in fact the function had a hard guard that rejected anything but `SAMPLED_SUMMARY`. Fixed in commit `ffdd334` with single-line guard widening + 3 new tests. Adds a small scope expansion to this PR (closes a separate TODO incidentally).
- **T2 helper location.** Plan had the actuals helper at `scripts/_actuals_helper.py` (per the original spec). Code review correctly flagged that pattern was inferior to relocating into `src/projections/scoring/`. Moved to `src/projections/scoring/actuals.py` (commit `1b7f9cd`) with rename to `actual_season_total` (was `actual_ppr_total` — misleading since it accepts any Ruleset). Plan's Task 10 imports were updated inline by the controller.
- **T6 `top_k_overlap` semantics.** Plan specified strict `/k` denominator (matching spec §3.2). Code review during T9 surfaced that the plan's T9 test fixture (3 QBs, asserts `top5_overlap == 1.0`) would have been unsatisfiable under strict `/k`. Implementer changed to `min(k, n)` denominator — standard Hit@k convention; identical to `/k` in production (cohorts always exceed K=24). Accepted with note that the spec text would benefit from a one-line clarification.
- **T9 `_p_elite_for` O(n²) fix.** Plan had a `for ... if gid == row[...]` scan per row. Caught in code review; fixed to precompute `samples_by_gsis: dict[str, np.ndarray]` once before `df.apply` (commit `72c6fcc`).
- **T10 `tabulate` dep + gsis_id format.** Plan's test fixture used `f"00-{pos}-{player_idx:04d}"` for synthetic gsis_ids — fails `GSIS_ID_PATTERN = ^\d{2}-\d{7}$`. Fixed to a `_gsis_id(pos, player_idx)` helper producing canonical format. `tabulate>=0.9` added to deps (was missing).

See `reports/upside_ranking_diagnostic.md` for the full verdict report and `reports/upside_ranking_diagnostic_table.csv` for the per-player drill-down.

---

## Snake-Draft Cheat Sheet — feature shipped (2026-05-16, on branch `feat/snake-cheat-sheet`)

**Status:** Spec + plan + impl on `feat/snake-cheat-sheet`. Third surface of the Draft Hub sub-project (auction $ generator → VORP → snake cheat sheet). Reads a `VorpTableSchema` parquet + `id_map.parquet` + `LeagueConfig`, emits a per-player table sorted by `(position canonical order, positional_rank)` with gap-based tier breaks (1..N for in-pool, NA otherwise). v1 scope: VORP + tier breaks only. ADP delta and p10/p90 confidence band deferred to follow-up specs. Spec at `docs/superpowers/specs/2026-05-16-snake-cheat-sheet-design.md`; plan at `docs/superpowers/plans/2026-05-16-snake-cheat-sheet.md`.

**Shipped surface:**
- `src/projections/draft/snake_cheat_sheet.py` — `generate_snake_cheat_sheet` public function + `_assign_tiers` private numpy helper (gap-based with fixed-N tiers; n≤N falls back to 1-per-tier).
- `src/projections/schemas.py` — appended `SnakeCheatSheetSchema` (9 cols including nullable Int64 `tier`).
- `src/projections/draft/__init__.py` — re-exports `generate_snake_cheat_sheet`.
- `scripts/generate_snake_cheat_sheet.py` — CLI with `--season --league-config --vorp-input --id-map --tiers-per-position --out` flags; CSV and parquet output supported; per-position stdout summary (top-3 with tier-1 cliff size) as eyeball mitigation.
- 22 tests in `tests/test_draft/test_snake_cheat_sheet.py`, 3 integration tests in `tests/test_scripts/test_generate_snake_cheat_sheet_cli.py`, 1 schema round-trip test appended to `tests/test_schemas/test_dataframe_schemas.py`. All passing.

**Decision log:**
- **Tier algorithm: gap-based, fixed N (default 8).** Captures "talent cliffs" rather than smoothing distributions into arbitrary buckets. N is configurable via `--tiers-per-position`. Alternatives (variable-N gap threshold, k-means, fixed buckets) documented in spec §3.2 and rejected with reason. `np.lexsort` tie-break prefers earlier (higher-rank) gap when tied magnitudes compete for the N-1th-largest slot.
- **Show all players, tier only in-pool.** Output includes out-of-pool players (positional_rank computed across both) so the sheet doubles as a waiver-wire lookup. Out-of-pool rows get `tier = NA`.
- **Display names from `id_map.parquet`, not `depth_charts`.** First draft of the spec named depth_charts as the name source — fact-check during spec-writing revealed `DepthChartsSchema` carries no name column. `IdMapSchema.full_name` is the canonical name source in this codebase (built by `build_id_map` from `nflreadpy.load_ff_playerids()`). Roster-wide, so 2026 pre-season name coverage is good.
- **ADP delta and confidence band deferred.** Spec §1.2 — each blocks on infrastructure that doesn't exist (no ADP ingest; no p10/p90 plumbed through `VorpTableSchema`). Follow-up specs.
- **Empty-input contract stricter than original spec §3.6 said.** Spec originally said "empty input → empty output." Actual impl calls `_select_pool` first, which raises "cannot fill N {slot} slots" because no players can fill any required position. Spec §3.6 + §5.1 #18 updated mid-implementation to match the as-implemented "raise on empty input + non-empty config" contract — failing loudly is correct when the caller asks for rankings at positions with no input.

**Risks logged (spec §6):**
- **No ADP signal means cheat sheet reflects model view, not room view.** Manual ADP cross-reference required during draft for v1.
- **Tier instability across runs.** Gap-based tiers can flip if a small VORP shift moves which gap is "Nth largest." Stdout `tier-1 size` per position surfaces cliff stability for eyeball-check.
- **`_select_pool` now has three callers** (`auction.py`, `vorp.py`, `snake_cheat_sheet.py`). Pool refactors must consider all three; auction test suite remains the regression gate.

**Plan-vs-execution deviations:**
- `LeagueConfig` requires `n_teams > 1`; plan's Task 4 + Task 5 corner-case tests originally specified `n_teams=1` configs which raise validation. Adapted to `n_teams=2, roster_slots={QB: 1}` — same invariants exercised.
- Plan's `_make_vorp_table` sizing of `{QB:4, RB:6, WR:6, TE:4}` for the display-name tests + sort-order test was too small for default 4-team config's RB requirements (2 RB × 4 teams = 8 RB starter slots, + FLEX). Grew to `{QB:8, RB:12, WR:12, TE:8}` (same sizing as the §5.1 #1-2 schema tests). Display-name semantics are pool-size-independent.
- **Spec §3.6 + §5.1 #18 updated mid-implementation** to match the stricter "empty input raises" behavior (commit `3a867f7`). Test originally named `test_empty_input_returns_empty` was renamed to `test_empty_input_raises` in the same commit.
- **Vectorized tier assignment** (code-review polish, commit `9c59924`): plan's `for idx, t in zip(...): tier_col[idx] = int(t)` was replaced with `tier_col[in_pool_idx] = tiers` — `pd.array(dtype=pd.Int64Dtype())` supports numpy fancy-indexed assignment.
- **`--season` parsed-but-unused in initial CLI (Task 8)**; threaded into stdout banner in Task 9 to close the spec §4 inconsistency.
- **Synthetic id_map position derivation fixed in Task 8** — plan-quoted `"QB" if gid.startswith("00-1") else "RB"` would have mismarked WR/TE rows as RB. Implementer derived position from the gsis_id prefix character properly.
- **F401 noqa carry-over:** `pytest` import was noqa-suppressed in Tasks 2-5 (unused); removed in Task 6 when `pytest.raises` was first used.
- **Pre-commit + venv interaction learning:** the inline `PATH=... git commit` syntax doesn't propagate to pre-commit hook subprocesses; must use `export PATH=...; git commit`. Discovered mid-implementation. Worth recording for future sessions.
- **`/simplify` follow-up cleanup commit** (`79b739c`): replaced stringly-typed `for pos_value in df["position"].unique()` with explicit `for pos in Position` enum iteration; deleted duplicated `_POSITION_ORDER` constant; replaced TOCTOU `path.exists()` with `try/except FileNotFoundError`; removed narrating "Stage N:" comments; promoted `_DISPLAY_NAME_FALLBACK` → `DISPLAY_NAME_FALLBACK` so tests import the constant instead of hardcoding "—".

**Recommended next direction:**
1. **ADP ingest + ADP-delta column** (`draft_ready_checklist.md` §2b.3). The biggest decision-relevance lift to the cheat sheet. FantasyPros has a free CSV export; Sleeper API exposes it. Either lands as an ingest spec + a small schema extension on `SnakeCheatSheetSchema`.
2. **Confidence band — p10/p90 floor/ceiling rank.** Plumb `season_p10` and `season_p90` through to the cheat sheet. Either extend `VorpTableSchema` to carry them (forces upstream/downstream changes) OR have this CLI re-aggregate from `weekly_projections` directly. Decide in the follow-up spec.
3. **Live snake-draft recommender** (`draft_ready_checklist.md` §2b.2). The other §2b consumer. Two viable approaches sketched: greedy (highest-VORP available at position of need) and lookahead (ADP-simulated opponents). The latter is ADP-blocked.
4. **Tier-stability variants.** If users find the default gap-based tiers too wobbly across runs, add `--tier-algorithm` flag accepting `gap` (default), `kmeans`, `fixed-buckets`. Trigger: user feedback after first real draft.
5. **Cross-Draft-Hub-CLI refactor follow-up** (surfaced by /simplify but out-of-scope for this PR): consolidate the duplicated `_make_config` / `_POSITION_ID_PREFIX` / `_make_vorp_table` test fixtures across `tests/test_draft/test_auction.py`, `test_vorp.py`, `test_snake_cheat_sheet.py` into a `tests/test_draft/conftest.py`; extract a shared `_write_output(df, path)` helper (the auction CLI version has a `.suffix` case-sensitivity bug — fix it in the same pass).

**Known pre-existing test failure (NOT introduced by this feature):**
`tests/test_models/test_decomposed_baseline.py::test_dispatch_default_model_class_for_wr_is_unchanged` — same stale pin flagged in PR #40 and PR #41 reviews. WR routing was flipped to `ensemble-decomposed` in the 2026-05-15 WR ensemble PR; this pre-gate test asserts the old `ensemble` value. Fails on `main` too. Worth a small follow-up to clean up.

See `docs/superpowers/specs/2026-05-16-snake-cheat-sheet-design.md` and `docs/superpowers/plans/2026-05-16-snake-cheat-sheet.md`. Draft-readiness status: `draft_ready_checklist.md` §2b.1 flipped to `[x]`.

---

## Preseason Projections — v1 framework + naive baseline shipped (2026-05-17, on branch `worktree-feat+preseason-projections`)

**Status:** New sub-package `src/projections/preseason/` ships v1 framework + `NaivePreseasonModel`. 4 modules (`features.py`, `model.py`, `project.py`, `backtest.py`), 3 new pandera schemas in `src/projections/schemas.py` (`PreseasonFeaturesSchema`, `PreseasonProjectionSchema`, `PreseasonBacktestSchema`), 2 new CLI scripts (`scripts/preseason_project_season.py`, `scripts/backtest_preseason.py`), 29 tests across the suite + 2 CLI integration tests. Spec at `docs/superpowers/specs/2026-05-17-preseason-projections-design.md`; plan at `docs/superpowers/plans/2026-05-17-preseason-projections.md`.

The baseline implements three branches: veterans via `prior_1_per_game × 16` (with prior_2 / prior_3 fallback); rookies via per-(position, stat) Gamma GLMs on `log(draft_pick_overall + 1)` trained on rookie-year season totals from `draft_picks ⋈ weekly_stats`; UDFAs imputed to pick=300. Distribution shape is degenerate (point-mass) for v1.0 — `mean = p10 = p50 = p90` — and the v1.5 trained-model spec is the next slot for adding real per-stat distribution width. fpts come from the canonical `projections.scoring.scoring_coefficients` map (no duplicated scoring math).

Backtest harness (`backtest.py`) is walk-forward: for each `target_season ∈ {2024, 2025}`, train on `[train_start, target_season-1]`, predict, aggregate `weekly_stats[season=target_season]` to per-player actual fpts, inner-join, compute per-position RMSE + Spearman top-50 + coverage diff, apply per-cell verdict (ADOPT / NULL / DO_NOT_ADOPT). v1.5+ ship gate: ≥6/8 cells ADOPT AND zero DO_NOT_ADOPT.

**Known v1.0 → v1.1 gap (flagged in spec §7.6, deferred):** the markdown report's per-position top-20 spot-check tables and player-name coverage-diff sidebars. The verdict tables + per-cell metric tables + coverage-diff counts ship in `write_backtest_report` (sufficient for the gate); the enhanced report variants require threading prediction+actual frames through `walk_forward_backtest`'s return type and is bounded follow-up.

**Next:**
1. **Produce the 2026 partition.** One-liner: `python scripts/preseason_project_season.py --season 2026 --ruleset espn_ppr`. Requires `data/raw/depth_charts/season=2026/` to be materialized first via `from projections.ingest.depth_charts import refresh_depth_charts; refresh_depth_charts(seasons=[2026])` — `depth_charts_2026.parquet` was confirmed available upstream via the nflverse HEAD probe (TODO #32 footer).
2. **Generate v1.0 characterization backtest.** `python scripts/backtest_preseason.py --model naive-preseason --target-seasons 2024,2025` — surfaces the floor RMSE / Spearman that v1.5 trained models will be benchmarked against.
3. **v1.5 spec: first trained model class.** Likely a GammaGLM on `(prior_1, prior_2, prior_3, age, depth_chart_rank, team)`. Backstop: LightGBM-quantile if Gamma underperforms.

Closes TODO #31 (preseason-projections "first plan should be brainstorm + roadmap"). Flips `draft_ready_checklist.md` §1a row 1 (2025 ingest) and §1b row 2 (`predict_season.py SEASON`) to `[x]`.

---

## RB Rushing + Receiving Decomposition Probe — verdicts 5x NULL (2026-05-16, on branch `feat/probe-rb-decomposition`)

**Status:** New probe `src/projections/backtest/rb_decomposition_probe.py` tests whether decomposing RB stats into two shared volume axes (carries, targets) x per-stat efficiency factors beats per-stat direct RidgeCV. 5 composed stats: rushing_yards, rushing_tds (carries axis) + receptions, receiving_yards, receiving_tds (targets axis). Sub-model = RidgeCV everywhere (decomposition-only test; factor-appropriate sub-models are separate cycles). Spec at `docs/superpowers/specs/2026-05-16-rb-decomposition-probe-design.md`.

**Per-stat verdicts:**

| Stat | n_paired | RMSE delta | 95% CI | Composite-fpts equiv | Verdict |
|---|---:|---:|---|---:|:---:|
| rushing_yards | 3291 | -0.0931 | [-0.1915, +0.0058] | -0.0093 fpts | NULL |
| rushing_tds | 3291 | +0.0010 | [-0.0014, +0.0033] | +0.0062 fpts | NULL |
| receptions | 3291 | -0.0004 | [-0.0022, +0.0016] | -0.0004 fpts | NULL |
| receiving_yards | 3291 | -0.0344 | [-0.0850, +0.0150] | -0.0034 fpts | NULL |
| receiving_tds | 3291 | -0.0003 | [-0.0012, +0.0006] | -0.0017 fpts | NULL |

**Coverage:** carries > 0 rate 0.9638-0.9761 across 2021-2024 (above 0.95 threshold every year); targets > 0 rate 0.7799-0.8518 across 2021-2024 (**BELOW THRESHOLD for all four eval years**). The receiving-axis coverage flag is structural for RBs — a meaningful fraction of RB-weeks have zero targets, which is a legitimate observation of low-volume / out-of-rotation usage rather than a data-quality issue. Receiving-stat verdicts therefore rest on the targets > 0 subset (rough effective n on the order of 0.80 × 3291 ≈ 2600); rushing-stat verdicts are at full eval coverage. The MARGINAL magnitude flag fires on 3 stats (receptions, receiving_yards, receiving_tds) whose |fpts delta| sits below the 0.005-fpts threshold per PR #31's retrospective rule.

**Plan-vs-execution deviations:**
- RB feature cache (`data/features/rb/...`) for seasons 2018-2020 predated the weather-column additions in commit `09e0d76` and didn't carry `wind_speed_mph` / `is_high_wind` / `temperature_f` / `is_grass_surface`, so `read_features` failed pandera validation on the first probe invocation. Resolved by running `scripts/refresh_features.py rb --seasons 2018-2024` once before the probe (85 s wall-clock); no code change. Probe itself ran in 5.4 s.
- Per-stat reports (spec §6 line listing 5 per-stat .md files): not produced. With all 5 stats NULL, per-stat reports would duplicate the summary table's verdict + CI + composite-fpts equivalent without adding decision-influencing content. If any future re-run produces a SIGNAL, generate per-stat reports for the SIGNAL stat(s) only.

**Mechanism interpretation:** Decomposition with RidgeCV on every sub-model is statistically indistinguishable from direct RidgeCV on RB rushing AND receiving stats. The four CIs that lean negative (everything except rushing_tds) cluster their upper bounds within 0.015 of zero — there's no hidden SIGNAL being masked by noise. This is the cleanest possible NULL outcome: same model class, same residual variance, every stat. Compare against the WR target-decomposition probe (PR #32), where the same Ridge-vs-Ridge recipe found a marginal SIGNAL on WR receptions; the same recipe finds nothing on RB receptions. Two distinct mechanistic stories rule out: (1) the volume / efficiency separation does not, by itself, expose RB-specific signal that direct RidgeCV misses; (2) PR #32 / PR #33's marginal WR-receptions SIGNAL does NOT generalize to RB even on the same stat — RB receiving-volume targets are too sparse (≤85% coverage) and too correlated with the rushing-volume features for the decomposition to find independent leverage.

**Recommended next direction:** Close the RB decomposition direction at this Ridge-only unit. Per spec §4 "all 5 NULL" branch, no integration plan is greenlit. Factor-appropriate RB sub-model probes (Poisson on carries / targets, Gamma on yards-per-X, logit on rate factors) are NOT next — those are conditional on at least one RB Ridge-vs-Ridge SIGNAL per spec §1.4 #3 — none here. With this PR landed, three consecutive factor-class / decomposition probes have now returned NULL (logit catch_rate PR #39, Tweedie yards_per_target PR #44, RB decomposition this PR); the decomposition-and-factor-class axis on receiving stats is empirically exhausted on Ridge-vs-class without independent mechanism evidence. Higher-leverage next directions remain: (1) refined-unit feature work under TODOs #24 / #25 that proved productive on WR / TE; (2) entirely different mechanism families (deeper-unit player-trajectory, new ingest sources). The RB decomposition recipe should not be re-tested without independent mechanism evidence that something has changed (new features, new sub-model classes proven elsewhere first).

See `reports/feature_probe_rb_decomposition_summary.md` for full per-stat tables + coverage flags + plan-vs-execution-deviations.

---

## Tweedie yards_per_target Probe — verdict `NULL` (2026-05-16, on branch `feat/probe-tweedie-yards-per-target`)

**Status:** New probe `src/projections/backtest/tweedie_yards_per_target_probe.py` tests whether replacing the yards_per_target efficiency sub-model class from `RidgeCV` on the ratio + clip(>=0) to `TweedieRegressor(power=1.5, link="log")` with alpha CV-selected lowers per-stat receiving_yards RMSE on WR rows. Spec at `docs/superpowers/specs/2026-05-16-tweedie-yards-per-target-probe-design.md`.

**Verdict:** `NULL` — RMSE delta -0.0121 yards (95% CI [-0.0564, +0.0353]), n_paired = 5195. Composite-fpts equivalent -0.0012 fpts. Magnitude flag fired: |delta| 0.0121 < 0.050 yards (|delta_fpts| < 0.005) per PR #31's retrospective rule (marginal zone — moot here because the CI brackets zero regardless).

**Mechanism interpretation:** The CI brackets zero with the upper limit well into positive territory (+0.0353 yards), so the Tweedie log-link sub-model is statistically indistinguishable from the Ridge-on-clipped-ratio incumbent on per-stat receiving_yards RMSE over the 2021-2024 pooled WR rows. Per-year breakdown shows directional noise around zero: 2021 +0.034, 2022 -0.036, 2023 +0.025, 2024 -0.075 — no single eval year's CI excludes zero (2024 comes closest at hi +0.007). The mechanistic story this rules out: Tweedie's compound-Poisson-Gamma shape on yards_per_target does NOT materially outperform the Ridge-on-ratio + clip(>=0) approximation on the WR data we have. Coverage is comfortably above 0.95 across all eval years (0.989-0.996), so the verdict is not muddied by the `targets > 0` filter.

**Mechanism caveat:** Incumbent arm (Ridge-decomp) is NOT current production for receiving_yards; production is direct RidgeCV via `ensemble-decomposed` (which decomposes Stat.RECEPTIONS only per PR #36/#38). A SIGNAL verdict at this gate would NOT have implied Tweedie-decomp beats current production; that comparison is the integration adoption-gate's question. The NULL verdict short-circuits that question — no integration cycle is warranted.

**Recommended next direction:** Close the yards_per_target factor-appropriate direction — the strictly-positive right-skewed efficiency factor's distributional shape is not large enough on real WR data to justify the class swap. Next slot per spec §6 is a `td_rate_per_target` factor-appropriate probe (Poisson or logistic, depending on whether td-per-target is treated as a count-per-trial or a Bernoulli rate) on a separate cycle. With both catch_rate (NULL, PR #39) and yards_per_target (NULL, this PR) now closed factor-appropriate, the factor-class-swap line of attack on WR receiving has produced two NULLs in a row — the recipe-change axis (decomposition itself, PR #36/#38) carries more weight than the sub-model-class axis on this data.

**Plan-vs-execution deviations:**
- **Negative-yardage filter on efficiency training rows.** Tweedie deviance requires y >= 0; ~0.6% of WR `targets > 0` rows have negative `receiving_yards` (real-data laterals / lost yards on receptions). `walk_forward_residuals` was tightened from `targets_train > 0` to `(targets_train > 0) & (yards_train >= 0.0)` on the efficiency-fit row mask, applied to BOTH arms so the comparison stays apples-to-apples. Eval rows are NOT filtered. Impact: minimal — <1% of training rows dropped per fold, applied symmetrically. Note: spec §3.1 forbade filtering `receiving_yards > 0` to preserve Tweedie's native handling of the y == 0 point mass; the y >= 0 mask honors that intent — y == 0 rows (the Tweedie point mass) are kept and only the unanticipated y < 0 NFL-laterals case is removed.
- **No ConvergenceWarning fired** during the run.
- **Wall-clock ~7 seconds** (well under the 5-15 min plan estimate; n_rows-per-fold smaller than worst-case).

See `reports/feature_probe_tweedie_yards_per_target_summary.md` for the full decision log + per-year tables + coverage + magnitude flag + deviations section.

---

## VORP Generator — feature shipped (2026-05-16, on branch `feat/vorp`)

**Status:** Spec + plan + impl on `feat/vorp`. Pool-boundary replacement-level method: replacement at position P is the season-mean projection of the worst player at P that makes it into the drafted pool (the `_select_pool` output, lifted from the auction module so both specs share one definition). VORP = `season_mean_fpts − replacement_fpts(P)`. Output parquet is the input contract for the already-shipped auction $ generator and the upcoming snake-draft cheat sheet. Spec at `docs/superpowers/specs/2026-05-16-vorp-design.md`; plan at `docs/superpowers/plans/2026-05-16-vorp.md`.

**Shipped surface:**
- `src/projections/draft/vorp.py` — `generate_vorp_table(projections, league_config)` public function.
- `src/projections/draft/_pool.py` — new module-private `_select_pool` (lifted from `auction.py`; generalized to accept inputs without a `vorp` column so VORP itself can call it).
- `src/projections/draft/auction.py` — re-imports `_select_pool` from `_pool`; no behavioral change.
- `src/projections/schemas.py` — appended `VorpTableSchema`.
- `scripts/generate_vorp_table.py` — CLI with `--season --league-config --projections-input --out` flags; CSV and parquet output; per-position stdout summary including `replacement_fpts` for eyeball sanity (mitigates the "pool-boundary is one specific definition" risk).
- ~28 new tests: 22 in `tests/test_draft/test_vorp.py`, 3 in `tests/test_draft/test_pool.py`, 3 in `tests/test_scripts/test_generate_vorp_table_cli.py`, 1 schema round-trip appended to `tests/test_schemas/test_dataframe_schemas.py`. All passing; mypy + ruff + format clean across 191 source files. The 21-test auction regression gate (`tests/test_draft/test_auction.py`) passes unchanged after the `_select_pool` lift.

**Decision log:**
- **Pool-boundary replacement-level (§3.1)** chosen over the two alternatives. §3.2 strict-positional (replacement = `N_teams × starters_at_P + 1`) ignores FLEX / SUPER_FLEX / BENCH and produces wrong replacement levels for any non-trivial roster. §3.3 bench-buffer adds a free parameter (`buffer_size`) that has no principled value. Pool-boundary naturally accounts for FLEX/SUPER_FLEX/BENCH composition and is internally consistent with the auction generator (both call the same `_select_pool`).
- **K/DST out of scope for v1.** TODO #10. VORP raises explicitly if `LeagueConfig.roster_slots` requires a position not present in the projections input — the pool composition is undefined otherwise, and silently producing partial output would propagate a sign error into the downstream auction $ table.
- **Column rename at the VORP boundary** (`season_mean` → `season_mean_fpts`), with an inline comment explaining the rationale. Cross-spec consistency (renaming the upstream projections column too) deferred per spec §6.
- **Refactor:** `_select_pool` lifted from `src/projections/draft/auction.py` into module-private `src/projections/draft/_pool.py`, generalized to accept inputs without a `vorp` column. Auction's 21-test regression gate passes unchanged.

**Risks logged (spec §6):**
- **Pool-boundary is one specific definition.** Magnitudes will differ from public ESPN / Yahoo / FantasyPros cheat sheets, which typically use strict-positional or implicit-bench replacement. **Mitigation shipped:** CLI emits per-position `replacement_fpts` in the stdout summary for user eyeball before trusting numbers downstream.

**Plan-vs-execution deviations:**
- **Spec contradiction caught mid-implementation.** Original §3.6 said VORP is "silent + downstream auction error" on a required-but-missing position. Code correctly raises, since with no projections for a required position the pool composition is undefined. Spec amended in commits `9504b0e` (§3.6 + §5.1 #17) and `6e4b719` (§5.4 #26 + §4 stdout example) to reflect the raise-on-missing behavior.
- **Lowercase `"espn_ppr"` bug in plan fixtures.** `Ruleset.espn_ppr().name` is uppercase `"ESPN_PPR"`. Task 4 implementer caught and fixed the fixture; Task 11 fixture was pre-fixed.
- **Multiple small cleanup commits between major tasks** — spec docstring tightenings, dead-branch removal, enum-vs-string-constant import substitution, test comment trim, defensive assertion additions. All from reviewer feedback, none material to the algorithm.

**Recommended next direction:**
1. **Snake-draft cheat sheet** (`draft_ready_checklist.md` §2b.1) — the other VORP consumer, same input contract (the parquet this spec produces), different output surface (per-position ordered list with VORP / ADP delta / tier / confidence band).
2. **`predict_season.py` generalization** (`draft_ready_checklist.md` §1b) — required for any post-2024 VORP run. The current projection generator is pinned to a specific season; generalize before season-start draft prep.
3. **K/DST projection generation** (TODO #10) — unblocks K/DST VORP and finishes the league-config story for standard ESPN/Yahoo formats.

See `docs/superpowers/specs/2026-05-16-vorp-design.md` and `docs/superpowers/plans/2026-05-16-vorp.md`. Draft-readiness status: `draft_ready_checklist.md` §2a.1 flipped to `[x]`.

---

## Auction Values $ Generator — feature shipped (2026-05-16, on branch `feat/auction-values`)

**Status:** Spec + plan + impl on `feat/auction-values`. First module of the Draft Hub sub-project. Standard SOS allocation: reserve `min_bid` per drafted slot, distribute remaining budget proportionally to positive VORP among the rostered pool. Strategy-agnostic — one $ per player; downstream live-bid recommender owns aggressiveness / roster-shape knobs. Spec at `docs/superpowers/specs/2026-05-16-auction-values-design.md`; plan at `docs/superpowers/plans/2026-05-16-auction-values.md`.

**Shipped surface:**
- `src/projections/draft/` — new subpackage (Draft Hub seed). `LeagueConfig` (frozen pydantic, shared with VORP/snake specs), `_select_pool` helper, `generate_auction_values` public function.
- `src/projections/schemas.py` — appended `AuctionValuesSchema`.
- `scripts/generate_auction_values.py` — CLI with `--season --league-config --vorp-input [--reference-prices] --out` flags; CSV and parquet output supported; per-position stdout summary as VORP-quality eyeball mitigation (spec §6 risk).
- `configs/league_espn_ppr_12team.json` + `configs/league_espn_half_10team.json` — example league configs.
- 33 tests in `tests/test_draft/`, 3 integration tests in `tests/test_scripts/test_generate_auction_values_cli.py`, 1 schema round-trip test appended to `tests/test_schemas/test_dataframe_schemas.py`. All passing; mypy + ruff + format clean across 186 source files.

**Decision log:**
- **Algorithm A only.** Pure VORP-to-$ (no per-position market scaling, no ADP anchor). Self-contained, no new ingest. `--reference-prices` flag allows pasting an external $ sheet for sanity comparison without baking calibration into the algorithm. Algorithms B/C (market scaling / ADP anchor) deferred to separate specs blocked on data-ingest scope that doesn't exist.
- **VORP is a sibling spec, not bundled.** This spec consumes a `vorp_table` parquet; the VORP spec will be its own (smaller) PR. Script errors clearly if the parquet is missing.
- **Strategy is downstream.** $ generator is strategy-agnostic; live-bid recommender will accept the strategy knobs in a follow-up spec.
- **Pool selection is projection-rank, not VORP-rank.** Spec §3 step 1 — actual drafts assign players to roster slots which have positional structure, so a high-VORP QB18 doesn't go on a roster when QB1-QB12 are already drafted. Pool fills position-specific → FLEX → SUPER_FLEX → BENCH.

**Risks logged (spec §6):**
- **Calibration vs real auction markets.** Algorithm A reflects model's view of value, not market clearing prices. Trigger to spec B/C if draft-day curve feels wrong.
- **VORP-spec coupling.** Broken VORP produces silently-broken $; schema invariants pass either way. **Mitigation shipped:** CLI emits per-position summary (top-3 $, in-pool count, min/median/max VORP within pool) for user eyeball before trusting output.

**Plan-vs-execution deviations:**
- Spec / plan worked-example listed `roster_size = 16` for the standard 12-team ESPN PPR config; actual sum is 17 (1 QB + 2 RB + 3 WR + 1 TE + 1 FLEX + 1 K + 1 DST + 7 BENCH). Spec corrected post-hoc. Pool-size invariant tests had the same off-by-one; implementer corrected mid-flight.
- Plan's `_bulk_position_rows` test helper embedded letters into the GSIS_ID-regex digit slots (`f"00-{position.value}{i:05d}"[:10]` → `00-QB000001`), which fails `AuctionValuesSchema`'s `\d{2}-\d{7}` constraint when the fixture is run through `validate`. Implementer swapped to a per-position digit prefix scheme (e.g. `00-1000001` for QB).
- Code review caught a real Important bug in the rounding-drift correction: with negative drift and small/skewed pools, the smallest-fractional candidates (the floor players) would be selected for `-1` adjustment, dropping them below `min_bid`. Reviewer reproduced with `n_teams=5, budget=2, min_bid=1, roster=QB:1, vorps=[1,1,1,0,0]` producing `[3,3,3,0,1]`. Fixed by excluding `rounded == min_bid` rows from the negative-drift candidate set. Regression test pinned.
- `LeagueConfig` initially permitted IR-only roster_slots (passing `min_length=1` but yielding `roster_size = 0`, which would have `ZeroDivisionError`'d the auction algorithm). Code review caught it before Task 5; added `model_validator(mode="after")` asserting `roster_size >= 1`.
- Task 7 used `env={**os.environ, "PYTHONPATH": str(repo_root / "src")}` in the integration test's subprocess — the project `.venv` editable install currently points at `.worktrees/feat-probe-logit-catch-rate/src`, so the subprocess can't import `projections.draft` otherwise. Defensive workaround; survives any future repointing.

**Recommended next direction:**
1. **VORP spec** — required dependency. Per-position replacement-level + `season_mean_fpts − replacement_fpts`. Small spec. Until VORP ships, `generate_auction_values` can be called against a hand-built VORP parquet but won't produce real draft-day values.
2. **Live auction bid recommender** — primary downstream consumer; owns aggressiveness / roster-shape strategy knobs.
3. **Snake-draft cheat sheet** (`draft_ready_checklist.md` §2b.1) — also consumes VORP; can ship in parallel.
4. **Algorithms B/C** (market scaling / ADP anchor) — only if draft-day experience with A reveals a market-divergence problem.

See `docs/superpowers/specs/2026-05-16-auction-values-design.md` and `docs/superpowers/plans/2026-05-16-auction-values.md`. Draft-readiness status: `draft_ready_checklist.md` §2c.1 flipped to `[x]`.

---

## Logit catch_rate Probe — verdict `NULL` (2026-05-16, on branch `feat/probe-logit-catch-rate`)

**Status:** New probe `src/projections/backtest/logit_catch_rate_probe.py` tests whether replacing the catch_rate efficiency sub-model class from `RidgeCV` on the ratio (current production via PR #36/#38) with `LogisticRegressionCV` via Bernoulli-trial row expansion (factor-appropriate for the [0, 1]-bounded ratio response) lowers per-stat receptions RMSE on WR rows. Both arms share the same shared-volume RidgeCV on `targets`; only the catch_rate efficiency sub-model class differs. Spec at `docs/superpowers/specs/2026-05-15-logit-catch-rate-probe-design.md`.

**Verdict:** `NULL` — RMSE Δ -0.0018 receptions (95% CI [-0.0047, +0.0009]), n_paired = 5195. Magnitude flag fired: |Δ| 0.0018 < 0.005 receptions threshold per PR #31's retrospective rule (marginal zone — but moot here because the CI brackets zero regardless).

**Mechanism interpretation:** The CI brackets zero (just barely on the upper side at +0.0009), so the logit-link sub-model is statistically indistinguishable from the Ridge-on-clipped-ratio incumbent on per-stat receptions RMSE over the 2021-2024 pooled WR rows. Per-year breakdown is consistent with this: 3 of 4 years lean slightly negative (2021, 2022, 2024) with 2022 the only year whose CI nearly excludes zero on the negative side (point -0.0052, hi +0.0007); 2023 is essentially flat at +0.0011. The mechanistic story this rules out: logit's proper [0, 1] support without a hard-clip does NOT meaningfully outperform the RidgeCV approximation on the WR data we have. Coverage is comfortably above 0.95 across all eval years (0.989-0.996), so the verdict is not muddied by the `targets > 0` filter.

**Recommended next direction:** Close the catch_rate factor-appropriate direction — the [0, 1]-bounded ratio's tail-calibration weakness is not large enough on real WR data to justify the class swap. Next slot per spec §6 is a `yards_per_target` factor-appropriate probe (log-link Gamma or Tweedie family on the strictly-positive, right-skewed efficiency factor) under the same shared-volume / single-factor-swap design pattern. That probe is the highest-leverage remaining factor-class swap because `yards_per_target` carries more receiving-yards variance than `catch_rate` carries receptions variance, and a Gaussian-on-ratio Ridge is a worse approximation to a Gamma response than to a Bernoulli-mean.

See `reports/feature_probe_logit_catch_rate_summary.md` for full decision log.

---

## WR Ensemble — Decomposed-Baseline Child A Swap — verdict `ADOPT` (binding) (2026-05-15, on branch `feat/wr-ensemble-decomposed-child`)

**Status:** New `wr_ensemble_decomposed()` factory swaps `EnsembleModel`'s child A from `wr_baseline` to `wr_decomposed_baseline`; lgb-nb child unchanged. Per-stat ensemble weights re-fit via pinball at q ∈ {0.10, 0.90}. Spec at `docs/superpowers/specs/2026-05-15-wr-ensemble-decomposed-child-design.md`.

**Per-cell verdicts:**

| Cell | Incumbent | Candidate | n_paired | RMSE Δ (fpts) | RMSE 95% CI | Spearman Δ | Verdict |
|---|---|---|---:|---:|---|---:|:---:|
| **Binding** (gates routing) | ensemble | ensemble-decomposed | 8460 | **-0.0038** | **[-0.0079, -0.0002]** | +0.0002 | **ADOPT** |
| Informational | decomposed-baseline | ensemble-decomposed | 8460 | -0.0074 | [-0.0234, +0.0089] | **+0.0041** (strict pos) | DO_NOT_ADOPT (RMSE CI brackets zero) |

**§1.3.5 outcome:** Routing flipped — `_PositionDispatch[Position.WR].default_model_class` changed from `"ensemble"` to `"ensemble-decomposed"` in `src/projections/models/__init__.py`. `tests/test_models/test_position_dispatch.py`'s `expected` dict updated. Backtest snapshot unchanged (pins only `baseline` model_class values; flip from ensemble→ensemble-decomposed is not in scope of the snapshot).

**Probe-vs-gate magnitude flag.** Binding RMSE Δ -0.0038 fpts is below the 0.005 fpts marginal-zone threshold from PR #31's retrospective. CI is strictly negative ([-0.0079, -0.0002]) so the routing flip is mechanically justified, but the absolute magnitude is small. Flagged so future contributors know this routing flip was made on a 4-millifpts improvement that's statistically conclusive at the pooled-CI level but in the marginal zone in absolute terms.

**Mechanism.** Informational-cell Spearman +0.0041 (strictly positive) is the cleanest evidence: ensemble's lgb-nb mixing adds rank-correlation lift even when child A is already decomposed. Two distinct mechanisms compound rather than cancel — decomposition improves the baseline's point predictions; lgb-nb captures residual non-linear structure ridges miss. Binding cell RMSE Δ -0.0038 (vs PR #36's predicted [0, -0.0103] range) lands in the bottom half of the band — substantial but not full compounding, consistent with some of decomposition's lift overlapping with what lgb-nb already captures.

**Side-effect fix (commit `975cd52`):** `QuantileDistribution.cdf` extrapolates past knots (mirrors `quantile()`), clipped to [0, 1]. Pre-fix asymmetry capped the joint Mixture(Q, X) cdf at `1 - weight*0.05`, breaking brentq inversion in `MixtureDistribution.quantile()` for q in tail. Discovered while building Task 2's mixture-tail unit tests. Test `test_quantile_cdf_clamps_at_endpoints` was pinning the defect — renamed to `_extrapolates_past_endpoints` with a more thorough assertion set.

**Recommended next direction:** factor-appropriate sub-model classes for `catch_rate` (logistic-link). Would lift the small adoption magnitude into the comfortable zone, address the [0, 1]-bounded ratio's tail-calibration weakness, and make decomposing additional stats (receiving_yards, receiving_tds) viable.

**Plan-vs-execution deviations.** Backtest wall-clock ~3 hours (vs PR #36's 34 min) — third model class + slow new `MixtureDistribution.quantile()` path against `QuantileDistribution` components + external workload competition. Plan's hypothesis about the bug location (`mixture._bracket_for_components`) was wrong; actual fix was `QuantileDistribution.cdf`. Plan's assertion about RECEPTIONS `component_b == ParametricNegativeBinomial` was wrong (RECEPTIONS isn't in `COUNT_STATS_FOR_NB`); corrected to QuantileDistribution. `scripts/adoption_gate.py` requires `--position WR` explicit on WR-only data. Stdout em-dashes wrote as cp1252 `0x97` instead of UTF-8 — fixed post-hoc.

See `reports/wr_ensemble_decomposed_summary.md` for the full decision log + per-cell + per-year tables.

---

## WR Target Decomposition Integration — verdict `DO_NOT_ADOPT` (binding) + `ADOPT` (informational) (2026-05-13, on branch `feat/wr-target-decomposition`)

**Status:** Production integration of `DecomposedBaselineModel` (peer to `BaselineModel`) with per-stat decomposition opt-in via constructor arg. v1 ships WR receptions-only decomposition (volume `targets` x efficiency `catch_rate` with sample-time clip `[0, 1]`). New `FrozenSampledDistribution` carries within-row coherent sampling through `score_distribution`; persistence uses `QuantileDistribution` summaries via the existing codec branch (no codec edits). Spec at `docs/superpowers/specs/2026-05-13-wr-target-decomposition-integration-design.md`.

**Per-cell verdicts:**

| Cell | Incumbent | Candidate | n_paired | RMSE Delta (fpts) | RMSE 95% CI | Spearman Delta | Verdict |
|---|---|---|---:|---:|---|---:|:---:|
| **Binding** (gates routing) | ensemble | decomposed-baseline | 8402 | **+0.0109** | **[-0.0080, +0.0285]** | -0.0052 | **DO_NOT_ADOPT** |
| Informational (probe-equivalent) | baseline | decomposed-baseline | 8402 | **-0.0103** | **[-0.0145, -0.0060]** | +0.00001 | ADOPT |

**§1.3.5 outcome:** Infrastructure-only ship. WR production routing unchanged (`_PositionDispatch[Position.WR].default_model_class` stays `"ensemble"`). `DecomposedBaselineModel` + `wr_decomposed_baseline` factory + `_WR_FACTORIES["decomposed-baseline"]` shipped as available infrastructure. No backtest snapshot update.

**Probe-vs-gate calibration:** probe predicted `-0.0042 fpts` on receptions (PR #32 Ridge-vs-Ridge); informational gate measured **-0.0103 fpts** at composite-fpts level -- **2.5x larger** in the same direction. **Favorable surprise**: decomposition recipe carries to composite-fpts more strongly than per-stat probe suggested, likely via within-row coherent sampling + better-calibrated factor-level variance. Binding gate sign-flipped to +0.0109 fpts vs ensemble -- mechanistically expected since `EnsembleModel`'s lgb-nb component provides lift that decomposed-baseline alone cannot recoup. **Magnitude is NOT in the PR #31 marginal-zone** (informational |Delta| > 0.005 fpts and CI strictly negative).

**Recommended next direction (per spec §1.3.5 informational-ADOPT branch).** Swap `BaselineModel -> DecomposedBaselineModel` inside `EnsembleModel`'s child A factory and re-fit ensemble weights. Run a new dual-run gate on `(EnsembleModel-with-decomposed-baseline, WR)` vs current `(EnsembleModel, WR)`. This would compound the lgb-nb contribution (proven at ensemble level) with the decomposition recipe (proven at baseline level). Spec the plan once this PR is merged.

**Deferred follow-ups (probe spec §7):** factor-appropriate sub-model classes (logistic for catch_rate, log-link Gamma for yards_per_target, Poisson for targets) -- now eligible since decomposition has proven informational lift; decomposition for receiving_yards / receiving_tds (NULL in probe); other positions (RB / QB / TE -- each its own probe + integration cycle).

**Plan-vs-execution deviations (all minor):**
- Worktree + venv routing: Windows POSIX-path resolution required `C:/Users/...` Windows paths instead of `/c/Users/...` for absolute `features_root` / `raw_root` kwargs.
- Test fixture `_synthetic_wr_fit_inputs_low_eff_variance` added in Task 3b to make the cross-stat coherence test's rho > 0.5 threshold achievable (standard fixture's analytical rho ~= 0.26 reflects realistic yards-per-target variance, not a broken mechanism).
- Backtest wall-clock ~34 min for 3-model 4-year WR-only (plan estimated 5-15 min; lgb-nb training dominated).
- 4 new `data/ensemble_weights/ensemble_wr_*.json` artifacts generated by the backtest's ensemble fits and committed alongside reports (pre-existing convention).

See `reports/wr_target_decomposition_summary.md` for the full decision log + per-cell tables + probe-vs-gate calibration + §1.3.5 outcome narrative.

---

## WR Receiving Stats Target Decomposition Probe — verdict **SIGNAL (marginal) — receptions cell** (2026-05-10, on branch `feat/probe-target-decomposition`)

**Status:** Probe-only spec shipped per `docs/superpowers/specs/2026-05-10-target-decomposition-probe-design.md` and plan `docs/superpowers/plans/2026-05-10-target-decomposition-probe.md`. **First model-architecture probe in the project** (Track 2A's prior probes have all measured feature additions via override parquets; this probe measures a prediction-recipe change with no override layer). New module `src/projections/backtest/target_decomposition_probe.py` walks forward over 2021–2024, fitting one shared volume RidgeCV (on `targets`) plus three efficiency RidgeCVs (on `catch_rate`, `yards_per_target`, `td_rate_per_target` filtered to `targets > 0`) plus three direct-comparator RidgeCVs (matching `BaselineModel.fit`). Sub-model class deliberately Ridge-only across both arms so any SIGNAL is attributable to *decomposition itself*, not to a model-class change. No new ingest, no schema changes, no production builders touched, no model factory added.

**Per-stat verdicts (pooled 2021–2024, paired-bootstrap CI on RMSE delta):**

| Stat | n_paired | RMSE direct | RMSE decomposed | Δ-RMSE | 95% CI | Verdict | Expected composite-fpts Δ |
|---|---:|---:|---:|---:|---|:---:|---:|
| receptions | 8460 | 2.0324 | 2.0282 | **-0.0042** | **[-0.0079, -0.0004]** | **SIGNAL** | **-0.0042 fpts** |
| receiving_yards | 8460 | 31.1654 | 31.1600 | -0.0054 | [-0.0601, +0.0492] | NULL | -0.0005 fpts |
| receiving_tds | 8460 | 0.4793 | 0.4788 | -0.0005 | [-0.0011, +0.0002] | NULL | -0.0029 fpts |

**Family verdict:** SIGNAL on 1 of 3 cells (receptions); NULL on the other two; no REGRESSION. Spec §4 branch: **"≥ 1 SIGNAL, no REGRESSION → greenlight integration plan"**, with §5 risk #1 caveat flag — the binding cell's expected composite-fpts Δ is -0.0042 fpts, just below the ~0.005 fpts threshold.

**Probe-vs-gate calibration risk (per spec §5 risk #1).** Per-stat RMSE Δ translated to expected composite-fpts contribution via the ESPN PPR coefficients (1.0 fpt/rec, 0.1 fpt/yd, 6.0 fpt/td). Net expected composite-fpts magnitude on the WR cell is -0.0076 fpts summed across the three stats; receptions alone contributes -0.0042 fpts. Per the PR #31 retrospective rule, probe binding-cell magnitudes under ~0.005 fpts (composite-fpts Δ) under coverage relaxation should be treated as MARGINAL, not SIGNAL. Coverage was strictly above 0.95 across all eval years (lowest 0.981 in 2023 eval), so the PR #31 rule's "coverage relaxation" condition does not apply here — but the magnitude alone falls in the marginal zone. The integration plan's adoption gate must weight CI strength (strictly below zero) against the small absolute magnitude.

**Factor orthogonality check (per spec §5 risk #2).** Pearson ρ between volume residual and efficiency residual per eval year, on rows with `targets > 0`. All 12 (stat × year) values have |ρ| < 0.05, well under the 0.2 caveat threshold. **Strongest mechanism-level finding from the probe:** the decomposition cleanly separates the two signal axes — no risk of systematic double-counting in the integration plan's `ProductDistribution` composition.

**Coverage:** 0.981–0.988 (eval) / 0.993–0.994 (train) across 2021–2024. Threshold 0.95 met with margin on every (year, population) cell. No relaxation invoked; PR #31 retrospective MARGINAL rule does not strictly apply (but the magnitude flag does — see above).

**Plan-vs-execution deviations (all minor, all caught during Tasks 1–5).**
- **Ruff Unicode rules + mypy strict on sklearn returns** (Task 1). Source-code docstrings and string literals required ASCII (ruff RUF001/RUF002 disallows Δ, ρ, § in `.py` files); typed local variables needed around sklearn `predict` returns (mypy strict's `[no-any-return]` on sklearn `Any`).
- **Bootstrap floor on synthetic fixtures** (Task 3). `paired_bootstrap_rmse_delta` requires `n_paired >= 100`; synthetic-fixture tests bumped `n_per_season` to 120. Real-data run clears this trivially (8460 paired rows per stat).
- **UTF-8 encoding on Windows** (Task 3). `path.write_text` requires explicit `encoding="utf-8"` because the markdown report bodies use em-dashes and Windows defaults to cp1252.
- **mypy `mypy_path` collision** (Task 4). Plan's combined invocation triggers a pre-existing "Source file found twice" error when run on `scripts/foo.py` + `tests/test_scripts/test_foo_cli.py` together (mypy_path collision); canonical `mypy src tests` is clean.

**What this closes:** target decomposition at the WR receiving cell × 2-factor (volume `targets` × efficiency `{catch_rate, yards_per_target, td_rate_per_target}`) × Ridge-only unit. Greenlights a per-position integration plan analogous to PR #21 / PR #29.

**Recommended next direction.** Per spec §7 named follow-ups, the integration plan must scope:
1. New `DecomposedBaselineModel` peer (subclass of `BaselineModel`) with per-stat decomposition opt-in via constructor arg.
2. Within-row coherent factor sampling — shared per-row `targets` draw flowing into all decomposed stats' composed `SampledDistribution`s. Requires a `ProductDistribution` or sample-set helper.
3. Factor-appropriate sub-model classes (logistic for catch_rate / td_rate_per_target, log-link Gamma for yards_per_target, Poisson / NB-2 for targets) **deferred** to a separate probe + integration cycle conditional on this integration's adoption-gate verdict.
4. Composite-fpts adoption gate on `(DecomposedBaselineModel, WR)` vs production `(EnsembleModel, WR)` with §1.3.5 per-position contingency matrix.

**Implementation option for the integration plan**: opt in to ONLY the SIGNAL stat (receptions) at the model level — leave receiving_yards and receiving_tds on direct ridges. Avoids high-variance ratio sub-models on the first integration cycle. The integration plan's contingency matrix should explicitly cover this option as the conservative path.

**Reports:** `reports/feature_probe_target_decomposition_summary.md`, `reports/feature_probe_target_decomposition_per_stat.csv`, 3 per-stat `.md` reports.

---

## Weather Refined-Unit RB+WR Integration — verdict **full-revert × 2** (probe-vs-gate divergence) (2026-05-09, on branch `feat/weather-refined-rb-wr`)

**Status:** Production strict-replace integration of the 8 refined weather cols (`is_cold_weather`, six surface one-hots, `is_primetime`) into `RbFeaturesSchema` + `WrFeaturesSchema`, replacing the v1 4-col bundle from PR #29, per `docs/superpowers/specs/2026-05-09-weather-refined-rb-wr-design.md`. Followed PR #29's `attach_weather_features` wiring + `scripts/backtest_dual.py` orchestration. **Both binding cells `(LightGBMNbModel, RB)` and `(LightGBMNbModel, WR)` returned `DO_NOT_ADOPT`** with point estimates opposite-signed from PR #30's probe predictions. Per §1.3.5 contingency, both positions full-revert; net code change vs main is **zero** (9 files restored to main's state in commit `c4ba548`). The PR ships only the spec, plan, gate reports, and this summary as historical record.

**Per-position dual-run gate verdicts (4 cells, all DO_NOT_ADOPT):**

| Position | Model class | n_paired | RMSE Δ (fpts) | RMSE 95% CI | Spearman Δ | Verdict |
|---|---|---:|---:|---|---:|:---:|
| RB | baseline       | 5273 | +0.0020 | [-0.0117, +0.0156] | -0.0015 | DO_NOT_ADOPT (NULL) |
| **RB** | **lightgbm-nb** | 5273 | **+0.0012** | **[-0.0064, +0.0090]** | -0.0002 | **DO_NOT_ADOPT (NULL)** |
| WR | baseline       | 8460 | +0.0120 | [+0.0013, +0.0228] | -0.0025 | DO_NOT_ADOPT (RMSE REGRESSION-shape; Spearman strictly negative) |
| **WR** | **lightgbm-nb** | 8460 | **+0.0060** | **[-0.0001, +0.0119]** | -0.0016 | **DO_NOT_ADOPT (RMSE NULL barely; Spearman strictly negative)** |

(Bolded rows are the §1.3.5 binding cells.) The 3 informational classes (`lightgbm`, `lightgbm-tuned`, `ensemble`) skipped per spec §1.3.4 (PR #27 / PR #29 precedent).

**Probe-vs-gate calibration: largest divergence in Track 2A history.** PR #30's swap probe predictions vs gate measurements:

| Position | Probe RMSE Δ | Probe CI | Gate RMSE Δ | Gate CI | Magnitude Δ | Sign |
|---|---:|---|---:|---|---:|---|
| RB | -0.0088 | [-0.0153, -0.0030] | +0.0012 | [-0.0064, +0.0090] | +0.0100 | flipped |
| WR | -0.0050 | [-0.0098, -0.0006] | +0.0060 | [-0.0001, +0.0119] | +0.0110 | flipped |

**~+0.011 fpts shift on both binding cells, with sign flipped.** Compare to historical calibration: PR #20 → #21 matched to 4 decimals; PR #25 → #26 within ~0.004 fpts; PR #25 → #27 within ~0.0017; PR #28 → #29 within ~0.0006 on both cells. **First Track 2A integration where the gate flips a binding-cell sign with the probe's CI strictly negative.** Spec §5 risk register anticipated: "smallest binding-cell magnitude in Track 2A history, just inside the per-cell noise floor of ~0.001-0.002 fpts; a small calibration error could flip WR's lgb-nb cell to MARGINAL or DO_NOT_ADOPT." Both binding cells flipped, not just one.

**Most likely mechanism:** the PR #30 probe's small-magnitude binding cells coincided with `--coverage-threshold 0.90` relaxation (the deepest in Track 2A; 2022 `is_cold_weather` non-NaN rate was 0.66 per (position, season)). Small-magnitude lift estimates from low-coverage features are the most fragile to sample-bootstrap variance. **Retrospective takeaway**: a probe binding-cell magnitude under ~0.005 fpts with coverage relaxation should be treated as MARGINAL, not SIGNAL, even if Phase 2's bootstrap CI test passes.

**Per-position §1.3.5 outcome — both positions full-revert.** Single revert commit `c4ba548` restores 9 files: `schemas.py`, `baseline.py`, `weather_features.py` (docstring), `tests/test_features/test_rb.py` + `test_wr.py`, `tests/test_features/test_cache.py`, `tests/test_scripts/test_tune_lightgbm.py`, `tests/test_schemas/test_dataframe_schemas.py`, `tests/backtest/model_metrics.json`. The PR's `git diff main` shows zero net code change.

**Closes:** the "broad-cut refined-unit weather at the in-builder unit" branch on the RB and WR cells from PR #30. Both are now empirically NULL at the production scale. Refined-unit-of-refined-unit candidates (continuous `kickoff_hour_et`, `is_london`, surface × position interactions, per-team weather acclimation, precipitation, wind direction) remain open under TODO #25 but are **deprioritized** — there's no evidence the refined unit is the binding constraint over v1, and the PR #30 probe is now retrospectively suspect for false-positive signal. None queued; future weather-related plans should require independent mechanism evidence before re-probing.

**Cross-class production-routing follow-up (RB and WR):** Closed-without-action. PR #29 logged the v1 cross-class flip question; with this PR's verdict, lgb-nb-with-anything is even less attractive (lgb-nb-with-refined cell measured +0.0012 RB / +0.0060 WR vs baseline-v1). Plan 8's `BaselineModel` routing for both RB and WR remains the right call.

**Spec gaps caught + fixed during execution:**
- `scripts/refresh_features.py` CLI takes a single position (PR #29 caught this; plan correctly invoked twice).
- `scripts/adoption_gate.py` dual-run mode requires single-model-class run dirs (PR #29 caught; this PR's plan referenced but didn't prescribe the workaround in Task 10 — surfaced at gate-run time as `MergeError: Merge keys are not unique`. Workaround applied: split each run's `results.parquet` by model_class into 4 subdirs, then run the gate 4 times). For any future refined-feature plan, prescribe this in the plan steps directly.
- Pre-commit mypy hook uses system Python (pre-existing pydantic v1/v2 conflict). Workaround: `PATH="/.venv/Scripts:$PATH" git commit`.
- Plan said "4 PR #29 weather tests per file"; actual was 5 (`outdoor_nan_data_propagates_nan` was missed). Plan corrected mid-flight (commit `559d4d9`).

See `reports/weather_refined_rb_wr_summary.md` for the full decision log + per-mode table + probe-vs-gate calibration + per-position §1.3.5 outcome narrative.

---

## Weather Refined-Unit Family Probe — verdict **SIGNAL** via lgb-nb composite (RB swap + WR augment + WR swap) (2026-05-09, on branch `feat/probe-weather-refined`)

**Status:** Probe-only spec shipped per `docs/superpowers/specs/2026-05-09-weather-refined-unit-probe-design.md` and plan `docs/superpowers/plans/2026-05-09-weather-refined-unit-probe.md`. Implements three refined-unit weather features (`is_cold_weather`, multi-class surface one-hot, `is_primetime`) on top of PR #28's `weather_features.py` module. No new ingest, no schema changes; production builders unchanged (PR #29's RB+WR integration consumes the v1 4-col subset and ignores the new override-only cols).

**Verdict:** **`SIGNAL`** per spec §1.3 criterion 3 (BaselineModel + lgb-nb composite via `--force-composite`).

**ADOPT cells (3 of 16):**
- **(lgb-nb augment, WR)** — composite RMSE Δ **-0.0051 fpts** (CI [-0.0097, -0.0006]); CI strictly negative.
- **(lgb-nb swap, RB)** — composite RMSE Δ **-0.0088 fpts** (CI [-0.0153, -0.0030]); CI strictly negative.
- **(lgb-nb swap, WR)** — composite RMSE Δ **-0.0050 fpts** (CI [-0.0098, -0.0006]); CI strictly negative.

**Refined-unit-specific decoding** per spec §1.2:
- **WR**: strict refinement (swap + augment ADOPT) → integration plan replaces v1 weather cols with refined 8-col bundle in `WrFeaturesSchema`.
- **RB**: replace-only (swap ADOPT, augment ~null) → integration replaces v1 with refined in `RbFeaturesSchema`.
- **QB / TE**: close at this cut. Refined-unit-of-refined-unit candidates remain on TODO #25.

**Greenlights** a per-position integration plan analogous to PR #29: replace v1 weather cols (`wind_speed_mph`, `is_high_wind`, `temperature_f`, `is_grass_surface`) with the refined 8-col bundle (`is_cold_weather`, `is_primetime`, plus 6 surface one-hots: `is_a_turf`, `is_astroturf`, `is_fieldturf`, `is_grass`, `is_matrixturf`, `is_sportturf`) in `RbFeaturesSchema` + `WrFeaturesSchema`. Production routing for both positions unchanged (stays on `BaselineModel` per Plan 8); the lgb-nb-only ADOPT verdict means the cross-class re-eval (would lgb-nb-with-refined-weather beat baseline at the position level?) is the natural follow-up question.

**Recurring QB augment regression** — sharper this round. (lgb-nb augment, QB) at composite RMSE Δ **+0.0099 fpts** (CI [+0.0002, +0.0202]) — CI strictly above 0. Worse than PR #28's +0.0077 fpts (CI bracketed 0). Pattern continues from PR #23 / #24 / #25 / #28 — adding context/team/trajectory/weather features to QB inputs consistently overfits on augment configurations across both BaselineModel and lgb-nb composite. Reinforces the "do not extend `QbFeaturesSchema` with weather features" rule established in PR #29.

**In-scope ingest-layer bug fix.** During Task 7's real-data audit, `is_primetime` rate was measured at 0.16% (vs the spec's predicted 12-15%). Investigation traced the cause to `_build_kickoff` in `src/projections/ingest/schedules.py` mis-tagging `nfl_data_py.import_schedules`'s `gametime` (ET wall-clock) as UTC via `pd.to_datetime(..., utc=True)`. Fixed at commit `56df07f` by parsing naive, then `tz_localize("America/New_York")` (handles EDT/EST switch via stdlib zoneinfo) → `tz_convert("UTC")`. Added regression test `test_build_kickoff_localizes_et_wall_clock_to_utc`. Corrected an existing test that had pinned the buggy behavior. After re-running schedules ingest + override regen, `is_primetime` rate measured 21.97% — within back-of-envelope expectation for ~6 of 32 teams playing primetime per week (TNF + SNF + MNF + Saturday + Thanksgiving + Christmas).

**Coverage caveat.** `is_cold_weather` non-NaN rate dips to 0.67 in 2022 across all 4 positions (well below the 0.90 threshold). Probe's pooled-baseline-rows coverage check still passed because pooled rate exceeds 0.90, but the 2022 cold-weather signal is diluted in fold splits. `is_primetime` coverage is 1.000 across all (position, season) cells. Documented per PR #29's coverage-claim hygiene rule.

**Plan-vs-execution deviation.** Augment-mode probe runs used a 8-col refined-only override (`weather_refined_only.parquet`) instead of the plan's prescribed 12-col full override. Reason: post-PR-29, v1 weather cols are in `RbFeaturesSchema` + `WrFeaturesSchema` baseline, so the full 12-col override would collide on those 4 cols in augment mode. Workaround: refined-only override for augment, full override + `--drop` for swap. v1 cols verified bit-identical between baseline and override pre-probe. Spec §1.2 augment-vs-swap semantics preserved. Documented in summary report's decision log.

**What this closes:** TODO #25's three refined-unit candidates at the in-builder-bundle unit (cold-weather threshold, multi-class surface, kickoff hour primetime). Refined-unit-of-refined-unit candidates remain open: continuous kickoff hour, `is_london` early-window cohort, surface×position interactions, per-team weather acclimation, precipitation (would require new ingest), wind direction (would require new ingest). None queued.

**Recommended next direction.** Per-position production-builder integration plan: extend `RbFeaturesSchema` + `WrFeaturesSchema` with the 8 refined cols (replacing the 4 v1 cols), wire `attach_weather_features` through `build_rb_features` + `build_wr_features` (both already do this, just with the v1 subset), update `baseline.py:_RB_FEATURE_COLUMNS` + `_WR_FEATURE_COLUMNS`, run dual-run adoption gate on `(lgb-nb, RB)` + `(lgb-nb, WR)`. Same shape as PR #29 but at the strict-refinement (replace, not augment) level. Production routing for both positions remains unchanged (BaselineModel per Plan 8); cross-class flip evaluation is a separate follow-up.

**Reports:** `reports/feature_probe_weather_refined_summary.md`, `reports/feature_probe_weather_refined_override_audit.md`, 4 per-(model, mode) `.md`/`.csv` files.

---

## Weather Features RB+WR Integration — verdicts: RB ADOPT, WR ADOPT (both ship-as-designed) (2026-05-08, on branch `feat/weather-features-rb-wr`)

**Status:** Production integration of the 4 weather features into `RbFeaturesSchema` + `WrFeaturesSchema` + `build_rb_features` + `build_wr_features` per `docs/superpowers/specs/2026-05-08-weather-features-rb-wr-design.md`. Wired `attach_weather_features` (already public from PR #28) into both builders via the existing `schedules` kwarg. Updated `baseline.py:_RB_FEATURE_COLUMNS` and `_WR_FEATURE_COLUMNS` (same recurring spec gap class as PR #21 / PR #26 / PR #27). No new ingest, no caller-script changes, no fixture extension (weather is per-game, not trailing-N).

**Per-position dual-run gate verdicts (4 cells):**

| Position | Model class | RMSE Δ | 95% CI | Verdict |
|---|---|---:|---|:---:|
| RB | baseline       | -0.0034 | [-0.0103, +0.0042] | DO_NOT_ADOPT (informational; not REGRESSION) |
| **RB** | **lightgbm-nb** | **-0.0077** | **[-0.0157, -0.0001]** | **ADOPT (binding)** |
| WR | baseline       | -0.0026 | [-0.0106, +0.0061] | DO_NOT_ADOPT (informational; not REGRESSION) |
| **WR** | **lightgbm-nb** | **-0.0104** | **[-0.0165, -0.0042]** | **ADOPT (binding)** |

**Probe-vs-gate calibration:** Probe predicted (lgb-nb, RB) -0.0081 / (lgb-nb, WR) -0.0110; gate measured -0.0077 / -0.0104. Both within ~5% of probe predictions and inside probe CIs. Track record extension: PR #20→#21 4-decimal match (RB); PR #25→#26 ~0.004 fpts (WR); PR #25→#27 ~0.0017 fpts (TE); this PR ~0.0004-0.0006 fpts on both binding cells.

**Per-position §1.3.5 outcome:** **Both positions hit the default ship-as-designed branch.** No modified-shape branch fired (would have required `(baseline, POS)` REGRESSION); no revert branch fired (would have required `(lgb-nb, POS)` MARGINAL or DO_NOT_ADOPT). Schema cols stay in `RbFeaturesSchema` + `WrFeaturesSchema`; `attach_weather_features` stays wired into both builders; `_RB_FEATURE_COLUMNS` + `_WR_FEATURE_COLUMNS` extensions stay in `baseline.py`.

**Second integration to bind on a non-default model class** (after PR #27 TE trajectory) and **first integration to bundle two positions into a single PR** with per-position contingency matrix (each position decided independently from the other). RB and WR production routings unchanged: both stay on `BaselineModel`.

**3 informational classes skipped** (lightgbm, lightgbm-tuned, ensemble) per spec §1.3.4 + PR #27 precedent — wall-time risk + TODO #29 lightgbm-tuned pruning candidate framing made the additional ~12 cells (4 per skipped class × 3 classes) low-value. Back-fillable by a follow-up `--model lightgbm,lightgbm-tuned,ensemble` backtest if any cross-class routing-flip discussion needs them.

**Coverage statistics (2021-2024 eval window, per Task 11 measurement on production builder output):** Weather-col coverage on RB / WR is **byte-perfectly identical** to PR #28's probe override (verified by reading `data/features_probe/weather.parquet` directly): wind/temp coverage ranges 67-98% per (position, season), uniformly 100% on `is_grass_surface`. Per-season variation: 2021 ~96%, 2022 ~67%, 2023 ~86%, 2024 ~98%. **PR #28 PM entry's coverage claim ("uniformly ≥92%") was overstated** — pooled 91.6% hides the 2022 trough. Documented for the record; future "coverage uniformly ≥X%" claims should be reported per-(position, season).

**Cross-class deferred follow-ups (per position):** RB and WR each route to `baseline` per Plan 8. With weather cols now in `RbFeaturesSchema` / `WrFeaturesSchema`, separate cross-class re-evals could justify flipping `_PositionDispatch[{RB|WR}].default_model_class` to `lightgbm-nb`. Not load-bearing for any current consumer; queue alongside the next RB- or WR-related work. Same shape as PR #27's TE follow-up.

**What this closes:** TODO #25's broad-cut weather family at the in-builder unit, on **both** the RB and WR ADOPT cells from PR #28. QB and TE remain DO_NOT_ADOPT at this unit per PR #28's probe; not re-tested in this PR's gate. Refined-unit candidates (cold-weather threshold, multi-class surface, kickoff hour, surface × position interactions, per-team weather acclimation, precipitation, wind direction) remain open under TODO #25; recommended priority `is_cold_weather` first.

**Spec gaps caught + fixed during execution:**
- `scripts/refresh_features.py` CLI takes a single position, not `rb wr` together as the plan suggested. Ran twice.
- `scripts/backtest.py` does not have a `--position` flag. Worked around via the `run_backtest(positions=...)` Python API in `scripts/backtest_dual.py`.
- `scripts/backtest.py --update-snapshot` overwrites the entire snapshot file. `scripts/backtest_dual.py` orchestrator preserves rows for non-target model classes.
- `scripts/adoption_gate.py` dual-run mode requires single-model-class run dirs. Each `_run_single_backtest.py` produces a multi-class results.parquet; split into per-model-class subdirs (`run_{baseline,candidate}_{baseline,lightgbm-nb}/`) before invoking the gate.
- Python import caching across the schema-revert boundary: first attempt at the dual backtest reused a single Python process; the in-memory schema classes did not refresh after `git checkout main -- src/projections/schemas.py`. Fixed by subprocess-ing `_run_single_backtest.py` for both runs.

**Follow-up — shipped in this PR:** Code-review reviewer flagged that 4 PRs in a row (PR #21 RB PBP, PR #26 WR trajectory, PR #27 TE trajectory, this PR) have hit the same `baseline.py:_<POS>_FEATURE_COLUMNS` spec gap. The parametrized regression test pinning `set(_<POS>_FEATURE_COLUMNS) == set(SCHEMA.columns) - identity` was added at `tests/test_models/test_baseline_feature_columns_match_schema.py` (5 cases — one per position plus an identity-cols sanity check). Closes the recurring-bug class structurally on every future schema extension.

See `reports/weather_features_rb_wr_summary.md` for the full decision log + per-mode table + probe-vs-gate calibration + per-position §1.3.5 outcome matrix.

---

## Weather Feature Family Probe — verdict **SIGNAL** via lgb-nb augment composite (RB + WR ADOPT) (2026-05-07, on branch `feat/probe-weather`)

**Status:** Probe-only spec shipped per `docs/superpowers/specs/2026-05-07-weather-feature-family-probe-design.md` and plan `docs/superpowers/plans/2026-05-07-weather-feature-family-probe.md`. Implements 4 weather features (`wind_speed_mph`, `is_high_wind` ≥20 mph threshold, `temperature_f`, `is_grass_surface`) sourced from existing `SchedulesSchema` columns — **no new ingest, no schema changes**. New module `src/projections/features/weather_features.py` (compute fns + `attach_weather_features` joiner + public `build_weather_overrides` assembler), override-generator script `scripts/build_weather_override.py`, weather override at `data/features_probe/weather.parquet` (56,652 rows). All tests pass; mypy strict + ruff + ruff format clean. CONTRIBUTING.md updated with a "Regenerating the weather override" subsection.

**Verdict:** **`SIGNAL` (durable)** per spec §1.3 criterion 3 — both BaselineModel + lgb-nb tested at composite via `--force-composite`. **Second SIGNAL family probe in a row** (after PR #25 trajectory) and **first probe where signal lives only in lgb-nb composite, not in BaselineModel** — RidgeCV cannot extract the bundle's non-linear thresholds (`is_high_wind` ≥20, surface category) even with explicit boolean encoding; tree splits do.

**ADOPT cells (2 of 16):**
- **(lgb-nb augment, RB)** — composite RMSE Δ **-0.0081 fpts** ([-0.0163, -0.0005]); CI strictly below 0.
- **(lgb-nb augment, WR)** — composite RMSE Δ **-0.0110 fpts** ([-0.0172, -0.0049]); CI strictly below 0.

QB and TE returned DO_NOT_ADOPT under lgb-nb augment composite (point estimates near zero, CIs bracket 0). BaselineModel returned 0 Phase-1 SIGNAL across both modes (0/120 augment, 0/120 swap). lgb-nb swap returned the degenerate all-zero composite — weather cols have no v1 counterparts, so the candidate-side drop+add is a no-op (spec §5.2 anticipated this).

**Coverage relaxation — `--coverage-threshold 0.90` (vs spec 0.95 default).** Outdoor-weather NaN rate measured at 8.39% (per `reports/feature_probe_weather_override_audit.md`); upstream `nfl_data_py` data quality issue (outdoor games where `wind` AND `temp` are both NaN despite no dome, concentrated in older 2018-2019 seasons). Per-(position, season) coverage in 2021-2024 eval window is uniformly ≥92% across all 4 positions. Spec §1.3 fallback invoked. Shallower than PR #25's 0.35 (deepest in Track 2 history); on par with PR #23's 0.90 precedent. Bias is symmetric across baseline + candidate arms under the probe's left-merge join.

**Recurring QB augment regression check — milder than PR #23/#24/#25 pattern.** Per-stat Phase 1 fired one QB regression cell (`rushing_yards` 2023, +0.0812 fpts CI [+0.0133, +0.1515]) — single year, single stat. Pooled QB rushing_yards is NULL across all 4 modes. Composite QB augment lgb-nb is +0.0077 fpts (CI [-0.0114, +0.0266]) — DO_NOT_ADOPT, brackets zero, **NOT REGRESSION**. PRs #23/#24/#25 saw composite-level QB augment regressions of +0.0268, +0.0276, and +0.0382 respectively; weather's +0.0077 is within noise. Plausibly weather information that's already partially captured by `roof_dome` + Vegas-implied `implied_team_total` doesn't deliver enough new QB-relevant signal to trigger the QB-specific overfit pattern. Worth re-checking the 2023 rushing_yards cell on real production data if a follow-up weather plan ever targets QB.

**Mechanism annotation:**
- **WR signal** (-0.0110 fpts) was predicted by spec §1.1 — wind suppresses downfield passing (`is_high_wind` boolean activates the regime), grass-vs-turf affects YAC. lgb-nb-only adoption confirms non-linear: the threshold encoding is doing more work than continuous wind speed in a linear model could.
- **RB signal** (-0.0081 fpts) was unexpected. Two plausible mechanisms (neither testable from probe alone): `is_grass_surface` (51% True) gives RBs a meaningfully different footing/cut-back regime; cold-weather games shift offensive balance toward rushing (passing efficiency drops, teams lean run; `temperature_f` continuous captures this regime).
- **No QB signal** — `roof_dome` + `implied_team_total` already proxy known weather risk for outdoor games; marginal lift from explicit wind/temp over those proxies is below the per-cell noise floor for QBs.
- **No TE signal** — likely sample-size driven (n_paired = 3,975 TE vs 5,273 RB / 8,470 WR). Not a clean mechanism rejection; could be revisited under a TE-specific refined-unit plan.

**Refined-unit candidates left unexplored** (per spec §1.4 — revisit-only-on-SIGNAL territory; now in scope, none queued): precipitation (would require new ingest, e.g., NOAA hourly historical keyed on stadium lat/lon), kickoff hour / time-of-day (extractable from existing `schedules.kickoff` UTC), `is_cold_weather` boolean (`temp < 32`, sibling to `is_high_wind`), multi-class surface encoding (one bool per surface code), surface × position interactions, per-team weather acclimation, wind direction (would require new ingest). Recommended priority order: cold-weather threshold → multi-class surface → kickoff hour.

**What this closes:** TODO #25's broad-cut weather family at the in-builder unit, on the RB + WR cells. QB + TE remain DO_NOT_ADOPT at this unit; refined-unit candidates remain open under the same TODO.

**Recommended follow-up:** Combined RB + WR integration plan routed through `LightGBMNbModel` only — the two ADOPT cells share the same model class and binding mode, so a single integration plan can extend `RbFeaturesSchema` + `WrFeaturesSchema` together, plumb the 4 weather cols through `build_rb_features` + `build_wr_features`, and run dual-run gates on `(LightGBMNbModel, RB)` + `(LightGBMNbModel, WR)` in parallel. **Precedent: PR #27 TE trajectory integration shipped a schema change for a non-default model class** while leaving baseline production routing unchanged — the weather plan should follow that pattern. Do NOT extend `QbFeaturesSchema` or `TeFeaturesSchema` in the same plan.

**Reports:** `reports/feature_probe_weather_summary.md` (decision log + per-mode table + mechanism annotation + coverage-relaxation note + follow-up recommendation), `reports/feature_probe_weather_override_audit.md` (override-generation audit), 4 per-(model, mode) .md/.csv files (`feature_probe_weather_{,baseline_,lgbnb_}{augment,swap}.{md,csv}`).

---

## TE Trajectory Features Integration — verdict **ADOPT** on `(LightGBMNbModel, TE)`; shipped (2026-05-04, on branch `feat/te-trajectory-features`)

**Status:** Production integration of the 4 trajectory features into `TeFeaturesSchema` + `build_te_features` per `docs/superpowers/specs/2026-05-04-te-trajectory-features-design.md`. Wired `attach_trajectory_features` into `build_te_features` via the existing `draft_picks` kwarg (plumbed in PR #26). Updated `baseline.py:_TE_FEATURE_COLUMNS` (same spec-gap class as PR #21 / PR #26). Extended shared `_build_position_weekly_stats` + `_build_position_supporting_frames` helpers parametrically so `baseline_weekly_stats_te` covers 17/17/4 weeks for trajectory's 8-game prior window (mirrors PR #26's WR fixture extension; QB/RB unchanged via default).

**Dual-run gate verdict on `(LightGBMNbModel, TE)`:** **`ADOPT`** (composite RMSE delta **-0.0090 fpts**, CI [-0.0171, -0.0013]). Probe predicted -0.0107 fpts; gate matched within ~0.0017 fpts (sharper calibration than PR #26's WR ~0.0043 gap). **First production integration in the project to bind on a non-default model class** (TE production routes to `baseline`; lgb-nb is where the probe's signal lived per PR #25 trajectory probe).

**Per-(model_class, TE) verdicts:**

| Model class | n_paired | RMSE Δ (fpts) | RMSE 95% CI | Spearman Δ | Spearman 95% CI | Verdict |
|---|---:|---:|---|---:|---|:---:|
| baseline | 4257 | -0.0100 | [-0.0280, +0.0093] | +0.0018 | [-0.0033, +0.0071] | DO_NOT_ADOPT (informational) |
| **lightgbm-nb** | 4257 | **-0.0090** | **[-0.0171, -0.0013]** | +0.0028 | [+0.0001, +0.0055] | **ADOPT (binding)** |

**Spec deviation:** spec §1.3.3 called for all 5 model classes; only the binding cell (lgb-nb) and the modified-shape contingency cell (baseline) were evaluated. An earlier `--model all` attempt was aborted after 3 hours wall time with no run dir produced — the per-stat pinball optimizer in EnsembleModel + lightgbm-tuned drove the runtime; the `--model lightgbm-nb`+`baseline` pair completed in ~33 min combined. The skipped cells (lightgbm, lightgbm-tuned, ensemble) are explicitly informational per spec §1.3.4 and not gating per spec §1.3.5; back-fillable by a follow-up backtest if the routing-flip discussion ever needs them.

**Modified-shape contingency:** `(baseline, TE)` returned DO_NOT_ADOPT at -0.0100 fpts with CI bracketing zero — **not REGRESSION**. Spec §1.3.5 modified-shape branch did NOT fire; ship as designed (`_TE_FEATURE_COLUMNS` extension stays in).

**Coverage statistics (eval window 2021-2024):** `age` 94.8%, `is_rookie` 94.8%, `volume_trend_l4_minus_prior_l4` 46.4%, `snap_pct_change_l4_vs_prior_l4` 75.6%. All 4 cols within ~5pp of probe coverage. Age range observed 20.0-40.0 (no clipping of the `ge=15, le=50` bound). Rookie rate ~17% over 7042 rows, consistent with NFL roster turnover.

**Cross-class deferred follow-up:** TE production routing remains on `baseline`. Naively stacking Plan 8's `(lgb-nb, TE)` baseline-vs-lgb-nb gap (+0.0028 fpts) with this PR's measured trajectory lift (-0.0090 fpts) suggests `lgb-nb-with-trajectory ≈ -0.0062 fpts` vs baseline-without-trajectory at the position level — small directional improvement, CI would likely bracket zero. A fresh cross-class re-eval is the right shape for that question; not load-bearing for any current consumer; queued under TODO #24.

**What this closes:** TODO #24's TE-cell branch at the trailing-8-game unit. Combined with PR #26's WR integration, the trailing-8-game-unit branch is now closed at all three of PR #25's ADOPT cells (WR baseline, WR lgb-nb, TE lgb-nb). Refined-unit candidates remain unexplored: per-position aging-curve interactions (`age²`), `is_2nd_year` / `is_3rd_year` flags, depth-chart-rank trends, longer trailing windows (l8 vs l16), `has_trajectory_history` indicator.

**Spec gaps caught + fixed during execution:**

- **Plan-vs-precedent inconsistency** (Task 2). The plan text said pass `ws` / `sc` (prior-mask-filtered frames) to `attach_trajectory_features`; the implementer correctly diverged and passed full `weekly_stats` / `snap_counts`, matching PR #26's `build_wr_features` precedent. The helper's rolling helpers do their own `.shift(1)` leakage shifting; passing prior-mask-filtered frames would strip the current-week index row and the merge would resolve all NaN. Documented in the commit message + inline comment.
- **PYTHONPATH workaround for the worktree's editable-install asymmetry.** Subagent reported the `.venv` resolves `projections` from main repo's `src/`, not the worktree's. Worktree-side scripts that read modified schemas need `PYTHONPATH=src` set explicitly. Not committed (environment concern, not code); applied to all real-data execution invocations in Phase 4.
- **Plan column-count drift:** plan's pre-task TE schema column count (21) was stale by 2 (Plan 3b rushing cols); actual was 23 → 27. Cosmetic, no behavior impact.

See `reports/te_trajectory_features_summary.md` for the full decision log + per-mode table + per-year breakdown + probe-vs-gate calibration.

---

## WR Trajectory Features Integration — verdict **ADOPT** on `(BaselineModel, WR)`; shipped (2026-05-04, on branch `feat/wr-trajectory-features`)

**Status:** Production integration of the 4 trajectory features into `WrFeaturesSchema` + `build_wr_features` per `docs/superpowers/specs/2026-05-03-wr-trajectory-features-design.md`. Promoted `_build_draft_lookup` from override-script-private to public `build_draft_lookup` in `src/projections/features/trajectory_features.py`. Wired `attach_trajectory_features` into `build_wr_features` with the new `draft_picks` kwarg; added the same kwarg to QB/RB/TE builders for plumbing symmetry (unused there, mirroring the existing `pbp` precedent). Updated `baseline.py:_WR_FEATURE_COLUMNS` (the spec gap PR #21 caught at `9895dee`). All 4 caller scripts (`refresh_features.py`, `train_baseline.py`, `predict_2024.py`, `sanity_check_baseline.py`) load + thread `draft_picks`. Full pytest suite (838 passed, 17 skipped); mypy strict + ruff + ruff format clean.

**Dual-run gate verdict on `(BaselineModel, WR)`:** **`ADOPT`** (composite RMSE delta **-0.0371 fpts**, CI [-0.0567, -0.0172]). Probe predicted -0.0414; gate matched the direction and landed within the probe CI (~0.004 fpts smaller magnitude — bootstrap noise, not a real divergence). **First production-builder integration since PR #21** (RB PBP cols).

**Per-(model_class, WR) verdicts** (informational per spec §1.3.5; binding cell is baseline):

| Model class | RMSE Δ | 95% CI | Spearman Δ | Verdict |
|---|---:|---|---:|:---:|
| **baseline**       | **-0.0371** | **[-0.0567, -0.0172]** | +0.0047 | **ADOPT (binding)** |
| lightgbm           | -0.0207 | [-0.0289, -0.0121] | +0.0026 | ADOPT |
| lightgbm-tuned     | +0.0025 | [-0.0056, +0.0106] | +0.0014 | DO_NOT_ADOPT |
| lightgbm-nb        | -0.0171 | [-0.0269, -0.0071] | +0.0020 | ADOPT |
| ensemble           | -0.0242 | [-0.0351, -0.0138] | +0.0019 | ADOPT |

4 of 5 model classes ADOPT. lgb-nb cell at -0.0171 cross-checks the probe's second WR ADOPT cell (probe predicted -0.0194). lightgbm-tuned is the sole DO_NOT_ADOPT (point estimate near zero) — consistent with TODO #29's pruning candidate framing. Pre-existing `(BaselineModel, WR)` per-year breakdown: 2021 -0.0553 (CI strictly negative), 2022 -0.0295 (CI brackets zero), 2023 -0.0397 (CI strictly negative), 2024 -0.0233 (CI brackets zero); pooled CI strictly negative because every year-point estimate is negative.

**Spec gaps caught + fixed during execution:**

- **Helper-input contract (commit `d1b3092`).** Spec §1.1 Task 5 + §2.3 instructed passing prior-mask-filtered `ws`/`sc` to `attach_trajectory_features`. The helper's internal `_volume_trend` and `compute_snap_pct_change` already use `.rolling(4).mean().shift(1)` for leakage safety — double-filtering produced 100% NaN trend cols. Fix: pass full unfiltered frames; existing 5 WR leakage tests confirm leakage safety. Direct regression test added at `a742d83` asserting `volume_trend_l4_minus_prior_l4 == 2.0` for a hand-computed Jefferson scenario.
- **Three Cluster A test-fixture leftovers** caught + fixed at commits `1f1f415` (`test_cache.py:_minimal_wr_features_row`), `33eea57` (7 lightgbm/ensemble synthetic fixtures special-casing `age` in [22, 30]), `807f046` (`test_tune_lightgbm.py:_WR_FEAT_COLUMNS`). Defense-in-depth grep for `opp_allowed_wr_fppg_l4` confirmed those were the only 3 missed sites.
- **Spec gap from PR #21 already anticipated:** `baseline.py:_WR_FEATURE_COLUMNS` is hardcoded; lightgbm derives from the schema dynamically. Implementation plan called this out as its own task (commit `36313d9`).

**Coverage statistics (2021-2024 eval window):** age 97.2%, is_rookie 97.2%, volume_trend_l4_minus_prior_l4 57.5%, snap_pct_change_l4_vs_prior_l4 73.9% — all within ~5pp of the probe's measured coverage. Gate didn't need a coverage-threshold flag (gate uses row-key matching, not NaN tolerance; the probe's `--coverage-threshold 0.35` is a different concern).

**What this closes:** TODO #24's "trailing-8-game unit" branch of the trajectory candidate. Refined-unit candidates (`age²`, `is_2nd_year` flags, longer trailing windows, depth-chart-rank trends, `has_trajectory_history` indicator) remain unexplored under the same TODO. None queued.

**TE follow-up status:** Per the trajectory probe (PR #25), TE adopted **only** under lgb-nb (-0.0107 fpts), not under BaselineModel. Per-position-routing decision required for any TE integration: either ship per-position routing to lgb-nb for TE only (precedent: Plan 6's QB-only ensemble suggestion), or ship the schema change for the lgb-nb code path while leaving baseline production routing unchanged. Not queued.

See `reports/wr_trajectory_features_summary.md` for the full decision log + per-mode table + probe-vs-gate calibration + per-year breakdown.

---

## Trajectory Family Probe — verdict **SIGNAL** via WR (both model classes) + TE (lgb-nb only) (2026-05-03, on branch `feat/probe-trajectory`)

**Status:** Probe-only spec shipped per `docs/superpowers/specs/2026-05-03-trajectory-feature-family-probe-design.md` and plan `docs/superpowers/plans/2026-05-03-trajectory-feature-family-probe.md`. Implements 4 pure compute fns (`compute_age`, `compute_is_rookie`, `compute_qb_volume_trend`, `compute_rb_volume_trend`, `compute_wr_te_volume_trend`, `compute_snap_pct_change`), `_volume_trend` shared helper, `attach_trajectory_features` joiner, public assembler `build_trajectory_overrides` in `src/projections/features/trajectory_features.py`, override-generator script `scripts/build_trajectory_override.py`, plus the new `refresh_draft_picks` ingest module and `DraftPicksSchema`. All tests pass; mypy strict + ruff + ruff format clean. CONTRIBUTING.md updated with a "Regenerating the trajectory override" subsection.

**Verdict:** **`SIGNAL` (durable)** per spec §1.3 criterion 3 — both BaselineModel + lgb-nb tested at composite via `--force-composite`. **First SIGNAL family probe since PR #20** (PBP team features bundle, RB-only). PR #22, #23, #24 all returned NULL durable.

**ADOPT cells (3 of 16):**
- **WR augment baseline** — composite RMSE Δ **-0.0414 fpts** ([-0.0606, -0.0230]); Spearman Δ +0.0058 ([+0.0026, +0.0092]).
- **WR augment lgb-nb (composite, forced)** — composite RMSE Δ **-0.0194 fpts** ([-0.0299, -0.0096]); Spearman Δ +0.0031 ([+0.0012, +0.0051]).
- **TE augment lgb-nb (composite, forced)** — composite RMSE Δ **-0.0107 fpts** ([-0.0191, -0.0028]); Spearman Δ +0.0032 ([+0.0004, +0.0063]).

WR is the dominant signal carrier: ADOPT under both model classes. TE is lgb-nb-only. RB is null (mechanism-consistent — RB rushing is more team-script-driven than career-arc-driven; PR #21's team-level PBP cols already covered RB's binding axis). QB is **regression on both model classes** — see "recurring QB augment regression" below.

**Bundle definition.** Four player-level features: `age` (biological age = `draft_age + (season - draft_year)` from `nfl_data_py.import_draft_picks`, with UDFA / pre-1980 fallback `season - inferred_draft_year + 22.0` where `inferred_draft_year` is earliest weekly_stats appearance — fallback fired on **22.6%** of rows), `is_rookie` (1.0 if `season == draft_year`), `volume_trend_l4_minus_prior_l4` (trailing-4 minus prior-4 mean on position-tailored stat: QB attempts / RB carries / WR/TE targets), `snap_pct_change_l4_vs_prior_l4` (same window cut on `SnapCountsSchema.offense_pct`). Plus audit-only `draft_year_inferred` boolean. Override: `data/features_probe/trajectory.parquet`, 56,652 rows.

**Coverage relaxation — major spec deviation, documented prominently.** All four probes invoked with `--coverage-threshold 0.35` (vs spec's 0.95 default; vs PR #22's 0.70 fallback for 2018-only structural cold-start). Trajectory's trend features are **structurally sparse — they require 8 prior active games per player**, which excludes ~50% of player-weeks across all years (rookies, returners, mid-season call-ups). Per spec §3.3: "fewer than 8 prior active games yields NaN." Per-position coverage of override-candidate columns vs baseline rows: QB (88.7% age / 37.8% volume_trend / 39.6% snap_pct), RB (96.6 / 53.7 / 66.6), WR (96.7 / 53.6 / 68.4), TE (95.4 / 44.7 / 71.1). This is **NOT silent NaN imputation** — the probe's left-merge produces NaN-on-NaN matches that are excluded from the per-paired-row delta, so the bias is symmetric on both baseline and candidate sides; the verdict applies to the surviving substantially-experienced-player subpopulation, which is also where these features have any defined value to begin with. Deepest threshold relaxation in Track 2A's history; future re-test or production-builder work must apply the same threshold (or scope the cohort explicitly).

**Recurring QB augment regression — now 4 instances across model classes.** This bundle adds **two new instances** to a growing pattern:

| PR | Family | Model class | Mode | RMSE Δ | CI |
|---|---|---|---|---:|---|
| #23 | red-zone | lgb-nb | augment | **+0.0268** | [+0.0082, +0.0449] |
| #24 | pressure | lgb-nb | augment | **+0.0276** | [+0.0077, +0.0472] |
| this | trajectory | **baseline** | augment | **+0.0382** | [+0.0155, +0.0600] |
| this | trajectory | lgb-nb | augment | **+0.0233** | [+0.0068, +0.0388] |

The pattern previously appeared only on lgb-nb; trajectory's bundle produces it on **both** Ridge baseline and lgb-nb composite. Suggests adding context features (team / pressure / trajectory) to QB inputs consistently overfits on augment configurations across model-class spectrum. Possibly because the QB v1 feature set is already small (~25 cols vs ~40+ for skill positions), so any bundle of 4 added cols disproportionately shifts the regularization posture. Worth flagging for any future QB feature work.

**What this closes:** TODO #24's age + role-trajectory candidates at the trailing-8-game unit. The bundled probe carried clear orthogonal signal at the WR (both model classes) and TE (lgb-nb only) cells. Greenlights a follow-up integration plan analogous to PR #20 → PR #21.

**Refined-unit candidates beyond trailing-8-game unit remain unexplored:** per-position aging-curve interaction terms (`age²` for older-RB drop), `is_2nd_year` / `is_3rd_year` flags (collinear with age but might unlock breakout-year signal), depth-chart-rank trends, longer trailing windows (l8 vs l16), treating sparsity as a feature (a `has_trajectory_history` indicator that flips on at game 8+, allowing the model to learn to use trajectory signals only when well-defined). None queued.

**Reports:** `reports/feature_probe_trajectory_summary.md` (decision log + per-mode table + mechanism annotation + coverage-relaxation note) + 4 per-(model, mode) .md/.csv files (`feature_probe_trajectory_{,lgbnb_}{augment,swap}.{md,csv}`).

---

## PBP Pressure Family Probe — verdict NULL (durable) at dropback-denominator cut, all 4 positions (2026-05-02, on branch `feat/probe-pbp-pressure`)

**Status:** Probe-only spec shipped per `docs/superpowers/specs/2026-05-02-pbp-pressure-feature-family-probe-design.md` and plan `docs/superpowers/plans/2026-05-02-pbp-pressure-feature-family-probe.md`. Implements 4 pure compute fns + `_per_game_rate` helper + attach helper + public assembler in `src/projections/features/pbp_pressure_features.py`, the override-generator script `scripts/build_pbp_pressure_override.py`, 19 synthetic-fixture tests + 4 CLI tests. mypy strict + ruff + ruff format clean. CONTRIBUTING.md updated with a "Regenerating the PBP pressure override" subsection.

**Verdict:** `NULL` (durable per spec §1.3 criterion 3 — both BaselineModel + lgb-nb tested at composite via `--force-composite`).

- All 4 mode × model reports (baseline + lgb-nb × augment + swap, 4 positions, 6 stats each = 120 pooled cells per mode × 4 modes = 480 total): 0 pooled Phase 1 SIGNAL.
- Phase 2 (composite, lgb-nb composite via `--force-composite`) on both modes: 0 ADOPT or MARGINAL across 8 verdict cells. All 8 DO_NOT_ADOPT.
- Two directional Phase-2 cells: **QB augment lgb-nb composite RMSE REGRESSION** at +0.0276 fpts CI [+0.0077, +0.0472] (CI strictly above 0 — same pattern as PR #23's red-zone QB augment regression at +0.0268); and **TE swap lgb-nb composite Spearman regression** at -0.0034 CI [-0.0067, -0.0002] (rank only; RMSE brackets zero so verdict is DO_NOT_ADOPT, not REGRESSION). The other 6 Phase 2 cells all bracket zero with point estimates near zero.
- Predicted mechanism (QB-side pressure exposure → `passing_yards` / `sacks` / `rushing_yards`) **not observed** — no QB-side cell fires SIGNAL anywhere; the only QB Phase 2 directional movement is in the wrong direction.

**What this closes:** TODO #3c's "Pressure rate allowed by O-line" candidate, at the dropback-denominator cut (`qb_dropback == 1`). The four bundled features do not carry orthogonal signal beyond v1 (or v1 + already-shipped PBP team features for RB) under either Ridge baseline or lgb-nb production model class.

**This closes Track 2A — all three TODO #3c team-level PBP families have now been probed:**
- PR #20 (pace/PROE/AYPS/EPA-resid bundle) — SIGNAL via RB; integrated into RB schema in PR #21 (-0.0124 fpts adoption gate).
- PR #23 (red-zone bundle) — durable NULL.
- PR #24 (this — pressure bundle) — durable NULL.

**Refined-unit candidates beyond `qb_dropback == 1` remain unexplored** but are unlikely to clear what the broad cut couldn't, absent independent evidence: alternate denominators (`pass_attempts + sacks` only), goal-line / 3rd-down / two-minute pressure subsets. None queued.

**Coverage:** default `--coverage-threshold 0.95` passed cleanly (pooled 96.6%; 2019–2024 uniformly 100%; 2018 cold-start 24.2% NaN, but eval window 2021–2024 unaffected). No threshold relaxation needed (PR #22 used 0.70, PR #23 used 0.90 — this probe didn't need either).

**Carry-over follow-ups noted by reviewers** (low-priority cleanup PR candidates, none blocking):
- Triplicate `_read_concat`, `_FANTASY_POSITIONS`, `_build_player_team_week_index`, `_parse_season_range` across 3 override scripts (PR #20, PR #23, PR #24) — extract to `scripts/_pbp_override_common.py`.
- `_trailing_4_mean` triplicated across 3 sibling feature modules — extract to `src/projections/features/_pbp_common.py` (could pair with the script-level extraction).
- Drop the redundant `& pbp[denom_col].notna()` clause in `_per_game_rate` (defensively redundant; `NaN == 1.0` is False).
- CLI test `test_main_writes_output` doesn't actually exercise the prior-season backfill path; the unit-level cross-season test covers the underlying invariant.

**Reports:** `reports/feature_probe_pbp_pressure_summary.md` (decision log + per-mode table + mechanism annotation) + 4 per-(model, mode) .md/.csv files (`feature_probe_pbp_pressure_{,lgbnb_}{augment,swap}.{md,csv}`).

---

## Track 2B — RB PBP cols × other model classes — directional improvement, no regression (2026-05-03, folded into branch `feat/probe-pbp-pressure`)

**Status:** Informational dual-run gate per the user's request to fold Track 2B into PR #24. Compared two `--model all` backtest runs: pre-PR-20 baseline (`run_20260429T003552Z`) vs post-PR-21 candidate (`run_20260503T014536Z` — generated today on the worktree's HEAD). Per-`(model_class, position)` paired-row RMSE delta + 1000-bootstrap CI on `(gsis_id, season, week, position)` keys. Reports the magnitude of the lift the lightgbm family (which auto-derives features from the schema dynamically) sees from the 4 RB PBP cols PR #21 shipped.

**Verdict:** **Directional improvement on RB across all 5 model classes; no regression on any cell.**

| Model class | RB RMSE Δ | 95% CI | Strictly negative? |
|---|---:|---|:---:|
| baseline | -0.0124 | [-0.0258, -0.0002] | **yes** (matches PR #21's -0.0124 to 4 decimals — methodology check) |
| lightgbm | -0.0141 | [-0.0278, +0.0005] | no (just barely brackets 0 on upper bound) |
| lightgbm-tuned | -0.0101 | [-0.0225, +0.0015] | no |
| lightgbm-nb | -0.0075 | [-0.0203, +0.0041] | no |
| ensemble | -0.0062 | [-0.0177, +0.0054] | no |

QB/WR/TE: zero or trivially-near-zero deltas across all 5 model classes (their schemas weren't touched in PR #21).

**Interpretation:** the per-feature signal from the 4 RB PBP cols progressively diffuses as model complexity grows (Optuna tuning → NB-2 dispersion → ensemble weighting). All point estimates are favorable, but only the simplest cells (`baseline`, `lightgbm`) reach strict statistical significance. No tree class hurts on RB. Adopting the cols system-wide (which the lightgbm family already does automatically) is at minimum neutral and at maximum slightly helpful.

**Probe approach abandoned** — the feature signal probe (`scripts/probe_feature_signal.py`) does NOT support the "drop existing cols, re-add via override" pattern that retrospective gating of already-shipped features would require. The probe applies `--drop` symmetrically and explicitly excludes override cols matching the drop list. Two abandoned probe attempts (`reports/track2b_rb_pbp_lgbnb_drop.md` and `reports/track2b_rb_pbp_lgbnb.md`) are preserved as record-of-experiment + as a forward-pointer for "do not use the probe for retrospective gating; use dual-run backtest instead."

**Limitations:** the two backtest runs were generated 4 days apart on slightly different code (post-run includes PR #22 + #23 + #24's spec/plan/code, but those don't touch RB feature builder or the lightgbm models). Same-commit dual-run with manipulated feature parquets would be more hermetic, but it's not worth the infrastructure cost for an informational pass that produces a clean directional signal here.

**Reports:** `reports/track2b_rb_pbp_other_models.md` (decision log + per-cell table + methodology + abandoned-probe appendix).

---

## PBP Red-Zone Family Probe — verdict NULL (durable) at RZ-broad cut, all 4 positions (2026-05-02, on branch `feat/probe-pbp-redzone`)

**Status:** Probe-only spec shipped per `docs/superpowers/specs/2026-05-02-pbp-redzone-feature-family-probe-design.md` and plan `docs/superpowers/plans/2026-05-02-pbp-redzone-feature-family-probe.md`. Implements 4 pure compute fns + attach helper + public assembler in `src/projections/features/pbp_redzone_features.py`, the override-generator script `scripts/build_pbp_redzone_override.py`, 19 synthetic-fixture tests + 4 CLI tests. mypy strict + ruff + ruff format clean. CONTRIBUTING.md updated with a "Regenerating the PBP red-zone override" subsection. .gitignore extended with `.claude/` (small chore folded into the branch).

**Verdict:** `NULL` (durable per spec §1.3 criterion 3 — both BaselineModel + lgb-nb tested at composite via `--force-composite`).

- All 4 mode × model reports (baseline + lgb-nb × augment + swap, 4 positions, 6 stats each = 96 pooled cells per mode × 4 modes): 0 pooled Phase 1 SIGNAL.
- Phase 2 (composite, lgb-nb composite via `--force-composite`) on both modes: 0 ADOPT or MARGINAL across 8 verdict cells. All 8 DO_NOT_ADOPT.
- Only directional Phase-2 cell: `QB augment lgb-nb composite RMSE +0.0268 fpts CI [+0.0082, +0.0449]` — strictly above 0, i.e. small but statistically-significant **regression** on QB when adding the 4 RZ features in augment mode under lgb-nb composite. The other 7 Phase 2 cells all bracket zero with point estimates near zero (RB/WR/TE flat).
- Predicted mechanism (TD efficiency) **not observed** — no `*_tds` cell fires SIGNAL anywhere. The bundle was chosen specifically to drive `*_tds` (Plan 5c's noisiest unmoved cell) and didn't.

**What this closes:** TODO #3c's "Red-zone usage shares (separate from full-field share)" candidate, at the team-level RZ-broad cut (`yardline_100 ≤ 20`). The four bundled features (`team_rz_pace_l4`, `team_rz_pass_rate_l4`, `team_def_rz_epa_allowed_l4`, `team_def_rz_pass_rate_allowed_l4`) do not carry orthogonal signal beyond v1 + already-shipped PBP team features under either Ridge baseline or lgb-nb production model class.

**Refined-unit candidates beyond `yardline_100 ≤ 20` that remain unexplored:** goal-line (`yardline_100 ≤ 5`), per-stat splits (un-bundling reverses the family-level prior framework), RZ-restricted EPA-residual (correlates with already-shipped full-field `team_def_epa_resid_l4`). None queued; the durable NULL across both model classes argues against any of these clearing what RZ-broad couldn't, absent independent evidence.

**What remains open in TODO #3c:** **Pressure rate allowed by O-line** — the third unexplored team-level family from TODO #3c. Different mechanism axis (offensive-line proxy via sack rate / scramble rate, not TD distribution). Curated PBP has `sack`, `qb_dropback`, `qb_scramble` so a pressure-proxy bundle is buildable without ingest extension. Natural next slot for a bundled probe under the same workflow.

**Coverage caveat:** Probe invoked with `--coverage-threshold 0.90` because the probe's hardcoded check is *pooled* and the structural 2018 cold-start (no Y-1 backfill, 73-77% per-position coverage) drags pooled to 94.7%. Per-season 2019–2024 coverage is uniformly ≥96.9% across all 4 positions — the eval window itself satisfies spec §1.3 criterion 1 ("≥95% per (position, season) pair"). Same precedent gap as PR #22 (which used 0.70). Not blocking; documented in summary report.

**Spec corrections caught + fixed during planning** (3 commits on the branch before implementation): §2.2 source switched from `weekly_stats` to `depth_charts` (matches PR #20's actual implementation; weekly_stats would produce ~50% coverage gaps from missing inactive-roster rows); §6.2 `read_pbp` (fictional function) replaced with `store.read_partition` per-season + `pd.concat` pattern; §6.4 CLI tests reframed as new coverage rather than "mirroring PR #20" (PR #20 shipped without CLI tests). Probe-output file count corrected from 16-32 files to 4-8 (probe writes one .csv + one .md per mode with all positions in long format, not per-position).

**Reports:** `reports/feature_probe_pbp_redzone_summary.md` (decision log + per-mode table + mechanism annotation) + 4 per-(model, mode) .md/.csv files (`feature_probe_pbp_redzone_{,lgbnb_}{augment,swap}.{md,csv}`).

---

## PBP Receiver Family Probe — verdict NULL (durable) for WR/TE; family closed at air-yards / aDOT cut (2026-05-01, on branch `feat/wr-te-pbp-features`)

**Status:** Probe-only spec shipped per `docs/superpowers/specs/2026-05-01-wr-te-pbp-receiver-features-design.md` and plan `docs/superpowers/plans/2026-05-01-wr-te-pbp-receiver-features.md`. Implements `_trailing_4_per_player_asof` as-of helper + 4 pure compute fns + assembler + validating wrapper in `src/projections/features/pbp_receiver_features.py`, the override-generator script `scripts/build_pbp_receiver_override.py`, 21 synthetic-fixture tests + 4 CLI tests. mypy strict + ruff + ruff format clean.

**Verdict:** `NULL` (durable per spec §1.3 criterion 3 — both BaselineModel + lightgbm-nb tested at composite via `--force-composite`).

- All 4 mode × model reports (baseline+lgb-nb × augment+swap, 2 positions, 12 stats each = 96 pooled cells): 0 pooled Phase 1 SIGNAL.
- Phase 2 (composite, lgb-nb production model) on both modes: 0 ADOPT or MARGINAL across 8 verdict cells. All 8 DO_NOT_ADOPT.
- Closest to SIGNAL: `WR swap composite RMSE -0.0052 fpts CI [-0.0118, +0.0013]` — point estimate favorable but CI brackets zero by +0.001 fpts.
- Per-year REGRESSION on TE `receiving_yards` 2022 (~+0.15 fpts in augment mode) replicates across baseline + lgb-nb — informational; pooled bootstrap correctly washes it out.

**What this closes:** TODO #3c's WR/TE refined-unit follow-up at the air-yards / aDOT cut. The four player-level PBP features tested (`aDOT_l4`, `deep_target_share_l4`, `yac_per_reception_l4`, `red_zone_target_share_l4`) do not carry orthogonal signal for receiver fantasy projections at the trailing-4-receiver-active-games unit, under either Ridge baseline or lgb-nb production model class.

**Refined-unit candidates beyond air-yards / aDOT that remain unexplored:** per-route-concept distributions (data not in curated PBP), target-quality residuals (per-throw difficulty modeling not in PBP), in-line vs flexed alignment for TE (not in PBP). None queued; revisit only with independent evidence (published study, third-party benchmark) suggesting the unit choice was the binding constraint.

**What remains open in TODO #3c:** Other PBP feature families at the *team* level not in PR #20's bundle — pressure rate allowed by O-line, red-zone usage shares (separate from this spec's receiver-level RZ target share), team pace alone (vs the bundled probe). Bundle 3-4 candidates per probe per the family-level prior.

**Spec gap (caught + worked around, not fixed in spec):** Spec §3.2 prescribed lgb-nb runs without `--force-composite`, but Phase 1 is RidgeCV-only regardless of `--model`, so bare lgb-nb runs are tautological with baseline. Re-ran with `--force-composite` to actually test lgb-nb at composite. The same gap exists in PR #20's spec §3.2; not exercised then because RB returned baseline SIGNAL and the conditional path never fired. Future probe-with-conditional-lgb-nb specs should include `--force-composite`.

**Coverage caveat:** 2018 coverage is structurally low (~50% non-null) because the curated PBP window starts in 2018 with no Y-1 backfill. 2019-2024 coverage is 75-87% per (position, season). Probe invoked with `--coverage-threshold 0.70`. The surviving-receiver row set is biased toward heavier-targeted WRs/TEs; bias is consistent across baseline + candidate sides so the comparison stays valid for the substantially-targeted subset.

**Reports:** `reports/feature_probe_pbp_receiver_summary.md` (decision log + addendum on the `--force-composite` spec gap) + 4 per-(model, mode) .md/.csv files (`feature_probe_pbp_receiver_{,lgbnb_}{augment,swap}.{md,csv}`).

---

## RB PBP Features Integration — verdict ADOPT on (BaselineModel, RB); shipped (2026-05-01, on branch `feat/rb-pbp-features`)

**Status:** Shipped per spec `docs/superpowers/specs/2026-05-01-rb-pbp-features-design.md` and plan `docs/superpowers/plans/2026-05-01-rb-pbp-features.md`. Extends `RbFeaturesSchema` with 4 nullable-float cols (`pace_l4`, `proe_l4`, `team_ayps_l4`, `team_def_epa_resid_l4`) and wires `attach_pbp_family_features` (extracted from `build_pbp_family_overrides` in PR #20) into `build_rb_features`. RB feature cache refreshed for 2018–2024 (151 partitions); full backtest + dual-run gate clean on the binding `(BaselineModel, RB)` cell; mypy strict + ruff + ruff format clean.

**Verdict:** `(BaselineModel, RB)` = `ADOPT`. Composite RMSE delta **-0.0124 fpts** (95% CI [-0.0255, -0.0006]); Spearman delta +0.0021 (CI [-0.0003, +0.0044]). The probe (PR #20) predicted -0.0124 fpts CI [-0.0249, -0.0001] in augment mode — the production gate matched the point estimate to 4 decimal places.

**Per-year breakdown (informational):** improvement concentrated in 2021 (-0.028 fpts, CI [-0.053, -0.005]) and 2022 (-0.021 fpts, CI [-0.046, +0.004]); 2023 flat (-0.004); 2024 mildly positive (+0.005, CI brackets zero). Pooled CI strictly negative because 2021–22 dominate the sample.

**Other model classes (informational, not gating):** `lightgbm-tuned`, `lightgbm-nb`, `ensemble` deferred to a follow-up — Optuna + NB-2 fitting are 1+ hour each, and per spec §1.3.5 only `(BaselineModel, RB)` binds the ship/revert decision. The lightgbm family auto-picks-up the 4 new cols (it derives features from the schema dynamically); no model-layer change needed there.

**Spec gap caught + fixed during execution:** the spec did not call out updating `src/projections/models/baseline.py:_RB_FEATURE_COLUMNS`, a hardcoded tuple the BaselineModel reads to know which schema cols to feed Ridge. Without the update, the gate measured all-zero deltas (candidate predictions identical to baseline). Fixed at commit `9895dee`. Future "add feature to position X" specs should include a model-feature-list checklist item.

**What this closes:** Spec §1.3 criteria 1–5 for RB; TODO #3c's RB-only sub-question. WR / TE remain open for a separate refined-unit spec (player aDOT for receivers, per-position EPA-residual à la Plan 9). QB explicitly excluded — PR #20's augment-mode `passing_yards` regression (+0.45 fpts) means PBP team-features do not transfer to QB at the team-level granularity.

**Reports:** `reports/rb_pbp_features_summary.md` (decision log + probe-vs-gate calibration table) + `reports/adoption_gate_rb_pbp_features_baseline.{md,csv}` (per-(model_class, RB) gate output).

---

## PBP Feature Family Probe — verdict SIGNAL via RB; greenlights production-builder follow-up (2026-04-30, on branch `feat/probe-pbp-family`)

**Status:** First family-level probe shipped per spec `docs/superpowers/specs/2026-04-30-pbp-feature-family-probe-design.md` and plan `docs/superpowers/plans/2026-04-30-pbp-feature-family-probe.md`. Implements 4 pure compute fns + assembler in `src/projections/features/pbp_team_features.py`, the `family_verdict_from_reports` helper in `src/projections/backtest/feature_probe.py`, and the `scripts/build_pbp_family_override.py` CLI. 14 synthetic-fixture tests pass; mypy strict + ruff + ruff format clean on touched files.

**Verdict:** `SIGNAL`. Phase 1 pooled SIGNAL on `(RB, rushing_yards)` in both augment and swap modes; Phase 2 ADOPT on RB in both modes (composite RMSE delta ~-0.012 fpts, CI strictly below 0). Signal concentrated on RB. QB regresses on `passing_yards` (+0.45 fpts) in augment mode; WR/TE net-zero. Conditional lgb-nb runs skipped per spec §3.2 (baseline already returned SIGNAL).

**What this greenlights:** A follow-up production-builder plan that integrates the PBP family into RB feature builders. RB-only first; verify which subset of the 4 features carries the rushing-yards signal under the production model class (ensemble Model D); WR/TE deferred to a separate spec with refined per-position units (player aDOT for receivers, per-position EPA-residual à la Plan 9). Do NOT integrate into QB without a swap-mode gate run.

**What this closes:** TODO #3c's open question about the 5–15% family-level prior. The prior is real for RB; not for QB/WR/TE at the team-level granularity tested here. Single-feature follow-ups (e.g., `pace_l4` alone) and refined-unit follow-ups (player aDOT, per-position EPA-resid) remain open as separate specs.

**Reports:** `reports/feature_probe_pbp_family_summary.md` (decision log) + 4 per-mode .md/.csv files (`feature_probe_pbp_family_{augment,swap}.{md,csv}`).

---

## Plan 9 retro option C — EPA-residual on lightgbm-nb — DO_NOT_ADOPT (4/4 positions) (2026-04-30, on branch `feat/probe-plan9-tree-models`)

**Status:** probe extended with `--force-composite` flag; 8 lightgbm-nb composite probes run against the existing Plan 9 override parquets (4 positions × {swap, augment}). All 8 cells DO_NOT_ADOPT. Plan 9's verdict (DO_NOT_ADOPT for the EPA-residual feature on baseline) generalizes to lightgbm-nb. Verification gates green: 51 probe tests pass; mypy / ruff / format clean.

### What shipped

- `scripts/probe_feature_signal.py`: new `--force-composite` flag bypasses Phase 1's pooled-SIGNAL gate so Phase 2 runs unconditionally. Use when `--model` is not the Ridge regressor used in Phase 1 — Phase 2's production-model fit may detect signal Phase 1's Ridge screen missed (e.g., trees on a feature Ridge can't use). Mutually exclusive with `--no-composite`.
- `src/projections/backtest/feature_probe.py`: `ProbeReport` gains a `phase2_skip_reason: str | None` field (`"no_signal"` | `"no_pooled_signal"` | `"user_disabled"` | `None`). Renderer now distinguishes the three skip cases and suggests `--force-composite` when only per-year SIGNAL fired (replaces a pre-existing bug that misattributed the skip to `--no-composite` regardless of cause).
- `_loosened_features_schema` extended: not only `Config.strict = False` (extras pass) but also marks all declared columns `required=False` (missing-after-`--drop` columns pass). Load-bearing for swap-mode Phase 2 — without it, the model's own `feature_schema.validate(features)` errored on every dropped column. Latent bug in the probe; surfaced once `--force-composite` made Phase 2 reachable in swap mode.
- Windows utf-8 stdout reconfigure at `main()` entry so `Δ` and `—` survive shell redirection. Pre-existing latent issue for the same reason as above.
- Test coverage: 8 new CLI tests (4 parse_args, 1 main integration, 3 render); 1 new feature_probe test for the schema fix. All 51 probe tests pass.
- 8 new reports under `reports/feature_probe_plan9_lgbnb_{swap,augment}_{QB,RB,WR,TE}.{md,csv}`.

### Probe verdicts (Plan 9 baseline retro vs lightgbm-nb composite)

`paired_bootstrap_rmse_delta`, n_bootstrap=1000, seed=42. Pairing key `(gsis_id, season, week, position)`. Plan 9 baseline column = adoption gate output from spec §6 (full backtest); lightgbm-nb column = `--force-composite` probe (Phase 2 only, walk-forward fit per holdout year on the same override parquets).

| Position | Mode | Plan 9 baseline RMSE Δ (95% CI) | lightgbm-nb RMSE Δ (95% CI) | lightgbm-nb verdict | Direction |
|---|---|---|---|---|---|
| QB | swap | +0.0005 ([-0.0125, +0.0144]) | +0.0004 ([-0.0139, +0.0159]) | DO_NOT_ADOPT (null) | Same (null) |
| QB | augment | +0.0013 (probe predict; null) | +0.0110 ([-0.0003, +0.0227]) | DO_NOT_ADOPT (borderline regression) | Worse |
| RB | swap | +0.0001 ([-0.0110, +0.0111]) | +0.0074 ([-0.0000, +0.0155]) | DO_NOT_ADOPT (borderline regression) | Worse |
| RB | augment | -0.0193 (probe predict; null) | +0.0042 ([-0.0021, +0.0116]) | DO_NOT_ADOPT (null) | Worse but null |
| TE | swap | -0.0037 ([-0.0121, +0.0050]) | +0.0083 ([+0.0008, +0.0161]) | DO_NOT_ADOPT (REGRESSION) | Worse, regression |
| TE | augment | +0.0045 (probe predict; null) | +0.0036 ([-0.0020, +0.0092]) | DO_NOT_ADOPT (null) | Same (null) |
| WR | swap | +0.0083 ([+0.0043, +0.0124]) (regression) | +0.0011 ([-0.0039, +0.0059]) | DO_NOT_ADOPT (null) | **Better** (regression → null) |
| WR | augment | +0.0140 (probe predict; null) | +0.0053 ([+0.0010, +0.0094]) | DO_NOT_ADOPT (REGRESSION) | Worse, regression |

### Findings

- **No (position, mode, model_class) cell ADOPTs.** EPA-residual feature does not clear the adoption gate at any tested model class. Plan 9's overall verdict is robust.
- **"Trees extract more signal" hypothesis is wrong.** lightgbm-nb does not systematically beat baseline on the same feature swap. The mechanism interpretation from Plan 9 (Ridge baseline is feature-saturated; opp signal is partially captured by `implied_team_total`, `spread`, NGS metrics, v1 fppg) extends to lightgbm-nb — the marginal information in EPA-residual is below the per-cell noise floor for both model classes.
- **Model class change reshuffles the noise without unlocking adoption.** Position-specific patterns are mixed: WR swap improves (regression → null) but WR augment worsens (null → regression); TE swap worsens (null → regression); QB swap is ~unchanged. The directional changes are within the per-cell noise floor (~0.08 fpts per Plan 8) so they don't represent systematic effects of model class.
- **No production routing change.** Plan 8's production defaults stand: QB→lightgbm-nb, RB→baseline, TE→baseline, WR→ensemble. The EPA-residual feature was reverted in Plan 9 and that decision is reaffirmed here.

### Decision-log entries

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-30 | Add `--force-composite` flag to the probe CLI | Phase 1 is hardcoded to RidgeCV; without a way to bypass Phase 1's pooled-SIGNAL gate, Phase 2's `--model lightgbm-nb` is unreachable on candidates where Ridge sees no signal — exactly the case option C wanted to test. The flag preserves the default gating; it's opt-in for non-Ridge model class evaluation. |
| 2026-04-30 | EPA-residual feature is closed across model classes | All 8 (position × mode) cells DO_NOT_ADOPT on lightgbm-nb. Combined with Plan 9's baseline DO_NOT_ADOPT verdicts and the post-Plan-9 augment retros, the feature is dead in 12/12 cells (4 positions × 3 model-class evaluations: baseline-swap, baseline-augment, lightgbm-nb-swap, lightgbm-nb-augment). Don't revisit. |
| 2026-04-30 | Latent probe bugs fixed: `_loosened_features_schema` made all declared columns optional; Windows utf-8 stdout reconfigure | Both bugs were unreachable until `--force-composite` enabled swap-mode Phase 2 against a non-trivial output (Plan 9 retro never fired Phase 2 successfully in swap mode). Surfaced + fixed in the same PR as the flag itself. |

### Next track

The project_management.md "Next track" from the Feature Signal Probe section stands: pivot to TODO #3b/#3c remaining PBP-derived feature candidates (pace, PROE, air-yards distributions, pressure rate allowed, redzone usage shares). Per the Probe section's directive, bundle 3-4 candidates into one probe + adoption gate at the family level — don't probe candidates one at a time.

When designing the bundled probe, **do not gate the next family by the EPA-residual result**: option C's null answer at lightgbm-nb means EPA-residual is closed, but it does NOT mean PBP features generally are. The mechanism (feature saturation in opponent-strength signal) is specific to that feature class.

---

## Feature Signal Probe — shipped (2026-04-30, on branch `feat/feature-signal-probe`)

**Status:** all 5 phases shipped. Final verification gates green: pytest pass, mypy clean, ruff clean. Plan 9 retro validation passes both §1.3 success criteria after a calibration fix (effect-size floor + pooled-only Phase 2 firing).

### What shipped

- `src/projections/backtest/feature_probe.py`: pure-stats module with `PerStatVerdict`, `ProbeReport`, `probe_per_stat`, `probe_composite`, `phase1_should_fire_phase2`, `_build_factory_with_columns`, `_loosened_features_schema`, `_verdict_for_per_stat` (with `effect_size_floor` parameter, default 0.05 fpts). Reuses Plan 8's `paired_bootstrap_*` helpers from `src/projections/backtest/adoption_gate.py` unchanged.
- `scripts/probe_feature_signal.py`: CLI mirroring `scripts/adoption_gate.py`'s shape. Argparse + parse_args + load_features_with_overrides + validate_override_coverage + render_markdown + render_csv + main(). New CLI args: `--coverage-threshold` (default 0.95), `--effect-size-floor` (default 0.05), `--no-composite`.
- Test coverage: 23 unit tests in `tests/test_backtest/test_feature_probe.py` + 23 CLI tests (including 4 integration tests with monkey-patched factories) in `tests/test_scripts/test_probe_feature_signal_cli.py`. No real backtest in the test suite — that's covered by Phase 4 retro.
- Plan 9 retro validation: 8 per-position reports under `reports/feature_probe_plan9_{swap,augment}_retro_{QB,RB,WR,TE}.{md,csv}`. Both criteria from spec §1.3 pass at the pooled level across all 4 positions.

### Calibration fix landed during Phase 4

Initial Plan 9 retro flagged 3 false-positive SIGNAL cells: 1 QB single-year, 1 RB single-year, 1 RB pooled-noise-floor. Two-part fix:

1. **Effect-size floor (default 0.05 fpts)**: SIGNAL/REGRESSION verdicts require `|rmse_delta| >= floor` in addition to the CI strictness condition. Below-floor effects are NULL even when statistically significant. The 0.05 default is conservative vs Plan 8's measured ~0.08 fpts per-cell noise floor.
2. **Pooled-only Phase 2 firing**: `phase1_should_fire_phase2` filters to `year_or_pooled == "pooled"` cells. Per-year SIGNAL cells stay informational — they may flag genuine per-year RMSE improvements that wash out at the pooled level.

Post-fix retro: 0 pooled SIGNAL cells across all 4 positions on the swap retro; criterion 2 passes (no augment row regresses where swap was null). 2 per-year SIGNAL cells preserved as informational (QB passing_yards 2023 = -0.29 fpts; RB rushing_yards 2024 = -0.17 fpts) — real per-year effects that the gate's pooled bootstrap correctly washes out.

### Decision-log entries

- **Probe is the canonical pre-spec screening step for any feature plan.** Before scoping a new feature plan touching feature columns (e.g., the remaining TODO #3b/3c PBP-derived candidates), generate an override parquet and run the probe. If the probe returns no pooled SIGNAL across positions, decompose the plan (bundle multiple candidates, change model class, or shelve) before writing the spec.
- **Probe is not a substitute for the adoption gate.** A pooled SIGNAL verdict + Phase 2 ADOPT prediction are necessary but not sufficient for shipping. Real adoption decisions still go through the full backtest + adoption gate.
- **Augment-not-swap is the safer default.** Plan 9's retrospective showed that swapping (drop + add) can be a strict downgrade for some positions where the dropped column carried orthogonal signal. Augment-only is the safer first probe; swap is a follow-up only when augment SIGNALs.
- **Effect-size floor (0.05 fpts default) prevents practical-significance false positives.** Statistically-significant effects below the noise floor (e.g., -0.001 fpts on n=3000+ paired rows) shouldn't fire Phase 2. Plan 8's measured per-cell noise floor (~0.08 fpts) is the empirical reference.

### Next track

The probe enables disciplined pursuit of TODO #3b (remaining PBP-derived feature candidates: pace, PROE, air-yards distributions, pressure rate allowed, redzone usage shares). Each candidate should be probed before scoping. The 5–15% family-level RMSE prior in TODO #3 should be applied at the family level — bundle 3–4 features into one probe + adoption gate, not one feature at a time.

---

## Plan 9 — PBP ingest + opp-adjusted EPA features — feature change DO_NOT_ADOPT (4/4 positions); plumbing ships (2026-04-29, on branch `feat/plan-9-pbp-ingest-opp-epa`)

**Status:** all 7 phases shipped. Verification gates green: 291+ pytest pass / 9 skipped (network only) / 0 fail; mypy 147 source files clean; ruff check + format clean. Snapshot regression check PASS (400 metrics within tolerance). **Zero per-position routing changes shipped** per the §6 zero-position-adopt branch — feature change reverted at commit `941b96c`. PBP plumbing ships unconditionally.

### Diagnosis recap

Plan 9 was the first feature-class plan after the post-Plan-8 pivot. Bundled PBP ingest with one feature consumer (`opp_epa_allowed_residual` — schedule-of-strength-adjusted EPA-per-play residual replacing v1 `opp_allowed_<pos>_fppg_l4`) per Plan 2a's "ingest + first feature builder" precedent. Adoption gate verdict: **DO_NOT_ADOPT for all 4 positions.** QB / RB / TE returned null results (RMSE + Spearman 95% CIs both bracket zero). WR returned a small but statistically significant **regression** on both RMSE (+0.0083 fpts) and Spearman (-0.0013).

The post-Plan-3e brainstorm hypothesized 5–15% RMSE improvement from PBP-derived features. The empirical result at the BaselineModel level is essentially flat. Three plausible mechanisms: (1) Ridge baseline is feature-saturated — opp signal partially captured by `implied_team_total`, `spread`, NGS metrics, v1 fppg; (2) the v1 fppg implicitly encodes some schedule-of-strength signal even without explicit residual computation; (3) WR-specific noise — opp-strength is more variable game-to-game for WR than other positions. Per Plan 8's lesson: don't chase noise floor; revert and pivot.

### What shipped (all 7 phases)

**Plumbing kept (per spec §6 step 6 zero-position-adopt branch):**
- `src/projections/ingest/pbp.py` — `refresh_pbp` covering `nfl_data_py.import_pbp_data` for 2018-2024 with a curated 27-column subset.
- `PbpSchema` in `src/projections/schemas.py`.
- Opt-in `--run-network` PBP smoke at `tests/test_ingest/test_api_drift.py::test_pbp_api_columns_and_schema`.
- `tests/test_ingest/test_pbp.py` — 6 ingest tests against the synthetic `fake_pbp_df` fixture.
- `tests/conftest.py::fake_pbp_df` (49 rows × 27 cols, all 9 `play_type` values + sacks + scrambles + posteam=NaN + epa=NaN edge cases) and `_build_synthetic_pbp` helper.
- `pbp` keyword arg threaded through 4 direct-builder scripts (`refresh_features.py`, `train_baseline.py`, `predict_2024.py`, `sanity_check_baseline.py`) and through every per-position `build_<pos>_features` signature with `_EMPTY_PBP` default — currently unused, reserved for the next PBP-driven feature plan.
- `scripts/adoption_gate.py` extended with `--baseline-run` / `--candidate-run` dual-run mode (cross-run pairing for feature-set vs feature-set comparisons) — load-bearing for every future feature-class plan.

**Reverted (per spec §6 step 6):**
- `src/projections/features/_opponent.py` — restored v1 `opp_allowed_fppg` + `_row_to_statline`. `opp_epa_allowed_residual` deleted.
- 4 FeaturesSchemas in `schemas.py` — `opp_allowed_<pos>_fppg_l4: Series[float] = pa.Field(ge=0, nullable=True)` restored.
- 4 per-position builder bodies — restored to v1 `opp_allowed_fppg(...)` calls + v1 merge.
- 4 `_<POS>_FEATURE_COLUMNS` in `models/baseline.py` — restored old column names.
- `tests/test_features/test_opponent.py` — bit-identical to v1 (8 new EPA-residual tests removed).
- `test_no_leakage_from_pbp_other_weeks` deleted from each per-position leakage file.

**Real-data drift caught at first ingest:** `nfl_data_py.import_pbp_data` "Downcasts floats" to float32 for memory; PbpSchema's `Series[float]` requires float64. The synthetic fixture used Python floats (which are float64 by default) so didn't surface this. 16 indicator/numeric columns now coerced to float64 in `_normalize_one_season` via the module-level `_FLOAT64_COLS` tuple. Real-data-drift item logged on TODO #16's running list.

**Performance:** the helper's per-(defteam, last_week) iteration over the prior_mask'd PBP was too slow at real-data scale (~7-10s per builder cell × ~600 cells = ~90 min for refresh_features). Optimization landed at commit `60458cb`: builder pre-filters PBP to just the trailing window weeks before the helper call, reducing iterations ~5× (per spec §9 follow-up). Regenerated feature cache in ~3-4 min. (The optimization was reverted with the helper deletion at `941b96c` since the helper itself is gone now; the broader pre-filter idiom can be re-applied if a future plan re-introduces a similar helper.)

### Phase 6 verdicts (run dispatched 2026-04-29 21:04)

Paired bootstrap, n_bootstrap=1000, seed=42. Baseline-run = pre-Plan-9 BaselineModel; candidate-run = post-Plan-9 BaselineModel (same model class, only `opp_allowed_<pos>_fppg_l4` → `opp_pass_epa_allowed_l4` / `opp_run_epa_allowed_l4` swapped). Pairing key `(gsis_id, season, week, position)`.

| Position | Verdict | RMSE delta (95% CI) | Spearman delta (95% CI) | n_paired |
|---|---|---|---|---:|
| QB | DO_NOT_ADOPT | +0.0005 ([-0.0125, +0.0144]) | -0.0001 ([-0.0029, +0.0024]) | 2676 |
| RB | DO_NOT_ADOPT | +0.0001 ([-0.0110, +0.0111]) | +0.0006 ([-0.0014, +0.0027]) | 5273 |
| TE | DO_NOT_ADOPT | -0.0037 ([-0.0121, +0.0050]) | +0.0000 ([-0.0028, +0.0027]) | 4257 |
| WR | DO_NOT_ADOPT | +0.0083 ([+0.0043, +0.0124]) | -0.0013 ([-0.0021, -0.0004]) | 8460 |

Per-year breakdowns: `reports/adoption_gate_plan9.csv`. Markdown report: `reports/adoption_gate_plan9.md`. Spec §6 carries the verdict + interpretation tables for record-of-decision.

### Per-position routing changes shipped

| Position | Pre-Plan-9 default | Post-Plan-9 default | Reason |
|----------|--------------------|---------------------|--------|
| QB       | lightgbm-nb        | lightgbm-nb         | No change. Plan 9 evaluated baseline-only; Plan 8's QB routing untouched. |
| RB       | baseline           | baseline            | No change. DO_NOT_ADOPT verdict reverted feature swap. |
| TE       | baseline           | baseline            | No change. DO_NOT_ADOPT verdict reverted feature swap. |
| WR       | ensemble           | ensemble            | No change. Plan 9 evaluated baseline-only; Plan 8's WR routing untouched. |

### Decision-log entries (Plan 9)

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-29 | PBP storage shape: raw per-play parquet, per-season partition, ~27-column curated subset | Mirrors `weekly_stats.py` template; future feature plans extend `_KEEP` additively without forcing a re-ingest. Total ~340K rows × 27 cols × 7 seasons ≈ 30-50 MB. |
| 2026-04-29 | "Opp-adjusted" interpretation = schedule-of-strength residual (per-play residual = EPA - offense's overall mean EPA in the same trailing window) | Standard nfl-stats formulation; the v1 fppg's flaw was lack of schedule adjustment, so any non-residual formulation would have shipped a renamed v1. |
| 2026-04-29 | Replace v1 fppg directly rather than running side-by-side | Side-by-side muddles the experiment (Ridge sees both, can't attribute the lift). Adoption gate decides per-position; revert is a one-commit undo. |
| 2026-04-29 | Adoption-gate CLI extended with `--baseline-run` / `--candidate-run` dual-run mode | Plan 8's CLI assumed model-class-vs-model-class within ONE run; Plan 9 needs feature-set-vs-feature-set across two runs. Same paired-bootstrap math, different inputs. Amortizes across every future feature-class plan. |
| 2026-04-29 | All 4 positions DO_NOT_ADOPT for opp-adjusted EPA-residual feature on BaselineModel | QB/RB/TE null results, WR significant regression. Per spec §6 step 6 zero-position-adopt branch: revert feature, ship plumbing. |
| 2026-04-29 | Builders keep `pbp: pd.DataFrame = _EMPTY_PBP` keyword arg post-revert | Lets Task 10's caller plumbing continue passing `pbp=` without script changes; reserves the threading for the next PBP-driven feature plan. `_EMPTY_PBP` module-level singleton avoids ruff B008. |
| 2026-04-29 | nfl_data_py float32 → float64 coercion added to `_normalize_one_season` | Real-data drift caught at first ingest in Phase 6: nfl_data_py "Downcasts floats" but PbpSchema's `Series[float]` is float64. Synthetic fixture used Python floats and didn't surface this. 16 columns now in `_FLOAT64_COLS`. Real-data drift entry on TODO #16. |

### Next track after Plan 9

PBP plumbing (Plan 9's TODO #3a) is shipped and validated. The next PBP-derived feature slice to brainstorm is one of TODO #3c's candidates:

1. **Pace** (plays per 60 min, neutral) — multiplies through volume features.
2. **PROE** (pass rate over expected, game-state adjusted) — per offense team.
3. **Player-level air yards / aDOT distributions** — richer downfield-target signal than NGS's season-to-date snapshot.

Plan 9's negative result on opp-EPA-residual at the BaselineModel level argues for evaluating these against multiple model classes (LightGBM, ensemble) in addition to BaselineModel — model class may dominate over feature class for marginal signals.

The remaining TODO #3c PBP-feature candidates (pressure rate, redzone usage shares, etc.) stay queued.

---

## Plan 8 — Adoption gate redesign — complete; ready for PR (2026-04-29, on branch `feat/plan-8-gate-redesign`)

**Status:** all 7 phases shipped. Final verification gates green: 464 pytest pass / 13 skipped (opt-in only) / 0 fail; mypy 145 source files clean; ruff check + format clean across 154 files. **Two production routing changes shipped** (QB → lightgbm-nb; WR → ensemble) per Phase 4 verdicts below.

### Diagnosis recap

Plans 3e / 5 / 5b / 5c / 7 / 6 all failed the prior §1.3 adoption gate. The streak decomposes into two compounding structural problems:

1. **§1.3 thresholds sit below the per-cell noise floor.** "Composite RMSE strictly lower on ≥12/16 cells AND not worse by >1% on any cell" + "weekly calibration no worse on any cell, mean delta ≥ +0.02" treat sampling variation as systematic regression. Smoking gun: Plan 6 hit 12/16 RMSE wins (meets the count!) but failed because TE 2024 was +1.24% worse — 0.24pp over the no-regression line, on a 1081-week cell. There is no significance test gating this; the noise floor on a single cell's RMSE is plausibly ≥1%.
2. **The calibration metric isn't load-bearing for any planned consumer.** Plan 5c PM and Plan 6 §96–99 already note this out loud — Draft Hub, start/sit, and the DFS lineup optimizer all consume mean and rank, not `[p10, p90]` coverage. Five plans of calibration work optimized a metric whose failure has no downstream cost. Plan 7's Phase 0 separately showed the assumed calibration mechanism was wrong (per-stat coverage doesn't decompose to composite coverage), so multiple plans were also pulling the wrong end of the distribution.

### What shipped (Phases 1–4)

- **Pure-stats module** at `src/projections/backtest/adoption_gate.py`: `BootstrapDelta`, `PositionVerdict`, `paired_bootstrap_rmse_delta`, `paired_bootstrap_spearman_delta`, `verdict_for_position` with the §1.3-replacement rule (RMSE: 95% CI strictly below 0; Spearman: lower CI > -0.02 catastrophic-regression floor; calibration informational, not gating).
- **CLI** at `scripts/adoption_gate.py`: reads any backtest run's per-row `results.parquet`, pairs rows on `(gsis_id, season, week)`, emits per-position adoption verdicts as markdown to stdout + optional CSV via `--csv-out`.
- **Per-position routing**: `_PositionDispatch.default_model_class` field with `__post_init__` validation; `production_model_for(position)` helper (single sanctioned entry point for "the production model for this position"); per-position defaults set per re-evaluation verdicts.
- **37 new tests** (Phase 1: 19 stats tests; Phase 2: 13 CLI tests; Phase 3: 5 routing tests). All pass; mypy + ruff + format clean across 145 source files.
- **Re-evaluation reports** committed under `reports/adoption_gate_*.{md,csv,stderr}` and `reports/adoption_gate_summary.md`.

### Phase 4 re-evaluation verdicts (run_20260429T003552Z)

Paired bootstrap, n_bootstrap=1000, seed=42. Pairing key `(gsis_id, season, week)`. ADOPT requires `rmse_delta` 95% CI strictly below 0 AND `spearman_delta` 95% CI strictly above -0.02.

| Position | Candidate      | Verdict       | RMSE delta (95% CI)             | Spearman delta (95% CI)        | n_paired |
|---|---|---|---|---|---:|
| QB | lightgbm       | DO_NOT_ADOPT  | -0.0233 ([-0.1239, +0.0758])    | +0.0155 ([+0.0010, +0.0296])   | 2676 |
| QB | lightgbm-tuned | **ADOPT**     | -0.1189 ([-0.2063, -0.0310])    | +0.0177 ([+0.0046, +0.0304])   | 2676 |
| QB | lightgbm-nb    | **ADOPT**     | **-0.1933 ([-0.2719, -0.1102])** | +0.0183 ([+0.0045, +0.0313])   | 2676 |
| QB | ensemble       | **ADOPT**     | -0.1760 ([-0.2274, -0.1242])    | +0.0184 ([+0.0098, +0.0262])   | 2676 |
| RB | lightgbm       | DO_NOT_ADOPT  | +0.1916 ([+0.1438, +0.2421])    | -0.0023 ([-0.0082, +0.0028])   | 5273 |
| RB | lightgbm-tuned | DO_NOT_ADOPT  | +0.1144 ([+0.0798, +0.1520])    | -0.0043 ([-0.0098, +0.0009])   | 5273 |
| RB | lightgbm-nb    | DO_NOT_ADOPT  | +0.0420 ([+0.0133, +0.0740])    | -0.0012 ([-0.0068, +0.0039])   | 5273 |
| RB | ensemble       | DO_NOT_ADOPT  | +0.0212 ([-0.0021, +0.0455])    | +0.0003 ([-0.0037, +0.0043])   | 5273 |
| TE | lightgbm       | DO_NOT_ADOPT  | +0.1553 ([+0.1096, +0.2060])    | +0.0043 ([-0.0052, +0.0132])   | 4257 |
| TE | lightgbm-tuned | DO_NOT_ADOPT  | +0.0879 ([+0.0468, +0.1322])    | +0.0082 ([-0.0003, +0.0170])   | 4257 |
| TE | lightgbm-nb    | DO_NOT_ADOPT  | +0.0028 ([-0.0289, +0.0422])    | +0.0071 ([-0.0014, +0.0160])   | 4257 |
| TE | ensemble       | DO_NOT_ADOPT  | -0.0208 ([-0.0454, +0.0097])    | +0.0076 ([+0.0016, +0.0137])   | 4257 |
| WR | lightgbm       | DO_NOT_ADOPT  | +0.1338 ([+0.0963, +0.1721])    | +0.0045 ([-0.0012, +0.0101])   | 8460 |
| WR | lightgbm-tuned | DO_NOT_ADOPT  | +0.0711 ([+0.0397, +0.1046])    | +0.0044 ([-0.0012, +0.0099])   | 8460 |
| WR | lightgbm-nb    | DO_NOT_ADOPT  | -0.0016 ([-0.0316, +0.0291])    | +0.0027 ([-0.0032, +0.0080])   | 8460 |
| WR | ensemble       | **ADOPT**     | -0.0320 ([-0.0531, -0.0092])    | +0.0069 ([+0.0028, +0.0109])   | 8460 |

### Per-position routing changes shipped

| Position | Pre-Plan-8 default | Post-Plan-8 default | Reason                           |
|----------|--------------------|---------------------|----------------------------------|
| QB       | baseline           | **lightgbm-nb**     | 3 ADOPTers; mechanical tie-break (most-negative rmse_delta.point) selects lightgbm-nb (-0.1933) over ensemble (-0.1760) and tuned (-0.1189). |
| RB       | baseline           | baseline            | No ADOPT verdict; every candidate regresses RB or has rank-correlation issues. |
| TE       | baseline           | baseline            | No ADOPT verdict; ensemble is closest (rmse_delta -0.021) but CI brackets zero. |
| WR       | baseline           | **ensemble**        | Sole ADOPTer; rmse_delta -0.032 fpts, both CIs strictly clear zero. n=8460 paired rows give it the statistical power. |

### Surprises vs spec §6 strong prior

Two findings deviate from the spec's prediction:

1. **QB winner is `lightgbm-nb`, not `ensemble`.** The spec expected ensemble to win QB. Reality: NB beats ensemble's RMSE point estimate by ~0.017 fpts; CIs overlap heavily but mechanical tie-break selects NB. Side benefit: NB is structurally simpler than ensemble (no MixtureDistribution / per-stat weight optimizer / 4-stage fit) — simpler is better when stat-equivalent.
2. **WR ADOPTs `ensemble`.** The spec said "WR's improvements were ≤0.55% per cell, pooled CI may or may not clear zero." It cleared. Larger sample (n=8460) gives the bootstrap enough power for a small-but-clean win.

### Snapshot regression gate audit (Phase 5)

`tests/backtest/tolerances.json` defaults vs measured per-cell noise floor (from Plan 8's bootstrap reports under `reports/adoption_gate_*.csv`, per-year breakdown rows):

| Metric kind            | snapshot default | Measured per-cell RMSE-delta CI half-width (absolute, fpts) | Translated to relative (÷ typical per-cell RMSE ≈ 6 fpts) | Verdict |
|---|---|---|---|---|
| `rmse_relative`        | 5.0%   | median 0.076; p75 0.100; max 0.217 | median 1.26%; p75 1.66%; max 3.61% | **Fine — ~3-4× headroom over the median noise floor** |
| `mae_relative`         | 5.0%   | (same scale as rmse) | (same) | **Fine** |
| `spearman_absolute`    | 0.02   | (Spearman per-year deltas have CI half-widths ~0.005–0.015) | n/a | **Fine — matches Plan 8's catastrophic-regression floor** |
| `calibration_absolute` | 0.03   | (calibration is informational under Plan 8; not load-bearing for adoption) | n/a | **Fine for code-regression purposes** |
| `mean_pred_relative`   | 10.0%  | (broader window, intentionally — guards against unintended mean shifts) | n/a | **Fine** |

**Conclusion**: snapshot.py's tolerances are above the per-cell noise floor with comfortable headroom; the regression gate is doing the right job (catching real code-induced numeric drift, not flagging sampling noise). No changes needed; no follow-up TODO filed.

The snapshot regression gate (catches code regression on a frozen model) and the new adoption gate (decides which class is the production default) answer different questions and stay independent. Both ship as-is.

### Phase 6 — §1.3 spec template

`docs/superpowers/specs/_adoption_gate_template.md` shipped. Future model-class specs inline-copy the §1.3 body into their own spec, so each spec carries the gate it was evaluated under as record-of-decision.

### Phase 7 — Final verification

All gates green at the tip of the branch:

- `mypy src tests` — 145 source files, no issues.
- `ruff check src tests scripts` + `ruff format --check src tests scripts` — clean across 154 files.
- `pytest` (excluding the slow `test_models` leakage tests + opt-in `--run-backtest` / `--run-network` gates): **464 passed, 13 skipped, 0 failed** in 7m50s.

### Next track after Plan 8

Feature-class work starting with **TODO #3 (PBP / EPA features)**. Five model-class swaps on identical features hit the same information ceiling; the next real RMSE lift (estimated 5–15%) lives in features, not in model class.

---

## Plan 6 — Model D ensemble (A + C-NB) — shipped as peer (run 2026-04-29)

**Verdict:** ship as peer. Per-(position, stat) calibration-aware weighted mixture of Model A and Model C-NB landed cleanly with the per-stat pinball optimizer behaving exactly as designed — yards stats heavily favor C-NB's tight QuantileDistribution; TD stats favor A's wider parametric distributions. **All three §1.3 adoption criteria failed** by narrow margins. The per-position split that motivated the plan is preserved (QB cells improve on every metric; RB/TE/WR cells regress on calibration), confirming the mechanism but also confirming Plan 7's lesson — per-stat coverage at [p10, p90] does not algebraically decompose to composite [p10, p90] coverage.

`EnsembleModel` (Model D) lands as a fifth peer of Models A, C, C-tuned, C-NB. Wraps a `BaselineModel` + `LightGBMNbModel` pair via `_EnsembleConfig` factories. The 4-stage `fit()` per spec §3.1 trains weight-fit children on `[S, Y-2]`, predicts the calibration year `Y-1`, fits per-(position, stat) weights via `scipy.optimize.minimize_scalar` on summed pinball loss at q ∈ {0.10, 0.90}, then re-fits prediction children on the full `[S, Y-1]` span. `MixtureDistribution` (new in `src/projections/distributions/mixture.py`) implements the `Distribution` Protocol structurally — pure CDF-pool composition with brentq-based quantile inversion. New codec MIXTURE branch persists `{family, weight, component_a, component_b}` recursively; `schema_version` bumps 1 → 2.

Snapshot extended 1504 → 1872 rows (368 new ensemble rows: 23 metrics × 4 positions × 4 years). 16 weight artifacts at `data/ensemble_weights/{model_id}.json` (4 positions × 4 folds, filename sanitizes `:` → `_` for NTFS). The 32 `season_calibration_*` rows still skip via the existing `SAMPLED_SUMMARY`-only family gate; TODO #28 stays open.

### Per-position model_ids (final fold, train 2018-2023, predict 2024)

| Position | Model A | Model C-NB | Model D Ensemble |
|---|---|---|---|
| QB | `baseline:qb:5e8fe380:2018-2023` | `lightgbm-nb:qb:4f40329c:2018-2023` | `ensemble:qb:3494f28a:2018-2023` |
| RB | `baseline:rb:...:2018-2023` | `lightgbm-nb:rb:...:2018-2023` | `ensemble:rb:9dec620c:2018-2023` |
| TE | `baseline:te:...:2018-2023` | `lightgbm-nb:te:...:2018-2023` | `ensemble:te:da0287a2:2018-2023` |
| WR | `baseline:wr:730abe91:2018-2023` | `lightgbm-nb:wr:b751ce19:2018-2023` | `ensemble:wr:6f075552:2018-2023` |

(Full model_ids per fold are in the `data/ensemble_weights/*.json` artifacts.)

### Adoption-gate verdict — DO NOT ADOPT Model D as default

Spec §1.3 required Model D to beat Model A on three criteria. **All three failed.**

| Criterion | Threshold | Actual (D vs A) | Pass? |
|---|---|---|---|
| Composite RMSE strictly lower on >=12/16 cells; max +1% worse | D < A on 12+; max +1% worse | D strictly lower on **12/16** (meets count); max +1.24% on TE 2024 (exceeds +1.0%) | **FAIL** (margin) |
| Spearman top-N within ±0.005 on every cell | All within ±0.005 | 4/16 outside ±0.005; max abs delta +0.0131 (QB 2021, **a +0.0131 IMPROVEMENT**) | **FAIL** |
| Calibration no worse on any cell; mean delta >= +0.02 | No regressions; mean ≥ +0.02 | D worse on 13/16; mean delta -0.058 | **FAIL** |

The Spearman criterion's "within ±0.005" is symmetric — it fails ensembles that improve rank ordering by >0.005, not just those that regress it. All 4 of Plan 6's Spearman violations are positive deltas (rank ordering *improves* on those cells). For purposes of "does D beat A on rank?" Plan 6 ties or wins on every cell except RB 2021 (delta -0.0055).

### Side-by-side per-cell comparison (16 cells)

`RMSE D-A %` is the percentage delta on composite RMSE (negative = D wins; threshold +1.00%). Spearman / Calib columns show D's value and the (D − A) delta.

| Cell | RMSE A | RMSE D | RMSE D-A % | Spearman A | Spearman D | Spearman D-A | Calib A | Calib D | Calib D-A |
|---|---|---|---|---|---|---|---|---|---|
| QB 2021 | 7.8342 | 7.6396 | -2.49% | 0.9342 | 0.9473 | +0.0131 | 0.6947 | 0.7143 | +0.0196 |
| QB 2022 | 7.2261 | 7.0432 | -2.53% | 0.9669 | 0.9667 | -0.0002 | 0.7458 | 0.7808 | +0.0350 |
| QB 2023 | 7.3092 | 7.1780 | -1.80% | 0.9454 | 0.9570 | +0.0116 | 0.7313 | 0.7299 | -0.0014 |
| QB 2024 | 7.6995 | 7.5061 | -2.51% | 0.9383 | 0.9437 | +0.0054 | 0.7018 | 0.7222 | +0.0204 |
| RB 2021 | 6.8486 | 6.8948 | +0.67% | 0.9700 | 0.9645 | -0.0055 | 0.7475 | 0.6563 | -0.0912 |
| RB 2022 | 6.6359 | 6.6197 | -0.24% | 0.9658 | 0.9680 | +0.0022 | 0.7415 | 0.6536 | -0.0879 |
| RB 2023 | 6.3143 | 6.3485 | +0.54% | 0.9665 | 0.9657 | -0.0008 | 0.7872 | 0.6789 | -0.1083 |
| RB 2024 | 6.4860 | 6.5070 | +0.32% | 0.9753 | 0.9762 | +0.0009 | 0.7568 | 0.6573 | -0.0995 |
| TE 2021 | 5.3365 | 5.2750 | -1.15% | 0.9655 | 0.9659 | +0.0004 | 0.7350 | 0.6748 | -0.0602 |
| TE 2022 | 5.2498 | 5.2024 | -0.90% | 0.9615 | 0.9642 | +0.0027 | 0.7647 | 0.6590 | -0.1057 |
| TE 2023 | 4.9422 | 4.9041 | -0.77% | 0.9704 | 0.9751 | +0.0047 | 0.7561 | 0.6767 | -0.0794 |
| TE 2024 | 5.0804 | 5.1435 | +1.24% | 0.9620 | 0.9593 | -0.0027 | 0.7345 | 0.6605 | -0.0740 |
| WR 2021 | 6.7333 | 6.6966 | -0.55% | 0.9699 | 0.9699 | +0.0000 | 0.6956 | 0.6339 | -0.0617 |
| WR 2022 | 6.6255 | 6.5910 | -0.52% | 0.9767 | 0.9754 | -0.0013 | 0.6970 | 0.6275 | -0.0695 |
| WR 2023 | 6.5159 | 6.4920 | -0.37% | 0.9680 | 0.9671 | -0.0009 | 0.7256 | 0.6415 | -0.0841 |
| WR 2024 | 6.6728 | 6.6398 | -0.49% | 0.9739 | 0.9721 | -0.0018 | 0.7109 | 0.6309 | -0.0800 |

### Per-position split — QB clean win on every metric; RB/TE/WR RMSE wins paired with calibration regressions

| Position | RMSE wins vs A | Mean Spearman delta | Mean calib delta vs A |
|---|---|---|---|
| QB | 4/4 | +0.0075 | **+0.0184** (positive) |
| RB | 1/4 | -0.0008 | -0.0968 |
| TE | 3/4 | +0.0013 | -0.0798 |
| WR | 4/4 | -0.0010 | -0.0738 |

**QB is the only position where Model D cleanly beats A on every metric on every fold.** RMSE -1.8% to -2.5% across 4/4 years; calibration mean +0.018 (3 of 4 years positive); Spearman gains in 3 of 4 years. Per the final-fold weight vector, the QB optimizer pulls passing_yards (0.20) and rushing_yards (0.12) heavily toward C-NB while leaving TDs and interceptions near-balanced — exactly the direction QB-specific gains in Plan 5c suggested.

**RB / TE / WR show the same pattern across the board:** RMSE improves on most cells (RB 1/4, TE 3/4, WR 4/4 wins), but [p10, p90] calibration regresses 6-11 percentage points. The mechanism is the same one Plan 5c diagnosed and Plan 7's Phase 0 confirmed empirically: NB-2 dispersion fitted on training residuals produces tight predictive intervals that don't survive held-out variance on RB/TE/WR; the per-stat pinball optimizer correctly identifies that yards distributions should pull heavily toward C-NB (where per-stat coverage is good), but the convolution into composite fantasy points doesn't preserve [p10, p90] coverage.

### Per-stat fitted weights — final fold

Across all 4 positions, the optimizer learned a clean per-stat pattern:

- **Yards stats** (passing / rushing / receiving): w_a ∈ [0.001, 0.20] — heavily C-NB.
- **TD stats** (passing / rushing / receiving TDs): w_a ∈ [0.61, 0.77] — moderately A.
- **Other counts** (interceptions / receptions / fumbles_lost): mixed, position-dependent.

This validates the design hypothesis from spec §1.1: A's wider parametric distributions help TD calibration; C-NB's tight QuantileDistribution distributions match yards p10/p90 well. What the design did NOT predict is that this per-stat optimum would not propagate to composite calibration on RB/TE/WR.

### Why this should work / does it work

Spec §1.1's mechanism hypothesis was correct in isolation (per-stat). The mixture variance formula `w·var_A + (1-w)·var_B + w(1-w)(mean_A − mean_B)²` does widen calibration intervals when component means differ. The pinball optimizer correctly identifies the per-stat optimum — visible in the clean yards-vs-TDs split.

**The composite [p10, p90] coverage problem is upstream of any per-stat fix.** Plan 7's diagnostic established that per-stat coverage at the central interval (p10/p90) doesn't decompose to composite coverage at the central interval — composite p10/p90 width is dominated by yards (weight ~6-8 fp per 100 yards), composite tail weight by counts (TD weight × 6 = single-row 6-18 fp jumps). When ensemble narrows yards (good for yards p10/p90) and widens TDs (good for TD p10/p90), the composite [p10, p90] band tightens around yards width but the composite tail behavior shifts in a way that increases the rate of actuals falling outside composite [p10, p90].

**TODO #30 follow-up #1 (composite-direct optimization via Monte Carlo) is the right next experiment if calibration is the priority.** Plan 6 confirms what Plan 7's diagnostic predicted: any per-stat-decoupled fix is fundamentally limited.

### Decision

**Default model selection:** Model A stays the production default. Models C, C-tuned, C-NB, and D all ship as peers; none is adopted. **Model D's QB cells beat Model A on every metric — if the project ever adopts a per-position default selection, the QB row of `POSITION_DISPATCH` could route through Model D while leaving RB/TE/WR routed through A.** Not implemented in this plan; flagged as a future routing experiment.

**Pivot:** The next track is determined by what we want from the modeling stack:
1. **Calibration priority** → composite-direct weight optimization via MC (TODO #30 follow-up #1). Same EnsembleModel infrastructure, replace pinball-on-per-stat with composite-Brier-on-MC. ~5-10x slower per fold; risks but might break the per-stat-vs-composite barrier.
2. **Mean-prediction priority** → feature-class tracks (TODO #3 PBP/EPA, TODO #23 target decomposition). Estimated 5-15% RMSE win on top of any model class. Independent of model class.
3. **Pivot to consumer tools** → Plan 4 (public Python API + CLI verbs). Modeling has reached "good enough" for downstream consumers; all four planned tools (Draft Hub, start/sit, DFS) consume mean and rank, not [p10, p90] coverage.

Pick one in the next session.

### Per-position model_ids on disk

Standalone artifacts at `data/ensemble_weights/ensemble_{pos}_{8hex}_{S}-{E}.json` (filename sanitizes `:` → `_` for NTFS). The joblib pickle at `models/artifacts/ensemble-{pos}-...joblib` is only created on `scripts/train_baseline.py`-style invocations; the backtest harness regenerates per-fold artifacts in-memory and does not write standalone files.

### Operational notes

- Backtest run: 2026-04-29, ~5h45m wall-clock for the full `--model all` regeneration on real data (5 model classes × 4 positions × 4 folds). Ensemble's 4-child-per-fold fit + per-stat pinball optimizer is the bottleneck; weight optimization alone is ~3h of the 5h45m total.
- Determinism re-check (`--check` after `--update-snapshot`) is **deferred** — re-running takes 5+ hours wall-clock. Plan 6 ships the snapshot from this single run; future re-runs (e.g., a touch on `ensemble.py`) should re-validate determinism before merging the resulting snapshot.
- Test runtime cost: `tests/test_models/test_ensemble_model_smoke.py` 4 new fit-based tests added ~14 min in CI. Phase 6 keeps the existing `@pytest.mark.backtest` gating; the quint-model smoke at `tests/test_backtest/test_harness_quint_model.py` is gated on real-data caches and does NOT run in lightweight CI.

---

## Plan 7 — Calibration-aware NB-2 fitting (Model C-NB-cal) — STOPPED at Phase 0 (2026-04-28)

**Verdict:** stop the plan. Spec premise was misaligned with empirical reality. Phase 0 ships as research output; Phase 1+ unexecuted. Branch `feat/plan-7-calibration-aware-nb` proposed for merge with just the diagnostic CLI + spec + plan + research note (record-of-decision). Filed TODO #30 for the right follow-up plan.

### What happened

Plan 7's spec assumed Plan 5c's "NB-2 distribution too narrow at the [p10, p90] tails" claim mapped directly to per-stat NB-2 distributions being too narrow at p10/p90. The Phase 0 diagnostic measured per-stat empirical [p10, p90] coverage on Plan 5c's C-NB output and showed the opposite: count NB-2 distributions are *over*-covering at [p10, p90] by ~16pp (mean gap **-0.169**; range -0.188 to -0.154 across all 16 cells). Yards distributions are well-calibrated (mean gap **+0.011**).

Pinball-loss fitting at q=0.10 / q=0.90 — Plan 7's exact mechanism — would tighten count distributions toward 0.80 nominal, which is the opposite direction needed to close the composite [p10, p90] coverage gap (-0.062 mean vs A). The composite gap mechanism lives in **upper-tail behavior beyond p90** — outside what Plan 7's loss function targets.

### Why per-stat over-coverage coexists with composite under-coverage

For low-mean count NB-2 (μ ≈ 0.4 typical for RB receiving_tds), discrete support concentrates ~95% of mass at {0, 1}, so the predicted [p10, p90] = [0, 1] trivially over-covers at the 80% nominal. The thin upper tail (P(X≥2) ≈ 0.05 model vs ~7-10% empirical) is what Plan 5c described as "too narrow" — the wording is imprecise; the narrowness is at p95-p99, not at p10/p90. Composite [p10, p90] under-coverage comes from upper-tail count outliers (TD weight × 6 = 12-18 fp jumps) exceeding composite p90 set by yards width.

Per-stat coverage at [p10, p90] does NOT decompose to composite coverage at [p10, p90] — convolution behavior at the central interval is dominated by the wider distribution's mean shift, not by the narrower distribution's own [p10, p90] coverage.

### Diagnostic CLI ships as reusable research output

`scripts/diagnose_calibration_breakdown.py` reads any per-row backtest parquet, filters by `model_id` prefix, and emits a per-(position, year) CSV decomposing per-stat coverage. It can be re-pointed at any future model-class output. 8 unit tests; mypy / ruff clean.

### What was missed at scoping

1. Trusted Plan 5c's mechanism statement without empirically checking it. A two-line python check at scoping would have caught the per-stat-over-coverage finding.
2. Per-stat coverage doesn't decompose composite coverage (convolution effect). The right diagnostic was counterfactual replacement (swap count distributions for A's; re-sample composite; measure closure). Shipped diagnostic measures a related-but-not-identical quantity that happens to surface the right verdict.
3. Missed the discreteness math: at μ ≈ 0.4, NB-2 mass at {0,1} trivially over-covers any [p10, p90] band at 80% nominal. Mechanical, catchable in 5 minutes.

Cost of getting it wrong: ~30 min conversation + 10 min compute. Cheap vs. building Phase 1 + Phase 2 and discovering empirically.

### Per-cell breakdown

Pasted into the research note: `docs/superpowers/research/2026-04-28-calibration-breakdown.md`.

### Next action

The composite calibration shortfall remains the binding constraint. Three candidate follow-up tracks (TODO #30 captures all three):

1. **Pinball-loss dispersion fit at upper-tail quantiles** (q=0.90, q=0.95 or q=0.95 only). Same machinery as Plan 7; right mechanism for the actual gap location.
2. **ZIP (zero-inflated Poisson) for count cells.** Handles zero mass + thin tail decoupled rather than via NB-2's single overdispersion knob. Fundamental family change.
3. **Mixture model: point mass at 0 + heavier-tailed integer distribution.** Most flexible, most code; defer until 1 and 2 are tried.

Or: accept the calibration shortfall as a known limitation and pivot to feature-class tracks (TODO #3, TODO #23) or Plan 6 (ensemble) instead. None of the planned downstream consumers (Draft Hub, start/sit, DFS) actually depend on a perfectly calibrated [p10, p90] — they consume mean and rank. Plan 5c PM already framed this as acceptable.

---

## Plan 5c — Hybrid LightGBM with NB-2 for Count Stats (Model C-NB) — shipped (run 2026-04-28)

**Closes:** the count-stat-bias mechanism identified in Plan 5b's diagnostic. Model C-NB strictly dominates Model C-tuned on RMSE (16/16 cells better) but still fails Model A's adoption gate — the gap moved from "tuning regression" to "calibration regression."

`LightGBMNbModel` (Model C-NB) lands as a fourth peer of Models A, C, C-tuned. Subclasses `LightGBMTunedModel`; for the 13 count cells Plan 3e routes through NB-2 in Ridge (`passing_tds`/`rushing_tds`/`receiving_tds`/`interceptions`/`fumbles_lost` × per-position target_stats), trains one `lgb.LGBMRegressor(objective="poisson")` per stat, fits NB-2 dispersion via the public `nb_dispersion_from_residuals` (relocated from `models/baseline.py` to `distributions/parametric.py` in Phase 0), and predicts via `ParametricNegativeBinomial(mu, dispersion)`. Yards/receptions stats unchanged from Model C-tuned (5-quantile + `QuantileDistribution`). Reuses Plan 5b's tuned hyperparameters from `data/tuned_params/lightgbm.json`. Per-row family is `MIXED`; per-stat families remain encoded inside the params blob via the existing codec.

Snapshot extended 1136 → 1504 rows (368 new lightgbm-nb rows; the 32 `season_calibration_*` rows are skipped by the SAMPLED_SUMMARY-only family gate, TODO #28 still open).

### Per-position model_ids

| Position | Model A | Model C | Model C-tuned | Model C-NB |
|---|---|---|---|---|
| WR | `baseline:wr:6d955427:2018-2023` | `lightgbm:wr:a4dd5a82:2018-2023` | `lightgbm-tuned:wr:62df14ad:2018-2023` | `lightgbm-nb:wr:dc445a2d:2018-2023` |
| QB | `baseline:qb:c98738f3:2018-2023` | `lightgbm:qb:06fadb3f:2018-2023` | `lightgbm-tuned:qb:fc902ed6:2018-2023` | `lightgbm-nb:qb:3ae5b940:2018-2023` |
| RB | `baseline:rb:5a86c8ee:2018-2023` | `lightgbm:rb:fb169c0e:2018-2023` | `lightgbm-tuned:rb:5d69fdfe:2018-2023` | `lightgbm-nb:rb:ba2e35cc:2018-2023` |
| TE | `baseline:te:9c00025b:2018-2023` | `lightgbm:te:bd4c2a5b:2018-2023` | `lightgbm-tuned:te:89dafdb6:2018-2023` | `lightgbm-nb:te:e76e590a:2018-2023` |

### Adoption-gate verdict — DO NOT ADOPT Model C-NB as default

Spec §1.3 required Model C-NB to beat Model A on three criteria. **All three failed.**

| Criterion | Threshold | Actual (C-NB vs A) | Pass? |
|---|---|---|---|
| Composite RMSE strictly lower on >=12/16 cells; max +1% worse | C-NB < A on 12+; max +1% worse | C-NB strictly lower on **11/16**; max +1.69% worse (TE 2024); 4/16 cells exceed 1% | **FAIL** |
| Spearman top-N within +-0.005 on every cell | All within ±0.005 | 4/16 outside ±0.005; max abs delta 0.0204 (QB 2021, **a +0.0204 IMPROVEMENT**) | **FAIL** |
| Calibration no worse on any cell; mean delta >= +0.02 | No regressions; mean ≥ +0.02 | C-NB worse on 13/16; mean delta -0.0617 | **FAIL** |

### Side-by-side per-cell comparison (16 cells × 4 metrics × 4 models)

`RMSE Cnb-A %` is the percentage delta on composite RMSE (negative = NB wins; threshold +1.00%). Spearman / Calib columns show NB's value and the (NB − A) delta.

| Cell | RMSE A | RMSE C | RMSE Ctuned | RMSE Cnb | RMSE Cnb-A % | Spearman A | Spearman Cnb | Spearman Cnb-A | Calib A | Calib Cnb | Calib Cnb-A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| QB 2021 | 7.8342 | 7.8386 | 7.7320 | 7.5651 | -3.43% | 0.9342 | 0.9546 | +0.0204 | 0.6947 | 0.7173 | +0.0226 |
| QB 2022 | 7.2261 | 7.2718 | 7.1220 | 7.0637 | -2.25% | 0.9669 | 0.9672 | +0.0003 | 0.7458 | 0.7686 | +0.0228 |
| QB 2023 | 7.3092 | 7.3213 | 7.2602 | 7.2152 | -1.29% | 0.9454 | 0.9569 | +0.0115 | 0.7313 | 0.7194 | -0.0119 |
| QB 2024 | 7.6995 | 7.5523 | 7.4845 | 7.4597 | -3.12% | 0.9383 | 0.9452 | +0.0069 | 0.7018 | 0.7178 | +0.0161 |
| RB 2021 | 6.8486 | 7.0688 | 6.9895 | 6.9292 | +1.18% | 0.9700 | 0.9623 | -0.0077 | 0.7475 | 0.6327 | -0.1148 |
| RB 2022 | 6.6359 | 6.8370 | 6.7032 | 6.6331 | -0.04% | 0.9658 | 0.9692 | +0.0034 | 0.7415 | 0.6446 | -0.0969 |
| RB 2023 | 6.3143 | 6.5069 | 6.4255 | 6.3744 | +0.95% | 0.9665 | 0.9640 | -0.0025 | 0.7872 | 0.6751 | -0.1121 |
| RB 2024 | 6.4860 | 6.6370 | 6.6242 | 6.5157 | +0.46% | 0.9753 | 0.9763 | +0.0010 | 0.7568 | 0.6512 | -0.1056 |
| TE 2021 | 5.3365 | 5.4636 | 5.3835 | 5.3137 | -0.43% | 0.9655 | 0.9640 | -0.0015 | 0.7350 | 0.6602 | -0.0748 |
| TE 2022 | 5.2498 | 5.3710 | 5.3197 | 5.2092 | -0.78% | 0.9615 | 0.9629 | +0.0015 | 0.7647 | 0.6710 | -0.0938 |
| TE 2023 | 4.9422 | 5.1324 | 5.0272 | 4.9314 | -0.22% | 0.9704 | 0.9742 | +0.0038 | 0.7561 | 0.6805 | -0.0756 |
| TE 2024 | 5.0804 | 5.2661 | 5.2302 | 5.1662 | +1.69% | 0.9620 | 0.9580 | -0.0040 | 0.7345 | 0.6586 | -0.0759 |
| WR 2021 | 6.7333 | 6.8837 | 6.8025 | 6.7244 | -0.13% | 0.9699 | 0.9679 | -0.0020 | 0.6956 | 0.6354 | -0.0602 |
| WR 2022 | 6.6255 | 6.7479 | 6.7149 | 6.6189 | -0.10% | 0.9767 | 0.9729 | -0.0038 | 0.6970 | 0.6256 | -0.0714 |
| WR 2023 | 6.5159 | 6.7129 | 6.5922 | 6.5380 | +0.34% | 0.9680 | 0.9645 | -0.0035 | 0.7256 | 0.6447 | -0.0809 |
| WR 2024 | 6.6728 | 6.7339 | 6.7218 | 6.6589 | -0.21% | 0.9739 | 0.9709 | -0.0030 | 0.7109 | 0.6362 | -0.0747 |

### Aggregate movement: NB strictly dominates Tuned on RMSE; calibration unchanged in aggregate

| Metric | Tuned vs A | NB vs A | NB vs Tuned |
|---|---|---|---|
| RMSE: cells where C-* beats A | 4/16 | 11/16 | 16/16 strict dominance |
| RMSE: max pct worse vs A | +2.95% | +1.69% | — |
| Spearman: cells outside ±0.005 vs A | 7/16 | 4/16 | improved |
| Spearman: max abs delta vs A | 0.0163 | 0.0204 (a +0.0204 *gain* on QB 2021) | — |
| Calibration: cells where C-* worse than A | 12/16 | 13/16 | NB on average +0.0013 vs Tuned |
| Calibration: mean delta vs A | -0.0630 | -0.0617 | +0.0013 (essentially unchanged) |

**NB strictly dominates Tuned on RMSE on every cell.** Replacing the 5-knot quantile prediction for count stats with a poisson-objective regressor + NB-2 dispersion eliminated the count-stat over-prediction Plan 5b diagnosed. The mean RMSE pct vs A moved from "Tuned regresses on 12/16 cells" to "NB beats A on 11/16 cells." But that improvement does not propagate to calibration — the mean p10/p90 coverage delta vs A stayed essentially flat (-0.0630 → -0.0617).

### Per-position split — QB clean win; RB/TE/WR are RMSE wins paired with calibration regressions

| Position | RMSE wins vs A | Mean calib delta vs A |
|---|---|---|
| QB | 4/4 | **+0.0124 (positive)** |
| RB | 1/4 | -0.1074 |
| TE | 3/4 | -0.0800 |
| WR | 3/4 | -0.0718 |

**QB is the only position where C-NB cleanly beats A on every metric:** RMSE -1.3% to -3.4% across 4/4 years, calibration +0.012 mean (3 of 4 years positive), Spearman +0 to +0.02. QB's count-stat distributions (passing_tds with mean ~1.5, interceptions ~0.7) are the exact zero-inflated count stats NB-2 was designed for, and the per-row mean is the dominant signal in QB scoring (rushing yards and passing yards together account for ~85% of fantasy points; passing TDs are the next ~10%).

**RB / TE / WR show the same pattern across the board:** RMSE improves (RB 1/4 wins, TE 3/4, WR 3/4 — close to a 50/50 split), but [p10, p90] calibration regresses 6-12 percentage points. The mechanism: NB-2 dispersion fitted on training residuals via conditional MLE produces tight predictive intervals when the per-row mean is well-fit (which the poisson booster does), but the residual variance on test data — particularly RB/TE/WR which have higher target variance and more regime drift between seasons — exceeds the training-fit dispersion. The fitted NB-2 distribution is therefore too narrow at the [p10, p90] tails on held-out years.

A practical illustration: RB 2024 has C-NB calib 0.6512 vs A 0.7568 — a 10 pp coverage drop. The NB-2 receiving_tds distribution for an average-volume RB with mu_hat ≈ 0.4 and fitted dispersion ≈ 5 has [p10, p90] of [0, 1] (NB-2 mode at 0; long right tail). When test-set actuals scatter to 2-3 TDs (a realistic RB game), they fall outside p90. The booster predicts the mean correctly; the dispersion under-estimates the heavy right tail.

### Why this should work / does it work

The Plan 5b diagnostic identified the mechanism: 5-knot QuantileDistribution linear interpolation + sort + clip produces a biased empirical mean on zero-inflated count stats (over-prediction of 30-60% on TE/WR receiving_tds, QB rushing_tds/fumbles, etc.). NB-2 with mean = mu_hat and dispersion fit on training residuals does not have this bias — the empirical mean of NB-2 samples ≈ mu_hat by construction.

**The mean-prediction fix worked.** NB strictly dominates Tuned on RMSE on every cell — closing the entire mean-prediction gap Plan 5 / 5b's quantile sub-models couldn't. RMSE moved from "C-tuned 4/16 wins, max +2.95% worse" to "C-NB 11/16 wins, max +1.69% worse" — a step closer to the §1.3 threshold but still short of 12/16.

**The calibration regression that Plan 5 / 5b had against A on the [p10, p90] interval did NOT close.** NB-2's narrow predictive interval at low mu trades RMSE for coverage. Plan 3e Phase 1 saw the same shape in Ridge (NB cells nudged calibration in the right direction in aggregate but did not solve the problem). Replacing GAMMA with NB-2 in Ridge, or quantile regression with NB-2 in LightGBM, both improve mean-prediction RMSE without solving the underlying coverage problem because the underlying residual variance on held-out years exceeds what a well-fit conditional distribution can represent without overfitting the noise floor.

Yards stats are unchanged from Model C-tuned (the test `test_yards_stat_predictions_match_tuned_baseline` pins yards-stat best_iters to be bit-exact identical between the two models) — the calibration regression on RB/TE/WR is fully attributable to the count-stat NB-2 path, not to anything in the inherited yards path.

### Decision

**Default model selection:** Model A stays the production default. Models C, C-tuned, and C-NB all ship as peers; none is adopted. Model C-NB strictly dominates Model C-tuned on RMSE — Model C-tuned is now arguably prunable (TODO followup).

**Pivot:** the next model-improvement track stays one of the three remaining options. With C-NB now showing what mean-prediction fixes alone can do, the calibration gap is the unambiguous binding constraint. Three remaining tracks:

- **Plan 6 — Model D ensemble.** Stack of (Model A, Model C-NB) per (position, stat) with calibration-aware weighting. Cheapest given Plan 5c's infrastructure. Covers the case where C-NB's mean-prediction wins on QB and Model A's calibration wins on RB/TE/WR.
- **Calibration-aware NB fitting.** Fit NB-2 dispersion to optimize p10/p90 coverage directly (quantile loss) rather than likelihood. Preserves C-NB's RMSE wins; targets the calibration regression directly. Risk: overfit to validation noise.
- **TODO #3 (PBP / EPA features)** + **TODO #23 (target decomposition)** — feature-class tracks. Independent of model class. Estimated 5-15% RMSE win on top of any model class.

Pick one in the next session. Plan 4 (public Python API + CLI verbs + free-tier hosting) remains the post-modeling milestone.

### Per-position model_ids on disk

Standalone artifacts at `models/artifacts/lightgbm-nb-{pos}-2018-2023-{hash}.joblib` (only created on `scripts/train_baseline.py`-style invocations; the backtest harness regenerates per-fold artifacts from the feature cache and does not write standalone files).

---

## Plan 5b — Optuna Tuning of Model C (Model C-tuned) — shipped (run 2026-04-28)

**Closes:** TODO #26 follow-up "if tuning closes the gap, revisit adoption."

`LightGBMTunedModel` (Model C-tuned) lands as a third peer of `BaselineModel` (Model A) and `LightGBMModel` (Model C) under the existing `Model` Protocol. Subclass overrides only `_hyperparams_for(stat)`, `code_hash`, and `model_id`; all training and prediction logic is inherited from `LightGBMModel`. Tuned hyperparameters live in `data/tuned_params/lightgbm.json` (checked in, dense across all 24 (position, stat) entries, content-hashed into `model_id`). 24 per-(position, stat) Optuna studies × 50 trials with TPE sampler + median pruner via `LightGBMPruningCallback`; trial scorer = sum of 5 pinball losses on 2023 val. Train 2018-2021; early-stop val 2022; trial scorer 2023. Tuned params reused across all 4 backtest folds. Backtest snapshot keyed by `(position, year, metric, model_class)` extended from 768 → 1136 rows (368 new lightgbm-tuned rows; SAMPLED_SUMMARY-vs-QUANTILE family asymmetry pinned, TODO #28 still open).

### Per-position model_ids

| Position | Model A model_id | Model C model_id | Model C-tuned model_id |
|---|---|---|---|
| WR | `baseline:wr:6d955427:2018-2023` | `lightgbm:wr:a4dd5a82:2018-2023` | `lightgbm-tuned:wr:62df14ad:2018-2023` |
| QB | `baseline:qb:c98738f3:2018-2023` | `lightgbm:qb:06fadb3f:2018-2023` | `lightgbm-tuned:qb:fc902ed6:2018-2023` |
| RB | `baseline:rb:5a86c8ee:2018-2023` | `lightgbm:rb:fb169c0e:2018-2023` | `lightgbm-tuned:rb:5d69fdfe:2018-2023` |
| TE | `baseline:te:9c00025b:2018-2023` | `lightgbm:te:bd4c2a5b:2018-2023` | `lightgbm-tuned:te:89dafdb6:2018-2023` |

### Adoption-gate verdict — DO NOT ADOPT Model C-tuned as default

Spec §1.3 required Model C-tuned to beat Model A on three criteria. **All three failed.**

| Criterion | Threshold | Actual (C-tuned vs A) | Pass? |
|---|---|---|---|
| Composite RMSE strictly lower on >=12 of 16 cells; not worse by >1% on any cell | C-tuned < A on 12+ cells; max +1% worse | C-tuned strictly lower on 4/16; max +2.95% worse (TE 2024); 8/16 cells exceed 1% | **FAIL** |
| Spearman top-N within +-0.005 of A on every cell | All 16 within +-0.005 | 9/16 within tolerance; 7 fail; worst +0.0163 (QB 2021) | **FAIL** |
| Weekly mean [p10,p90] coverage no worse on any cell; mean improvement >= +0.02 | No regressions; mean delta >= +0.02 | C-tuned no worse on 5/16 (all 4 QB cells + TE 2021 marginal); mean delta -0.0630 | **FAIL** |

### Side-by-side metric comparison (16 cells)

A vs C vs C-tuned with the C-tuned − A deltas on RMSE, Spearman, and calibration. `RMSE Ctuned-A %` is the percentage delta on composite RMSE; positive = C-tuned worse; threshold is +1.00% per criterion 1.

| Cell | RMSE A | RMSE C | RMSE C-tuned | RMSE Ctuned-A % | MAE A | MAE C | MAE Ctuned | Spearman A | Spearman C | Spearman Ctuned | Spearman Ctuned-A | Calib A | Calib C | Calib Ctuned | Calib Ctuned-A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| QB 2021 | 7.8342 | 7.8386 | 7.7320 | -1.30% | 6.3606 | 6.4652 | 6.3581 | 0.9342 | 0.9501 | 0.9505 | +0.0163 | 0.6947 | 0.6857 | 0.7128 | +0.0180 |
| QB 2022 | 7.2261 | 7.2718 | 7.1220 | -1.44% | 5.7093 | 5.7649 | 5.6199 | 0.9669 | 0.9655 | 0.9668 | -0.0001 | 0.7458 | 0.7595 | 0.7854 | +0.0396 |
| QB 2023 | 7.3092 | 7.3213 | 7.2602 | -0.67% | 5.8796 | 5.9636 | 5.8887 | 0.9454 | 0.9560 | 0.9576 | +0.0122 | 0.7313 | 0.6955 | 0.7448 | +0.0134 |
| QB 2024 | 7.6995 | 7.5523 | 7.4845 | -2.79% | 6.0788 | 6.1338 | 6.0594 | 0.9383 | 0.9450 | 0.9460 | +0.0078 | 0.7018 | 0.6944 | 0.7281 | +0.0263 |
| RB 2021 | 6.8486 | 7.0688 | 6.9895 | +2.06% | 5.2108 | 5.6643 | 5.5522 | 0.9700 | 0.9565 | 0.9611 | -0.0089 | 0.7475 | 0.6008 | 0.6167 | -0.1308 |
| RB 2022 | 6.6359 | 6.8370 | 6.7032 | +1.01% | 5.0383 | 5.4613 | 5.3138 | 0.9658 | 0.9603 | 0.9674 | +0.0016 | 0.7415 | 0.6221 | 0.6379 | -0.1037 |
| RB 2023 | 6.3143 | 6.5069 | 6.4255 | +1.76% | 4.7179 | 5.0392 | 5.0039 | 0.9665 | 0.9600 | 0.9634 | -0.0032 | 0.7872 | 0.6354 | 0.6590 | -0.1281 |
| RB 2024 | 6.4860 | 6.6370 | 6.6242 | +2.13% | 4.9290 | 5.1591 | 5.1982 | 0.9753 | 0.9746 | 0.9763 | +0.0009 | 0.7568 | 0.6231 | 0.6307 | -0.1261 |
| TE 2021 | 5.3365 | 5.4636 | 5.3835 | +0.88% | 3.8932 | 4.1989 | 4.1390 | 0.9655 | 0.9598 | 0.9617 | -0.0038 | 0.7350 | 0.6398 | 0.6670 | -0.0680 |
| TE 2022 | 5.2498 | 5.3710 | 5.3197 | +1.33% | 3.6970 | 4.1489 | 4.0954 | 0.9615 | 0.9606 | 0.9621 | +0.0006 | 0.7647 | 0.6415 | 0.6700 | -0.0947 |
| TE 2023 | 4.9422 | 5.1324 | 5.0272 | +1.72% | 3.5439 | 4.0228 | 3.9442 | 0.9704 | 0.9709 | 0.9743 | +0.0039 | 0.7561 | 0.6711 | 0.6900 | -0.0662 |
| TE 2024 | 5.0804 | 5.2661 | 5.2302 | +2.95% | 3.7446 | 4.1408 | 4.1022 | 0.9620 | 0.9568 | 0.9582 | -0.0038 | 0.7345 | 0.6401 | 0.6735 | -0.0611 |
| WR 2021 | 6.7333 | 6.8837 | 6.8025 | +1.03% | 5.0891 | 5.4583 | 5.3996 | 0.9699 | 0.9624 | 0.9651 | -0.0048 | 0.6956 | 0.6107 | 0.6321 | -0.0635 |
| WR 2022 | 6.6255 | 6.7479 | 6.7149 | +1.35% | 5.0221 | 5.3711 | 5.3580 | 0.9767 | 0.9670 | 0.9690 | -0.0076 | 0.6970 | 0.6004 | 0.6151 | -0.0818 |
| WR 2023 | 6.5159 | 6.7129 | 6.5922 | +1.17% | 4.7814 | 5.2292 | 5.0860 | 0.9680 | 0.9590 | 0.9628 | -0.0051 | 0.7256 | 0.6220 | 0.6379 | -0.0877 |
| WR 2024 | 6.6728 | 6.7339 | 6.7218 | +0.74% | 4.9437 | 5.1907 | 5.2066 | 0.9739 | 0.9669 | 0.9673 | -0.0066 | 0.7109 | 0.6128 | 0.6177 | -0.0933 |

### Tuning helped — but not enough

Aggregate movement vs untuned Model C (Plan 5):

| Metric | Untuned C vs A | Tuned C vs A | Delta from tuning |
|---|---|---|---|
| RMSE: cells where C beats A | 1/16 | 4/16 | +3 cells |
| RMSE: max pct worse | +3.85% | +2.95% | -0.90 pp |
| Spearman: cells outside ±0.005 | 12/16 | 7/16 | -5 cells |
| Spearman: max abs delta | 0.0135 | 0.0163 | +0.0028 (worse on outlier) |
| Calibration: cells where C-tuned worse | 15/16 | 11/16 | -4 cells |
| Calibration: mean delta | -0.0857 | -0.0630 | +0.0227 |

**QB cells responded strongly to tuning.** All 4 QB years now strictly beat Model A on RMSE (-0.67% to -2.79%) and on calibration (+0.013 to +0.040 deltas; all positive). The hand-set defaults from Plan 5 had QBs landing in only 1 cell of "C wins"; tuning produced 4-for-4. QB tuning preferred shallow trees (`max_depth` mostly 3) with moderate `num_leaves` (33-127) and meaningful `reg_alpha` on count stats — i.e., regularize harder than the defaults did.

**RB / TE / WR cells improved but did not flip.** All three positions still regress on RMSE (1-3% worse), Spearman (mostly within tolerance now but some still outside), and calibration (still 6-13 pp under A). Tuning compressed the gaps but didn't eliminate them. The Plan 5 post-mortem hypothesis #1 (per-stat sub-models lack a shared prior; small-data positions overfit) and #4 (multi-output training would let the model borrow strength across stats) are the most plausible remaining mechanisms — tuning operates *within* per-stat sub-models and so cannot address the fundamental "each sub-model fits its own noise" problem.

### Why Model C-tuned still lost (refined hypothesis)

Plan 5's post-mortem listed four candidate mechanisms; tuning addresses only #2:

1. **No shared prior across the 5 quantile sub-models per (position, stat).** Tuning made each sub-model's hyperparameters more conservative on average (lower `learning_rate`, more aggressive `min_child_samples`), but each still fits independently. **Not addressed.**
2. **Hand-set hyperparameters were sub-optimal.** Tuning addressed this directly. QBs benefited materially; smaller-data positions partially benefited.
3. **5-quantile interpolation is too coarse.** Tuning didn't change the quantile grid. **Not addressed.**
4. **Per-stat independent training discards shared signal across stats.** Tuning operates at the per-stat level. **Not addressed.**

Mechanisms #1 and #4 jointly explain why QB tunes well while WR/RB/TE don't: QB has fewer rows but each row has one strong target (passing yards) with high signal-to-noise; the other QB stats are zero-inflated counts that benefit from the tuning's harder regularization. RB/TE/WR have similar-sized datasets but each (position, stat) sub-model is a small-data fit on its own — total dataset size is fine, but per-sub-model is starved. A multi-output gradient-boosted model trained jointly across the 6 stats per position (and arguably across all 4 positions) is the natural next experiment.

### Decision

**Default model selection:** Model A stays the production default. Model C and Model C-tuned both ship as peers; neither is adopted. **No Plan 5c is filed** — the diagnostic verdict is unambiguous on the criteria, and per-fold tuning would not address mechanisms #1 / #3 / #4.

**Pivot:** the next model-improvement track is one of:
- **Plan 6 — Model D ensemble.** Stacked predictor (Model A + Model C + Model C-tuned). Even with all three losing head-to-head against A on most cells, a per-cell weighted ensemble could beat A — particularly on QB where C-tuned now has a robust edge. Cheapest experiment given Plan 5b's infrastructure.
- **TODO #3 (PBP / EPA features).** Feature track. Independent of model class. Estimated 5-15% RMSE win on top of any model class.
- **TODO #23 (target decomposition).** Volume × efficiency factorization. Independent of model class. Estimated 3-10% RMSE win.
- **Future Plan: multi-output LightGBM / shared-prior training** — addresses mechanisms #1 / #4 directly. Not yet specced; would inherit the LightGBM machinery from Plan 5.

The user picks one in the next session. Plan 4 (public Python API + CLI verbs + free-tier hosting) remains the post-modeling milestone.

### Per-position model_ids on disk

Standalone artifacts at `models/artifacts/lightgbm-tuned-{pos}-2018-2023-{hash}.joblib`. Backtest harness regenerates per-fold artifacts via the feature cache; standalone artifacts are for ad-hoc prediction / sanity checks.

---

## Plan 5 — LightGBM with Quantile Regression (Model C) — shipped (run 2026-04-27)

**Closes:** TODO #26.

`LightGBMModel` (Model C) lands as a peer of `BaselineModel` (Model A) under
the existing `Model` Protocol. Per-stat sub-models trained at quantiles
[0.05, 0.10, 0.50, 0.90, 0.95]; per-row prediction sorts to enforce
non-crossing, clips to [0, inf) for non-negative stats, wraps in
`QuantileDistribution`, and runs through the unchanged `score_distribution`
scoring layer. New `DistributionFamily.QUANTILE` + codec branch.
`POSITION_DISPATCH` extended with `factories: dict[str, Callable]` keyed by
model class name. Backtest harness gains `--model {baseline,lightgbm,both}`;
snapshot file renamed `baseline_metrics.json` → `model_metrics.json` and
rows keyed by `(position, year, metric, model_class)` (400 → 768 rows; LightGBM
skips 32 season_calibration_* rows per the harness gate that limits
season-aggregation to SAMPLED_SUMMARY family — see Task 18 follow-up).

### Per-position model_ids

| Position | Model A model_id (current) | Model C model_id (this plan) |
|---|---|---|
| WR | (Plan 3e Phase 1: `baseline:wr:6d955427:2018-2023`) | `lightgbm:wr:a4dd5a82:2018-2023` |
| QB | (Plan 3e Phase 1: `baseline:qb:c98738f3:2018-2023`) | `lightgbm:qb:06fadb3f:2018-2023` |
| RB | (Plan 3e Phase 1: `baseline:rb:5a86c8ee:2018-2023`) | `lightgbm:rb:fb169c0e:2018-2023` |
| TE | (Plan 3e Phase 1: `baseline:te:9c00025b:2018-2023`) | `lightgbm:te:bd4c2a5b:2018-2023` |

### Adoption-gate verdict — DO NOT ADOPT Model C as default

Spec §1.3 required Model C to beat Model A on three criteria. **All three failed.**

| Criterion | Threshold | Actual | Pass? |
|---|---|---|---|
| Composite RMSE strictly lower on >=12 of 16 cells; not worse by >1% on any cell | C <= A on 12+ cells; max +1% worse | C strictly lower on 1/16; max +3.85% worse (TE 2023); 11/16 cells exceed 1% | **FAIL** |
| Spearman top-N within +-0.005 of A on every cell | All 16 within +-0.005 | 4/16 within tolerance; 12 fail; worst -0.0135 (RB 2021) | **FAIL** |
| Weekly mean [p10,p90] coverage no worse on any cell; mean improvement >= +0.02 | No regressions; mean delta >= +0.02 | C no worse on 1/16 (QB 2022 +0.0137); mean delta -0.0857 | **FAIL** |

### Side-by-side metric comparison (16 cells)

Per-cell deltas (Model C - Model A) and the cell winner. `tie` indicates the absolute pct-delta is within the tolerance band (0-1% on RMSE/MAE; ±0.005 on Spearman; ±0.005 on calibration). `A` / `C` indicate a strict winner.

| Cell | composite_rmse (A) | composite_rmse (C) | RMSE pct delta | composite_mae A | composite_mae C | spearman A | spearman C | spearman delta | calib_p10p90 A | calib_p10p90 C | calib delta | RMSE winner | MAE winner | Spearman winner | Calib winner |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| QB 2021 | 7.8342 | 7.8386 | +0.06% | 6.3606 | 6.4652 | 0.9342 | 0.9501 | +0.0159 | 0.6947 | 0.6857 | -0.0090 | tie | A | C | A |
| QB 2022 | 7.2261 | 7.2718 | +0.63% | 5.7093 | 5.7649 | 0.9669 | 0.9655 | -0.0014 | 0.7458 | 0.7595 | +0.0137 | tie | tie | tie | C |
| QB 2023 | 7.3092 | 7.3213 | +0.17% | 5.8796 | 5.9636 | 0.9454 | 0.9560 | +0.0106 | 0.7313 | 0.6955 | -0.0358 | tie | A | C | A |
| QB 2024 | 7.6995 | 7.5523 | -1.91% | 6.0788 | 6.1338 | 0.9383 | 0.9450 | +0.0067 | 0.7018 | 0.6944 | -0.0073 | C | tie | C | A |
| RB 2021 | 6.8486 | 7.0688 | +3.22% | 5.2108 | 5.6643 | 0.9700 | 0.9565 | -0.0135 | 0.7475 | 0.6008 | -0.1468 | A | A | A | A |
| RB 2022 | 6.6359 | 6.8370 | +3.03% | 5.0383 | 5.4613 | 0.9658 | 0.9603 | -0.0055 | 0.7415 | 0.6221 | -0.1195 | A | A | A | A |
| RB 2023 | 6.3143 | 6.5069 | +3.05% | 4.7179 | 5.0392 | 0.9665 | 0.9600 | -0.0065 | 0.7872 | 0.6354 | -0.1518 | A | A | A | A |
| RB 2024 | 6.4860 | 6.6370 | +2.33% | 4.9290 | 5.1591 | 0.9753 | 0.9746 | -0.0007 | 0.7568 | 0.6231 | -0.1337 | A | A | tie | A |
| TE 2021 | 5.3365 | 5.4636 | +2.38% | 3.8932 | 4.1989 | 0.9655 | 0.9598 | -0.0057 | 0.7350 | 0.6398 | -0.0951 | A | A | A | A |
| TE 2022 | 5.2498 | 5.3710 | +2.31% | 3.6970 | 4.1489 | 0.9615 | 0.9606 | -0.0009 | 0.7647 | 0.6415 | -0.1232 | A | A | tie | A |
| TE 2023 | 4.9422 | 5.1324 | +3.85% | 3.5439 | 4.0228 | 0.9704 | 0.9709 | +0.0005 | 0.7561 | 0.6711 | -0.0851 | A | A | tie | A |
| TE 2024 | 5.0804 | 5.2661 | +3.66% | 3.7446 | 4.1408 | 0.9620 | 0.9568 | -0.0052 | 0.7345 | 0.6401 | -0.0944 | A | A | A | A |
| WR 2021 | 6.7333 | 6.8837 | +2.23% | 5.0891 | 5.4583 | 0.9699 | 0.9624 | -0.0076 | 0.6956 | 0.6107 | -0.0849 | A | A | A | A |
| WR 2022 | 6.6255 | 6.7479 | +1.85% | 5.0221 | 5.3711 | 0.9767 | 0.9670 | -0.0096 | 0.6970 | 0.6004 | -0.0966 | A | A | A | A |
| WR 2023 | 6.5159 | 6.7129 | +3.02% | 4.7814 | 5.2292 | 0.9680 | 0.9590 | -0.0090 | 0.7256 | 0.6220 | -0.1036 | A | A | A | A |
| WR 2024 | 6.6728 | 6.7339 | +0.92% | 4.9437 | 5.1907 | 0.9739 | 0.9669 | -0.0070 | 0.7109 | 0.6128 | -0.0981 | tie | A | A | A |

### Why Model C lost — initial analysis

LightGBM-with-defaults systematically under-covers and underperforms Ridge on RB/TE/WR; only QBs see meaningful improvement. Plausible causes (none investigated in Plan 5; deferred):

1. **Quantile-loss training does not regularize against under-confidence.** The per-stat sub-models train independently at p5 / p10 / p50 / p90 / p95, with no shared prior. With ~6-15K rows per (position, stat), each sub-model fits noise that pushes the predicted interval inward. Ridge's L2 prior + post-hoc parametric variance (Plan 3e NB-2 for counts; Normal/Gamma for the rest) is more conservative.
2. **Hand-set hyperparameters, not tuned.** Plan 5 §1.3 explicitly deferred hyperparameter tuning to a focused follow-up "if results justify." `n_estimators=2000` + `learning_rate=0.05` + `num_leaves=31` is a reasonable starting point but not optimal for any specific stat.
3. **5-quantile interpolation is too coarse.** Tail accuracy depends on knot density; 5 knots over [0.05, 0.95] interpolates linearly between p10 and p50 (40% mass) and between p50 and p90 (40% mass) — coarse enough to lose structure where the underlying distribution has skew.
4. **Per-stat independent training discards shared signal across stats.** A multi-output model trained jointly across the 6 stats per position would let it borrow strength.

### Next steps

**Default model selection**: keep Model A as the production default. Model C ships as a peer for future iteration but is not adopted today.

**Followup plans (none in scope for Plan 5):**
1. **Plan 5b — Hyperparameter tuning for Model C.** Optuna-based search per (position, stat, quantile) sub-model. If tuning closes the gap, revisit adoption.
2. **Plan 6 — Model D (ensemble of Model A + Model C).** Even though Model C lost head-to-head, a stacked predictor (e.g., per-cell weighted average with weights fit on a held-out year) could still beat A alone, particularly on QB where C has a real edge. Worth trying once the adoption-gate infra is in place.
3. **TODO #28 (filed below)** — widen `aggregate_to_season` to accept `QUANTILE` family so LightGBM cells get season_calibration_* metrics.

After Plan 5 + (5b? 6?): Plan 4 (public Python API + CLI verbs + free-tier hosting), then Draft Hub.

### Per-position model_ids on disk

Standalone artifacts at `models/artifacts/lightgbm-{pos}-2018-2023-{hash}.joblib`. Backtest harness regenerates per-fold artifacts via the feature cache; standalone artifacts are for ad-hoc prediction / sanity checks.

---

## Plan 3e Phase 3 — Per-tertile bucketing — REVERTED (run 2026-04-27)

**Closes:** Nothing new — the routing change did not survive validation. TODO #22 stays closed against Plan 3e overall, but the shipped Plan 3e state is now Phase 0 (diagnostic CLI) + Phase 1 (NB for count stats); Phase 2 + Phase 3 are both attempted-and-reverted (infrastructure preserved).

After Phase 3 merged on-branch, the empirical signal was unambiguous: per-tertile variance bucketing regressed weekly mean coverage by **0.016** (0.726 → 0.710) and season mean coverage by **0.062** (0.461 → 0.399) vs the Plan 3d baseline. QB cells gained ~+0.02 weekly across all 4 years (their residuals are more uniformly homoscedastic across `mu_hat` tertiles, so bucketing produced near-equal per-bucket params and was harmless). RB/WR/TE cells regressed substantially (-0.025 to -0.037 weekly per cell, and as much as -0.135 on the worst season cell).

### Mechanism

The per-bucket variance estimator does not capture within-bucket residual asymmetry on positions whose low-pred buckets mix mostly-zero actuals with occasional big-game actuals. The bottom bucket gets a tighter std/shape/dispersion, which narrows the [p10, p90] interval on rows with the smallest predicted means — exactly where residuals are heteroscedastic *upward* (zero-inflated tails on count stats; right-skew on small-yards rows). Result: the central interval shrinks where actuals don't, and coverage drops. The right answer is quantile-based fitting (Plan 5 / quantile regression territory) — fit variance to minimize p10/p90 quantile loss directly rather than maximize residual likelihood.

### Decision

Revert Phase 3's routing in `BaselineModel.fit` and `BaselineModel.build_stat_distributions`. **Keep** the bucketing helpers (`_compute_tertile_cuts`, `_assign_bucket_indices`, `_per_bucket_normal_std_from_residuals`, `_per_bucket_gamma_alpha_from_residuals`, `_per_bucket_nb_dispersion_from_residuals`, `_per_bucket_student_t_params_from_residuals`), the widened `variance_params` value type (`float | list[float]`), and their unit tests as future infrastructure for plans that combine bucketing with non-symmetric within-bucket estimators or quantile-based fits.

### Verification

After revert + retrain + re-snapshot, the snapshot returns to Phase 1's baseline (commit `0078223`) **bit-for-bit**: weekly mean 0.733 → 0.733; season mean 0.428 → 0.428; max abs delta across all 400 metrics is 0.00000. Variance_params reverts to scalar shape (`{"std": X}` / `{"shape": X}` / `{"dispersion": X}`); per-position model_ids change because the source-file hash changes, but the underlying numbers don't.

### Final shipped state for Plan 3e

- Phase 0: diagnostic CLI (`scripts/diagnose_calibration.py`) + research report.
- Phase 1: `ParametricNegativeBinomial` for the 10 zero-inflated count stats (`*_tds`, `interceptions`, `fumbles_lost`); conditional MLE dispersion estimator with NB-2 / "size" parameterization; codec branch for the new family.
- Phase 2: Student-t routing for `*_yards` — attempted, reverted. `ParametricStudentT` class, codec branch, and `_student_t_params_from_residuals` estimator preserved as infrastructure.
- Phase 3: per-tertile variance bucketing across all routed families — attempted, reverted. Bucketing helpers + widened `variance_params` type preserved as infrastructure.

Spec calibration targets (min cell coverage ≥ 0.65; mean delta ≥ +0.10) still **not met** by the shipped state. Follow-up plans below.

### Follow-up plan candidates (post-merge brainstorming)

1. **ZIP (zero-inflated Poisson) for count cells** if NB still undercovers — handles the zero mass directly rather than via dispersion.
2. **Cross-week residual correlation modeling for season under-dispersion.** Season aggregation currently sums independent weekly draws; in reality, a player's good/bad weeks correlate (matchup quality, role, health). Modeling that correlation would widen season-total variance directly without touching weekly distributions. This is the canonical fix for the 0.30–0.50 season under-dispersion that has persisted through every Plan 3e attempt.
3. **Calibration-aware fitting.** Plan 3e fitted variance via residual MLE / method-of-moments; the empirical signal then says coverage missed. Direct fits that minimize a calibration loss (e.g., quantile loss at p10/p90) rather than a likelihood would close the loop. This is a structural shift in the fitting paradigm and worth its own spec — the bucketing infrastructure preserved on-branch is a natural building block here.

---

## Plan 3e Phase 3 — Per-tertile variance bucketing (run 2026-04-27, on branch `feat/plan-3e-calibration-tightening`)

**Closes:** Plan 3e overall (Phases 0 + 1 + 2-attempted-and-reverted + 3); TODO #22 closed.

Phase 3 is the cross-cutting fix: every (position, stat) cell now persists 33rd/67th-percentile cuts on `mu_hat` from the training set + a 3-element list of variance parameters (one per tertile bucket). At predict time, each row is routed to its bucket via `np.searchsorted` and the corresponding parameter is selected. Applies to all families currently in use (NORMAL, GAMMA, NEGATIVE_BINOMIAL).

**Phase 3 delivered:**
- `BaselineModel.variance_params` shape generalized from `dict[Stat, dict[str, float]]` to `dict[Stat, dict[str, float | list[float]]]`.
- 5 new helpers: `_compute_tertile_cuts`, `_assign_bucket_indices`, `_per_bucket_normal_std_from_residuals`, `_per_bucket_gamma_alpha_from_residuals`, `_per_bucket_nb_dispersion_from_residuals` (and `_per_bucket_student_t_params_from_residuals` as future infrastructure).
- `BaselineModel.fit` rewritten to compute tertile cuts + per-bucket parameters per family.
- `BaselineModel.build_stat_distributions` rewritten to look up bucket per row + select per-bucket parameter.
- Codec unchanged (per-row distributions still emit concrete scalar params); mixed-family regression test added.
- Standalone artifacts retrained.
- Snapshot regenerated.

### Coverage delta vs Plan 3d baseline (pre-Plan-3e at commit `fe55d5b`)

| Metric | Pre-Plan-3e (3d at `fe55d5b`) | Post-Phase-3 | Delta |
|---|---|---|---|
| Weekly mean `calibration_p10p90` | 0.726 | 0.710 | **-0.016** |
| Weekly min `calibration_p10p90` | 0.675 (QB/2021) | 0.663 (WR/2022) | -0.012 |
| Season mean `season_calibration_p10p90` | 0.461 | 0.399 | **-0.062** |
| Season min `season_calibration_p10p90` | 0.313 (QB/2022) | 0.293 (QB/2021) | -0.020 |
| ALL-32-cells mean `[p10, p90]` coverage delta | — | — | **-0.039** |
| ALL-32-cells min coverage | 0.313 | 0.293 | -0.020 |

**Compared to Phase 1 alone** (snapshot at `0078223`, pre-bucketing): weekly mean 0.733 → 0.710 (-0.023); season mean 0.428 → 0.399 (-0.030); all-32-cells mean delta -0.026.

**Per-cell weekly highlights:**
- QB cells gained on bucketing: 2021 +0.020, 2022 +0.024, 2023 +0.025, 2024 +0.019 (QB cells are now the only weekly cells with positive deltas vs Plan 3d).
- RB cells regressed -0.025 to -0.033 across all 4 years.
- TE cells regressed -0.017 to -0.032 across all 4 years.
- WR cells regressed -0.030 to -0.037 across all 4 years (WR/2024 is the worst weekly miss).

**Per-cell season highlights:**
- Worst season-coverage regressions: WR/2022 -0.135, RB/2024 -0.121, WR/2023 -0.119, RB/2021 -0.090, WR/2021 -0.096.
- Only QB/2023 (+0.013) and QB/2022 (0.000) season cells held or improved.

### Per-position model_ids

| Position | model_id |
|---|---|
| WR | `baseline:wr:a1fe2727:2018-2023` |
| QB | `baseline:qb:5333a44e:2018-2023` |
| RB | `baseline:rb:078c171c:2018-2023` |
| TE | `baseline:te:f460c50f:2018-2023` |

### Sample variance_params shape (one stat per family per position)

- WR receiving_yards (NORMAL): `{'bucket_cuts': [38.288, 55.137], 'std_per_bucket': [25.599, 33.781, 41.593]}`
- WR receptions (GAMMA): `{'bucket_cuts': [2.984, 4.253], 'shape_per_bucket': [1.752, 2.787, 3.822]}`
- WR receiving_tds (NEGATIVE_BINOMIAL): `{'bucket_cuts': [0.226, 0.334], 'dispersion_per_bucket': [4.828, 1000.0, 1000.0]}`
- QB passing_yards (NORMAL): `{'bucket_cuts': [220.099, 250.908], 'std_per_bucket': [87.075, 75.772, 76.945]}`
- QB passing_tds (NEGATIVE_BINOMIAL): `{'bucket_cuts': [1.294, 1.650], 'dispersion_per_bucket': [1000.0, 1000.0, 1000.0]}`
- RB rushing_yards (NORMAL): `{'bucket_cuts': [40.814, 57.406], 'std_per_bucket': [28.511, 33.948, 37.936]}`
- TE receiving_yards (NORMAL): `{'bucket_cuts': [27.605, 40.259], 'std_per_bucket': [19.853, 24.328, 33.730]}`

### Spec target verification

**Both spec targets MISSED — and Phase 3 regressed coverage rather than improving it.**

- Min cell coverage across all 32 cells: 0.293 (target ≥ 0.65). **Not met.** No appreciable movement from Plan 3d (0.313).
- Mean coverage delta across all 32 cells: -0.039 (target ≥ +0.10). **Not met; regressed.**

### Mechanism of the regression

Per-tertile bucketing reduces variance in the bottom + middle buckets relative to the unbucketed pooled estimate. The bottom bucket now uses a tighter std/shape/dispersion, which narrows the [p10, p90] interval on the half of the dataset with the smallest predicted means — exactly the half where residuals are heteroscedastic *upward* (zero-inflated tails on count stats; right-skew on small-yards rows). Result: the central interval shrinks where the actuals don't, and coverage drops.

QB cells are the exception (uniform +0.02 weekly gains): QB residual variance is more uniformly homoscedastic across mu_hat tertiles than RB/WR/TE, so bucketing produces ~equal per-bucket params and avoids the asymmetric narrowing effect. RB/WR/TE — where heteroscedasticity is sharpest — are exactly where bucketing hurts most.

### Known shortfalls / follow-up plans

Recommended follow-up plans (none of these is in scope for Plan 3e — they are post-merge work):

1. **Revert Phase 3 if RB/WR/TE coverage matters more than QB.** The Phase-1 snapshot (`0078223`) had better mean coverage than Phase 3 (0.733 vs 0.710 weekly; 0.428 vs 0.399 season). A clean revert to Phase 1 is a reasonable call. Plan 3e Phase 3 ships the per-tertile mechanism + tests; reversing the routing is a one-commit follow-up.
2. **Asymmetric residual modeling.** Bucketing collapses the residual distribution to a single std/shape per bucket, which still assumes symmetric tails within each bucket. The data has zero-inflation (count stats) and right-skew (small-yards rows) that bucketing on its own can't capture. Follow-up plans:
   - **ZIP (zero-inflated Poisson) for count cells** if NB still undercovers — handles the zero mass directly rather than via dispersion.
   - **Per-bucket family choice** rather than per-cell — e.g., use NORMAL on the high-mean bucket of receiving_yards but Student-t on the low-mean bucket where the long right tail dominates.
3. **Cross-week residual correlation modeling for season under-dispersion.** Season aggregation currently sums independent weekly draws; in reality, a player's good/bad weeks correlate (matchup quality, role, health). Modeling that correlation would widen season-total variance directly without touching weekly distributions. This is the canonical fix for the 0.30–0.50 season under-dispersion that has persisted through every Phase 3e attempt.
4. **Calibration-aware fitting.** Plan 3e fitted variance via residual MLE / method-of-moments; the empirical signal then says coverage missed. Direct fits that minimize a calibration loss (e.g., quantile loss at p10/p90) rather than a likelihood would close the loop. This is a structural shift in the fitting paradigm and worth its own spec.

---

## Plan 3e Phase 2 — Student-t for yards stats — ATTEMPTED + REVERTED (run 2026-04-27)

**Closes:** Nothing — the routing change did not survive validation. TODO #22
remains in progress; Phase 3 (variance bucketing) is the next attempt.

Phase 2 attempted to route every `*_yards` stat (passing/rushing/receiving
yards across QB/RB/TE/WR) from `NORMAL` to `STUDENT_T` based on Phase 0's
per-cell AIC signal favoring heavy tails (delta `[-2160, -317]` across the 5
yards-stat cells). The new `ParametricStudentT(loc, scale, df)` distribution
class, `DistributionFamily.STUDENT_T` enum value, codec branches, and
`_student_t_params_from_residuals` MLE estimator were all built and wired
through `BaselineModel.fit` and `build_stat_distributions`.

### Empirical finding: weekly coverage regressed by ~1.5–2 pts uniformly

After retraining the standalone artifacts and regenerating the snapshot,
weekly `calibration_p10p90` dropped roughly 1.5–2 pts uniformly across
RB/WR/TE cells with no offsetting season-coverage gain. The regression was
not noise: it appeared on every position-year cell that contained a `*_yards`
stat in the points decomposition.

### Root cause: heavy tails narrow the [p10, p90] shoulder

The mechanism is structural, not a bug. Student-t with the data's empirical
tail shape (df ~5–8 across the yards stats) puts more probability mass in
the extreme outer tails and *less* in the central shoulder of the
distribution than `NORMAL` at similar total std. Since our success metric
is `[p10, p90]` coverage — i.e. the share of actuals that land in the
central 80% interval — Student-t's heavier extremes shrink that interval
and lose coverage even when its full-distribution likelihood is better.

Phase 0's AIC signal was correct on its own terms (Student-t is a closer
fit to the full residual distribution), but **AIC is not a calibration
metric for the central interval.** The two objectives can diverge structurally
when the underlying data has heavy tails — preferring the heavier-tailed family
on AIC simultaneously deprefers it on `[p10, p90]` coverage.

### Decision: revert Phase 2 routing; keep the infrastructure

Per user decision, the factory routing was reverted in this commit. After
revert, **zero stats route to `STUDENT_T`** across all 4 positions. Yards
stats are back to `NORMAL` everywhere; the snapshot returns bit-exactly to
the Phase 1 baseline at commit `0078223` (verified via Step 7 coverage
delta = 0.000 on weekly mean / season mean / weekly min / season min).

The `ParametricStudentT` class, `DistributionFamily.STUDENT_T` enum value,
codec round-trip, `_student_t_params_from_residuals` estimator, and the
`STUDENT_T` branches in `BaselineModel.fit` / `build_stat_distributions`
all remain in-tree as future infrastructure. Their dedicated unit tests
(`tests/test_distributions/test_student_t.py`,
`tests/test_distributions/test_codec.py::test_codec_round_trip_student_t`,
and the two estimator tests in `tests/test_models/test_baseline.py`) are
unchanged. Any future plan can wire them up; the current code is correct
and validated.

### Lesson learned

Phase 0's family-fit AIC signal preferred Student-t for yards stats, and
that signal was technically correct: Student-t *is* a better full-
distribution fit than Normal on these residuals. But AIC measures full-
distribution agreement, not central `[p10, p90]` coverage — and Plan 3e's
success metric is calibration of the central interval. When the underlying
data is heavy-tailed, the two objectives can diverge structurally: the
heavier-tailed family wins on AIC and loses on central coverage. **For
Plan 3e and any future calibration-tightening phase, the family choice
must be evaluated against the calibration metric directly, not via AIC
proxy.**

### Forward pointer

Phase 3 (per-tertile variance bucketing) is the next attempt at improving
weekly coverage. It addresses a different Phase 0 root cause (pervasive
heteroscedasticity, 18 of 24 cells with variance-bucket ratio > 1.5) and
operates orthogonally to family choice — it can be wired on top of any
future family swap.

### Per-position model_ids (after revert)

| Position | model_id |
|---|---|
| WR | `baseline:wr:6d955427:2018-2023` |
| QB | `baseline:qb:c98738f3:2018-2023` |
| RB | `baseline:rb:5a86c8ee:2018-2023` |
| TE | `baseline:te:9c00025b:2018-2023` |

(Code hashes rotate from Phase 1's because the `baseline.py` module docstring
+ `_*_DIST_FAMILIES` dicts changed.)

---

## Plan 3e Phase 1 — Negative Binomial for count stats (run 2026-04-27, on branch `feat/plan-3e-calibration-tightening`)

**Closes:** TODO #22 progress; Phase 0 complete; Phases 2-3 in progress on this branch.

Phase 1 routes the 10 zero-inflated count stats (every `*_tds` + `interceptions` + `fumbles_lost` across QB/RB/TE/WR) from GAMMA to NEGATIVE_BINOMIAL via the new `ParametricNegativeBinomial` family. Conditional MLE estimator (`_negative_binomial_dispersion_from_residuals`) fits dispersion per stat, addressing Phase 0's marginal-vs-conditional AIC asymmetry caveat in production.

**Phase 1 delivered:**
- `ParametricNegativeBinomial(mean, dispersion)` distribution class implementing the Distribution Protocol; standard NB-2 parameterization (var = mean + mean²/dispersion).
- `DistributionFamily.NEGATIVE_BINOMIAL` enum value + codec branches in `pack_per_stat_params` / `unpack_per_stat_params`.
- `_negative_binomial_dispersion_from_residuals` conditional-MLE estimator (`scipy.optimize.minimize_scalar` bounded; `_NB_DISPERSION_CLIP = (0.01, 1000.0)`).
- `BaselineModel.fit` and `BaselineModel.build_stat_distributions` route NB stats correctly.
- All 4 per-position factories (_WR/QB/RB/TE_DIST_FAMILIES) updated.
- Standalone artifacts retrained (4 `models/artifacts/baseline-{pos}-...joblib` files; new `model_id` per position because `code_hash` rotates).
- Snapshot regenerated; gate passes.
- Bug fix landed mid-phase (commit `865ccfb`): inverted `_scipy_n_p()` conversion was producing wrong NB variance; fixed to standard NB-2.

### Coverage delta vs Phase 0 baseline

Pre-Phase-1 baseline = Plan 3d's snapshot at merge commit `fe55d5b`.

| Metric | Pre-Phase-1 | Post-Phase-1 | Delta |
|---|---|---|---|
| Weekly mean `calibration_p10p90` | 0.726 | 0.733 | +0.007 |
| Weekly min `calibration_p10p90` | 0.675 (QB/2021) | 0.695 (QB/2021) | +0.020 |
| Season mean `season_calibration_p10p90` | 0.461 | 0.428 | -0.033 |
| Season min `season_calibration_p10p90` | 0.313 (QB/2022) | 0.293 (QB/2021) | -0.020 |

**Weekly coverage improved modestly across all positions, with the largest gains on QB and TE:**
- QB cells: 2021 +0.020, 2023 +0.022 (the Phase 0 diagnostic flagged QB as the worst-calibrated position).
- TE cells: 2022 +0.015, 2023 +0.021, 2024 +0.018.
- WR/RB cells: small mixed deltas in `[-0.009, +0.004]`, all within tolerance.

**Season coverage regressed across most cells.** This is an expected secondary effect: Phase 0's GAMMA fits had inflated variance on count stats, so when independent weekly distributions were summed for the season Monte Carlo, the over-wide weekly tails partially compensated for the missing inter-week covariance. Replacing GAMMA with NB-2 (which correctly tightens count-stat variance per the conditional MLE fit) removes that compensating slack, exposing the true season-aggregation under-dispersion. Worst-affected cells: WR/2022 -0.074, WR/2023 -0.075, RB/2024 -0.073, RB/2022 -0.059. Phase 2 (Student-t for yards) and Phase 3 (variance bucketing) should not directly address this, but season-level inter-week correlation (a Plan-3e follow-up or post-3e item) will.

**Per-stat MAE/RMSE shifts on NB-routed stats are below the 0.01 noise floor across all 16 cells** — NB-2 and GAMMA agree on the conditional mean by construction; only the variance/shape changes, which feeds into calibration metrics, not point-prediction metrics.

### Per-position model_ids

| Position | model_id |
|---|---|
| WR | `baseline:wr:6964f45a:2018-2023` |
| QB | `baseline:qb:178a0438:2018-2023` |
| RB | `baseline:rb:0d8180b1:2018-2023` |
| TE | `baseline:te:ae33da15:2018-2023` |

### Next: Phase 2 (Student-t for yards stats) on this same branch.

---

## Plan 3e Phase 0 — Calibration diagnostic (run 2026-04-26, on branch `feat/plan-3e-calibration-tightening`)

**Closes:** None. TODO #22 (Plan 3e calibration tightening) stays open — Phase 0
delivers the diagnostic only; the full Plan 3e tightening closes #22. Phase 0
surfaced 3 root causes that the spec amendment (next gate) will translate into
Phase 1+ implementation work.

Phase 0 = a `scripts/diagnose_calibration.py` CLI plus a research report
(`docs/superpowers/research/2026-04-26-calibration-diagnosis.md`) that fits
alternative distribution families against per-row residuals from the latest
backtest run and identifies why weekly + season `[p10, p90]` coverage
under-disperses to 0.30–0.55 vs the 0.80 target. The spec amendment that adds
Phase 1+ implementation phases to
`docs/superpowers/specs/2026-04-26-plan-3e-calibration-tightening-design.md` is
the next gate before any model code changes (per spec section 3 decision gate).

### Diagnostic findings (3 root causes)

1. **Zero-inflated count stats are catastrophically miscalibrated under
   GAMMA.** `coverage_p10p90 = 0.0` across every (position, stat) cell for
   `*_tds`, `interceptions`, and `fumbles_lost` — the fitted GAMMA's p10 sits
   above zero while the modal residual is exactly zero. Root cause is family
   choice; the recommendation is a family swap to negative-binomial / zero-
   inflated negative binomial.
2. **Continuous yards stats are heavy-tailed.** Student-t fits beat Normal on
   AIC by `delta in [-2160, -317]` across 5 yards-stat cells (passing/rushing/
   receiving × position). Recommendation is a family swap to Student-t for the
   `*_yards` stats.
3. **Heteroscedasticity is pervasive.** 18 of 24 (position, stat) cells have
   variance-bucket ratio > 1.5 (top vs bottom predicted-mean tertile).
   Variance bucketing is needed independent of family choice and combines with
   the family swaps above.

See `docs/superpowers/research/2026-04-26-calibration-diagnosis.md` for the full
per-cell table, recommended fixes, and selection methodology.

### Next gate

The spec amendment (Plan 3e Phase 1+) is the next gate before any model code
changes. Re-invocation of `superpowers:brainstorming` happens in the next
user-driven session to scope Phase 1 (family-family swaps), Phase 2 (variance
bucketing), and a final regression-gate phase against the 3d snapshot.

---

## Plan 3d — Real Monte Carlo season aggregation (run 2026-04-26, on branch `feat/plan-3d-monte-carlo-season`)

**Closes:** TODO #13 (per-row seeds), TODO #14 (SAMPLED_SUMMARY family), TODO #19 (gate non-determinism by demonstration).

Held-out years: 2021–2024 (same as Plan 3c). Snapshot at 400 rows
(368 weekly metrics from 3c + 32 new season-calibration rows from 3d).
Full gate runtime: 292.73 seconds.

### Composite metrics by (position, year)

| Position | Year | composite_rmse | composite_mae | spearman_topN | calibration_p10p90 | calibration_le_p90 |
|---|---|---|---|---|---|---|
| QB | 2021 | 7.841 | 6.357 | 0.933 | 0.675 | 0.857 |
| QB | 2022 | 7.240 | 5.703 | 0.968 | 0.737 | 0.845 |
| QB | 2023 | 7.324 | 5.868 | 0.945 | 0.709 | 0.831 |
| QB | 2024 | 7.722 | 6.072 | 0.938 | 0.699 | 0.842 |
| RB | 2021 | 6.864 | 5.147 | 0.970 | 0.745 | 0.846 |
| RB | 2022 | 6.631 | 4.965 | 0.967 | 0.746 | 0.851 |
| RB | 2023 | 6.322 | 4.641 | 0.967 | 0.791 | 0.867 |
| RB | 2024 | 6.487 | 4.853 | 0.975 | 0.766 | 0.863 |
| TE | 2021 | 5.352 | 3.856 | 0.966 | 0.727 | 0.845 |
| TE | 2022 | 5.282 | 3.670 | 0.960 | 0.750 | 0.830 |
| TE | 2023 | 4.978 | 3.527 | 0.969 | 0.735 | 0.821 |
| TE | 2024 | 5.101 | 3.712 | 0.962 | 0.717 | 0.823 |
| WR | 2021 | 6.746 | 5.040 | 0.970 | 0.700 | 0.827 |
| WR | 2022 | 6.633 | 4.975 | 0.977 | 0.693 | 0.831 |
| WR | 2023 | 6.531 | 4.737 | 0.968 | 0.723 | 0.832 |
| WR | 2024 | 6.693 | 4.899 | 0.975 | 0.707 | 0.825 |

Drift from Plan 3c snapshot was within tolerance for every weekly metric
(largest absolute drift: `RB/2021/calibration_p10p90` 0.7536 -> 0.7452,
abs delta 0.0084 vs 0.03 tolerance; largest relative drift: `RB/2024/composite_mae`
+0.165% vs 5% tolerance). 77 of 368 existing rows show non-zero drift; all
are within tolerance. Cause: the per-row seed change in `score_distribution`
(closes TODO #13) reorders Monte Carlo draws, but the underlying regression
math is unchanged. See `/tmp/3d-pre-snapshot-drift.txt` for the raw
`--check` output.

### Season-total calibration (new in Plan 3d)

| Position | Year | season_calibration_p10p90 | season_calibration_le_p90 |
|---|---|---|---|
| QB | 2021 | 0.317 | 0.976 |
| QB | 2022 | 0.313 | 0.928 |
| QB | 2023 | 0.388 | 0.900 |
| QB | 2024 | 0.377 | 0.935 |
| RB | 2021 | 0.521 | 0.896 |
| RB | 2022 | 0.478 | 0.853 |
| RB | 2023 | 0.413 | 0.857 |
| RB | 2024 | 0.516 | 0.879 |
| TE | 2021 | 0.500 | 0.925 |
| TE | 2022 | 0.474 | 0.853 |
| TE | 2023 | 0.540 | 0.876 |
| TE | 2024 | 0.432 | 0.890 |
| WR | 2021 | 0.505 | 0.881 |
| WR | 2022 | 0.563 | 0.898 |
| WR | 2023 | 0.562 | 0.881 |
| WR | 2024 | 0.479 | 0.863 |

Season-total `[p10, p90]` coverage is well below target (0.80) — typically
0.30–0.55 across cells, worst on QB (0.31–0.39). This inherits 3c's weekly
under-dispersion: when independent under-dispersed weekly distributions are
summed (with no covariance), the season distribution under-disperses further
because variances add but the systematic miss does not cancel. `<= p90`
coverage is closer to target (0.85–0.98) — the upper-tail stretch from
gamma summation partially masks the under-dispersion at p10. Plan 3e is
the calibration-tightening follow-up.

### Decision log (Plan 3d)

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-26 | params blob format = per-stat distribution params | Three orders of magnitude smaller than persisting full sample arrays; decomposable; deterministic regeneration via seed. |
| 2026-04-26 | Per-row seed = sha256 of `(gsis_id, season, week, ruleset.name)` truncated to 32 bits | Deterministic across processes (Python `hash()` is salt-randomized via PYTHONHASHSEED); independent across rows; reproducible. |
| 2026-04-26 | Aggregator regenerates per-week samples rather than persisting them | Storage 1000x smaller; regeneration is O(seconds); samples are deterministic given seed. |
| 2026-04-26 | Modal-position resolution for traded players | Deterministic; rare edge case; documented in docstring. |
| 2026-04-26 | Calibration tightening (MLE gamma alpha / variance buckets) explicitly deferred to Plan 3e | 3d's snapshot reflects under-dispersed calibration as the regression floor; tightening is a separable model-quality improvement. |

### Current status (as of 2026-04-26)

**Projections Core — Plan 3d (real Monte Carlo season aggregation) merged to `main` at commit `fe55d5b` (PR #9).**

**Predecessors:**
- Plan 1 (Foundations) merged at `8f02a6c`.
- Plan 2a merged at `7926090`; Plan 2b merged at `af325ea`.
- Plan 3a (WR Model A baseline) merged at `598ab9c`.
- Plan 3b (QB / RB / TE Model A baselines) merged at `c4a0401`.
- Plan 3c (walk-forward backtest gate) merged at `3db71a6` (PR #8).

### Next action

**Recommended: Plan 3e — calibration tightening.** Replace
`_gamma_alpha_from_residuals`'s method-of-moments with an MLE fit, and/or
add per-stat residual-variance bucketing by predicted-mean tertile, to
move weekly + season calibration coverage toward 0.80. The under-dispersion
shows up most acutely on QB season totals (p10–p90 coverage 0.31–0.39);
expect the largest tightening to come from QB-stat MLE fits.

---

## Plan 3c — Walk-forward backtest gate (run 2026-04-26, on branch `feat/plan-3c-backtest-harness`)

Held-out years: 2021, 2022, 2023, 2024 (4 years × 4 positions = 16 fits per gate run).
Train window: expanding from 2018 → year-1.
Snapshot file: `tests/backtest/baseline_metrics.json` (368 rows committed).
Gate: `pytest -m backtest --run-backtest` — opt-in, pre-PR. Full run: 133 seconds.
Default-on smoke: `tests/backtest/test_backtest_smoke.py` — one (WR, 2024) cell, ~15s.

### Composite metrics by (position, year)

| Position | Year | composite_rmse | composite_mae | spearman_topN | calibration_p10p90 | calibration_le_p90 |
|---|---|---|---|---|---|---|
| QB | 2021 | 7.846 | 6.364 | 0.933 | 0.677 | 0.860 |
| QB | 2022 | 7.234 | 5.702 | 0.967 | 0.740 | 0.848 |
| QB | 2023 | 7.323 | 5.868 | 0.945 | 0.712 | 0.834 |
| QB | 2024 | 7.714 | 6.068 | 0.939 | 0.702 | 0.844 |
| RB | 2021 | 6.868 | 5.143 | 0.970 | 0.754 | 0.849 |
| RB | 2022 | 6.635 | 4.963 | 0.967 | 0.753 | 0.851 |
| RB | 2023 | 6.324 | 4.636 | 0.967 | 0.796 | 0.868 |
| RB | 2024 | 6.486 | 4.845 | 0.975 | 0.769 | 0.862 |
| TE | 2021 | 5.351 | 3.857 | 0.966 | 0.720 | 0.841 |
| TE | 2022 | 5.278 | 3.671 | 0.960 | 0.753 | 0.831 |
| TE | 2023 | 4.973 | 3.527 | 0.970 | 0.738 | 0.825 |
| TE | 2024 | 5.098 | 3.714 | 0.962 | 0.716 | 0.821 |
| WR | 2021 | 6.743 | 5.044 | 0.970 | 0.698 | 0.827 |
| WR | 2022 | 6.631 | 4.979 | 0.977 | 0.694 | 0.833 |
| WR | 2023 | 6.529 | 4.742 | 0.968 | 0.726 | 0.833 |
| WR | 2024 | 6.691 | 4.903 | 0.975 | 0.702 | 0.825 |

### Naive baseline comparison (informational)

Naive = per-player trailing-4-game stat mean, with cold-start fallback to
per-position mean. **Model A beats naive on composite RMSE by 5–11% on
every (position, year) cell** — no inverted cells.

| Position | Naive composite RMSE range | Model A vs naive |
|---|---|---|
| QB | 7.83 – 8.53 | -6.2% to -10.8% |
| RB | 6.77 – 7.45 | -5.4% to -7.8% |
| TE | 5.30 – 5.67 | -5.6% to -6.2% |
| WR | 7.02 – 7.17 | -5.5% to -7.3% |

Spearman top-N: model and naive are tied within ±0.01 across all 16 cells.
Trailing-4-mean is already a very strong rank-correlation baseline (because
"good players keep being good"); Model A's value-add is in lower
RMSE / MAE on per-stat and composite metrics, not in ranking signal.

Calibration (`[p10, p90]` coverage): 0.67–0.80 across cells, target 0.80 — under-dispersed
in the same direction as 3a/3b's WR sanity check. Plan 3d's MLE-fit gamma α / variance
bucketing should tighten this; Plan 3c locks the current numbers in as the regression floor.

### Phase 6 unplanned-but-necessary fixes

Two issues surfaced during the first end-to-end run; both fixed in scope:

- **`score_distribution` perf vectorization** (commit `dc122a7`). The original per-sample Python loop building a Pydantic StatLine per sample × per stat × per row dominated harness runtime — 20–30 minutes for the full 16-cell run. Spec section 1.2 had deferred this perf TODO to Plan 3d, but the runtime made the gate functionally unrunnable, so vectorization was pulled forward. Math is bit-identical (linear scoring rule + same RNG draw order); existing scoring tests pass unchanged. Full gate now runs in 133s.
- **`tests/conftest.py` marker filter** (commit `4b5aea0`). The original `"backtest" in item.keywords` filter over-matched any test under `tests/backtest/` (pytest's keywords include path-derived components), wrongly skipping the default-on smoke test under the `--run-backtest` gate. Fixed by switching to `item.get_closest_marker("backtest")`. Network filter switched to the same idiom for consistency.

---

## Plan 3b — 2024 sanity check (run on branch `feat/plan-3b-qb-rb-te-baseline`)

Held-out year is 2024 (same as 3a; `nfl_data_py` has not yet published 2025). Each position trained on 2018-2023. Per-position evals are stdout-only — Plan 3c owns CI threshold gating.

### WR (retrained under Plan 3b's `BaselineModel` constructor)

```
Loading artifact: models\artifacts\baseline-wr-2018-2023-a2f581cf.joblib
model_id: baseline:wr:a2f581cf:2018-2023

=== WR 2024 sanity check (n=2048 player-weeks) ===

-- Per-stat fit --
            receptions  rmse= 2.051  mae= 1.543  mean_pred= 2.892  mean_actual= 3.116
       receiving_yards  rmse=31.198  mae=22.938  mean_pred=36.237  mean_actual=39.204
         receiving_tds  rmse= 0.495  mae= 0.347  mean_pred= 0.212  mean_actual= 0.256
         rushing_yards  rmse= 3.944  mae= 1.914  mean_pred= 1.311  mean_actual= 1.005
           rushing_tds  rmse= 0.086  mae= 0.017  mean_pred= 0.010  mean_actual= 0.007
          fumbles_lost  rmse= 0.122  mae= 0.033  mean_pred= 0.018  mean_actual= 0.015

-- Composite (PPR points) --
  mean prediction:  rmse=6.780  mae=4.910
  top-N season-total rank correlation (Spearman, all WRs): 0.971

-- Calibration --
  fraction in [p10, p90]: 0.708  (target ~ 0.80)
  fraction <= p90:        0.815  (target ~ 0.90)

=== End sanity check (informational; not a CI gate) ===
```

### QB

```
Loading artifact: models\artifacts\baseline-qb-2018-2023-3907548e.joblib
model_id: baseline:qb:3907548e:2018-2023

=== QB 2024 sanity check (n=684 player-weeks) ===

-- Per-stat fit --
         passing_yards  rmse=84.538  mae=68.175  mean_pred=199.516  mean_actual=192.405
           passing_tds  rmse= 1.068  mae= 0.866  mean_pred= 1.219  mean_actual= 1.219
         interceptions  rmse= 0.829  mae= 0.699  mean_pred= 0.684  mean_actual= 0.585
         rushing_yards  rmse=17.880  mae=13.369  mean_pred=18.163  mean_actual=17.197
           rushing_tds  rmse= 0.440  mae= 0.287  mean_pred= 0.191  mean_actual= 0.171
          fumbles_lost  rmse= 0.396  mae= 0.304  mean_pred= 0.205  mean_actual= 0.171

-- Composite (PPR points) --
  mean prediction:  rmse=7.810  mae=6.281
  top-N season-total rank correlation (Spearman, all QBs): 0.928

-- Calibration --
  fraction in [p10, p90]: 0.667  (target ~ 0.80)
  fraction <= p90:        0.860  (target ~ 0.90)

=== End sanity check (informational; not a CI gate) ===
```

### RB

```
Loading artifact: models\artifacts\baseline-rb-2018-2023-a7f565e9.joblib
model_id: baseline:rb:a7f565e9:2018-2023

=== RB 2024 sanity check (n=1316 player-weeks) ===

-- Per-stat fit --
         rushing_yards  rmse=30.294  mae=22.628  mean_pred=38.617  mean_actual=39.458
           rushing_tds  rmse= 0.531  mae= 0.373  mean_pred= 0.267  mean_actual= 0.296
            receptions  rmse= 1.523  mae= 1.174  mean_pred= 1.751  mean_actual= 1.734
       receiving_yards  rmse=15.410  mae=11.127  mean_pred=12.767  mean_actual=13.127
         receiving_tds  rmse= 0.248  mae= 0.118  mean_pred= 0.065  mean_actual= 0.064
          fumbles_lost  rmse= 0.213  mae= 0.093  mean_pred= 0.052  mean_actual= 0.047

-- Composite (PPR points) --
  mean prediction:  rmse=6.517  mae=4.802
  top-N season-total rank correlation (Spearman, all RBs): 0.975

-- Calibration --
  fraction in [p10, p90]: 0.773  (target ~ 0.80)
  fraction <= p90:        0.851  (target ~ 0.90)

=== End sanity check (informational; not a CI gate) ===
```

### TE

```
Loading artifact: models\artifacts\baseline-te-2018-2023-4706d589.joblib
model_id: baseline:te:4706d589:2018-2023

=== TE 2024 sanity check (n=1081 player-weeks) ===

-- Per-stat fit --
            receptions  rmse= 1.911  mae= 1.372  mean_pred= 2.271  mean_actual= 2.596
       receiving_yards  rmse=22.476  mae=16.371  mean_pred=23.030  mean_actual=26.175
         receiving_tds  rmse= 0.397  mae= 0.286  mean_pred= 0.191  mean_actual= 0.166
         rushing_yards  rmse= 4.423  mae= 0.399  mean_pred= 0.131  mean_actual= 0.256
           rushing_tds  rmse= 0.114  mae= 0.008  mean_pred= 0.002  mean_actual= 0.006
          fumbles_lost  rmse= 0.138  mae= 0.035  mean_pred= 0.016  mean_actual= 0.019

-- Composite (PPR points) --
  mean prediction:  rmse=5.143  mae=3.716
  top-N season-total rank correlation (Spearman, all TEs): 0.960

-- Calibration --
  fraction in [p10, p90]: 0.741  (target ~ 0.80)
  fraction <= p90:        0.821  (target ~ 0.90)

=== End sanity check (informational; not a CI gate) ===
```

The WR retrain in Phase 6 produced a new `model_id` (`a2f581cf` vs 3a's `925f492b`) because Plan 3b modified `baseline.py` (which is part of the hashed code-files list); substantively the predictions match the merged 3a artifact's output to within numerical noise.

---

## Plan 3a — 2024 WR sanity check (run 2026-04-25, on branch `feat/plan-3a-wr-model-a`)

Held-out year is 2024 not 2025 (spec called for 2025; `nfl_data_py` has not yet published 2025 data).

```
Loading artifact: models/artifacts/wr-baseline-2018-2023-925f492b.joblib
model_id: baseline:wr:925f492b:2018-2023

=== 2024 sanity check (n=2048 player-weeks) ===

-- Per-stat fit --
            receptions  rmse= 2.049  mae= 1.541  mean_pred= 2.900  mean_actual= 3.116
       receiving_yards  rmse=31.186  mae=22.946  mean_pred=36.331  mean_actual=39.204
         receiving_tds  rmse= 0.495  mae= 0.348  mean_pred= 0.212  mean_actual= 0.256
         rushing_yards  rmse= 3.945  mae= 1.917  mean_pred= 1.314  mean_actual= 1.005
           rushing_tds  rmse= 0.086  mae= 0.017  mean_pred= 0.010  mean_actual= 0.007
          fumbles_lost  rmse= 0.122  mae= 0.033  mean_pred= 0.018  mean_actual= 0.015

-- Composite (PPR points) --
  mean prediction:  rmse=6.775  mae=4.908
  top-N season-total rank correlation (Spearman, all WRs): 0.971

-- Calibration --
  fraction in [p10, p90]: 0.708  (target ~ 0.80)
  fraction <= p90:        0.816  (target ~ 0.90)
```

Soft-threshold check vs. spec §6.3:
- Spearman top-30 correlation ≥ 0.4 — **MET** (0.971 — very high, the model captures relative WR ranking well).
- Calibration `[p10, p90]` coverage in 70–90% range — **borderline MET** (70.8%; right at the lower bound). The predicted distributions are slightly too narrow (under-dispersed). Plan 3c's backtest harness can formalize this and motivate either MLE-fit gamma α (TODO note in spec §3.4) or per-stat residual variance buckets.
- Per-stat RMSE within 2× of naive-baseline RMSE — **n/a until we compute the naive baseline**; track for future.

Per-stat means are systematically slightly *under* actual (e.g., receptions 2.90 vs 3.12, receiving_yards 36.3 vs 39.2) — Ridge has shrunk toward the league mean, which is expected behavior. The bias is small enough that the rank correlation is preserved.

**Plan 3a deliverable: pipeline works end-to-end on real data.** Bad numbers would feed into Plan 3c's threshold-setting; the sanity numbers here are good enough that the pipeline is the load-bearing artifact, not the model itself.

---

## Current status (as of 2026-05-07)

**Projections Core — Weather family probe SHIPPED with verdict SIGNAL via lgb-nb augment composite (RB + WR). Sixth family probe in Track 2; second consecutive SIGNAL.** The full multi-track scoreboard:

**Track 2A (team-level PBP families):**
- PR #20 (pace/PROE/AYPS/EPA-resid bundle, 2026-04-30) — **SIGNAL via RB**; integrated into `RbFeaturesSchema` in PR #21 (`(BaselineModel, RB)` adoption gate -0.0124 fpts).
- PR #22 (WR/TE receiver-level air-yards / aDOT bundle, 2026-05-01) — **NULL durable**.
- PR #23 (red-zone bundle, 2026-05-02) — **NULL durable**; QB augment lgb-nb regression at +0.0268.
- PR #24 (pressure bundle + Track 2B, 2026-05-02 / 2026-05-03) — **pressure NULL durable**; QB augment lgb-nb regression at +0.0276; Track 2B: RB PBP cols transfer directionally-favorably to all 4 tree-model classes.

**Track 2 player-trajectory features:**
- **PR #25 (trajectory probe, 2026-05-03) — SIGNAL via WR (both model classes) + TE (lgb-nb only).** Three ADOPT cells: WR augment baseline -0.0414 fpts, WR augment lgb-nb -0.0194 fpts, TE augment lgb-nb -0.0107 fpts.
- **PR #26 (WR trajectory integration, 2026-05-04) — ADOPT** on `(BaselineModel, WR)` at -0.0371 fpts (CI [-0.0567, -0.0172]); 4 of 5 model classes ADOPT in the gate. Probe-vs-gate calibration: probe -0.0414, gate -0.0371 (~0.004 fpts gap, within probe CI).
- **PR #27 (TE trajectory integration, 2026-05-04) — ADOPT** on `(LightGBMNbModel, TE)` at -0.0090 fpts (CI [-0.0171, -0.0013]). First production integration in the project to bind on a non-default model class (TE production routes to baseline; lgb-nb is where the probe's signal lived). Establishes the precedent for shipping schema changes for non-default model classes.

**Track 2 weather features (this entry):**
- **PR #28 (weather probe, 2026-05-07, this) — SIGNAL via lgb-nb augment composite (RB + WR ADOPT)**. Two ADOPT cells: lgb-nb augment RB -0.0081 fpts (CI [-0.0163, -0.0005]), lgb-nb augment WR -0.0110 fpts (CI [-0.0172, -0.0049]). **First probe where signal lives only in lgb-nb composite, not in BaselineModel** — RidgeCV cannot extract the bundle's non-linear thresholds (`is_high_wind` ≥20, surface category) even with explicit boolean encoding; tree splits do. Greenlights a combined RB+WR integration plan routed through `LightGBMNbModel` only (PR #27 precedent).

**Recurring QB augment regression — pattern continues but milder.** PR #23, #24 had QB augment lgb-nb composite regressions (+0.0268, +0.0276). PR #25 extended to both BaselineModel (+0.0382) and lgb-nb (+0.0233). PR #28 (weather): per-stat-only regression on QB rushing_yards 2023 (+0.0812), pooled QB Phase 2 lgb-nb augment is +0.0077 fpts (CI brackets 0, NOT REGRESSION). Plausibly weather information already partially captured by `roof_dome` + `implied_team_total` doesn't deliver enough new QB-relevant signal to trigger the QB-specific overfit pattern as strongly.

**Spec deviation in PR #28 — coverage threshold relaxation.** Weather probes invoked with `--coverage-threshold 0.90` (vs spec's 0.95 default). Outdoor-weather NaN rate measured at 8.39% (per `reports/feature_probe_weather_override_audit.md`) — upstream `nfl_data_py` data quality, concentrated in older 2018-2019 seasons. Per-(position, season) coverage in 2021-2024 eval window is uniformly ≥92%. Bias is symmetric across baseline + candidate arms under the probe's left-merge join. On par with PR #23's 0.90 precedent; shallower than PR #25's 0.35 (deepest in Track 2 history).

**Track 2B finding:** the RB PBP cols' signal transfers to the lightgbm family (which auto-derives features from the schema dynamically) but with smaller and noisier effects than the baseline lift. Untuned `lightgbm` shows the strongest point estimate (-0.0141 fpts, CI [-0.0278, +0.0005] — just barely brackets 0 on the upper bound); `lightgbm-tuned` / `lightgbm-nb` / `ensemble` all show -0.006 to -0.010 fpts directional improvement with CIs that bracket zero. No model class regresses on RB. No spillover to QB/WR/TE (their schemas weren't touched). See `reports/track2b_rb_pbp_other_models.md` for the per-cell table + methodology + appendix on probe attempts that did NOT work for retrospective gating of already-shipped cols.

**Predecessors shipped since the previous 2026-04-27 status snapshot:**
- **Plan 8** (adoption gate redesign) — replaced PR #10–#15-era §1.3 thresholds; weekly `[p10, p90]` calibration dropped as a gating metric (not load-bearing for planned downstream consumers).
- **Plan 9** (PBP ingest plumbing + opp-adjusted EPA-residual feature) — PBP plumbing shipped (`src/projections/ingest/pbp.py`, `PbpSchema`, `pbp` kwarg threaded through builders); opp-EPA-residual feature DO_NOT_ADOPT 4/4 positions and reverted at `941b96c`.
- **Plan 9 retro option C** (lightgbm-nb on EPA-residual) — DO_NOT_ADOPT 4/4 cells. Closed across model classes; do not revisit.
- **Feature Signal Probe** — pre-spec screening tool (`scripts/probe_feature_signal.py`); canonical first step before any new feature plan.
- **PBP Feature Family Probe (PR #20)** — bundled the 4 PBP team-features into one probe; verdict SIGNAL via RB.
- **RB PBP Features Integration (PR #21)** — production builder + ADOPT for `(BaselineModel, RB)`.
- **PBP Receiver Family Probe (PR #22)** — receiver-level air-yards / aDOT — NULL durable.
- **PBP Red-Zone Family Probe (PR #23)** — team-level RZ — NULL durable.
- **PBP Pressure Family Probe (PR #24)** — team-level pressure — NULL durable.
- **Trajectory Feature Family Probe (PR #25)** — player-level age + role-trajectory — SIGNAL via WR + TE.
- **WR Trajectory Features Integration (PR #26)** — production builder + ADOPT on `(BaselineModel, WR)` at -0.0371 fpts.
- **TE Trajectory Features Integration (PR #27)** — production builder + ADOPT on `(LightGBMNbModel, TE)` at -0.0090 fpts; first non-default-model-class binding.
- **Weather Feature Family Probe (PR #28, this)** — game-environment features — **SIGNAL via lgb-nb augment composite (RB + WR)**.

**Predecessors (longer history):**
- Plan 1 / Plan 2a / Plan 2b / Plan 3a / Plan 3b / Plan 3c / Plan 3d / Plan 3e (Phase 0 + Phase 1) / Plan 5 / Plan 5b / Plan 5c / Plan 6 / Plan 7. See per-plan blocks below.

## Next action

**Weather family probe shipped (PR #28).** Verdict SIGNAL via lgb-nb augment composite — RB ADOPT -0.0081 fpts (CI [-0.0163, -0.0005]) and WR ADOPT -0.0110 fpts (CI [-0.0172, -0.0049]). Baseline returned no Phase 1 SIGNAL across either mode; signal lives only in tree models (the bundle's non-linear thresholds — `is_high_wind` ≥20, surface category — are invisible to RidgeCV even with explicit boolean encoding). lgb-nb swap is degenerate (no v1 weather counterparts to drop). QB and TE returned DO_NOT_ADOPT under lgb-nb augment composite.

**Recommended next direction (pick one):**

1. **Combined RB+WR weather features integration plan, routed through `LightGBMNbModel` only.** The two ADOPT cells share the same model class and binding mode (lgb-nb augment), so a single integration plan can extend `RbFeaturesSchema` + `WrFeaturesSchema` together, plumb the 4 weather cols through `build_rb_features` + `build_wr_features`, and run dual-run gates on `(LightGBMNbModel, RB)` and `(LightGBMNbModel, WR)` in parallel. **Precedent: PR #27** TE trajectory integration shipped a schema change for a non-default model class while leaving baseline production routing unchanged. Do NOT extend `QbFeaturesSchema` or `TeFeaturesSchema` in the same plan (QB DO_NOT_ADOPT + recurring augment-regression pattern; TE DO_NOT_ADOPT + small sample). Expected gate magnitudes: RB ~-0.008 fpts, WR ~-0.011 fpts (probe predictions; gate may differ ±0.003 fpts per PR #25/#26/#27 calibration history).

2. **Refined weather units** (lower priority; revisit-only-on-SIGNAL territory now in scope per spec §1.4): cold-weather threshold (`is_cold_weather = temp < 32`, sibling shape to `is_high_wind`), multi-class surface encoding (one bool per surface code), kickoff hour / time-of-day (free, extractable from existing `schedules.kickoff`), surface × position interactions, per-team weather acclimation, precipitation (would require new ingest — NOAA hourly historical), wind direction (would require new ingest). Recommended priority order if scoped: cold-weather threshold → multi-class surface → kickoff hour.

3. **Pivot to next mechanism axis: target decomposition (TODO #23).** Volume × efficiency decomposition for each fantasy stat. 3-10% RMSE estimated independent of model class. Architecturally larger than weather integration but a different signal axis after consecutive context/team/trajectory/weather feature additions.

**Track 2A status (PBP team-level families):** Complete. All three TODO #3c team-level PBP feature families probed. Two NULL, one SIGNAL→ADOPT (PR #20→#21). No additional team-level candidates queued.

**Track 2B status (RB PBP cols × other model classes):** Closed (folded into PR #24). RB PBP cols transfer to all 4 tree-model classes directionally; only baseline reaches strict CI<0; no class regresses.

**Refined-unit candidates (low-priority, none queued):**
- Weather refined units: cold-weather threshold, multi-class surface, kickoff hour, surface × position, per-team acclimation, precipitation, wind direction.
- Trajectory refined units: per-position aging-curve interaction terms (`age²`), `is_2nd_year` / `is_3rd_year` flags, depth-chart-rank trends, longer trailing windows (l8 vs l16), `has_trajectory_history` indicator.
- PBP refined units: per-route-concept distributions, goal-line / 3rd-down splits, alternate pressure denominators. The cumulative durable-NULL signal across the broad cuts argues against any of these clearing absent independent evidence the unit choice was the binding constraint.

**Followup housekeeping (low-priority):**
- Model C-tuned is strictly dominated by Model C-NB on RMSE — TODO #29 captures the pruning when ready.
- Triplicate `_read_concat` / `_FANTASY_POSITIONS` / `_build_player_team_week_index` / `_parse_season_range` across 3 override scripts (PR #20, #23, #24) — extract to `scripts/_pbp_override_common.py`.
- `_trailing_4_mean` triplicated across 3 sibling feature modules — extract to `src/projections/features/_pbp_common.py`.
- Drop redundant `notna()` clause in `_per_game_rate` (`pbp_pressure_features.py:64`).

After the above: Plan 4 (public API + CLI verbs + free-tier hosting), then Draft Hub.

---

## Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-25 | Plan 3a held-out year is 2024, not 2025 | `nfl_data_py` has not yet published 2025 data despite the simulated date being post-2025-season. Training window shifted to 2018-2023. Architecture unaffected; 3c's walk-forward backtest will revisit. |
| 2026-04-25 | Per-stat independent `RidgeCV` sub-models for Model A | Closest match to spec wording (§3.1); per-stat residuals are debuggable; per-stat-independence assumption is "option D" / TODO #1 territory. |
| 2026-04-25 | `Model` as `typing.Protocol` (not `abc.ABC`); not `@runtime_checkable` | Structural typing matches existing `Distribution` Protocol; no isinstance checks needed in callers. |
| 2026-04-25 | One `BaselineModel` class with per-position factories (`wr_baseline()`, future `qb_/rb_/te_baseline()`) | Minimizes 3a→3b copy; per-position quirks expressed as config (`target_stats`, `feature_columns`, `dist_families`). |
| 2026-04-25 | `model_id = "baseline:<pos>:<8-char-code-hash>:<train-start>-<train-end>"` written into every projection row | Stable, reproducible, traceable. Persisted into `ProjectionWeeklySchema.model_id` so we always know which model produced which projection. |
| 2026-04-25 | `code_hash` covers 8 source files | `models/base.py`, `models/baseline.py`, `features/wr.py`, `features/_shared.py`, `features/_rolling.py`, `features/_opponent.py`, `scoring/score.py`, `scoring/score_distribution.py`. Anything whose change should invalidate the artifact. |
| 2026-04-25 | Method of moments for gamma α with clip to `[0.01, 100]` | Closed-form; MLE via `scipy.optimize` is a follow-up if calibration is bad. Plan 3a's calibration is borderline (70.8% in [p10, p90]) — TODO note for 3c. |
| 2026-04-25 | Greek letters in source converted to ASCII (`alpha`, `mu`) | Ruff RUF002/RUF003 flag Greek letters as ambiguous-unicode. Spec/plan markdown can keep them; source files use ASCII transliterations. |
| 2026-04-25 | Per-row sample seed in `score_distribution` is fixed at `42` for v1 | Documented in `predict_distribution` docstring + TODO #13. Cross-row sample correlation; fine for per-row stats; matters when callers combine samples (DFS lineup variance). Defer fix to Plan 3c or DFS work. |
| 2026-04-25 | `family="SAMPLED"` but `params` is summary-only blob | Documented in `predict_distribution` docstring + TODO #14. Per-row p-quantile columns carry the actual distributional info. Decide between SAMPLED_SUMMARY enum value vs. full samples blob before Plan 3c's backtest output consumes the rows. |
| 2026-04-25 | WR builder's traded-player fix: dedupe shares to highest share per gsis_id | v1 hack documented inline + TODO #15. Proper fix restructures `trailing_n_share_in_group` to expose team, lets callers join on (gsis_id, team). Tackle in Plan 3b. |
| 2026-04-25 | TODO #15 closed before Plan 3b kickoff: helper returns `[gsis_id, team, share_l<n>]`; WR/RB/TE builders join on `(gsis_id, team)` | Picks the share for the player's depth-chart-current team — semantically more correct than the v1 highest-share proxy and removes the dedupe hack. RB/TE builders inherit the fix automatically when 3b trains them on real data. |
| 2026-04-25 | TODO #8 closed before Plan 3b kickoff: opt-in `pytest -m network --run-network` smokes per ingest source | One smoke per source (weekly_stats, depth_charts, ngs × 3 stat_types, schedules, id_map, snap_counts) asserts every raw column the normalize step depends on is present, then runs normalize end-to-end so pandera surfaces dtype drift. Post-bump procedure documented in `CONTRIBUTING.md`. |
| 2026-04-25 | Plan 3b: BaselineModel gains required `feature_schema` + `code_hash_files` constructor args | Replaces hardcoded WR references; per-position config stays per-factory. Existing 3a artifact unloadable; retrain in Phase 6 (TODO #17 closed). |
| 2026-04-25 | Plan 3b: TE model includes rushing as target stat (Taysom Hill) | Q3 brainstorm decision; Phase 1 added `rushing_*_per_game_l4` to `TeFeaturesSchema` and `build_te_features`; cost is two columns and a fixture row. |
| 2026-04-25 | Plan 3b: NORMAL/GAMMA convention extended mechanically; POISSON deferred | WR's family choices carry to QB/RB/TE without per-position tuning. POISSON for low-mean integer counts (interceptions, fumbles_lost) deferred to 3c contingent on calibration evidence. |
| 2026-04-25 | Plan 3b: centralized `POSITION_DISPATCH` registry in `models/__init__.py` | One canonical "what positions the system knows about" answer. Reused by CLI scripts and future 3c backtest harness. Adding a position is one new line. |
| 2026-04-25 | Plan 3b: per-position test files (mirrors `tests/test_features/`) | Q6 brainstorm decision. Six new files; failure isolation per position is worth ~210 lines of necessary duplication. |
| 2026-04-25 | Plan 3b: smoke test parametrized across all four positions | Q6 brainstorm B; catches "I broke RB silently" earlier than the per-position test files. ~20s smoke runtime acceptable. |
| 2026-04-25 | Plan 3b: three WR-specific scripts deleted; replaced by position-arg-driven generalized scripts | Q1 brainstorm C. Avoids producing four near-duplicate scripts after 3b. |
| 2026-04-25 | Plan 3b real-data drift: `*_yards_per_game_l4` schema bound dropped to allow negative trailing means | Underlying weekly_stats yards columns allow negative values (sacks/TFL/kneels); commits `fa864ac` and `e25eb57` relax the bound on the trailing means and on `passing_yards_per_game_std`. |
| 2026-04-25 | Plan 3b real-data drift: bye-week + dedupe filters ported from WR to QB/RB/TE | WR had these in 3a (TODO #9a, #9c); QB/RB/TE feature builders inherit the same shape. Commits `f79806a` (bye filter) and `54b6d95` (dedupe). |
| 2026-04-26 | Plan 3c gate is opt-in `pytest -m backtest --run-backtest`, not default-on | A full gate run is ~2 minutes; default-on adds material drag to every dev iteration. Default-on smoke covering one (WR, 2024) cell catches harness wiring bugs cheaply. |
| 2026-04-26 | Snapshot at (position, year, metric) granularity (368 rows); per-metric-type tolerances | Per-year visibility is the whole point of multi-year backtest; aggregating loses the "regressed only on 2022" signal. Tolerances grouped by metric type keeps maintenance low; per-row overrides added empirically as we observe noise. |
| 2026-04-26 | Held-out years 2021-2024 (skip 2019 / 2020) | 2019's 1-season train window is too small; 2020 is COVID-shortened structural outlier. Each held-out year has at least 3 seasons of training history. |
| 2026-04-26 | Plan 3c uses summed weekly means as season totals (degenerate aggregation); real Monte Carlo aggregation deferred to Plan 3d | Decouples gating infrastructure from season-distribution design. Plan 3d converges TODOs #13 / #14 and calibration tightening. |
| 2026-04-26 | Feature cache invalidation is manual via `scripts/refresh_features.py`; auto-invalidation deferred (TODO #21) | Manual is documented in CONTRIBUTING.md and produces a clear FileNotFoundError pointing at the refresh command. Auto-invalidation via code-hash is straightforward but adds surface area; defer until manual produces a real-world bug. |
| 2026-04-26 | `score_distribution` perf vectorization pulled forward from Plan 3d into Plan 3c | Spec section 1.2 deferred the perf TODO under "feature caching means we predict once per (player-week, year), not per training fold." Phase 6 demonstrated this was wrong: the per-sample Python loop still dominated at 20-30 minutes for the full harness. Math is bit-identical (linear scoring rule); fix is mechanically safe. |
| 2026-04-29 | Pivot to gate redesign (Plan 8) before any further model-improvement work; feature-class track (TODO #3 PBP / EPA) is the next-up modeling lift after Plan 8 lands | PR-10-through-PR-15 diagnosis: §1.3's per-cell thresholds are below the noise floor (Plan 6 failed by 0.24pp on a single noisy cell after winning 12/16 RMSE) and the calibration criterion isn't load-bearing for any planned downstream consumer. Five model-class swaps on identical features hit the same information ceiling — feature work is the next real lift, not another model class. |

---

## Plan 2b — historical (as of 2026-04-24)

**Projections Core — Plan 2b (QB/RB/TE feature builders) merged to `main` at commit `af325ea`.**

**Predecessors:**
- Plan 1 (Foundations) merged at `8f02a6c`.
- Dev tooling merged via `feat/dev-tooling`.
- Plan 2a (Ingest expansion + WR feature builder) merged at `7926090`.

**Plan 2b delivered:**
- `build_qb_features`, `build_rb_features`, `build_te_features` — pure-function builders mirroring `build_wr_features`'s shape.
- Three new feature schemas (`QbFeaturesSchema`, `RbFeaturesSchema`, `TeFeaturesSchema`).
- `WeeklyStatsSchema` extended with `attempts`, `completions`, `sacks` for QB features.
- Generalized `trailing_n_share_in_group` helper in `_rolling.py` (migrated from `wr.py`'s local helper).
- ~45 new tests (~200 total). 5 leakage tests per position (15 new).

---

## Next action

**Recommended: Plan 3 — Model A baseline + season aggregation + first-class backtest harness.**

All 4 offensive skill positions (QB/RB/WR/TE) now have feature builders. Plan 3 trains the v1 model per position, aggregates weekly outputs to season distributions (Monte Carlo with bye + availability), and stands up the backtest harness that gates future model changes.

K and DST builders (TODO #10) can land in parallel with Plan 3 — they're independent.

---

## Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-11 | Full ingest migration off `nfl_data_py` to `nflreadpy` (TODO #32 option 3) | Every `_fetch_raw_*` now calls `nflreadpy.load_*(seasons=...).to_pandas()`; `nfl_data_py` dropped from `pyproject.toml`; `polars>=1.0` added as transitive dep (nflreadpy returns polars frames). All 8 sources confirmed live against nflreadpy via the opt-in API drift smokes. Historical re-ingest covered 2018-2025 for every source except depth_charts (see TODO #34). Two normalize-step fixes flushed out by the live re-ingest: (a) `schedules._normalize_one_season` now coerces `season`/`week` to int64 (polars→pandas returns int32, nfl_data_py returned int64 implicitly); (b) `weekly_stats._normalize_one_season` runs the `Position`-isin filter before the team-code normalize so defensive rows with NaN `opponent_team` no longer trip `normalize_team_code`. |
| 2026-05-11 | depth_charts 2025+ deferred — upstream switched to a snapshot-by-timestamp release schema | nflverse migrated the `depth_charts_<year>.parquet` release in 2025 from per-game-week rows (`club_code`, `season`, `week`, `depth_team`, `depth_position`) to a snapshot-by-timestamp feed (`dt`, `pos_slot`, `pos_rank`, `pos_abb`, `pos_grp`, ...). `refresh_depth_charts` now raises `NotImplementedError` for any season whose payload lacks the legacy columns; the derivation logic (closest-prior-snapshot per game day, mapped to canonical `(position, depth_rank)`) is TODO #34. 2018-2024 partitions are unaffected and were re-ingested cleanly. |
| 2026-05-11 | `weekly_stats` ingest fetches `stats_player/stats_player_week_<season>.parquet` directly (TODO #32 option 1) | Shipped 2026-05-11; superseded same-day by option 3 above. `_fetch_raw_weekly` now goes through `nflreadpy.load_player_stats`. Direct-URL fetch was bit-identical to the nflreadpy path so the switch was zero-blast-radius. |
| 2026-05-11 | `WeeklyStatsSchema.receiving_air_yards` bound widened to `ge=-100` (was -50) | New nflverse release surfaced one 2025 Tyler Higbee row at -92 (target behind LOS on a screen / shovel). Empirical 2018-2024 floor was -33; -100 gives headroom for similar anomalies without losing the schema's role as a sanity gate. |
| 2026-05-11 | Existing 2018-2024 weekly_stats partitions left untouched; only 2025+ written from the new URL | Re-ingesting 2018-2024 from the new format would add ~22% more rows per season (zero-stat weeks + the new K rows) and could quietly shift trailing-window features in the backtest harness. Defer the historical re-ingest until the full nflreadpy migration plan (option 3) lands, so it happens once. |
| 2026-04-24 | `rushing_qb` boolean threshold = 5.0 carries/game over trailing 4; `passing_down_back` = 4.0 targets/game | Rough heuristics from feel. Not load-bearing; revisit at backtest time if categorization matters |
| 2026-04-24 | TE target_share denominator includes WR + RB + TE (full pass-catching group) | TEs usually have only one fantasy-relevant player per team, so same-position-share would be ~1.0 and useless. Full-group share captures meaningful gradient |
| 2026-04-24 | RB target_share denominator includes WR + RB + TE (full pass-catching group) | A workhorse RB getting 5 targets/game on a 30-target offense is meaningfully different from one getting 5 on a 20-target offense. Full-group denominator captures team passing volume, not just RB-on-RB share |
| 2026-04-24 | Migrate `_trailing_4_share_per_team` from `wr.py` to `_rolling.py` as `trailing_n_share_in_group` | RB needs target_share against the full pass-catching group (not just RBs); TE needs the same. Generalize once, in the shared helper module, rather than duplicate in three builders |
| 2026-04-24 | Extend `WeeklyStatsSchema` with `attempts`, `completions`, `sacks` | QB feature builder needs these source columns. All three are present in raw `nfl_data_py.import_weekly_data` output. Same incremental pattern as 2a's extension for `targets`/`carries`/`receiving_air_yards` |
| 2026-04-24 | One bundled PR for QB/RB/TE (not three per-position PRs) | Repetitive, interlinked work; reviewing all three together catches drift. Each position lands as its own commit inside the bundle for easy retrospection |
| 2026-04-24 | All 4 position builders use parallel files (no WR/TE shared base) | Each position's feature list will diverge over time as we add play-by-play-derived features. Premature DRY hurts later. Shared logic lives in `_rolling.py` / `_opponent.py` / `_shared.py` (the latter added 2026-04-25 in PR #4 cleanup, hoisting `prior_mask` / `exact_week_mask` / `build_game_environment` out of `wr.py`) |
| 2026-04-24 | K and DST split out into a future plan; 2b covers QB/RB/TE only | K needs FG-attempt data not in `WeeklyStatsSchema`; DST is team-level not player-level and needs play-by-play. Both should wait for the data they need rather than ship degraded v0 features |
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
- **Plan 5** — Model C (LightGBM with quantile regression). Adopt only if it beats Model A on the backtest harness. Detailed scope in TODO #26. One of three model-improvement tracks identified post-Plan-3e (alongside TODO #3 PBP features and TODO #23 target decomposition).

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
