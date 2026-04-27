# Project Management

Running log of project status, decisions, and next steps. Append new entries at the top; keep the bottom as the long-tail backlog. Specs live in `docs/superpowers/specs/`, plans in `docs/superpowers/plans/`, single-task TODOs in `TODO.md`.

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

## Current status (as of 2026-04-26)

**Projections Core — Plan 3e Phase 0 (calibration diagnostic) complete on branch `feat/plan-3e-calibration-tightening`.** Diagnostic CLI + research report committed; no model code changes yet. Spec amendment (Phase 1+) is the next gate per the spec's section 3 decision gate.

**Plan 3d (real Monte Carlo season aggregation) merged to `main` at commit `fe55d5b` (PR #9).**

**Predecessors:**
- Plan 1 (Foundations) merged at `8f02a6c`.
- Plan 2a merged at `7926090`; Plan 2b merged at `af325ea`.
- Plan 3a (WR Model A baseline) merged at `598ab9c`.
- Plan 3b (QB / RB / TE Model A baselines) merged at `c4a0401`.
- Plan 3c (walk-forward backtest gate) merged at `3db71a6` (PR #8).

**Plan 3e Phase 0 delivered (current branch, not yet merged):**
- New `scripts/diagnose_calibration.py` CLI: loads the latest `data/backtest/run_<ts>/` per-row results, fits alternative distribution families (Student-t, lognormal, negative-binomial) against per-stat residuals, and emits a per-cell summary CSV + per-cell QQ/residual plots + a recommendation column.
- New `tests/test_scripts/test_diagnose_calibration.py` (21 tests) covering the loader, residual extraction, summary stats, alternative-family fits (including the degenerate Student-t guardrail), recommendation logic, and a smoke test of `main()`.
- Research report committed at `docs/superpowers/research/2026-04-26-calibration-diagnosis.md` identifying 3 root causes (zero-inflation under GAMMA, heavy tails on `*_yards`, pervasive heteroscedasticity).
- TODO #22 (Plan 3e — calibration tightening) stays open; the diagnostic surfaces root causes but does not implement fixes.

**Plan 3d delivered:**
- New `src/projections/aggregation/season.py`: `aggregate_to_season` real Monte Carlo season aggregator (regenerates per-week samples from the per-row seed; modal-position resolution for traded players).
- New `src/projections/distributions/codec.py`: `pack_per_stat_params` / `unpack_per_stat_params` codec for `ProjectionWeeklySchema.params`.
- `derive_row_seed` in `src/projections/scoring/score_distribution.py`: stable 32-bit per-row seed via sha256 of `(gsis_id, season, week, ruleset.name)`. Consumed by both `predict_distribution` and `aggregate_to_season`.
- `BaselineModel.predict_distribution` now writes per-row seeds + per-stat params blob.
- New `DistributionFamily.SAMPLED_SUMMARY` enum value; new `ProjectionSeasonSchema`.
- Season-calibration metrics (`season_calibration_p10p90`, `season_calibration_le_p90`) wired into the harness; pinned to `calibration_absolute` tolerance classifier. Snapshot expanded from 368 → 400 rows.
- `scripts/backtest.py` writes per-row + per-player results to `data/backtest/run_<ts>/`.
- Default-on smoke asserts season metrics are present and finite.
- Full gate runtime: 292.73s. Drift on the 368 existing weekly rows is within tolerance for every cell (max abs delta 0.0084 vs 0.03 tolerance; max rel delta +0.165% vs 5% tolerance). Cause: per-row seed change reorders Monte Carlo draws; underlying regression math unchanged.
- Season `[p10, p90]` coverage well below target (0.30–0.55 vs 0.80) — inherits 3c's weekly under-dispersion. Plan 3e is the calibration-tightening follow-up.
- TODO #13 (per-row seeds), TODO #14 (SAMPLED_SUMMARY family), TODO #19 (gate non-determinism by demonstration) closed.
- TODO #22 filed (Plan 3e — calibration tightening).

## Next action

**Recommended: Plan 3e Phase 1+ — locked in by spec amendment after Phase 0 evidence; brainstorming session to follow.**

Phase 0 produced a research report
(`docs/superpowers/research/2026-04-26-calibration-diagnosis.md`) identifying 3
root causes for the 0.30–0.55 weekly + season `[p10, p90]` under-dispersion:
zero-inflated count stats catastrophically miscalibrated under GAMMA,
heavy-tailed continuous yards stats (Student-t beats Normal by AIC delta
`-2160 to -317`), and pervasive heteroscedasticity (18 of 24 cells with
variance-ratio > 1.5). Per the spec section 3 decision gate, the next gate is a
spec amendment to
`docs/superpowers/specs/2026-04-26-plan-3e-calibration-tightening-design.md`
adding Phase 1+ implementation phases (family swaps, variance bucketing,
regression gate against the 3d snapshot). Re-invoke `superpowers:brainstorming`
for that scoping in the next user-driven session.

After 3e: Plan 4 (public Python API + CLI verbs + free-tier web hosting).

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
