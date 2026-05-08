# Weather Features RB+WR Integration — Summary Report

**Status:** Both positions ship as designed.
- **RB:** ADOPT (ship-as-designed) on `(LightGBMNbModel, RB)`.
- **WR:** ADOPT (ship-as-designed) on `(LightGBMNbModel, WR)`.

**Branch:** `feat/weather-features-rb-wr`
**Spec:** `docs/superpowers/specs/2026-05-08-weather-features-rb-wr-design.md`
**Plan:** `docs/superpowers/plans/2026-05-08-weather-features-rb-wr.md`
**Date:** 2026-05-08

## Decision

Both binding cells `(LightGBMNbModel, RB)` and `(LightGBMNbModel, WR)` returned **ADOPT** in the dual-run adoption gate. Both contingency cells `(BaselineModel, *)` returned DO_NOT_ADOPT (CIs bracket zero) — **not** REGRESSION. Per spec §1.3.5, the default ship-as-designed branch fires for **both** positions: schema cols stay in `RbFeaturesSchema` + `WrFeaturesSchema`; `attach_weather_features` stays wired into both builders; `_RB_FEATURE_COLUMNS` + `_WR_FEATURE_COLUMNS` extensions stay in `baseline.py`.

## Binding-cell shift — second non-default-class integration; first bundled-position integration

PR #21 (RB PBP) and PR #26 (WR trajectory) bound the ship decision on `(BaselineModel, position)` because baseline was each position's `default_model_class`. PR #27 (TE trajectory) was the first integration to bind on a non-default class — `(LightGBMNbModel, TE)`. This PR is the **second** non-default-class integration (binds on `(lgb-nb, RB)` and `(lgb-nb, WR)`) AND the **first** to bundle two positions into a single PR. Both binding cells share the same model class + mode (lgb-nb augment composite) per PR #28's probe; PR #20→#21 and PR #25→#26/#27 had only one binding cell each. RB and WR production routings stay on `baseline`; flipping `_PositionDispatch[POS].default_model_class` is a deferred per-position cross-class follow-up.

## Probe-vs-gate calibration per position

| Position | Source | Composite RMSE Δ on (lgb-nb, POS) | 95% CI |
|---|---|---:|---|
| RB | PR #28 probe (predicted) | **-0.0081** | [-0.0163, -0.0005] |
| RB | This PR's gate (measured) | **-0.0077** | [-0.0157, -0.0001] |
| WR | PR #28 probe (predicted) | **-0.0110** | [-0.0172, -0.0049] |
| WR | This PR's gate (measured) | **-0.0104** | [-0.0165, -0.0042] |

**Excellent calibration on both binding cells.** Gate measured magnitudes are ~5% smaller than probe predictions in both cases — well within the probe CI on each cell. Track record:
- PR #20→#21 matched to 4 decimal places on RB (-0.0124 → -0.0124).
- PR #25→#26 matched within ~0.004 fpts on WR (-0.0414 → -0.0371; within probe CI).
- PR #25→#27 matched within ~0.0017 fpts on TE (-0.0107 → -0.0090).
- This PR: RB (-0.0081 → -0.0077, gap 0.0004) and WR (-0.0110 → -0.0104, gap 0.0006) — both within ~10% of probe.

## Per-(model_class, position) verdicts (4 cells)

| Position | Model class | RMSE Δ | 95% CI | Spearman Δ | 95% CI | Verdict |
|---|---|---:|---|---:|---|:---:|
| **RB** | baseline       | -0.0034 | [-0.0103, +0.0042] | +0.0002 | [-0.0012, +0.0017] | DO_NOT_ADOPT (informational; not REGRESSION) |
| **RB** | **lightgbm-nb** | **-0.0077** | **[-0.0157, -0.0001]** | +0.0004 | [-0.0008, +0.0016] | **ADOPT (binding)** |
| **WR** | baseline       | -0.0026 | [-0.0106, +0.0061] | -0.0005 | [-0.0020, +0.0010] | DO_NOT_ADOPT (informational; not REGRESSION) |
| **WR** | **lightgbm-nb** | **-0.0104** | **[-0.0165, -0.0042]** | +0.0021 | [+0.0008, +0.0034] | **ADOPT (binding)** |

n_paired: RB 5273; WR 8460. n_bootstrap: 1000; seed: 42.

**3 informational classes** (lightgbm, lightgbm-tuned, ensemble) **were not run** in the gate per spec §1.3.4 + PR #27 precedent. Reasons: (a) wall-time risk — PR #27's `--model all` aborted after 3 hours; (b) TODO #29 already flags `lightgbm-tuned` as a pruning candidate (dominated 16/16 by lgb-nb on RMSE); (c) ensemble's MIXED-family rows would need `aggregate_to_season` widening (TODO #28). Back-fillable by a follow-up `--model lightgbm,lightgbm-tuned,ensemble` backtest if any cross-class routing-flip discussion needs them.

## Per-position §1.3.5 contingency matrix outcome

| Position | (lgb-nb, POS) | (baseline, POS) | Branch fired | Action taken |
|---|:---:|:---:|:---:|---|
| **RB** | ADOPT | DO_NOT_ADOPT (not REGRESSION) | **ship-as-designed** | All weather edits kept (schema + builder + `_RB_FEATURE_COLUMNS`). |
| **WR** | ADOPT | DO_NOT_ADOPT (not REGRESSION) | **ship-as-designed** | All weather edits kept (schema + builder + `_WR_FEATURE_COLUMNS`). |

**Both positions hit the default branch.** No modified-shape branch fired (would have required `(baseline, POS)` REGRESSION — CI strictly above zero). No revert branch fired (would have required `(lgb-nb, POS)` MARGINAL or DO_NOT_ADOPT). Worst-case combined outcomes (RB modified-shape + WR full-revert, etc.) did not materialize.

## Coverage statistics (2021-2024 eval window)

Per Task 11's measurement on the production builder output, post-refresh-features:

| Position | Season | Rows | wind_speed_mph | is_high_wind | temperature_f | is_grass_surface |
|---|---:|---:|---:|---:|---:|---:|
| RB | 2021 | 1946 | 96.3% | 96.3% | 96.3% | 100.0% |
| RB | 2022 | 1940 | 67.7% | 67.7% | 67.7% | 100.0% |
| RB | 2023 | 1850 | 85.7% | 85.7% | 85.7% | 100.0% |
| RB | 2024 | 1844 | 98.3% | 98.3% | 98.3% | 100.0% |
| WR | 2021 | 3114 | 96.2% | 96.2% | 96.2% | 100.0% |
| WR | 2022 | 3077 | 66.8% | 66.8% | 66.8% | 100.0% |
| WR | 2023 | 3114 | 85.7% | 85.7% | 85.7% | 100.0% |
| WR | 2024 | 3018 | 98.5% | 98.5% | 98.5% | 100.0% |

**Production builder coverage byte-perfectly matches PR #28's probe override coverage** (verified by reading the probe parquet at `data/features_probe/weather.parquet` directly — same per-(position, season) percentages to the tenth of a percent on every cell). The `compute_weather_features` + `attach_weather_features` helpers shipped in PR #28 and consumed by the production builder produce identical output to the probe. No builder-wiring divergence.

**Documentation correction vs PR #28 PM entry:** the 2026-05-07 weather probe PM entry stated "per-(position, season) coverage in the 2021-2024 eval window is uniformly ≥92% across all 4 positions." That claim was **overstated** — actual probe (and now production) coverage shows 2022 at ~67% and 2023 at ~86% per (RB, WR). The pooled audit metric of "8.39% NaN rate" hides the per-season variation. The probe was nonetheless run on this same data and returned SIGNAL on the lgb-nb augment composite for both RB and WR; the gate has now reproduced those verdicts on production-pipeline output. Future "coverage uniformly ≥X%" claims should be reported per (position, season), not pooled.

## Threshold note

The gate did not require `--coverage-threshold` adjustment — its row-key-matching pairing logic is independent of the probe's pooled coverage check (per PR #26 spec note). Default `0.95` accepted; no fallback needed. (The probe used `--coverage-threshold 0.90` but that flag is probe-only, not gate-side.)

## Cross-class deferred follow-up — per position

**RB** production routes to `baseline` per Plan 8 (2026-04-29). With weather cols now in `RbFeaturesSchema`, a separate cross-class re-eval (`scripts/adoption_gate.py --position RB` comparing `lightgbm-nb` candidate to `baseline` baseline at the position level — not the within-class with-vs-without comparison this PR ran) could justify flipping `_PositionDispatch[RB].default_model_class` to `lightgbm-nb`. Not load-bearing for any current consumer; queue alongside the next RB-related work.

**WR** production routes to `baseline`. Same shape as RB above; the `_PositionDispatch[WR].default_model_class` flip question is parallel.

Naive arithmetic suggests the cross-class re-eval may bind near zero for both positions. From Plan 8's published numbers, `lgb-nb` for RB was modestly worse than baseline at the position level pre-this-PR; this PR adds -0.0077 fpts to lgb-nb RB. Whether the net flips the verdict needs the actual re-eval — naive stacking is illustrative, not predictive.

## What this closes

TODO #25's broad-cut weather family at the in-builder unit, on **both** the RB and WR ADOPT cells from PR #28. QB and TE remain DO_NOT_ADOPT at this unit per PR #28's probe; not re-tested in this PR's gate. Refined-unit candidates remain open under TODO #25: `is_cold_weather` (`temp < 32`, sibling shape to `is_high_wind`), multi-class surface encoding (one bool per surface code), kickoff hour / time-of-day, surface × position interactions, per-team weather acclimation, precipitation (would require new ingest), wind direction (would require new ingest). Recommended priority order: cold-weather threshold → multi-class surface → kickoff hour. None queued.

## Spec gaps caught + fixed during execution

- **`scripts/refresh_features.py` CLI takes a single position**, not `rb wr` together as the plan suggested. Ran twice: once for `rb`, once for `wr`. (Plan §6 Task 10 mis-specified.)
- **`scripts/backtest.py` does not have a `--position` flag**, contrary to the plan's invocation. Worked around via the `run_backtest(positions=...)` Python API in the custom orchestrator.
- **`scripts/backtest.py --update-snapshot` overwrites the entire snapshot file** rather than merging. Worked around with `scripts/backtest_dual.py` orchestrator that preserves rows for non-target model classes.
- **`scripts/adoption_gate.py` dual-run mode requires single-model-class run dirs.** Each `_run_single_backtest.py` produces a multi-class results.parquet; split into per-model-class subdirs (`run_{baseline,candidate}_{baseline,lightgbm-nb}/`) before invoking the gate.
- **Python import caching across the schema-revert boundary.** First attempt at the dual backtest reused a single Python process across baseline+candidate runs; the in-memory schema classes did not refresh after `git checkout main -- src/projections/schemas.py`, so the baseline-side `pandera.validate` raised on `wind_speed_mph` (the cache had been refreshed without it but the in-process class still expected it). Fixed by subprocess-ing `_run_single_backtest.py` for both runs.
- **PR #28 PM entry's coverage claim ("uniformly ≥92%")** was inaccurate at the per-(position, season) granularity; actual coverage on 2022 is ~67%. Documented above for the record.

## Follow-up recommendations (non-blocking)

- **Recurring-bug-class regression test for `_<POS>_FEATURE_COLUMNS`.** Code-review reviewer flagged that 4 PRs in a row (PR #21 RB PBP, PR #26 WR trajectory, PR #27 TE trajectory, this PR) have hit the same spec-gap pattern: schema gets a new feature, lightgbm auto-picks-up via dynamic schema derivation, but `baseline.py:_<POS>_FEATURE_COLUMNS` is hardcoded and must be updated explicitly. A 5-line parametrized test pinning `set(_POS_FEATURE_COLUMNS) == set(SCHEMA.columns) - identity` would catch this structurally on every future schema extension. Worth a follow-up commit / mini-PR after this one merges.

## Reports + artifacts

- `reports/adoption_gate_weather_features_rb_wr.md` — concatenated stdout from 4 gate invocations.
- `reports/adoption_gate_weather_baseline_RB.csv`, `_baseline_WR.csv`, `_lightgbm-nb_RB.csv`, `_lightgbm-nb_WR.csv` — per-cell CSV outputs.
- `data/backtest/run_baseline/` + `data/backtest/run_candidate/` — raw per-row dual-run results parquets (gitignored — regenerable via `scripts/backtest_dual.py`).
- `tests/backtest/model_metrics.json` — snapshot updated; 384 of 1872 rows replaced (RB+WR × baseline+lgb-nb); 1488 preserved (QB+TE all classes; RB+WR × {lightgbm, lightgbm-tuned, ensemble}).

## Next track

- **Refined-unit weather candidates (TODO #25).** First sibling probe: `is_cold_weather` (`temp < 32`, structural sibling to `is_high_wind`). Same probe-first workflow; bundle 3-4 refined-unit features.
- **TODO #29 lightgbm-tuned pruning.** Now that we've shipped without re-running tuned/lightgbm/ensemble for RB+WR, a separate housekeeping PR can clean out tuned references.
- **Cross-class production-routing flips for RB and WR** (deferred per-position above).
