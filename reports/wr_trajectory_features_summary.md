# WR Trajectory Features Integration — Summary Report

**Status:** **ADOPT** on `(BaselineModel, WR)` — composite RMSE delta **-0.0371 fpts** ([-0.0567, -0.0172]).
**Branch:** `feat/wr-trajectory-features`
**Spec:** `docs/superpowers/specs/2026-05-03-wr-trajectory-features-design.md`
**Plan:** `docs/superpowers/plans/2026-05-03-wr-trajectory-features.md`
**Date:** 2026-05-04

## Decision

**Ship.** The 4 trajectory features (`age`, `is_rookie`, `volume_trend_l4_minus_prior_l4`, `snap_pct_change_l4_vs_prior_l4`) are integrated into `WrFeaturesSchema` + `build_wr_features` per spec §1.1; the dual-run adoption gate confirms `(BaselineModel, WR)` ADOPT'd at -0.0371 fpts (CI strictly negative). 4 of 5 model classes ADOPT.

## Probe-vs-gate calibration

| Source | Composite RMSE Δ on (BaselineModel, WR) augment | 95% CI |
|---|---:|---|
| PR #25 probe (predicted) | -0.0414 | [-0.0606, -0.0230] |
| This PR's gate (measured) | **-0.0371** | **[-0.0567, -0.0172]** |

**Probe predicted -0.0414; gate measured -0.0371.** Difference 0.0043 fpts — well within both intervals. Both directions match (negative = candidate wins). Both CIs strictly exclude zero. The gate's slightly-smaller magnitude is consistent with the probe-vs-gate pattern observed in PR #20→#21 (probe magnitude was matched to 4 decimals there, but trajectory's ~3x larger absolute lift gave more room for rounding-style drift in the bootstrap CI generation). Mechanism interpretation: probe uses an override parquet that left-merges trajectory features onto baseline rows; production builder produces them in-place — both should give identical training data, and the small gap is bootstrap noise rather than a real divergence.

## Per-(model_class, WR) verdicts

| Model class | n_paired | RMSE Δ (fpts) | RMSE 95% CI | Spearman Δ | Spearman 95% CI | Verdict |
|---|---:|---:|---|---:|---|:---:|
| **baseline** | 8460 | **-0.0371** | **[-0.0567, -0.0172]** | +0.0047 | [+0.0015, +0.0081] | **ADOPT (binding)** |
| lightgbm | 8460 | -0.0207 | [-0.0289, -0.0121] | +0.0026 | [+0.0005, +0.0047] | ADOPT |
| lightgbm-tuned | 8460 | +0.0025 | [-0.0056, +0.0106] | +0.0014 | [-0.0003, +0.0032] | DO_NOT_ADOPT |
| lightgbm-nb | 8460 | -0.0171 | [-0.0269, -0.0071] | +0.0020 | [+0.0002, +0.0038] | ADOPT |
| ensemble | 8460 | -0.0242 | [-0.0351, -0.0138] | +0.0019 | [+0.0001, +0.0040] | ADOPT |

**4 of 5 classes ADOPT.** lightgbm-tuned is the only DO_NOT_ADOPT — its point estimate is essentially zero (+0.0025) and the CI brackets zero on both sides, consistent with the broader pattern that lightgbm-tuned is dominated by lightgbm-nb (TODO #29 pruning candidate). The lgb-nb cell at -0.0171 cross-checks the probe's second WR ADOPT cell (probe predicted -0.0194; gate measured -0.0171; same direction, similar magnitude). ensemble shows the largest magnitude after baseline at -0.0242 (probe didn't directly measure ensemble; informational).

## Per-year breakdown — `(BaselineModel, WR)` binding cell

| Year | n_paired | RMSE Δ | RMSE 95% CI | Spearman Δ | Spearman 95% CI |
|---|---:|---:|---|---:|---|
| 2021 | 2109 | **-0.0553** | **[-0.0940, -0.0179]** | +0.0068 | [+0.0001, +0.0133] |
| 2022 | 2102 | -0.0295 | [-0.0728, +0.0079] | +0.0052 | [-0.0015, +0.0119] |
| 2023 | 2201 | -0.0397 | [-0.0767, -0.0039] | +0.0034 | [-0.0026, +0.0093] |
| 2024 | 2048 | -0.0233 | [-0.0571, +0.0120] | +0.0035 | [-0.0030, +0.0095] |

The pooled RMSE improvement is concentrated in 2021 + 2023 (both CIs strictly negative). 2022 + 2024 are net-positive but their year-only CIs bracket zero. Spearman is uniformly +0.003 to +0.007 across all 4 years (rank-quality lift is small but consistent). The pooled CI is strictly negative because every year-point estimate is negative; year-only CIs that bracket zero are typical for the smaller per-year sample size (~2100 rows vs 8460 pooled).

## Coverage statistics (eval window 2021-2024)

| Column | Eval (2021-2024) coverage | Full (2018-2024) coverage | Probe (PR #25) coverage | Match? |
|---|---:|---:|---:|:---:|
| `age` | 97.2% | 96.7% | 96.7% | ✓ |
| `is_rookie` | 97.2% | 96.7% | 96.7% | ✓ |
| `volume_trend_l4_minus_prior_l4` | 57.5% | 51.6% | 53.6% | ✓ (+~4pp) |
| `snap_pct_change_l4_vs_prior_l4` | 73.9% | 66.7% | 68.4% | ✓ (+~5pp) |

All 4 columns within ~5pp of the probe's measured coverage. The slight uplift on the trend cols comes from the production builder running the bye-week filter + depth-chart dedupe before the trajectory join (so the surviving cohort has slightly more 8+ active games), versus the probe override which iterated all positions. The age column shows 11,976 finite rows in 2021-2024 with mean=24.7, std=2.8, range=21-35 — biologically plausible. is_rookie: ~20% rookie rate (2417/9559+347 NaN) consistent with NFL roster turnover.

The bound `age ge=15, le=50` was conservative; observed range is 21-35. No values clip the bound. `volume_trend_l4_minus_prior_l4` and `snap_pct_change_l4_vs_prior_l4` bracket zero and are bounded as expected (snap_pct_change in [-1, 1] by construction; volume_trend unbounded but observed within ±10).

## Spec gap caught + fixed

The spec at §1.1 Task 5 + §2.3 instructed passing prior-mask-filtered `ws`/`sc` to `attach_trajectory_features`. **This was wrong.** The helper's internal `_volume_trend` and `compute_snap_pct_change` already use `.rolling(4).mean().shift(1)` for leakage safety. Double-filtering left the input with no row for the current week, so the rolling+shift produced no output rows for the current week — `volume_trend_l4_minus_prior_l4` and `snap_pct_change_l4_vs_prior_l4` would have been 100% NaN on every output row.

The bug was caught during Phase 3 implementation (`baseline.py:_WR_FEATURE_COLUMNS` extension) when downstream model tests collapsed via `BaselineModel.fit`-time `dropna`. Fix at commit `d1b3092` (`fix(wr): pass full weekly_stats/snap_counts to attach_trajectory_features`): pass full unfiltered frames; helper's internal `.shift(1)` ensures leakage safety. The 5 existing WR leakage tests verified no regression. Direct regression test added at commit `a742d83` (`test(wr): regression test for trajectory trend non-NaN on 8+ history`) asserting `volume_trend_l4_minus_prior_l4 == 2.0` for a hand-computed Jefferson scenario via `pytest.approx`.

The spec/plan are historical record and not amended mid-PR (matches the repo's convention from prior PRs); this report + the PM decision-log entry document the spec gap explicitly.

## Cluster A leftovers caught + fixed

Phase 1 (Cluster A) added 4 trajectory cols to `WrFeaturesSchema` and updated `tests/test_schemas/test_dataframe_schemas.py` fixtures. Three downstream test fixtures were missed — caught and fixed during the integration:

1. `tests/test_features/test_cache.py:_minimal_wr_features_row` — fixed at commit `1f1f415` (Cluster B prereq).
2. 7 lightgbm/ensemble synthetic random fixtures used `rng.uniform(0.0, 0.5)` for `age`, violating `ge=15` — special-cased at commit `33eea57` (Cluster C bundle).
3. `tests/test_scripts/test_tune_lightgbm.py:_WR_FEAT_COLUMNS` — fixed at commit `807f046` (Phase 4 verification).

Defense-in-depth grep for `opp_allowed_wr_fppg_l4` (a distinctive WR-only column) confirmed those were the only 3 missed sites.

## What this closes

**TODO #24's "trailing-8-game unit" branch of the trajectory candidate.** The bundled trajectory probe carried clear signal at the WR cells; the production builder integrates the 4 features at the same unit. **First production-builder integration since PR #21** (RB PBP cols).

**Refined-unit candidates beyond the trailing-8-game unit remain unexplored** under the same TODO:
- per-position aging-curve interaction terms (`age²` for older-RB drop, etc.)
- `is_2nd_year` / `is_3rd_year` flags (collinear with age but might unlock breakout-year signal)
- depth-chart-rank trends
- longer trailing windows (l8 vs l16)
- treating sparsity as a feature (a `has_trajectory_history` indicator that flips on at game 8+)

None queued.

## TE follow-up status

Per the trajectory probe (PR #25), TE adopted **only** under lgb-nb (-0.0107 fpts), not under BaselineModel. **Per-position-routing decision required for any TE integration:** either (a) ship per-position routing to lgb-nb for TE only (precedent: Plan 6's QB-only ensemble suggestion), or (b) ship the schema change for the lgb-nb code path while leaving baseline production routing unchanged. Either is its own decision; not queued.

## Next track

The trajectory family integration on WR is shipped; the remaining trajectory probe-ADOPT cell (TE under lgb-nb) is deferred to a separate decision per the routing question above. Other Track 2 candidates remain open per PM "Next action" §3 (TODO #25 weather features) and the broader feature/model-class roadmap.
