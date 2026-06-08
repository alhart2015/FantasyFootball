# WR Target Decomposition Integration — verdict DO_NOT_ADOPT (binding) + ADOPT (informational) (2026-05-13, on branch `feat/wr-target-decomposition`)

**Spec:** `docs/superpowers/specs/2026-05-13-wr-target-decomposition-integration-design.md`
**Plan:** `docs/superpowers/plans/2026-05-13-wr-target-decomposition-integration.md`
**Builds on:** PR #32 (WR target decomposition probe; SIGNAL on receptions cell, NULL on receiving_yards / _tds).

**Status.** Production integration of `DecomposedBaselineModel` (peer to `BaselineModel`) with per-stat decomposition opt-in. v1 ships WR receptions-only decomposition (volume = `targets`, efficiency = `catch_rate`). `FrozenSampledDistribution` carries within-row coherent sampling through `score_distribution`; persistence uses `QuantileDistribution` summaries via the existing codec branch (no codec edits). **Binding gate `(DecomposedBaselineModel, WR)` vs `(EnsembleModel, WR)` returned `DO_NOT_ADOPT`; informational gate `(DecomposedBaselineModel, WR)` vs `(BaselineModel, WR)` returned `ADOPT`.**

## Per-cell verdicts

| Cell | Incumbent | Candidate | n_paired | RMSE delta (fpts) | RMSE 95% CI | Spearman delta | Spearman 95% CI | Verdict |
|---|---|---|---:|---:|---|---:|---|:---:|
| **Binding** (gates routing) | ensemble | decomposed-baseline | 8402 | +0.0109 | [-0.0080, +0.0285] | -0.0052 | [-0.0087, -0.0018] | **DO_NOT_ADOPT** |
| Informational (probe-equivalent) | baseline | decomposed-baseline | 8402 | -0.0103 | [-0.0145, -0.0060] | +0.00001 | [-0.0006, +0.0006] | ADOPT |

## Probe-vs-gate calibration

The probe (PR #32) predicted per-stat Delta-RMSE on receptions of **-0.0042 fpts** with the ESPN PPR reception coefficient of 1.0 fpt/rec -> expected composite-fpts contribution **-0.0042 fpts**.

| Comparison | Probe Delta-RMSE | Gate Delta-RMSE | Magnitude ratio | Sign |
|---|---:|---:|---:|---|
| Informational (vs `baseline`) | -0.0042 fpts | **-0.0103 fpts** | **2.5x larger** | Same (favorable) |
| Binding (vs `ensemble`) | -0.0042 fpts | +0.0109 fpts | sign flip | Flipped -- candidate underperforms ensemble |

**Probe-vs-gate magnitude flag.** Per PR #31's retrospective rule, the probe's binding-cell composite-fpts magnitude was -0.0042 fpts -- below the ~0.005 fpts marginal-zone threshold. Coverage was strictly above 0.95 across all eval years, so the rule does not strictly fire. The integration's informational cell measured **2.5x the probe magnitude** in the same direction (favorable surprise -- decomposition recipe at composite-fpts level is BETTER than the per-stat receptions-only RMSE prediction suggested). The informational cell's CI is strictly negative, validating the decomposition mechanism. The binding cell's sign flip is mechanistically expected: `EnsembleModel` combines `BaselineModel` + `LightGBMNbModel` and the lgb-nb component contributes lift that decomposition cannot independently outpace.

## Coverage

`targets > 0` rate on WR rows per eval year (production feature cache, no override):

(Coverage was strictly above 0.95 across all 4 eval years per the probe report; no relaxation invoked at the gate; default `--coverage-threshold 0.95` met with margin. See `reports/feature_probe_target_decomposition_summary.md` for the per-eval-year detail.)

## §1.3.5 outcome

**Branch: DO_NOT_ADOPT (binding) + ADOPT (informational) -> infrastructure-only ship.**

- `_PositionDispatch[Position.WR].default_model_class` UNCHANGED at `"ensemble"`. Production routing for WR continues to use `EnsembleModel`.
- `DecomposedBaselineModel` class, `wr_decomposed_baseline` factory, and `_WR_FACTORIES["decomposed-baseline"]` registration STAY in tree as available infrastructure.
- `FrozenSampledDistribution` and `_persistable_dists_for_packing` hook stay in tree.
- `scripts/backtest.py --model decomposed-baseline` STAYS available (with the WR-only positions restriction from the Task 4 fix).
- No backtest snapshot update (`tests/backtest/model_metrics.json` is unchanged -- WR production routing didn't change, so the snapshot is still valid).

## Mechanism interpretation

The informational cell's strict-negative RMSE delta confirms that the decomposition recipe (volume x efficiency for receptions) carries to composite-fpts at a magnitude **larger than the probe's per-stat Ridge-vs-Ridge prediction**. The 2.5x amplification likely reflects:

1. **Within-row coherent sampling** (the `FrozenSampledDistribution` cross-stat correlation path) lifts the composite-fpts CI shape, marginally helping mean accuracy when scoring composes receptions x `reception_pts` with other receiving stats.
2. **Variance modeling at the factor level** (Normal residual std on `catch_rate` ratio, clipped to [0, 1]) produces a better-calibrated receptions distribution than `ParametricGamma` did at the same mu -- the Gamma family's shape parameter was estimated globally and may have been miscalibrated for low-target weeks.

The binding-cell sign flip reflects `EnsembleModel`'s structural lift: its mixture of `BaselineModel` + `LightGBMNbModel` with per-stat pinball-fit weights provides a baseline that `DecomposedBaselineModel` (a Ridge-only model class, even with decomposition) cannot match on the WR cell. The gap (+0.0109 fpts) is small in absolute terms but positive in sign -- decomposition recipe is not enough to recoup what lgb-nb adds to ensemble.

## Recommended next direction

Per spec §1.3.5 "informational ADOPT" branch: **the natural next plan slot is swapping `BaselineModel -> DecomposedBaselineModel` inside `EnsembleModel`'s child A factory** and re-fitting the ensemble weights. This would compound the lgb-nb contribution (proven at the ensemble level) with the decomposition recipe (proven at the baseline level). The new dual-run gate would compare:

- Candidate: `EnsembleModel(child_a=wr_decomposed_baseline, child_b=wr_lightgbm_nb)`
- Incumbent: `EnsembleModel(child_a=wr_baseline, child_b=wr_lightgbm_nb)` (current production)

Expected magnitude: somewhere between -0.0103 fpts (decomposed-baseline-alone improvement) and the ensemble's existing lift. If positive, the routing decision becomes "ensemble stays, but the production ensemble's baseline child is decomposed-baseline." Spec the plan once this PR is merged; it's a follow-up cycle, not in scope for this integration.

## Deferred follow-ups (named, per probe spec §7)

1. **Factor-appropriate sub-model classes** (logistic for `catch_rate`, log-link Gamma for `yards_per_target`, Poisson for `targets`) per probe spec §7.4. Gated on this integration's verdict -- now eligible since decomposition has proven informational lift.
2. **Decomposition for `receiving_yards` / `receiving_tds`** (NULL in probe). Conditional on factor-appropriate sub-models closing those NULL probes first.
3. **Other positions** (RB, QB, TE) -- each its own probe + integration cycle.

## Plan-vs-execution deviations

1. **Worktree + venv routing.** The plan correctly anticipated this (`PYTHONPATH=src ../../../.venv/Scripts/python.exe`), but the executor encountered a Windows path-resolution issue: `/c/Users/...` POSIX paths resolve to `\c\Users\...` (relative) under Windows Python's `Path`. Workaround: use `C:/Users/HartAlden/FantasyFootball/data/...` for `features_root` / `raw_root` kwargs. The plan should document this for the next cross-worktree backtest.
2. **Test fixture `_synthetic_wr_fit_inputs_low_eff_variance` added** in Task 3b. The plan's coherence-test threshold (rho > 0.5) wasn't achievable with the standard fixture (analytical rho ~= 0.26 for the standard yards-per-target variance). The low-eff-variance fixture is a calibration helper, not a workaround for a broken mechanism -- the architectural shared-volume sampling is correct.
3. **Backtest wall-clock: ~34 minutes** for 3-model 4-year WR-only run. Ensemble's 4-stage fit dominated. Plan estimated 5-15 minutes; reality was longer due to LightGBM training time per fold.
4. **4 new `data/ensemble_weights/ensemble_wr_*.json` artifacts** were generated during the backtest's ensemble fits and committed alongside the reports. Pre-existing convention (per PR #35 which also added ensemble weight JSONs alongside its data work).
5. **Spec §1.1 describes a `_sampled_from_quantiles` helper that does not exist.** The spec said `build_stat_distributions` would summarize per-row composed samples into 19 quantiles, wrap them in `QuantileDistribution`, then re-expand to `FrozenSampledDistribution` via `_sampled_from_quantiles` before scoring. The implementation is simpler and avoids the lossy round-trip: `build_stat_distributions` composes directly into `FrozenSampledDistribution`, scoring consumes that live instance (preserving cross-stat correlation via the `n == len` branch), and quantization to `QuantileDistribution` happens only at `_persistable_dists_for_packing` time — downstream of scoring, just before codec packing. The code is correct; the spec's intermediate-quantization step was unnecessary architectural complexity that fell out during implementation. A future contributor reading the spec should treat §1.1's quantize→re-expand sentence as superseded by §3.1.5's "FrozenSampledDistribution is what `build_stat_distributions` emits; QuantileDistribution is only the persisted form" guidance (which is correct in the spec).

## Reports

- `reports/wr_target_decomposition_summary.md` -- this file.
- `reports/adoption_gate_wr_decomposed_baseline_vs_ensemble.{md,csv}` -- binding cell.
- `reports/adoption_gate_wr_decomposed_baseline_vs_baseline.{md,csv}` -- informational cell.

## What this PR ships

- `src/projections/distributions/sampled.py` -- `FrozenSampledDistribution` with write-protection
- `src/projections/models/decomposed_baseline.py` -- `DecompositionSpec`, `DecomposedBaselineModel`, `wr_decomposed_baseline` factory, `_persistable_dists_for_packing` override (QuantileDistribution persistence path)
- `src/projections/models/baseline.py` -- `_persistable_dists_for_packing` hook, `_RIDGE_ALPHA_GRID` constant
- `src/projections/models/__init__.py` -- registration of `wr_decomposed_baseline` in `_WR_FACTORIES["decomposed-baseline"]`; `default_model_class` UNCHANGED at `"ensemble"`
- `src/projections/backtest/harness.py` -- `DecomposedBaselineModel` added to `cast` union
- `src/projections/aggregation/season.py` -- comment noting cross-stat correlation loss at season re-scoring
- `scripts/backtest.py` -- `"decomposed-baseline"` choice + WR-only positions restriction
- `tests/test_distributions/test_sampled.py`, `tests/test_models/test_decomposed_baseline.py`, `tests/test_scripts/test_backtest_cli.py` -- test coverage
