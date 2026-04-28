# Plan 7 — Calibration-aware NB-2 fitting (Model C-NB-cal) — Design

**Status:** **STOPPED at Phase 0** (2026-04-28). Premise was misaligned with the actual mechanism. See `docs/superpowers/research/2026-04-28-calibration-breakdown.md` for the diagnostic finding and `project_management.md` for the verdict. Spec preserved as record-of-decision.
**Date:** 2026-04-28
**Author:** alden + claude
**Builds on:** Plan 5c (PR #13, merged at `166ea97`) — depends on `LightGBMNbModel`, `nb_dispersion_from_residuals` in `src/projections/distributions/parametric.py`, the `POSITION_DISPATCH.factories` dict, and the snapshot keying by `(position, year, metric, model_class)`. Branched from `main` at `166ea97`.

> **Why stopped:** Phase 0 measured per-stat empirical coverage on Plan 5c's C-NB output. Per-stat NB-2 count distributions are *over*-covering at [p10, p90] (gap mean -0.16 across 16 cells; counts cover ~96% vs nominal 80%) while yards stats are well-calibrated (gap ≈ 0). Plan 7 assumed NB-2 is *under*-covering at p10/p90; pinball fitting at q=0.10/0.90 would *narrow* the count distributions, opposite the direction needed for the composite [p10, p90] coverage gap. The composite gap mechanism lives in upper-tail (p95+) behavior — beyond what this plan's loss function targets. See research note for full analysis. Follow-up TODO #30 captures the right next plan: upper-tail count calibration.

---

## 1. Overview

Plan 5c shipped Model C-NB and confirmed the mean-prediction fix worked: Model C-NB strictly dominates Model C-tuned on composite RMSE on every (position, year) cell (16/16) and beats Model A on 11/16 with a worst-case +1.69%. **The binding constraint moved from mean-prediction to calibration.** Model C-NB's weekly `[p10, p90]` coverage regresses vs Model A by mean -0.062 across 16 cells, with RB/TE/WR cells losing 6–12 percentage points of coverage. QB cells are clean wins on every metric.

The mechanism Plan 5c documented: NB-2 dispersion fit via conditional MLE on training residuals (`nb_dispersion_from_residuals`) produces tight predictive intervals when the per-row mean is well-fit, but residual variance on held-out years exceeds the training-fit dispersion — particularly for RB / TE / WR where target variance is larger and regime drift between seasons is real. The fitted NB-2 distribution is therefore too narrow at the [p10, p90] tails on held-out years.

**The fix is to swap the dispersion estimator's objective** — keep everything else in C-NB unchanged. Mean μ continues to come from `lgb.LGBMRegressor(objective="poisson")`. Dispersion α is re-fit by minimizing pinball loss at q=0.10 and q=0.90 on a held-out validation year, instead of maximizing the conditional NB-2 log-likelihood on training residuals. The mechanism the regression cures: pinball loss is the strictly proper score for the [p10, p90] quantiles; minimizing it yields the α whose predictive intervals match the empirical [p10, p90] of the validation distribution, which transfers to held-out years better than train-set dispersion does when train-vs-test variance shifts.

Yards-stat distributions stay as `QuantileDistribution` inherited from C-tuned, untouched. Plan 5c noted "calibration regression on RB/TE/WR is fully attributable to the count-stat NB-2 path" *vs Model C-tuned*, but the absolute coverage shortfall vs Model A is split between yards and counts in unmeasured proportions. Phase 0 is a hard gate that quantifies this split before Phase 1 implementation begins.

This plan delivers Model C-NB-cal (`LightGBMNbCalModel`) as a fifth peer model class. Subclasses `LightGBMNbModel`; overrides only the dispersion-fitting step (and `code_hash`/`model_id`). All count-stat poisson booster training, all yards-stat quantile training, and all predict-time logic are inherited unchanged.

### 1.1 Goals (in scope)

- New public estimator `nb_dispersion_from_pinball(*, mu_hat, actual, quantiles=(0.10, 0.90)) -> float` in `src/projections/distributions/parametric.py`. Bounded 1-d minimization over α in `_NB_DISPERSION_CLIP`; objective is the sum of pinball losses at the requested quantiles, evaluated using `ParametricNegativeBinomial.quantile`. Same return-shape and same clipping behavior as `nb_dispersion_from_residuals`.
- New `LightGBMNbCalModel` subclassing `LightGBMNbModel`. Overrides `fit` to swap the dispersion-fitting call only:
  - `mu_hat_train, y_train` → `mu_hat_val, y_val` for the dispersion fit (μ is the booster's prediction on the validation slice; y is the true target on the validation slice).
  - `nb_dispersion_from_residuals(...)` → `nb_dispersion_from_pinball(mu_hat=mu_hat_val, actual=y_val, quantiles=(0.10, 0.90))`.
  - All other training (the poisson booster fit, yards-stat quantile sub-models, the train/val split selection) inherited unchanged.
- Overrides `code_hash` and `model_id` to reflect the `lightgbm-nb-cal:` prefix and the new file in the hash.
- Per-position factories `qb_lightgbm_nb_cal`, `rb_lightgbm_nb_cal`, `te_lightgbm_nb_cal`, `wr_lightgbm_nb_cal` mirroring the Plan 5c factory shape.
- `POSITION_DISPATCH.factories` extended with `"lightgbm-nb-cal"` per position.
- Backtest harness `--model` selector accepts `lightgbm-nb-cal`. The `--model all` value runs all five classes (`baseline + lightgbm + lightgbm-tuned + lightgbm-nb + lightgbm-nb-cal`).
- Backtest snapshot extended 1504 → 1872 rows (368 new under `model_class="lightgbm-nb-cal"`; same 32 `season_calibration_*` rows skipped per the existing SAMPLED_SUMMARY/MIXED family gate, TODO #28 still open and unrelated).
- Default-on smoke covers all five model classes.
- Per-position parametrized smoke for the new class.
- Phase 0 diagnostic CLI `scripts/diagnose_calibration_breakdown.py`: decomposes Plan 5c's `[p10, p90]` coverage gap vs A into count-stat vs yards-stat contributions; emits a CSV breakdown and a recommendation column. Drives the Phase 0 gating decision.
- Diagnostic report appended to `project_management.md` with the per-cell A vs C-NB vs C-NB-cal comparison and the adoption-gate verdict.
- TODO.md updated with Plan 7 verdict.

### 1.2 Non-goals (deferred)

- **No changes to mean-prediction.** The poisson booster, its hyperparameters, its early-stopping val slice, and its `predict()` output are bit-identical to Model C-NB. Yards-stat training and prediction are unchanged.
- **No new training objectives.** Still `objective="poisson"` for count stats and pinball-loss quantile heads for yards stats.
- **No yards-stat calibration adjustment.** Out of scope unless Phase 0 forces a re-scope; in that case the yards-side fix becomes its own follow-up spec, not folded into this plan.
- **No μ-bucketed or mean-varying α(μ) parameterization.** Single scalar dispersion per (position, stat), same shape as today. The Plan 3e Phase 3 bucketing infrastructure stays preserved as future work; revisiting it under pinball loss is a candidate follow-up plan if Phase 2's scalar-α gate fails.
- **No asymmetric pinball weighting.** Equal weight on q=0.10 and q=0.90. Asymmetric weighting (e.g., heavier upper tail for DFS GPP) is a candidate follow-up if Phase 2 measurement shows an upper-tail regression.
- **No removal or deprecation of any existing model class** during this plan. Models A, C, C-tuned, C-NB, C-NB-cal all coexist. Pruning Model C-tuned and/or Model C (both strictly dominated on RMSE by C-NB) remains housekeeping for a follow-up commit per TODO #29.
- **No widening of `aggregate_to_season` to MIXED or QUANTILE families** (TODO #28). Model C-NB-cal rows skip the same 32 `season_calibration_*` metrics that Model C / C-tuned / C-NB skip today.
- **No new ingest, no PBP / EPA, no target decomposition, no multi-output training.**
- **No application of calibration-aware NB-2 fitting to the Ridge / Plan 3e NB path.** Ridge isn't on the model-improvement track; if it returns to active development the same fix can be retrofitted then.

### 1.3 Adoption gate

Scoped to the constraint this plan actually changes (calibration). Mean prediction is bit-identical to C-NB, so RMSE moves only insofar as Monte Carlo sampling under a different α reorders draws — small and bounded by C-NB's measured envelope.

| Criterion | Threshold |
|---|---|
| Calibration mean delta vs A across 16 cells | ≥ 0 |
| Per-cell calibration vs A | No cell worse than -0.02 |
| RMSE | Allowed to move within ±1.69% (Model C-NB's measured envelope vs A on the same data) |
| Spearman | Within ±0.005 of C-NB (dispersion change shouldn't reorder ranks) |

**Plan 7 is complete on either of:**

- **Model C-NB-cal passes the gate.** → C-NB-cal replaces C-NB in the dispatch table in the same PR. TODO #29 prune (drop Model C-tuned, possibly Model C) becomes the same housekeeping commit. Calibration-aware track closed; pivot to Plan 6 (ensemble) or feature tracks (#3, #23).
- **Model C-NB-cal fails the gate.** → Ship as fifth peer (no dispatch swap). Document the per-cell deltas; identify which sub-mechanism the residual gap is attributable to (per-quantile pinball loss change, fold-instability, dispersion saturation at the clip boundary). Calibration-aware track closed; pivot to Plan 6 or feature tracks. The asymmetric-weighting and μ-bucketed follow-ups (§1.2) only land if Phase 2 evidence specifically points at them.

### 1.4 Phase 0 — diagnostic gate (mandatory; gates whether plan continues)

Before Phase 1 implementation begins, decompose Plan 5c's measured -0.062 mean coverage gap vs A into **count-stat contribution** and **yards-stat contribution**.

The mechanism: total fantasy-point [p10, p90] coverage error is a function of per-stat coverage errors weighted by per-stat fantasy-point variance contributions. A new CLI `scripts/diagnose_calibration_breakdown.py` reads the existing 5c per-row backtest output (`data/backtest/run_<ts>/per_row.parquet` from the C-NB run), computes per-stat empirical [p10, p90] coverage, weights by per-stat fantasy-point variance contribution, and emits a CSV with one row per (position, stat, year) plus per-cell summary rows attributing the total coverage gap to the two paths.

**Decision rule:**

- **Counts contribute ≥ 50% of the coverage gap** → Phase 1 proceeds; this plan is the right answer.
- **Yards contribute ≥ 50% of the coverage gap** → Phase 0 ships as research output; file a yards-side calibration spec as a separate plan; **stop Plan 7 here**. The yards-stat path uses `QuantileDistribution`, not NB-2, and the fix shape (post-hoc conformal widening of the [p10, p90] band) is mechanistically different from this plan's NB-2 dispersion swap.
- **Roughly 50/50 (each between 40% and 60%)** → Phase 1 proceeds with documented expectation that the calibration delta will close at most half the gap; file the yards-side follow-up as a known dependency before this plan ships.

Phase 0 is its own commit (diagnostic script + tests + CSV output + decision recorded in `project_management.md`). The Phase 0 commit is the gate.

---

## 2. Architecture

```
src/projections/
├── distributions/
│   ├── parametric.py            [+ nb_dispersion_from_pinball estimator]
│   └── ...                      [no other changes]
├── models/
│   ├── lightgbm_nb.py           [unchanged]
│   ├── lightgbm_nb_cal.py       [NEW: LightGBMNbCalModel subclasses LightGBMNbModel]
│   └── __init__.py              [+ "lightgbm-nb-cal" factories per position]
└── backtest/
    └── ...                      [unchanged; harness already dispatches by model_class]

scripts/
├── diagnose_calibration_breakdown.py   [NEW: Phase 0 diagnostic CLI]
└── backtest.py                          [+ "lightgbm-nb-cal" in --model choices]

tests/
├── test_distributions/
│   └── test_nb_dispersion_pinball.py   [NEW: estimator unit tests]
├── test_models/
│   ├── test_lightgbm_nb_cal.py          [NEW: subclass behavior; dispersion swap]
│   └── test_lightgbm_nb_cal_smoke.py    [NEW: per-position parametrized smoke]
├── test_scripts/
│   └── test_diagnose_calibration_breakdown.py   [NEW: CLI tests]
├── test_backtest/
│   └── test_harness_quint_model.py      [NEW: 5-model harness smoke]
└── backtest/
    └── model_metrics.json               [extended 1504 → 1872 rows]
```

### 2.1 Estimator: `nb_dispersion_from_pinball`

```python
def nb_dispersion_from_pinball(
    *,
    mu_hat: np.ndarray,
    actual: np.ndarray,
    quantiles: tuple[float, ...] = (0.10, 0.90),
) -> float:
    """Fit NB-2 dispersion alpha by minimizing sum of pinball losses at the
    requested quantiles, holding mu_hat fixed.

    For each candidate alpha, predicts ParametricNegativeBinomial(mu, alpha)
    quantile values on the input rows and computes the standard pinball loss
    sum_i (q - 1{y_i < q_hat_i}) * (y_i - q_hat_i). Minimizes the sum across
    the requested quantiles via bounded 1-d optimization in alpha space, with
    the same _NB_DISPERSION_CLIP and clip-snap semantics as
    nb_dispersion_from_residuals.

    Designed for use on a held-out validation slice (mu_hat is the booster's
    prediction on val rows, actual is the true target on val rows).
    """
```

Same return type, same clip behavior, same degenerate-input handling as `nb_dispersion_from_residuals`. Single-shot scalar optimization per (position, stat); no iterative or multi-shot fitting.

### 2.2 Model class: `LightGBMNbCalModel`

Subclass of `LightGBMNbModel`; overrides:

- `code_hash` — adds `lightgbm_nb_cal.py` to the hash file list.
- `model_id` — prefix `lightgbm-nb-cal:` instead of `lightgbm-nb:`.
- `fit(features, weekly_stats)` — identical to parent except the count-stat dispersion fit:
  - `nb_dispersion_from_residuals(mu_hat=mu_hat_train, actual=y_train)` →
  - `nb_dispersion_from_pinball(mu_hat=mu_hat_val, actual=y_val, quantiles=(0.10, 0.90))`
  - The validation slice is the same `val_season = max(seasons)` slice that the parent already uses for the booster's early-stop. No new split logic; no leakage (α is a separate parameter not optimized via the booster's early-stop callback).
- `predict_distribution` — inherited verbatim. NB-2 distribution for count stats, QuantileDistribution for yards. Per-row family stays `MIXED`.

### 2.3 Per-position model_ids (post-Phase-2; placeholder hashes)

| Position | Model A | Model C-NB | Model C-NB-cal |
|---|---|---|---|
| WR | `baseline:wr:6d955427:2018-2023` | `lightgbm-nb:wr:dc445a2d:2018-2023` | `lightgbm-nb-cal:wr:<hash>:2018-2023` |
| QB | `baseline:qb:c98738f3:2018-2023` | `lightgbm-nb:qb:3ae5b940:2018-2023` | `lightgbm-nb-cal:qb:<hash>:2018-2023` |
| RB | `baseline:rb:5a86c8ee:2018-2023` | `lightgbm-nb:rb:ba2e35cc:2018-2023` | `lightgbm-nb-cal:rb:<hash>:2018-2023` |
| TE | `baseline:te:9c00025b:2018-2023` | `lightgbm-nb:te:e76e590a:2018-2023` | `lightgbm-nb-cal:te:<hash>:2018-2023` |

---

## 3. Phase plan

### Phase 0 — diagnostic gate

1. Implement `scripts/diagnose_calibration_breakdown.py` reading the latest 5c per-row backtest output.
2. Write tests for the breakdown logic (per-stat coverage, variance weighting, attribution math).
3. Run on real 5c output; record the count vs yards split in `docs/superpowers/research/2026-04-28-calibration-breakdown.md`.
4. Apply the Phase 0 decision rule (§1.4); record verdict in `project_management.md`. Either continue to Phase 1 or stop and file the yards-side follow-up plan.

### Phase 1 — implementation

1. Add `nb_dispersion_from_pinball` to `src/projections/distributions/parametric.py` + unit tests covering the strictly-proper minimum, clip behavior, and degenerate inputs (all-zero actuals, single-row inputs, quantile-edge cases).
2. Add `src/projections/models/lightgbm_nb_cal.py` with `LightGBMNbCalModel` + tests covering the dispersion-swap (parent's MLE α and child's pinball α should differ on a synthetic dataset where train-residual and val-residual variances diverge).
3. Wire the four per-position factories.
4. Extend `POSITION_DISPATCH.factories` with `"lightgbm-nb-cal"`.
5. Extend `scripts/backtest.py --model` choices and `--model all` enumeration.
6. Per-position parametrized smoke + harness 5-model smoke.
7. Confirm yards-stat predictions are bit-identical to C-NB on a fixture (analogous to C-NB's `test_yards_stat_predictions_match_tuned_baseline`).

### Phase 2 — backtest run + adoption verdict

1. Run full walk-forward harness with `--model lightgbm-nb-cal`. Snapshot extends 1504 → 1872 rows.
2. Build the per-cell A vs C-NB vs C-NB-cal comparison table (same shape as Plan 5c's table).
3. Apply the §1.3 adoption gate. Document per-cell deltas and per-mechanism diagnosis.
4. Additionally report **per-quantile pinball loss change** at q=0.10 and q=0.90 separately (not just the [p10, p90] coverage delta). This is the load-bearing diagnostic for the upper-tail-distortion risk flagged in §4.

### Phase 3 — final review + PR

1. Update `project_management.md` with the Plan 7 entry: per-cell A/C-NB/C-NB-cal table, gate verdict, adoption decision, and (if gate passed) dispatch-swap commit.
2. Update TODO.md with Plan 7 verdict.
3. If gate passed: prune Model C-tuned (TODO #29) and consider pruning Model C (untuned) in the same PR. Snapshot rows for pruned classes are deleted in the same commit as the dispatch change.
4. Open PR.

---

## 4. Risks and mitigations

- **Risk: 2022 is a single year; α may overfit fold variance.** Mitigation: Phase 2 reports per-fold pinball loss using each fold's own α to bound the variance. If results are noisy across folds, a follow-up plan upgrades to multi-year validation or rolling-window. Decision rule: if any (position, stat) cell's α varies by more than 5x across folds, flag for follow-up.
- **Risk: pinball-fit α widens upper tail too much, hurting DFS GPP signal.** Mitigation: Phase 2 reports per-quantile pinball loss change separately (not just composite). If q=0.90 pinball worsens by more than 10% on any cell, flag asymmetric-weighted fitting (§1.2 deferred goal) as the follow-up plan.
- **Risk: dispersion saturates at the `_NB_DISPERSION_CLIP[0]` (= 0.01) boundary on cells where the validation set is too narrow to support large α.** Mitigation: clip-snap is preserved from `nb_dispersion_from_residuals`; Phase 2 reports clip-saturation per cell. If saturation is widespread, the follow-up is wider clip bounds and / or a regularized objective.
- **Risk: leakage from re-using the booster's early-stop val year for dispersion fitting.** Mitigation: the booster's early-stop is a single-scalar selection (best_iter); α-fit is a separate single-scalar selection on the same slice. The two scalars don't share gradient information; α-fit cannot leak booster generalization data. Documented in the estimator docstring and mirrored in tests.
- **Risk: Phase 0 diagnostic shows yards dominate the coverage gap, invalidating the plan's premise.** Mitigation: the Phase 0 gate exists exactly for this scenario. Stop Plan 7, file yards-side calibration plan separately.

## 5. Out-of-scope / explicit deferrals

- **Plan 6 (Model D ensemble).** Independent track. Can run before, after, or in parallel with this plan.
- **TODO #29 (prune Model C-tuned).** Done as housekeeping in this plan only if §1.3 adoption gate passes. Otherwise stays open.
- **TODO #28 (widen `aggregate_to_season` for QUANTILE / MIXED).** Not blocked by this plan; the same gap remains for the new model class.
- **Asymmetric pinball weighting** (§1.2). Follow-up if Phase 2 evidence specifically supports it.
- **μ-bucketed α(μ) parameterization** (§1.2). Follow-up; Plan 3e Phase 3 infrastructure preserved.
- **Yards-stat calibration adjustment.** Follow-up only if Phase 0 forces a re-scope.
- **Application to Ridge / Plan 3e NB path.** Out of scope; retrofit only if Ridge returns to active development.

---

## 6. Acceptance criteria

- Phase 0 diagnostic CSV committed; count-vs-yards split decision recorded in `project_management.md`.
- New estimator `nb_dispersion_from_pinball` lands with unit tests; verified to differ from `nb_dispersion_from_residuals` on a synthetic dataset where train/val variances diverge.
- `LightGBMNbCalModel` implements only the dispersion-swap; yards-stat predictions are bit-identical to C-NB on a fixture.
- Backtest snapshot extended 1504 → 1872 rows; harness smokes pass for all five model classes.
- Per-cell A vs C-NB vs C-NB-cal comparison table committed to `project_management.md`.
- §1.3 adoption-gate verdict applied with explicit pass / fail per criterion. Dispatch swap landed if and only if the gate passes.
- Per-quantile pinball loss change reported separately at q=0.10 and q=0.90 for upper-tail-distortion diagnosis.
- TODO.md updated with Plan 7 verdict.
- All `pytest -v`, `mypy src tests`, `ruff check src tests`, `ruff format --check src tests` pass.
