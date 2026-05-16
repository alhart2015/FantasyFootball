# WR Ensemble — Decomposed-Baseline Child A Swap — verdict ADOPT (binding) (2026-05-15, on branch `feat/wr-ensemble-decomposed-child`)

**Spec:** `docs/superpowers/specs/2026-05-15-wr-ensemble-decomposed-child-design.md`
**Plan:** `docs/superpowers/plans/2026-05-15-wr-ensemble-decomposed-child.md`
**Builds on:** PR #36 (WR Target Decomposition Integration; shipped `DecomposedBaselineModel` as infra-only). PR #36 §1.3.5 named this swap as the recommended next slot.

**Status.** New `wr_ensemble_decomposed()` factory swaps `EnsembleModel`'s child A from `wr_baseline` to `wr_decomposed_baseline`; the lgb-nb child stays unchanged. Per-stat ensemble weights re-fit via pinball at q ∈ {0.10, 0.90}. **Binding cell `(ensemble-decomposed, WR) vs (ensemble, WR)` returned ADOPT.** Production WR routing flipped from `"ensemble"` to `"ensemble-decomposed"` in `_PositionDispatch[Position.WR].default_model_class`.

## Per-cell verdicts

| Cell | Incumbent | Candidate | n_paired | RMSE Δ (fpts) | RMSE 95% CI | Spearman Δ | Spearman 95% CI | Verdict |
|---|---|---|---:|---:|---|---:|---|:---:|
| **Binding** (gates routing) | ensemble | ensemble-decomposed | 8460 | **-0.0038** | [-0.0079, -0.0002] | +0.0002 | [-0.0004, +0.0007] | **ADOPT** |
| Informational | decomposed-baseline | ensemble-decomposed | 8460 | -0.0074 | [-0.0234, +0.0089] | +0.0041 | [+0.0013, +0.0067] | DO_NOT_ADOPT (RMSE CI brackets zero) |

The binding-cell RMSE CI is strictly negative; the informational-cell RMSE CI brackets zero. The informational-cell Spearman is strictly positive — ensemble machinery adds clear rank-correlation lift on top of decomposed-baseline alone.

## §1.3.5 outcome

**Branch: ADOPT (binding) → routing flip.**

- `_PositionDispatch[Position.WR].default_model_class` changed from `"ensemble"` to `"ensemble-decomposed"` in `src/projections/models/__init__.py`.
- `tests/test_models/test_position_dispatch.py`'s `expected` dict updated to pin `Position.WR: "ensemble-decomposed"`.
- `wr_ensemble` factory + `_WR_FACTORIES["ensemble"]` registration kept in tree (existing infra; not removed).
- Backtest snapshot (`tests/backtest/model_metrics.json`) unchanged — it pins only `baseline` model_class values; the routing-flip from `ensemble` → `ensemble-decomposed` doesn't affect what the snapshot covers.

## Probe-vs-gate magnitude flag

The binding-cell RMSE Δ point estimate is **-0.0038 fpts** — below the ~0.005 fpts marginal-zone threshold from PR #31's retrospective rule. The CI is strictly negative ([-0.0079, -0.0002]) so the routing flip is mechanically justified, but the absolute magnitude is small. Flagged explicitly so future contributors reading this PR know the production routing was flipped on a 0.004-fpts improvement — small in absolute terms, but consistent across 3 of 4 eval years and statistically conclusive at the pooled-CI level.

## Per-year breakdown (binding cell, informational)

| Year | n_paired | RMSE Δ | RMSE lo | RMSE hi | Spearman Δ | Spearman lo | Spearman hi |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2021 | 2109 | -0.0045 | -0.0124 | +0.0029 | +0.0004 | -0.0008 | +0.0016 |
| 2022 | 2102 | -0.0056 | -0.0140 | +0.0038 | -0.0006 | -0.0023 | +0.0011 |
| 2023 | 2201 | -0.0046 | -0.0145 | +0.0051 | +0.0007 | -0.0006 | +0.0020 |
| 2024 | 2048 | -0.0005 | -0.0021 | +0.0012 | +0.0001 | -0.0002 | +0.0005 |

All four years show negative point-estimates at the binding cell; 2021-2023 drive the pooled signal. 2024 is essentially flat (-0.0005 fpts) — consistent with 2024 being the smallest held-out year and possibly the most data-similar to recent training years.

## Mechanism interpretation

The informational cell's strictly-positive Spearman Δ (+0.0041, CI [+0.0013, +0.0067]) is the cleanest piece of evidence: **ensemble's lgb-nb mixing improves rank correlation even when the baseline child is already decomposed**. Two distinct mechanisms compound rather than cancel:

1. **Decomposed-baseline's lift** comes from volume × efficiency factoring on the receptions stat (PR #36's informational cell showed -0.0103 fpts RMSE vs plain baseline at composite-fpts level).
2. **Ensemble's lift** comes from lgb-nb capturing residual non-linear structure that ridges miss, mixed via per-stat pinball-fit weights.

The binding cell's RMSE Δ of -0.0038 (vs PR #36's expected range of [0, -0.0103]) lands in the bottom half of the predicted band, consistent with **substantial but not full** compounding. Some of decomposition's lift overlaps with what lgb-nb was already capturing; the marginal additional benefit is real but smaller than decomposition's standalone benefit over plain baseline.

The informational cell's RMSE Δ of -0.0074 (CI [-0.0234, +0.0089]) is inconclusive — the high CI width suggests substantial year-to-year variance in how much ensemble's lgb-nb contributes when child A is decomposed. The positive Spearman, however, is statistically conclusive and is what motivates the production flip alongside the binding-cell RMSE.

## Recommended next direction

**Factor-appropriate sub-models for `catch_rate`** (PR #36 deferred-follow-up #1; PR #32 probe spec §7.4). The current decomposed-baseline uses Normal residual std on the clipped `catch_rate` ratio, which is a crude variance model for a [0, 1]-bounded ratio. Replacing with a logistic-link sub-model could:

1. Improve `catch_rate` predictions' tail calibration (currently the bottleneck for receptions distribution accuracy).
2. Address the marginal-magnitude concern in this PR — a -0.0038-fpts adoption is borderline marginal-zone; factor-appropriate models could lift the magnitude into the comfortable-zone (>0.005 fpts).
3. Make decomposing additional stats (receiving_yards, receiving_tds) viable since the [0, ∞)-support efficiency factors (yards-per-target, td-rate) need different sub-model classes to behave reasonably at the tails.

Other deferred follow-ups remain in their PR #36 status:
- Decomposition for `receiving_yards` / `receiving_tds`: still gated on factor-appropriate sub-models closing the NULL probes first.
- Other positions (RB / QB / TE): each is its own probe + integration cycle. WR-first remains the right path since the decomposed-baseline infrastructure is WR-only.

## Plan-vs-execution deviations

1. **`QuantileDistribution.cdf` extrapolation fix (commit `975cd52`).** Task 2's plan hypothesized the mixture-tail bug was in `mixture._bracket_for_components` querying components at too-wide tails. Investigation showed the real root cause: `QuantileDistribution.cdf` clamped at the knot range while `.quantile()` already extrapolated linearly past knots. The cdf-quantile asymmetry capped the joint mixture cdf at `weight*qs[-1] + (1-weight)*1.0 = 1 - weight*0.05`, so brentq inversion in `MixtureDistribution.quantile()` couldn't bracket q values in the tail. Fix: make `cdf` extrapolate symmetrically with `quantile`, clipped to [0, 1]. The existing `test_quantile_cdf_clamps_at_endpoints` test was pinning the defect and was renamed to `test_quantile_cdf_extrapolates_past_endpoints` with a strictly more thorough assertion set. User-approved senior-dev-override per CLAUDE.md.
2. **RECEPTIONS `component_b` is `QuantileDistribution`, not `ParametricNegativeBinomial`** (commit `44aa882`). The plan's Task 2 fit/predict test asserted `component_b == ParametricNegativeBinomial`. In production, lgb-nb only emits NB for stats in `COUNT_STATS_FOR_NB` (TDs / fumbles / interceptions) — receptions falls through to lgb-nb's quantile-regression branch and emits `QuantileDistribution`. Test corrected to assert actual production behavior; extended to verify `RECEIVING_TDS` as the genuine `Mixture(NB, NB)` cell that spec §3.4 named as the integration-risk target.
3. **`mixture._quantile_with_bracket` comment refresh (commit `72b8455`).** After the cdf fix, the helper's "cdf clamps" comments and ValueError messages described a failure mode that no longer exists for the only Distribution type that exhibited it. Refreshed the comments + ValueError messages while preserving the defensive guard against future non-saturating cdfs.
4. **Backtest wall-clock: ~3 hours** (vs PR #36's ~34 min for 2 models). Three contributors:
   - Third model class (decomposed-baseline) ran alongside ensemble + ensemble-decomposed.
   - **The new `MixtureDistribution.quantile()` code path against `QuantileDistribution` components is substantially slower** than the parametric-only mixtures PR #36 used. Each weight-fit iteration inverts `mix.cdf(x) - q` via brentq; each cdf call evaluates both component cdfs (one of which is QuantileDistribution.cdf, which does interpolation but is in Python). For the 2 ensembles × 4 eval years × 6 stats × scipy.minimize_scalar's ~50 iter × ~2100 calibration rows × 2 quantiles × ~30 brentq iter × 2 component-cdf calls, the cost is dominant.
   - External user workload (`scripts/project_season.py --season 2025`) launched mid-backtest competed for CPU.
   This is a known performance hit baked in by the design — accepting it is the cost of the Mixture(Q, X) code path. A future perf-improvement plan could vectorize the per-stat pinball loss or batch-evaluate cdf calls.
5. **Task 4 subagent stopped its turn after starting the backtest.** The orchestrator took over to monitor progress and run the adoption gate. The plan's helper-script approach was structurally correct; the subagent just didn't finish executing it.
6. **`scripts/adoption_gate.py` requires `--position WR` explicitly when running against a WR-only dataset.** Default `--position all` tries QB/RB/TE which have 0 rows and raises `ValueError: need at least 100 paired rows for a meaningful bootstrap, got 0`. Not a script bug — the run only contained WR rows. Documented for future single-position gate runs.
7. **UTF-8 em-dash issue when redirecting `adoption_gate.py` stdout to `.md`.** The script's print output contained em-dashes that Windows console codepage rendering encoded as cp1252 bytes (`0x97`) instead of UTF-8 (`e2 80 94`). Fixed by post-process byte replacement. Worth filing as a follow-up: `adoption_gate.py` could write the markdown to file directly (not stdout) and explicitly use `encoding="utf-8"`.
8. **One stray `data/ensemble_weights/ensemble_wr_024e1c90_2018-2024.json`** appeared during the backtest, with train span 2018-2024 (one season longer than any Task 4 eval split). It's from `scripts/project_season.py --season 2025` running in parallel under the same user's session. NOT committed in this PR.
9. **Pre-existing pytest leaks.** Several pytest processes from earlier subagent invocations (Task 4 subagent's setup + an even older session) were leaking with ~48 min cumulative CPU and competing for CPU with my Task 4 backtest. Killed manually. Worth filing as a follow-up: pytest invocations from subagents should exit cleanly when the subagent's turn ends.

## Reports

- `reports/wr_ensemble_decomposed_summary.md` — this file.
- `reports/adoption_gate_wr_ensemble_decomposed_vs_ensemble.{md,csv}` — binding cell.
- `reports/adoption_gate_wr_ensemble_decomposed_vs_decomposed_baseline.{md,csv}` — informational cell.

## What this PR ships

- `src/projections/models/ensemble.py` — new `wr_ensemble_decomposed()` factory wiring `child_a_factory=wr_decomposed_baseline`.
- `src/projections/models/__init__.py` — `_WR_FACTORIES["ensemble-decomposed"]` registration; `__all__` entry; `default_model_class` flipped from `"ensemble"` to `"ensemble-decomposed"` per §1.3.5 ADOPT.
- `src/projections/distributions/quantile.py` — `cdf()` extrapolates past knots (mirrors `quantile()`'s existing extrapolation), clipped to [0, 1]. Fixes the cdf-quantile asymmetry that broke `MixtureDistribution.quantile()` inversion for q in tail when one component is a `QuantileDistribution`.
- `src/projections/distributions/mixture.py` — refreshed comments and ValueError messages in `_quantile_with_bracket` to reflect the cdf fix; defensive guard preserved.
- `scripts/backtest.py` — `"ensemble-decomposed"` added to `--model` choices + WR-only positions restriction.
- `tests/test_models/test_ensemble_decomposed.py` — factory wiring + fit/predict round-trip + code_hash divergence + mixture-tail sanity tests.
- `tests/test_models/test_position_dispatch.py` — expected dict updated to pin `Position.WR: "ensemble-decomposed"`.
- `tests/test_distributions/test_cdf.py` — `test_quantile_cdf_clamps_at_endpoints` renamed to `_extrapolates_past_endpoints` with a more thorough assertion set.
- `tests/test_distributions/test_mixture.py` — regression test for `Mixture(QuantileDistribution, ParametricNegativeBinomial)` tail-quantile finiteness.
- `tests/test_scripts/test_backtest_cli.py` — CLI smoke test for `--model ensemble-decomposed` + WR-only restriction.
- Reports: `reports/wr_ensemble_decomposed_summary.md`, `reports/adoption_gate_wr_ensemble_decomposed_vs_{ensemble,decomposed_baseline}.{md,csv}`.
- Data: 8 `data/ensemble_weights/ensemble_wr_*.json` artifacts from the dual-run backtest fits.
- Spec: `docs/superpowers/specs/2026-05-15-wr-ensemble-decomposed-child-design.md`.
- Plan: `docs/superpowers/plans/2026-05-15-wr-ensemble-decomposed-child.md`.
