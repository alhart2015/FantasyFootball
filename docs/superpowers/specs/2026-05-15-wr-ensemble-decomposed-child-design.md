# WR Ensemble — Decomposed-Baseline Child A Swap — Design

**Status:** draft (brainstorming, 2026-05-15). Ready for user review.
**Date:** 2026-05-15
**Author:** alden + claude
**Sub-project of:** FantasyFootball / Projections Core
**Builds on:**
- WR Target Decomposition Integration (PR #36, merged 2026-05-14) — shipped `DecomposedBaselineModel`, `wr_decomposed_baseline` factory, `FrozenSampledDistribution`, and registered `_WR_FACTORIES["decomposed-baseline"]`. Binding cell `(DecomposedBaselineModel, WR) vs (EnsembleModel, WR)` returned `DO_NOT_ADOPT` (RMSE Δ +0.0109, CI [-0.0080, +0.0285]; Spearman Δ -0.0052). **Informational cell `(DecomposedBaselineModel, WR) vs (BaselineModel, WR)` returned `ADOPT`** at RMSE Δ -0.0103 fpts (CI [-0.0145, -0.0060]) — 2.5× the probe's predicted magnitude, same direction. WR production routing stays on `ensemble`. PR #36 spec at `docs/superpowers/specs/2026-05-13-wr-target-decomposition-integration-design.md` §1.3.5 names this swap as the recommended next slot.
- WR Target Decomposition Probe (PR #32, merged 2026-05-10) — original `feat/probe-target-decomposition` probe; `docs/superpowers/specs/2026-05-10-target-decomposition-probe-design.md`.

**Branch:** `feat/wr-ensemble-decomposed-child` cut from `main` at `4af7573` (PR #36 merge commit).

---

## 1. Overview

PR #36 shipped `DecomposedBaselineModel` as available infrastructure but left WR production routing on `EnsembleModel(child_a=wr_baseline, child_b=wr_lightgbm_nb)`. The informational cell proved the decomposition recipe carries to composite-fpts at a magnitude larger than the per-stat probe predicted, but the binding cell against ensemble was DO_NOT_ADOPT — `EnsembleModel`'s lgb-nb component contributes lift that decomposed-baseline alone cannot recoup.

This spec evaluates the natural follow-up: **does ensemble lift compound with the decomposition recipe**? Concretely, swap `EnsembleModel`'s child A factory from `wr_baseline` to `wr_decomposed_baseline`, re-fit per-stat ensemble weights via pinball on the calibration year, and run a fresh adoption gate against the current production `EnsembleModel`.

The new model class is **not new code** — it's `EnsembleModel` with a different child-A factory. The factory wiring is the only production-code addition. All persistence, mixture machinery, and codec handling are inherited unchanged from `EnsembleModel`. The `_persistable_dists_for_packing` hook on `DecomposedBaselineModel` already produces `QuantileDistribution` summaries for decomposed stats, and the codec's existing `MIXTURE` branch in `_pack_single` / `_unpack_single` recursively handles `MixtureDistribution(component_a=QuantileDistribution, component_b=ParametricNegativeBinomial)` — that code path exists today but has not been exercised in production (BaselineModel emits parametric distributions; the existing production ensemble's mixture components are both parametric).

The shipping decision is binary and bound to the **`(EnsembleModel-with-decomposed-baseline-child, WR)` vs `(EnsembleModel, WR)`** verdict.

### 1.1 Goals (in scope)

- **New factory `wr_ensemble_decomposed()` in `src/projections/models/ensemble.py`** — zero-arg, returns `EnsembleModel` wired with `child_a_factory=wr_decomposed_baseline` and `child_b_factory=wr_lightgbm_nb`. ~6 lines mirroring the existing `wr_ensemble()` factory.

- **Register `"ensemble-decomposed"` in `_WR_FACTORIES`** in `src/projections/models/__init__.py`. Add to `__all__`. Do **not** change `_PositionDispatch[Position.WR].default_model_class` in the initial PR commits — the flip is conditional on the adoption-gate verdict (§1.3.5).

- **Add `"ensemble-decomposed"` to `scripts/backtest.py`'s `--model` choices** and to its WR-only positions restriction (mirroring the `decomposed-baseline` handling from PR #36).

- **Add `"ensemble-decomposed"` to the `cast` union in `src/projections/backtest/harness.py`** alongside `DecomposedBaselineModel` (also from PR #36).

- **Tests under `tests/test_models/test_ensemble.py` (or `tests/test_models/test_ensemble_decomposed.py` if cleaner)**:
  - Unit — `wr_ensemble_decomposed()` factory returns an `EnsembleModel` whose `_config.child_a_factory()` returns a `DecomposedBaselineModel` (and whose `child_b_factory()` returns an `LightGBMNbModel`); `position == WR`; `target_stats == _WR_TARGET_STATS`.
  - Unit — `EnsembleModel` with decomposed-baseline child A fits cleanly on a synthetic 3-year WR fixture, persists weights JSON to a tmp_path-overridden `weights_dir`, and produces a fitted `model_id` distinct from `wr_ensemble().fit(same data).model_id` (different code_hash due to different child A code_hash).
  - Unit — `predict_distribution` returns a `ProjectionWeeklySchema`-validating frame; `family` column is `MIXED`; per-row `params` blob round-trips through `unpack_per_stat_params` to per-stat `MixtureDistribution` dicts where the decomposed stats' `component_a` is a `QuantileDistribution` (not `ParametricGamma` / `ParametricNormal`).
  - Unit — `EnsembleModel.code_hash` differs between `wr_ensemble()` and `wr_ensemble_decomposed()` post-fit on identical data, because child A's code_hash differs. Confirms persistence keying separates the two ensembles' weight artifacts.
  - Integration — `_WR_FACTORIES["ensemble-decomposed"]()` returns an unfitted `EnsembleModel` with the correct child wiring; `POSITION_DISPATCH[Position.WR].factories["ensemble-decomposed"]` resolves; `default_model_class` is still `"ensemble"` (not flipped pre-gate).
  - Integration — `Model`-protocol conformance via the existing pattern in `tests/test_models/test_position_dispatch.py` (PR #36 added a parametrized check; just ensure `"ensemble-decomposed"` is covered).

- **Run the adoption gate. Two cells (one binding, one informational).** Both produced from a single dual-run backtest via `scripts/backtest_dual.py` per PR #29 / PR #36 pattern:
  - **Binding:** `(EnsembleModel-with-decomposed-baseline-child, WR)` vs `(EnsembleModel, WR)` — gates production routing flip.
  - **Informational:** `(EnsembleModel-with-decomposed-baseline-child, WR)` vs `(DecomposedBaselineModel, WR)` — tests whether ensemble's mixture machinery still adds lift on top of decomposition. Mechanistic check: if positive, lgb-nb contribution survives the child-A swap.

- **Reports:**
  - `reports/wr_ensemble_decomposed_summary.md` — verdicts + 95% CI for both cells; per-year breakdowns; §1.3.5 outcome narrative; deferred-follow-up disposition.
  - `reports/adoption_gate_wr_ensemble_decomposed_vs_ensemble.{md,csv}` — binding cell.
  - `reports/adoption_gate_wr_ensemble_decomposed_vs_decomposed_baseline.{md,csv}` — informational cell.

- **Per-§1.3.5 outcome execution.** Conditional on verdict (see §1.3.5 below).

- **PM/TODO update.** `project_management.md` decision-log entry at top with verdicts and §1.3.5 outcome; `TODO.md` updated to close (or extend) the "ensemble-child swap" follow-up named in PR #36's summary.

### 1.2 Non-goals (deferred)

- **No factor-appropriate sub-model classes** (logistic for `catch_rate`, log-link Gamma for `yards_per_target`, Poisson for `targets`). Named follow-up per PR #32 spec §7.4 and PR #36 deferred-follow-up #1 — separate probe + integration cycle. Including it here would bundle two independent design variables into one gate and obscure attribution.
- **No decomposition for `receiving_yards` or `receiving_tds`.** Both NULL in PR #32 probe; PR #36 deferred-follow-up #2 names a refined-unit re-probe as the prerequisite. Adding them here without a re-probe would compound the same risk PR #36 explicitly avoided.
- **No other positions.** RB / QB / TE each get their own probe + integration cycle. WR-only because the decomposed-baseline infrastructure is WR-only (PR #36 §1.1).
- **No changes to `EnsembleModel.fit` / `predict_distribution` / `_fit_weight_for_stat`.** The class is intentionally child-agnostic — swapping a factory is the only edit needed. If the gate reveals a numerical issue with `MixtureDistribution(QuantileDistribution, NegativeBinomial)` (e.g., the `_quantile_with_bracket` helper in `mixture.py` mishandles a `QuantileDistribution` component on tail-quantile lookups), that becomes its own scoped fix — not in scope here unless the unit tests in §1.1 surface it.
- **No new Distribution classes, codec branches, or schema edits.** All persistence and mixture machinery is reused.
- **No `--coverage-threshold` change.** Default 0.95 per PR #36 (probe coverage was 0.981+; same data this time).
- **No regression-test additions on `_<POS>_FEATURE_COLUMNS` parity.** No feature-additions in this PR.
- **No model_id prefix change.** Both `wr_ensemble()` and `wr_ensemble_decomposed()` produce `model_id` strings starting with `"ensemble:"` and are distinguished by `code_hash` only. Adding a `model_class_suffix` field to `_EnsembleConfig` to disambiguate the prefix is premature — `model_id` is opaque to callers, and the `data/ensemble_weights/` filename derives from `model_id` so weight artifacts already separate cleanly. See §5 risk #3.

### 1.3 Adoption gate

Adoption decisions are per position. For Position WR, the adoption gate compares `ensemble-decomposed` against the incumbent `_PositionDispatch[Position.WR].default_model_class` which is `ensemble`.

**Inputs.** Per-row predictions from both classes for the same `(gsis_id, season, week)` rows across all held-out years (2021–2024), pulled from a dual-run `scripts/backtest_dual.py` invocation per PR #29 / PR #36 pattern. After pairing, WR contributes ~8,400 paired rows (matching PR #36's binding cell n_paired = 8,402).

**Statistical machinery.** Paired bootstrap with `n_bootstrap=1000`, deterministic seed `42`. Resampling unit is the paired player-week — both candidate and incumbent are scored on the same draw.

**Per-position metrics.**
- **RMSE delta** (`candidate − incumbent`): pooled across all held-out years. Negative = candidate wins.
- **Spearman delta**: per-year Spearman computed within each held-out year, then averaged unweighted across years.

Per-cell breakdowns (one row per held-out year) are emitted for inspection but do **not** gate adoption.

**Verdict rule** (identical to PR #36 § 1.3):
```
PASS_RMSE      := rmse_delta.hi_95     <  0.0
PASS_SPEARMAN  := spearman_delta.lo_95 > -0.02

if  PASS_RMSE and  PASS_SPEARMAN:  ADOPT
if  PASS_RMSE and !PASS_SPEARMAN:  MARGINAL — investigate before adopting
if !PASS_RMSE and  PASS_SPEARMAN:  DO_NOT_ADOPT
if !PASS_RMSE and !PASS_SPEARMAN:  DO_NOT_ADOPT
```

**Adoption is manual.** `scripts/adoption_gate.py` emits a report; a human reads the verdicts and edits `_PositionDispatch[Position.WR].default_model_class` if the binding cell verdict is `ADOPT`. The CLI never writes to source.

**Tooling.**
```
python -m scripts.adoption_gate \
  --baseline-run data/backtest/run_<ts_ensemble> \
  --candidate-run data/backtest/run_<ts_ensemble_decomposed> \
  --candidate ensemble-decomposed \
  --csv-out reports/adoption_gate_wr_ensemble_decomposed_vs_ensemble.csv
```

A second invocation against `--baseline-run` for `decomposed-baseline` produces the informational cell's report.

### 1.3.5 Per-position §1.3.5 outcome matrix

Conditional on the **binding** `(EnsembleModel-with-decomposed-baseline-child, WR)` vs `(EnsembleModel, WR)` verdict. The informational cell informs interpretation but does not change the branch.

- **ADOPT (binding)** — Flip `_PositionDispatch[Position.WR].default_model_class` from `"ensemble"` to `"ensemble-decomposed"` in `src/projections/models/__init__.py`. Keep `wr_ensemble` factory available. Existing `data/ensemble_weights/ensemble_wr_*.json` artifacts for both ensembles remain in-tree (no GC; they are addressed by distinct `model_id` strings). PM logs the flip + magnitude. **Update the backtest snapshot** (`tests/backtest/model_metrics.json`) since WR production routing changed.

- **MARGINAL (binding RMSE PASS, Spearman fail)** — Treat as DO_NOT_ADOPT for production routing per template's "investigate before adopting" guidance, but keep the `wr_ensemble_decomposed` factory + `_WR_FACTORIES["ensemble-decomposed"]` registration as available infrastructure. PM logs the verdict + the Spearman-floor-violation magnitude. No backtest snapshot update (production routing unchanged).

- **DO_NOT_ADOPT (binding)** — Keep WR on `ensemble` in production. Keep `wr_ensemble_decomposed` factory + registration as available infrastructure. **The informational cell determines the recommended follow-up:**
  - **Informational ADOPT** (vs `decomposed-baseline`) — Ensemble's lgb-nb contribution still adds lift on top of decomposition, but not enough to outperform the current production ensemble. This is the expected-if-disappointing outcome. PM logs the closure of the ensemble-child swap direction. Next recommended slot: factor-appropriate sub-models (PR #36 deferred-follow-up #1) on `catch_rate`, which addresses the variance-modeling concern raised in PR #36's mechanism-interpretation §, since the Normal-on-clipped-ratio variance model is the most-likely source of mis-calibration constraining decomposition's lift.
  - **Informational MARGINAL / DO_NOT_ADOPT** (vs `decomposed-baseline`) — Ensemble machinery does not compound with decomposition. Two distinct signals are getting cancelled or absorbed. PM logs the closure of the ensemble-child swap direction in stronger terms. Next recommended slot may be a diagnostic probe rather than another integration — e.g., per-stat ablation of which mixture weights changed materially when child A switched from baseline to decomposed-baseline, to attribute the absorption mechanism. Skip factor-appropriate sub-models until the mechanism is understood; they would compound the same risk.
  - **Informational REGRESSION** (vs `decomposed-baseline`) — Ensemble-with-decomposed-child is actively worse than decomposed-baseline alone. Strong evidence that the mixture is hurting decomposed-baseline's predictions, possibly via a numerical issue in `MixtureDistribution`'s tail-quantile handling on `QuantileDistribution` components, or via the calibration year's per-stat weights pinning to a regime that doesn't generalize. **Full revert** of `wr_ensemble_decomposed` factory + `_WR_FACTORIES["ensemble-decomposed"]` registration + backtest CLI choice. Spec + plan + reports stay as historical record. PM logs the closure + the mechanism hypothesis.

- **REGRESSION (binding RMSE CI strictly > 0)** — `EnsembleModel-with-decomposed-baseline-child` is actively worse than `EnsembleModel`. Full revert per PR #31 / PR #36 REGRESSION precedent: remove `wr_ensemble_decomposed` factory + `_WR_FACTORIES["ensemble-decomposed"]` registration + backtest CLI choice + harness `cast` union entry. Keep this spec as historical record. PM logs the closure + the magnitude.

**Expected magnitude.** PR #36's informational cell (`decomposed-baseline` vs `baseline`) was RMSE Δ -0.0103 fpts. The binding cell here (`ensemble-decomposed` vs `ensemble`) is the same delta type measured one layer up. Plausible range: somewhere between 0 and -0.0103 fpts, biased toward zero because much of decomposition's lift may already be captured by lgb-nb's modeling. A binding-cell magnitude < 0.005 fpts on a marginal CI should trigger the same marginal-zone flag PR #36 documented.

### 1.4 Success criteria

The spec is complete iff all of:

1. **Factory + registration + tests land cleanly.** `pytest -v` (full suite), `mypy src tests scripts` (zero violations), `ruff check src tests scripts` (zero violations), `ruff format --check src tests scripts` (no drift).
2. **`wr_ensemble_decomposed` round-trips through the standard backtest pipeline** — `scripts/backtest.py --model ensemble-decomposed --positions wr` produces a `results.parquet` validating against `ProjectionWeeklySchema`. The unit-test smoke against synthetic data is a precondition; the full backtest is the real-data check.
3. **The dual-run backtest + adoption gate runs successfully on WR** for the binding cell (`ensemble-decomposed` vs `ensemble`) and the informational cell (`ensemble-decomposed` vs `decomposed-baseline`), both across the standard `2021-2024` eval window with `--coverage-threshold 0.95`.
4. **The summary report (`reports/wr_ensemble_decomposed_summary.md`) records all of:**
   - Both gate cells' verdicts + 95% CI on RMSE Δ + Spearman Δ.
   - Per-year breakdowns.
   - The §1.3.5 outcome narrative and any routing-flip executed.
   - The deferred-follow-up disposition per §1.3.5 branch.
   - Probe-vs-gate magnitude flag check on the binding cell (informational only; does not change the verdict).
5. **The §1.3.5 outcome is executed** before the PR is merged — routing flip on ADOPT (plus snapshot update), no-flip on MARGINAL / DO_NOT_ADOPT (with infra kept), or full revert on REGRESSION.

---

## 2. Inputs

### 2.1 Source data — already in cache

All inputs already exist in the feature cache and weekly stats (same as PR #36):

| Column | Type | Source | Use |
|---|---|---|---|
| WR features | `WrFeaturesSchema` | `data/features/wr/season=YYYY/week=WW/part.parquet` | X matrix for both child A (decomposed-baseline) and child B (lgb-nb) inside the new ensemble. |
| WR target stats | per `WeeklyStatsSchema` | same partition | y for both children; child A's decomposed stat (`receptions`) uses the same `targets`/`receptions` columns as PR #36. |

**No new ingest, no schema changes, no feature-cache refresh.**

### 2.2 No new caller-script changes

`scripts/backtest_dual.py`, `scripts/adoption_gate.py`, and `scripts/refresh_features.py` are unchanged.

`scripts/backtest.py` gets one line for the new `--model` choice + one entry in the WR-only restriction (mirroring PR #36's `decomposed-baseline` handling).

---

## 3. Architecture

### 3.1 New factory `wr_ensemble_decomposed`

```python
# src/projections/models/ensemble.py

def wr_ensemble_decomposed() -> EnsembleModel:
    """Construct an unfitted WR ensemble whose child A is decomposed-baseline.

    Differs from `wr_ensemble` only in `child_a_factory`. The pinball-weight-fit
    calibration step (Stage 3 of EnsembleModel.fit) re-runs against the
    decomposed-baseline-vs-lgb-nb children, producing per-stat weights tuned to
    the new child A.
    """
    return EnsembleModel(
        config=_EnsembleConfig(
            position=Position.WR,
            target_stats=_WR_TARGET_STATS,
            child_a_factory=wr_decomposed_baseline,
            child_b_factory=wr_lightgbm_nb,
        )
    )
```

Imports needed: `wr_decomposed_baseline` from `projections.models.decomposed_baseline` (already in the module's namespace via existing exports — no new circular-import risk; `decomposed_baseline.py` doesn't import `ensemble.py`).

Type note: `_EnsembleConfig.child_a_factory: Callable[[], BaselineModel]`. `wr_decomposed_baseline()` returns `DecomposedBaselineModel` which subclasses `BaselineModel`. `Callable` is covariant in its return type — mypy accepts this without an explicit cast or wrapper.

### 3.2 Registry + CLI + harness wiring

`src/projections/models/__init__.py`:
- Add `wr_ensemble_decomposed` to the existing `from projections.models.ensemble import (...)` block.
- Add `"ensemble-decomposed": wr_ensemble_decomposed` to `_WR_FACTORIES`.
- Add `"wr_ensemble_decomposed"` to `__all__`.
- **Do not** change `_PositionDispatch[Position.WR].default_model_class` — flip is the §1.3.5 outcome step.

`scripts/backtest.py`:
- Add `"ensemble-decomposed"` to the `--model` `choices=[...]` list.
- Add `"ensemble-decomposed"` to the WR-only positions-restriction block alongside `"decomposed-baseline"`.

`src/projections/backtest/harness.py`:
- The existing `cast` union covering `BaselineModel | DecomposedBaselineModel | LightGBMModel | LightGBMNbModel | LightGBMTunedModel | EnsembleModel` already includes `EnsembleModel` — no change needed. The new factory returns an `EnsembleModel` instance.

### 3.3 Persistence — no edits

- `EnsembleModel` already persists per-position weight artifacts to `data/ensemble_weights/{sanitized_model_id}.json`. The new ensemble's `model_id` includes its distinct `code_hash` (composed from child A's code_hash, which differs between baseline and decomposed-baseline), so the weight artifact path will be distinct from the existing `wr_ensemble` artifact. No collision.
- The codec already supports `MIXTURE` with arbitrary registered component families. `MixtureDistribution(component_a=QuantileDistribution, component_b=ParametricNegativeBinomial)` recurses through `_pack_single` / `_unpack_single` cleanly — `QuantileDistribution` has a branch in both; `MixtureDistribution` has a branch in both. No codec edits.

### 3.4 Mixture-of-(QuantileDistribution, NegativeBinomial) — verification

This is the first production code path that mixes a `QuantileDistribution` with a `ParametricNegativeBinomial` inside a `MixtureDistribution`. The path exists in code but has not been exercised in production (BaselineModel emits parametric distributions; the current production ensemble's components are always parametric). The unit tests in §1.1 must explicitly cover:

- Per-row `predict_distribution` output's `params` blob round-trips through `unpack_per_stat_params` to per-stat `MixtureDistribution` instances where the decomposed stats' `component_a` is a `QuantileDistribution`.
- `_fit_weight_for_stat` produces a finite per-stat weight on a synthetic calibration year where child A's predictions for `RECEPTIONS` are `QuantileDistribution`. This exercises `_bracket_for_components` + `_quantile_with_bracket` against a `QuantileDistribution` component.
- `MixtureDistribution(QuantileDistribution, ParametricNegativeBinomial, weight=0.5).quantile(0.10)` and `.quantile(0.90)` return finite values (not NaN / inf). Sanity check on tail behavior.

If any of these surface a numerical issue, it's in scope to fix the underlying helper (likely `mixture._quantile_with_bracket`) before the real-data gate runs — see §5 risk #2.

### 3.5 Schema — no edits

`WrFeaturesSchema`, `WeeklyStatsSchema`, `ProjectionWeeklySchema` all unchanged. The new ensemble emits `family = MIXED` per-row identical to the existing ensemble.

---

## 4. Execution sequence

Order matters; later tasks depend on earlier ones.

1. **`wr_ensemble_decomposed` factory + unit tests on factory wiring + code_hash divergence** — net-new factory, no behavior change to `EnsembleModel`. Verifies the type-level wiring lands cleanly.
2. **Synthetic-data fit + predict tests on the new ensemble** — exercises Stage 1–4 of `EnsembleModel.fit` end-to-end with decomposed-baseline child A, including the `MixtureDistribution(QuantileDistribution, ParametricNegativeBinomial)` round-trip through the codec. **This is the task most likely to surface a numerical issue in `mixture.py`** — if any test fails on tail quantiles or bracket finding, scope a fix to `mixture.py` here (not deferred).
3. **Registry + CLI + harness wiring + integration tests** — `_WR_FACTORIES["ensemble-decomposed"]`, `scripts/backtest.py` choice, harness cast union check. `default_model_class` stays on `"ensemble"`.
4. **Real-data dual-run backtest + adoption gate (binding + informational)** — operational task; no source edits. Produce the binding + informational cell reports.
5. **Per-§1.3.5 outcome execution + writeup + PM/TODO update** — conditional source edits per the verdict matrix, plus `reports/wr_ensemble_decomposed_summary.md` + `project_management.md` + `TODO.md`. On ADOPT, also update `tests/backtest/model_metrics.json` (production routing changed).

Each task touches ≤ 5 files per CLAUDE.md "phased execution" rule.

---

## 5. Risk register

1. **Expected-magnitude risk (highest).** The expected RMSE Δ on the binding cell is plausibly in the marginal zone (between 0 and -0.0103 fpts, biased toward zero). If the binding cell ADOPTs at a magnitude < 0.005 fpts on a borderline CI, the routing flip is real but small, and the marginal-zone flag from PR #31's retrospective applies. **Mitigation:** the §1.3.5 ADOPT branch is the same regardless of magnitude (production gets a real, if small, improvement). The summary report calls out the marginal-zone flag explicitly so the routing-flip decision is made with eyes open. PR #31's retrospective rule fired only when a probe ADOPT cell sign-flipped at the gate — here we're at the integration layer one step further, so the rule does not strictly apply but the framing carries.

2. **`MixtureDistribution(QuantileDistribution, NegativeBinomial)` tail-quantile risk.** The mixture's `.quantile(q)` uses `_quantile_with_bracket` + `_bracket_for_components`. The bracket helper queries each component's `.quantile` at `q=0.01` and `q=0.99` (or similar tail anchors) to set the root-finding bounds. `QuantileDistribution.quantile(0.99)` returns the value at the highest persisted quantile knot (q=0.95) — there is no extrapolation. If `_bracket_for_components` queries beyond the persisted range, the bracket could be too narrow on the upper tail, causing root-finding to fail or pin to the bracket edge. **Mitigation:** §3.4 names this as a test target. If a unit test surfaces NaN / inf or pinned-edge values, the fix scope is `_bracket_for_components` (probably querying at q ∈ [0.05, 0.95] for QuantileDistribution components rather than [0.01, 0.99]) — small, scoped change in the mixture-helpers layer.

3. **`model_id` prefix collision.** Both `wr_ensemble()` and `wr_ensemble_decomposed()` produce `model_id` strings of the form `"ensemble:wr:<code_hash>:<years>"`. Distinct only by `code_hash`. **Risk:** a future caller that filters by `model_id.startswith("ensemble:")` to identify "the ensemble model" will match both variants. **Mitigation:** no such caller exists today (`model_id` is opaque to production code; the `data/ensemble_weights/` filename derives directly from the sanitized full `model_id`, so weight artifacts are correctly separated). If a future caller needs to distinguish the variants, add a `model_class_suffix` field to `_EnsembleConfig` and a unit test pinning the prefix. Not in scope for v1.

4. **Calibration-year weight pinning.** `EnsembleModel.fit`'s Stage 3 fits per-stat weights via pinball at q ∈ {0.10, 0.90} on the calibration year (last train year). If the decomposed child A's `RECEPTIONS` `QuantileDistribution` happens to align better with truth than the parametric Gamma did on the calibration year specifically, but worse on test years, the pinball-fit weight will overweight child A on a quirk of the calibration year. **Mitigation:** the per-year breakdown in the adoption-gate report exposes this — if one held-out year drives the verdict and the others are flat, the per-year sub-CIs will show it. The §1.3.5 MARGINAL / DO_NOT_ADOPT branches account for this case.

5. **Re-fitting cost.** `EnsembleModel.fit` runs Stage 1 (weight-fit children on [S, Y-2]) + Stage 4 (re-fit children on full [S, Y-1]) — both children fit twice per fold. The new ensemble adds the decomposition sub-models' fits inside child A both times. Per PR #36's plan-vs-execution deviation #3, the backtest wall-clock was ~34 minutes for 3-model 4-year WR-only. Adding `ensemble-decomposed` to the dual-run brings it to 2-model 4-year, somewhere around 25 minutes. **Mitigation:** none needed — wall-clock is acceptable. Plan task notes the expected duration so the executor doesn't think the run is hung.

6. **Worktree-venv routing** (recurring from PR #36 #1). Backtest invocation needs the worktree's `src/projections/models/ensemble.py` to be the version imported, not the main repo's. **Mitigation:** the plan task explicitly invokes via `PYTHONPATH=<worktree>/src ../../../.venv/Scripts/python.exe ...` or installs editably into an isolated venv. Either works; plan picks. Document the absolute-path workaround for Windows from PR #36's deviations.

---

## 6. Testing

`tests/test_models/test_ensemble_decomposed.py` (or extend `tests/test_models/test_ensemble.py` — plan picks):

1. **Factory wiring** — `wr_ensemble_decomposed()` returns an `EnsembleModel`; `_config.child_a_factory()` returns a `DecomposedBaselineModel`; `_config.child_b_factory()` returns an `LightGBMNbModel`; `position == Position.WR`; `target_stats == _WR_TARGET_STATS`.
2. **`code_hash` divergence** — fit `wr_ensemble()` and `wr_ensemble_decomposed()` on the same synthetic 3-year WR fixture; assert their `code_hash` strings differ; assert their `model_id` strings differ; assert their `data/ensemble_weights/{sanitized_id}.json` paths differ (using a tmp `weights_dir` per `_EnsembleConfig`).
3. **`predict_distribution` round-trip** — fit on synthetic fixture; predict on a held-out subset; validate against `ProjectionWeeklySchema`; unpack a row's `params` blob; for `RECEPTIONS`, assert the per-stat entry is a `MixtureDistribution` whose `component_a` is a `QuantileDistribution` and whose `component_b` is a `ParametricNegativeBinomial`; for non-decomposed WR stats (e.g., `RECEIVING_YARDS`), assert `component_a` is the parametric type from `_WR_DIST_FAMILIES`.
4. **Mixture tail-quantile sanity** — construct a `MixtureDistribution(QuantileDistribution(...), ParametricNegativeBinomial(...), weight=0.5)` directly (no fit, no ensemble); assert `.quantile(0.10)`, `.quantile(0.50)`, `.quantile(0.90)` all return finite values; assert `.quantile(0.99)` returns finite (this is the tail-bracket risk from §5 #2). If this fails, the §1.1 plan task scopes a fix to `mixture._bracket_for_components` before moving on.
5. **`_fit_weight_for_stat` on QuantileDistribution components** — construct a synthetic calibration year where `components_a` is a list of `QuantileDistribution` instances and `components_b` is a list of `ParametricNegativeBinomial`; assert the returned weight is in `(0.001, 0.999)` and finite; assert no `RuntimeWarning` was raised by the grid-search fallback.
6. **Registry + dispatch** — `POSITION_DISPATCH[Position.WR].factories["ensemble-decomposed"]()` returns an unfitted `EnsembleModel`; `_PositionDispatch[Position.WR].default_model_class == "ensemble"` (not flipped pre-gate); `"ensemble-decomposed"` is in `__all__`.
7. **CLI smoke** — `scripts/backtest.py --model ensemble-decomposed --positions wr ...` invocation parses successfully (mocked / dry-run if a full smoke is heavy); cross-position `--positions qb` raises (mirroring PR #36's `decomposed-baseline` WR-only restriction).
8. **Model-protocol conformance** — the existing position-dispatch test pattern in `tests/test_models/test_position_dispatch.py` already iterates `_WR_FACTORIES`; ensure `"ensemble-decomposed"` is covered by the parametrized test (PR #36 added this for `decomposed-baseline`; the existing test should auto-pick up the new key if it iterates `factories.items()`).

Real-data dual-run gate evidence captured in the PR per CLAUDE.md "Forced verification" rule.

---

## 7. Reports

`reports/wr_ensemble_decomposed_summary.md`:

- Both adoption-gate cells: verdict + 95% CI on RMSE Δ + Spearman Δ + n_paired + per-year breakdowns.
- Probe-vs-gate magnitude flag check: was the binding cell's RMSE Δ magnitude under 0.005 fpts? If yes, document the marginal-zone caveat (informational only — does not change the verdict).
- The §1.3.5 outcome narrative and any routing-flip executed.
- Deferred-follow-up disposition per §1.3.5 branch.
- Plan-vs-execution deviations (if any).

`reports/adoption_gate_wr_ensemble_decomposed_vs_ensemble.{md,csv}` — binding cell (produced by `scripts/adoption_gate.py`).

`reports/adoption_gate_wr_ensemble_decomposed_vs_decomposed_baseline.{md,csv}` — informational cell (produced by `scripts/adoption_gate.py`).

---

## 8. Estimated scope

~5 plan tasks, single session, one real-data backtest run.

| Task | Surface | Files touched |
|---|---|---|
| 1. Factory + factory-wiring unit tests | Factory + registry | `src/projections/models/ensemble.py`, `src/projections/models/__init__.py`, `tests/test_models/test_ensemble_decomposed.py` (new) |
| 2. Synthetic-data fit/predict tests + mixture-tail tests (with optional `mixture.py` fix if §5 #2 fires) | Predict path + codec exercise | extend `tests/test_models/test_ensemble_decomposed.py`, possibly `src/projections/distributions/mixture.py` if a numerical issue surfaces |
| 3. CLI + harness wiring + registry tests | CLI + harness | `scripts/backtest.py`, `tests/test_scripts/test_backtest_cli.py`, `tests/test_models/test_position_dispatch.py` (parametrize coverage) |
| 4. Real-data dual-run backtest + adoption gate (binding + informational) | Reports + artifacts | `reports/adoption_gate_wr_ensemble_decomposed_vs_ensemble.{md,csv}`, `reports/adoption_gate_wr_ensemble_decomposed_vs_decomposed_baseline.{md,csv}`, possibly `data/backtest/run_<ts>/` artifacts (manual, not committed wholesale) |
| 5. Per-§1.3.5 outcome execution + writeup + PM/TODO update | Decision log + conditional source edits | `reports/wr_ensemble_decomposed_summary.md`, `project_management.md`, `TODO.md`, conditionally `src/projections/models/__init__.py` (`default_model_class` flip on ADOPT) + `tests/backtest/model_metrics.json` snapshot update on ADOPT, OR full revert on REGRESSION |

End-to-end wall-clock: dual-run backtest is ~25–35 minutes per PR #36's deviations. Source edits + tests are small. Should fit in one focused session.

---

## 9. Implementation plan handoff

After this spec is approved and committed on `feat/wr-ensemble-decomposed-child`, the next step is the writing-plans skill to produce `docs/superpowers/plans/2026-05-15-wr-ensemble-decomposed-child.md` decomposing the 5 tasks above into per-task implementation steps with explicit phase boundaries and per-task verification commands.
