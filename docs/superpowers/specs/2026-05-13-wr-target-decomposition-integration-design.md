# WR Target Decomposition Integration — Design

**Status:** approved (brainstorming, 2026-05-13). Ready for implementation plan.
**Date:** 2026-05-13
**Author:** alden + claude
**Sub-project of:** FantasyFootball / Projections Core
**Builds on:**
- WR Target Decomposition Probe (PR #32, merged 2026-05-10) — shipped `src/projections/backtest/target_decomposition_probe.py` + `scripts/probe_target_decomposition.py` and ran the walk-forward Ridge-vs-Ridge probe on 2021–2024 WR rows. **Verdict: SIGNAL (marginal) on the receptions cell only.** Δ-RMSE −0.0042 fpts, CI [−0.0079, −0.0004], n_paired = 8460. `receiving_yards` and `receiving_tds` returned NULL. Factor orthogonality clean (|ρ| < 0.05 across all 12 year × stat cells). Probe spec at `docs/superpowers/specs/2026-05-10-target-decomposition-probe-design.md` §7 names the four follow-ups this spec scopes.
- Plan 3d (TODO #14, closed) — `DistributionFamily.SAMPLED_SUMMARY` exists; `pack_per_stat_params` codec supports `QUANTILE` family, which is the persistence vehicle for per-row composed-sample summaries.

**Branch:** `feat/wr-target-decomposition` cut from `main` at `0b52a9d` (post-nflreadpy migration).

---

## 1. Overview

PR #32's probe greenlit a per-position integration via spec branch §4 "≥ 1 SIGNAL, no REGRESSION." The probe also flagged a magnitude caveat: receptions Δ-RMSE × ESPN PPR reception coefficient = −0.0042 fpts composite-fpts contribution — below the ~0.005 fpts marginal-zone threshold from PR #31's retrospective rule. Coverage was strictly above 0.95 across all eval years, so the rule does not strictly apply, but the magnitude alone falls in the marginal zone. The adoption gate must weight CI strength against absolute magnitude.

This spec builds a new model class `DecomposedBaselineModel` (peer to `BaselineModel`) that supports per-stat decomposition opt-in via a constructor `decomposed_stats` mapping. **v1 ships a single WR factory configured for `receptions`-only decomposition** (`volume_stat=TARGETS`, `efficiency=catch_rate` with clip `[0, 1]`); `receiving_yards` and `receiving_tds` fall through to direct-RidgeCV identical to current `BaselineModel`. The architecture supports cross-stat coherent sampling (multiple decomposed stats sharing a volume factor get a single per-row volume draw flowing into each composed sample array), but v1 does not exercise it — the integration plan names it as v2 follow-up territory.

WR is currently routed to `EnsembleModel` (`_PositionDispatch[Position.WR].default_model_class = "ensemble"`, Plan 8 verdict, 2026-04-29). The binding adoption-gate cell per probe spec §7.6 is `(DecomposedBaselineModel, WR)` vs `(EnsembleModel, WR)`. An additional informational cell `(DecomposedBaselineModel, WR)` vs `(BaselineModel, WR)` is run because it reproduces the probe's Ridge-vs-Ridge comparison at the composite-fpts level — without it, a binding-cell DO_NOT_ADOPT cannot distinguish "decomposition does not carry to composite-fpts" from "decomposition helps baseline but ensemble's lgb-nb contribution outpaces the lift."

The shipping decision is binary and bound to the **`(DecomposedBaselineModel, WR)` vs `(EnsembleModel, WR)`** verdict. **The informational `(DecomposedBaselineModel, WR)` vs `(BaselineModel, WR)` cell is not gating** — it is captured in the summary report to interpret the binding cell. Other model classes' verdicts are not produced (this PR builds one new class; cross-class re-evals against unchanged classes have no new comparator).

This is the **first model-architecture integration in the project** (PR #21 / #26 / #27 / #29 were feature-set extensions to schemas + builders; PR #32 was the first model-architecture probe; this PR is the first model-architecture production integration). The same §1.3.5 contingency-matrix shape applies as PR #29 / PR #31.

### 1.1 Goals (in scope)

- New module **`src/projections/models/decomposed_baseline.py`** containing:
  - `DecompositionSpec` frozen dataclass: `volume_stat: Stat`, `efficiency_label: str`, `efficiency_clip_hi: float`. Mirrors the probe's `_StatDecomp` shape.
  - `DecomposedBaselineModel` — subclass of `BaselineModel` with one new dataclass field `decomposed_stats: Mapping[Stat, DecompositionSpec] = field(default_factory=dict)`, plus the persisted-state fields needed for the factor sub-models (see §3.1.2 for the exact list).
  - `wr_decomposed_baseline()` factory configured for receptions-only decomposition.

- New module **`src/projections/distributions/sampled.py`** containing:
  - `FrozenSampledDistribution(samples: NDArray[np.float64])` — Distribution-Protocol-conforming dataclass whose `.sample(n, rng)` returns `self.samples` directly when `n == len(self.samples)`, else falls back to `rng.choice(self.samples, size=n, replace=True)`. **The `n == len` path is what preserves within-row cross-stat correlation when `score_distribution` calls `.sample(n_samples)` on multiple composed distributions that share an underlying volume draw.** Mean / std / quantile / cdf are computed directly off `self.samples` (no resampling). The class is intentionally separate from `SampledDistribution` (which retains its documented `rng.choice` re-sampling semantics for any existing caller that depends on it).

- **Codec persistence path.** Decomposed stats persist per row as `QuantileDistribution` summaries (existing `QUANTILE` codec branch — no codec edits required). At `build_stat_distributions` time, the per-row composed sample array (length 10,000) is summarized into 19 quantiles at q ∈ {0.05, 0.10, …, 0.95} and wrapped in `QuantileDistribution` for persistence, then re-expanded to a `FrozenSampledDistribution` via an internal `_sampled_from_quantiles` helper before being handed to `score_distribution`. The `FrozenSampledDistribution` instance is what carries the within-row correlation through the per-row `score_distribution` call; the persisted `QuantileDistribution` is the on-disk summary.

- **Update `src/projections/models/__init__.py`** to register `wr_decomposed_baseline` in `_WR_FACTORIES` under the key `"decomposed-baseline"`. Do **not** change `_PositionDispatch[Position.WR].default_model_class` in this PR's initial state — the flip is conditional on the adoption-gate verdict (§1.3.5). Exports updated in `__all__`.

- **Update `src/projections/models/baseline.py`** with no production behavior changes: only the helpers needed for `DecomposedBaselineModel` to reuse (e.g., promote `_x_frame_with_bool_coercion` to a module-level helper if needed, or rely on the subclass-method-resolution-order to call it on `self`). Prefer the latter; subclass inherits the bound method as-is.

- **Tests under `tests/test_models/test_decomposed_baseline.py`:**
  - Unit — `DecompositionSpec` fields + immutability.
  - Unit — fit on synthetic 4-year WR-shaped frame; verify per-stat ridges populated: direct ridges for non-decomposed stats; volume + efficiency ridges for decomposed stats; volume residual std persisted; efficiency residual std persisted.
  - Unit — `build_stat_distributions` produces per-row dicts with `ParametricNormal`/`ParametricGamma`/`ParametricNegativeBinomial` for non-decomposed stats (matching `BaselineModel` family map) AND `QuantileDistribution` for decomposed stats.
  - Unit — within-row cross-stat coherence (synthetic config decomposing **both** receptions and receiving_yards on shared `Stat.TARGETS`): verify the two decomposed stats' `FrozenSampledDistribution` arrays are Pearson-correlated with ρ > 0.5 within a row (they share a volume draw; this is the architectural guarantee). v1 production config is receptions-only, but this test exercises the coherent-sampling code path that v2 will exercise in production.
  - Unit — determinism: same row seed → same composed samples; different row seed → different composed samples; correlation property preserved across seeds.
  - Unit — `FrozenSampledDistribution.sample(n)` returns `self.samples` exactly when `n == len(self.samples)`; resamples when `n != len`.
  - Unit — `predict_distribution` returns a `ProjectionWeeklySchema`-validating frame; `family` column is `SAMPLED_SUMMARY` (unchanged from `BaselineModel`); `params` blob round-trips through `unpack_per_stat_params` to per-stat dicts where decomposed stats appear as `QuantileDistribution`.
  - Unit — `model_id` is `"decomposed-baseline:<position>:<8-char-hash>:<train-start>-<train-end>"`.
  - Integration — factory `wr_decomposed_baseline()` constructs an unfitted model with the correct receptions-only `decomposed_stats` config; `fit` + `predict_distribution` round-trip on a synthetic WR fixture passes `ProjectionWeeklySchema.validate`.
  - Integration — `Model`-protocol conformance via the existing `tests/test_models/test_protocol.py` pattern (if absent, add one matching the BaselineModel conformance test).

- **Run the adoption gate. Two cells (one binding, one informational).** Both produced from a single dual-run backtest (`scripts/backtest_dual.py` per PR #29's pattern):
  - **Binding:** `(DecomposedBaselineModel, WR)` vs `(EnsembleModel, WR)` — gates production routing flip.
  - **Informational:** `(DecomposedBaselineModel, WR)` vs `(BaselineModel, WR)` — apples-to-apples test of the decomposition recipe at the composite-fpts level.

- **Reports:**
  - `reports/wr_target_decomposition_summary.md` — probe-vs-gate calibration table; both cells' verdicts + 95% CI; §1.3.5 outcome narrative; coverage statistics on receptions decomposition (`targets > 0` rate per eval year); recommendation on production routing per §1.3.5; deferred-follow-up disposition.
  - `reports/adoption_gate_wr_decomposed_baseline_vs_ensemble.{md,csv}` — binding cell verdict.
  - `reports/adoption_gate_wr_decomposed_baseline_vs_baseline.{md,csv}` — informational cell verdict.

- **Per-§1.3.5 outcome execution.** Conditional on verdict (see §1.3.5 below).

- **PM/TODO update.** `project_management.md` decision-log entry at top with verdicts and §1.3.5 outcome; `TODO.md` entry under #23 (target decomposition direction) updated with the integration result.

### 1.2 Non-goals (deferred)

- **No decomposition for `receiving_yards` or `receiving_tds` in v1.** Both returned NULL in the probe; including them in the v1 production config adds variance without expected mean lift, and risks the binding-cell verdict by inflating the composite-fpts RMSE on stats where the architectural prediction is "no signal." The opt-in architecture supports trivially flipping them on in a future PR, contingent on a refined probe (e.g., factor-appropriate sub-models per spec §7.3 of the probe spec).
- **No factor-appropriate sub-model classes** (logistic for `catch_rate`, Gamma log-link for `yards_per_target`, Poisson for `targets`). Named follow-up per probe spec §7.4 — separate probe + integration cycle, gated on this integration's adoption-gate verdict. v1 uses RidgeCV for every factor sub-model (matches the probe's deliberate model-class choice).
- **No other positions.** RB / QB / TE each get their own decomposition probe + integration cycle. WR-first because the probe was WR-only. Per the probe spec §1.4: RB's natural decomposition is dual-mode (`carries × yards_per_carry` for rushing + `targets × yards_per_target` for receiving); QB is the most complex (full passing chain plus rushing/sack/scramble adjustments); TE mirrors WR receiving. None scoped here.
- **No new `ProductDistribution` Distribution-Protocol class.** v1 persists decomposed stats as `QuantileDistribution` summaries and rehydrates via the existing `QUANTILE` codec branch. A future codec extension that persists factor parameters directly (volume mu/sigma + efficiency mu/sigma per row, recomposing on demand) is the right shape if downstream features ever need the factor structure (e.g., DFS within-game correlation modeling, named in TODO #1 option D). Not in scope for v1.
- **No `EnsembleModel` integration of `DecomposedBaselineModel` as a child.** EnsembleModel's child factories are constants (`BaselineModel` + `LightGBMNbModel`) — swapping `BaselineModel → DecomposedBaselineModel` inside the ensemble is a separate cross-class question (does ensemble-with-decomposed-baseline beat ensemble-with-baseline?) deferred until v1's verdict comes in. The decision tree:
  - If v1 ADOPTs and WR routes to `decomposed-baseline`: the ensemble-with-decomposed-baseline question becomes "can we win further by re-ensembling?" — open follow-up.
  - If v1 is MARGINAL / DO_NOT_ADOPT vs ensemble but the informational cell is ADOPT vs baseline: the natural follow-up is to swap `BaselineModel → DecomposedBaselineModel` inside `EnsembleModel`'s child A factory and re-fit weights. **This becomes the recommended next plan** — the decomposed-baseline-as-ensemble-child question is then the active follow-up. Documented in the summary report.
- **No `--coverage-threshold` relaxation.** The probe's `targets > 0` coverage was 0.981–0.988 (eval) / 0.993–0.994 (train) — well above the default 0.95. The integration gate uses the same default. No relaxation is invoked; any documented relaxation would be a probe-vs-gate discontinuity and a documented caveat.
- **No regression test on `_<POS>_FEATURE_COLUMNS` parity.** That covers feature-additions (the recurring class fixed in PR #29 / #31's mid-flight catches and the post-PR-29 structural test at `tests/test_models/test_baseline_feature_columns_match_schema.py`). This PR adds no features. The feature-columns parity test continues to gate any future feature-additions.

### 1.3 Adoption gate

Adoption decisions are per position. For Position WR, the adoption gate compares `decomposed-baseline` against the incumbent `_PositionDispatch[Position.WR].default_model_class` which is `ensemble`.

**Inputs.** Per-row predictions from both classes for the same `(gsis_id, season, week)` rows across all held-out years (2021–2024), pulled from a dual-run `scripts/backtest_dual.py` invocation per PR #29's pattern. After pairing, WR contributes ~8,460 paired rows (matching the probe's n_paired).

**Statistical machinery.** Paired bootstrap with `n_bootstrap=1000`, deterministic seed `42`. Resampling unit is the paired player-week — both candidate and incumbent are scored on the same draw.

**Per-position metrics.**
- **RMSE delta** (`candidate − incumbent`): pooled across all held-out years. Negative = candidate wins.
- **Spearman delta**: per-year Spearman computed within each held-out year, then averaged unweighted across years.

Per-cell breakdowns (one row per held-out year) are emitted for inspection but do **not** gate adoption.

**Verdict rule.**
```
PASS_RMSE      := rmse_delta.hi_95     <  0.0
PASS_SPEARMAN  := spearman_delta.lo_95 > -0.02

if  PASS_RMSE and  PASS_SPEARMAN:  ADOPT
if  PASS_RMSE and !PASS_SPEARMAN:  MARGINAL — investigate before adopting
if !PASS_RMSE and  PASS_SPEARMAN:  DO_NOT_ADOPT
if !PASS_RMSE and !PASS_SPEARMAN:  DO_NOT_ADOPT
```

**What this gate does not check.**
- No per-cell pass/fail — per-year deltas are informational; only the position-pooled CI gates.
- No Spearman-improvement requirement — only the catastrophic-regression floor (`-0.02`).
- No calibration check.
- No probe-magnitude floor — but see §1.3.5 for how the marginal-zone magnitude flag is handled.

**Adoption is manual.** `scripts/adoption_gate.py` emits a report; a human reads the verdicts and edits `_PositionDispatch[Position.WR].default_model_class` if the binding cell verdict is `ADOPT`. The CLI never writes to source.

**Tooling.**
```
python -m scripts.adoption_gate \
  --baseline-run data/backtest/run_<ts_ensemble> \
  --candidate-run data/backtest/run_<ts_decomposed> \
  --candidate decomposed-baseline \
  --csv-out reports/adoption_gate_wr_decomposed_baseline_vs_ensemble.csv
```

A second invocation against the `--baseline-run` for `baseline` produces the informational cell's report.

### 1.3.5 Per-position §1.3.5 outcome matrix

Conditional on the **binding** `(DecomposedBaselineModel, WR)` vs `(EnsembleModel, WR)` verdict. The informational cell informs interpretation but does not change the branch.

- **ADOPT (binding)** — Flip `_PositionDispatch[Position.WR].default_model_class` from `"ensemble"` to `"decomposed-baseline"` in `src/projections/models/__init__.py`. Keep `wr_ensemble` factory available. Existing `data/ensemble_weights/ensemble_wr_*.json` artifacts remain in-tree (no GC). PM logs the flip + magnitude.

- **MARGINAL (binding RMSE PASS, Spearman fail)** — Treat as DO_NOT_ADOPT for production routing per template's "investigate before adopting" guidance, but ship the `DecomposedBaselineModel` class + `wr_decomposed_baseline` factory as available infrastructure. PM logs the verdict + the Spearman-floor-violation magnitude.

- **DO_NOT_ADOPT (binding)** — Keep WR on `ensemble` in production. Ship the `DecomposedBaselineModel` class + `wr_decomposed_baseline` factory as available infrastructure (registered in `_WR_FACTORIES` but not the default). **The informational cell determines the recommended follow-up:**
  - **Informational ADOPT** (vs `baseline`) — Strong signal that decomposition helps baseline-class WR but cannot outpace ensemble. PM logs the recommended next plan: swap `BaselineModel → DecomposedBaselineModel` inside `EnsembleModel`'s child-A factory, re-fit ensemble weights, run a new dual-run gate on `(EnsembleModel-with-decomposed-baseline, WR)` vs current `(EnsembleModel, WR)`. This is the natural next slot — it isolates the decomposition contribution under the production-ensemble structure.
  - **Informational MARGINAL / DO_NOT_ADOPT** (vs `baseline`) — Decomposition signal does not carry to composite-fpts at all. Close target-decomposition at the WR receiving cell × 2-factor unit. PM logs the closure. Refined-unit candidates (3-factor decomposition, factor-appropriate sub-model classes) remain open under TODO #23 but require independent mechanism evidence before re-probing per PR #31's retrospective rule.
  - **Informational REGRESSION** (vs `baseline`) — Decomposition is actively worse than direct ridges at the composite-fpts level despite the probe's per-stat receptions signal. **Full revert** of `decomposed_baseline.py` + factory registration + `_WR_FACTORIES` entry. Spec + plan + reports stay as historical record. PM logs the closure of the WR decomposition direction in stronger terms; future revisits require either factor-appropriate sub-models or a refined volume axis.

- **REGRESSION (binding RMSE CI strictly > 0)** — `DecomposedBaselineModel` is actively worse than `EnsembleModel`. Full revert per PR #31 precedent: remove production code paths (the factory + the `_WR_FACTORIES` entry); keep `decomposed_baseline.py` module as in-tree historical record (so anyone reading the PR can see what was tried). PM logs the closure + the magnitude.

**Probe-vs-gate magnitude flag.** Per PR #31's retrospective rule, the probe's binding-cell composite-fpts implied magnitude was −0.0042 fpts — below the ~0.005 fpts marginal-zone threshold. Coverage was strictly above 0.95, so the rule does not strictly fire, but the integration-gate magnitudes should be reported alongside CI sign for clarity. If the binding cell ADOPTs at a composite-fpts magnitude < 0.005 fpts on a borderline CI, the summary report calls out the marginal-zone flag explicitly so the routing-flip decision is made with eyes open.

### 1.4 Success criteria

The spec is complete iff all of:

1. **Module + class + factory + tests land cleanly.** `pytest -v` (full suite), `mypy src tests scripts` (zero violations), `ruff check src tests scripts` (zero violations), `ruff format --check src tests scripts` (no drift).
2. **`DecomposedBaselineModel` round-trips through the standard backtest pipeline** — `scripts/backtest.py --model decomposed-baseline --position wr` (or equivalent invocation supported by `scripts/backtest_dual.py`) produces a `results.parquet` validating against the run's expected schema.
3. **The dual-run backtest + adoption gate runs successfully on WR** for the binding cell `(decomposed-baseline vs ensemble)` and the informational cell `(decomposed-baseline vs baseline)`, both across the standard `2021-2024` eval window with `--coverage-threshold 0.95` (no relaxation; probe coverage was 0.981+).
4. **The summary report (`reports/wr_target_decomposition_summary.md`) records all of:**
   - The probe's predicted per-stat Δ-RMSE on receptions (`-0.0042 fpts` × ESPN PPR coefficient = `-0.0042 fpts` composite-fpts implied magnitude on the receptions contribution; aggregate-stat composite-fpts is a different quantity, hence the probe-vs-gate calibration table).
   - The gate's measured composite-fpts Δ-RMSE for both cells with 95% CI + Spearman Δ.
   - Coverage statistics on `targets > 0` rate per eval year for WR (per-eval-year + pooled).
   - The §1.3.5 outcome narrative and any routing-flip executed.
   - The deferred-follow-up recommendation (per §1.3.5 branch, e.g., ensemble-child swap; or factor-appropriate sub-models).
5. **The §1.3.5 outcome is executed** before the PR is merged — routing flip on ADOPT, no-flip on MARGINAL / DO_NOT_ADOPT (with infra shipped), or full revert on REGRESSION (or informational REGRESSION on DO_NOT_ADOPT binding).

If criterion 1 fails, fix and rerun. If criterion 2 fails, the model is wrong — fix before running the gate. Criterion 3 is mechanical. Criterion 5 is the binding decision.

---

## 2. Inputs

### 2.1 Source data — already in cache

All inputs already exist in the feature cache and weekly stats (same as the probe):

| Column | Type | Source | Use |
|---|---|---|---|
| WR features | `WrFeaturesSchema` | `data/features/wr/season=YYYY/week=WW/part.parquet` (populated by `scripts/refresh_features.py`) | X matrix for all ridges (1 shared volume + 1 efficiency for receptions + 5 direct comparators for non-decomposed stats). |
| `targets` | `Series[int]` | `WeeklyStatsSchema.targets` | y for the volume sub-model; filter predicate (`> 0`) for the efficiency sub-model. |
| `receptions` | `Series[int]` | `WeeklyStatsSchema.receptions` | y for `catch_rate` (= receptions / targets) — the lone decomposed stat in v1. |
| Other WR target stats | per `WeeklyStatsSchema` | same partition | y for direct ridges (`receiving_yards`, `receiving_tds`, `rushing_yards`, `rushing_tds`, `fumbles_lost`) — identical to BaselineModel's path for those stats. |

**No new ingest, no schema changes.** The receptions-only v1 configuration consumes only existing data; no feature-cache refresh is required.

### 2.2 No new caller-script changes

`scripts/backtest_dual.py`, `scripts/adoption_gate.py`, and `scripts/refresh_features.py` are unchanged. The dual-run backtest invokes via `--model decomposed-baseline` (mapped through `_WR_FACTORIES["decomposed-baseline"]` once registered). PR #29's gate-orchestration patterns apply directly: split each run's `results.parquet` by `model_class` into per-class subdirs if `scripts/adoption_gate.py` dual-run mode requires single-model-class run dirs (the existing PR #29 workaround).

---

## 3. Architecture

### 3.1 `DecomposedBaselineModel`

#### 3.1.1 Class shape

Subclass of `BaselineModel`. Reuses:
- `_x_frame_with_bool_coercion` (inherited method on `self`)
- `model_id`-derivation pattern (overridden to use `"decomposed-baseline"` prefix)
- `predict_distribution`'s per-row loop (inherited; the override is in `build_stat_distributions`)
- `save` / `load` (inherited; `load`'s `isinstance` check rejects mismatched class — same defense as `BaselineModel.load`, no override needed)

Overrides:
- `fit(features, weekly_stats)` — extended to fit the volume + efficiency sub-models for any stat in `decomposed_stats`, in addition to the direct ridges for non-decomposed stats. Code structure mirrors `target_decomposition_probe.py`'s `walk_forward_residuals` per-iteration body.
- `build_stat_distributions(features)` — produces `FrozenSampledDistribution` for decomposed stats (per-row sample arrays carrying within-row cross-stat correlation when multiple stats share a volume factor) and parametric distributions for non-decomposed stats (unchanged path). Conversion to `QuantileDistribution` happens at codec-persistence time inside `predict_distribution`, not here — see §3.1.5 / §3.1.6 for the layering rationale.
- `model_id` (property) — `f"decomposed-baseline:{position}:{code_hash}:{train_start}-{train_end}"`.

#### 3.1.2 New dataclass fields

```
decomposed_stats: Mapping[Stat, DecompositionSpec] = field(default_factory=dict)
volume_ridges: dict[Stat, RidgeCV] = field(default_factory=dict)
    # Keyed by volume_stat (e.g., Stat.TARGETS). One ridge per unique volume_stat
    # across all entries in decomposed_stats. v1: one entry, Stat.TARGETS.
efficiency_ridges: dict[Stat, RidgeCV] = field(default_factory=dict)
    # Keyed by decomposed composite stat (e.g., Stat.RECEPTIONS). One ridge per
    # decomposed stat.
volume_variance: dict[Stat, float] = field(default_factory=dict)
    # Per-volume-stat Normal residual std for sampling. v1: {Stat.TARGETS: float}.
efficiency_variance: dict[Stat, float] = field(default_factory=dict)
    # Per-decomposed-stat Normal residual std on the ratio for sampling.
    # v1: {Stat.RECEPTIONS: float} (residual std of catch_rate).
```

All four new state dicts are `field(default_factory=dict)` so an unfitted instance is constructable. `fit()` populates them; `build_stat_distributions` raises `RuntimeError` if a decomposed stat lacks a volume_ridge / efficiency_ridge entry post-fit.

#### 3.1.3 `fit()` algorithm

Reuses `BaselineModel.fit`'s join + NaN-drop + median-imputation logic exactly. After the existing per-stat direct ridge fit loop, an additional pass fits the decomposition sub-models:

```
For each unique volume_stat across decomposed_stats:
  y_vol = joined[volume_stat.value].to_numpy(dtype=float64)
  volume_ridges[volume_stat] = RidgeCV(alphas).fit(x, y_vol)
  mu_vol = volume_ridges[volume_stat].predict(x)
  volume_variance[volume_stat] = max(std(y_vol - mu_vol), 1e-6)

For each (composite_stat, decomp_spec) in decomposed_stats.items():
  mask = joined[decomp_spec.volume_stat.value] > 0
  x_pos = x[mask]
  y_pos = joined.loc[mask, composite_stat.value].to_numpy(dtype=float64)
  vol_pos = joined.loc[mask, decomp_spec.volume_stat.value].to_numpy(dtype=float64)
  ratio = y_pos / vol_pos   # safe — mask guarantees vol_pos > 0
  efficiency_ridges[composite_stat] = RidgeCV(alphas).fit(x_pos, ratio)
  mu_eff = efficiency_ridges[composite_stat].predict(x_pos)
  efficiency_variance[composite_stat] = max(std(ratio - mu_eff), 1e-6)
```

For composite stats that appear in `decomposed_stats`, the direct ridge in `self.ridges` is **also fit** (matching the probe's structure where both arms are computed). This is informational; `build_stat_distributions` uses only the decomposed path for those stats. Persisting both lets a diagnostic CLI inspect the direct-vs-decomposed gap if needed and lets a future config flip a stat from decomposed → direct without re-training.

(Self-justification check: is this wasteful? It costs one extra RidgeCV fit per decomposed stat. For v1 with one decomposed stat, that's one extra ridge fit at train time, ~milliseconds. Cheap insurance; keep it.)

Train-time NaN policy, median imputation, and code-hash computation: unchanged from `BaselineModel.fit`.

#### 3.1.4 `build_stat_distributions()` algorithm

```
non_decomposed_stats = [s for s in self.target_stats if s not in self.decomposed_stats]

# Existing path for non-decomposed stats — produces parametric distributions per row.
parametric_rows = super().build_stat_distributions(features, only_stats=non_decomposed_stats)
# (Implies a minor refactor of BaselineModel.build_stat_distributions to accept an
#  `only_stats` filter. If reluctant to touch BaselineModel signature, the override can
#  duplicate the per-row loop locally for the non-decomposed stats — slight code-dup but
#  no superclass-signature change. Plan task picks one.)

# Decomposed path — per-row coherent sampling, summarize to QuantileDistribution.
x_frame = self._x_frame_with_bool_coercion(features).fillna(self.feature_means)
x = x_frame.to_numpy(dtype=float64)

per_volume_mu: dict[Stat, NDArray] = {
    vs: self.volume_ridges[vs].predict(x).astype(float64)
    for vs in {s.volume_stat for s in self.decomposed_stats.values()}
}
per_decomposed_mu_eff: dict[Stat, NDArray] = {
    cs: self.efficiency_ridges[cs].predict(x).astype(float64)
    for cs in self.decomposed_stats
}

# Seed derivation per row: derive_row_seed(...) + sub-seed offset by integer
# factor identifier so volume + efficiency draws within a row are independent
# but reproducible. Sub-seed: 0 for shared-volume-stat per row, 1..K for the
# K-th decomposed stat's efficiency factor. See §3.2 for the exact derivation.

For each row i in features.iterrows():
    row_seed = derive_row_seed(gsis_id=..., season=..., week=..., ruleset_name=...)
    # Per-row volume draws (one per unique volume_stat): shared across all
    # decomposed stats with the same volume_stat.
    vol_samples: dict[Stat, NDArray] = {}
    for vs, mu_vol_arr in per_volume_mu.items():
        sigma_vol = self.volume_variance[vs]
        rng = np.random.default_rng(row_seed)
        raw = rng.normal(loc=mu_vol_arr[i], scale=sigma_vol, size=N_SAMPLES)
        vol_samples[vs] = np.maximum(raw, 0.0)

    decomposed_dists_for_row: dict[Stat, Distribution] = {}
    for j, (composite_stat, decomp_spec) in enumerate(self.decomposed_stats.items(), start=1):
        sigma_eff = self.efficiency_variance[composite_stat]
        rng_eff = np.random.default_rng(row_seed + j)
        raw_eff = rng_eff.normal(
            loc=per_decomposed_mu_eff[composite_stat][i],
            scale=sigma_eff,
            size=N_SAMPLES,
        )
        eff_samples = np.clip(raw_eff, 0.0, decomp_spec.efficiency_clip_hi)
        composed = vol_samples[decomp_spec.volume_stat] * eff_samples
        # IMPORTANT: emit a live FrozenSampledDistribution carrying the per-row
        # sample array. Two decomposed stats sharing a volume_stat receive the
        # SAME vol_samples[volume_stat] array → element-wise correlation is
        # baked into the composed sample arrays. score_distribution downstream
        # calls .sample(n_samples=10_000) on each; the FrozenSampledDistribution
        # n == len branch returns the underlying array verbatim, preserving
        # cross-stat correlation. The QuantileDistribution conversion happens
        # later, at codec-persistence time (see §3.1.6).
        decomposed_dists_for_row[composite_stat] = FrozenSampledDistribution(samples=composed)

    # Merge parametric + decomposed dists for this row.
    parametric_rows[i].update(decomposed_dists_for_row)

return parametric_rows
```

**N_SAMPLES.** Match `BaselineModel.predict_distribution`'s `n_samples=10_000` for consistency. Per-row Monte Carlo cost: one row × 2 normal draws × 10K = small.

**`_PERSISTED_QUANTILES`.** Used at persistence time (§3.1.6), NOT here. 19 quantiles at q ∈ {0.05, 0.10, …, 0.95}. Per-row persistence cost: 19 floats per decomposed stat in the params blob — negligible vs the existing per-stat parametric encoding. Recomposable to high fidelity via `QuantileDistribution`'s linear-interpolation `.sample` (with the documented loss of cross-stat correlation on rehydration — §3.1.5 + §5 risk #4).

#### 3.1.5 Coherent-sampling correctness inside `score_distribution`

`predict_distribution` (inherited) calls `score_distribution(stat_dists, ruleset, n_samples=10_000, seed=...)`. `score_distribution` iterates per-stat distributions and calls `dist.sample(n_samples, rng=rng)`. To preserve within-row cross-stat correlation, each decomposed stat must return **the same per-row sample array** so that when `score_distribution` sums them with coefficients, the volume-driven correlation is intact.

The persisted-as-QuantileDistribution → in-memory-as-FrozenSampledDistribution conversion is the mechanism. At the boundary just before `score_distribution` is called (inside the inherited `predict_distribution` loop), an internal helper `_inflate_dists_for_scoring(per_row_dists, n_samples=10_000)` converts every `QuantileDistribution` entry into a `FrozenSampledDistribution(samples=q_dist.sample(n_samples, rng))` — where the `rng` is **the same per-row rng used for the score_distribution call**.

Concretely: per row, the rng is seeded with `derive_row_seed(...)` (existing). Inside `_inflate_dists_for_scoring`, the per-stat sample draws happen **after** the volume × efficiency composition (which is materialized in the QuantileDistribution's quantiles). Two decomposed stats sharing a volume factor produce two QuantileDistribution objects whose internal quantile arrays were derived from the same per-row volume samples. Inflating both back via `q_dist.sample(N)` from the **same rng-seed** does not reproduce the original cross-stat correlation — QuantileDistribution's `.sample` does inverse-CDF sampling against independently-drawn uniforms, which loses the cross-stat link.

**Implication for v1 (receptions-only):** the cross-stat coherent-sampling guarantee is only *architecturally true* when decomposed stats are emitted as live `FrozenSampledDistribution` instances inside `build_stat_distributions`, **NOT** when they're persisted as `QuantileDistribution` and rehydrated. The v1 production path is therefore:

1. `build_stat_distributions` produces `FrozenSampledDistribution` (not `QuantileDistribution`) for decomposed stats — these carry the actual cross-stat-correlated per-row sample arrays.
2. `predict_distribution`'s inherited loop calls `score_distribution(stat_dists, ruleset, n_samples=10_000, ...)`. `score_distribution` calls `.sample(10_000)` on each; the `FrozenSampledDistribution`'s `n == len` branch returns the underlying array verbatim → cross-stat correlation preserved.
3. **Persistence step (separate)** — before persisting to the params blob, `predict_distribution` converts each `FrozenSampledDistribution` → `QuantileDistribution` via `_quantile_summary_of(samples, quantiles=_PERSISTED_QUANTILES)`. The persisted blob is the quantile summary; the scoring step has already happened on the live in-memory `FrozenSampledDistribution`.

The persisted-form correlation loss is documented in §5 risk #4 and is acceptable for v1's use case (no post-hoc re-scoring from persisted params blobs).

#### 3.1.6 `predict_distribution` override (minimal)

The inherited `BaselineModel.predict_distribution` loop is:
```
stat_dists_per_row = self.build_stat_distributions(features)
for (_idx, feat_row), stat_dists in zip(...):
    seed = derive_row_seed(...)
    points = score_distribution(stat_dists, ruleset, n_samples=10_000, seed=seed)
    family_blob = pack_per_stat_params(stat_dists)
    ...
```

The override changes the `pack_per_stat_params` call: per-stat dists with `FrozenSampledDistribution` are first converted to `QuantileDistribution` (via `_quantile_summary_of`) for persistence, then packed:

```
persistable_dists = {
    stat: (
        _quantile_summary_of(dist.samples, quantiles=_PERSISTED_QUANTILES)
        if isinstance(dist, FrozenSampledDistribution)
        else dist
    )
    for stat, dist in stat_dists.items()
}
family_blob = pack_per_stat_params(persistable_dists)
```

This is a small, well-scoped override that doesn't touch the per-row loop's iteration shape. Implementation can either:
- (A) Override `predict_distribution` wholesale (copy-paste from BaselineModel, modify the pack step), OR
- (B) Extract a `_pack_per_row(stat_dists)` hook on BaselineModel that DecomposedBaselineModel overrides.

Plan task picks. **Recommendation: (B)** — single-method override is cleaner than copy-pasting a 50-line loop with a 3-line modification. The hook addition to BaselineModel is a small, defensible change that doesn't alter BaselineModel's behavior.

### 3.2 `FrozenSampledDistribution`

```python
@dataclass(slots=True, frozen=True)
class FrozenSampledDistribution:
    """Sampled distribution whose per-call sample(n) returns the underlying
    array verbatim when n == len(samples), enabling cross-stat coherent
    sampling at score_distribution time.

    Cf. SampledDistribution (in scoring/score_distribution.py), which always
    re-samples via rng.choice. The two coexist; consumers pick based on
    whether they need within-call sample-ordering preservation.
    """

    samples: NDArray[np.float64]

    def mean(self) -> float: return float(self.samples.mean())
    def std(self) -> float: return float(self.samples.std())
    def quantile(self, q: float) -> float:
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must be in (0, 1), got {q}")
        return float(np.quantile(self.samples, q))
    def cdf(self, x: float) -> float:
        return float((self.samples <= x).mean())
    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        if n == len(self.samples):
            return self.samples
        rng = rng if rng is not None else np.random.default_rng()
        return rng.choice(self.samples, size=n, replace=True)
```

Location: `src/projections/scoring/score_distribution.py` (next to `SampledDistribution`) or a new `src/projections/distributions/sampled.py` module promoted alongside the existing ones. The current location of `SampledDistribution` is in `scoring/` for layering reasons (it's the output of `score_distribution`); `FrozenSampledDistribution` is structurally a Distribution but is consumed by the model layer, not produced by the scoring layer. **Recommend a new module `src/projections/distributions/sampled.py`** that hosts both `SampledDistribution` (moved from `scoring/score_distribution.py`) and `FrozenSampledDistribution`. The move keeps `Distribution` implementations together. The scoring module re-imports `SampledDistribution` from the new location to preserve its public API.

If the move is non-trivial (test imports / public API impact), fall back to keeping `FrozenSampledDistribution` next to `SampledDistribution` in `scoring/score_distribution.py`. The plan task picks; the architecture is the same either way.

### 3.3 Factory + dispatch registration

```python
# src/projections/models/decomposed_baseline.py

_WR_RECEIVING_DECOMPOSITION: Final[Mapping[Stat, DecompositionSpec]] = {
    Stat.RECEPTIONS: DecompositionSpec(
        volume_stat=Stat.TARGETS,
        efficiency_label="catch_rate",
        efficiency_clip_hi=1.0,
    ),
}


def wr_decomposed_baseline() -> DecomposedBaselineModel:
    """Construct an unfitted WR decomposed-baseline model.

    v1 config: receptions decomposed via targets × catch_rate; all other
    WR target stats fall through to direct RidgeCV.
    """
    return DecomposedBaselineModel(
        position=Position.WR,
        target_stats=_WR_TARGET_STATS,
        feature_columns=_WR_FEATURE_COLUMNS,
        dist_families=_WR_DIST_FAMILIES,
        feature_schema=WrFeaturesSchema,
        code_hash_files=_default_code_hash_files("wr.py")
            + (_decomposed_baseline_module_path(),),
        decomposed_stats=_WR_RECEIVING_DECOMPOSITION,
    )
```

`_decomposed_baseline_module_path()` returns the path to `src/projections/models/decomposed_baseline.py`. The code-hash files tuple is extended so any edit to `decomposed_baseline.py` propagates to the model_id (matching the `_default_code_hash_files` pattern for BaselineModel which already includes `baseline.py`).

`src/projections/models/__init__.py`:
- Add `from projections.models.decomposed_baseline import DecomposedBaselineModel, wr_decomposed_baseline`.
- Add `"decomposed-baseline": wr_decomposed_baseline` to `_WR_FACTORIES`.
- Update `__all__` with `"DecomposedBaselineModel"` and `"wr_decomposed_baseline"`.
- **Do not** change `_PositionDispatch[Position.WR].default_model_class` in the initial PR commits. The flip (if ADOPT) is a separate commit captured in the §1.3.5 outcome execution.

### 3.4 Codec — no edits

The codec already supports `QUANTILE`. `pack_per_stat_params` packs `QuantileDistribution` instances via `_pack_single`'s existing branch. `unpack_per_stat_params` decodes back to `QuantileDistribution`. No changes to `src/projections/distributions/codec.py`.

### 3.5 Schema — no edits

`WrFeaturesSchema` and `WeeklyStatsSchema` are unchanged (the integration uses existing columns). `ProjectionWeeklySchema` already supports `family = SAMPLED_SUMMARY` (Plan 3d) so the row-level family for `decomposed-baseline` predictions matches `baseline` predictions exactly. No schema edits.

---

## 4. Execution sequence

Order matters; later tasks depend on earlier ones.

1. **`FrozenSampledDistribution` + tests** — net-new class + unit tests on the `n == len` semantics. Foundation for everything downstream; cleanest standalone task.
2. **`DecompositionSpec` + `DecomposedBaselineModel.fit` + fit tests** — net-new module, no `predict_distribution` yet. Verifies the train-side decomposition recipe lands cleanly.
3. **`DecomposedBaselineModel.build_stat_distributions` + `predict_distribution` override + per-row predict tests** — wires the per-row coherent sampling + quantile summarization. Includes the multi-stat coherence test on a synthetic 2-stat config (even though v1 production is 1-stat).
4. **`wr_decomposed_baseline` factory + dispatch registration + factory + Model-protocol-conformance tests** — registers in `_WR_FACTORIES`. Does **not** flip `default_model_class`.
5. **Dual-run backtest + adoption gate** — operational task; no source edits. Produce the binding + informational cell reports. **The `.venv` is shared with main repo; the worktree's source is read via `PYTHONPATH=<worktree>/src` prefixed on every invocation, OR a `pip install -e .` inside an isolated worktree venv. Plan task picks; either works.**
6. **Per-§1.3.5 outcome execution + writeup + PM/TODO update** — conditional source edits per the verdict matrix, plus `reports/wr_target_decomposition_summary.md` + `project_management.md` + `TODO.md`.

Each task touches ≤ 5 files per CLAUDE.md "phased execution" rule.

---

## 5. Risk register

1. **Probe-vs-gate magnitude calibration risk (highest).** Probe binding cell was −0.0042 fpts composite-fpts implied magnitude on the receptions stat alone (Ridge-vs-Ridge). The gate's binding cell is decomposed-baseline-on-WR (all 6 stats) vs ensemble-on-WR — a different comparison. PR #30 → #31 saw a +0.011 fpts shift on both binding cells with sign flipped between probe and gate (the largest Track 2A divergence). v1's marginal-zone probe magnitude makes a sign-flip-at-gate plausible. **Mitigation:** the informational cell (`decomposed-baseline` vs `baseline`) is structurally the apples-to-apples reproduction of the probe at the composite-fpts level; if both cells flip vs probe predictions, that's a strong "decomposition did not carry to composite-fpts" signal regardless of the binding cell's outcome. Document both magnitudes in the summary report regardless of verdict.
2. **Variance under-fitting on the `catch_rate` factor.** Population-level Normal residual std on a [0, 1]-bounded ratio is a crude variance model — true variance varies sharply with target volume (low-target rows have noisy catch rates; high-target rows are more deterministic). The clipped-Normal sampling produces a per-row distribution whose mean is correct (= mu_volume × mu_eff up to clip effects) but whose tails may be poorly calibrated. **Mitigation:** the existing `weekly_calibration_*` snapshot metrics catch tail miscalibration; the summary report flags the receptions cell's p10/p90 coverage vs BaselineModel's Gamma-on-receptions parameterization. If coverage degrades materially, this becomes a known follow-up: factor-appropriate sub-models (logistic for catch_rate) per probe spec §7.4, which natively handle the [0, 1] support.
3. **Within-row coherent sampling is dormant in v1.** Receptions-only mode never exercises the cross-stat correlation guarantee (the architectural payoff named in probe spec §7.3). **Mitigation:** unit test asserts the guarantee on a synthetic 2-stat configuration; the production v1 path uses the same code, just with one decomposed stat. Future v2 expansion (e.g., adding `receiving_yards` decomposition after factor-appropriate sub-models close the probe) inherits the architectural correctness for free.
4. **QuantileDistribution rehydration loses cross-stat correlation.** Documented in §3.1.5; mitigated by the persistence-after-scoring boundary in `predict_distribution`. **Risk:** a future caller that reads back the persisted params blob and recomposes per-stat distributions for downstream scoring (e.g., re-scoring under a different ruleset post-hoc) loses the cross-stat correlation. **Mitigation:** the v1 use case (training + immediate scoring inside `predict_distribution`) does not exercise re-scoring; the documented limitation goes into a code comment on `_pack_per_row` and a TODO entry. If re-scoring becomes load-bearing, that's the natural prompt to add a true `ProductDistribution` codec branch.
5. **Subclass fragility on dataclass inheritance.** Adding fields to a `@dataclass` subclass of `@dataclass` requires the subclass fields to all have defaults (the parent class has post-init-required fields with no defaults). **Mitigation:** all four new fields use `field(default_factory=dict)`, so an unfitted instance is constructable. Defaults match BaselineModel's existing-state-field pattern.
6. **`_x_frame_with_bool_coercion` method-resolution / re-use.** Inherited as-is — no override needed. **Risk:** a future BaselineModel refactor that changes this helper's signature breaks DecomposedBaselineModel silently. **Mitigation:** the Model-protocol conformance test exercises `fit` + `predict_distribution` end-to-end, which calls the helper indirectly. A direct unit test on the helper is unnecessary additional surface.
7. **Worktree-venv routing.** Implementation phase needs the worktree's `src/projections/models/decomposed_baseline.py` to be the version imported, not the main repo's. **Mitigation:** the plan task explicitly invokes via `PYTHONPATH=<worktree>/src ../../../.venv/Scripts/python.exe ...` or installs editably into an isolated venv. Either works; plan picks.

---

## 6. Testing

`tests/test_models/test_decomposed_baseline.py`:

1. **`DecompositionSpec`** — frozen dataclass: field assignment raises, equality on field values, copying via `dataclasses.replace`.
2. **`FrozenSampledDistribution`** — `n == len` returns `self.samples` (assert array identity / `np.shares_memory`); `n != len` resamples (rng-dependent). Mean / std / quantile / cdf match numpy reference values on a 10K-sample fixture.
3. **Fit happy path** — synthetic 4-year WR-shaped frame (mirrors PR #21 / #26's `baseline_features_wr` fixture coverage); verify `volume_ridges[Stat.TARGETS]`, `efficiency_ridges[Stat.RECEPTIONS]`, `volume_variance[Stat.TARGETS]`, `efficiency_variance[Stat.RECEPTIONS]` populated. Verify `self.ridges` contains entries for every target_stat including the decomposed one (per §3.1.3 "fit both arms"). Verify `model_id == "decomposed-baseline:wr:<hash>:<start>-<end>"`.
4. **Empty `decomposed_stats`** — constructing with `decomposed_stats={}` and calling `fit` should produce a model behaviorally indistinguishable from `BaselineModel` (assert on `predict_distribution` output equality on a fixed seed). Verifies the opt-in arch doesn't regress the empty-config case.
5. **`build_stat_distributions` per-stat types** — non-decomposed stats produce parametric distributions matching `_WR_DIST_FAMILIES`; decomposed stats produce `FrozenSampledDistribution` (NOT `QuantileDistribution` — the quantile conversion happens at persistence time, not at build time).
6. **Within-row cross-stat coherence (architectural)** — construct with `decomposed_stats={RECEPTIONS: ..., RECEIVING_YARDS: DecompositionSpec(volume_stat=TARGETS, efficiency_label="yards_per_target", efficiency_clip_hi=+inf)}`. Fit on synthetic frame. On a single eval row, the two `FrozenSampledDistribution` instances must have Pearson correlation > 0.5 element-wise (because they share the volume draw). Asserts the architectural guarantee even though v1 production is 1-stat.
7. **`predict_distribution` round-trip** — fit, predict, validate against `ProjectionWeeklySchema`, unpack the `params` blob, assert decomposed stats appear as `QuantileDistribution` (the persisted form), non-decomposed stats appear in their parametric form.
8. **`predict_distribution` determinism** — same seed → same per-row mean / p10 / p50 / p90 across two invocations.
9. **`wr_decomposed_baseline` factory** — Model-protocol conformance (mirrors the BaselineModel factory test if one exists; if not, add a minimal one).
10. **`_WR_FACTORIES` registration** — `POSITION_DISPATCH[Position.WR].factories["decomposed-baseline"]()` returns an unfitted `DecomposedBaselineModel`; `_PositionDispatch[Position.WR].default_model_class` is still `"ensemble"` (not flipped pre-gate).

Real-data smoke + adoption gate evidence captured in the PR per CLAUDE.md "Forced verification" rule.

---

## 7. Reports

`reports/wr_target_decomposition_summary.md`:

- Probe-vs-gate calibration table: probe Δ-RMSE (per stat, receptions only) → expected composite-fpts contribution; gate Δ-composite-fpts-RMSE (both cells).
- Both adoption-gate cells: verdict + 95% CI on RMSE Δ + Spearman Δ + n_paired + per-year breakdowns.
- Coverage statistics: `targets > 0` rate per eval year on WR rows (eval + train); pooled.
- Per-§1.3.5 outcome narrative.
- Deferred-follow-up disposition per §1.3.5 branch.
- Probe-vs-gate magnitude flag check: was the binding cell's composite-fpts magnitude under 0.005 fpts? If yes, document the marginal-zone caveat per PR #31's retrospective rule (informational only — does not change the verdict).

`reports/adoption_gate_wr_decomposed_baseline_vs_ensemble.{md,csv}` and `reports/adoption_gate_wr_decomposed_baseline_vs_baseline.{md,csv}` — produced by `scripts/adoption_gate.py`.

---

## 8. Estimated scope

~6 plan tasks, single session, no overnight backtests. Per CLAUDE.md "phased execution" each task touches ≤ 5 files.

| Task | Surface | Files touched |
|---|---|---|
| 1. `FrozenSampledDistribution` + tests | Distribution layer | `src/projections/distributions/sampled.py` (new) OR `src/projections/scoring/score_distribution.py` (extended), `src/projections/distributions/__init__.py`, `tests/test_distributions/test_sampled.py` |
| 2. `DecompositionSpec` + `DecomposedBaselineModel.fit` + fit tests | Model core | `src/projections/models/decomposed_baseline.py` (new), `src/projections/models/baseline.py` (optional hook extraction per §3.1.6), `tests/test_models/test_decomposed_baseline.py` (new) |
| 3. `build_stat_distributions` + `predict_distribution` override + tests | Model predict | extend the module from Task 2, extend the test from Task 2, possibly `src/projections/models/baseline.py` for the `_pack_per_row` hook (per §3.1.6) |
| 4. `wr_decomposed_baseline` factory + dispatch + Model-protocol conformance | Factory + registry | extend `src/projections/models/decomposed_baseline.py`, `src/projections/models/__init__.py`, factory test in `tests/test_models/test_decomposed_baseline.py` |
| 5. Real-data dual-run backtest + adoption gate (binding + informational) | Reports + artifacts | `reports/adoption_gate_wr_decomposed_baseline_vs_ensemble.{md,csv}`, `reports/adoption_gate_wr_decomposed_baseline_vs_baseline.{md,csv}`, possibly `data/backtest/run_<ts>/` artifacts (manual, not committed wholesale) |
| 6. Per-§1.3.5 outcome execution + writeup + PM/TODO update | Decision log + conditional source edits | `reports/wr_target_decomposition_summary.md`, `project_management.md`, `TODO.md`, conditionally `src/projections/models/__init__.py` (`default_model_class` flip on ADOPT) or full revert per REGRESSION branch |

End-to-end wall-clock: model fit is the same cost as BaselineModel + one extra RidgeCV per training fold. Dual-run backtest runs at standard wall-time. Should fit in one focused session, modulo the §1.3.5 conditional execution which is fast (one-line source edit + writeup).

---

## 9. Implementation plan handoff

After this spec is approved and committed on `feat/wr-target-decomposition`, the next step is the writing-plans skill to produce `docs/superpowers/plans/2026-05-13-wr-target-decomposition-integration.md` decomposing the 6 tasks above into per-task implementation steps with explicit phase boundaries and per-task verification commands.
