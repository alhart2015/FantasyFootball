# Weather Refined-Unit RB+WR Integration — Summary

**Date:** 2026-05-09
**Branch:** `feat/weather-refined-rb-wr`
**Spec:** `docs/superpowers/specs/2026-05-09-weather-refined-rb-wr-design.md`
**Plan:** `docs/superpowers/plans/2026-05-09-weather-refined-rb-wr.md`
**Predecessors:** PR #28 (broad-cut weather probe → SIGNAL via lgb-nb augment) → PR #29 (v1 RB+WR integration → ADOPT both) → PR #30 (refined-unit probe → SIGNAL via lgb-nb composite, RB swap + WR swap+augment ADOPT) → this PR.

## Verdict: **both positions full-revert** per §1.3.5 contingency

The adoption gate's binding cells `(LightGBMNbModel, RB)` and `(LightGBMNbModel, WR)` both returned `DO_NOT_ADOPT` with point estimates **opposite** in sign from PR #30's probe predictions. Per the spec §1.3.5 contingency matrix, both positions full-revert: schema swap + `_<POS>_FEATURE_COLUMNS` swap + builder-boundary tests + cluster-A fixture defaults all rolled back to main's pre-PR state. Net code change vs main: **zero** (the 9 modified files are byte-identical to main).

This is the **first integration in the Track 2A workflow to fully revert on a probe-vs-gate divergence**. The spec's risk register anticipated this outcome ("smallest binding-cell magnitude in Track 2A history, just inside the per-cell noise floor of ~0.001-0.002 fpts; a small calibration error could flip WR's lgb-nb cell to MARGINAL or DO_NOT_ADOPT"). Both binding cells flipped, not just one.

## Per-(model_class, position) gate verdicts

Dual-run via `scripts/backtest_dual.py` orchestration (PR #29's pattern). Candidate run: this branch (refined 8-col bundle in schemas). Baseline run: main at `51e61f5` (v1 4-col bundle). 4 gate cells × 2 model classes = 8 verdict cells (4 positions × 2 model classes); evaluated only the 4 binding/contingency cells per §1.3.4.

| Cell | n_paired | RMSE Δ (fpts) | RMSE 95% CI | Spearman Δ | Spearman 95% CI | Verdict |
|---|---:|---:|---|---:|---|:---:|
| (baseline, RB)    | 5273 | +0.0020 | [-0.0117, +0.0156] | -0.0015 | [-0.0040, +0.0010] | DO_NOT_ADOPT (NULL — CI brackets 0) |
| (baseline, WR)    | 8460 | +0.0120 | [+0.0013, +0.0228] | -0.0025 | [-0.0045, -0.0005] | DO_NOT_ADOPT (RMSE REGRESSION shape — CI strictly above 0; Spearman strictly negative) |
| **(lgb-nb, RB)**    | 5273 | **+0.0012** | **[-0.0064, +0.0090]** | -0.0002 | [-0.0015, +0.0011] | **DO_NOT_ADOPT (NULL — both CIs bracket 0)** |
| **(lgb-nb, WR)**    | 8460 | **+0.0060** | **[-0.0001, +0.0119]** | -0.0016 | [-0.0029, -0.0002] | **DO_NOT_ADOPT (RMSE NULL just barely; Spearman strictly negative)** |

(Bolded rows are the binding cells per §1.3.5.)

The skipped 3 informational classes (`lightgbm`, `lightgbm-tuned`, `ensemble`) per spec §1.3.4 were not run; back-fillable by a follow-up `--model lightgbm,lightgbm-tuned,ensemble` backtest if a future cross-class re-eval is ever scoped (low priority — both binding model classes returned NULL).

## Probe-vs-gate calibration: largest divergence in Track 2A history

PR #30 swap probe predictions vs measured:

| Position | Probe RMSE Δ (fpts) | Probe 95% CI | Gate RMSE Δ (fpts) | Gate 95% CI | Magnitude Δ | Sign |
|---|---:|---|---:|---|---:|---|
| RB | -0.0088 | [-0.0153, -0.0030] | **+0.0012** | [-0.0064, +0.0090] | **+0.0100** | **flipped** |
| WR | -0.0050 | [-0.0098, -0.0006] | **+0.0060** | [-0.0001, +0.0119] | **+0.0110** | **flipped** |

**~+0.011 fpts shift on both binding cells, with sign flipped.** Compare to historical calibration:
- PR #20 → #21 (RB v1 PBP): probe -0.0124 → gate -0.0124 (matched to 4 decimals).
- PR #25 → #26 (WR trajectory): probe -0.0414 → gate -0.0371 (within probe CI).
- PR #25 → #27 (TE trajectory): probe -0.0107 → gate -0.0090 (within probe CI).
- PR #28 → #29 (RB+WR v1 weather): probe -0.0081 / -0.0110 → gate -0.0077 / -0.0104 (within ~0.0006 fpts on both cells).

PR #30 → this PR is the first Track 2A integration where the **gate flips a binding-cell sign** with the probe's CI strictly negative. Three plausible mechanisms (none disambiguable from this PR's data alone — all "probe found ephemeral signal"):

1. **Bootstrap noise + multiple comparison.** The probe ran 16 (model × mode × position) cells; ADOPT decisions on 3 cells across the position-decoding §1.2 matrix have a non-trivial false-positive rate at α=0.05 even with strict CI tests. The probe's decision logic was robust against single-cell false positives but not against systematic small-magnitude drift across multiple correlated cells.
2. **Probe-vs-production join shape.** The probe's "swap" mode used `weather.parquet` override + `--drop` of v1 cols; the production builder's swap is `attach_weather_features` returning 12 cols → schema strict-filter dropping the 4 v1. The two paths should be mathematically equivalent, but a subtle difference (e.g., NaN-row dropna semantics during the join, or stale fold-split caching in the probe's bootstrap) could systematically shift estimates by ~0.01 fpts.
3. **Genuine no-signal.** The refined cols carry no marginal lift over v1 at the production scale; the probe's signal was a coincidence. Mechanism-consistent with the close call: PR #30's probe coverage relaxation (--coverage-threshold 0.90) was a known precedent risk; per-(position, season) is_cold_weather coverage in 2022 was 67% (the "deepest threshold relaxation" in Track 2A flagged in PR #30). The 2022-trough-on-cold-weather is a feature with low effective sample size, and small-magnitude lift estimates from low-coverage features are the most fragile.

The third mechanism is the most parsimonious. Recommended retrospective takeaway: **a probe binding-cell magnitude under ~0.005 fpts with coverage relaxation should be treated as MARGINAL, not SIGNAL**, even if Phase 2's bootstrap CI test passes. The current probe spec's threshold could be tightened in a future revision.

## Per-position §1.3.5 outcome: **full-revert × 2**

Per the spec's contingency matrix:
- **(lgb-nb, RB) DO_NOT_ADOPT** → full-revert RB.
- **(lgb-nb, WR) DO_NOT_ADOPT** → full-revert WR.

Combined: full-revert both. Single revert commit (`c4ba548`) restores the 9 affected files to main's state:
- `src/projections/schemas.py` — refined 8 cols → v1 4 cols on `RbFeaturesSchema` + `WrFeaturesSchema`.
- `src/projections/models/baseline.py` — refined names → v1 names in `_RB_FEATURE_COLUMNS` + `_WR_FEATURE_COLUMNS`.
- `src/projections/features/weather_features.py` — module docstring restored.
- `tests/test_features/test_rb.py` + `test_wr.py` — original PR #29 v1 builder-boundary weather tests restored.
- `tests/test_features/test_cache.py` — v1 fixture defaults restored.
- `tests/test_scripts/test_tune_lightgbm.py` — v1 fixture defaults restored.
- `tests/test_schemas/test_dataframe_schemas.py` — v1 fixture defaults restored.
- `tests/backtest/model_metrics.json` — refined snapshot deltas reverted to main's v1 snapshot rows for RB+WR × baseline+lgb-nb.

The `git diff main` for this PR shows zero net code change — only the spec, plan, and reports differ from main. The refined-unit weather work is now historical record under TODO #25.

The contingency note in the spec deliberately did NOT preserve a "modified-shape" branch (PR #29's pattern of "keep schema, revert _<POS>_FEATURE_COLUMNS") because the strict-replace nature of this PR makes that shape ill-defined: reverting baseline.py's hardcoded list to v1 names while the schema has refined names would point baseline at non-existent columns. Full-revert is the cleanest contingency.

## Coverage statistics on the refined cols (run prior to gate)

Per `reports/weather_refined_rb_wr_coverage.txt` (working artifact from Plan Task 8). Refreshed RB + WR feature caches for 2018-2024 against the refined schema; sampled coverage on 2021-2024 eval window:

| Position | Season | n rows | is_cold_weather | surface (×6) | is_primetime |
|---|---|---:|---:|---:|---:|
| RB | 2021 | 1852 | 0.961 | 1.000 | 1.000 |
| RB | 2022 | 1850 | 0.662 | 0.978 | 1.000 |
| RB | 2023 | 1771 | 0.851 | 0.872 | 1.000 |
| RB | 2024 | 1758 | 0.982 | 0.993 | 1.000 |
| WR | 2021 | 2961 | 0.961 | 1.000 | 1.000 |
| WR | 2022 | 2935 | 0.651 | 0.977 | 1.000 |
| WR | 2023 | 2964 | 0.850 | 0.872 | 1.000 |
| WR | 2024 | 2879 | 0.984 | 0.993 | 1.000 |

Cross-checked against PR #30's `reports/feature_probe_weather_refined_override_audit.md`: all cells within ±2pp of probe-audit values. **Builder wiring is correct** — the verdict divergence is not explained by a coverage / wiring bug. The 2022 `is_cold_weather` 0.66/0.65 trough is the same upstream-NaN pattern PR #30 documented and was a known risk going into this gate (spec §5 noted "the smallest binding-cell magnitude … with coverage relaxation").

`is_primetime` 100.0% non-NaN per (position, season) confirms the post-`56df07f` schedules state (no ET-as-UTC bug).

## What this closes / what's still open

**Closes (under TODO #25):**
- The "broad-cut refined-unit weather at the in-builder unit" branch on the RB and WR ADOPT cells from PR #30. Both are now empirically NULL at the production scale.

**Still open (under TODO #25):**
- Refined-unit-of-refined-unit candidates: continuous `kickoff_hour_et` (vs binary `is_primetime`), `is_london` (`kickoff_hour_et < 11`), surface × position interactions, per-team weather acclimation, precipitation (would require new ingest), wind direction (new ingest). **None queued.** Given this PR's verdict, refined-weather work should be deprioritized — there's no evidence the refined unit is the binding constraint over v1, and no probe-level evidence (the PR #30 probe is now retrospectively suspect for false-positive signal). A future weather-related plan should require independent mechanism evidence (e.g., a per-stat passing-deep-vs-short × weather decomposition with prior data backing) before re-probing.

**Cross-class production-routing follow-up (RB and WR):**
PR #29 logged the cross-class flip question for v1 weather. With this PR's verdict, **the question is even less load-bearing** — the lgb-nb-with-refined cell measured +0.0012 RB / +0.0060 WR vs baseline-v1, which makes lgb-nb-with-anything an unattractive routing flip candidate at present. Plan 8's `BaselineModel` routing for both RB and WR remains the right call. The PR #29-deferred follow-up is closed-without-action; no separate ticket needed.

## Spec gaps caught + fixed during execution

1. **`scripts/refresh_features.py` CLI takes a single position** — same gap PR #29 caught. Plan correctly called for two invocations (rb, then wr); execution followed.
2. **`scripts/adoption_gate.py` dual-run mode requires single-model-class run dirs** — same gap PR #29 caught. The plan referenced this as a known issue but didn't prescribe the workaround in the Task 10 steps. Discovered at gate-run time when the merge raised `MergeError: Merge keys are not unique`. Workaround: split each run's `results.parquet` by model_class into per-model-class subdirs (`run_baseline_baseline/`, `run_baseline_lightgbm-nb/`, `run_candidate_baseline/`, `run_candidate_lightgbm-nb/`), then run the gate per (model_class) per (position) — 4 invocations total. The split logic is documented inline in `scripts/_coverage_check_refined.py`'s sibling pattern; for any future refined-feature plan, prescribe this in the plan steps directly.
3. **Pre-commit mypy hook uses system Python** — pre-existing pydantic v1 vs venv pydantic v2 conflict. Workaround: `PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit ...` to put the venv's mypy first. Out of scope to fix the pre-commit config in this PR.
4. **Spec said "5 PR #29 weather tests per file"; plan said "4"** — plan corrected mid-flight (commit `559d4d9`). The 5th test (`outdoor_nan_data_propagates_nan`) was rewritten in Task 6 then reverted along with everything else.

## Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-05-09 | Spec → plan → execute on `feat/weather-refined-rb-wr` | Per spec §7 phasing rule + CLAUDE.md workflow rule. |
| 2026-05-09 | Modified-shape contingency = full-revert (not PR #29's "keep schema, revert baseline.py") | Strict-replace integration leaves no clean modified-shape: reverting `_<POS>_FEATURE_COLUMNS` to v1 names while schema has refined names points baseline at non-existent cols. User-decided in brainstorming. |
| 2026-05-09 | Bundle RB + WR in single PR | PR #29 precedent: same model class binds for both positions; per-position contingency matrix runs independently within the same PR. |
| 2026-05-09 | Skip 3 informational model classes per §1.3.4 | Wall-time risk + TODO #29 lightgbm-tuned dominated. |
| 2026-05-09 | Run gate per (model_class) per (position) — 4 invocations | adoption_gate dual-run mode requires single-class run dirs; split was needed at execution time. |
| 2026-05-09 | Both positions DO_NOT_ADOPT → full-revert per §1.3.5 | Probe-vs-gate verdict divergence; spec §5 risk fired. |

## Reports

- 4 per-(model_class, position) `.md` + `.csv` adoption gate reports under `reports/adoption_gate_weather_refined_*.{md,csv}`.
- This summary: `reports/weather_refined_rb_wr_summary.md`.
- Coverage cross-check working artifact: `reports/weather_refined_rb_wr_coverage.txt`.
