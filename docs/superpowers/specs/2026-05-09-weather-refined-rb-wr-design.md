# Weather Refined-Unit RB+WR Integration — Design

**Status:** approved (brainstorming, 2026-05-09). Ready for implementation plan.
**Date:** 2026-05-09
**Author:** alden + claude
**Sub-project of:** FantasyFootball / Projections Core
**Builds on:**
- Weather Refined-Unit Family Probe (PR #30, merged at `51e61f5`) — verdict `SIGNAL` via `LightGBMNbModel` composite. Three Phase-2 ADOPT cells: `(lgb-nb swap, RB)` -0.0088 fpts (CI [-0.0153, -0.0030]); `(lgb-nb swap, WR)` -0.0050 fpts (CI [-0.0098, -0.0006]); `(lgb-nb augment, WR)` -0.0051 fpts (CI [-0.0097, -0.0006]). Per spec §1.2 decoding: WR exhibits **strict refinement** (swap + augment ADOPT — refined beats v1 *and* adds on top of v1); RB exhibits **replace-only** (swap ADOPT, augment ~NULL — refined replaces v1 cleanly but doesn't add when stacked). The 8-feature refined bundle (`is_cold_weather`, six surface one-hots `is_a_turf` / `is_astroturf` / `is_fieldturf` / `is_grass` / `is_matrixturf` / `is_sportturf`, `is_primetime`) is greenlit for production integration as a strict-replace of the v1 4-col bundle on RB+WR. QB and TE returned DO_NOT_ADOPT under all 4 (mode × model) cells; QB augment composite is a recurring regression at +0.0099 fpts (CI strictly above 0).
- Weather Features RB+WR Integration (PR #29, merged at `2026-05-08`) — shipped the v1 4-col bundle (`wind_speed_mph`, `is_high_wind`, `temperature_f`, `is_grass_surface`) into `RbFeaturesSchema` + `WrFeaturesSchema`. Wired `attach_weather_features(out, schedules)` into `build_rb_features` (rb.py:251) and `build_wr_features` (wr.py:289). Updated `baseline.py:_RB_FEATURE_COLUMNS` (lines 397–400) and `_WR_FEATURE_COLUMNS` (lines 297–300) with the same 4 v1 names. Verdicts: RB ADOPT (-0.0077 fpts) and WR ADOPT (-0.0104 fpts) at lgb-nb composite. Set the precedent that `attach_weather_features` is the single integration seam for any future weather-feature shipment — this PR's strict-replace consumes that seam unchanged.
- Weather Feature Family Probe (PR #28, merged at `39be213`) — original broad-cut weather probe. Established `compute_weather_features` + `attach_weather_features` + `build_weather_overrides` as the public surface in `src/projections/features/weather_features.py`. The helpers already produce all 12 weather cols (4 v1 + 8 refined), unchanged across PR #29 and PR #30; this PR consumes the refined 8 via schema-level filtering and leaves the helper signature alone.
- TE Trajectory Features Integration (PR #27, merged at `5a9352f`) — set the durable precedent for non-default-binding integrations: schema change for the `LightGBMNbModel` code path, evaluate `baseline + lgb-nb` in the gate (binding + contingency cells), production routing stays on `BaselineModel`, defer cross-class flip to a separate post-merge re-eval. PR #29 reused the same shape; this PR does too.
- RB PBP Features Integration (PR #21, merged at `9895dee`) — set the `_<POS>_FEATURE_COLUMNS` extension precedent: lightgbm family derives feature lists from `*FeaturesSchema.to_schema().columns.keys()` dynamically; baseline.py's hardcoded per-position tuple must be edited explicitly or `BaselineModel` will not see the new features even though the schema validates. PR #26, #27, #29 each re-caught the same gap. The parametrized regression test added in PR #29 (`tests/test_models/test_baseline_feature_columns_match_schema.py`) now structurally enforces this — any drift between `_<POS>_FEATURE_COLUMNS` and the schema fails the test.

**Branch:** `feat/weather-refined-rb-wr` cut from `main` at `51e61f5`. Worktree at `.worktrees/feat-weather-refined-rb-wr/`.

---

## 1. Overview

PR #30's refined-unit probe delivered three ADOPT cells across `(lgb-nb, RB)` swap, `(lgb-nb, WR)` swap, and `(lgb-nb, WR)` augment. The verdict structure (per spec §1.2 decoding) decomposes into a single integration shape: **replace the v1 4-col weather bundle with the refined 8-col bundle in `RbFeaturesSchema` and `WrFeaturesSchema`**. WR's strict-refinement (swap + augment both ADOPT) and RB's replace-only (swap ADOPT, augment ~NULL) collapse to the same code change — the 8-col schema replaces the 4-col schema; the difference between strict-refinement and replace-only is a probe-side semantic that doesn't propagate to the integration surface.

This PR is structurally simpler than PR #29 (which had to wire `attach_weather_features` into the builders for the first time). All the integration plumbing — the import, the call site, the schedules kwarg, the caller scripts — is already in place from PR #29. The substance of this PR is:

1. **Schema-level swap**: remove the 4 v1 cols from `RbFeaturesSchema` + `WrFeaturesSchema`, add the 8 refined cols.
2. **`baseline.py` hardcoded list swap**: same 4-out / 8-in edit in `_RB_FEATURE_COLUMNS` and `_WR_FEATURE_COLUMNS`.
3. **No builder code changes**. `attach_weather_features` returns all 12 cols (unchanged since PR #28); pandera's `strict="filter"` on the schema validate at the end of each builder drops the 4 v1 cols and keeps the 8 refined cols. Confirmed: `RbFeaturesSchema.Config.strict = "filter"` (schemas.py:674) and `WrFeaturesSchema.Config.strict = "filter"` (schemas.py:553); `build_rb_features` ends with `return RbFeaturesSchema.validate(out)` (rb.py:253) and `build_wr_features` ends with `return WrFeaturesSchema.validate(out)` (wr.py:291). The validate-and-return pattern is the canonical "drop v1 from output without changing builder code" mechanism.
4. **Test updates**: rewrite the four PR #29 builder-boundary weather tests to assert refined-bundle behavior (cold-weather, surface-multiclass, primetime-kickoff, bye-week-fallback). Cluster-A defensive grep for the 4 v1 col names; swap defaults at every synthetic-row site.
5. **Module docstring update**: `src/projections/features/weather_features.py` still labels itself "Probe-only — features land in the override parquet, not in `*FeaturesSchema`." That framing is stale post-PR #29 (v1 integrated) and stale-er post this PR (refined integrated). Update to reflect that both bundles are now integrated.

**Critical shape (matching PR #27, #29 precedent): the binding cells are `(LightGBMNbModel, RB)` and `(LightGBMNbModel, WR)`, not `(BaselineModel, *)`.** PR #30's probe ADOPT'd both positions only under lgb-nb; BaselineModel returned 0/120 Phase-1 SIGNAL across both modes for all 4 positions, so Phase 2 was skipped entirely on baseline. Because both positions' production `default_model_class` stays at `baseline` (Plan 8 verdict 2026-04-29; unchanged by PR #29), shipping the refined bundle does **not** automatically improve RB or WR production output — the cols are persisted in the schema for the model classes that demonstrably benefit (`lightgbm-nb`, likely `lightgbm` and `ensemble` though those are not gated here), but production routing stays on baseline pending a separate cross-class re-eval. Same deferred follow-up as PR #29 (which logged it for both RB and WR); this PR simply updates the deferred-follow-up entries to reflect the refined-cols-in-schema state.

**Why combined RB+WR PR (vs split into two sequential PRs):** the binding cells share the same model class (lgb-nb) and the same integration mode (swap, where v1 is replaced by refined). The two positions' schemas and builders are independent at the edit-locus level. The backtest+gate run dominates wall-clock and parallelizes naturally with `--position RB,WR --model baseline,lightgbm-nb`. Per §1.3.5, the per-position contingency matrix is structured so independent revert outcomes ship cleanly within the same PR. PR #29 set this bundling precedent; this PR follows it.

The shipping decision is binary and bound to the **lgb-nb** verdict, **per position independently**: if `(LightGBMNbModel, POS)` returns `ADOPT`, ship that position; if `MARGINAL` or `DO_NOT_ADOPT`, full-revert that position (see §1.3.5 contingency note on why the modified-shape branch from PR #29 doesn't apply cleanly to a strict-replace integration). Other model classes' verdicts (`LightGBMModel`, `LightGBMTunedModel`, `EnsembleModel`) are explicitly informational, **not run** in the gate per PR #27 / PR #29 precedent.

### 1.1 Goals (in scope)

- **Schema swap on `WrFeaturesSchema` (`schemas.py:484-557`).** Remove the 4 v1 weather cols (lines 547–550) along with the existing PR #29 / PR #28 header comment block (lines 542–546); replace with the 8 refined cols and a fresh header comment block:

  ```python
  # Weather features — refined-unit replace per PR #30 verdict (RB swap, WR
  # swap+augment ADOPT under lgb-nb composite). Replaces the v1 4-col bundle
  # from PR #29 (wind_speed_mph, is_high_wind, temperature_f,
  # is_grass_surface) with the 8 refined cols below. Sourced from existing
  # SchedulesSchema cols (temp, surface, kickoff). Domes filled (temp=70 →
  # is_cold_weather=0; surface flags reflect the actual surface code with no
  # override since stadia keep their playing surface across roof states;
  # is_primetime is independent of roof). 2022 has a known coverage trough
  # on is_cold_weather (~0.67 non-NaN per (position, season)) due to upstream
  # NaN temp on outdoor games — nullable=True absorbs; pooled-row CI is
  # symmetric across baseline/candidate dropna; documented in PR #30 audit.
  is_cold_weather: Series[float] = pa.Field(ge=0, le=1, nullable=True)
  is_a_turf: Series[float] = pa.Field(ge=0, le=1, nullable=True)
  is_astroturf: Series[float] = pa.Field(ge=0, le=1, nullable=True)
  is_fieldturf: Series[float] = pa.Field(ge=0, le=1, nullable=True)
  is_grass: Series[float] = pa.Field(ge=0, le=1, nullable=True)
  is_matrixturf: Series[float] = pa.Field(ge=0, le=1, nullable=True)
  is_sportturf: Series[float] = pa.Field(ge=0, le=1, nullable=True)
  is_primetime: Series[float] = pa.Field(ge=0, le=1, nullable=True)
  ```

  All 8 cols are `Float64` boolean-style (1.0 / 0.0 / NaN) per the helpers' actual output dtypes (`weather_features.py:_compute_is_cold_weather`, `_compute_surface_onehot`, `_compute_is_primetime`). Bounds `ge=0, le=1` on every col, `nullable=True` on every col.

- **Schema swap on `RbFeaturesSchema` (`schemas.py:611-675`).** Identical edit shape. Remove the 4 v1 weather cols (lines 668–671) + header comment block (lines 663–667); insert the same 8 refined cols + header comment block in the same slot.

- **`baseline.py` hardcoded feature-list swap (`models/baseline.py`).**
  - `_WR_FEATURE_COLUMNS` (line 266): remove the 4 v1 names at lines 297–300; insert the 8 refined names in the same slot. Keep the existing `# Weather features (PR #28...)` comment by replacing it with a comment that references PR #30's strict-replace verdict.
  - `_RB_FEATURE_COLUMNS` (line 366): same edit; remove lines 397–400; insert 8 refined names at the same slot. Same comment swap.
  - The lightgbm family (`models/lightgbm.py:120-122`) derives `_RB_FEATURE_COLUMNS` and `_WR_FEATURE_COLUMNS` from `RbFeaturesSchema.to_schema().columns.keys()` and `WrFeaturesSchema.to_schema().columns.keys()` directly. `lightgbm_nb.py` and `lightgbm_tuned.py` import from `lightgbm.py`. Once Phase 1 lands the schema swap, the lightgbm family auto-picks-up the 8 refined cols and drops the 4 v1 cols. **No edits to lightgbm files.** Phase 3 includes a smoke-test assertion that the dynamically-derived RB and WR feature lists post-schema-swap contain the 8 refined names and **do not** contain the 4 v1 names.
  - The `tests/test_models/test_baseline_feature_columns_match_schema.py` regression test added in PR #29 verifies `set(_<POS>_FEATURE_COLUMNS) == set(SCHEMA.columns) - identity` for each position; once both Phase 1 (schema swap) and Phase 3 (`_<POS>_FEATURE_COLUMNS` swap) land, the test passes for both positions. If only one lands without the other, the test fails. The test thus structurally enforces the schema/baseline.py invariant on every future feature-list edit.

- **No builder code changes in `features/rb.py` or `features/wr.py`.** PR #29 wired `attach_weather_features(out, sch)` into both at `rb.py:251` and `wr.py:289`. The helper still returns all 12 weather cols (unchanged from PR #28; reused unchanged by PR #29 and PR #30). Pandera's `strict="filter"` on the schema validate at the end of each builder (`rb.py:253` / `wr.py:291`) drops the 4 v1 cols at the boundary. Cost: ~4 trivial column ops per builder invocation (negligible). Benefit: zero builder churn; `attach_weather_features` and the upstream `build_weather_overrides` assembler stay usable by `scripts/build_weather_override.py` for any future probe re-run.

- **Tests at the builder boundary (`tests/test_features/test_rb.py`, `tests/test_features/test_wr.py`).** Drop or rewrite the four PR #29 weather tests in each file, replacing with refined-shape analogs. Edits to each file are parallel:
  - **Drop**: `test_<pos>_features_attach_weather_dome_fill` (asserts `wind_speed_mph == 0.0`, `temperature_f == 70.0`, `is_high_wind == 0.0` — those cols won't be in the schema after this PR), `test_<pos>_features_attach_weather_outdoor_high_wind`, `test_<pos>_features_attach_weather_grass_surface`. The bye-week-fallback test stays in shape but its assertions update.
  - **Add**: `test_<pos>_features_attach_weather_refined_dome_fill` — synthetic `roof="dome"` schedules row at 1pm ET. Assert `is_cold_weather == 0.0` (falls out of dome's `temperature_f=70` fill), all 6 surface flags reflect the actual surface code (no roof-based override; stadium keeps its playing surface), `is_primetime == 0.0`.
  - **Add**: `test_<pos>_features_attach_weather_refined_cold_outdoor` — synthetic outdoor `wind=10, temp=28, roof=NaN, surface="grass", kickoff=1pm ET`. Assert `is_cold_weather == 1.0`, `is_grass == 1.0`, all other surface flags `0.0`, `is_primetime == 0.0`.
  - **Add**: `test_<pos>_features_attach_weather_refined_surface_multiclass_<code>` (one per code, parametrized) — for each of the 6 codes in `_SURFACE_CODES`, synthesize a schedules row with that surface; assert `is_<code> == 1.0` and all 5 other surface flags `0.0`.
  - **Add**: `test_<pos>_features_attach_weather_refined_primetime_kickoff` — kickoff timestamp 8:20pm ET (TNF / SNF window): assert `is_primetime == 1.0`. Same player-week with kickoff 1pm ET: assert `is_primetime == 0.0`. Same player-week with kickoff `pd.NaT` in schedules: assert `is_primetime` is NaN.
  - **Rewrite (not add)**: `test_<pos>_features_attach_weather_refined_bye_week_fallback` — synthetic player-week where the schedule join returns no row. Assert all 8 weather cols are NaN and the `nullable=True` flag accepts validation.

- **Cluster-A defensive grep — swap v1 defaults for refined defaults at every synthetic-row construction site.** Anchored on the v1 col names being removed:

  ```bash
  grep -rn "wind_speed_mph" tests/   # the canonical v1 col present at every synthetic-row site
  ```

  Per the survey done during this spec's brainstorming, 7 test files reference `wind_speed_mph`. Two are out-of-scope (they test the helpers, which still produce 12 cols):
  - `tests/test_features/test_weather_features.py` — unit tests for `compute_weather_features` / `attach_weather_features` / `build_weather_overrides`. **Keep v1 references** — these helpers' contract is unchanged.
  - `tests/test_scripts/test_build_weather_override_cli.py` — CLI test for `scripts/build_weather_override.py`. The probe override still produces 12 cols. **Keep v1 references.**

  Five are in-scope (need the swap):
  - `tests/test_features/test_rb.py` — covered by the test rewrites above.
  - `tests/test_features/test_wr.py` — covered.
  - `tests/test_features/test_cache.py` — synthetic minimal RB / WR features rows. Drop the 4 v1 defaults; add 8 refined defaults (`is_cold_weather=0.0, is_a_turf=0.0, is_astroturf=0.0, is_fieldturf=0.0, is_grass=1.0, is_matrixturf=0.0, is_sportturf=0.0, is_primetime=0.0` — `is_grass=1.0` so the multi-class one-hot row sums to 1).
  - `tests/test_scripts/test_tune_lightgbm.py` — synthetic features fixture. Same swap.
  - `tests/test_schemas/test_dataframe_schemas.py` — schema validation tests with explicit v1 col references. Same swap.

  After the swap, re-run the grep. Expectation: only the two helper-test files match. Implementation plan asserts this.

- **Update `weather_features.py` module docstring** (`src/projections/features/weather_features.py:1-12`). Current docstring says `"""Weather feature computes for the weather refined-unit family probe (PR #28 broad-cut + this PR's refinements). … Probe-only — features land in the override parquet, not in *FeaturesSchema. Integration follow-up is conditional on the family-probe verdict per docs/superpowers/specs/2026-05-09-weather-refined-unit-probe-design.md."""` Update to reflect that **both** v1 (PR #29) and refined (this PR) are integrated; v1 is superseded; the helper is the canonical code path for both probe overrides and production builders. One-paragraph rewrite; see §3.6 for the proposed text.

- **No `tests/conftest.py` extension.** Weather is per-game (not trailing-N); the existing `baseline_weekly_stats_*` / `baseline_features_*` fixtures already cover the relevant week ranges. Same as PR #29.

- **No caller-script changes.** `schedules` is already loaded and threaded through all 4 caller scripts (`scripts/refresh_features.py`, `scripts/train_baseline.py`, `scripts/predict_2024.py`, `scripts/sanity_check_baseline.py`) for all 4 positions per PR #29.

- **Refresh both feature caches**: `python scripts/refresh_features.py rb wr --seasons 2018-2024`. Mandatory: the existing caches under `data/features/rb/` and `data/features/wr/` were written with the v1 cols and will fail validation against the refined schema. Manual; output not committed (lives under `data/features/{rb,wr}/...`, gitignored convention).

- **Run the backtest snapshot regen**: `python scripts/backtest.py --position RB,WR --model baseline,lightgbm-nb --update-snapshot`. 4 cells (RB+WR × baseline+lgb-nb). Snapshot updates committed (`tests/backtest/model_metrics.json` rows for these 4 cells).

- **Run the adoption gate**: dual-run mode against the pre-PR `main` SHA. Output: `reports/adoption_gate_weather_refined_rb_wr.{md,csv}`. Commit. (If the gate exposes a `--coverage-threshold` flag and rejects a default of 0.95, match PR #30's `--coverage-threshold 0.90` precedent given the `is_cold_weather` 2022 trough.)

- **Write the summary report** (`reports/weather_refined_rb_wr_summary.md`) consolidating: probe-predicted vs gate-measured magnitudes per position; per-(model_class, position) verdicts (4 cells); per-position §1.3.5 outcome (ship-as-designed / full-revert per position); coverage statistics on the 8 cols at the eval window cross-checked against PR #30's `feature_probe_weather_refined_override_audit.md`; explicit binding-cell-shift note (lgb-nb, not baseline) and the deferred cross-class production-routing question for both RB and WR.

- **Update `project_management.md` and `TODO.md` #25** per §6.

### 1.2 Non-goals (deferred)

- **No QB / TE schema changes.** PR #30 returned DO_NOT_ADOPT on QB and TE for both modes under lgb-nb. QB augment composite was a recurring regression at +0.0099 fpts (CI strictly above 0; sharper than PR #28's +0.0077 with CI bracketing 0). The "do not extend `QbFeaturesSchema` with weather features" rule established in PR #28 / PR #29 reinforces here. TE's smaller n_paired (4257 vs 5273 RB / 8470 WR) and weaker mechanism story leave it as a "close at this cut" outcome. Refined-unit-of-refined-unit candidates remain open under TODO #25 (continuous `kickoff_hour_et`, `is_london`, surface × position interactions, per-team weather acclimation, precipitation, wind direction). None queued in this PR.
- **No production routing flips.** `_PositionDispatch[RB|WR].default_model_class` stays at `baseline` per Plan 8. The cross-class flip question (does `lgb-nb-with-refined-weather` beat `baseline-without-weather` for RB or WR at the position level?) is a separate cross-class re-eval — same shape PR #27 deferred for TE trajectory, and PR #29 deferred for RB and WR with v1 weather. With refined cols replacing v1 in the schemas, the deferred re-eval question now applies to refined; the spec flags it explicitly per position in the summary report.
- **No new ingest.** Weather sourced from existing `SchedulesSchema` columns (`temp`, `surface`, `kickoff`) at `data/raw/schedules/...`. The opt-in `--run-network` smoke at `tests/test_ingest/test_api_drift.py` already covers `nfl_data_py.import_schedules` column-rename drift. Note: PR #30 fixed an in-scope ingest-layer bug in `_build_kickoff` (commit `56df07f`) that mis-tagged ET wall-clock as UTC; the fix is on `main` and load-bearing for `is_primetime`. This PR consumes the corrected partitions; no further ingest changes.
- **No helper changes in `weather_features.py`.** `compute_weather_features`, `attach_weather_features`, and `build_weather_overrides` all stay identical to PR #30's code (which itself is the PR #28 + PR #29 carry-forward extended with the refined computes). The helper still returns 12 cols (4 v1 + 8 refined); pandera filters at the schema boundary. The only edit to `weather_features.py` in this PR is the module docstring.
- **No probe re-run.** PR #30's probe is the binding evidence; this PR consumes its verdict and runs the production gate. No re-probing at any unit.
- **No 5-class gate run** per spec §1.3.3 — `baseline + lgb-nb` only, matching PR #27 / PR #29 precedent. The 3 skipped classes (`lightgbm`, `lightgbm-tuned`, `ensemble`) are explicitly informational per spec §1.3.4 and back-fillable by a follow-up backtest if a cross-class routing-flip discussion ever needs them. TODO #29 already flags `lightgbm-tuned` as a pruning candidate (dominated 16/16 by lgb-nb on RMSE in earlier runs); spending wall-clock on it for two positions × refined cols doesn't move any decision.
- **No cluster-A grep extension to the helper test files.** `tests/test_features/test_weather_features.py` and `tests/test_scripts/test_build_weather_override_cli.py` test the helper / probe-override-CLI surface, both of which still produce 12 cols. Their v1 references are correct and stay.
- **No deprecation of the v1 col names in `compute_weather_features`'s output.** The function still returns `wind_speed_mph` / `is_high_wind` / `temperature_f` / `is_grass_surface` alongside the 8 refined cols. They're filtered by the schema at the production builder boundary; the probe override path keeps them. A future cleanup that removes v1 from `compute_weather_features` would require updating `build_weather_overrides`, the override CLI, the helper tests, and the probe override script, and would make any future re-probe at a v1-related axis impossible. Out of scope here.
- **No CONTRIBUTING.md changes.** PR #28 added the "Regenerating the weather override" subsection; PR #29 updated it for v1 integration; PR #30 updated it for the refined-unit family bundle. The override path is unchanged in this PR.
- **No spec / plan / report changes for prior work.** PR #28 / #29 / #30 specs, plans, and reports stay as historical record.

### 1.3 Success criteria

The spec is complete iff all of:

1. **Schema swap + `_<POS>_FEATURE_COLUMNS` swap + tests + cluster-A fixture defaults land cleanly.** `pytest -v` (full suite), `mypy src tests` (zero violations), `ruff check src tests scripts` (zero violations), `ruff format --check src tests` (no drift). The `tests/test_models/test_baseline_feature_columns_match_schema.py` regression test passes for all 4 positions (the QB and TE entries don't change since QB/TE schemas are untouched).

2. **Refreshed RB + WR feature caches validate against the swapped schemas** at every `(season, week)` partition. The 8 refined cols are present; the 4 v1 cols are absent.

3. **The full backtest + adoption gate runs successfully on RB + WR × `baseline + lgb-nb`** across the standard `2021-2024` holdout years. 2 positions × 2 model classes = 4 gate cells. The 3 informational classes (`lightgbm`, `lightgbm-tuned`, `ensemble`) are explicitly skipped per §1.3.4.

4. **Other model classes are explicitly informational.** The summary report records what cells were *not* run and the rationale (PR #27 / PR #29 precedent + wall-time risk + TODO #29 lightgbm-tuned dominated).

5. **The summary report (`reports/weather_refined_rb_wr_summary.md`) records all of:**
   - The probe's predicted composite RMSE deltas: **(lgb-nb swap, RB) -0.0088 fpts**, **(lgb-nb swap, WR) -0.0050 fpts** (using swap as the binding-mode prediction, since this PR's integration is a strict-replace = swap shape; the WR augment +0.0051 prediction is informational), from PR #30's `feature_probe_weather_refined_lgbnb_swap.csv`.
   - The gate's measured composite RMSE deltas on `(lgb-nb, RB)` and `(lgb-nb, WR)` with 95% CIs.
   - The per-(model_class, position) verdict table for the 4 cells run (baseline + lgb-nb × RB + WR).
   - Per-position coverage of the 8 new cols at the eval window (2021–2024) and on the full 2018–2024 history; cross-checked against PR #30's audit (`reports/feature_probe_weather_refined_override_audit.md`). Specific check: per-(position, season) `is_cold_weather` non-NaN rate matches the probe's audit (2021 ~98%, 2022 ~67%, 2023 ~85%, 2024 ~98%). `is_primetime` rate ~22% pooled (post-`56df07f` ingest fix, which is on `main`).
   - Explicit note that the binding cells are `(LightGBMNbModel, *)` per §1, both production routings remain on `baseline`, and the cross-class flip question is deferred to a separate per-position follow-up for each of RB and WR (refining the deferred entry that PR #29 logged; the schema state being checked has changed from v1 to refined).
   - The 2-position §1.3.5 contingency outcome (which branch fired for each position; whether ship-as-designed or full-revert).

6. **The shipping decision is bound per position to the `(LightGBMNbModel, POS)` verdict**, with the §1.3.5 contingency applied independently to each of RB and WR. Per-position table immediately below.

If criterion 1 fails, fix and rerun. If criterion 2 fails, the schema swap is wrong — fix before running the gate (most likely cause: a stale v1 col leaked into the cache via a missing `--seasons` arg on the refresh, or an old partition wasn't overwritten). Criterion 3 is mechanical (the gate either runs or doesn't). Criterion 6 is the binding decision.

#### 1.3.5 Per-position contingency matrix

Each position's ship/revert decision is independent — no shared "ship together or revert together" coupling. The matrix below applies to each position separately:

| `(lgb-nb, POS)` | `(baseline, POS)` | Action for POS |
|---|---|:---|
| ADOPT | not REGRESSION | **Ship as designed**: schema swap + `_<POS>_FEATURE_COLUMNS` swap stay. |
| ADOPT | REGRESSION | **Full-revert that position**: undo schema swap + `_<POS>_FEATURE_COLUMNS` swap for that position. (The PR #29 / PR #27 "modified-shape" branch — keep schema, revert `_<POS>_FEATURE_COLUMNS` so baseline doesn't see new cols — does **not** apply here. PR #29's `_<POS>_FEATURE_COLUMNS` was *extending* with new names; reverting that extension left baseline pointed at a working subset. Here the change is a *replace* — reverting `_<POS>_FEATURE_COLUMNS` to v1 names would point baseline at columns that no longer exist in the schema. Cleanest contingency is full-revert. Probability is low: PR #30 baseline returned 0/120 SIGNAL Phase-1 cells across both modes, so baseline composite REGRESSION on this PR's gate would itself be evidence of a probe-vs-gate calibration failure. Same applies to the next row.) |
| MARGINAL or DO_NOT_ADOPT | (any) | **Full-revert that position**: undo schema swap + `_<POS>_FEATURE_COLUMNS` swap. Keep cluster-A test fixture defaults that landed in Phase 4 (cheap to leave; useful for any future weather revisit at a refined-unit-of-refined-unit cut). Document divergence in summary; close the strict-replace branch of TODO #25 for that position. |

Worst-case combined outcome: both positions full-revert. The PR ships zero schema or `_<POS>_FEATURE_COLUMNS` changes; the test rewrites and module docstring update can stay (they're consistent with the helper-level state) or revert too — implementation plan's Phase 6 owns the per-position decomposition.

The probe's evidence on `(BaselineModel, *)` was the strongest "no signal on baseline" evidence in Track 2A history (0/120 SIGNAL across both modes for all 4 positions). The gate flipping baseline RB or WR to REGRESSION (CI strictly above 0) is unlikely but not zero. Per-position revert is pre-decided so the implementation plan doesn't have to negotiate it post-gate.

---

## 2. Inputs

### 2.1 Schedules source

`schedules` partitions read via `read_partition(raw_root, "schedules", season=s)` for the eval range. Already loaded by all 4 caller scripts and passed into `build_rb_features` / `build_wr_features`. **Critical**: includes the post-`56df07f` ET-wall-clock fix from PR #30 — kickoff timestamps are correctly localized so `is_primetime` fires at the expected ~22% rate. Verify before running the gate that `data/raw/schedules` is the post-fix partition (a clean re-ingest after `56df07f` is the canonical state on `main` post-PR-30).

### 2.2 Player-team-week index inside builders

`build_rb_features` and `build_wr_features` both produce internal `(gsis_id, season, week, team, opponent, ...)` frames from `depth_charts` (filtered to RBs / WRs in `as_of_week`, deduped per the Plan 3b drift fixes) inner-joined with `schedules` (bye-week filter from the same Plan 3b drift). PR #29 wired `attach_weather_features(out, sch)` into both builders post-assembly. The integration reuses this seam — `attach_weather_features` joins on `(season, week, team)` from the index frame, returns 12 weather cols, then schema validation drops 4 v1 and keeps 8 refined.

### 2.3 Schedules contract

`schedules` must satisfy `SchedulesSchema` per the existing ingest. The `attach_weather_features` helper (per PR #28) trusts this contract; defensive normalization is not added. PR #30's `_compute_surface_onehot` raises `ValueError` if `surface` carries a code not in the pinned `_SURFACE_CODES` tuple — a future `nfl_data_py` upgrade adding a new surface code triggers a hard fail and forces a deliberate code review (intended; see PR #30 weather_features.py:78-87).

If `schedules` is empty, the helper's left-merge produces all-NaN rows on the 12 weather cols. Schema's `nullable=True` on all 8 refined cols accepts this.

---

## 3. Code shape

### 3.1 `attach_weather_features` (unchanged)

Public function in `src/projections/features/weather_features.py`, signature unchanged from PR #28:

```python
def attach_weather_features(
    index: pd.DataFrame,
    schedules: pd.DataFrame,
) -> pd.DataFrame:
    """Left-merge the twelve weather features onto a player-team-week index."""
```

Returns 12 weather cols (4 v1 + 8 refined). Row count preserved. No edits in this PR.

### 3.2 `build_rb_features` (unchanged)

`src/projections/features/rb.py:251` already calls:

```python
out = attach_weather_features(out, sch)
```

Followed at line 253 by:

```python
return RbFeaturesSchema.validate(out)
```

With `RbFeaturesSchema.Config.strict = "filter"` (schemas.py:674), the validate returns a frame with only the 8 refined weather cols (not the 4 v1) once Phase 1 swaps the schema. **No edits to rb.py in this PR.**

### 3.3 `build_wr_features` (unchanged)

Identical structure: `wr.py:289` calls `out = attach_weather_features(out, sch)`, `wr.py:291` returns `WrFeaturesSchema.validate(out)`. **No edits to wr.py in this PR.**

### 3.4 Schema swap

In `src/projections/schemas.py`:

**`WrFeaturesSchema` (lines 484–557):** delete the 5-line header comment + 4 col defs at lines 542–550 (the PR #28 / PR #29 v1 weather block). Insert in the same slot:

```python
# Weather features — refined-unit replace per PR #30 verdict (RB swap, WR
# swap+augment ADOPT under lgb-nb composite). Replaces the v1 4-col bundle
# from PR #29 (wind_speed_mph, is_high_wind, temperature_f,
# is_grass_surface) with the 8 refined cols below. Sourced from existing
# SchedulesSchema cols (temp, surface, kickoff). Domes filled (temp=70 →
# is_cold_weather=0; surface flags reflect the actual surface code with no
# override since stadia keep their playing surface across roof states;
# is_primetime is independent of roof). 2022 has a known coverage trough
# on is_cold_weather (~0.67 non-NaN per (position, season)) due to upstream
# NaN temp on outdoor games — nullable=True absorbs; pooled-row CI is
# symmetric across baseline/candidate dropna; documented in PR #30 audit.
is_cold_weather: Series[float] = pa.Field(ge=0, le=1, nullable=True)
is_a_turf: Series[float] = pa.Field(ge=0, le=1, nullable=True)
is_astroturf: Series[float] = pa.Field(ge=0, le=1, nullable=True)
is_fieldturf: Series[float] = pa.Field(ge=0, le=1, nullable=True)
is_grass: Series[float] = pa.Field(ge=0, le=1, nullable=True)
is_matrixturf: Series[float] = pa.Field(ge=0, le=1, nullable=True)
is_sportturf: Series[float] = pa.Field(ge=0, le=1, nullable=True)
is_primetime: Series[float] = pa.Field(ge=0, le=1, nullable=True)
```

**`RbFeaturesSchema` (lines 611–675):** identical edit — delete the 5-line header comment + 4 col defs at lines 663–671; insert the same 8 refined cols + same header comment block in the same slot.

`Config.strict = "filter"` and `Config.coerce = True` are unchanged on both schemas.

### 3.5 `baseline.py` hardcoded feature lists

In `src/projections/models/baseline.py`:

**`_WR_FEATURE_COLUMNS` (line 266):** delete the 4 v1 names at lines 297–300 along with the existing `# Weather features (PR #28...)` comment at lines 293–296. Insert in the same slot:

```python
# Weather features — refined-unit replace per PR #30 verdict (this PR's
# strict-replace integration). lightgbm derives feature lists from
# WrFeaturesSchema dynamically and auto-picks-up; baseline.py is hardcoded
# so must be updated explicitly. Same rule recurs at every per-position
# feature-list edit (PR #21, #26, #27, #29 each caught it once).
"is_cold_weather",
"is_a_turf",
"is_astroturf",
"is_fieldturf",
"is_grass",
"is_matrixturf",
"is_sportturf",
"is_primetime",
```

**`_RB_FEATURE_COLUMNS` (line 366):** identical edit at lines 393–400.

The `tests/test_models/test_baseline_feature_columns_match_schema.py` test verifies post-edit that `set(_<POS>_FEATURE_COLUMNS) == set(SCHEMA.columns) - identity` for all 4 positions; once Phase 1 (schema swap) and Phase 3 (`_<POS>_FEATURE_COLUMNS` swap) both land, the test passes. If only one lands, the test fails — structural enforcement of the invariant.

The lightgbm family auto-picks-up via dynamic schema derivation:
- `lightgbm.py:120`: `_RB_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(RbFeaturesSchema.to_schema().columns.keys())` — auto-syncs to the swapped schema.
- `lightgbm.py:122`: same for WR.
- `lightgbm_nb.py` and `lightgbm_tuned.py` import these from `lightgbm.py` (via `lightgbm_nb.py:45,51` / `lightgbm_tuned.py:25,31`).

**No edits to lightgbm files.** Phase 3 includes a smoke-test assertion that the dynamically-derived RB and WR lists post-schema-swap contain the 8 refined names and **do not** contain the 4 v1 names.

### 3.6 `weather_features.py` module docstring

Current (`src/projections/features/weather_features.py:1-12`):

```python
"""Weather feature computes for the weather refined-unit family probe (PR #28
broad-cut + this PR's refinements).

Sourced from `SchedulesSchema` columns (`wind`, `temp`, `roof`, `surface`,
`kickoff`) already in `data/raw/schedules`. Dome / closed-roof games are
filled per spec §3.5: a controlled environment has no weather, so wind=0 /
temp=70 is semantically correct, not "imputed missing."

Probe-only — features land in the override parquet, not in
`*FeaturesSchema`. Integration follow-up is conditional on the family-probe
verdict per `docs/superpowers/specs/2026-05-09-weather-refined-unit-probe-design.md`.
"""
```

Replace with:

```python
"""Weather feature computes for both the v1 broad-cut bundle (PR #28 probe →
PR #29 RB+WR integration) and the refined-unit bundle (PR #30 probe →
PR-#-this-spec strict-replace integration).

Sourced from `SchedulesSchema` columns (`wind`, `temp`, `roof`, `surface`,
`kickoff`) already in `data/raw/schedules`. Dome / closed-roof games are
filled per the original PR #28 spec §3.5: a controlled environment has no
weather, so `wind=0` / `temp=70` is semantically correct, not "imputed
missing."

`compute_weather_features` returns 12 cols: 4 v1 (`wind_speed_mph`,
`is_high_wind`, `temperature_f`, `is_grass_surface`) and 8 refined
(`is_cold_weather`, six surface one-hots, `is_primetime`). The v1 4-col
bundle was superseded in `RbFeaturesSchema` / `WrFeaturesSchema` by the
refined 8-col bundle in PR-#-this-spec; the helper still returns all 12
because `build_weather_overrides` and `scripts/build_weather_override.py`
consume the full set for any future probe re-run. Pandera's `strict="filter"`
on the production builders' schema validates filters the v1 cols at the
boundary.
"""
```

(The "PR-#-this-spec" placeholder gets filled in with the actual PR number once the PR is opened.)

### 3.7 Tests

In `tests/test_features/test_rb.py` and `tests/test_features/test_wr.py` (parallel additions in each file):

**Drop or rewrite the four PR #29 builder-boundary weather tests:**

- `test_<pos>_features_attach_weather_dome_fill` → drop. Replaced by `test_<pos>_features_attach_weather_refined_dome_fill`.
- `test_<pos>_features_attach_weather_outdoor_high_wind` → drop. The "high wind" axis isn't in the refined bundle (PR #30's 4 binding refinements are cold-weather, multi-class surface, primetime — wind threshold dropped per probe §1.2). The wind axis stays in `compute_weather_features`'s output but is filtered out at the schema boundary; helper-level wind tests stay in `test_weather_features.py`.
- `test_<pos>_features_attach_weather_grass_surface` → drop. Replaced by `test_<pos>_features_attach_weather_refined_surface_multiclass_<code>` (parametrized).
- `test_<pos>_features_attach_weather_bye_week_fallback` → rewrite. Same shape (synthetic player-week with no schedule join), assertions update to the 8 refined cols all NaN.

**Add the refined-bundle builder-boundary tests:**

- `test_<pos>_features_attach_weather_refined_dome_fill` — synthetic `roof="dome"` schedules row, `kickoff` 1pm ET (so `is_primetime=0`). Assert `is_cold_weather == 0.0` (falls out of dome's `temperature_f=70.0` fill, computed by `_compute_is_cold_weather`), all 6 surface flags reflect the actual surface code (e.g., `is_fieldturf=1.0` for a dome stadium with fieldturf), `is_primetime == 0.0`.
- `test_<pos>_features_attach_weather_refined_cold_outdoor` — synthetic outdoor `wind=10.0, temp=28.0, roof=NaN, surface="grass", kickoff=Sunday 1pm ET`. Assert `is_cold_weather == 1.0`, `is_grass == 1.0`, all other surface flags `0.0`, `is_primetime == 0.0`.
- `test_<pos>_features_attach_weather_refined_surface_multiclass_<code>` — parametrized across the 6 codes in `_SURFACE_CODES`. For each code, synthesize a schedules row with that surface; assert `is_<code> == 1.0` and all 5 other surface flags `0.0`. (Per-code coverage: when this test exits, every surface flag has been exercised at the builder-boundary level. PR #30's helper-level tests in `test_weather_features.py` already cover the lower-level case; this is the integration-flow check.)
- `test_<pos>_features_attach_weather_refined_primetime_kickoff` — three sub-cases on the same player-week with different kickoff timestamps. Sub-case 1: `kickoff` 8:20pm ET (TNF / SNF window) → `is_primetime == 1.0`. Sub-case 2: `kickoff` 1pm ET → `is_primetime == 0.0`. Sub-case 3: `kickoff` is `pd.NaT` → `is_primetime` is NaN.
- `test_<pos>_features_attach_weather_refined_bye_week_fallback` — synthetic player-week where the schedule join returns no row (player on a team with no schedule entry in `as_of_week`). Assert all 8 refined weather cols are NaN; assert schema validation passes (`nullable=True` on every refined col).

**Cluster-A defensive grep** (per §1.1):

```bash
grep -rn "wind_speed_mph" tests/   # canonical v1 col
```

In-scope sites: `tests/test_features/test_cache.py`, `tests/test_scripts/test_tune_lightgbm.py`, `tests/test_schemas/test_dataframe_schemas.py`. At each site that builds a synthetic minimal RB or WR features row, drop the 4 v1 defaults and add 8 refined defaults:

```python
"is_cold_weather": 0.0,
"is_a_turf": 0.0,
"is_astroturf": 0.0,
"is_fieldturf": 0.0,
"is_grass": 1.0,
"is_matrixturf": 0.0,
"is_sportturf": 0.0,
"is_primetime": 0.0,
```

(`is_grass=1.0` and the other 5 surface flags `0.0` so the multi-class one-hot row sums to 1, which is the realistic per-row state for a non-NaN surface.)

After the swap, re-run the grep. Expectation: only 2 files match (`tests/test_features/test_weather_features.py`, `tests/test_scripts/test_build_weather_override_cli.py`) — both are out-of-scope helper tests. Plan asserts this post-edit.

`tests/test_features/test_weather_features.py`: **no changes.** PR #30 already added per-helper unit tests for `_compute_is_cold_weather`, `_compute_surface_onehot`, `_compute_is_primetime`; those tests stay. The v1-col-related tests also stay (the helper still produces them).

`tests/test_scripts/test_build_weather_override_cli.py`: **no changes.** The CLI test asserts the override produces 12 cols.

`tests/test_models/test_baseline_feature_columns_match_schema.py`: **no changes**, but verify it still passes after Phase 1 + Phase 3 land. (If Phase 1 lands without Phase 3, this test catches the gap immediately.)

`tests/conftest.py`: **no changes.**

`tests/test_features/conftest.py`: **no changes.**

---

## 4. Real-data execution sequence (run-once, reports committed)

1. Code changes land (Phases 1–4 per §7) + tests pass + lint + typecheck clean (criterion §1.3.1).
2. Verify `data/raw/schedules/` is the post-`56df07f` corrected state. Per PR #30, the fix is on `main`; the working-tree-as-cut-from-main inherits it. If a future audit ever surfaces stale partitions, regenerate via `python -c "from projections.ingest.refresh import refresh; refresh(data_root=Path('data'), seasons=range(2018, 2025), only=['schedules'])"`.
3. `python scripts/refresh_features.py rb wr --seasons 2018-2024` — regenerates RB + WR feature caches against the swapped schema. Verify schema validation passes on every (season, week) partition for both positions (criterion §1.3.2). **Mandatory** because the existing caches were written with v1 cols; the schema swap rejects them.
4. **Coverage cross-check** (criterion §1.3.5 evidence): per-(position, season) NaN rate on the 8 refined cols for the eval window 2021–2024. Should approximate PR #30's per-(position, season) coverage from `feature_probe_weather_refined_override_audit.md`:
   - `is_cold_weather`: 2021 ~98%, 2022 ~67%, 2023 ~85%, 2024 ~98% non-NaN.
   - 6 surface one-hots: pooled ~98% non-NaN; 2.17% per-row NaN rate from rows with unparseable / unknown surface codes.
   - `is_primetime`: 100% non-NaN per (position, season) post-`56df07f`.
   If coverage is materially different (>5pp shift on any cell), the schema swap or builder wiring is wrong — investigate before running the gate.
5. `python scripts/backtest.py --position RB,WR --model baseline,lightgbm-nb --update-snapshot` — runs the walk-forward backtest on RB + WR × `baseline + lgb-nb` on holdout years 2021–2024. Captures per-row prediction frames (criterion §1.3.3). 4 cells total. Snapshot updates committed: `tests/backtest/model_metrics.json` rows for these 4 cells; ensemble weight files (`data/ensemble_weights/ensemble_{rb,wr}_*.json`) are *not* expected to regen (§1.3.4 explicitly skips ensemble; the existing weights remain valid for the v1-column ensemble snapshot, which is now out of date but isolated — flagged in §5 Risks).

   Per PR #29's spec: the dual-run gate may need to subprocess `_run_single_backtest.py` per model class to avoid the schema-import-caching trap (the in-memory `RbFeaturesSchema` / `WrFeaturesSchema` doesn't refresh after a `git checkout main -- src/projections/schemas.py` within a single Python process). The implementation plan front-loads this and routes through the per-model-class dual-run pattern from `scripts/backtest_dual.py`.
6. `python scripts/adoption_gate.py --position RB,WR --baseline-run <pre-pr-sha> --candidate-run <branch-sha>` — produces per-(model_class, position) verdicts for the 4 cells. Output: `reports/adoption_gate_weather_refined_rb_wr.{md,csv}`. Commit. (If the gate exposes a `--coverage-threshold` flag and rejects a default of 0.95, match probe's `0.90` per the `is_cold_weather` 2022 trough.)
7. Write `reports/weather_refined_rb_wr_summary.md` with: probe-predicted (-0.0088 RB / -0.0050 WR via swap; -0.0051 WR via augment as informational) vs gate-measured magnitudes per position; per-(model_class, position) verdicts (4 cells); coverage statistics from step 4; binding-cell shift rationale; ship/revert decision per position per §1.3.5; explicit deferred-follow-up note for the cross-class production routing question per position. Commit.
8. **For each position independently per §1.3.5:**
   - **`(lgb-nb, POS)` ADOPT AND `(baseline, POS)` not REGRESSION** → ship-as-designed: keep schema swap + `_<POS>_FEATURE_COLUMNS` swap.
   - **`(lgb-nb, POS)` ADOPT AND `(baseline, POS)` REGRESSION** → full-revert that position (schema swap + `_<POS>_FEATURE_COLUMNS` swap unwound). Re-run that position × both model classes' backtest cells (4 rows in `tests/backtest/model_metrics.json`).
   - **`(lgb-nb, POS)` MARGINAL or DO_NOT_ADOPT** → full-revert that position. Re-run that position × both model classes' backtest cells.
9. **Update `project_management.md`** with a top-of-file decision-log entry (template: PR #29's entry). **Update `TODO.md` #25** with measured magnitudes per position and the per-position shipped state. Refresh per-position cross-class production-routing follow-up note (PR #29 added it for v1 weather; this PR updates it to refined weather).
10. Push branch + open PR.

---

## 5. Risks

- **Probe-vs-gate verdict divergence per position.** PR #30's binding magnitudes are -0.0088 (RB swap) and -0.0050 (WR swap) — the latter is the smallest binding-cell magnitude in Track 2A history, just inside the per-cell noise floor of ~0.001-0.002 fpts. A small calibration error could flip WR's lgb-nb cell to MARGINAL or DO_NOT_ADOPT. Per-position revert path defined in §1.3.5. RB has more headroom (-0.0088 with CI hi -0.0030) so flip risk is lower.

- **Probe-vs-gate magnitude divergence.** Track record of probe → gate calibration: PR #20 → #21 matched to 4 decimals (RB v1 PBP); PR #25 → #26 within ~0.004 fpts (WR trajectory); PR #25 → #27 within ~0.0017 fpts (TE trajectory); PR #28 → #29 within ~0.0004-0.0006 fpts on both binding cells (RB v1 -0.0081 → -0.0077; WR v1 -0.0110 → -0.0104). A ~10% calibration error at WR's -0.0050 magnitude is ~0.0005 fpts — within the per-cell noise floor and the historical track record. The §1.3.5 rule binds on the verdict label, not magnitude.

- **`(baseline, *)` REGRESSION risk.** PR #30 found 0/120 SIGNAL Phase-1 cells on baseline across both modes for all 4 positions — the strongest "no signal on baseline" evidence to date. Baseline composite REGRESSION (CI strictly above 0) on this PR's gate is unlikely but not zero. The §1.3.5 full-revert path covers it (modified-shape doesn't apply to a strict-replace, per the matrix note). The recurring **QB augment regression** (now sharper at +0.0099 fpts in PR #30 — CI strictly above 0) is the canary: if the same overfit pattern manifests on baseline RB or WR despite probe evidence to the contrary, the modified-shape branch is unavailable and the position fully reverts. The deferred QB / TE refined-unit work would not be revisited here.

- **`baseline.py:_<POS>_FEATURE_COLUMNS` miss / drift.** PR #21 / #26 / #27 / #29 each caught the same gap. PR #29 added the parametrized regression test `tests/test_models/test_baseline_feature_columns_match_schema.py` to structurally enforce the invariant. This PR's Phase 1 and Phase 3 must both land for the test to pass — Phase 1 alone (schema swap without baseline.py swap) makes the test fail. The implementation plan's task ordering ensures Phase 3 lands before any pre-commit. **Net of PR #29's regression test, this risk class is now structurally addressed**; the test catches it on every commit.

- **Coverage shift between probe override and production builder.** The probe override at `data/features_probe/weather.parquet` was built from the same `attach_weather_features` helper this PR's production builders call (already in place from PR #29). Coverage should match almost exactly. Step 4 of §4 cross-checks builder-output coverage against the probe audit before running the gate. Materially different coverage (>5pp shift) signals a builder wiring bug or a stale schedule partition — fix before gate.

- **Cluster-A test fixture leftovers — straightforward, but the swap is bigger than PR #29's add.** PR #29's defensive grep had 4 col defaults to add at each site. This PR's swap drops 4 v1 and adds 8 refined — twice the line-count delta per site. Front-loaded grep + post-edit grep verification per §1.1. Sites are enumerated explicitly: `tests/test_features/test_cache.py`, `tests/test_scripts/test_tune_lightgbm.py`, `tests/test_schemas/test_dataframe_schemas.py`. The 2 helper-test files (`tests/test_features/test_weather_features.py`, `tests/test_scripts/test_build_weather_override_cli.py`) intentionally stay v1-aware.

- **Feature cache invalidation for both RB and WR.** Same as PR #29 / PR #21 / PR #26 / PR #27. The schema swap rejects old cache rows that have v1 cols. §4 step 3 calls this out and runs the refresh explicitly before any backtest invocation that reads the cache.

- **EnsembleModel weight regen — limited blast radius, but staler than PR #29's case.** EnsembleModel pulls from C-NB internally per Plan 6. The lgb-nb backtest cell change (RB and WR) means `data/ensemble_weights/ensemble_{rb,wr}_*.json` weight files would regen IF EnsembleModel were re-run as part of the snapshot. §1.3.4 explicitly skips ensemble; this PR does NOT re-run EnsembleModel. PR #29 did the same — so the existing ensemble weight files for RB / WR currently reflect the v1-weather lgb-nb predictions; after this PR they reflect a state two integration-PRs stale (no v1, no refined either, since ensemble fit ran before PR #29 wired weather at all — or possibly v1-weather depending on the post-PR #29 weight cycle). Acceptable per Plan 6; flagged for any future ensemble-related work to refit before committing to a routing flip.

- **Schema-import-caching during dual-run gate.** Per PR #29's spec gap caught (`scripts/backtest_dual.py` orchestration): a single Python process holds the in-memory `RbFeaturesSchema` / `WrFeaturesSchema` from import time; running `_run_single_backtest.py` for the candidate after `git checkout main -- src/projections/schemas.py` for the baseline reads stale schema state. PR #29's fix was to subprocess each per-model-class run; same subprocess pattern applies here. The implementation plan's Phase 5 calls this out as a known trap with the PR #29 workaround pre-applied.

- **`_compute_surface_onehot` ValueError on a new upstream surface code.** PR #30 pinned `_SURFACE_CODES` to the 6 codes present in 2018–2024 schedules. A future `nfl_data_py` upgrade adding a new code (e.g., a new turf brand) would raise `ValueError` from `_compute_surface_onehot`. The error is intended as a hard fail forcing deliberate pin update; not silent dropout. The opt-in `--run-network` smoke at `tests/test_ingest/test_api_drift.py` doesn't cover surface-code drift directly but would catch the schema-level column changes. Risk is post-merge, not blocking this PR.

- **Recurring QB augment regression — guardrail only.** PR #30's QB augment composite was +0.0099 fpts (CI strictly above 0), the sharpest QB augment regression in Track 2A history. QB is not in this PR's scope. The summary report flags it as a guardrail per the rule "do not extend `QbFeaturesSchema` with weather features without independent evidence."

- **Bigger swap = bigger blast radius if a fix is needed mid-PR.** The schema swap removes 4 cols and adds 8; if a Phase 1–4 commit breaks the test suite in a way that needs partial revert, the recovery path (revert just the 4 v1 deletes? revert just the 8 refined adds?) is more nuanced than PR #29's "revert the 4-col addition." Implementation plan's per-task structure makes the schema swap a single atomic edit.

- **Module docstring is the most-likely-stale part of the codebase.** The current `weather_features.py` docstring still says "Probe-only" — stale since PR #29. This PR updates it to reflect both bundles integrated. Implementation plan's Phase 4 includes the docstring rewrite as a single edit. If a future bundle (e.g., refined-unit-of-refined-unit per TODO #25) lands, the docstring needs another update; flagged for that future PR.

---

## 6. Documentation updates on merge

- **`project_management.md`:** append a top-of-file decision-log entry. Title: "Weather Refined-Unit RB+WR Integration — verdicts: RB <verdict>, WR <verdict>". Format matches PR #29's entry — title, status, verdict, what shipped or reverted per position, magnitude, probe-vs-gate calibration note per position, plus an explicit "second strict-replace integration after PR #29 v1 weather; first to swap rather than extend schemas" section. Cross-link PR #30 (probe), PR #29 (v1), and this PR.
- **`TODO.md` #25:** record the production integration outcome per position (shipped / reverted, with measured magnitude). Cross-reference the summary report and PR #30's probe entry. Note: refined-unit broad-cut closed for RB and WR ADOPT cells from PR #30; QB and TE remain DO_NOT_ADOPT (now a sharper "QB augment regression" guardrail per PR #30); refined-unit-of-refined-unit candidates remain open under TODO #25 (continuous `kickoff_hour_et`, `is_london`, surface × position interactions, per-team weather acclimation, precipitation, wind direction). None queued.
- **(Cross-class production routing follow-up — per position):** update the existing TODO #25 sub-note that PR #29 added with the refined-cols-in-schema state. New text: "RB / WR production routes to `baseline` per Plan 8 (2026-04-29). With *refined* weather cols now in `RbFeaturesSchema` / `WrFeaturesSchema` (post-this-PR), a cross-class re-eval (`scripts/adoption_gate.py --baseline-run <pre-PR> --candidate-run <post-PR> --position {RB|WR}` comparing `lightgbm-nb` candidate to `baseline` baseline) could justify flipping `_PositionDispatch[{RB|WR}].default_model_class` to `lightgbm-nb`. Not load-bearing for any current consumer; queue alongside the next {RB|WR}-related work."
- **`docs/superpowers/specs/2026-05-09-weather-refined-unit-probe-design.md`:** no changes. The probe spec stays as historical record.
- **`docs/superpowers/specs/2026-05-08-weather-features-rb-wr-design.md`:** no changes. PR #29's spec stays as historical record.
- **`CONTRIBUTING.md`:** no changes. PR #28 / #29 / #30 each touched their respective override/integration subsections; the override path is unchanged in this PR; the production builder path is unchanged in code (only schema names change).

---

## 7. Implementation phasing

The implementation plan should structure work in phases per the CLAUDE.md "PHASED EXECUTION" rule (≤5 files per phase). Suggested phasing:

- **Phase 1 — Schema swap (1 file).** `schemas.py` (drop 4 v1 + add 8 refined cols in `RbFeaturesSchema`; identical edit in `WrFeaturesSchema`). Verify (criterion §1.3.1 partial): `pytest -v -k "schemas"` passes; `RbFeaturesSchema.validate(...)` and `WrFeaturesSchema.validate(...)` accept frames with 8 refined cols and reject frames missing them. The full-suite `pytest -v` will still fail at this point because the existing tests reference v1 cols and the schema regression test (`tests/test_models/test_baseline_feature_columns_match_schema.py`) will fail until Phase 3 lands — that's expected and tracked.

- **Phase 2 — `weather_features.py` module docstring (1 file).** Update the module docstring per §3.6. No code change. Verify: `pytest -v` does not regress (the docstring is metadata only). `mypy src tests` clean.

- **Phase 3 — `_<POS>_FEATURE_COLUMNS` swap (1 file).** `models/baseline.py` (drop 4 v1 + add 8 refined names in `_RB_FEATURE_COLUMNS`; identical edit in `_WR_FEATURE_COLUMNS`). Verify: `tests/test_models/test_baseline_feature_columns_match_schema.py` passes for both RB and WR (the QB and TE entries don't change since QB/TE schemas are untouched). Smoke test (1-line asserts in a new unit test or a temporary REPL check): the dynamically-derived `_RB_FEATURE_COLUMNS` and `_WR_FEATURE_COLUMNS` from `lightgbm.py` post-swap contain `is_cold_weather`, `is_a_turf`, …, `is_primetime` and **do not** contain `wind_speed_mph`, `is_high_wind`, `temperature_f`, `is_grass_surface`. Once Phase 1 + Phase 3 both land, the schema regression test passes; before that it fails.

- **Phase 4 — Tests + cluster-A leftover fixtures (≤5 files).** Drop or rewrite the 4 PR #29 weather tests in `tests/test_features/test_rb.py` and `tests/test_features/test_wr.py`; add the 5 refined-bundle tests in each (10 new tests across 2 files). Cluster-A grep for `wind_speed_mph` and add 8 refined defaults / drop 4 v1 defaults at every in-scope site (`tests/test_features/test_cache.py`, `tests/test_scripts/test_tune_lightgbm.py`, `tests/test_schemas/test_dataframe_schemas.py`). Verify: full pytest suite passes; mypy + ruff clean. Post-edit grep confirms only the 2 helper-test files match `wind_speed_mph`.

- **Phase 5 — Real-data execution + reports (no code).** §4 steps 2–7. Output: refreshed RB + WR caches, backtest snapshot delta (`tests/backtest/model_metrics.json` rows for the 4 cells), adoption gate report (`reports/adoption_gate_weather_refined_rb_wr.{md,csv}`), summary report (`reports/weather_refined_rb_wr_summary.md`). The 4-cell gate result determines which §1.3.5 branch fires for each position.

- **Phase 6 — Conditional code adjustments + documentation (1–5 files).** Branches per position independently per §1.3.5. Possible per-position outcomes:
  - **Both positions ship-as-designed** (default branch, expected from probe evidence): no code adjustments. Update `project_management.md` + `TODO.md` per §6 (2 files).
  - **One or both positions full-revert** (`(baseline, POS)` REGRESSION OR `(lgb-nb, POS)` MARGINAL/DO_NOT_ADOPT): revert the schema swap in `schemas.py`, the `_<POS>_FEATURE_COLUMNS` swap in `baseline.py`, and the cluster-A fixture changes (or keep them — defensible either way; cheap to leave). Re-run that position × both model classes' backtest cells (4 rows). Update `project_management.md` + `TODO.md` documenting the divergence. 4-5 files total per affected position.

This phasing keeps each step ≤5 files. Phase 1 will produce test failures until Phases 2–4 land — that's expected, tracked, and the implementation plan resolves it within a single sequenced commit chain. Phase 6's per-position conditional structure keeps the implementation plan's per-task definition crisp — only the actual outcomes execute (typically: both positions ship-as-designed, no Phase 6 code adjustments needed).
