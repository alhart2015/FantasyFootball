# Weather Refined-Unit RB+WR Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the v1 4-col weather bundle with the refined 8-col bundle in `RbFeaturesSchema` + `WrFeaturesSchema` (and the matching `_<POS>_FEATURE_COLUMNS` in `baseline.py`); run the dual-run adoption gate on `(baseline + lightgbm-nb) × (RB + WR)`; ship per the §1.3.5 contingency.

**Architecture:** Schema-level swap. `attach_weather_features` (already wired into both builders by PR #29) returns 12 weather cols; pandera's `strict="filter"` on each builder's terminal `SCHEMA.validate(out)` filters down to the 8 schema cols. No builder code changes; the only code edits are `schemas.py` (drop 4 v1, add 8 refined per position) and `baseline.py` (matching swap in `_RB_FEATURE_COLUMNS` + `_WR_FEATURE_COLUMNS`). Tests + cluster-A fixtures need parallel updates. Real-data execution reuses PR #29's `scripts/backtest_dual.py` orchestration unchanged.

**Tech Stack:** Python 3.11, pandera, pandas, lightgbm, pytest, mypy, ruff. PR #29's `scripts/backtest_dual.py` + `scripts/_run_single_backtest.py` for the dual-run gate orchestration.

**Spec:** `docs/superpowers/specs/2026-05-09-weather-refined-rb-wr-design.md`.

**Branch:** `feat/weather-refined-rb-wr` (worktree: `.worktrees/feat-weather-refined-rb-wr/`). Cut from `main` at `51e61f5`.

---

## Phase shape

The spec lays out 6 phases. Plan structures these as 10 tasks. Each task is a single commit unless noted. Between tasks 1–6, the test suite is in a known broken state (the schema swap leaves stale references in tests/fixtures); Tasks 1 → 6 form a sequenced commit chain that ends in a clean pytest. Task 6 is the "verify everything is green" gate.

| Task | Phase | Files | Test state after task |
|---|---|---|---|
| 1 | Phase 1 | `src/projections/schemas.py` | Broken: schema-regression test fails (RB+WR), fixture-driven tests fail, builder-boundary v1 weather tests fail. |
| 2 | Phase 3 | `src/projections/models/baseline.py` | Schema-regression test passes; fixture + builder tests still broken. |
| 3 | Phase 4a | `tests/test_features/test_cache.py` | test_cache passes; other fixture + builder tests still broken. |
| 4 | Phase 4a | `tests/test_scripts/test_tune_lightgbm.py` | tune_lightgbm passes; remaining schema/builder tests broken. |
| 5 | Phase 4a | `tests/test_schemas/test_dataframe_schemas.py` | All cluster-A fixture sites fixed; only builder-boundary tests broken. |
| 6 | Phase 4b | `tests/test_features/test_rb.py`, `tests/test_features/test_wr.py` | Full suite passes (modulo Phase 4c lint/format). |
| 7 | Phase 2 + 4c | `src/projections/features/weather_features.py` | No new test impact; full suite still passes. Run `pytest -v` + `mypy src tests` + `ruff check src tests` + `ruff format --check src tests` to verify. |
| 8 | Phase 5a | `data/features/{rb,wr}/...` (gitignored) + coverage cross-check note | Caches refreshed against the refined schema. |
| 9 | Phase 5b | `tests/backtest/model_metrics.json` | Snapshot regenerated for `(RB,WR) × (baseline,lightgbm-nb)`. |
| 10 | Phase 5c + 6 | `reports/adoption_gate_weather_refined_*.{md,csv}`, `reports/weather_refined_rb_wr_summary.md`, `project_management.md`, `TODO.md` | Reports + decision log committed. Per-position §1.3.5 contingency applied. |

**Conditional revert task** (only if §1.3.5 fires): see §"Phase 6 conditional contingency" near the end of this plan.

---

## Task 1: Phase 1 — Schema swap (RbFeaturesSchema + WrFeaturesSchema)

**Files:**
- Modify: `src/projections/schemas.py:484-558` (WrFeaturesSchema), `src/projections/schemas.py:611-676` (RbFeaturesSchema)

**Context:** Each schema currently carries the 4 v1 weather cols inserted by PR #29. The swap drops them and adds 8 refined cols (`is_cold_weather`, six surface one-hots, `is_primetime`). `Config.strict = "filter"` and `Config.coerce = True` are unchanged on both schemas. After this task, the lightgbm family's dynamically-derived feature lists (`lightgbm.py:120,122`) auto-pick-up the swap; baseline.py's hardcoded lists do not — Task 2 fixes that.

- [ ] **Step 1: Re-read both schema blocks before editing.**

  Run (verifies line numbers haven't drifted; CLAUDE.md context-decay rule):
  ```bash
  sed -n '540,560p' src/projections/schemas.py
  sed -n '660,680p' src/projections/schemas.py
  ```

  Expected: see the PR #29 weather block at lines 542–550 (WR) and 663–671 (RB), preceded in each case by a 5-line `# Weather features (PR #28 family probe + 2026-05-08 RB+WR integration` comment.

- [ ] **Step 2: Edit `WrFeaturesSchema` weather block.**

  In `src/projections/schemas.py`, replace lines 542–550 (5-line header comment + 4 v1 col defs) with:

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

  (Indentation: 4 spaces — class body. Match the surrounding cols.)

- [ ] **Step 3: Edit `RbFeaturesSchema` weather block.**

  Replace lines 663–671 (5-line header comment + 4 v1 col defs) with the **identical** 12-line block from Step 2.

- [ ] **Step 4: Re-read both blocks to verify the edits applied correctly.**

  CLAUDE.md edit-integrity rule. Run:
  ```bash
  sed -n '540,565p' src/projections/schemas.py
  sed -n '660,685p' src/projections/schemas.py
  ```
  Expected: each block now shows the refined header comment + 8 refined col defs followed by the unchanged `class Config:` block.

- [ ] **Step 5: Run targeted schema tests.**

  ```bash
  PYTHONPATH=src pytest tests/test_schemas/ -v -k "wr_features or rb_features" --no-cov
  ```

  Expected: **failing tests in `test_dataframe_schemas.py`** that explicitly assert presence of v1 cols (3 sites — see Task 5). Schema-class-existence tests still pass; cols-set-membership tests fail. **This is expected** and is fixed in Task 5.

- [ ] **Step 6: Confirm the schema-regression test fails for RB and WR.**

  ```bash
  PYTHONPATH=src pytest tests/test_models/test_baseline_feature_columns_match_schema.py -v --no-cov
  ```

  Expected: 2 of the 4 parametrized cases fail (`[RB-...]` and `[WR-...]`); QB and TE pass; identity sanity check passes. The failure messages name the missing cols (`is_cold_weather`, `is_a_turf`, …, `is_primetime`) and the extras (`wind_speed_mph`, `is_high_wind`, `temperature_f`, `is_grass_surface`). **This is expected** — Task 2 fixes it.

- [ ] **Step 7: Commit.**

  ```bash
  git add src/projections/schemas.py
  git commit -m "$(cat <<'EOF'
  feat(schemas): swap v1 weather cols for refined 8-col bundle on Rb/WrFeaturesSchema

  Replaces v1 4-col weather bundle (wind_speed_mph, is_high_wind,
  temperature_f, is_grass_surface) with the 8 refined cols
  (is_cold_weather, six surface one-hots, is_primetime) per PR #30
  verdict. Phase 1 of the strict-replace integration; Phase 3 will swap
  the matching baseline.py per-position feature lists. Schema strict=filter
  on the builder boundary handles the col set automatically without
  builder code changes.

  Schema-regression test (test_baseline_feature_columns_match_schema.py)
  fails for RB+WR until Phase 3 lands; cluster-A fixture tests fail
  until Phase 4a; builder-boundary weather tests fail until Phase 4b.
  Expected; documented in plan.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 2: Phase 3 — `_<POS>_FEATURE_COLUMNS` swap in baseline.py

**Files:**
- Modify: `src/projections/models/baseline.py:266-301` (`_WR_FEATURE_COLUMNS`), `src/projections/models/baseline.py:366-401` (`_RB_FEATURE_COLUMNS`)

**Context:** Both per-position feature tuples currently end with the 4 v1 weather names appended by PR #29 (lines 295–300 for WR, 392–400 for RB). Replace those names with the 8 refined names. lightgbm family auto-syncs via dynamic schema derivation (`lightgbm.py:120,122` → `tuple(<Schema>.to_schema().columns.keys())`); only baseline.py needs explicit edits.

- [ ] **Step 1: Re-read both `_<POS>_FEATURE_COLUMNS` blocks.**

  ```bash
  sed -n '266,302p' src/projections/models/baseline.py
  sed -n '366,402p' src/projections/models/baseline.py
  ```

  Expected: see PR #29's weather entries at the end of each tuple, preceded by a 2-line (`_WR_FEATURE_COLUMNS`) or 4-line (`_RB_FEATURE_COLUMNS`) comment.

- [ ] **Step 2: Edit `_WR_FEATURE_COLUMNS`.**

  In `src/projections/models/baseline.py`, replace lines 295–300 (the 2-line comment + 4 v1 names) with:

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

  (Trailing comma after `"is_primetime"` is required because the closing `)` of the tuple is on the next line.)

- [ ] **Step 3: Edit `_RB_FEATURE_COLUMNS`.**

  Replace lines 392–400 (the 4-line comment + 4 v1 names) with the **identical** 13-line block from Step 2 (the 5-line comment is the same; the 8 col-name strings are the same).

- [ ] **Step 4: Re-read both edited blocks to verify.**

  ```bash
  sed -n '266,310p' src/projections/models/baseline.py
  sed -n '366,410p' src/projections/models/baseline.py
  ```

  Expected: each tuple ends with `"is_primetime",` followed by `)`.

- [ ] **Step 5: Run the schema-regression test.**

  ```bash
  PYTHONPATH=src pytest tests/test_models/test_baseline_feature_columns_match_schema.py -v --no-cov
  ```

  Expected: all 4 parametrized cases pass + identity sanity check passes.

- [ ] **Step 6: Smoke-check the lightgbm dynamic feature list.**

  ```bash
  PYTHONPATH=src python -c "
  from projections.models.lightgbm import _RB_FEATURE_COLUMNS as RB, _WR_FEATURE_COLUMNS as WR
  refined = {'is_cold_weather', 'is_a_turf', 'is_astroturf', 'is_fieldturf', 'is_grass', 'is_matrixturf', 'is_sportturf', 'is_primetime'}
  v1 = {'wind_speed_mph', 'is_high_wind', 'temperature_f', 'is_grass_surface'}
  assert refined.issubset(RB) and refined.issubset(WR), 'refined cols missing from lightgbm dynamic list'
  assert v1.isdisjoint(RB) and v1.isdisjoint(WR), 'v1 cols leaked into lightgbm dynamic list'
  print('OK')
  "
  ```

  Expected: prints `OK` and exits 0.

- [ ] **Step 7: Commit.**

  ```bash
  git add src/projections/models/baseline.py
  git commit -m "$(cat <<'EOF'
  feat(baseline): swap v1 weather names for refined 8-col bundle in _<POS>_FEATURE_COLUMNS

  Mirrors Task 1's schema swap: drop wind_speed_mph / is_high_wind /
  temperature_f / is_grass_surface from _RB_FEATURE_COLUMNS and
  _WR_FEATURE_COLUMNS; add is_cold_weather / six surface one-hots /
  is_primetime. lightgbm family auto-syncs via dynamic schema
  derivation; baseline.py is hardcoded so this swap is explicit.

  After this commit, tests/test_models/test_baseline_feature_columns_match_schema.py
  passes for all 4 positions. cluster-A fixture tests + builder-boundary
  tests still fail until Phases 4a + 4b.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 3: Phase 4a (1/3) — Cluster-A fixture swap in `test_cache.py`

**Files:**
- Modify: `tests/test_features/test_cache.py:54-57` (single minimal-row dict with 4 v1 defaults)

**Context:** This file builds a single synthetic minimal-row dict for the cache test. It currently has 4 v1 weather defaults at lines 54–57. Drop those; add 8 refined defaults.

- [ ] **Step 1: Re-read the affected block.**

  ```bash
  sed -n '40,80p' tests/test_features/test_cache.py
  ```

  Expected: see the dict literal containing `"wind_speed_mph": 8.0,` etc. at lines 54–57. Note the surrounding context (likely a per-position feature row used to test cache round-trip).

- [ ] **Step 2: Replace the 4 v1 default lines with 8 refined defaults.**

  Replace lines 54–57:
  ```python
          "wind_speed_mph": 8.0,
          "is_high_wind": 0.0,
          "temperature_f": 60.0,
          "is_grass_surface": 0.0,
  ```
  with:
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

  (`is_grass=1.0` and the 5 other surface flags `0.0` so the multi-class one-hot row sums to 1, matching realistic per-row state for a non-NaN surface.)

- [ ] **Step 3: Re-read to verify.**

  ```bash
  sed -n '40,80p' tests/test_features/test_cache.py
  ```

  Expected: the dict literal now has 8 refined cols where the 4 v1 cols used to be.

- [ ] **Step 4: Run the cache tests.**

  ```bash
  PYTHONPATH=src pytest tests/test_features/test_cache.py -v --no-cov
  ```

  Expected: all tests pass.

- [ ] **Step 5: Commit.**

  ```bash
  git add tests/test_features/test_cache.py
  git commit -m "$(cat <<'EOF'
  test(cache): swap v1 weather defaults for refined 8-col defaults in fixture

  Cluster-A fixture site 1 of 3. Drops wind_speed_mph / is_high_wind /
  temperature_f / is_grass_surface defaults from the synthetic minimal-row
  dict; adds is_cold_weather / six surface one-hots / is_primetime
  (is_grass=1.0, others 0.0 so multi-class one-hot sums to 1).

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 4: Phase 4a (2/3) — Cluster-A fixture swap in `test_tune_lightgbm.py`

**Files:**
- Modify: `tests/test_scripts/test_tune_lightgbm.py:60-66, 120-145` (two sites: feature-list tuple + synthetic frame builder)

**Context:** This file has two sites referencing the v1 weather cols:
1. A feature-list tuple at lines 60–66 enumerating expected columns.
2. A synthetic random-data frame builder at lines 120–145 that constructs random values for each weather col.

- [ ] **Step 1: Re-read both sites.**

  ```bash
  sed -n '55,70p' tests/test_scripts/test_tune_lightgbm.py
  sed -n '115,150p' tests/test_scripts/test_tune_lightgbm.py
  ```

  Expected at first site (~60–66): a tuple containing `"wind_speed_mph"`, `"is_high_wind"`, `"temperature_f"`, `"is_grass_surface"` listed sequentially.

  Expected at second site (~120–145): comments + `cols = (...)` containing the 4 v1 names (lines ~127–130), and 4 corresponding `df["<col>"] = rng.<distribution>(...)` assignments (lines ~140–143).

- [ ] **Step 2: Replace the feature-list tuple entries.**

  In the first site (line 63–66 region), replace:
  ```python
      "wind_speed_mph",
      "is_high_wind",
      "temperature_f",
      "is_grass_surface",
  ```
  with:
  ```python
      "is_cold_weather",
      "is_a_turf",
      "is_astroturf",
      "is_fieldturf",
      "is_grass",
      "is_matrixturf",
      "is_sportturf",
      "is_primetime",
  ```

- [ ] **Step 3: Replace the second-site `cols = (...)` and the `df[...]` assignments.**

  Replace the comment block at lines ~122–124 (`# Weather cols need bounded / categorical values …`) and the tuple at ~127–130 with:

  ```python
      # Weather refined cols are all 0/1 boolean indicators (is_cold_weather,
      # six surface one-hots, is_primetime). The schema is bypassed here, but
      # tune_lightgbm fits on these values so they must be in-range.
      cols = (
          "is_cold_weather",
          "is_a_turf",
          "is_astroturf",
          "is_fieldturf",
          "is_grass",
          "is_matrixturf",
          "is_sportturf",
          "is_primetime",
      )
  ```

  Then replace the 4 `df["<v1_col>"] = ...` assignments at lines ~140–143 with 8 assignments:

  ```python
      df["is_cold_weather"] = rng.integers(0, 2, size=len(df)).astype(np.float64)
      # Surface one-hot: pick one code per row, set that flag to 1.0, others to 0.0.
      surface_idx = rng.integers(0, 6, size=len(df))
      surface_cols = ("is_a_turf", "is_astroturf", "is_fieldturf", "is_grass", "is_matrixturf", "is_sportturf")
      for i, col in enumerate(surface_cols):
          df[col] = (surface_idx == i).astype(np.float64)
      df["is_primetime"] = rng.integers(0, 2, size=len(df)).astype(np.float64)
  ```

- [ ] **Step 4: Re-read both sites to verify.**

  ```bash
  sed -n '55,75p' tests/test_scripts/test_tune_lightgbm.py
  sed -n '115,155p' tests/test_scripts/test_tune_lightgbm.py
  ```

  Expected: first site tuple has 8 refined names; second site has the comment + 8-name `cols` tuple + the 8 assignments above.

- [ ] **Step 5: Run the tune_lightgbm tests.**

  ```bash
  PYTHONPATH=src pytest tests/test_scripts/test_tune_lightgbm.py -v --no-cov
  ```

  Expected: all tests pass.

- [ ] **Step 6: Commit.**

  ```bash
  git add tests/test_scripts/test_tune_lightgbm.py
  git commit -m "$(cat <<'EOF'
  test(tune_lightgbm): swap v1 weather defaults for refined 8-col defaults at fixture sites

  Cluster-A fixture site 2 of 3. Two sites in this file reference v1
  weather cols: a feature-list tuple and a synthetic random-data frame
  builder. Both updated to the refined 8-col bundle. Surface one-hot
  generated as one-per-row (random index in [0,6) → that flag 1.0,
  others 0.0).

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 5: Phase 4a (3/3) — Cluster-A fixture swap in `test_dataframe_schemas.py`

**Files:**
- Modify: `tests/test_schemas/test_dataframe_schemas.py:420-475, 575-630` (4 sites total: 2 RB-schema + 2 WR-schema validation tests)

**Context:** This file has 4 sites referencing the v1 weather cols, each in a `pd.DataFrame({...})` literal that exercises schema validation. From Step 0 grep: lines 423–426, 470–473, 581–584, 624–627.

- [ ] **Step 1: Re-read all 4 sites.**

  ```bash
  sed -n '418,478p' tests/test_schemas/test_dataframe_schemas.py
  sed -n '578,632p' tests/test_schemas/test_dataframe_schemas.py
  ```

  Expected: each site is a `pd.DataFrame({"<col>": [<val>], ...})` literal containing 4 v1 weather cols sequentially: `"wind_speed_mph": [8.0], "is_high_wind": [0.0], "temperature_f": [65.0], "is_grass_surface": [1.0],`.

- [ ] **Step 2: Replace each of the 4 sites.**

  At each of the 4 sites, replace the 4 v1 col entries:
  ```python
      "wind_speed_mph": [8.0],
      "is_high_wind": [0.0],
      "temperature_f": [65.0],
      "is_grass_surface": [1.0],
  ```
  with the 8 refined col entries:
  ```python
      "is_cold_weather": [0.0],
      "is_a_turf": [0.0],
      "is_astroturf": [0.0],
      "is_fieldturf": [0.0],
      "is_grass": [1.0],
      "is_matrixturf": [0.0],
      "is_sportturf": [0.0],
      "is_primetime": [0.0],
  ```

  Use Edit's `replace_all=True` if the 4-line pattern is byte-identical across all 4 sites. Verify post-edit count matches: 4 sites changed; total line delta is `+16` (8 added × 4 sites − 4 removed × 4 sites = 32 − 16 = 16) before pre-commit auto-formatting.

  **Note:** if any site has different values for the v1 cols (e.g., different temperature), the `replace_all` won't apply uniformly — switch to per-site Edits in that case. From Step 0 grep all 4 sites had identical values, so `replace_all=True` should work.

- [ ] **Step 3: Re-read all 4 sites to verify.**

  ```bash
  sed -n '418,478p' tests/test_schemas/test_dataframe_schemas.py
  sed -n '578,632p' tests/test_schemas/test_dataframe_schemas.py
  ```

  Expected: each site now has 8 refined col entries.

- [ ] **Step 4: Run the schema tests.**

  ```bash
  PYTHONPATH=src pytest tests/test_schemas/ -v --no-cov
  ```

  Expected: all tests pass.

- [ ] **Step 5: Cluster-A grep verification — no fixture leftovers.**

  ```bash
  PYTHONPATH=src grep -rn "wind_speed_mph" tests/ | grep -v "test_weather_features.py\|test_build_weather_override_cli.py"
  ```

  Expected: empty output. Only the 2 helper-test files (which intentionally still reference v1) match the broader grep; no other in-scope test file should reference v1 cols. **If any other file matches, investigate before continuing.** (Tasks 6 and 7 will rewrite the builder-boundary tests in test_rb.py + test_wr.py separately; they currently still reference v1 names but that's tracked.)

  Actually allow `test_rb.py` and `test_wr.py` matches at this stage — they get rewritten in Task 6. The expected match set after this task is exactly:
  ```
  tests/test_features/test_weather_features.py     (helper, intentional)
  tests/test_scripts/test_build_weather_override_cli.py  (helper, intentional)
  tests/test_features/test_rb.py                    (rewritten in Task 6)
  tests/test_features/test_wr.py                    (rewritten in Task 6)
  ```

- [ ] **Step 6: Commit.**

  ```bash
  git add tests/test_schemas/test_dataframe_schemas.py
  git commit -m "$(cat <<'EOF'
  test(schemas): swap v1 weather defaults for refined 8-col defaults at 4 fixture sites

  Cluster-A fixture site 3 of 3. Four DataFrame literals in this file
  exercise Rb/WrFeaturesSchema validation; each had 4 v1 weather col
  entries. All four updated to the refined 8-col bundle.

  After this commit cluster-A fixture sweep is complete. Builder-boundary
  weather tests in test_rb.py + test_wr.py still reference v1 cols;
  Task 6 rewrites those.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 6: Phase 4b — Builder-boundary test rewrites in `test_rb.py` + `test_wr.py`

**Files:**
- Modify: `tests/test_features/test_rb.py:402-595` (5 PR #29 weather tests)
- Modify: `tests/test_features/test_wr.py:613-805` (5 PR #29 weather tests)

**Context:** PR #29 added 5 builder-boundary weather tests per file:
1. `test_build_<pos>_features_attach_weather_dome_fill` — drop.
2. `test_build_<pos>_features_attach_weather_outdoor_high_wind` — drop.
3. `test_build_<pos>_features_attach_weather_grass_surface` — drop.
4. `test_build_<pos>_features_attach_weather_bye_week_fallback` — rewrite assertions to refined cols.
5. `test_build_<pos>_features_attach_weather_outdoor_nan_data_propagates_nan` — rewrite to assert NaN propagation through the refined cols (outdoor + NaN temp → `is_cold_weather` is NaN; missing `surface` → 6 surface flags NaN).

Add 5 new tests per file:
- `test_build_<pos>_features_attach_weather_refined_dome_fill`
- `test_build_<pos>_features_attach_weather_refined_cold_outdoor`
- `test_build_<pos>_features_attach_weather_refined_surface_multiclass` (parametrized over 6 codes)
- `test_build_<pos>_features_attach_weather_refined_primetime_kickoff`
- `test_build_<pos>_features_attach_weather_refined_bye_week_fallback` (replaces the old bye-week test in the rewrite path; named "refined" to be clear)

Edits to test_rb.py and test_wr.py are **parallel** in shape — every assertion structure mirrors between RB and WR. The synthetic-fixture builders, schedules construction, depth-charts construction, and downstream `build_<pos>_features` invocation are already in place from PR #29; only the weather-related setup (schedules row's `wind`/`temp`/`roof`/`surface`/`kickoff`) and the assertions change.

- [ ] **Step 1: Re-read the existing PR #29 weather test block in `test_rb.py`.**

  ```bash
  sed -n '400,600p' tests/test_features/test_rb.py
  ```

  Expected: 5 tests at lines ~402, ~435, ~469, ~520, ~557, each with their own synthetic schedules row construction and assertions on `wind_speed_mph` / `is_high_wind` / `temperature_f` / `is_grass_surface`.

- [ ] **Step 2: Re-read the existing PR #29 weather test block in `test_wr.py`.**

  ```bash
  sed -n '610,810p' tests/test_features/test_wr.py
  ```

  Expected: 5 parallel-shaped tests at lines ~613, ~646, ~680, ~731, ~768.

- [ ] **Step 3: Edit `test_rb.py` — drop the 3 v1-only tests + rewrite 2.**

  In `tests/test_features/test_rb.py`, replace the **entire 5-test block** spanning the 5 functions (from `def test_build_rb_features_attach_weather_dome_fill` at line ~402 through the end of `test_build_rb_features_attach_weather_outdoor_nan_data_propagates_nan` at line ~595 — read the file to find the exact range and the function that comes after, then replace from the start of test #1 through just before the next non-weather function) with the new 5-test block:

  ```python
  def test_build_rb_features_attach_weather_refined_dome_fill(
      baseline_weekly_stats_rb: pd.DataFrame,
      baseline_depth_charts_rb: pd.DataFrame,
      baseline_ngs_rushing: pd.DataFrame,
      baseline_snap_counts_rb: pd.DataFrame,
      baseline_id_map: pd.DataFrame,
  ) -> None:
      """Dome game: temperature_f filled to 70 by attach_weather_features →
      is_cold_weather=0; surface flags reflect the actual stadium surface
      (no roof-based override); is_primetime=0 for a 1pm ET dome kickoff.
      """
      schedules = _build_schedules_with_weather(
          roof="dome",
          surface="fieldturf",
          wind=None,
          temp=None,
          kickoff_hour_local=13,
      )
      out = build_rb_features(
          weekly_stats=baseline_weekly_stats_rb,
          depth_charts=baseline_depth_charts_rb,
          ngs_rushing=baseline_ngs_rushing,
          snap_counts=baseline_snap_counts_rb,
          id_map=baseline_id_map,
          schedules=schedules,
          as_of_week=5,
      )
      assert not out.empty, "fixture should produce at least one row"
      assert (out["is_cold_weather"] == 0.0).all(), "dome temp=70 fill → is_cold_weather=0"
      assert (out["is_fieldturf"] == 1.0).all(), "fieldturf row → is_fieldturf=1"
      for surf in ("is_a_turf", "is_astroturf", "is_grass", "is_matrixturf", "is_sportturf"):
          assert (out[surf] == 0.0).all(), f"{surf} should be 0 for fieldturf row"
      assert (out["is_primetime"] == 0.0).all(), "1pm ET kickoff → is_primetime=0"


  def test_build_rb_features_attach_weather_refined_cold_outdoor(
      baseline_weekly_stats_rb: pd.DataFrame,
      baseline_depth_charts_rb: pd.DataFrame,
      baseline_ngs_rushing: pd.DataFrame,
      baseline_snap_counts_rb: pd.DataFrame,
      baseline_id_map: pd.DataFrame,
  ) -> None:
      """Outdoor cold game: temp=28 → is_cold_weather=1; grass surface →
      is_grass=1, others 0; 1pm ET kickoff → is_primetime=0.
      """
      schedules = _build_schedules_with_weather(
          roof=None,
          surface="grass",
          wind=10.0,
          temp=28.0,
          kickoff_hour_local=13,
      )
      out = build_rb_features(
          weekly_stats=baseline_weekly_stats_rb,
          depth_charts=baseline_depth_charts_rb,
          ngs_rushing=baseline_ngs_rushing,
          snap_counts=baseline_snap_counts_rb,
          id_map=baseline_id_map,
          schedules=schedules,
          as_of_week=5,
      )
      assert not out.empty
      assert (out["is_cold_weather"] == 1.0).all(), "temp=28 → is_cold_weather=1"
      assert (out["is_grass"] == 1.0).all(), "grass row → is_grass=1"
      for surf in ("is_a_turf", "is_astroturf", "is_fieldturf", "is_matrixturf", "is_sportturf"):
          assert (out[surf] == 0.0).all(), f"{surf} should be 0 for grass row"
      assert (out["is_primetime"] == 0.0).all()


  @pytest.mark.parametrize(
      "surface_code,expected_col",
      [
          ("a_turf", "is_a_turf"),
          ("astroturf", "is_astroturf"),
          ("fieldturf", "is_fieldturf"),
          ("grass", "is_grass"),
          ("matrixturf", "is_matrixturf"),
          ("sportturf", "is_sportturf"),
      ],
  )
  def test_build_rb_features_attach_weather_refined_surface_multiclass(
      surface_code: str,
      expected_col: str,
      baseline_weekly_stats_rb: pd.DataFrame,
      baseline_depth_charts_rb: pd.DataFrame,
      baseline_ngs_rushing: pd.DataFrame,
      baseline_snap_counts_rb: pd.DataFrame,
      baseline_id_map: pd.DataFrame,
  ) -> None:
      """For each of the 6 surface codes, the matching one-hot col is 1.0
      and the other 5 are 0.0 in the builder output.
      """
      schedules = _build_schedules_with_weather(
          roof=None,
          surface=surface_code,
          wind=5.0,
          temp=60.0,
          kickoff_hour_local=13,
      )
      out = build_rb_features(
          weekly_stats=baseline_weekly_stats_rb,
          depth_charts=baseline_depth_charts_rb,
          ngs_rushing=baseline_ngs_rushing,
          snap_counts=baseline_snap_counts_rb,
          id_map=baseline_id_map,
          schedules=schedules,
          as_of_week=5,
      )
      assert not out.empty
      surface_cols = ("is_a_turf", "is_astroturf", "is_fieldturf", "is_grass", "is_matrixturf", "is_sportturf")
      for col in surface_cols:
          if col == expected_col:
              assert (out[col] == 1.0).all(), f"{col} should be 1 for surface={surface_code}"
          else:
              assert (out[col] == 0.0).all(), f"{col} should be 0 for surface={surface_code}"


  def test_build_rb_features_attach_weather_refined_primetime_kickoff(
      baseline_weekly_stats_rb: pd.DataFrame,
      baseline_depth_charts_rb: pd.DataFrame,
      baseline_ngs_rushing: pd.DataFrame,
      baseline_snap_counts_rb: pd.DataFrame,
      baseline_id_map: pd.DataFrame,
  ) -> None:
      """8:20pm ET kickoff → is_primetime=1; 1pm ET → is_primetime=0."""
      for hour, expected in ((20.333, 1.0), (13.0, 0.0)):  # 8:20pm = 20.333 (hour + 20/60)
          schedules = _build_schedules_with_weather(
              roof=None,
              surface="grass",
              wind=5.0,
              temp=60.0,
              kickoff_hour_local=hour,
          )
          out = build_rb_features(
              weekly_stats=baseline_weekly_stats_rb,
              depth_charts=baseline_depth_charts_rb,
              ngs_rushing=baseline_ngs_rushing,
              snap_counts=baseline_snap_counts_rb,
              id_map=baseline_id_map,
              schedules=schedules,
              as_of_week=5,
          )
          assert (out["is_primetime"] == expected).all(), (
              f"kickoff hour {hour} ET should yield is_primetime={expected}"
          )


  def test_build_rb_features_attach_weather_refined_bye_week_fallback(
      baseline_weekly_stats_rb: pd.DataFrame,
      baseline_depth_charts_rb: pd.DataFrame,
      baseline_ngs_rushing: pd.DataFrame,
      baseline_snap_counts_rb: pd.DataFrame,
      baseline_id_map: pd.DataFrame,
  ) -> None:
      """Bye-week / missing-schedule case: empty schedules frame → builders
      produce an empty output (since the bye-week filter inside the builder
      drops any team with no schedule row in `as_of_week`). The 8 refined
      weather cols never get populated; schema's nullable=True absorbs.
      """
      empty_schedules = _build_empty_schedules()
      out = build_rb_features(
          weekly_stats=baseline_weekly_stats_rb,
          depth_charts=baseline_depth_charts_rb,
          ngs_rushing=baseline_ngs_rushing,
          snap_counts=baseline_snap_counts_rb,
          id_map=baseline_id_map,
          schedules=empty_schedules,
          as_of_week=5,
      )
      # With the bye-week filter, an empty schedules → no rostered teams → empty output.
      # The schema validation still runs and accepts the empty frame because all 8
      # refined cols are nullable=True.
      assert out.empty or out[
          ["is_cold_weather", "is_a_turf", "is_astroturf", "is_fieldturf",
           "is_grass", "is_matrixturf", "is_sportturf", "is_primetime"]
      ].isna().all().all(), (
          "bye-week / missing-schedule rows should have all 8 refined weather cols NaN"
      )
  ```

  **Implementation notes:**
  - `_build_schedules_with_weather(roof, surface, wind, temp, kickoff_hour_local)` and `_build_empty_schedules()` are local helpers added by PR #29 (verify by re-reading `tests/test_features/test_rb.py` for their existing definitions). If their signatures differ from what's used above, adjust the keyword arguments to match the existing helper signatures. **Do not invent new helpers**; PR #29's are sufficient.
  - PR #29's tests passed `kickoff_utc` directly; the new tests use `kickoff_hour_local` semantically (the helper converts to UTC). Verify how PR #29's helper translates the local hour to a UTC-aware timestamp; mirror it. If the helper accepts a `kickoff_utc` directly, compute it from the local hour (`pd.Timestamp("2024-09-15 13:00", tz="America/New_York").tz_convert("UTC")` for 1pm ET, etc.).
  - Drop the `outdoor_nan_data_propagates_nan` test entirely OR rewrite as `test_build_rb_features_attach_weather_refined_outdoor_nan_temp_propagates_to_is_cold_weather` — it currently asserts NaN on `wind_speed_mph` / `temperature_f` (filtered out by schema post-this-PR); a useful rewrite asserts that outdoor + NaN `temp` → `is_cold_weather` is NaN in builder output. Decide based on test clarity at write time; if rewriting, append to the 5 new tests above (so 6 tests total). **Recommended:** drop entirely; the helper-level test in `test_weather_features.py` covers `_compute_is_cold_weather`'s NaN propagation, and the bye-week-fallback test above covers schema acceptance of NaN refined cols.

- [ ] **Step 4: Edit `test_wr.py` — parallel structure.**

  Replace the 5 PR #29 weather tests in `tests/test_features/test_wr.py` (~lines 613–805) with 5 new tests that **mirror Step 3 exactly**, substituting:
  - `build_rb_features` → `build_wr_features`
  - `baseline_weekly_stats_rb` → `baseline_weekly_stats_wr`
  - `baseline_depth_charts_rb` → `baseline_depth_charts_wr`
  - `baseline_ngs_rushing` → `baseline_ngs_receiving` (or whatever WR's NGS fixture is named — check by reading existing test_wr.py imports)
  - `baseline_snap_counts_rb` → `baseline_snap_counts_wr`
  - `test_build_rb_features_*` → `test_build_wr_features_*`
  - WR's `build_wr_features` signature may also need `draft_picks`, `pbp` kwargs (per PR #26 / PR #29 plumbing). Check PR #29's existing weather tests to see what kwargs they pass; mirror exactly.

- [ ] **Step 5: Run RB + WR builder tests.**

  ```bash
  PYTHONPATH=src pytest tests/test_features/test_rb.py tests/test_features/test_wr.py -v --no-cov
  ```

  Expected: all tests pass. The 5 new refined-bundle tests per file pass; the 3 dropped tests are gone; the 2 rewritten/dropped legacy tests no longer reference v1 cols.

  **If failures occur:** the most likely cause is a fixture-signature mismatch between this plan's test code and the actual local helpers. Read the failing test's traceback; adjust the keyword args in the test to match the existing local helpers. Do **not** modify the helpers themselves.

- [ ] **Step 6: Cluster-A grep verification — no leftovers in `test_rb.py` / `test_wr.py`.**

  ```bash
  PYTHONPATH=src grep -n "wind_speed_mph\|is_high_wind\|is_grass_surface" tests/test_features/test_rb.py tests/test_features/test_wr.py
  ```

  Expected: empty output. If any line matches, the test rewrite missed a leftover; investigate and fix before continuing.

  Note: the term `temperature_f` is also a v1 col but it appears in the helper-level signature (`_compute_is_cold_weather(temperature_f: pd.Series)`) — that's *not* a v1 col reference, it's a parameter name. Don't grep `temperature_f` in this verification because it would match the helper signature. (If the grep above is empty and a follow-up grep on `temperature_f` shows only param-name uses, you're fine.)

- [ ] **Step 7: Commit.**

  ```bash
  git add tests/test_features/test_rb.py tests/test_features/test_wr.py
  git commit -m "$(cat <<'EOF'
  test(rb,wr): rewrite builder-boundary weather tests for refined 8-col bundle

  Drop 3 of PR #29's 5 builder-boundary weather tests (dome_fill,
  outdoor_high_wind, grass_surface — all asserted on v1 cols which are
  no longer in the schema) per file. Drop or rewrite the 4th and 5th
  (bye_week_fallback, outdoor_nan_data_propagates_nan).

  Add 5 new tests per file mirroring the refined bundle:
  - refined_dome_fill: temp=70 fill → is_cold_weather=0; surface flags
    reflect actual code; is_primetime=0 for 1pm kickoff.
  - refined_cold_outdoor: temp=28 → is_cold_weather=1; grass → is_grass=1.
  - refined_surface_multiclass: parametrized over 6 codes; matching
    one-hot is 1.0, others 0.0.
  - refined_primetime_kickoff: 8:20pm ET → is_primetime=1; 1pm → 0.
  - refined_bye_week_fallback: empty schedules → empty output (or all-NaN
    refined cols); schema nullable=True absorbs.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 7: Phase 2 + 4c — Module docstring update + full-suite verification

**Files:**
- Modify: `src/projections/features/weather_features.py:1-12` (module docstring)
- Verify: full repo

**Context:** The current docstring still says "Probe-only — features land in the override parquet, not in `*FeaturesSchema`." Stale post-PR #29; doubly stale post-this-PR. Update to reflect that both v1 and refined bundles are integrated. Then run the full verification suite per CLAUDE.md FORCED VERIFICATION.

- [ ] **Step 1: Re-read the current docstring.**

  ```bash
  sed -n '1,15p' src/projections/features/weather_features.py
  ```

  Expected: 12-line docstring starting with `"""Weather feature computes for the weather refined-unit family probe (PR #28 broad-cut + this PR's refinements).`.

- [ ] **Step 2: Replace the docstring.**

  Replace lines 1–12 with:

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

  (`PR-#-this-spec` placeholder gets filled in with the actual PR number once the PR is opened — Task 10.)

- [ ] **Step 3: Run full pytest.**

  ```bash
  PYTHONPATH=src pytest -v --no-cov 2>&1 | tail -30
  ```

  Expected: all tests pass. Pay attention to:
  - `tests/test_models/test_baseline_feature_columns_match_schema.py` (4 cases pass).
  - `tests/test_schemas/` (all schema tests pass).
  - `tests/test_features/test_rb.py`, `test_wr.py` (5 new refined-bundle tests pass per file).
  - `tests/test_features/test_cache.py`, `tests/test_scripts/test_tune_lightgbm.py`, `tests/test_schemas/test_dataframe_schemas.py` (cluster-A fixture sites pass).
  - `tests/test_features/test_weather_features.py` and `tests/test_scripts/test_build_weather_override_cli.py` (helper tests, unchanged — still reference v1 cols).

- [ ] **Step 4: Run mypy.**

  ```bash
  PYTHONPATH=src mypy src tests
  ```

  Expected: 0 violations.

- [ ] **Step 5: Run ruff check + format.**

  ```bash
  ruff check src tests
  ruff format --check src tests
  ```

  Expected: 0 violations on both.

- [ ] **Step 6: Run the schemas/ingest/store integration sweep.**

  Per CLAUDE.md FORCED VERIFICATION rule for tasks touching pandera schemas:

  ```bash
  PYTHONPATH=src pytest -v -k "ingest or store or schemas" --no-cov 2>&1 | tail -20
  ```

  Expected: all pass.

- [ ] **Step 7: Final cluster-A grep — verify only 2 helper-test files match v1 cols.**

  ```bash
  PYTHONPATH=src grep -rn "wind_speed_mph" tests/
  ```

  Expected: matches only in `tests/test_features/test_weather_features.py` (helper tests) and `tests/test_scripts/test_build_weather_override_cli.py` (probe override CLI tests). No other match.

- [ ] **Step 8: Commit.**

  ```bash
  git add src/projections/features/weather_features.py
  git commit -m "$(cat <<'EOF'
  docs(weather_features): update module docstring for both bundles integrated

  PR #28 probe → PR #29 v1 integration shipped the broad-cut bundle.
  PR #30 probe → this PR shipped the refined bundle as a strict-replace
  swap. Helper still returns all 12 cols (4 v1 + 8 refined) because
  build_weather_overrides + the probe override CLI consume the full set.
  Pandera strict=filter on production builders filters v1 at the
  schema boundary.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 8: Phase 5a — Refresh feature caches + coverage cross-check

**Files:**
- Modify: `data/features/{rb,wr}/season=YYYY/week=WW/part.parquet` (gitignored)
- Verify: `data/raw/schedules/` is post-`56df07f` corrected state

**Context:** The existing RB + WR feature caches were written before this PR's schema swap; they have v1 cols. The refined-cols schema rejects them — caches must be regenerated. `scripts/refresh_features.py` accepts a single position per invocation (PR #29 caught this gap), so call twice. After refresh, cross-check coverage of the 8 refined cols against PR #30's audit.

- [ ] **Step 1: Verify schedules state.**

  ```bash
  PYTHONPATH=src python -c "
  from pathlib import Path
  import pandas as pd
  from projections.store import read_partition
  s = read_partition(Path('data/raw'), 'schedules', season=2024)
  primetime_rate = ((s['kickoff'].dt.tz_convert('America/New_York').dt.hour + s['kickoff'].dt.tz_convert('America/New_York').dt.minute/60.0) >= 18.0).mean()
  print(f'2024 primetime rate: {primetime_rate:.4f}')
  "
  ```

  Expected: ~0.18-0.25 (consistent with ~6-of-32 primetime teams per week × 18 weeks). If <0.05, the partitions are pre-`56df07f` (the ET-wall-clock-as-UTC bug); regenerate via:
  ```bash
  PYTHONPATH=src python -c "from pathlib import Path; from projections.ingest.refresh import refresh; refresh(data_root=Path('data'), seasons=range(2018, 2025), only=['schedules'])"
  ```

- [ ] **Step 2: Refresh RB features.**

  ```bash
  PYTHONPATH=src python scripts/refresh_features.py rb --seasons 2018-2024
  ```

  Expected: prints per-season-per-week partition counts; total > 0; no schema-validation errors. **If a schema-validation error fires here**, the RB builder is producing rows that don't satisfy the refined schema — re-read the error, identify which col, and check whether the helper's output for that col matches the schema's bounds (should always, since `compute_weather_features` outputs `Float64` 0/1/NaN with `nullable=True`).

- [ ] **Step 3: Refresh WR features.**

  ```bash
  PYTHONPATH=src python scripts/refresh_features.py wr --seasons 2018-2024
  ```

  Expected: same shape as Step 2.

- [ ] **Step 4: Coverage cross-check on the 8 refined cols.**

  ```bash
  PYTHONPATH=src python -c "
  from pathlib import Path
  import pandas as pd
  cols = ['is_cold_weather', 'is_a_turf', 'is_astroturf', 'is_fieldturf', 'is_grass', 'is_matrixturf', 'is_sportturf', 'is_primetime']
  for pos in ('rb', 'wr'):
      print(f'=== {pos.upper()} ===')
      frames = []
      for season in range(2021, 2025):
          for week in range(1, 19):
              p = Path(f'data/features/{pos}/season={season}/week={week:02d}/part.parquet')
              if p.exists():
                  frames.append(pd.read_parquet(p))
      df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
      print(f'  rows: {len(df)}')
      for c in cols:
          rate = df[c].notna().mean() if c in df.columns else None
          print(f'  {c}: non-NaN rate = {rate:.4f}' if rate is not None else f'  {c}: MISSING')
  "
  ```

  Expected: `is_cold_weather` ~85-90% pooled (2022 ~67% drags it down); 6 surface one-hots ~98% non-NaN; `is_primetime` 100%. **If `is_primetime` is <0.95**, the schedules partitions are pre-`56df07f` — go back to Step 1.

  Cross-check against PR #30's `reports/feature_probe_weather_refined_override_audit.md` per-(position, season). Materially-different coverage (>5pp shift) signals a builder wiring bug — investigate before continuing.

- [ ] **Step 5: Document the coverage in a scratch file (no commit yet).**

  Save coverage output to `reports/weather_refined_rb_wr_coverage.txt` for later inclusion in the summary report (Task 10):

  ```bash
  PYTHONPATH=src python -c "<<the script from Step 4 above>>" > reports/weather_refined_rb_wr_coverage.txt
  ```

  This file is not committed yet (Task 10 incorporates it into the summary report).

- [ ] **Step 6: No commit at this task.**

  Feature caches are gitignored. The coverage scratch file is a working artifact for Task 10.

---

## Task 9: Phase 5b — Backtest dual-run snapshot regen via `backtest_dual.py`

**Files:**
- Modify: `tests/backtest/model_metrics.json` (snapshot rows for `(RB,WR) × (baseline,lightgbm-nb)`)
- Generate: `data/backtest/run_baseline/`, `data/backtest/run_candidate/` (gitignored)

**Context:** PR #29 shipped `scripts/backtest_dual.py` exactly for this orchestration: candidate run with current branch's schema state, then `git checkout main` to revert schemas, refresh caches against main, run baseline, restore HEAD, refresh caches back, merge candidate metrics into the snapshot. The script is hardcoded for `(RB,WR) × (baseline,lightgbm-nb)` — exactly this PR's gate scope. Wall-time: ~30-60 min depending on machine.

- [ ] **Step 1: Verify branch state is clean and on `feat/weather-refined-rb-wr`.**

  ```bash
  git status
  git rev-parse --abbrev-ref HEAD
  ```

  Expected: working tree clean; branch `feat/weather-refined-rb-wr`. **If working tree is dirty**, commit or stash before running `backtest_dual.py` — the script does `git checkout main` and `git checkout HEAD` on `_FLIPPED_FILES`, which would conflict.

- [ ] **Step 2: Run `backtest_dual.py`.**

  ```bash
  PYTHONPATH=src python scripts/backtest_dual.py 2>&1 | tee /tmp/backtest_dual.log
  ```

  Expected output sequence (per backtest_dual.py:65-114):
  - "=== Phase 1: candidate run …" — runs `_run_single_backtest.py` against the current branch (refined cols). Wall-time ~15-30 min.
  - "=== Phase 2: roll back source to main HEAD ===" — `git checkout main -- <flipped files>`.
  - "=== Phase 3: refresh feature caches (now without weather cols) ===" — `refresh_features.py rb` + `refresh_features.py wr` against main's schema (v1 cols).
  - "=== Phase 4: baseline run …" — runs `_run_single_backtest.py` against main (v1 cols). Wall-time ~15-30 min.
  - "=== Phase 5: restore source to HEAD …" — `git checkout HEAD -- <flipped files>` (back to refined cols).
  - "=== Phase 6: refresh feature caches (back to with weather cols) ===" — refresh against refined cols.
  - "=== Phase 7: merge candidate metrics into snapshot ===" — read `metrics.parquet`, preserve non-target rows in `tests/backtest/model_metrics.json`, write merged.
  - "=== Done ===" — prints run dirs + snapshot path.

  **If the script crashes** mid-flight, the `try/finally` block restores HEAD source files automatically (line 81-83). Working-tree state should still be safe; check `git status` and resume from the failed phase manually.

- [ ] **Step 3: Verify the snapshot delta.**

  ```bash
  git diff tests/backtest/model_metrics.json | head -80
  ```

  Expected: per-(RB|WR, baseline|lightgbm-nb) rows have updated metric values; rows for QB/TE and other model classes are byte-identical (preserved by the merge logic in `backtest_dual.py:97-104`).

- [ ] **Step 4: Run the snapshot regression test to confirm consistency.**

  ```bash
  PYTHONPATH=src pytest tests/test_backtest/ -v --no-cov 2>&1 | tail -20
  ```

  Expected: snapshot tests pass against the merged JSON.

- [ ] **Step 5: Commit the snapshot.**

  ```bash
  git add tests/backtest/model_metrics.json
  git commit -m "$(cat <<'EOF'
  snapshot(backtest): regen RB+WR × baseline+lgb-nb after refined weather swap

  Dual-run via scripts/backtest_dual.py: candidate run on this branch
  (refined 8-col weather), baseline run on main (v1 4-col weather).
  Per-row predictions for RB+WR × baseline+lightgbm-nb now reflect the
  schema swap; QB/TE rows and other model classes preserved byte-identical
  via backtest_dual.py's merge logic.

  Wall-time: ~30-60 min for the full dual-run.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 10: Phase 5c + Phase 6 — Adoption gate + summary report + PM/TODO updates

**Files:**
- Generate: `reports/adoption_gate_weather_refined_rb.{md,csv}`, `reports/adoption_gate_weather_refined_wr.{md,csv}`
- Generate: `reports/weather_refined_rb_wr_summary.md`
- Modify: `project_management.md`, `TODO.md`

**Context:** Run the adoption gate twice (once per position; `adoption_gate.py --position` accepts a single value), write the summary report consolidating both, then update PM + TODO per §1.3.5 outcome. This is the final task in the per-PR work — branch is then ready for PR creation.

- [ ] **Step 1: Run adoption gate for RB.**

  ```bash
  PYTHONPATH=src python scripts/adoption_gate.py \
    --baseline-run data/backtest/run_baseline \
    --candidate-run data/backtest/run_candidate \
    --position RB \
    --csv-out reports/adoption_gate_weather_refined_rb.csv \
    > reports/adoption_gate_weather_refined_rb.md
  ```

  Expected: produces an `.md` report with the verdict table + a `.csv` with the per-row metrics. The two binding cells: `(baseline, RB)` and `(lightgbm-nb, RB)`. Probe-predicted `(lgb-nb swap, RB)` was -0.0088 fpts; expected gate magnitude similar (within ~0.0010 fpts per probe-vs-gate calibration history).

- [ ] **Step 2: Run adoption gate for WR.**

  ```bash
  PYTHONPATH=src python scripts/adoption_gate.py \
    --baseline-run data/backtest/run_baseline \
    --candidate-run data/backtest/run_candidate \
    --position WR \
    --csv-out reports/adoption_gate_weather_refined_wr.csv \
    > reports/adoption_gate_weather_refined_wr.md
  ```

  Expected: similar shape; binding cell `(lightgbm-nb, WR)`; probe-predicted -0.0050 fpts.

- [ ] **Step 3: Read both reports and identify the §1.3.5 branch per position.**

  ```bash
  cat reports/adoption_gate_weather_refined_rb.md
  cat reports/adoption_gate_weather_refined_wr.md
  ```

  For each position:
  - **`(lgb-nb, POS)` ADOPT AND `(baseline, POS)` not REGRESSION** → ship-as-designed (default expected branch).
  - **`(lgb-nb, POS)` ADOPT AND `(baseline, POS)` REGRESSION** → full-revert that position (per §1.3.5 contingency note).
  - **`(lgb-nb, POS)` MARGINAL or DO_NOT_ADOPT** → full-revert that position.

  If either position requires full-revert, follow the **Phase 6 conditional contingency** below before continuing to Step 4.

- [ ] **Step 4: Write the summary report.**

  Create `reports/weather_refined_rb_wr_summary.md` with the following sections (model on PR #29's `weather_features_rb_wr_summary.md` shape):

  - **Header**: date, branch, spec link, plan link, predecessors (PR #28, #29, #30), verdict.
  - **Per-mode verdict table**: 4 cells × {ΔRMSE, CI, Spearman Δ, verdict}.
  - **Probe-vs-gate calibration**: probe-predicted (-0.0088 RB swap, -0.0050 WR swap) vs gate-measured magnitudes per binding cell. Note magnitude delta (typically <0.001 fpts per the calibration history).
  - **Per-position §1.3.5 outcome**: which branch fired for RB, which for WR (typically both ship-as-designed).
  - **Coverage statistics** (from Task 8 step 5's scratch file): per-(position, season) on `is_cold_weather` / multi-class surface / `is_primetime`. Cross-check against PR #30's audit.
  - **Binding-cell shift note**: lgb-nb is binding, baseline is informational/contingency. Production routing for both RB and WR remains on `BaselineModel` per Plan 8.
  - **Cross-class deferred follow-up (per position)**: with refined cols in schema, a cross-class re-eval could justify flipping `_PositionDispatch[POS].default_model_class` to `lightgbm-nb`. Same shape as PR #29's deferred entry; this PR's update changes "v1 weather" → "refined weather".
  - **Spec gaps caught + fixed during execution** (if any): document any deviation from the plan — placeholder until execution surfaces them.

  Save to `reports/weather_refined_rb_wr_summary.md`.

- [ ] **Step 5: Update `project_management.md`.**

  Append a top-of-file decision-log entry. Format mirrors PR #29's entry (`project_management.md:41-77`); shape:

  ```markdown
  ## Weather Refined-Unit RB+WR Integration — verdicts: RB <verdict>, WR <verdict> (2026-05-09, on branch `feat/weather-refined-rb-wr`)

  **Status:** Production strict-replace integration of the 8 refined weather cols into `RbFeaturesSchema` + `WrFeaturesSchema` (replacing the v1 4 cols from PR #29) per `docs/superpowers/specs/2026-05-09-weather-refined-rb-wr-design.md`. <Followed PR #29's wiring> ... <verdicts> ... <ship/revert decisions>.

  **Per-position dual-run gate verdicts (4 cells):**
  <table per Task 10 step 4>

  **Probe-vs-gate calibration:** Probe-predicted (lgb-nb swap, RB) -0.0088 / (lgb-nb swap, WR) -0.0050; gate-measured <values>. <within X fpts of probe predictions; track record extension>.

  **Per-position §1.3.5 outcome:** <which branches fired>.

  **Coverage statistics (2021-2024 eval window):** <from coverage cross-check>.

  **Cross-class deferred follow-ups (per position):** <update PR #29's entries to reflect refined cols>.

  **What this closes:** TODO #25's refined-unit (in-builder) cut on the RB and WR ADOPT cells from PR #30. QB and TE remain DO_NOT_ADOPT at this unit. Refined-unit-of-refined-unit candidates (continuous kickoff hour, is_london, surface × position interactions, per-team weather acclimation, precipitation, wind direction) remain open under TODO #25; none queued.

  **Spec gaps caught + fixed during execution:** <if any>.

  See `reports/weather_refined_rb_wr_summary.md` for the full decision log + per-mode table + per-position §1.3.5 outcome + probe-vs-gate calibration.

  ---
  ```

- [ ] **Step 6: Update `TODO.md` #25.**

  Find TODO #25 (the broad-cut weather backlog entry); append:

  ```markdown
  **Update 2026-05-09 (Refined-unit RB+WR strict-replace integration, branch `feat/weather-refined-rb-wr`):** Production strict-replace integration of the 8 refined weather cols (is_cold_weather, six surface one-hots, is_primetime) into `RbFeaturesSchema` + `WrFeaturesSchema`, replacing the v1 4-col bundle from PR #29. Dual-run gate verdicts on RB+WR × baseline+lgb-nb: <verdicts>. Probe-predicted -0.0088 RB / -0.0050 WR; gate-measured <values>. <Per-position outcome>. The "broad-cut weather family at the in-builder unit" is now closed at the refined-unit level for RB and WR. Refined-unit-of-refined-unit candidates remain open: continuous kickoff hour, is_london, surface × position interactions, per-team weather acclimation, precipitation (would require new ingest), wind direction (new ingest). None queued; recommended priority continuous kickoff hour first if a follow-up is scoped. See `reports/weather_refined_rb_wr_summary.md`.
  ```

  Also: update PR #29's deferred cross-class follow-up sub-bullet (find by searching for "RB / WR production routes to `baseline` per Plan 8" in TODO.md) to reflect refined cols in schema.

- [ ] **Step 7: Fill in the `PR-#-this-spec` placeholder in `weather_features.py` docstring.**

  Once the PR number is known (after `gh pr create`), replace `PR-#-this-spec` with the actual PR number (e.g., `PR #31`). This is a follow-up after the PR is opened; the placeholder is acceptable for the initial commit chain.

- [ ] **Step 8: Final pytest + lint check.**

  ```bash
  PYTHONPATH=src pytest -v --no-cov 2>&1 | tail -10
  PYTHONPATH=src mypy src tests
  ruff check src tests scripts
  ruff format --check src tests
  ```

  All clean.

- [ ] **Step 9: Commit reports + PM/TODO updates.**

  ```bash
  git add reports/adoption_gate_weather_refined_rb.{md,csv} \
          reports/adoption_gate_weather_refined_wr.{md,csv} \
          reports/weather_refined_rb_wr_summary.md \
          project_management.md \
          TODO.md
  git commit -m "$(cat <<'EOF'
  report(weather-refined-rb-wr): adoption gate + summary + PM/TODO updates

  Adoption gate verdicts on RB+WR × baseline+lgb-nb:
  - <(baseline, RB)>: <verdict + magnitude>
  - <(lgb-nb, RB)>: <verdict + magnitude>
  - <(baseline, WR)>: <verdict + magnitude>
  - <(lgb-nb, WR)>: <verdict + magnitude>

  Per-position §1.3.5 outcome: <which branches fired>.

  See reports/weather_refined_rb_wr_summary.md for full decision log,
  probe-vs-gate calibration, and per-position contingency narrative.

  Closes TODO #25 broad-cut weather at the refined-unit level for RB+WR.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

- [ ] **Step 10: Open PR.**

  ```bash
  git push -u origin feat/weather-refined-rb-wr
  gh pr create --title "feat: refined-unit weather strict-replace on RB+WR" --body "$(cat <<'EOF'
  ## Summary
  - Strict-replace integration of the 8 refined weather cols into `RbFeaturesSchema` + `WrFeaturesSchema`, replacing the v1 4-col bundle from PR #29.
  - Dual-run gate verdicts: <fill in per Task 10 step 3>.
  - Production routing for RB+WR remains on `BaselineModel` per Plan 8; cross-class flip deferred per PR #29 / PR #27 precedent.

  ## Test plan
  - [x] `pytest -v` clean
  - [x] `mypy src tests` zero violations
  - [x] `ruff check src tests scripts` zero violations
  - [x] `ruff format --check src tests` no drift
  - [x] Schema-regression test (`test_baseline_feature_columns_match_schema.py`) passes for all 4 positions
  - [x] Cluster-A grep — only 2 helper-test files match v1 col names
  - [x] Adoption gate dual-run produces RB+WR × baseline+lgb-nb verdicts

  See `reports/weather_refined_rb_wr_summary.md` for full decision log.
  See `docs/superpowers/specs/2026-05-09-weather-refined-rb-wr-design.md` for spec.
  See `docs/superpowers/plans/2026-05-09-weather-refined-rb-wr.md` for plan.

  🤖 Generated with [Claude Code](https://claude.com/claude-code)
  EOF
  )"
  ```

  Output: PR URL.

---

## Phase 6 conditional contingency (only if §1.3.5 fires)

Per the spec's §1.3.5 contingency matrix, **only execute this section if the gate produces a non-ship-as-designed verdict for either position**. The expected branch is "both positions ship-as-designed" given the probe's evidence (0/120 baseline SIGNAL across both modes, ADOPT cells with CIs strictly negative).

### If `(lgb-nb, POS)` MARGINAL or DO_NOT_ADOPT, OR `(baseline, POS)` REGRESSION, for either RB or WR:

- [ ] **Revert that position's changes.**

  In `src/projections/schemas.py`, restore the v1 weather block (5-line comment + 4 v1 col defs) for that position only — leave the other position's refined block in place.

  In `src/projections/models/baseline.py`, restore the v1 names in that position's `_<POS>_FEATURE_COLUMNS` only.

  In `tests/test_features/test_<pos>.py`, optionally restore the original PR #29 weather tests (cheap to leave the new refined tests dropped — they would fail against a v1 schema). Cluster-A fixture defaults: cheap to leave with refined defaults; harmless.

- [ ] **Re-refresh that position's feature cache.**

  ```bash
  PYTHONPATH=src python scripts/refresh_features.py <pos> --seasons 2018-2024
  ```

- [ ] **Re-run that position's backtest cells.**

  ```bash
  PYTHONPATH=src python scripts/_run_single_backtest.py data/backtest/run_revert_<pos>
  ```

  Then merge those metrics into `tests/backtest/model_metrics.json` for that position's rows × baseline+lgb-nb only.

- [ ] **Document divergence in summary report.**

  Add a "Per-position §1.3.5 contingency outcome" subsection to `reports/weather_refined_rb_wr_summary.md` explaining which position reverted, why (what verdict fired), and what was kept (the other position's schema swap, cluster-A fixtures).

- [ ] **Commit + continue with Task 10 Step 5+ (PM/TODO updates).**

  Update `project_management.md` and `TODO.md` to reflect the per-position partial-revert outcome. The deferred cross-class follow-up note for the reverted position drops back to the v1-cols-in-schema state (or stays the same if the other position kept refined).

---

## Self-review checklist (run after this plan is complete and before handing to engineer)

The plan author has already done this review; documenting for traceability:

1. **Spec coverage:** every spec §1.1 goal is mapped to a task. Schema swap → Task 1; baseline.py swap → Task 2; cluster-A → Tasks 3–5; builder-boundary tests → Task 6; module docstring → Task 7; cache refresh + coverage → Task 8; backtest dual-run → Task 9; gate + summary + PM/TODO → Task 10. Spec §1.3.5 contingency → "Phase 6 conditional contingency" section. ✓

2. **Placeholder scan:** "PR-#-this-spec" and "<verdict>" / "<value>" placeholders intentionally remain in Tasks 7 and 10 — they're filled in at PR-open time and gate-completion time respectively. No "TBD" / "TODO" / "implement later" patterns. ✓

3. **Type / name consistency:** col names checked against `_SURFACE_CODES` in `weather_features.py` (a_turf, astroturf, fieldturf, grass, matrixturf, sportturf — all 6); `is_cold_weather`, `is_primetime` checked against helper output. `_RB_FEATURE_COLUMNS`, `_WR_FEATURE_COLUMNS` symbol names match `baseline.py`. Schema class names (`RbFeaturesSchema`, `WrFeaturesSchema`) match `schemas.py`. ✓
