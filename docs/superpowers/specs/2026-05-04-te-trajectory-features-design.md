# TE Trajectory Features Integration — Design

**Status:** approved (brainstorming, 2026-05-04). Ready for implementation plan.
**Date:** 2026-05-04
**Author:** alden + claude
**Sub-project of:** FantasyFootball / Projections Core
**Builds on:**
- Trajectory Feature Family Probe (PR #25, merged at `d045eb8`) — shipped the 4 trajectory compute fns + `attach_trajectory_features` joiner + `build_trajectory_overrides` assembler in `src/projections/features/trajectory_features.py`, the `refresh_draft_picks` ingest module + `DraftPicksSchema`, and the override-generator script. Probe verdict was `SIGNAL`; the binding cells were three Phase-2 ADOPTs: WR augment baseline (-0.0414 fpts), WR augment lgb-nb (-0.0194 fpts), **TE augment lgb-nb (-0.0107 fpts)**.
- WR Trajectory Features Integration (PR #26, merged at `884d025`) — promoted `build_draft_lookup` to public in `src/projections/features/trajectory_features.py`; added the `draft_picks` kwarg with `_EMPTY_DRAFT_PICKS` default to all 4 `build_<pos>_features` signatures (currently unused on QB / RB / TE — accept-and-ignore plumbing); plumbed `draft_picks` loading + threading through all 4 caller scripts (`refresh_features.py`, `train_baseline.py`, `predict_2024.py`, `sanity_check_baseline.py`). All ingest, helper, and caller-side plumbing is therefore already in place when this spec lands.

**Branch:** `feat/te-trajectory-features` cut from `main` at `884d025`.

---

## 1. Overview

PR #26 shipped trajectory features into `WrFeaturesSchema` and `build_wr_features` with a `(BaselineModel, WR)` ADOPT gate verdict at -0.0371 fpts. This spec executes the parallel TE integration: append the same 4 nullable-float columns to `TeFeaturesSchema`, wire `attach_trajectory_features` into `build_te_features` (the helper already supports `Position.TE`), refresh the TE feature cache, and run the dual-run adoption gate to verify the probe's prediction holds on the binding cell.

**Critical difference from PR #26: the binding cell is `(LightGBMNbModel, TE)`, not `(BaselineModel, TE)`.** PR #25's trajectory probe ADOPT'd TE only under lgb-nb (-0.0107 fpts, CI [-0.0191, -0.0028]); TE's BaselineModel cell was DO_NOT_ADOPT (CI bracketed zero, point estimate near zero — not a regression, just no signal at the linear-Ridge model class). Because TE's production `default_model_class` is still `baseline` (Plan 8 verdict 2026-04-29), shipping these features does **not** automatically improve TE production output — the feature cols are persisted in the schema for the model classes that demonstrably benefit (`lightgbm-nb`, likely `lightgbm` and `ensemble`), but production routing stays on baseline pending a separate cross-class re-eval.

The work is intentionally narrow: TE only. WR is already shipped (PR #26). QB returned regression on both model classes in PR #25's probe — explicitly do-not-integrate. RB returned net-zero (mechanism-consistent: PR #21's team-level PBP cols already cover RB's binding axis).

The shipping decision is binary and bound to the **lgb-nb** production class for TE: if the adoption gate's `(LightGBMNbModel, TE)` verdict is `ADOPT`, ship. If it is `MARGINAL` or `DO_NOT_ADOPT`, revert per Plan 9 discipline. Other model classes' verdicts (`BaselineModel`, `LightGBMModel`, `LightGBMTunedModel`, `EnsembleModel`) are informational — captured in the summary report but not gating. This is the **first integration spec in the project to bind on a non-default model class**; the §1.3.5 success rule below makes the criterion explicit so the divergence from PR #21 / PR #26's `(BaselineModel, position)` pattern is on-the-record.

### 1.1 Goals (in scope)

- Append 4 nullable-float columns to `TeFeaturesSchema` in `src/projections/schemas.py` (identical column names + dtypes + bounds as PR #26's `WrFeaturesSchema` extension):
  - `age: Series[float] = pa.Field(ge=15, le=50, nullable=True)` — biological age in the target season; primary path uses `draft_age + (season - draft_year)` from `nfl_data_py.import_draft_picks`, fallback path uses `season - inferred_draft_year + 22.0` where `inferred_draft_year` is earliest weekly_stats appearance.
  - `is_rookie: Series[float] = pa.Field(ge=0, le=1, nullable=True)` — 1.0 if `season == draft_year` else 0.0; same draft_lookup + inferred-fallback resolution as `age`.
  - `volume_trend_l4_minus_prior_l4: Series[float] = pa.Field(nullable=True)` — trailing-4-active-games mean of `targets` minus prior-4-active-games mean (per PR #25's `compute_wr_te_volume_trend`, which uses `targets` for both WR and TE). NaN for players with fewer than 8 prior active games.
  - `snap_pct_change_l4_vs_prior_l4: Series[float] = pa.Field(ge=-1, le=1, nullable=True)` — same window cut on `SnapCountsSchema.offense_pct`. Bounded since `offense_pct ∈ [0, 1]`.

  The audit-only `draft_year_inferred` boolean from the helper is **not** added — same precedent as PR #26 (production schema drops it via `strict="filter"`).

- Modify `build_te_features` in `src/projections/features/te.py` to:
  - Consume the existing `draft_picks: pd.DataFrame = _EMPTY_DRAFT_PICKS` kwarg (currently plumbed-but-unused since PR #26).
  - After the existing `out` frame has `(gsis_id, season, week, team, opponent, ...)`, build a slim index `out[["gsis_id", "season", "week", "team", "opponent"]].rename(columns={"opponent": "opp"})`.
  - Build `draft_lookup = build_draft_lookup(draft_picks)`.
  - Call `attach_trajectory_features(idx, ws, sc, draft_lookup, Position.TE)` — note: pass the **unfiltered-by-position** prior-mask-filtered weekly_stats (`ws`, not a TE-only slice) and snap_counts (`sc`, not a TE-only slice). The helper's `_volume_trend` filters on `(Position.WR, Position.TE)` internally. This matches PR #26's spec gap fix at commit `d1b3092` (the helper does its own leakage shifting via `.shift(1)` / `.shift(5)`; an external prior-mask filter would double-shift to 100% NaN).
  - Take the 4 trajectory feature columns (drop `team`, `opp`, `draft_year_inferred`, identity cols already in `out`); merge onto `out` on `(gsis_id, season, week)`.
  - The final `TeFeaturesSchema.validate(out)` enforces presence + dtype + bounds.

- Update `src/projections/models/baseline.py:_TE_FEATURE_COLUMNS` (line 408) to include the 4 new column names. **This is the same spec gap PR #21 caught at commit `9895dee` (RB) and PR #26 caught for WR** — `baseline.py` hardcodes the per-position feature tuple while the lightgbm family derives from `TeFeaturesSchema.to_schema().columns.keys()` filtered through `_NON_FEATURE_COLUMNS` dynamically. Without this update, `BaselineModel.fit` for TE will not see the new features even though the schema validates. The implementation plan calls this out as its own dedicated task with a smoke-test verification (assert the 4 names appear in `_TE_FEATURE_COLUMNS` post-edit).

- Update `tests/test_features/test_te.py`:
  - Extend the existing happy-path test to assert the 4 new columns are present, float-typed, and bounded per the schema.
  - Add `test_te_features_attach_trajectory_join_drafted_player` — synthetic fixture: a drafted TE (in `draft_lookup`) with 8+ prior active games. Assert `age = draft_age + (season - draft_year)`, `is_rookie = 0.0` (not their draft year), `volume_trend_l4_minus_prior_l4` matches a hand-computed mean-of-trailing-4-targets minus mean-of-prior-4-targets, `snap_pct_change_l4_vs_prior_l4` matches the analog on `offense_pct`.
  - Add `test_te_features_attach_trajectory_join_rookie` — rookie TE (`season == draft_year`): `is_rookie = 1.0`, `age` matches `draft_age` exactly, `volume_trend_l4_minus_prior_l4 = NaN` (no prior 8 games), `snap_pct_change_l4_vs_prior_l4 = NaN`.
  - Add `test_te_features_attach_trajectory_join_udfa` — TE with no `draft_lookup` entry: uses inferred-draft-year fallback. Assert `is_rookie = (season == inferred_draft_year)`, `age = season - inferred_draft_year + 22.0`.
  - Add `test_te_features_empty_draft_picks_fallback` — `draft_picks=_EMPTY_DRAFT_PICKS`: every row routes to inferred fallback, no errors raised, schema validates.

- Add a `te_draft_picks` fixture to `tests/test_features/conftest.py` covering the canonical TE gsis_ids used across the TE fixture set, mirroring the `wr_draft_picks` fixture that PR #26 added (4 entries: a veteran with finite `draft_age`, a second veteran, a NaN-`draft_age` branch, and a rookie drafted in the test season).

- Extend `tests/conftest.py:baseline_weekly_stats_te` (and any analogous `baseline_features_te` builder) to cover **17 weeks of 2023 + 17 weeks of 2024 + 4 weeks of 2025** — same precedent PR #26 set on the WR equivalent (`baseline_weekly_stats_wr`). Trajectory's trailing-4-minus-prior-4 trends require 8+ active games of history; without 2023, every (l4 - prior_l4) row in 2024-only fixtures would be NaN and `BaselineModel.fit`'s dropna would empty the TE training set. Same baseline_features_te schedule / depth-chart / NGS coverage extension to match.

- Synthetic-fixture grep + special-casing for `age` / `is_rookie` / trend cols on lightgbm/ensemble TE smoke fixtures — the `_minimal_te_features_row` (or analog) helper in `tests/test_features/test_cache.py` and the per-model TE smoke fixtures (PR #26's commits `1f1f415`, `33eea57`, `807f046` did this for WR). Use a defensive grep for `opp_allowed_te_fppg_l4` to find every site that builds a synthetic TE row; add `age` ∈ [22, 30], `is_rookie` ∈ {0, 1}, both trend cols at small finite values (e.g., 0.5, 0.05).

- Refresh the TE feature cache: `python scripts/refresh_features.py te --seasons 2018-2024`. (Manual; run-time output, not committed.)

- Run the full backtest + adoption gate on TE only, all 5 model classes (`baseline`, `lightgbm`, `lightgbm-tuned`, `lightgbm-nb`, `ensemble`). Commit the resulting `reports/adoption_gate_te_trajectory_features.{md,csv}`.

- Write `reports/te_trajectory_features_summary.md` consolidating: probe-predicted vs gate-measured magnitudes; per-model-class verdicts; the binding-cell shift rationale; coverage statistics; ship/revert decision; the cross-class TE production routing question explicitly flagged as deferred follow-up.

- On `(LightGBMNbModel, TE)` `ADOPT` verdict: update `project_management.md` decision log + `TODO.md` #24 (record shipped, with measured magnitude; cross-link to PR #25 and PR #26). On `MARGINAL` / `DO_NOT_ADOPT`: revert builder + schema changes (the spec leaves the test fixtures and any helper changes in place for any future TE-trajectory revisit).

### 1.2 Non-goals (deferred)

- **No production routing flip from `baseline` to `lightgbm-nb` for TE.** This is a real follow-up question — does `lgb-nb-with-trajectory` beat `baseline-without-trajectory` for TE at the position level, sufficient to justify flipping `_PositionDispatch[TE].default_model_class`? — but it's a *cross-class* question that PR #25's *within-class* probe doesn't directly answer. The dual-run gate this spec runs will produce both `(baseline, TE)` and `(lgb-nb, TE)` cells, but neither answers the cross-class question (each compares a class-against-itself with-vs-without trajectory). A separate Plan-8-style cross-class re-eval is the right shape for that question; it can run anytime after this spec ships, with the trajectory cols already in the TE schema.
- **No QB / RB / WR schema changes.** WR shipped in PR #26. QB returned regression on both model classes in PR #25 (explicitly do-not-integrate). RB returned net-zero (PR #21's team-level PBP cols already cover RB's binding axis).
- **No new ingest.** `refresh_draft_picks` shipped in PR #25 and the partition exists at `data/raw/draft_picks/season=YYYY/part.parquet`. The opt-in `--run-network` smoke at `tests/test_ingest/test_api_drift.py` already covers `nfl_data_py.import_draft_picks` column-rename drift.
- **No new helper extractions.** PR #26 promoted `build_draft_lookup` to public; `attach_trajectory_features` was public from PR #25; the `Position.TE` branch in `attach_trajectory_features` already exists. No `trajectory_features.py` edits in this PR.
- **No caller-script changes.** PR #26 plumbed `draft_picks` through all 4 caller scripts (`refresh_features.py`, `train_baseline.py`, `predict_2024.py`, `sanity_check_baseline.py`) for all 4 positions. Already wired. No script edits in this PR.
- **No per-feature ablation.** The probe tested all 4 features bundled. Production-Ridge regularization shrinks uninformative coefficients toward 0; lgb's tree splits ignore unused features. Shipping all 4 doesn't degrade prediction quality vs shipping the 1-2 load-bearing ones.
- **No new probe machinery.** The probe code is not modified. The summary report compares probe-vs-gate calibration but does not re-run the probe.
- **No spec / plan file changes for prior work.** PR #25 / PR #26 specs, plans, and reports stay as historical record.

### 1.3 Success criteria

The spec is complete iff all of:

1. **Schema + builder + tests + fixture extensions land cleanly.** `pytest -v` (full suite), `mypy src tests` (zero violations), `ruff check src tests scripts` (zero violations), `ruff format --check` (no drift).
2. **Refreshed TE feature cache validates against the extended schema** at every `(season, week)` partition.
3. **The full backtest + adoption gate runs successfully on TE** for all 5 model classes (baseline / lightgbm / lightgbm-tuned / lightgbm-nb / ensemble) across the standard `2021-2024` holdout years, **with `--coverage-threshold 0.35`** to match PR #26's structural-sparsity precedent for trajectory features.
4. **The summary report (`reports/te_trajectory_features_summary.md`) records all of:**
   - The probe's predicted composite RMSE delta on `(LightGBMNbModel, TE) augment`: **-0.0107 fpts** (from PR #25's `feature_probe_trajectory_lgbnb_augment.csv`).
   - The gate's measured composite RMSE delta on `(LightGBMNbModel, TE)` with 95% CI.
   - The per-(model_class, TE) verdicts for the other 4 model classes — informational; the baseline cell is the cross-check on the probe's TE baseline DO_NOT_ADOPT prediction.
   - Per-position coverage of the 4 new columns at the eval window (2021-2024) and on the full 2018-2024 history; documented next to the threshold relaxation rationale.
   - Explicit note that the binding-cell shift from PR #21 / PR #26's `(BaselineModel, position)` pattern was per spec §1 — TE production routing remains on `baseline`; the cross-class flip question is deferred to a separate follow-up.
5. **The shipping decision is bound to the `(LightGBMNbModel, TE)` verdict:**
   - **(lgb-nb, TE) `ADOPT`** → merge PR; update decision logs.
   - **(lgb-nb, TE) `ADOPT` AND (baseline, TE) `REGRESSION`** (CI of RMSE delta strictly above 0) → ship a *modified* shape: keep the 4 cols in `TeFeaturesSchema`, but **do not** add them to `_TE_FEATURE_COLUMNS` in `baseline.py`. This protects baseline TE production output (which is the current default) from a measurable regression while still exposing the cols to lightgbm-family TE for the eventual cross-class flip. The summary report flags the divergence and documents the rationale. Probability of this branch firing is low (probe was DO_NOT_ADOPT, not REGRESSION, on baseline TE; magnitude was within bootstrap noise of zero), but the spec pre-decides the response so the implementation plan doesn't have to.
   - **(lgb-nb, TE) `MARGINAL` or `DO_NOT_ADOPT`** → revert builder + schema changes. Document the divergence in the summary; close TODO #24's TE-cell exploration as "probe SIGNAL did not reproduce in production gate."

If criterion 1 fails, fix and rerun. If criterion 2 fails, the builder is wrong — fix before running the gate. Criterion 3 is mechanical (the gate either runs or doesn't). Criterion 5 is the binding decision.

---

## 2. Inputs

### 2.1 Draft-picks source

Same as PR #26: `draft_picks` partitions read via `read_partition(raw_root, "draft_picks", season=s)` for the full historical range `[1980, seasons.stop)`. The caller scripts (already plumbed in PR #26) handle the load via `_read_concat(raw_root, "draft_picks", list(range(1980, seasons.stop)))`.

Empty fallback: if `data/raw/draft_picks/` does not exist, `build_te_features` produces an empty `draft_lookup`, every row routes to the inferred-fallback `age` and `is_rookie` resolution, and the schema's `nullable=True` accepts the resulting NaN for cold-start cases. The implementation plan ensures the production execution sequence (§4) verifies the partition exists before refreshing features.

### 2.2 Player-team-week index inside `build_te_features`

`build_te_features` already produces an internal `(gsis_id, season, week, team, opponent)` frame from `depth_charts` (filtered to TEs in `as_of_week`, deduped per the Plan 3b drift fixes for the TE builder) inner-joined with `schedules` (bye-week filter from the same Plan 3b drift). The trajectory integration reuses this frame, renames `opponent → opp` to match `attach_trajectory_features`'s contract, passes it through.

No new ingest source. No new schema-level changes outside `TeFeaturesSchema`.

### 2.3 Weekly_stats and snap_counts contracts

Both must satisfy their respective schemas (caller's responsibility; existing convention). The `weekly_stats` and `snap_counts` passed to `attach_trajectory_features` are the **prior-mask-filtered** frames (`ws`, `sc`) — the existing leakage filter `prior_mask` already restricts these to rows strictly before `(season, week)`, which is the correct backfill scope for trailing-8-active-games rolling. They are **not** position-filtered (the helper's `_volume_trend` filters on `(Position.WR, Position.TE)` internally for TE).

If `weekly_stats` is empty, the helper's compute fns each return empty frames and the left-merges produce all-NaN rows. The schema's `nullable=True` accepts this.

---

## 3. Code shape

### 3.1 `build_te_features` integration

In `src/projections/features/te.py`, after the existing assembly that produces `out` with `(gsis_id, season, week, team, opponent, ...)` columns:

```python
# Trajectory features (PR #25 family probe + this spec's TE integration).
# Helper does its own position filter and rolling, so pass the unfiltered
# (but prior-mask-filtered) weekly_stats / snap_counts.
draft_lookup = build_draft_lookup(draft_picks)
traj_idx = (
    out[["gsis_id", "season", "week", "team", "opponent"]]
    .rename(columns={"opponent": "opp"})
)
traj = attach_trajectory_features(traj_idx, ws, sc, draft_lookup, Position.TE)
out = out.merge(
    traj[
        [
            "gsis_id",
            "season",
            "week",
            "age",
            "is_rookie",
            "volume_trend_l4_minus_prior_l4",
            "snap_pct_change_l4_vs_prior_l4",
        ]
    ],
    on=["gsis_id", "season", "week"],
    how="left",
)
```

The function signature already accepts `draft_picks: pd.DataFrame = _EMPTY_DRAFT_PICKS` (PR #26 plumbing). No signature change.

### 3.2 Schema change

In `src/projections/schemas.py`, append to `TeFeaturesSchema` (identical 4-line addition to PR #26's `WrFeaturesSchema` extension):

```python
# Trajectory features (PR #25 family probe + 2026-05-04 TE integration
# spec). All four are structurally sparse: age + is_rookie need a
# draft_picks lookup hit (or the inferred fallback) and so cover ~95% of
# TE player-weeks per the probe; the trend cols need 8 prior active
# games and so cover ~45-71% of TE player-weeks. NaN where coverage is
# missing; BaselineModel imputes with feature mean, lightgbm consumes
# NaN natively. See spec §1.3 criterion 4 for the coverage measurement
# requirement.
age: Series[float] = pa.Field(ge=15, le=50, nullable=True)
is_rookie: Series[float] = pa.Field(ge=0, le=1, nullable=True)
volume_trend_l4_minus_prior_l4: Series[float] = pa.Field(nullable=True)
snap_pct_change_l4_vs_prior_l4: Series[float] = pa.Field(ge=-1, le=1, nullable=True)
```

`age` lower bound 15 conservative; upper bound 50 covers Brady-era outliers. `snap_pct_change_l4_vs_prior_l4` bounded since `offense_pct ∈ [0, 1]`. `volume_trend_l4_minus_prior_l4` unbounded.

Strict mode ("filter") on the schema's `Config` already drops `draft_year_inferred` and any other helper-output columns the production schema doesn't declare; no other change needed.

### 3.3 `baseline.py` hardcoded feature list

In `src/projections/models/baseline.py`, extend `_TE_FEATURE_COLUMNS` (line 408-427) to include the 4 new column names:

```python
_TE_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    # ... existing 18 columns ...
    "opp_allowed_te_fppg_l4",
    # Trajectory features (PR #25 family probe + 2026-05-04 TE integration).
    "age",
    "is_rookie",
    "volume_trend_l4_minus_prior_l4",
    "snap_pct_change_l4_vs_prior_l4",
)
```

The lightgbm family (`models/lightgbm.py:122` and the inheriting `lightgbm_tuned.py` / `lightgbm_nb.py` / `ensemble.py`) derives feature columns dynamically from `TeFeaturesSchema.to_schema().columns.keys()` filtered through `_NON_FEATURE_COLUMNS`, so the lightgbm family auto-picks-up the 4 new columns when the schema change lands. **No edit needed to the lightgbm files** — the implementation plan asserts this with a smoke test that reads the dynamically-derived TE feature list and verifies the 4 new columns appear post-schema-change.

**§1.3.5 contingency:** if `(baseline, TE)` returns `REGRESSION` AND `(lgb-nb, TE)` ADOPTs, the modified-shape ship path in §1.3.5 leaves `_TE_FEATURE_COLUMNS` *unchanged* (no 4-col addition). The lightgbm family still picks up the cols via the schema; baseline TE doesn't see them. The implementation plan's Phase 5 handles this branch by reverting the `_TE_FEATURE_COLUMNS` edit and re-running the baseline TE backtest cell so `model_metrics.json` reflects the without-trajectory baseline (the lightgbm family cells stay as-is from Phase 4's snapshot regen).

### 3.4 Tests

In `tests/test_features/test_te.py`:

- Extend the existing happy-path test to assert the 4 new columns are present, float-typed, and bounded per the schema.
- Add `test_te_features_attach_trajectory_join_drafted_player` — synthetic fixture: a drafted TE (in `draft_lookup`) with 8+ prior active games. Assert all 4 cols match hand-computed expectations.
- Add `test_te_features_attach_trajectory_join_rookie` — rookie TE (`season == draft_year`): `is_rookie = 1.0`, `age` matches `draft_age` exactly, both trend cols `NaN`.
- Add `test_te_features_attach_trajectory_join_udfa` — TE with no `draft_lookup` entry: uses inferred-draft-year fallback.
- Add `test_te_features_empty_draft_picks_fallback` — `draft_picks=_EMPTY_DRAFT_PICKS`: every row routes to inferred fallback, no errors raised, schema validates.

In `tests/test_features/conftest.py`:

- Add a `te_draft_picks` fixture mirroring PR #26's `wr_draft_picks` — 4 synthetic entries: veteran TE drafted in 2020 (finite `draft_age`), veteran TE drafted in 2019 (finite `draft_age`), veteran TE with NaN `draft_age` branch, rookie TE drafted in 2024.

In `tests/conftest.py`:

- Extend `baseline_weekly_stats_te` (and the parallel `baseline_features_te` builder) to cover **17 weeks of 2023 + 17 weeks of 2024 + 4 weeks of 2025** — same precedent PR #26 set on the WR equivalents at the same file. Trajectory trends need 8+ prior active games; without 2023 history, the trend cols are NaN on every 2024 row and `BaselineModel.fit`'s dropna empties the training set. Same depth-chart / NGS / schedule coverage extension to match.

In synthetic TE fixtures across `tests/test_features/test_cache.py`, the lightgbm/ensemble TE smoke fixtures (`tests/test_models/test_lightgbm_te.py`, `tests/test_models/test_baseline_te.py`, etc.), and any other site that constructs a minimal TE features row:

- Special-case `age` (any value in [22, 30]), `is_rookie` (0 or 1), `volume_trend_l4_minus_prior_l4` (small finite, e.g., 0.5), `snap_pct_change_l4_vs_prior_l4` (small finite, e.g., 0.05).
- The implementation plan includes a defensive grep for `opp_allowed_te_fppg_l4` (the last existing column in `_TE_FEATURE_COLUMNS`) to find every minimal-TE-row construction site, mirroring PR #26's WR grep. Same pattern caught 3 cluster-A leftovers for WR (commits `1f1f415`, `33eea57`, `807f046`); TE likely has analogous sites.

---

## 4. Real-data execution sequence (run-once, reports committed)

1. Code changes land + tests pass + lint + typecheck clean (criterion §1.3.1).
2. Verify `data/raw/draft_picks/` exists with seasons covering at least 1980-2024. If not, `python -c "from projections.ingest.refresh_draft_picks import refresh_draft_picks; refresh_draft_picks(data_root=Path('data'), seasons=range(1980, 2025))"` first.
3. `python scripts/refresh_features.py te --seasons 2018-2024` — regenerates TE feature cache. Verify schema validation passes on every (season, week) partition (criterion §1.3.2). Output is not committed (lives under `data/features/te/...`, gitignored convention).
4. Quick coverage check: per-(season, week) NaN rate on `age` / `is_rookie` / `volume_trend_l4_minus_prior_l4` / `snap_pct_change_l4_vs_prior_l4` for the eval window 2021-2024. Should approximate the probe's TE coverage (95.4% / 95.4% / 44.7% / 71.1%). If materially different, the builder wiring is wrong — investigate before running the gate.
5. `python scripts/backtest.py --position TE --update-snapshot` — runs the walk-forward backtest on all 5 model classes for TE on holdout years 2021-2024. Captures per-row prediction frames (criterion §1.3.3). Snapshot updates committed.
6. `python scripts/adoption_gate.py --position TE --baseline-run <pre-pr-sha> --candidate-run <branch-sha> --coverage-threshold 0.35` — produces per-(model_class, TE) verdicts. Output: `reports/adoption_gate_te_trajectory_features.{md,csv}`. Commit.
7. Write `reports/te_trajectory_features_summary.md` with: probe-predicted (-0.0107) vs gate-measured magnitudes on `(lgb-nb, TE)`; per-(model_class, TE) verdicts; coverage statistics from step 4; binding-cell shift rationale; ship/revert decision per §1.3.5; explicit deferred-follow-up note for the cross-class TE production routing question. Commit.
8. **If `(lgb-nb, TE)` verdict is `ADOPT`** (and `(baseline, TE)` is not REGRESSION): update `project_management.md` (top-of-file decision-log entry, format matches PR #26's entry but with the `(LightGBMNbModel, TE)` binding-cell explicit) + `TODO.md` #24 (record TE shipped, with measured magnitude). Push branch + open PR.
9. **If `(lgb-nb, TE)` verdict is `ADOPT` AND `(baseline, TE)` is `REGRESSION`:** revert *only* the `_TE_FEATURE_COLUMNS` extension in `baseline.py`. Keep the schema edit + builder edit + tests. Push branch + open PR with the modified-shape note prominent in the summary. (Probability low — see §5.)
10. **If `(lgb-nb, TE)` verdict is `MARGINAL` or `DO_NOT_ADOPT`:** revert the schema + builder changes (keep the test fixtures and any ancillary helper extensions). Document the divergence in `project_management.md` + `TODO.md` #24. Push branch with the revert + summary; open PR labeled "documentation-only" (no code change to ship).

---

## 5. Risks

- **Probe-vs-gate divergence on magnitude.** The probe predicted -0.0107 fpts on `(LightGBMNbModel, TE) augment`; the real gate could measure something materially different. PR #20 → PR #21 matched to 4 decimals on RB; PR #25 → PR #26 matched to ~10% on WR (-0.0414 → -0.0371, within probe CI). For TE at the smaller -0.0107 magnitude, ~10% calibration error is ~0.001 fpts — within the per-cell noise floor. The §1.3.5 rule binds on the verdict label, not magnitude.
- **Probe-vs-gate divergence on verdict.** TE's probe magnitude is the smallest of the 3 ADOPT cells. A small calibration error in the wrong direction could flip the gate to MARGINAL or DO_NOT_ADOPT. The revert path (§4 step 10) is defined.
- **`(baseline, TE)` REGRESSION risk.** Probe was DO_NOT_ADOPT (CI bracketed zero) on baseline TE, not REGRESSION (CI strictly above zero). For the production gate to flip baseline TE to REGRESSION, the gate would need to measure a strictly-positive RMSE delta on baseline TE — possible but unlikely given the probe's null result. The §1.3.5 modified-shape ship path covers this; the implementation plan's Phase 5 is conditional on the gate verdict.
- **`baseline.py:_TE_FEATURE_COLUMNS` miss.** PR #21 caught this at commit `9895dee` (RB), PR #26 caught it for WR — same pattern recurs for TE. The implementation plan explicitly schedules it as its own task with a smoke-test verification.
- **Coverage threshold gate failure.** If the production builder produces materially different NaN coverage than the probe override (e.g., TE bye-week filter or depth-chart dedupe differs from the probe's index path), the gate could spuriously fail on `--coverage-threshold 0.35`. Mitigate per §4 step 4: cross-check coverage stats post-refresh-features against the probe's measured coverage before running the gate. If coverage diverges, fix the builder, not the threshold.
- **`draft_picks` partition missing on a fresh checkout.** §4 step 2 covers it; the implementation plan also asserts the file exists in a Phase 0 sanity check before running the gate.
- **Empty-draft-picks silent degradation.** If `_EMPTY_DRAFT_PICKS` is passed (the kwarg default), `age` falls back to `season - inferred_year + 22.0` for every row — predictions still run, but the feature value is biased and audit-only. PR #26 already updated all 4 caller scripts to load + thread `draft_picks`; this PR inherits that wiring, no new caller-side risk.
- **TE feature cache invalidation.** Adding 4 columns to `TeFeaturesSchema` invalidates the existing TE cache under `data/features/te/...` — the schema validate would reject old rows missing the new columns. The spec calls this out in §4 step 3 and runs the refresh explicitly before any backtest invocation that reads the cache. Same pattern as PR #21's RB cache invalidation and PR #26's WR cache invalidation.
- **Ensemble-model snapshot regen.** EnsembleModel weights live under `data/ensemble_weights/ensemble_te_*.json`. Adding features to the TE schema changes the per-stat sub-model fit and therefore the ensemble weight optimizer's per-stat optima. The full backtest snapshot regen (§4 step 5) will rewrite both `tests/backtest/model_metrics.json` rows AND `data/ensemble_weights/ensemble_te_*.json` files. The implementation plan notes the ensemble weight files as expected snapshot churn, not a regression signal. Same pattern as PR #26's WR ensemble weight churn.
- **First binding-cell shift in project history.** PR #21 / PR #26 set the precedent that integration specs bind on `(BaselineModel, position)`. This spec deliberately breaks that precedent because TE's probe SIGNAL lives on lgb-nb, not baseline. Documented prominently in §1, §1.3.5, and the summary report. Future "integrate feature family X into position P" specs that follow this pattern (probe SIGNAL on a non-default class) will reference this spec as precedent.

---

## 6. Documentation updates on merge

- **`project_management.md`:** Append a top-of-file decision-log entry. Format matches PR #26's entry — title, status, verdict, what shipped or reverted, magnitude, probe-vs-gate calibration note, **plus** an explicit binding-cell-shift section noting that this is the project's first integration to bind on a non-default model class.
- **`TODO.md` #24:** Record the production integration outcome for TE (shipped / reverted / shipped-modified-shape, with measured magnitude). Cross-reference the summary report and PR #26's WR shipped state. Note that the "trailing-8-game-unit" branch of the trajectory candidate is now closed for both WR (PR #26) and TE (this PR) at all three ADOPT cells from PR #25; refined-unit candidates (`age²`, `is_2nd_year` flags, longer trailing windows, `has_trajectory_history` indicator) remain unexplored under the same TODO.
- **`docs/superpowers/specs/2026-05-03-trajectory-feature-family-probe-design.md`:** No changes. The probe spec stays as historical record.
- **`docs/superpowers/specs/2026-05-03-wr-trajectory-features-design.md`:** No changes. PR #26's spec stays as historical record.
- **`CONTRIBUTING.md`:** No changes. PR #26 already updated the "Regenerating the trajectory override" subsection to cover the production-builder hookup; the TE addition is mechanical reuse of the same path.
- **(Cross-class TE production routing follow-up):** Add a `TODO.md` note (under TODO #24 or a new entry) flagging the cross-class question explicitly: "TE production routes to `baseline` per Plan 8 (2026-04-29). With trajectory cols now in `TeFeaturesSchema`, a cross-class re-eval (`scripts/adoption_gate.py --baseline-run <pre-PR> --candidate-run <post-PR> --position TE` comparing `lightgbm-nb` candidate to `baseline` baseline) could justify flipping `_PositionDispatch[TE].default_model_class` to `lightgbm-nb`. Not load-bearing for any current consumer; queue alongside the next TE-related work."

---

## 7. Implementation phasing

The implementation plan should structure work in phases per the CLAUDE.md "PHASED EXECUTION" rule (≤5 files per phase). Suggested phasing:

- **Phase 1 — Schema + builder (2 files).** `schemas.py` (4 columns to `TeFeaturesSchema`); `features/te.py` (wire `attach_trajectory_features`, consume the existing `draft_picks` kwarg). Verify: existing TE feature builder tests still pass (likely some will fail due to the new schema cols not being present in fixtures — that's expected, fix in Phase 2); pyrightcheck/mypy clean on these 2 files standalone.
- **Phase 2 — Test fixtures + new TE tests (3 files).** `tests/test_features/conftest.py` (add `te_draft_picks` fixture); `tests/conftest.py` (extend `baseline_weekly_stats_te` + `baseline_features_te` to 17/17/4 weeks); `tests/test_features/test_te.py` (4 new trajectory tests + happy-path extension). Verify: `pytest tests/test_features/test_te.py` passes; full TE feature builder tests pass.
- **Phase 3 — `_TE_FEATURE_COLUMNS` + cluster-A leftover fixtures (≤5 files).** `models/baseline.py` (extend `_TE_FEATURE_COLUMNS`); defensive grep for `opp_allowed_te_fppg_l4` to find every site building a synthetic TE features row (likely `tests/test_features/test_cache.py`, `tests/test_models/test_lightgbm_te.py`, `tests/test_models/test_baseline_te.py`, possibly more); add `age`/`is_rookie`/trend col defaults to each. Verify: smoke test asserts the 4 new names in `_TE_FEATURE_COLUMNS`; smoke test asserts the lightgbm-family dynamic feature derivation includes the 4 new names; full pytest suite passes; mypy + ruff clean.
- **Phase 4 — Real-data execution + reports (no code).** §4 steps 2-7. Output: refreshed TE cache, backtest snapshot delta (`tests/backtest/model_metrics.json` rows for TE × all 5 classes), `data/ensemble_weights/ensemble_te_*.json` files regen, adoption gate report (`reports/adoption_gate_te_trajectory_features.{md,csv}`), summary report (`reports/te_trajectory_features_summary.md`). The gate result determines which branch of Phase 5 fires.
- **Phase 5 — Conditional code adjustments + documentation (1-5 files).** Branches on Phase 4's gate verdict per §1.3.5:
  - **(lgb-nb, TE) ADOPT, (baseline, TE) not REGRESSION** (ship-as-designed): no code adjustments. Update `project_management.md` + `TODO.md` per §6 (2 files).
  - **(lgb-nb, TE) ADOPT AND (baseline, TE) REGRESSION** (ship-modified-shape): revert the 4-col extension in `models/baseline.py:_TE_FEATURE_COLUMNS` (1 file). Re-run the baseline TE backtest cell only (snapshot regen for the 4 baseline TE rows in `tests/backtest/model_metrics.json`; ensemble weights stay as-is since EnsembleModel pulls from C-NB which is schema-derived). Update `project_management.md` + `TODO.md` (+ summary report addendum noting the modified-shape branch fired). 3-4 files total.
  - **(lgb-nb, TE) MARGINAL or DO_NOT_ADOPT** (revert): revert the schema edit in `schemas.py`, the builder edit in `features/te.py`, the `_TE_FEATURE_COLUMNS` extension in `baseline.py`, and the cluster-A fixture additions. Keep the `te_draft_picks` fixture and the `baseline_weekly_stats_te` extension (useful for any future TE-trajectory revisit). Re-run snapshot regen for TE × all 5 classes. Update `project_management.md` + `TODO.md` documenting the divergence. 4-5 files total.

This phasing keeps each step ≤5 files. Phase 1 may produce test failures (fixtures don't yet have the new schema cols) — that's expected and Phase 2 resolves them in the same logical unit. Phase 3 isolates the baseline feature-list edit and the cluster-A fixture grep into a single phase so the smoke-test verification covers both. Phase 5's conditional structure keeps the implementation plan's per-task definition crisp — only one branch executes.
