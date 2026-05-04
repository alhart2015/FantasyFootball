# WR Trajectory Features Integration — Design

**Status:** approved (brainstorming, 2026-05-03). Ready for implementation plan.
**Date:** 2026-05-03
**Author:** alden + claude
**Sub-project of:** FantasyFootball / Projections Core
**Builds on:** Trajectory Feature Family Probe (PR #25, merged at `d045eb8`) — the probe shipped 4 trajectory compute fns + assembler in `src/projections/features/trajectory_features.py`, the new `refresh_draft_picks` ingest module + `DraftPicksSchema`, and an override-generator script. The probe returned `SIGNAL` with three Phase-2 ADOPT cells: WR augment baseline (-0.0414 fpts), WR augment lgb-nb (-0.0194 fpts), TE augment lgb-nb (-0.0107 fpts). This spec promotes the WR cell into the production WR feature pipeline.
**Branch:** `feat/wr-trajectory-features` cut from `main` at `d045eb8`.

---

## 1. Overview

The trajectory family probe greenlit a follow-up integration for the binding cells. WR is the strongest signal carrier — it ADOPT'd under both BaselineModel and lgb-nb in augment mode. This spec executes the WR integration: append 4 nullable-float columns to `WrFeaturesSchema`, modify `build_wr_features` to compute and join the 4 trajectory features onto WR rows, refresh the WR feature cache, and run the full backtest + adoption gate to verify the probe's ADOPT prediction holds on `(BaselineModel, WR)`.

The work is intentionally narrow: WR only. QB / RB / TE feature schemas and builders are not touched. The probe explicitly returned regression on QB augment-mode (both model classes), net-zero on RB (mechanism-consistent — RB rushing is more team-script-driven than career-arc-driven; PR #21's team-level PBP cols already covered RB's binding axis), and ADOPT on TE only under lgb-nb (not under BaselineModel). Integrating trajectory features into TE is a separate spec — it must grapple with whether to ship per-position routing (lgb-nb for TE only, baseline elsewhere) or to ship the schema change for the lgb-nb code path while leaving baseline production routing unchanged.

The shipping decision is binary and bound to the production model class (BaselineModel, "Model A"): if the adoption gate's per-position verdict on `(BaselineModel, WR)` is `ADOPT`, ship. If it is `MARGINAL` or `DO_NOT_ADOPT`, revert per Plan 9 discipline. Other model classes' verdicts (`LightGBMTunedModel`, `LightGBMNbModel`, `EnsembleModel`) are informational — captured in the summary report but not gating.

### 1.1 Goals (in scope)

- Append 4 nullable-float columns to `WrFeaturesSchema` in `src/projections/schemas.py`:
  - `age: Series[float] = pa.Field(ge=15, le=50, nullable=True)` — biological age in the target season; primary path uses `draft_age + (season - draft_year)` from `nfl_data_py.import_draft_picks`, fallback path uses `season - inferred_draft_year + 22.0` where `inferred_draft_year` is earliest weekly_stats appearance.
  - `is_rookie: Series[float] = pa.Field(ge=0, le=1, nullable=True)` — 1.0 if `season == draft_year` else 0.0; same draft_lookup + inferred-fallback resolution as `age`.
  - `volume_trend_l4_minus_prior_l4: Series[float] = pa.Field(nullable=True)` — trailing-4-active-games mean of `targets` minus prior-4-active-games mean. NaN for players with fewer than 8 prior active games.
  - `snap_pct_change_l4_vs_prior_l4: Series[float] = pa.Field(ge=-1, le=1, nullable=True)` — same window cut on `SnapCountsSchema.offense_pct`. Bounded since `offense_pct ∈ [0, 1]`.

  The audit-only `draft_year_inferred` boolean column from the probe override is **not** added — it is metadata for tracking fallback frequency, not a feature. The probe over-the-wire shape preserves it because `build_trajectory_overrides` writes it for downstream audit; the production schema drops it via `strict="filter"`.

- Promote `_build_draft_lookup` from `scripts/build_trajectory_override.py` to a public helper `build_draft_lookup(draft_picks: pd.DataFrame) -> DraftLookup` in `src/projections/features/trajectory_features.py`. The override script imports it; `build_wr_features` consumes it. One canonical conversion path, no script-side duplication. The override script's local definition is removed.

- Modify `build_wr_features` in `src/projections/features/wr.py` to:
  - Accept a new keyword arg `draft_picks: pd.DataFrame = _EMPTY_DRAFT_PICKS` (mirrors the existing unused-but-plumbed `pbp` kwarg pattern).
  - After the existing `out` frame has `(gsis_id, season, week, team, opponent)`, build a slim index `out[["gsis_id", "season", "week", "team", "opponent"]].rename(columns={"opponent": "opp"})`.
  - Build `draft_lookup = build_draft_lookup(draft_picks)`.
  - Call `attach_trajectory_features(idx, ws, sc, draft_lookup, Position.WR)` — note: pass the **unfiltered-by-position** prior-mask-filtered weekly_stats (`ws`, not `ws_wr`) and snap_counts (`sc`, not `sc_wr`). The helper's `_volume_trend` filter handles position internally.
  - Take the 4 trajectory feature columns (drop `team`, `opp`, `draft_year_inferred`, identity cols already in `out`); merge onto `out` on `(gsis_id, season, week)`.
  - The final `WrFeaturesSchema.validate(out)` enforces presence + dtype + bounds.

- Update `src/projections/models/baseline.py:_WR_FEATURE_COLUMNS` (line 266) to include the 4 new column names. **This is the spec gap PR #21 caught at commit `9895dee` for `_RB_FEATURE_COLUMNS`** — `baseline.py` hardcodes the per-position feature tuple while the lightgbm family derives from `WrFeaturesSchema.to_schema().columns.keys()` dynamically. Without this update, `BaselineModel.fit` will not see the new features even though the schema validates. The implementation plan calls this out as its own dedicated task.

- Update `tests/test_features/test_wr.py`:
  - Extend the existing happy-path test to assert the 4 new columns are present and float-typed.
  - Add `test_wr_features_attach_trajectory_join` — synthetic fixture: a WR with 8+ prior active games has a numerically-verifiable `volume_trend_l4_minus_prior_l4` and `snap_pct_change_l4_vs_prior_l4`; a rookie WR has `is_rookie=1.0`, `age` matching `draft_age + (season - draft_year)`, and `volume_trend=NaN`; a UDFA WR with no draft_lookup entry uses the inferred-draft-year fallback path.
  - Add a small unit test for the promoted `build_draft_lookup` helper covering the NaN-`draft_age` branch (rare but real per `DraftPicksSchema`).

- Add a new keyword arg `draft_picks: pd.DataFrame = _EMPTY_DRAFT_PICKS` to the `build_qb_features`, `build_rb_features`, and `build_te_features` signatures for symmetry with the new WR builder signature — accept-and-ignore today, consumed when their own integration plans fire. **This matches the existing `pbp` plumbing precedent** (TODO #3a: "`pbp` keyword arg is threaded through every per-position `build_<pos>_features` signature with `_EMPTY_PBP` default — currently unused by builders, reserved for the next PBP-driven feature plan"). Without it, `refresh_features.py` would need an asymmetric `if pos == "wr": ... else: ...` branch on the call site, which is ugly and divergent from the `pbp` precedent.

- Modify `scripts/refresh_features.py` to thread `draft_picks` through to all 4 `build_<pos>_features` calls (mirrors how `pbp` is threaded today). Load via `_read_concat(raw_root, "draft_picks", list(range(1980, seasons.stop)))` matching the override script's existing convention.

- Modify the three other direct-builder scripts that call `build_wr_features` — `scripts/train_baseline.py`, `scripts/predict_2024.py`, `scripts/sanity_check_baseline.py` — to load and pass `draft_picks` (same `_read_concat` pattern). Without these the WR builder defaults to empty draft_picks and every row routes to the inferred-fallback path silently — predictions still run, but `age` is wrong.

- Refresh the WR feature cache: `python scripts/refresh_features.py wr --seasons 2018-2024`. (Manual; run-time output, not committed.)

- Run the full backtest + adoption gate on WR only, all 4 model classes. Commit the resulting `reports/adoption_gate_wr_trajectory_features.{md,csv}`.

- Write `reports/wr_trajectory_features_summary.md` consolidating: probe-predicted vs gate-measured magnitudes; per-model-class verdicts; coverage statistics; ship/revert decision.

- On `ADOPT` verdict: update `project_management.md` decision log + `TODO.md` #24 (record shipped, with measured magnitude; cross-link to TODO #3c precedent). On `MARGINAL` / `DO_NOT_ADOPT`: revert builder + schema changes (the spec leaves the `build_draft_lookup` promotion in place — useful for the deferred TE follow-up regardless of WR's outcome).

### 1.2 Non-goals (deferred)

- **No QB / RB / TE schema changes.** TE adopted only under lgb-nb (-0.0107 fpts), which raises the per-position-routing question (TE production routes to `BaselineModel` today). A TE integration spec must pick: (a) ship per-position routing to lgb-nb for TE only, precedent Plan 6's QB-only ensemble suggestion; or (b) ship the schema change for the lgb-nb code path while leaving baseline production routing unchanged. Either is its own decision; out of scope here. RB returned net-zero (mechanism-consistent: PR #21's team-level PBP already covers RB's binding axis). QB returned regression on **both** model classes — explicitly do-not-integrate.
- **No new ingest.** `refresh_draft_picks` shipped in PR #25 and the partition exists at `data/raw/draft_picks/season=YYYY/part.parquet`. The opt-in `--run-network` smoke at `tests/test_ingest/test_api_drift.py` already covers `nfl_data_py.import_draft_picks` column-rename drift.
- **No per-feature ablation.** The probe tested all 4 features bundled. Production-Ridge regularization shrinks uninformative coefficients toward 0, so shipping all 4 doesn't degrade prediction quality vs shipping the 1-2 load-bearing ones. A per-feature ablation is a "nice to know" follow-up, not a prerequisite.
- **No widening to other model classes' production routing.** `POSITION_DISPATCH[WR].factories['default']` stays at `BaselineModel`. The new features are added to the schema (which is shared across model classes), so all model classes consume them in the gate run, but the production routing decision is unchanged.
- **No new probe machinery.** The probe code (`src/projections/backtest/feature_probe.py`, `scripts/probe_feature_signal.py`) is not modified. The summary report compares probe-vs-gate calibration but does not re-run the probe.
- **No spec / plan file changes for prior work.** PR #25's spec, plan, and reports stay as historical record.
- **No retroactive integration of `draft_year_inferred` audit metadata into a separate audit table.** The override parquet preserves it for one-shot audit (the probe summary measured 22.6% fallback rate); production does not need an ongoing audit channel for it. If fallback frequency becomes a debugging concern after shipping, surface it via a one-off CLI rather than a persisted column.

### 1.3 Success criteria

The spec is complete iff all of:

1. **Schema + builder + tests + script-plumbing land cleanly.** `pytest -v` (full suite), `mypy src tests` (zero violations), `ruff check src tests scripts` (zero violations), `ruff format --check` (no drift).
2. **Refreshed WR feature cache validates against the extended schema** at every `(season, week)` partition.
3. **The full backtest + adoption gate runs successfully on WR** for all 4 model classes (baseline / lightgbm-tuned / lightgbm-nb / ensemble) across the standard `2021-2024` holdout years, **with `--coverage-threshold 0.35`** to match the probe's structural-sparsity precedent.
4. **The summary report (`reports/wr_trajectory_features_summary.md`) records all of:**
   - The probe's predicted composite RMSE delta on `(BaselineModel, WR) augment`: **-0.0414 fpts** (from PR #25's `feature_probe_trajectory_augment.csv`).
   - The gate's measured composite RMSE delta on `(BaselineModel, WR)` with 95% CI.
   - The per-(model_class, WR) verdicts for the other 3 model classes — informational; the lgb-nb cell is the cross-check on the probe's second WR ADOPT cell (-0.0194 fpts).
   - Per-position coverage of the 4 new columns at the eval window (2021-2024) and on the full 2018-2024 history; documented next to the threshold relaxation rationale.
5. **The shipping decision matches the gate verdict on `(BaselineModel, WR)`:**
   - `ADOPT` → merge PR; update decision logs.
   - `MARGINAL` or `DO_NOT_ADOPT` → revert builder + schema changes (keep the `build_draft_lookup` promotion); document the divergence.

If criterion 1 fails, fix and rerun. If criterion 2 fails, the builder is wrong — fix before running the gate. Criterion 3 is mechanical (the gate either runs or doesn't). Criterion 5 is the binding decision.

---

## 2. Inputs

### 2.1 Draft-picks source

`draft_picks` partitions read via `read_partition(raw_root, "draft_picks", season=s)` for the full historical range `[1980, seasons.stop)` — PR #25's `refresh_draft_picks` partitions one file per draft year going back to 1980, and `build_draft_lookup` collapses the multi-year frame into `{gsis_id: (draft_year, draft_age)}`. Range goes back to 1980 because `nfl_data_py.import_draft_picks` exposes that as the practical lower bound for player-level coverage; older players (pre-1980 draft) route to the inferred-fallback path.

Empty fallback: if `data/raw/draft_picks/` does not exist (clean checkout, partition not refreshed yet), `build_wr_features` produces an empty `draft_lookup`, every row routes to the inferred-fallback `age` and `is_rookie` resolution, and the schema's `nullable=True` accepts the resulting NaN for cold-start cases. Predictions still run; `age` is biased toward the 22.0 fallback offset. The implementation plan ensures the production execution sequence (§4) refreshes draft_picks before refreshing features.

### 2.2 Player-team-week index inside `build_wr_features`

`build_wr_features` already produces an internal `(gsis_id, season, week, team, opponent)` frame from `depth_charts` (filtered to WRs in `as_of_week`, deduped per TODO #9c) inner-joined with `schedules` (bye-week filter per TODO #9a). The trajectory integration reuses this frame, renames `opponent → opp` to match `attach_trajectory_features`'s contract, passes it through.

No new ingest source. No new schema-level changes outside `WrFeaturesSchema`.

### 2.3 Weekly_stats and snap_counts contracts

Both must satisfy their respective schemas (caller's responsibility; existing convention). The `attach_trajectory_features` helper (per `trajectory_features.py:273`) trusts these contracts; defensive normalization is not added.

The `weekly_stats` and `snap_counts` passed to the helper are the **prior-mask-filtered** frames (`ws`, `sc`) — the existing leakage filter `prior_mask` already restricts these to rows strictly before `(season, week)`, which is the correct backfill scope for trailing-8-active-games rolling. They are **not** position-filtered (the helper's `_volume_trend` filters to `Position.WR.value` internally).

If `weekly_stats` is empty (would happen at season 1, week 1 of the entire history with no prior coverage), the helper's compute fns each return empty frames and the left-merges produce all-NaN rows — same shape as a successful call where every row's trailing-8 has fewer than 8 prior games. The schema's `nullable=True` accepts this.

---

## 3. Code shape

### 3.1 Promoted helper `build_draft_lookup`

Public function in `src/projections/features/trajectory_features.py`:

```python
def build_draft_lookup(draft_picks: pd.DataFrame) -> DraftLookup:
    """Convert a draft_picks DataFrame into a {gsis_id: (draft_year, draft_age)} lookup.

    Args:
        draft_picks: frame matching ``DraftPicksSchema``. Must include
            ``gsis_id``, ``draft_year``, ``draft_age``. ``draft_age`` may be
            NaN (rare; per ``nfl_data_py``, missing for players whose
            birthdate isn't in the source). Empty frame returns ``{}``.

    Returns:
        Lookup keyed by ``gsis_id``. Missing keys (UDFAs, pre-1980
        draftees) route to the inferred-draft-year fallback inside
        ``compute_age`` / ``compute_is_rookie``.
    """
    if draft_picks.empty:
        return {}
    return {
        str(row["gsis_id"]): (
            int(row["draft_year"]),
            float(row["draft_age"]) if pd.notna(row["draft_age"]) else float("nan"),
        )
        for _, row in draft_picks.iterrows()
    }
```

`scripts/build_trajectory_override.py:106-114` (the existing private `_build_draft_lookup`) is removed; the script imports `build_draft_lookup` from `trajectory_features`.

### 3.2 `build_wr_features` integration

In `src/projections/features/wr.py`, after the existing assembly that produces `out` with `(gsis_id, season, week, team, opponent, ...)` columns:

```python
# Trajectory features (PR #25 family probe + this spec's WR integration).
# Helper does its own position filter and rolling, so pass the unfiltered
# (but prior-mask-filtered) weekly_stats / snap_counts.
draft_lookup = build_draft_lookup(draft_picks)
traj_idx = (
    out[["gsis_id", "season", "week", "team", "opponent"]]
    .rename(columns={"opponent": "opp"})
)
traj = attach_trajectory_features(traj_idx, ws, sc, draft_lookup, Position.WR)
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

The function signature gains one keyword arg `draft_picks: pd.DataFrame = _EMPTY_DRAFT_PICKS`. The unused-but-plumbed `pbp` kwarg precedent extends to `draft_picks` — TE / RB / QB builders gain the same default for symmetry across positions; this spec only wires it through WR.

### 3.3 Schema change

In `src/projections/schemas.py`, append to `WrFeaturesSchema`:

```python
# Trajectory features (PR #25 family probe + 2026-05-03 WR integration
# spec). All four are structurally sparse: age + is_rookie need a
# draft_picks lookup hit (or the inferred fallback) and so cover ~88-97%
# of player-weeks; the trend cols need 8 prior active games and so cover
# ~50% of player-weeks. NaN where coverage is missing; BaselineModel
# imputes with feature mean, lightgbm consumes NaN natively. See spec
# §1.3 criterion 4 for the coverage measurement requirement.
age: Series[float] = pa.Field(ge=15, le=50, nullable=True)
is_rookie: Series[float] = pa.Field(ge=0, le=1, nullable=True)
volume_trend_l4_minus_prior_l4: Series[float] = pa.Field(nullable=True)
snap_pct_change_l4_vs_prior_l4: Series[float] = pa.Field(ge=-1, le=1, nullable=True)
```

`age` lower bound 15 is conservative (a few HS draftees historically; modern era is ≥21). Upper bound 50 covers Tom Brady at 47 with margin. `snap_pct_change_l4_vs_prior_l4` bounded since `offense_pct ∈ [0, 1]` so the difference is `[-1, 1]`. `volume_trend_l4_minus_prior_l4` is unbounded (can go arbitrarily negative for declining-usage players who lost role).

Strict mode ("filter") on the schema's `Config` already drops `draft_year_inferred` and any other helper-output columns that the production schema doesn't declare; no other change needed.

### 3.4 `baseline.py` hardcoded feature list

In `src/projections/models/baseline.py`, extend `_WR_FEATURE_COLUMNS` (line 266-288) to include the 4 new column names:

```python
_WR_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    # ... existing 21 columns ...
    "opp_allowed_wr_fppg_l4",
    # Trajectory features (PR #25 family probe + 2026-05-03 WR integration).
    "age",
    "is_rookie",
    "volume_trend_l4_minus_prior_l4",
    "snap_pct_change_l4_vs_prior_l4",
)
```

`src/projections/models/lightgbm.py:122` (and the inheriting `lightgbm_tuned.py` / `lightgbm_nb.py` via the imported `_WR_FEATURE_COLUMNS`) derives feature columns dynamically from `WrFeaturesSchema.to_schema().columns.keys()` filtered through `_NON_FEATURE_COLUMNS`, so the lightgbm family auto-picks-up the 4 new columns when the schema change lands. **No edit needed to the lightgbm files** — the implementation plan asserts this with a smoke test that reads `_filter_features(_WR_FEATURE_COLUMNS)` and verifies the 4 new columns appear post-schema-change.

### 3.5 Tests

In `tests/test_features/test_wr.py`:

- Extend the existing happy-path test to assert the 4 new columns are present, float-typed, and bounded per the schema.
- Add `test_wr_features_attach_trajectory_join_drafted_player` — synthetic fixture: a drafted WR (in `draft_lookup`) with 8+ prior active games. Assert: `age = draft_age + (season - draft_year)`, `is_rookie = 0.0` (not their draft year), `volume_trend_l4_minus_prior_l4` matches a hand-computed mean-of-trailing-4-targets minus mean-of-prior-4-targets, `snap_pct_change_l4_vs_prior_l4` matches the analog on `offense_pct`.
- Add `test_wr_features_attach_trajectory_join_rookie` — rookie WR (`season == draft_year`): `is_rookie = 1.0`, `age` matches `draft_age` exactly, `volume_trend_l4_minus_prior_l4 = NaN` (no prior 8 games), `snap_pct_change_l4_vs_prior_l4 = NaN`.
- Add `test_wr_features_attach_trajectory_join_udfa` — WR with no `draft_lookup` entry: uses inferred-draft-year fallback. Assert `is_rookie = (season == inferred_draft_year)`, `age = season - inferred_draft_year + 22.0`.
- Add `test_wr_features_empty_draft_picks_fallback` — `draft_picks=_EMPTY_DRAFT_PICKS`: every row routes to inferred fallback, no errors raised, schema validates.

In `tests/test_features/test_trajectory_features.py`:

- Add `test_build_draft_lookup_*` covering: empty frame returns `{}`; happy-path drafted player produces `(draft_year, draft_age)` tuple; NaN-`draft_age` branch produces `(draft_year, float('nan'))`.

In `tests/test_scripts/test_build_trajectory_override_cli.py`:

- The existing tests should continue to pass after `_build_draft_lookup` is removed and replaced with the imported `build_draft_lookup`. The contract is unchanged (one canonical path now).

---

## 4. Real-data execution sequence (run-once, reports committed)

1. Code changes land + tests pass + lint + typecheck clean (criterion §1.3.1).
2. Verify `data/raw/draft_picks/` exists with seasons covering at least 1980-2024. If not, `python -m scripts.refresh_draft_picks --seasons 1980-2024` first (the partition was created by PR #25 ingest but a fresh checkout may not have it).
3. `python scripts/refresh_features.py wr --seasons 2018-2024` — regenerates WR feature cache. Verify schema validation passes on every (season, week) partition (criterion §1.3.2). Output is not committed (lives under `data/features/wr/...`, gitignored convention).
4. Quick coverage check (one-liner script or notebook): per-(season, week) NaN rate on `age` / `is_rookie` / `volume_trend_l4_minus_prior_l4` / `snap_pct_change_l4_vs_prior_l4` for the eval window 2021-2024. Should match the probe's coverage ranges (WR: ~96.7% age / ~53.6% volume_trend / ~68.4% snap_pct). If materially different, the builder wiring is wrong — investigate before running the gate.
5. `python scripts/backtest.py --position WR --update-snapshot` — runs the walk-forward backtest on all 4 model classes for WR on holdout years 2021-2024. Captures per-row prediction frames (criterion §1.3.3). Snapshot updates committed.
6. `python scripts/adoption_gate.py --position WR --baseline-run <pre-pr-sha> --candidate-run <branch-sha> --coverage-threshold 0.35` — produces per-(model_class, WR) verdicts. **Note the threshold flag** — without it, the gate's pooled coverage check fails on the structurally-sparse trend cols. Output: `reports/adoption_gate_wr_trajectory_features.{md,csv}`. Commit.
7. Write `reports/wr_trajectory_features_summary.md` with: probe-predicted vs gate-measured magnitudes, per-(model_class, WR) verdicts, coverage statistics from step 4, ship/revert decision (criterion §1.3.4 + §1.3.5). Commit.
8. **If `(BaselineModel, WR)` verdict is `ADOPT`:** update `project_management.md` (top-of-file decision-log entry, format matches PR #21's entry) + `TODO.md` #24 (record shipped, with measured magnitude; close the trailing-8-game-unit branch of the trajectory candidate). Push branch + open PR.
9. **If `(BaselineModel, WR)` verdict is `MARGINAL` or `DO_NOT_ADOPT`:** revert the schema + builder changes in a new commit (keep the `build_draft_lookup` promotion — useful for the TE follow-up regardless). Document the divergence in `project_management.md` + `TODO.md` #24. Push branch with the revert + summary; open PR labeled "documentation-only" (no code change to ship).

---

## 5. Risks

- **Probe-vs-gate divergence on magnitude.** The probe predicted -0.0414 fpts on `(BaselineModel, WR) augment`; the real gate could measure something materially different. The §1.3.5 rule binds on the verdict label, not magnitude, so this is recorded but not gating. If divergence is dramatic in either direction, the summary report flags it for future probe-tuning. PR #20→#21 matched to 4 decimals (-0.0124 → -0.0124); PR #25's WR cell is ~3x larger, so the absolute error could be larger too.
- **Probe-vs-gate divergence on verdict.** If `(BaselineModel, WR)` returns `MARGINAL` or `DO_NOT_ADOPT`, the spec's revert path fires. This is the Plan 9-style outcome and is documented.
- **`baseline.py:_WR_FEATURE_COLUMNS` miss.** PR #21 caught this at commit `9895dee`; the implementation plan explicitly schedules it as its own task with a smoke-test verification (assert the 4 names appear in the tuple post-edit) to make the failure mode loud.
- **Coverage threshold gate failure.** If the production builder produces materially different NaN coverage than the probe override (e.g., bye-week filter or depth-chart dedupe differs), the gate could spuriously fail on `--coverage-threshold 0.35`. Mitigate per §4 step 4: cross-check coverage stats post-refresh-features against the probe's measured coverage before running the gate. If coverage diverges, fix the builder, not the threshold.
- **`draft_picks` partition missing on a fresh checkout.** §4 step 2 covers it; the implementation plan also asserts the file exists in a Phase 0 sanity check before running the gate.
- **Empty-draft-picks silent degradation.** If `_EMPTY_DRAFT_PICKS` is passed (e.g., a future caller forgets to wire `draft_picks`), `age` falls back to `season - inferred_year + 22.0` for every row — predictions still run, but the feature value is biased and audit-only. Mitigation: every direct-builder caller must be updated in this PR (§1.1 lists `train_baseline.py`, `predict_2024.py`, `sanity_check_baseline.py`); a future caller would surface the issue via per-position coverage drift in the next backtest snapshot.
- **`refresh_features.py` regression on QB/RB/TE.** Threading `draft_picks` through all 4 builders requires QB/RB/TE to accept-and-ignore the new kwarg. The default `_EMPTY_DRAFT_PICKS` makes this a no-op for those positions; the smoke test that runs all 4 position builders on a synthetic fixture (precedent: Plan 3b's parametrized smoke) catches accidental signature mismatches.
- **TODO #29 entanglement.** The lightgbm-tuned class is a pruning candidate (Model C-NB strictly dominates on RMSE). When the gate runs all 4 model classes, lightgbm-tuned will produce one of the 4 verdicts. This is informational per §1.3.4; the pruning is queued but not dependent on this spec.
- **WR feature cache invalidation.** Adding 4 columns to `WrFeaturesSchema` invalidates the existing WR cache under `data/features/wr/...` — the schema validate would reject old rows missing the new columns, and `read_partition` will fail. The spec calls this out in §4 step 3 and runs the refresh explicitly before any backtest invocation that reads the cache. Same pattern as PR #21's RB cache invalidation.

---

## 6. Documentation updates on merge

- **`project_management.md`:** Append a top-of-file decision-log entry. Format matches PR #21's entry — title, status, verdict, what shipped or reverted, magnitude, probe-vs-gate calibration note.
- **`TODO.md` #24:** Record the production integration outcome (shipped / reverted, with measured magnitude). Cross-reference the summary report. If `ADOPT`, close the "trailing-8-game-unit" branch of the trajectory candidate (refined-unit candidates from PR #25 — `age²`, `is_2nd_year` flags, longer trailing windows, `has_trajectory_history` indicator — remain unexplored under the same TODO).
- **`docs/superpowers/specs/2026-05-03-trajectory-feature-family-probe-design.md`:** No changes. The probe spec stays as-is.
- **`CONTRIBUTING.md`:** Update the "Regenerating the trajectory override" subsection to note that the override is now also referenced by the production WR builder via the `attach_trajectory_features` helper; the override path remains useful for probe re-runs (e.g., if a future spec re-tests the unit). No new instructions.

---

## 7. Implementation phasing

The implementation plan should structure work in phases per the CLAUDE.md "PHASED EXECUTION" rule (≤5 files per phase). Suggested phasing for the implementation plan to expand:

- **Phase 1 — Schema + helper promotion + override-script + helper tests (4 files).** `schemas.py` (4 columns to `WrFeaturesSchema`); `trajectory_features.py` (promote `build_draft_lookup`); `scripts/build_trajectory_override.py` (remove private `_build_draft_lookup`, import from `trajectory_features`); `tests/test_features/test_trajectory_features.py` (add `build_draft_lookup` tests). Verify: existing 48 tests pass; new helper tests pass; `scripts/build_trajectory_override.py --help` runs (smoke).
- **Phase 2 — WR builder integration + tests (2 files).** `features/wr.py` (wire `attach_trajectory_features` + `draft_picks` kwarg); `tests/test_features/test_wr.py` (4 new tests). Verify: full WR feature builder tests pass; happy-path emits all 25 columns.
- **Phase 3 — Per-position builder kwarg additions + baseline.py (4 files).** `features/qb.py`, `features/rb.py`, `features/te.py` (add `draft_picks: pd.DataFrame = _EMPTY_DRAFT_PICKS` kwarg, accept-and-ignore — mirror the existing `pbp` precedent); `models/baseline.py` (extend `_WR_FEATURE_COLUMNS` with the 4 new column names). Verify: existing per-position feature builder tests pass unchanged; smoke test asserts `_filter_features(_WR_FEATURE_COLUMNS)` (lightgbm) includes the 4 new names; the `_WR_FEATURE_COLUMNS` baseline tuple includes them too.
- **Phase 4 — Caller-script plumbing (4 files).** `scripts/refresh_features.py`, `scripts/train_baseline.py`, `scripts/predict_2024.py`, `scripts/sanity_check_baseline.py` (load `draft_picks` via `_read_concat`, pass to all 4 build_<pos>_features calls). Verify: each script runs end-to-end on the synthetic-fixture path used by its CLI tests; no signature-mismatch errors.
- **Phase 5 — Real-data execution + reports (no code).** §4 steps 2-7. Output: refreshed WR cache, backtest snapshot delta, adoption gate report, summary report. The gate result determines whether Phase 6 is "ship" or "revert."
- **Phase 6 — Documentation + decision-log update (2 files).** `project_management.md`, `TODO.md`. Conditional on Phase 5's gate verdict.

This phasing keeps each step ≤4 files, sequenced so a Phase-N failure doesn't block Phase-(N+1)'s diagnostic value.
