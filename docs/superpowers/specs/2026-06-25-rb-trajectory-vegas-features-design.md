# RB feature lift — trajectory + Vegas-preseason signal probe → conditional integration

**Date:** 2026-06-25
**Branch:** `feat/rb-feature-signal-probe`
**Status:** spec
**TODO:** #55 (DFS — model RB off `baseline`)

## 1. Problem & motivation

The DFS Layer-1 edge study (#54) returned **STOP**, and the per-position
breakdown (`reports/dfs_per_position_significance_2026-06-25.md`) showed the
loss is concentrated in **RB**: fraction **0.430** (95% CI 0.397–0.463) on the
no-effort `baseline` model, and the **largest disagreement subset** (1009 cells
/ 356 clusters), so RB is the single biggest drag on the pooled verdict. #55
named RB the highest-leverage modeling target.

Two cheap paths to lift RB are now closed by direct evidence:

1. **Decomposition** (the literal "WR-style" suggestion in #55) — the RB
   decomposition probe (`reports/feature_probe_rb_decomposition_summary.md`,
   2026-05-16) returned **NULL on all five composed stats**, and the receiving
   axis fails the 0.95 coverage bar (RBs frequently have 0 targets: 78–85%/yr).
2. **Model-class swap on the existing features** — a fresh RB-only walk-forward
   bake-off (2026-06-25) shows **`baseline` beats both `lightgbm-nb` and
   `ensemble` on every metric**, pooled and in nearly every year-cell:

   | Pooled (2021–2024) | baseline | ensemble | lightgbm-nb |
   |---|---|---|---|
   | composite_mae ↓ | **4.987** | 5.105 | 5.146 |
   | composite_rmse ↓ | **6.566** | 6.590 | 6.612 |
   | spearman_topN ↑ | **0.9694** | 0.9685 | 0.9673 |

   (This bake-off was run fresh this session via the RB-only walk-forward driver
   `scripts/_rb_model_bakeoff.py` over 2021–2024; Phase 0 persists it to
   `reports/rb_model_bakeoff_2026-06-25.md` alongside the decomposition summary
   so the numbers are checkable, not just cited inline.)

When a flexible gradient-boosted model (lightgbm) cannot beat a linear one
(ridge) on the **same** features, that is the textbook **signal-limited**
signature: there is no additional nonlinear/interaction signal in the current RB
feature set for a richer model to exploit. RB is therefore signal-limited **on
the current feature set** — a richer function class won't help — so the only
remaining lever is **new feature signal, not another model swap**. (This is a
claim about the *current* features only; adding new signal reopens the
model-class question, which §4 Phase 2 re-tests rather than assuming baseline
still wins.)

RB has a concrete, unexploited feature gap. It is missing exactly two feature
families that WR (and, for Vegas, QB) received and that materially helped them,
and that have **never been probed for RB**:

- **Trajectory** (WR PR #25, also TE PR #27): `age`, `is_rookie`,
  `volume_trend_l4_minus_prior_l4`, `snap_pct_change_l4_vs_prior_l4`.
- **Vegas preseason context** (WR+QB PR #51): `preseason_implied_team_total`,
  `preseason_spread`, `season_avg_implied_team_total`, `season_avg_spread`.

RB already has weather (PR #28) and PBP team-level features (PR #21) that WR
lacks — so this is a targeted two-family gap, not a wholesale one.

## 2. Goal & success criteria

Determine whether adding these families lifts RB, and if so, ship the lift. In
priority order:

1. **Real metric (reported, NOT gated):** RB's DFS edge-study fraction moves up —
   ideally its 95% CI reaches or exceeds **0.50** — and ideally nudges the pooled
   verdict off STOP. This is the *observational confirmation* the whole effort is
   aimed at, but it is **not** a pass/fail gate: the project does not branch on it
   (the gates that decide what ships are G1/G2 in §6). Phase 3 reports the
   before/after fraction honestly whichever way it lands. (Per #55, a single
   position may not flip the pool; we measure, not assume.)
2. **Cheap gate (necessary, decisive):** the feature family helps the model on the
   walk-forward backtest — the dual-run **adoption gate** returns ADOPT
   (`baseline` on augmented features vs `baseline` on old features), and a model
   class on the augmented features beats the augmented baseline in the re-run
   bake-off. Precise criteria and the ship-vs-revert branches are in §6 (G2).
3. **Honest dead-end (acceptable outcome):** if both families probe **NULL**,
   **STOP** — RB is confirmed unliftable with the signal available — write it up,
   update TODO/PM, and ship nothing. This is reached for ~1–2 hrs of compute and
   zero integration risk.

"Beat baseline" is defined precisely in §6 (decision gates).

## 3. Non-goals / out of scope

- **Decomposition** for RB — dead (NULL probe, coverage failure).
- **New modeling architectures** (usage/opportunity decomposition, committee/role
  models). Logged as a future option; not this slice.
- **Other positions** (QB still loses with a real model — separate question; TE
  is ~neutral).
- **DFS Layer 2** (salary-cap optimizer, ROI backtest) — gated on a real edge,
  unchanged by this work.
- **Vectorizing the `expected_points(...).apply(axis=1)` edge-study emit**
  (#55's noted ~75-min bottleneck). Phase 3 re-runs the emit only once or twice;
  we defer the perf refactor unless iteration proves painful. If we do it, it is
  a **separate** slice, not folded in here.

## 4. Architecture & phases

The work follows the repo's established **probe → integrate → adoption-gate**
discipline (the same path weather/PBP/trajectory/Vegas took, position by
position). Phases with hard decision gates between them.

### Phase 0 — Persist the baseline-to-beat (no source changes)

Run `scripts/_rb_model_bakeoff.py` (already authored this session) and write its
output to `reports/rb_model_bakeoff_2026-06-25.md`, so the §1 "baseline beats
lightgbm-nb and ensemble" claim rests on a committed artifact, not an inline
citation. This is the *old-feature-set* baseline the Phase 2 re-bake-off is
compared against. Trivial; do it first so the rest of the work has a fixed
reference point.

### Phase 1 — Signal probe (cheap, no source changes)

Use the existing pre-spec screening tool `scripts/probe_feature_signal.py`
(→ `src/projections/backtest/feature_probe.py`), which merges a candidate-column
**override parquet** onto the baseline feature set and emits per-stat
Δ-CV-RMSE bootstrap CIs with SIGNAL / NULL / REGRESSION verdicts (augment mode).

**Build the two override parquets** by **reusing the existing builder scripts**
(do NOT write new ones — per CLAUDE.md "reuse before writing"):

- `scripts/build_trajectory_override.py` already emits the trajectory family
  (`age`, `is_rookie`, `volume_trend_l4_minus_prior_l4`,
  `snap_pct_change_l4_vs_prior_l4`) for **all four** fantasy positions in one
  parquet, calling `attach_trajectory_features(index, weekly_stats, snap_counts,
  build_draft_lookup(draft_picks), position)` internally. RB is already one of
  the looped positions.
- `scripts/build_vegas_team_context_override.py` already emits the Vegas family
  (`preseason_implied_team_total`, `preseason_spread`,
  `season_avg_implied_team_total`, `season_avg_spread`) via
  `attach_vegas_team_context_features` over raw `schedules` (position-agnostic;
  merged on `(season, week, team)`).

Confirm each script's output covers RB rows; if a script needs an RB-inclusive
invocation flag, add it minimally rather than forking the builder. The override
parquets land in `data/features_probe/` (gitignored convention; regenerable).
No source module (`RbFeaturesSchema`, `build_rb_features`, any model) is touched
in this phase.

**Run the probe** for each family — **explicitly `--position RB`** (the probe
defaults to all four positions and would crash on QB/WR where these columns
already exist in the cache), with the **coverage threshold relaxed** to match
the structural sparsity of each family (the default is 0.95 and both families
fall below it):

```
python -m scripts.probe_feature_signal --candidate-name augment_rb_trajectory \
    --position RB --override data/features_probe/rb_trajectory.parquet \
    --coverage-threshold 0.35 \
    --csv-out reports/feature_probe_rb_trajectory.csv
python -m scripts.probe_feature_signal --candidate-name augment_rb_vegas \
    --position RB --override data/features_probe/rb_vegas_preseason.parquet \
    --coverage-threshold 0.90 \
    --csv-out reports/feature_probe_rb_vegas.csv
```

Coverage thresholds follow the established precedents: Vegas `season_avg_*` is
NaN at week 1 by construction → `0.90` (per the WR+QB Vegas probe spec,
`2026-05-17-vegas-team-context-probe-design.md`); trajectory's trend columns need
~8 prior active games and cover only ~45–71% of player-weeks → a relaxed floor
(~`0.35`, matching the PR #25 WR trajectory probe). **Confirm the exact RB
non-null fractions from the override and set the threshold just below the
binding column** rather than hardcoding these figures blindly; the harness
reports per-column coverage and aborts with `OverrideCoverageError` if a column
falls under the threshold.

**Gate G1** (see §6). Both NULL → STOP. Any SIGNAL → Phase 2 for the
signaling family (or families).

### Phase 2 — Integrate the signaling family + re-test model classes

For each family that SIGNALed in Phase 1:

1. **Schema** — add the family's columns to `RbFeaturesSchema`
   (`src/projections/schemas.py:687`), matching the WR precedent dtypes exactly
   (verified at `schemas.py:604-607`): all four are
   `Series[float] = pa.Field(nullable=True)` — `age` (`ge=15, le=50`),
   `is_rookie` (`ge=0, le=1`), `volume_trend_l4_minus_prior_l4` (unbounded),
   `snap_pct_change_l4_vs_prior_l4` (`ge=-1, le=1`); Vegas columns float nullable.
   (`is_rookie`/`age` are float in the *feature* schema, not bool/int — do not
   invent new dtypes.)
2. **Builder** — wire the attach helper into `build_rb_features`
   (`src/projections/features/rb.py`) following the `build_wr_features` template
   (`features/wr.py:257-268`): build the player-team-week `index`, call
   `attach_trajectory_features(index, weekly_stats, snap_counts,
   build_draft_lookup(draft_picks), Position.RB)` on the **full, un-`prior_mask`-
   filtered** `weekly_stats`/`snap_counts` (the helper's trailing-8/season-age
   windows need full history, not the per-week slice — WR comments this
   explicitly), and/or `attach_vegas_team_context_features` over `schedules`.
   `draft_picks` is already an accepted (currently-ignored) parameter
   (`rb.py:68`); `refresh_features.py` already loads and passes it. This is
   materially more than "pass draft_picks" — the index construction and
   full-frame history wiring are the real work; mirror WR. `df =
   SCHEMA.validate(df)` with reassignment at the module boundary.
3. **Model column list** — add the columns to `_RB_FEATURE_COLUMNS`
   (`src/projections/models/baseline.py:386`). lightgbm derives its feature list
   from `RbFeaturesSchema` dynamically and picks them up automatically; the
   hardcoded baseline list must be updated explicitly (the documented
   pattern in the WR/QB column lists).
4. **Rebuild the RB feature cache** — `python scripts/refresh_features.py rb`
   for all cached seasons. (Keep the *pre-feature* RB cache/run from Phase 0 so
   the dual-run gate in step 5 has a clean before/after pair.)
5. **Re-evaluate on the augmented features.** Two distinct questions, two
   distinct instruments:
   - **(a) Do the features help? — the ship gate.** `scripts/adoption_gate.py`
     is a *paired* gate over two backtest **run directories** (each a
     `results.parquet`), **not** a snapshot comparison. So: produce two RB
     `baseline` backtest runs via `scripts/backtest.py --report` (which writes
     `data/backtest/run_<ts>/results.parquet`) — one on the **old** features
     (Phase 0 / pre-rebuild cache), one on the **augmented** features — then run
     the **dual-run** gate (`--baseline-run <old> --candidate-run <new>
     --candidate baseline`). ADOPT here = the feature family helps the baseline
     model (criteria in §6 G2).
   - **(b) Which model class is best on the augmented features? — selection.**
     Re-run the RB-only walk-forward **bake-off** {`baseline`, `lightgbm-nb`,
     `ensemble`} on the augmented cache (the §1 / Phase 0 driver) and compare
     **absolute** `composite_rmse`/`composite_mae`/`spearman_topN`. New signal is
     precisely what could let lightgbm/ensemble finally beat ridge; the §1
     verdict was on the old feature set and does not bind. (The adoption gate
     emits a *paired delta*, not an absolute per-class ranking — that is why
     class selection uses the bake-off's absolute table, while the ship decision
     uses the gate.)

**Gate G2** (see §6) consumes both: the gate verdict (a) decides whether features
ship at all; the bake-off (b) decides which class becomes the default.

### Phase 3 — Flip default + re-validate on DFS

If a model class wins G2:

1. Set `POSITION_DISPATCH[Position.RB].default_model_class` to the winner
   (`src/projections/models/__init__.py:194`).
2. Update the committed backtest snapshot (`tests/backtest/model_metrics.json`)
   via `python scripts/backtest.py --update-snapshot` for the affected rows, and
   verify `--check` passes.
3. Re-run the **DFS edge study + per-position significance**, reusing
   `scripts/_dfs_per_position_analysis.py` and its persisted universe cache. The
   real cache path is `data/dfs_universe_2021-2024_<sig>.parquet`, where `<sig>`
   is derived from the usage floor + per-position model routing
   (`_dfs_per_position_analysis.py:39,46`). Because flipping RB's
   `default_model_class` changes that signature, the analysis writes a **new**
   cache file and rebuilds the RB cells automatically — no manual invalidation.
   Report RB's new fraction + CI and whether the pooled verdict moved.
4. Update TODO #55 + `project_management.md` with the verdict (lift shipped /
   improved-but-still-STOP / NULL dead-end), including the before/after RB
   fraction.

## 5. Components touched

| Phase | Files | Nature |
|---|---|---|
| 0 | `reports/rb_model_bakeoff_2026-06-25.md` (new); driver `scripts/_rb_model_bakeoff.py` (already authored) | No source changes |
| 1 | **reuse** `scripts/build_trajectory_override.py` + `scripts/build_vegas_team_context_override.py` (minimal RB-inclusive tweak only if needed), `data/features_probe/rb_*.parquet` (new, gitignored), `reports/feature_probe_rb_*.csv` (new) | No source changes |
| 2 | `src/projections/schemas.py`, `src/projections/features/rb.py`, `src/projections/models/baseline.py`; `data/features/rb/*` cache rebuilt; `data/backtest/run_*/` (old + augmented) | Schema + builder + model column list |
| 3 | `src/projections/models/__init__.py` (dispatch default), `tests/backtest/model_metrics.json` (snapshot); TODO/PM | Default flip + snapshot + docs |

Mirrors WR trajectory (PR #25), WR+QB Vegas (PR #51), RB weather (PR #28).

## 6. Decision gates (precise)

These gates use the harness's *actual* verdict mechanics (verified against
`feature_probe.py` and `adoption_gate.py`), not invented labels.

**Gate G1 (after Phase 1).** Use `family_verdict_from_reports`
(`feature_probe.py:112`), which returns SIGNAL/NULL per family. Under the hood a
**per-stat** verdict is SIGNAL iff the paired-bootstrap 95% CI on Δ-RMSE is
strictly below 0 **and** `|point| ≥ effect_size_floor` (default **0.05** fpts —
this absolute-magnitude gate is what collapses CI-significant-but-tiny effects
straight to NULL; there is no separate "MARGINAL" label at the per-stat level,
and the floor *is* the magnitude discipline of
`[[feedback_probe_threshold_retrospective]]`). Pooled-across-years is the primary
read — `phase1_should_fire_phase2` (`feature_probe.py:102`) fires only on a
**pooled** SIGNAL, so a lone single-year SIGNAL that is NULL pooled does not
qualify. (Note: `family_verdict_from_reports` also treats a Phase-2 composite
`MARGINAL` as SIGNAL; for G1 we read the **Phase-1** family verdict, the cheap
screen, so this does not apply here.)
- **Both families NULL → STOP.** Write up, update TODO/PM, end.
- **Any family SIGNAL → Phase 2** for that family only.

**Gate G2 (after Phase 2).** Two independent decisions, per the two instruments
in Phase 2 step 5:

*Ship decision (do the features ship at all?).* The **dual-run adoption gate**
(`baseline` on old features vs `baseline` on augmented features) returns ADOPT.
The gate's actual ADOPT criterion (`verdict_for_position`, `adoption_gate.py`)
is exactly two conditions: `PASS_RMSE` (composite ΔRMSE `hi_95 < 0`) **and**
`PASS_SPEARMAN` (`spearman.lo_95 > -0.02`). **MAE and calibration are *not* gate
inputs** — they are reported for context only, not pass/fail conditions. ("Did
the augmented baseline strictly improve?" therefore has a precise meaning: the
dual-run gate verdict is ADOPT. Nothing softer.)

*Default decision (which class becomes the default?).* From the Phase 2(b)
bake-off on augmented features, take the model class with the lowest **absolute**
`composite_rmse`, provided it does not regress `spearman_topN` vs the augmented
baseline. (Absolute, because the bake-off is the only instrument that ranks all
three classes; the gate only does paired deltas.)

Combined outcomes:
- **Gate ADOPT + a non-baseline class wins the bake-off → Phase 3**, flipping the
  default to that class.
- **Gate ADOPT + baseline still wins the bake-off → ship the features, keep
  `baseline`** as the default, update the snapshot, and still run the Phase 3
  edge-study re-check (the improved baseline may move RB's fraction even with no
  class flip).
- **Gate does NOT ADOPT → STOP**, revert/abandon the integration (do not ship
  dead feature columns), document. (No "point estimate looked good" escape — the
  gate verdict is the sole ship criterion.)

## 7. Testing

- **Phase 1** is read-only (override build + probe). No new unit tests required
  beyond the probe harness's existing coverage; the override builder gets a smoke
  assertion that output columns are present and identity-keyed.
- **Phase 2** follows the WR trajectory/Vegas integration test templates:
  - `RbFeaturesSchema` accepts the new columns with the declared dtypes and
    rejects wrong dtypes (extend the existing schema tests).
  - A `build_rb_features` test asserting each new column is **populated**
    (not all-NaN) on a real fixture and that non-null coverage clears a stated
    floor. Both families are *structurally* below the probe's default 0.95
    coverage (which is why Phase 1 relaxes the threshold): Vegas `season_avg_*`
    is NaN at week 1 by construction (≈0.90+ overall), and trajectory's trend
    columns need ~8 prior active games (≈45–71% of player-weeks). The test's
    floor should match the relaxed Phase-1 thresholds, not 0.95.
  - The dtype-regression seam per CLAUDE.md: `pytest -v -k "ingest or store or
    schemas"` must stay green (RB cache rebuild is a store/schema path).
- **End-of-effort checklist** (CLAUDE.md §4) on every phase that touches source:
  `pytest -v` (or stated subset), `mypy src tests`, `ruff check`, `ruff format
  --check`.

## 8. Risks & mitigations

- **Both NULL is genuinely likely** given the signal-limited signature. Mitigation:
  Phase 1 is cheap and designed as the early exit; a NULL result is a *successful*
  outcome of the spec (a cheap, honest dead-end), not a failure.
- **Trajectory features barely vary week-to-week** (`age`/`is_rookie` are
  season-constant). They may help season-level calibration but add little to the
  weekly DFS-disagreement metric. The probe measures weekly Δ-RMSE directly, so
  this shows up as NULL if real.
- **Coverage** is structurally below the probe's 0.95 default for both families
  (Vegas week-1 `season_avg_*` NaN; trajectory's ~8-game trend window; trajectory
  also depends on the `draft_picks` join, missing for some UDFAs/older players).
  Mitigation: Phase 1 relaxes `--coverage-threshold` per family to the documented
  precedents (§4) and confirms the exact RB non-null fractions before setting the
  floor; Phase 2 asserts the matching floor and the schema marks the columns
  nullable (per the WR precedent). A probe that aborts on `OverrideCoverageError`
  is a misconfigured threshold, not a NULL result — don't conflate them.
- **Model backtest is slow** (~30 min single-process for the RB-only 3-class
  bake-off). Mitigation: scope re-runs to RB-only; accept the cost; the emit
  vectorization stays out of scope.
- **Necessary ≠ sufficient.** Even a clean lift only beats *Sleeper-alone* and
  may not flip the pooled verdict (QB/TE still drag). Phase 3 reports the truth
  either way; "improved but still STOP" is an explicitly allowed terminal state.
- **Don't ship dead columns.** If G2 fully fails, the integration is reverted
  (§6) so the cache/schema don't carry inert features.

## 9. Open questions

- **Exact RB coverage fractions:** the relaxed `--coverage-threshold` values in §4
  (0.90 Vegas, ~0.35 trajectory) follow the WR/QB precedents; confirm the actual
  RB per-column non-null fractions from the built override parquets and set each
  threshold just below the binding column (Phase 1, first task).
- **Probe eval-year span:** 2021–2024 to match the edge study and the §1 bake-off;
  the probe's `train_start` default (2018) is retained.
