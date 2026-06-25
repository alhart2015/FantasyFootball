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

When a flexible gradient-boosted model (lightgbm) cannot beat a linear one
(ridge) on the **same** features, that is the textbook **signal-limited**
signature: there is no additional nonlinear/interaction signal in the current RB
feature set for a richer model to exploit. RB is signal-limited, not
model-limited. **Therefore the only lever with a real chance of moving RB is new
feature signal — not another model swap.**

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

1. **Real metric (confirmation):** RB's DFS edge-study fraction moves up — ideally
   its 95% CI reaches or exceeds **0.50** — and ideally nudges the pooled verdict
   off STOP. (Per #55, a single position may not flip the pool; we measure, not
   assume.)
2. **Cheap gate (necessary):** some model class (`baseline` / `lightgbm-nb` /
   `ensemble`) on the **augmented** RB features beats the **current** `baseline`
   on the walk-forward model backtest (composite RMSE + MAE, with `spearman_topN`
   not regressing), confirmed through the existing **adoption gate**.
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
position). Three phases with hard decision gates between them.

### Phase 1 — Signal probe (cheap, no source changes)

Use the existing pre-spec screening tool `scripts/probe_feature_signal.py`
(→ `src/projections/backtest/feature_probe.py`), which merges a candidate-column
**override parquet** onto the baseline feature set and emits per-stat
Δ-CV-RMSE bootstrap CIs with SIGNAL / NULL / REGRESSION verdicts (augment mode).

**Build the two override parquets** in `data/features_probe/` (gitignored
convention; regenerable):

- `rb_trajectory.parquet` — columns `age`, `is_rookie`,
  `volume_trend_l4_minus_prior_l4`, `snap_pct_change_l4_vs_prior_l4`, keyed on
  `gsis_id` / `season` / `week`, produced by `attach_trajectory_features`
  (`src/projections/features/trajectory_features.py`) over the RB feature base +
  raw `draft_picks`.
- `rb_vegas_preseason.parquet` — columns `preseason_implied_team_total`,
  `preseason_spread`, `season_avg_implied_team_total`, `season_avg_spread`,
  keyed identically, produced by `attach_vegas_team_context_features`
  (`src/projections/features/vegas_team_context_features.py`) over raw
  `schedules`.

A small builder script (`scripts/build_rb_probe_overrides.py`) constructs these
from the same raw inputs `build_rb_features` already consumes (`draft_picks`
present back to 2000; `schedules` present). It does **not** touch
`RbFeaturesSchema`, `build_rb_features`, or any model.

**Run the probe** for each family (augment mode), over eval years 2021–2024 to
match the edge study and the model bake-off:

```
python -m scripts.probe_feature_signal --candidate-name augment_rb_trajectory \
    --override data/features_probe/rb_trajectory.parquet \
    --csv-out reports/feature_probe_rb_trajectory.csv
python -m scripts.probe_feature_signal --candidate-name augment_rb_vegas \
    --override data/features_probe/rb_vegas_preseason.parquet \
    --csv-out reports/feature_probe_rb_vegas.csv
```

The probe restricts itself to RB automatically by the override's identity keys /
position; coverage per candidate is reported by the harness.

**Gate G1** (see §6). Both NULL → STOP. Any SIGNAL → Phase 2 for the
signaling family (or families).

### Phase 2 — Integrate the signaling family + re-test model classes

For each family that SIGNALed in Phase 1:

1. **Schema** — add the family's columns to `RbFeaturesSchema`
   (`src/projections/schemas.py:687`) with the correct nullable dtypes per
   CLAUDE.md: `age` → `pd.Int64Dtype()` (nullable int) or float per the WR
   precedent; `is_rookie` → bool; the trend/Vegas floats → float64;
   `is_rookie` follows the WR precedent's dtype exactly. Match WR's schema
   declarations for these identical columns rather than inventing new dtypes.
2. **Builder** — wire the attach helper into `build_rb_features`
   (`src/projections/features/rb.py`), passing `draft_picks` (already an
   accepted, currently-reserved parameter) and/or `schedules` (already consumed).
   `df = SCHEMA.validate(df)` with reassignment at the module boundary.
3. **Model column list** — add the columns to `_RB_FEATURE_COLUMNS`
   (`src/projections/models/baseline.py:386`). lightgbm derives its feature list
   from `RbFeaturesSchema` dynamically and picks them up automatically; the
   hardcoded baseline list must be updated explicitly (the documented
   pattern in the WR/QB column lists).
4. **Rebuild the RB feature cache** — `python scripts/refresh_features.py rb`
   for all cached seasons.
5. **Re-evaluate on the augmented features:**
   - Run `scripts/adoption_gate.py` for RB: `baseline`-with-features vs the
     current committed `baseline` snapshot (the ship gate — composite ΔRMSE
     with coverage guards).
   - Re-run the **model-class bake-off** {`baseline`, `lightgbm-nb`,
     `ensemble`} on the augmented features (the same RB-only walk-forward used
     in §1), because new signal is precisely what could let lightgbm/ensemble
     finally beat ridge. The §1 verdict was on the **old** feature set and does
     not bind here.

**Gate G2** (see §6). Pick the winning model class if any beats current baseline;
else STOP-with-features-but-no-model-change (document; possibly still re-run the
edge study if the augmented baseline itself improved — see §6).

### Phase 3 — Flip default + re-validate on DFS

If a model class wins G2:

1. Set `POSITION_DISPATCH[Position.RB].default_model_class` to the winner
   (`src/projections/models/__init__.py:194`).
2. Update the committed backtest snapshot (`tests/backtest/model_metrics.json`)
   via `python scripts/backtest.py --update-snapshot` for the affected rows, and
   verify `--check` passes.
3. Re-run the **DFS edge study + per-position significance**, reusing
   `scripts/_dfs_per_position_analysis.py` and the persisted universe cache
   (`data/dfs_universe_2021-2024.parquet`; the cache is keyed on usage floor +
   model routing, so changing RB's routing correctly invalidates/rebuilds the RB
   cells). Report RB's new fraction + CI and whether the pooled verdict moved.
4. Update TODO #55 + `project_management.md` with the verdict (lift shipped /
   improved-but-still-STOP / NULL dead-end), including the before/after RB
   fraction.

## 5. Components touched

| Phase | Files | Nature |
|---|---|---|
| 1 | `scripts/build_rb_probe_overrides.py` (new), `data/features_probe/rb_*.parquet` (new, gitignored), `reports/feature_probe_rb_*.csv/.md` (new) | No source changes |
| 2 | `src/projections/schemas.py`, `src/projections/features/rb.py`, `src/projections/models/baseline.py`; `data/features/rb/*` cache rebuilt | Schema + builder + model column list |
| 3 | `src/projections/models/__init__.py` (dispatch default), `tests/backtest/model_metrics.json` (snapshot); TODO/PM | Default flip + snapshot + docs |

Mirrors WR trajectory (PR #25), WR+QB Vegas (PR #51), RB weather (PR #28).

## 6. Decision gates (precise)

**Gate G1 (after Phase 1).** A family **SIGNALs** if it has at least one per-stat
verdict of SIGNAL under the probe's standard criterion (paired-bootstrap 95% CI
on Δ-RMSE excludes 0 in the improving direction) that is **not** flagged
MARGINAL by the harness's magnitude check (consistent with the
binding-cell/magnitude discipline in `[[feedback_probe_threshold_retrospective]]`
— a CI-significant but <0.005-fpts-equivalent improvement is treated as
MARGINAL, not actionable). Pooled-across-years is the primary read; a lone
single-year SIGNAL that is NULL pooled does not qualify.
- **Both families NULL/MARGINAL → STOP.** Write up, update TODO/PM, end.
- **Any family SIGNALs (non-marginal) → Phase 2** for that family only.

**Gate G2 (after Phase 2).** "Beat baseline" = the adoption gate's composite
verdict is **ADOPT** for at least one model class — i.e. composite `RMSE`
improves with the lower CI bound favorable (the gate's own criterion), `MAE` does
not regress, `spearman_topN` does not regress, and coverage
(`calibration_p10p90`) stays within tolerance. Among ADOPT model classes, pick
the lowest composite_rmse.
- **Some model class ADOPTs → Phase 3** with that class as the new default.
- **No model class ADOPTs, but the augmented `baseline` strictly improved over
  the old baseline** (RMSE down, nothing regressed) → still **flip nothing**
  (RB stays `baseline`), but the *features* ship and we **re-run the edge study**
  to see if even the improved baseline moves RB's fraction. (The default is
  already `baseline`; "shipping" here means the new feature columns are now in
  the cache + schema and the committed snapshot is updated.)
- **No improvement anywhere → STOP**, revert/abandon the integration (do not ship
  dead feature columns), document.

## 7. Testing

- **Phase 1** is read-only (override build + probe). No new unit tests required
  beyond the probe harness's existing coverage; the override builder gets a smoke
  assertion that output columns are present and identity-keyed.
- **Phase 2** follows the WR trajectory/Vegas integration test templates:
  - `RbFeaturesSchema` accepts the new columns with the declared dtypes and
    rejects wrong dtypes (extend the existing schema tests).
  - A `build_rb_features` test asserting each new column is **populated**
    (not all-NaN) on a real fixture and that non-null coverage clears a stated
    threshold (Vegas-preseason near-complete; trajectory lower due to the
    draft_picks join for UDFAs/veterans — state the expected floor).
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
- **Coverage on trajectory** depends on the `draft_picks` join (missing for some
  UDFAs/older players). Mitigation: Phase 1 reports coverage; Phase 2 asserts a
  floor and the schema marks the columns nullable (per the WR precedent).
- **Model backtest is slow** (~30 min single-process for the RB-only 3-class
  bake-off). Mitigation: scope re-runs to RB-only; accept the cost; the emit
  vectorization stays out of scope.
- **Necessary ≠ sufficient.** Even a clean lift only beats *Sleeper-alone* and
  may not flip the pooled verdict (QB/TE still drag). Phase 3 reports the truth
  either way; "improved but still STOP" is an explicitly allowed terminal state.
- **Don't ship dead columns.** If G2 fully fails, the integration is reverted
  (§6) so the cache/schema don't carry inert features.

## 9. Open questions

- **Dtypes for `age`/`is_rookie`:** resolve by matching WR's `WrFeaturesSchema`
  declarations exactly (don't invent). Confirmed during Phase 2 task 1.
- **Probe eval-year span:** 2021–2024 to match the edge study and the §1 bake-off;
  the probe's `train_start` default (2018) is retained.
