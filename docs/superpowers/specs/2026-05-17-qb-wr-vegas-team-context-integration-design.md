# QB + WR Vegas team-context integration — design

**Status:** draft, awaiting review.
**Branch:** `feat/qb-wr-vegas-team-context-integration`.
**Predecessor:** PR #49 (TODO #33c probe) — `docs/superpowers/specs/2026-05-17-vegas-team-context-probe-design.md`.
**Predecessor verdict:** SIGNAL at `lgb-nb × swap` composite — QB ADOPT (ΔRMSE −0.0587 fpts, CI [−0.092, −0.028]) + WR ADOPT (ΔRMSE −0.0130 fpts, CI [−0.022, −0.003]). RB just missed ADOPT; TE NULL.

## 1. Goal

Promote the four Vegas team-context features (`preseason_implied_team_total`, `preseason_spread`, `season_avg_implied_team_total`, `season_avg_spread`) from probe-only-override status to first-class members of `QbFeaturesSchema` and `WrFeaturesSchema`. Wire them into `LightGBMNbModel`'s feature list for QB and WR only as a **schema-swap** (drop the existing per-game `implied_team_total` + `spread`, add the four new cols). All other model classes are untouched. Validate via three dual-run adoption gates.

## 1.1 Scope

In scope:

- Schema additions on `QbFeaturesSchema` and `WrFeaturesSchema`.
- Feature-builder wire-up for `build_qb_features` and `build_wr_features`.
- lgb-nb feature-list overrides for QB and WR factories.
- Code-hash hygiene: `features/vegas_team_context_features.py` added to the lgb-nb hash file set.
- Three adoption-gate runs (Phase 2): `(lgb-nb, QB)`, `(lgb-nb, WR)`, `(ensemble-decomposed, WR)`.
- Integration-verdict report under `reports/qb_wr_vegas_team_context_integration_summary.md`.
- TODO #33c entry updated with integration outcome.

Explicitly out of scope:

- RB integration. RB just-missed-ADOPT under `season_avg_*`-bearing probe; deferred to a separate `preseason_*`-only follow-up probe per TODO #33c entry.
- TE integration. TE returned NULL across all probe cells; closed.
- External preseason Vegas signals (May win totals, OC/HC tenure, FA-acquisition flag). Listed in the predecessor's "Next direction" caveat — not in this spec.
- `BaselineModel` / `DecomposedBaselineModel` feature-list changes. The Ridge children of `wr_ensemble_decomposed` must remain on their pre-#49 feature list to preserve the calibration-aware-weighting trade-off — Ridge AUGMENT regressed on QB in the probe.
- `LightGBMModel` (untuned) and `LightGBMTunedModel` feature-list changes. These classes derive their feature list from the schema and therefore receive the AUGMENT treatment automatically. Probe verdict for lgb augment composite was NULL on QB and WR — no shipped regression on these classes — and both are pruning candidates per TODO #29. Validated incidentally by the dual-run gate's pre/post baseline runs but not gated.

## 1.2 Success criteria

| # | Gate | Predicted ΔRMSE (composite, fpts) | ADOPT criterion |
|---|---|---|---|
| 1 | `(lgb-nb, QB)` — probe-replication | ≈ −0.0587 (CI [−0.092, −0.028]) | CI strictly below 0 |
| 2 | `(lgb-nb, WR)` — probe-replication | ≈ −0.0130 (CI [−0.022, −0.003]) | CI strictly below 0 |
| 3 | `(ensemble-decomposed, WR)` — production gate | direction uncertain (pinball weights re-fit) | ADOPT if CI < 0; MARGINAL accepted if CI brackets 0 with point estimate ≤ 0; STOP if CI > 0 |

In addition:

- Ridge child of `wr_ensemble_decomposed` has byte-identical predictions pre vs post (the Ridge feature list does not change; the Ridge children of the ensemble are weight-fit on a feature frame that has four extra columns but does not select them).
- All pre-existing tests pass; mypy strict + ruff clean.
- New tests added per § 4 below.

## 1.3 Non-goals

- Closing the Chase-250-vs-403 elite-magnitude gap. The probe's −0.06 fpts ΔRMSE at QB is ≈ 1–2% per-week composite — necessary but not sufficient for elite-magnitude. Documented as a caveat; this spec does not pretend to close it.

## 2. Architecture

### 2.1 Schema additions (`src/projections/schemas.py`)

Add four nullable Float64 columns to **both** `QbFeaturesSchema` and `WrFeaturesSchema`:

```python
preseason_implied_team_total: pa.Field(
    dtype="Float64", nullable=True, ge=0,
    description="Vegas-implied team total from this team's week-1 game, broadcast across all weeks.",
)
preseason_spread: pa.Field(
    dtype="Float64", nullable=True,
    description="Team-perspective spread (favorite negative) from this team's week-1 game, broadcast across all weeks.",
)
season_avg_implied_team_total: pa.Field(
    dtype="Float64", nullable=True, ge=0,
    description="Expanding mean of implied_team_total over weeks 1..N-1 (leakage-safe; NaN at week 1).",
)
season_avg_spread: pa.Field(
    dtype="Float64", nullable=True,
    description="Expanding mean of team-perspective spread over weeks 1..N-1 (NaN at week 1).",
)
```

Nullability rationale:

- `season_avg_*` is NaN at week 1 of every season by construction (expanding mean of an empty prior window). This is by design, not a data-quality issue.
- `preseason_*` is NaN only on the rare team-with-no-week-1-schedule-row edge case (e.g., historical preseason-week renumbering). Carries the same NaN handling as other team-level features (downstream median imputation in `BaselineModel.fit` — but lgb-nb ingests NaN directly through LightGBM's native missing-handling, so no imputation is required for lgb-nb's path).

No bound on `*_spread` (can be negative — favorite); `ge=0` on `*_implied_team_total` matches the existing `implied_team_total` bound.

### 2.2 Feature-builder wire-up (`src/projections/features/qb.py` + `wr.py`)

At the bottom of each builder, after the existing schema-validation step, call:

```python
from projections.features.vegas_team_context_features import attach_vegas_team_context_features

# ... existing builder logic produces `out` with schema-validated cols including
# implied_team_total + spread ...

out = attach_vegas_team_context_features(out, schedules)
return QbFeaturesSchema.validate(out)  # re-validate with four new cols populated
```

The existing per-game `implied_team_total` + `spread` columns **remain** in the builder output — they are dropped only from lgb-nb's feature list (§ 2.3), not from the schema. `BaselineModel.fit` continues to select them via its hardcoded `_QB_FEATURE_COLUMNS` / `_WR_FEATURE_COLUMNS` lists. This is the load-bearing decoupling that lets the schema-swap apply to lgb-nb only.

`attach_vegas_team_context_features` is a thin wrapper around `compute_vegas_team_context_features` — already public in `vegas_team_context_features.py` since PR #49. No refactor required.

TE + RB builders are not touched.

### 2.3 lgb-nb feature-list overrides (`src/projections/models/lightgbm_nb.py`)

Add at module scope, alongside the existing imports from `lightgbm.py`:

```python
_VEGAS_SWAP_REPLACE: Final[frozenset[str]] = frozenset({"implied_team_total", "spread"})
_VEGAS_SWAP_ADD: Final[tuple[str, ...]] = (
    "preseason_implied_team_total",
    "preseason_spread",
    "season_avg_implied_team_total",
    "season_avg_spread",
)

def _swap_for(cols: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(c for c in cols if c not in _VEGAS_SWAP_REPLACE) + _VEGAS_SWAP_ADD

_QB_FEATURE_COLUMNS_NB: Final[tuple[str, ...]] = _swap_for(_filter_features(_QB_FEATURE_COLUMNS))
_WR_FEATURE_COLUMNS_NB: Final[tuple[str, ...]] = _swap_for(_filter_features(_WR_FEATURE_COLUMNS))
```

Update the two factories:

```python
def qb_lightgbm_nb() -> LightGBMNbModel:
    return LightGBMNbModel(
        config=_LightGBMConfig(
            position=Position.QB,
            target_stats=_QB_TARGET_STATS,
            feature_columns=_QB_FEATURE_COLUMNS_NB,  # swap-treatment
            feature_schema=QbFeaturesSchema,
            non_negative_stats=_QB_NON_NEGATIVE,
        )
    )

def wr_lightgbm_nb() -> LightGBMNbModel:
    return LightGBMNbModel(
        config=_LightGBMConfig(
            position=Position.WR,
            target_stats=_WR_TARGET_STATS,
            feature_columns=_WR_FEATURE_COLUMNS_NB,  # swap-treatment
            feature_schema=WrFeaturesSchema,
            non_negative_stats=_WR_NON_NEGATIVE,
        )
    )
```

`te_lightgbm_nb()` and `rb_lightgbm_nb()` are unchanged: they still pass `_filter_features(_TE_FEATURE_COLUMNS)` / `_filter_features(_RB_FEATURE_COLUMNS)`, and `TeFeaturesSchema` / `RbFeaturesSchema` do not gain any new cols.

### 2.4 Code-hash hygiene (`src/projections/models/lightgbm_nb.py`)

Extend `_code_hash_files_nb` to include `features/vegas_team_context_features.py`. The current set already covers `features/qb.py`, `features/wr.py`, and `_shared.py`; the new module is a transitive dep of QB + WR builders post-integration, and a bug fix landed only in `vegas_team_context_features.py` would otherwise fail to invalidate cached `model_id`s.

Add unconditionally for all four positions — TE and RB lgb-nb model_ids will rebuild once when this hash list changes, then stabilize. Conditional logic (e.g., only-include-when-QB-or-WR) would complicate the call site for no real gain.

### 2.5 Ensemble wiring (no change)

`wr_ensemble_decomposed()` calls `wr_lightgbm_nb` as its child B factory. After § 2.3, calling `wr_lightgbm_nb()` returns a model with the new swap feature list — so the ensemble's child B automatically picks up the integration. No edits to `models/ensemble.py` or `models/decomposed_baseline.py`.

The Ridge decomposed child (child A) uses `_WR_FEATURE_COLUMNS` from `baseline.py` — hardcoded, does **not** auto-pick up schema changes. It remains on the pre-integration feature list. The pinball-weight-fit Stage 3 of `EnsembleModel.fit` will re-fit weights against a child B that now consumes Vegas cols and drops per-game cols; weights may shift in either direction. Gate #3 in § 1.2 is the load-bearing check on whether this re-fit produces a net composite improvement at the production route.

## 3. Mechanism interpretation (carried from probe spec § 6)

Trees overfit per-game `implied_team_total` / `spread` line noise; smoother preseason and season-to-date variants generalize better. Ridge cannot extract this — Phase 1 AUGMENT actively *regressed* on QB pooled `passing_yards` in both augment and swap modes (Ridge augment +0.45 fpts; Ridge swap NULL). The SIGNAL only emerges when the four new cols **replace** (not augment) the per-game cols **under tree models**.

This is the same mechanism reading as the probe verdict: keeping the Ridge child on the per-game cols preserves Ridge's signal in the ensemble, and putting only lgb-nb on the four new cols isolates the tree-overfit-on-line-noise correction.

## 4. Tests

### 4.1 Schema tests (`tests/test_schemas/`)

Mirror the existing pattern for other nullable Float64 cols (e.g., `implied_team_total`):

- `QbFeaturesSchema.validate` accepts a valid frame carrying all four new cols.
- Negative `preseason_implied_team_total` is rejected (`ge=0`).
- Negative `season_avg_implied_team_total` is rejected (`ge=0`).
- NaN is accepted on all four cols (nullability).
- Wrong dtype (e.g., float64 instead of Float64) is rejected.

Same coverage for `WrFeaturesSchema`.

### 4.2 Builder tests (`tests/test_features/test_qb_features.py` + `test_wr_features.py`)

- Synthetic-fixture integration test: builder output has the four new cols, dtype `Float64`, values match what `compute_vegas_team_context_features` would produce on the same schedules fixture.
- Week-1 NaN: `season_avg_implied_team_total` and `season_avg_spread` are NaN at the team's first week of every season.
- Bye-week handling: the existing builder bye-week filter (introduced in TODO #16's drift list) drops players whose team has no schedule row in `as_of_week` *before* the Vegas-cols attach. Verify: (a) every surviving row has non-NaN `preseason_*` cols (team played week 1, so broadcast value exists); (b) `season_avg_*` cols are NaN at week 1 only. The rare "team had no week-1 game" edge case (preseason renumbering / expansion mid-season) would surface as NaN `preseason_*` on a non-week-1 row — assert that this does not occur in any 2018–2024 fixture row; if it ever does, document the failure under TODO #16 rather than masking with imputation.
- Schema-validation round-trip: builder output validates against the new schema.

### 4.3 Model feature-list tests (`tests/test_models/`)

- `qb_lightgbm_nb().config.feature_columns` excludes `"implied_team_total"` and `"spread"`, includes all four new cols.
- `wr_lightgbm_nb().config.feature_columns` — same.
- `qb_lightgbm().config.feature_columns` and `qb_lightgbm_tuned().config.feature_columns` **include** `"implied_team_total"` and `"spread"` (augment-by-default for non-production classes — explicit assertion documents the asymmetry).
- `te_lightgbm_nb().config.feature_columns` is byte-identical to its pre-integration value (no `preseason_*` / `season_avg_*` cols leak into TE).
- `rb_lightgbm_nb().config.feature_columns` — same.

### 4.4 Integration smoke (`tests/test_models/`)

Existing per-position lgb-nb fit/predict tests already exercise `_LightGBMConfig.feature_columns` end-to-end. Extend with a tiny synthetic fixture carrying the four new cols at non-NaN values and assert `predict_distribution` returns a `MIXED` family frame with the expected shape.

### 4.5 Code-hash invalidation test (`tests/test_models/`)

Extend the existing `_code_hash_files_nb`-related test (the one that asserts editing a hashed file changes `LightGBMNbModel.code_hash`) to cover `features/vegas_team_context_features.py` as a hashed dep.

## 5. Adoption gates (Phase 2)

Run `scripts/adoption_gate.py` in `--baseline-run / --candidate-run` dual-run mode (load-bearing pattern from TODO #3a) for three (model, position) pairs.

### 5.1 Gate runs

Pre-integration baseline (`--baseline-run`): on `main` HEAD (post-#49 merge — note that PR #49 was a probe-only change and did not touch schemas or builders, so the pre-integration feature builders + schemas live unchanged at this commit), regenerate feature parquets, train each model class, save backtest snapshots.

Post-integration candidate (`--candidate-run`): refresh feature parquets on this branch (four new cols populated), train each model class, save backtest snapshots.

Three gates:

1. `--model lightgbm-nb --position QB`
2. `--model lightgbm-nb --position WR`
3. `--model ensemble-decomposed --position WR`

### 5.2 ADOPT / MARGINAL / REGRESSION criteria

Per probe-spec convention (composite ΔRMSE, bootstrap 95% CI):

- **ADOPT**: composite ΔRMSE CI strictly below 0.
- **MARGINAL**: composite ΔRMSE CI brackets 0 with point estimate ≤ 0.
- **REGRESSION**: composite ΔRMSE CI strictly above 0.

Gate-specific ship/stop policy:

| Gate | ADOPT | MARGINAL | REGRESSION |
|---|---|---|---|
| 1 (lgb-nb QB) | ship | stop, debug feature-builder bug | stop, debug feature-builder bug |
| 2 (lgb-nb WR) | ship | stop, debug feature-builder bug | stop, debug feature-builder bug |
| 3 (ensemble-decomposed WR) | ship | ship + flag pinball-weight stability in PR description | stop, investigate; decision: ship lgb-nb integration anyway (gates 1+2 ADOPT on isolated children) vs. revert |

If gates 1 and 2 both miss the probe point estimate by >50% (e.g., observed ΔRMSE < −0.030 fpts at QB instead of ≈ −0.059), this signals a builder-side bug — most likely sign-convention drift in `_shared.build_game_environment` flow-through, week-1 broadcast off-by-one, or a join-key mismatch in `attach_vegas_team_context_features`. **Do not ship until reconciled with the probe's override audit (`reports/feature_probe_vegas_team_context_override_audit.md`).**

### 5.3 Report

`reports/qb_wr_vegas_team_context_integration_summary.md` — three gate verdicts in a table, point estimates vs. probe predictions, MARGINAL/REGRESSION caveats if any, link back to predecessor probe summary.

## 6. Phasing (per CLAUDE.md ≤5 files per phase)

### Phase 0 — schema + builder

Files touched (4):

- `src/projections/schemas.py` (add 8 cols total — 4 in each of QbFeaturesSchema + WrFeaturesSchema).
- `src/projections/features/qb.py` (add `attach_vegas_team_context_features` call).
- `src/projections/features/wr.py` (same).
- New tests under `tests/test_schemas/` and `tests/test_features/`.

Verification: `pytest -v tests/test_schemas tests/test_features` clean; `mypy src tests`; `ruff check + format --check src tests`.

### Phase 1 — lgb-nb feature-list override

Files touched (2):

- `src/projections/models/lightgbm_nb.py` (add `_swap_for` helper, two override tuples, factory updates, extend `_code_hash_files_nb`).
- New / extended tests under `tests/test_models/`.

Verification: `pytest -v tests/test_models`; `mypy src tests`; `ruff check + format --check src tests`. Full-suite pytest narrow subset: `pytest -k "lightgbm_nb or lightgbm or ensemble or features"`.

### Phase 2 — gates + report

No source changes. Steps:

1. Switch to `main` (PR #49 merge), refresh feature parquets via `scripts/refresh_features.py`, train each model class, save baseline snapshots under a named timestamp directory.
2. Switch back to this branch, refresh feature parquets (now populated with the four new cols), train, save candidate snapshots.
3. Run the three `adoption_gate.py --baseline-run ... --candidate-run ...` invocations.
4. Write `reports/qb_wr_vegas_team_context_integration_summary.md`.
5. Update TODO #33c entry with integration verdict (ADOPT / MARGINAL / REGRESSION per gate; ship/stop decision).
6. Update `project_management.md` with the Phase 2 outcome.

## 7. Caveats (carried from PR #49)

- ΔRMSE −0.06 fpts at QB is ≈ 1–2% per-week composite. **The Chase 250→403 elite-magnitude gap is not closed by this integration alone.** Necessary but not sufficient.
- If gates 1 + 2 ADOPT but the elite-magnitude problem persists in `reports/qb_wr_vegas_team_context_integration_summary.md`'s post-integration retrospective on 2024 actuals, the next direction is **external preseason Vegas data** (genuine May win totals, OC/HC tenure, FA-acquisition flag, projected pace, projected pass rate). That is a separate spec; the present integration's job is to ship the cheap signal that's already in `spread_line` / `total_line`.
- RB just-missed-ADOPT — the binding constraint may be that `season_avg_*` leaks early-season info that hurts the team-strength estimate; a `preseason_*`-only follow-up probe is queued separately. TE NULL — closed.

## 8. Reverse compatibility

- Schema changes are additive (new nullable cols); existing feature parquets on disk will be re-generated on the next `scripts/refresh_features.py` run. No migration path needed — feature parquets are regenerable from raw ingest.
- Cached model_ids for QB + WR lgb-nb will change (feature-list change → code-hash change → model_id change). This is correct: callers reading these snapshot keys must re-fit. Existing backtest snapshots under `tests/backtest/model_metrics.json` will be partially invalidated for QB + WR lgb-nb rows; those rows will be re-written by Phase 2's candidate snapshot regeneration.
- BaselineModel + DecomposedBaselineModel + LightGBMModel + LightGBMTunedModel code-hashes and model_ids are unchanged (hardcoded feature lists; schema-derived lists pick up the four cols but the feature list source-of-truth shifts only on the lgb-nb side).

## 9. References

- Predecessor probe spec: `docs/superpowers/specs/2026-05-17-vegas-team-context-probe-design.md`.
- Predecessor probe plan: `docs/superpowers/plans/2026-05-17-vegas-team-context-probe.md`.
- Predecessor probe summary: `reports/feature_probe_vegas_team_context_summary.md`.
- Predecessor probe override audit (sign-convention and coverage sanity): `reports/feature_probe_vegas_team_context_override_audit.md`.
- RB PBP integration as a structural template: `docs/superpowers/specs/2026-05-01-rb-pbp-features-design.md`.
- WR ensemble-decomposed child spec (defines the production WR route): `docs/superpowers/specs/2026-05-15-wr-ensemble-decomposed-child-design.md`.
- Plan 5c (lgb-nb): TODO #26 entry in `TODO.md`; Plan 6 (ensemble): TODO #29 entry.
- Plan 9 / dual-run gate harness: `scripts/adoption_gate.py`.
