# Feature Signal Probe — Design

**Status:** approved (brainstorming, 2026-04-30). Ready for implementation plan.
**Date:** 2026-04-30
**Author:** alden + claude
**Sub-project of:** FantasyFootball / Projections Core
**Builds on:** Plan 8 (PR #16, merged at `6675359`) and Plan 9 (PR #17, merged at `220298f`) — depends on the per-row `results.parquet` shape and the pure-stats `paired_bootstrap_*` helpers in `src/projections/backtest/adoption_gate.py`. Branched from `main` at `220298f` onto `feat/feature-signal-probe`.

---

## 1. Overview

Plan 9 spent 16 tasks shipping a feature swap whose Δ-RMSE point estimate at the BaselineModel level turned out to be `+0.0005` fpts on QB and `+0.0001` on RB — both well below Plan 8's measured per-cell noise floor (~1.3% RMSE half-width, ≈ 0.08 fpts). The post-mortem identified the root cause as a planning miscalibration: the "5–15% RMSE" estimate in TODO #3 was a *family*-level prior across six PBP-derived features, but Plan 9 applied it to a single-feature swap and never decomposed.

The Feature Signal Probe is the cheapest tool that prevents this class of failure: a CLI + pure-stats module that takes a baseline feature set and a candidate-column override, and reports per-stat Δ-CV-RMSE bootstrap CIs (Phase 1) plus, conditionally, composite fpts ΔRMSE (Phase 2) — answering "is there enough signal here to be worth scoping a full plan around?"

The probe is a **screening** tool, not a model-evaluation tool. A `SIGNAL` verdict is a green light to write a spec; a `NULL` verdict says "don't write the spec, the candidate is below the noise floor." It is also explicitly *not* a replacement for the adoption gate: the adoption gate is the final word on whether a feature change ships.

### 1.1 Goals (in scope)

- New pure-stats module `src/projections/backtest/feature_probe.py` exporting:
  - `PerStatVerdict` dataclass: `(position, stat, year_or_pooled, rmse_delta: BootstrapDelta, r_squared_delta: float, verdict: Literal["SIGNAL", "NULL", "REGRESSION"])`. Reuses Plan 8's `BootstrapDelta` from `adoption_gate.py`.
  - `probe_per_stat(features_baseline, features_candidate, weekly_stats, *, position, holdout_years, n_bootstrap=1000, seed=42) -> list[PerStatVerdict]` — one verdict per `(stat, holdout_year)` plus one position-pooled-across-years row per stat.
  - `ProbeReport` dataclass bundling Phase 1 verdicts + Phase 2 verdicts (when fired) for rendering.
  - `phase1_should_fire_phase2(verdicts: list[PerStatVerdict]) -> bool` — `True` iff any cell has `verdict == "SIGNAL"`.
- New CLI `scripts/probe_feature_signal.py`:
  - Args: `--candidate-name <label>` (required, used in report header), `--baseline-features <path>` (default `data/features/`), `--override <parquet>` (repeatable, required to define the candidate), `--drop <col1,col2>` (optional, comma-separated), `--model {baseline,lightgbm-nb}` (default `baseline`), `--position {QB,RB,WR,TE}` (repeatable, default all 4), `--seasons <S-E>` (default `2018-2024`), `--holdout-years <S-E>` (default `2021-2024`), `--n-bootstrap <int>` (default 1000), `--seed <int>` (default 42), `--csv-out <path>` (optional), `--composite/--no-composite` (default `--composite` — Phase 2 still gated by Phase 1 result).
  - Reads cached features via `projections.features.cache.read_features`, applies overrides + drops, runs Phase 1, conditionally runs Phase 2, prints a per-position markdown report to stdout, optionally writes a CSV.
  - Always exits 0; this is informational.
- New test files: `tests/test_backtest/test_feature_probe.py` (stats module) and `tests/test_scripts/test_probe_feature_signal_cli.py` (CLI), following the Plan 8 test pattern (synthetic fixtures, no network, no real backtest).
- One example `--override` parquet committed under `tests/test_backtest/fixtures/` for the CLI test, demonstrating the file shape.

### 1.2 Non-goals (deferred)

- **No new feature builder code.** The probe consumes whatever the user produces externally as override parquets. Generation is the user's responsibility (notebook, ad-hoc script, modified `refresh_features.py`).
- **No automatic adoption decision.** The probe predicts the gate verdict; only the real adoption gate ships routing changes.
- **No persistent run artifacts under `data/`.** Outputs go to stdout + optionally `reports/`. The probe is rerunnable; nothing snapshots its history.
- **No multi-candidate sweeps.** One probe invocation = one candidate comparison. To compare three variants (e.g., augment vs swap vs drop), invoke three times. Aggregating across runs is a future tool if it ever matters.
- **No widening to other model classes beyond `baseline` and `lightgbm-nb`.** `lightgbm`, `lightgbm-tuned`, and `ensemble` are accessible via the same factory dispatch but adding them as `--model` choices is unnecessary right now: `lightgbm-nb` strictly dominates `lightgbm-tuned` on RMSE (Plan 5c verdict) and is the natural tree-model probe target.
- **No within-row covariance modeling.** Paired bootstrap on `(gsis_id, season, week)` rows treats each row as exchangeable. Block bootstrap by `(season, week)` would handle within-week correlation but Plan 8 already validated this is overkill for the adoption gate; the probe inherits that judgment.
- **No tuning of the per-stat verdict thresholds.** `verdict == "SIGNAL"` iff the bootstrap CI is strictly below 0; same statistical rule as Plan 8's adoption gate, just applied per stat instead of per composite.
- **No probe of LightGBM/NB-specific hyperparameters.** The probe uses the production-tuned hyperparameters from `data/tuned_params/lightgbm.json` (Plan 5b) when `--model lightgbm-nb` is selected. Hyperparameter tuning of a candidate feature is out of scope and arguably wrong (an over-tuned probe answers a different question).

### 1.3 Success criteria for trusting the probe's verdict

The probe is useful iff its verdict reliably predicts the adoption-gate verdict. Two retroactive sanity checks pin this down at v1:

1. **Plan 9 retro: the probe must return zero `SIGNAL` cells at the *pooled* level on every position when fed the post-Plan-9 EPA residual columns as overrides + the v1 `opp_allowed_*_fppg_l4` as drops** (i.e., reproduce the Plan-9 swap as a probe input). The adoption gate said `DO_NOT_ADOPT` × 4; the probe must agree on the pooled-across-years per-stat verdicts because that is what `phase1_should_fire_phase2` consumes (per-year `SIGNAL` cells are informational only — they may flag genuine per-year RMSE improvements that wash out at the pooled level, which is exactly the per-year-vs-pooled distinction the probe is designed to surface). Anything else means the probe over-fires Phase 2 on noise — fixable by tightening the verdict rule (e.g., raising `--effect-size-floor`), but a v1 release-blocker if it happens after the default calibration.

2. **Plan 9 retro inverse: the probe's *augment* variant must not produce a verdict worse than the *swap* variant.** Concretely, the swap retro returned `REGRESSION` on WR (RMSE delta strictly above zero). The augment variant — same override, no drop — strictly dominates swap in information content (Ridge can shrink the new column to zero coefficient if it's redundant), so augment must not produce `REGRESSION` on any cell where swap is `NULL` or better. This is a *no-worse-than* check, not a *must-SIGNAL* check: if Ridge shrinks the candidate column to zero, augment ≈ baseline → all `NULL`, which is fine — it just means the new column carries no orthogonal signal under the regularization the production model uses.

If criterion 1 holds but criterion 2 fails (augment regresses where swap merely was null), the probe is unstable in a way that erodes trust — investigate before merging. If both hold, the probe is calibrated and ready to use as the gate for future feature plans.

These two checks are run-once real-data validations (see §7.2), against the live PBP partitions on disk. They are not committed tests — committed tests use synthetic fixtures.

---

## 2. Inputs

### 2.1 CLI args

```
python -m scripts.probe_feature_signal \
  --candidate-name "opp_epa_residual_swap" \
  --baseline-features data/features \
  --override data/features_probe/opp_epa_residual.parquet \
  --drop opp_allowed_qb_fppg_l4,opp_allowed_rb_fppg_l4,opp_allowed_wr_fppg_l4,opp_allowed_te_fppg_l4 \
  --model baseline \
  --csv-out reports/feature_probe_opp_epa_swap.csv
```

All args have defaults except `--candidate-name` and at least one of `--override` or `--drop`. The CLI rejects with a clear error if both `--override` and `--drop` are absent (a no-op probe is meaningless).

`--position` (repeatable) filters to a subset; default is all 4. Per-position dispatch matches `POSITION_DISPATCH` exactly — same factory the adoption gate uses, same feature schema, same target stats.

`--composite / --no-composite`: Phase 2 is opt-in-able to skip even on `SIGNAL`, useful when running a quick screen. Default is `--composite` (run Phase 2 if Phase 1 fires it).

### 2.2 Override parquet shape

```
columns:
  gsis_id    : str (pyarrow)            — required, GSIS-id-format-checked
  season     : Int64 nullable          — required
  week       : Int64 nullable          — required
  <candidate columns>                  — one or more, any name
```

The probe rejects with a clear error if:
- The override is missing any of `gsis_id`/`season`/`week`.
- An override column name collides with an existing baseline feature column AND is not in `--drop` (silent override would mask which feature is actually being tested).
- Override row coverage of the joined baseline rows is below 95% on any (position, season) pair (silent NaN-imputation under Ridge would mask the candidate's signal — see §8 risks).

Multiple `--override` files are concatenated columnwise on the `(gsis_id, season, week)` key before the join. Different override files can carry different candidate columns (e.g., one for QB-relevant, one for WR-relevant), but each `(gsis_id, season, week)` row must appear in at most one file per candidate column.

The drop list is applied **after** override join, so an override file can both *replace* (override + drop) and *augment* (override only) within the same probe invocation depending on the `--drop` arg.

### 2.3 What the probe loads

For each `--position`:
1. `features_baseline = read_features(position, season)` for each season in `--seasons` range, concatenated.
2. Left-join overrides on `(gsis_id, season, week)`. Validate coverage threshold.
3. Drop `--drop` columns from the schema's expected feature list (see §3 step 1).
4. `weekly_stats = read_partition(...)` for the same season range — used as fit truth, identical to `BaselineModel.fit`'s upstream load.

The probe never writes to `data/features/`. The override parquet is the only durable artifact representing the candidate.

---

## 3. Phase 1 — per-stat screening

Per-stat ΔRMSE on a Ridge regressor is the cheapest way to detect "does this column carry signal independent of the existing features." If a column adds zero per-stat predictive accuracy across all stats, no downstream composition can recover signal — scoring is just `Σ stat_i × weight_i`, which preserves zero.

For each `(position, target_stat, holdout_year)` triple:

1. Look up the position's `BaselineModel` factory; its `feature_columns` tuple is the baseline feature list. Apply `--drop` to produce `baseline_cols`. Append the override-supplied candidate column names to produce `candidate_cols` (which is `baseline_cols ∪ overrides`).
2. Inner-join features with `weekly_stats` on `(gsis_id, season, week)` and drop NaN rows on `candidate_cols ∪ {target_stat}`. This is the same join+dropna logic as `BaselineModel.fit` lines 487–522. Both feature sets must be evaluated on the **same** post-dropna row set so the bootstrap is paired — a row that is NaN on the candidate but valid on the baseline must be dropped from both.
3. Train Ridge on `[seasons.start, holdout_year - 1]` rows with `baseline_cols`. Train a second Ridge on the same rows with `candidate_cols`. Identical RidgeCV alpha grid and CV strategy as `BaselineModel.fit` (the `np.logspace(-3, 3, 13)` alphas).
4. Predict on `holdout_year` rows; compute per-row squared residuals `sq_resid_baseline` and `sq_resid_candidate` (each is `(y - ŷ)²` for one row).
5. Paired bootstrap (n=1000, seed=42) of `√mean(sq_resid_candidate) - √mean(sq_resid_baseline)` across rows. This is the per-stat Δ-CV-RMSE point estimate + 95% CI. **Reuse `paired_bootstrap_rmse_delta` from `src/projections/backtest/adoption_gate.py`** rather than re-implementing — same statistical machinery, same seed convention.
6. Compute in-sample ΔR² as a diagnostic: `1 - (sum sq_resid_candidate / sum (y - ȳ)²)` minus the same for baseline. Reported but not used for verdict. (The `r_squared_delta` field on `PerStatVerdict` carries this number — distinct from the per-row squared residuals above.)

Per-cell verdict (one per `(position, stat, holdout_year)`):
- `SIGNAL` if `rmse_delta.hi_95 < 0` (candidate strictly improves CV-RMSE; tiny effect can pass with enough rows).
- `REGRESSION` if `rmse_delta.lo_95 > 0` (candidate strictly worsens; rare but possible if the column is noise that competes for L2 budget).
- `NULL` otherwise (CI brackets 0).

Per-position-pooled rows (one per `(position, stat, "pooled")`): same bootstrap, but resample over all 4 holdout years' rows together. The pooled CI is what the screening verdict reports as the headline; per-year rows are informational.

`phase1_should_fire_phase2` returns `True` iff any per-cell or pooled verdict in the run is `SIGNAL`.

---

## 4. Phase 2 — composite (gated by Phase 1)

Phase 2 fires iff Phase 1 returned at least one `SIGNAL` cell **and** `--composite` is on. It answers "would this pass the adoption gate?"

Phase 2 is **walk-forward**: one fit per `(position, holdout_year)`, training on `[seasons.start, holdout_year - 1]` and predicting on `holdout_year`. Per-year predictions are then concatenated and the bootstrap pools across years — identical to what the backtest harness + production adoption gate do.

For each `(position, holdout_year)` in the active position × year set:

1. Use `POSITION_DISPATCH[position].factories[args.model]()` to build a fresh unfitted model instance (`--model baseline` → BaselineModel; `--model lightgbm-nb` → LightGBMNbModel). Build two: one keyed to `baseline_cols`, one keyed to `candidate_cols` (via the `_build_factory_with_columns` helper described in §6.1).
2. Fit baseline-feature model on `[seasons.start, holdout_year - 1]` features (with `--drop` applied), predict on `holdout_year`. Same for candidate-feature model. Each `predict_distribution` call produces a per-row prediction frame with the same shape as the backtest harness's `results.parquet` (per-row `mean`, `p10`, `p50`, `p90`, etc.).
3. After the per-year loop, concatenate the per-row prediction frames into two long DataFrames `baseline_predictions` and `candidate_predictions` (each ~3,000–8,000 rows depending on position, pooled across the active holdout years).
4. Hand both to `paired_bootstrap_rmse_delta` and `paired_bootstrap_spearman_delta` from `src/projections/backtest/adoption_gate.py`. The gate's existing pairing on `(gsis_id, season, week)` works as-is.
5. Apply `verdict_for_position` from the same module, which returns the `ADOPT` / `MARGINAL` / `DO_NOT_ADOPT` decision under the same rule the production adoption gate uses.

The Phase 2 verdict is the probe's **prediction** of the adoption-gate verdict for this candidate under the chosen model class. If the user later runs the full backtest + adoption-gate workflow on the same candidate, the verdicts should match — divergence between probe and gate is a probe bug.

The probe does not write to `data/backtest/`. Phase 2 holds prediction frames in memory only; reports go to stdout + optional CSV.

---

## 5. Outputs

Markdown to stdout. One header block, one Phase 1 section per position, one Phase 1 summary, optionally one Phase 2 section per position + summary.

```
# Feature signal probe — opp_epa_residual_swap

Baseline features: data/features
Overrides:        data/features_probe/opp_epa_residual.parquet
Drops:            opp_allowed_qb_fppg_l4, opp_allowed_rb_fppg_l4, opp_allowed_wr_fppg_l4, opp_allowed_te_fppg_l4
Model class:      baseline
Holdout years:    2021-2024
n_bootstrap:      1000, seed: 42

## Phase 1 — per-stat screening

### QB
  stat               year      n  rmse_delta   ci_lo    ci_hi    r²_delta  verdict
  passing_yards      2021     665   +0.0125   -0.4521   +0.4895    -0.0001  NULL
  passing_yards      2022     657   ...
  ...
  passing_yards      pooled  2676   +0.0050   -0.1023   +0.1112    -0.0001  NULL
  passing_tds        pooled  2676   ...
  ...

### RB / WR / TE — same shape

## Phase 1 verdict

No SIGNAL cells across any (position, stat, holdout). Probe predicts adoption gate would return DO_NOT_ADOPT for every position. **Phase 2 skipped.**
```

Or, if Phase 1 fires Phase 2:

```
## Phase 1 verdict

SIGNAL on QB / passing_yards / 2024 (rmse_delta -0.42 fpts, CI [-0.71, -0.13]).
1 cell triggers Phase 2 — running composite evaluation...

## Phase 2 — composite ΔRMSE

| Position | Verdict | RMSE delta (95% CI)             | Spearman delta (95% CI)       | n_paired |
|---|---|---|---|---:|
| QB | DO_NOT_ADOPT | -0.0420 ([-0.1850, +0.1011]) | +0.0012 ([-0.0028, +0.0052]) | 2676 |
| RB | DO_NOT_ADOPT | ... | ... | 5273 |
...

## Probe verdict

Phase 1: 1/96 cells SIGNAL.
Phase 2: 0/4 positions ADOPT.
Predicted adoption gate: DO_NOT_ADOPT × 4. Recommend not scoping a full plan around this candidate.
```

`--csv-out` produces a long-format CSV suitable for scripting: one row per `(phase, position, stat_or_composite, year_or_pooled, metric_name, point, lo_95, hi_95, verdict)`. Same shape convention as `reports/adoption_gate_*.csv` so existing tooling reads both.

---

## 6. Code shape

### 6.1 New module `src/projections/backtest/feature_probe.py`

Pure stats module — same shape as `src/projections/backtest/adoption_gate.py`. Imports `BootstrapDelta`, `PositionVerdict`, `paired_bootstrap_rmse_delta`, `paired_bootstrap_spearman_delta`, `verdict_for_position` from `adoption_gate.py` and reuses them unchanged.

```python
@dataclass(frozen=True)
class PerStatVerdict:
    position: Position
    stat: Stat
    year_or_pooled: int | Literal["pooled"]
    n_paired: int
    rmse_delta: BootstrapDelta
    r_squared_delta: float                                  # in-sample, diagnostic
    verdict: Literal["SIGNAL", "NULL", "REGRESSION"]


@dataclass(frozen=True)
class ProbeReport:
    candidate_name: str
    model_class: str                                        # "baseline" or "lightgbm-nb"
    phase1: list[PerStatVerdict]                            # per (position, stat, year/pooled)
    phase2: list[PositionVerdict] | None                    # None if Phase 2 not fired


def probe_per_stat(
    *,
    position: Position,
    features_baseline_cols: list[str],
    features_candidate_cols: list[str],
    features: pd.DataFrame,                                 # already-joined, post-drop, post-override
    weekly_stats: pd.DataFrame,
    target_stats: tuple[Stat, ...],
    holdout_years: tuple[int, ...],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[PerStatVerdict]: ...


def probe_composite(
    *,
    position: Position,
    factory_baseline: Callable[[], Model],                  # produces unfitted model w/ baseline cols
    factory_candidate: Callable[[], Model],                 # produces unfitted model w/ candidate cols
    features_baseline: pd.DataFrame,
    features_candidate: pd.DataFrame,
    weekly_stats: pd.DataFrame,
    holdout_years: tuple[int, ...],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> PositionVerdict: ...


def phase1_should_fire_phase2(verdicts: list[PerStatVerdict]) -> bool: ...
```

The factory injection in `probe_composite` is what lets us swap in a candidate-feature model: the position factory's default `_X_FEATURE_COLUMNS` constant is replaced by a tuple closure that returns `candidate_cols` instead. Concretely, `feature_probe._build_factory_with_columns(position, model_class, columns)` returns a zero-arg callable suitable to pass as `factory_candidate`. This avoids monkey-patching the production factory tuples.

### 6.2 New CLI `scripts/probe_feature_signal.py`

Argparse + I/O glue + markdown rendering. Mirrors `scripts/adoption_gate.py`'s shape:

- `parse_args()` (extracted for testability — same pattern Plan 9 introduced for `adoption_gate.py`).
- `load_features_with_overrides(...) -> pd.DataFrame` (pure function — no I/O after the initial reads, easy to test against a synthetic baseline + override).
- `validate_override_coverage(...)` — raises a clear error with row-count detail if coverage threshold breached.
- `render_markdown(report: ProbeReport) -> str` and `render_csv(report: ProbeReport) -> str`.
- `main(argv: list[str] | None = None)` — argparse → load → probe → render → print/write.

### 6.3 Tests

- `tests/test_backtest/test_feature_probe.py` — synthetic-fixture coverage of `probe_per_stat` and `probe_composite`. Cases:
  - Per-stat: zero-effect column → `NULL` verdict; high-signal column → `SIGNAL`; noise column with negative coefficient under Ridge regularization → `NULL`/`REGRESSION` boundary.
  - Composite: gated correctly by phase 1; fires only when Phase 1 returns `SIGNAL`.
  - `phase1_should_fire_phase2` truth table.
  - Determinism: same seed → same verdicts.
- `tests/test_scripts/test_probe_feature_signal_cli.py` — end-to-end CLI pattern (mirrors `tests/test_scripts/test_adoption_gate_cli.py`):
  - Happy path with a small fixture override.
  - `--override` + `--drop` combined → swap mode.
  - `--override` only → augment mode.
  - `--drop` only → ablation mode.
  - Coverage-below-threshold → clear error.
  - Override-column-collides-with-baseline-column-without-drop → clear error.
  - `--no-composite` skips Phase 2 even on `SIGNAL`.
- `tests/test_backtest/fixtures/probe_override_example.parquet` — the canonical example override committed alongside the tests, generated by a `make_fixture()` helper in the test module.

---

## 7. Test plan + retrospective validation

### 7.1 Synthetic-fixture tests (committed)

Per §6.3 above. All run in CI under `pytest`; no network, no real data dependencies.

### 7.2 Real-data validation (run-once, not committed)

Two retrospective probes against the merged Plan 9 work, satisfying the §1.3 success criteria:

1. **Plan 9 swap retro:** generate an override parquet at `data/features_probe/plan9_swap_retro.parquet` with the EPA residual columns from a one-off run of the (deleted) Plan 9 `opp_epa_allowed_residual` helper against the `data/raw/pbp/` partitions that PR #17 plumbed in. Then:
   ```
   python -m scripts.probe_feature_signal \
     --candidate-name "plan9_swap_retro" \
     --override data/features_probe/plan9_swap_retro.parquet \
     --drop opp_allowed_qb_fppg_l4,opp_allowed_rb_fppg_l4,opp_allowed_wr_fppg_l4,opp_allowed_te_fppg_l4 \
     --csv-out reports/feature_probe_plan9_swap_retro.csv
   ```
   Expected: probe returns `NULL` (no `SIGNAL`) per criterion 1.

2. **Plan 9 augment retro:** same override, no drop:
   ```
   python -m scripts.probe_feature_signal \
     --candidate-name "plan9_augment_retro" \
     --override data/features_probe/plan9_swap_retro.parquet \
     --csv-out reports/feature_probe_plan9_augment_retro.csv
   ```
   Expected: probe returns `SIGNAL` on at least one cell per criterion 2.

If both pass, the probe is calibrated — merge. If criterion 1 fails (false positive on a known-no-signal candidate), tighten the per-stat verdict rule before merging. If criterion 2 fails (false negative on a strictly-more-info candidate), the probe is too insensitive — investigate before relying on it.

These two reports get committed to `reports/` as record-of-validation, the same way Plan 8's adoption gate re-evaluation reports were. The override parquet under `data/features_probe/` is **not** committed — it's regenerable from the live PBP partitions and shouldn't bloat the repo.

### 7.3 Standard verification gates

Per `CLAUDE.md` end-of-effort checklist: `pytest -v` (full suite), `mypy src tests`, `ruff check src tests scripts`, `ruff format --check src tests scripts`. All zero-violation.

---

## 8. Risks

- **Override coverage gaps mask signal.** If the override parquet covers, say, only 80% of the baseline's `(gsis_id, season, week)` rows, Ridge fills the rest with NaN → drop, halving the training set on the candidate side, biasing Δ-CV-RMSE. The 95% coverage threshold is a hard reject; the user must produce an override that covers ≥95% of rows or fix their generation script. Lower thresholds would silently mislead.
- **Bool coercion.** Candidate columns of dtype `bool` get coerced via `BaselineModel._x_frame_with_bool_coercion` (int8). Already covered by the existing `_x_frame_with_bool_coercion` path; the probe inherits it for free.
- **LightGBM-NB cost.** A full Phase 2 run with `--model lightgbm-nb` is ~1–2 hr (24 cells × 4 folds × ~1–3 min/cell for NB tree training). Acceptable as the slow path; the documentation will state this clearly so the user doesn't kick it off without intent.
- **Phase 2 false positives at composite level.** Plan 6 and Plan 7's diagnostic both showed per-stat optimization doesn't decompose to composite calibration. The probe deliberately does the *opposite* — it gates Phase 2 on per-stat *signal*, not composite signal. So a per-stat `SIGNAL` cell can still produce a Phase 2 `DO_NOT_ADOPT` (the per-stat win didn't survive composition). This is correct behavior: the probe says "worth running the full evaluation," not "guaranteed to pass the gate." The Phase 2 result is the actual gate prediction.
- **Probe is not a substitute for the adoption gate.** A `SIGNAL` verdict is necessary but not sufficient for shipping. The full backtest + adoption gate is still required for any feature change that reaches production routing. Documenting this explicitly in the CLI's `--help` text and the report's footer is part of the implementation.
- **Probe verdict drift between runs.** Bootstrap with fixed seed is deterministic; same input → same verdict. Override parquet regeneration (e.g., re-running the helper that produced it) may produce subtly different values due to upstream `nfl_data_py` data updates — verdicts may shift accordingly. The report header captures the override path's `mtime` for traceability.

---

## 9. Documentation updates on merge

- **`project_management.md`:**
  - Append a "Feature Signal Probe" entry at the top (status: complete; calibration verified against Plan 9 retro).
  - Add a decision-log row noting the probe is the canonical pre-spec screening step for any future feature plan, citing TODO #3b/3c (remaining PBP-derived feature candidates) as the immediate beneficiary.
- **`TODO.md`:**
  - No new entries; this closes a process gap rather than opening a workstream.
  - Optionally cross-reference from TODO #3b ("remaining PBP-derived feature candidates") noting that the probe must be run before scoping each candidate.
- **`CONTRIBUTING.md`:**
  - Add a one-paragraph "Feature plan workflow" subsection: before scoping any feature plan, generate an override parquet and run `scripts/probe_feature_signal.py`. If the probe returns `NULL` × all positions, decompose the plan (bundle multiple candidates, change model class, or shelve) before writing the spec.
- **`docs/superpowers/specs/_feature_plan_template.md`** (optional, follow-up if useful): a §1.3-equivalent template for feature plans that requires citing a probe report. Defer until we've used the probe a few times and know what shape this needs.
