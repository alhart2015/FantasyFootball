# Plan 3c — Walk-forward backtest harness with snapshot-diff gating — Design

**Status:** approved (brainstorming)
**Date:** 2026-04-26
**Author:** alden + claude
**Builds on:** `2026-04-25-plan-3b-qb-rb-te-baseline-design.md` (the per-position Model A baselines this plan exercises). Closes TODO #4 (feature parquet storage).
**Plan-3 series context:**
- **Plan 3a (merged at `598ab9c`):** Model A on WR only. Pinned the per-week interface, joblib persistence, first real-data ingest.
- **Plan 3b (merged at `c4a0401`):** Generalized Model A to QB / RB / TE. Four trained artifacts; per-position 2024 sanity checks; `POSITION_DISPATCH` registry.
- **Plan 3c (this design):** Walk-forward backtest harness + snapshot-diff CI gate, with feature parquet caching to make it tractable. Uses summed weekly means as season totals (degenerate aggregation).
- **Plan 3d (next, not in this design):** Real Monte Carlo season-distribution aggregation. Closes TODO #13 (per-row seeds) and TODO #14 (SAMPLED_SUMMARY family). Adds season-total calibration to the gate.

---

## 1. Overview

3a/3b validated that Model A produces sensible per-week distributions on real data and recorded informational sanity-check numbers in `project_management.md`. Those numbers do not gate anything: nothing fails CI if a future change quietly worsens the model. Plan 3c turns the existing sanity-check shape into a **regression gate** by running a walk-forward backtest across four held-out years, comparing the resulting metrics against a committed snapshot, and failing pytest if any metric regresses beyond its tolerance.

Plan 3c is **deliberately scoped to gating**, not to producing publishable season distributions. The season-aggregation question (Monte Carlo over per-week distributions, with bye-week and availability handling) is its own design surface — it forces the per-row-seed and SAMPLED_SUMMARY decisions to converge with the harness — and is deferred to Plan 3d. Plan 3c's "season total" is just `groupby('gsis_id').sum()` of weekly mean predictions, which is enough for Spearman top-N gating.

### 1.1 Goals

- A new `src/projections/backtest/` package: walk-forward driver, metrics, naive baseline computation, snapshot read/write/diff.
- A new feature-cache layer at `data/features/{position}/season=YYYY/week=WW/part.parquet` with a `scripts/refresh_features.py` writer and an `src/projections/features/cache.py` reader. Closes TODO #4.
- A committed snapshot file at `tests/backtest/baseline_metrics.json` capturing the v1 Model A metrics across (position, year, metric).
- A committed tolerance-config file at `tests/backtest/tolerances.json` defining per-metric-type defaults plus an empty `overrides` list for per-row exceptions.
- A pytest test at `tests/backtest/test_backtest_gate.py` gated behind `@pytest.mark.backtest` and `--run-backtest`, plus a default-on smoke test that exercises a single (position, year) cell to keep wiring under test without paying full runtime.
- A CLI runner at `scripts/backtest.py` with three modes: `--check` (default; runs harness, diffs snapshot, prints gate result), `--update-snapshot` (regenerates snapshot from a fresh run), `--report` (prints model + naive metrics side by side, no gate).
- Phase 6 produces the v1 snapshot, runs the gate, and records results in `project_management.md` mirroring the 3a/3b sanity-check sections.

### 1.2 Non-goals (deferred)

- **Real Monte Carlo season-distribution aggregation** → Plan 3d. Plan 3c sums weekly means; season-total calibration metrics are not in the gate.
- **TODO #13 (per-row seeds in `score_distribution`)** → Plan 3d. Cross-row sample correlation does not affect Plan 3c's metrics because we only consume per-row means/quantiles, not joint samples.
- **TODO #14 (SAMPLED_SUMMARY family decision)** → Plan 3d.
- **`score_distribution` vectorization perf TODO** → Plan 3d. The harness scale (~16 fits × ~17 weeks × small player count per position) does not stress this code path because feature caching means we predict once per (player-week, year), not per training fold.
- **Catastrophic absolute floors** ("Spearman < 0.5 means broken regardless of snapshot") → defer until snapshot-only gating produces a real-world false negative.
- **GitHub Actions CI workflow** → out of scope per project direction (CI deferred indefinitely). The gate runs as opt-in pytest invoked manually pre-PR.
- **K / DST positions** → TODO #10. Harness operates only on the four positions Model A covers.

---

## 2. Architecture

### 2.1 New packages and files

```
src/projections/
├── backtest/                     # NEW PACKAGE
│   ├── __init__.py               # exports: run_backtest, GateResult, diff_snapshot
│   ├── harness.py                # walk-forward driver: train, predict, score per (position, year)
│   ├── metrics.py                # per-stat RMSE/MAE, composite RMSE/MAE, Spearman, calibration
│   ├── naive.py                  # per-player trailing-4-game stat mean baseline
│   └── snapshot.py               # snapshot file IO + diff logic + tolerance application
└── features/
    └── cache.py                  # NEW: write_features / read_features (consumed by harness + future plans)

scripts/
├── refresh_features.py           # NEW: rebuild data/features/ from data/raw/
└── backtest.py                   # NEW: CLI runner

tests/backtest/                   # NEW TEST DIRECTORY
├── baseline_metrics.json         # gated snapshot, ~352 rows
├── tolerances.json               # per-metric-type defaults + per-row overrides
├── test_backtest_gate.py         # @pytest.mark.backtest test that runs harness + diffs snapshot
└── test_backtest_smoke.py        # default-on smoke covering one (position, year) cell

tests/test_backtest/              # NEW (mirrors tests/test_models/)
├── test_harness.py               # unit tests against synthetic fixtures
├── test_metrics.py               # unit tests for RMSE/Spearman/calibration computations
├── test_naive.py                 # unit tests for naive-baseline computation
└── test_snapshot.py              # unit tests for diff logic + tolerance application
```

No changes to `src/projections/models/`, `src/projections/features/{qb,rb,te,wr}.py`, `src/projections/ingest/`, or `src/projections/store/`. Plan 3c is additive everywhere except `pyproject.toml` (registers the `backtest` marker and adds the `--run-backtest` CLI flag through `tests/conftest.py`).

### 2.2 Feature cache layer

**Read path (`src/projections/features/cache.py`):**

```python
def read_features(
    position: Position,
    season: int,
    *,
    weeks: Iterable[int] | None = None,
    features_root: Path = Path("data/features"),
) -> pd.DataFrame:
    """Load cached features for a (position, season). Returns concatenated rows
    across the requested weeks (default: all available weeks for the season),
    re-validated against the appropriate FeaturesSchema.
    Raises FileNotFoundError if the cache for that (position, season) is missing."""
```

Returns a single DataFrame validated against the appropriate `*FeaturesSchema` (looked up via `POSITION_DISPATCH`). Strict — does not silently fall through to in-place feature computation; the harness assumes the cache is populated and surfaces a clear error if not.

**Write path (`scripts/refresh_features.py`):**

```
python scripts/refresh_features.py {position|all} [--seasons RANGE] [--data-root PATH]
```

Iterates the cartesian product (positions × seasons × weeks), calls each builder via `POSITION_DISPATCH[position].feature_builder(...)`, validates against the appropriate schema, and writes through `store.write_partition`. Idempotent — `write_partition` overwrites by design.

**Layout:** `data/features/{position}/season=YYYY/week=WW/part.parquet`. `{position}` is lowercase (`qb`, `rb`, `te`, `wr`) to match the existing `data/raw/{table}/season=YYYY/week=WW/...` and `data/projections/weekly/ruleset=.../season=YYYY/week=WW/...` patterns.

**Invalidation:** v1 is **manual**. If the user touches a feature builder, they re-run `scripts/refresh_features.py`. No code-hash auto-invalidation. Documented in CONTRIBUTING.md as "After touching `src/projections/features/`."

**Gitignore:** `data/features/` is added to `.gitignore` alongside the existing `data/raw/` and `data/projections/`.

### 2.3 `run_backtest` driver

```python
@dataclass(frozen=True, slots=True)
class BacktestRun:
    timestamp: datetime
    metrics: pd.DataFrame  # one row per (position, year, metric); columns: position, year, metric, value
    naive_metrics: pd.DataFrame  # same shape; informational only
    per_row_results: pd.DataFrame  # per-(position, year, week, gsis_id) actuals + preds for diagnosis

def run_backtest(
    *,
    held_out_years: Iterable[int] = (2021, 2022, 2023, 2024),
    positions: Iterable[Position] | None = None,
    train_start: int = 2018,
    features_root: Path = Path("data/features"),
    raw_root: Path = Path("data/raw"),
    ruleset: Ruleset | None = None,  # defaults to Ruleset.espn_ppr() inside the function
) -> BacktestRun:
    """Walk-forward backtest. For each (position, year), train Model A on
    cached features for [train_start, year-1], predict each week of `year`
    from cached features, score against actuals from data/raw/weekly_stats."""
```

For each (position, year):

1. Read cached training features for `[train_start, year-1]` via `cache.read_features`.
2. Read cached prediction features for `year`.
3. Read held-out actuals from `data/raw/weekly_stats/season={year}/...`.
4. Construct `BaselineModel` via `POSITION_DISPATCH[position].factory()`.
5. `model.fit(train_features, train_actuals)`.
6. `model.predict_distribution(predict_features, ruleset)` (already returns a DataFrame validated against `ProjectionWeeklySchema`).
7. Compute model metrics via `backtest.metrics.compute(...)`.
8. Compute naive baseline metrics via `backtest.naive.compute(...)`.
9. Append per-(position, year, metric) rows to the in-memory result.

Total: 16 fits per run (4 positions × 4 held-out years). At ~1–3s per RidgeCV fit + cheap I/O for cached features, runtime budget is **30–90s**. Empirically validated in Phase 6.

### 2.4 Snapshot file and tolerance config

**`tests/backtest/baseline_metrics.json`:**

```json
[
  { "position": "QB", "year": 2021, "metric": "composite_rmse", "value": 7.81 },
  { "position": "QB", "year": 2021, "metric": "composite_mae", "value": 6.28 },
  { "position": "QB", "year": 2021, "metric": "spearman_topN", "value": 0.928 },
  { "position": "QB", "year": 2021, "metric": "calibration_p10p90", "value": 0.667 },
  { "position": "QB", "year": 2021, "metric": "calibration_le_p90", "value": 0.860 },
  { "position": "QB", "year": 2021, "metric": "passing_yards_rmse", "value": 84.5 },
  { "position": "QB", "year": 2021, "metric": "passing_yards_mae", "value": 68.2 },
  { "position": "QB", "year": 2021, "metric": "passing_yards_mean_pred", "value": 199.5 },
  ...
]
```

Sorted lexicographically by `(metric, position, year)` so PR diffs stay clean and locality-friendly.

**`tests/backtest/tolerances.json`:**

```json
{
  "defaults": {
    "rmse_relative": 0.05,
    "mae_relative": 0.05,
    "spearman_absolute": 0.02,
    "calibration_absolute": 0.03,
    "mean_pred_relative": 0.10
  },
  "overrides": []
}
```

The `overrides` list is initially empty. As we discover genuinely-noisy cells (e.g., RB rushing_tds RMSE in 2022 may be unstable due to rare events on small sample), we add objects of shape `{ "position": "...", "year": ..., "metric": "...", "tolerance_kind": "rmse_relative", "tolerance_value": 0.10, "rationale": "..." }`. Adding a row to `overrides` requires the same PR review as updating the snapshot.

**Direction-aware comparison logic** (in `snapshot.py`):

| Metric | Direction | Default tolerance type |
|---|---|---|
| `*_rmse`, `*_mae`, `composite_rmse`, `composite_mae` | regress UP (worse = larger) | relative |
| `*_mean_pred` | regress AWAY from snapshot value (either direction) | relative |
| `spearman_topN` | regress DOWN (worse = smaller) | absolute |
| `calibration_p10p90` | regress AWAY from 0.80 target (either direction) | absolute |
| `calibration_le_p90` | regress AWAY from 0.90 target (either direction) | absolute |

The metric-name → tolerance-kind mapping is suffix-based and lives in `snapshot.py`. New metrics added later have to register their tolerance kind explicitly — failing closed if a metric appears in the snapshot whose tolerance kind isn't known.

### 2.5 Gated metrics list

For each (position, year), the snapshot contains rows for:

1. **Per-target-stat fit** (per `Stat` in `target_stats` for that position): `{stat}_rmse`, `{stat}_mae`, `{stat}_mean_pred`. Catches Ridge collapsing to zero; catches per-stat regressions even when the composite is fine.
2. **Composite (PPR points)**: `composite_rmse`, `composite_mae`.
3. **Rank**: `spearman_topN` — Spearman correlation between predicted and actual *summed-mean* season totals across all players for that (position, year).
4. **Calibration (weekly)**: `calibration_p10p90`, `calibration_le_p90`.

Per position, that's `(num_target_stats × 3) + 4` metrics. Across positions:
- WR: 6 stats × 3 + 4 = 22 metrics
- QB: 6 stats × 3 + 4 = 22 metrics
- RB: 6 stats × 3 + 4 = 22 metrics
- TE: 6 stats × 3 + 4 = 22 metrics

Total: 22 × 4 positions × 4 years = **352 snapshot rows**. Tractable JSON; PR diffs stay readable because the sort order localizes related cells together.

Naive-baseline metrics are computed at the same granularity but written to `data/backtest/run_<ts>/naive_metrics.json` (gitignored), not into the gated snapshot.

---

## 3. Naive baseline (informational only)

**Definition:** for each held-out (position, year, week, gsis_id, stat), the naive prediction is the player's **trailing-4-game mean** of that stat across all games strictly prior to (year, week) in time — i.e., earlier weeks of `year` when available, falling back to weeks of prior seasons. This is allowed to use earlier weeks of the held-out year because those games have already been observed at the simulated time of prediction; no held-out-year leakage occurs.

If the player has fewer than 4 prior games available (cold start), fall back to the **per-position mean** of that stat across all (year', week', player') with `year' < year` (training window only — never the held-out year itself). The cold-start fallback uses train-window data only to avoid the degenerate "league mean of the held-out year" comparator.

**Composite (PPR points):** for each player-week, compute the naive prediction by feeding the per-stat naive predictions into the existing `score(StatLine, ruleset)` function. (Not the distribution-composing `score_distribution` — naive is point-estimate only.)

Naive metrics are computed by the harness alongside model metrics and printed in `scripts/backtest.py --report`. They are **not** written to the gated snapshot file. Their purpose is solely to give the user (and PR reviewers) a sense of how much value Model A is adding over the trailing-4 view it's already constructed from. If Model A's composite RMSE is within 5% of naive's, that's a signal to reconsider feature engineering — but it doesn't fail the gate. Plan 5/6 (Model C/D) revisits the comparison.

---

## 4. Test mechanism

### 4.1 Opt-in gate

`pytest -m backtest --run-backtest` runs the full gate. The pattern mirrors the existing `-m network --run-network` setup for opt-in `nfl_data_py` API-drift smokes (TODO #8 → closed).

`pyproject.toml` registers the marker:

```toml
[tool.pytest.ini_options]
markers = [
  "network: opt-in tests that require nfl_data_py network access",
  "backtest: opt-in tests that run the full walk-forward backtest gate",
]
```

`tests/conftest.py` (existing) registers `--run-backtest` and applies skip logic to `@pytest.mark.backtest` tests when the flag is absent. Same shape as the existing `--run-network` plumbing.

### 4.2 Default-on smoke

`tests/backtest/test_backtest_smoke.py` runs `run_backtest(held_out_years=[2024], positions=[Position.WR])` — one cell — and asserts the result has the expected schema and non-NaN metrics. Catches "I broke the harness import path" / "I broke the snapshot diff signature" without paying full runtime. Runs in default `pytest -v`. Budget: ~10s.

### 4.3 Unit tests under `tests/test_backtest/`

Use synthetic in-memory fixtures (mirrors the convention from `tests/test_features/conftest.py` / `tests/test_models/`). No real `nfl_data_py` data. Fixtures construct a small synthetic feature DataFrame conforming to `WrFeaturesSchema`, a small actuals DataFrame, and exercise:

- `metrics.compute(...)` — known per-stat RMSE / MAE / Spearman / calibration values from the fixture.
- `naive.compute(...)` — known trailing-4-game predictions.
- `snapshot.diff(...)` — known passes / fails / tolerance edge cases (regression exactly at tolerance, regression just past tolerance, regression in the wrong direction for the metric kind).
- `harness.run_backtest(...)` — end-to-end on synthetic features, asserting the output schema and that all four positions are exercised.

Total expected new tests: ~30–40. Brings repo total to ~325 (from 289 post-3b).

---

## 5. CLI runner

`scripts/backtest.py` has three modes:

```
python scripts/backtest.py --check              # default; runs harness, diffs snapshot, exits 0/1
python scripts/backtest.py --update-snapshot    # runs harness, overwrites baseline_metrics.json
python scripts/backtest.py --report             # runs harness, prints model + naive metrics side by side, no gate
```

`--check` is what the pytest test invokes internally (via `run_backtest` + `diff_snapshot`, not subprocess).
`--update-snapshot` is what the user runs after intentionally improving the model. It writes the new `baseline_metrics.json` and prints a diff summary so the user can confirm changes are in the expected direction before committing.
`--report` is for ad-hoc inspection during development; output mirrors `sanity_check_baseline.py`'s human-readable format extended across years and positions.

All three modes share the same `run_backtest` call. They differ only in what they do with the result.

---

## 6. Implementation phases

Each phase is one or more commits on `feat/plan-3c-backtest-harness`. PR opens after Phase 6.

### Phase 1 — Feature cache layer (closes TODO #4)

- Add `src/projections/features/cache.py` with `read_features(...)`.
- Add `scripts/refresh_features.py` driving builder × seasons × weeks.
- Add unit tests under `tests/test_features/test_cache.py`.
- Add `data/features/` to `.gitignore`.
- Update CONTRIBUTING.md "After touching `src/projections/features/`" note.
- Verify with manual smoke: `python scripts/refresh_features.py all --seasons 2018-2024`.

### Phase 2 — Harness skeleton

- Add `src/projections/backtest/__init__.py`, `harness.py`. `run_backtest` returns a `BacktestRun` with `metrics=pd.DataFrame()` (empty) — no real metrics yet, just driver wiring.
- Unit test: `run_backtest(held_out_years=[2024], positions=[Position.WR])` returns a `BacktestRun` with the expected attribute shape.

### Phase 3 — Metrics + naive baseline

- Add `metrics.py` (per-stat RMSE/MAE/mean_pred, composite RMSE/MAE, Spearman top-N, calibration).
- Add `naive.py` (per-player trailing-4-game stat mean, fallback to per-position mean, composite via `score(StatLine, ruleset)`).
- Wire both into `harness.run_backtest`. Now `BacktestRun.metrics` and `.naive_metrics` are populated.
- Unit tests: known-value tests for each metric on synthetic fixtures.

### Phase 4 — Snapshot read/write/diff + tolerance config

- Add `snapshot.py` (`read_snapshot`, `write_snapshot`, `diff_snapshot` returning a `GateResult`).
- Add tolerance kind registry; suffix-based metric-to-kind mapping.
- Add direction-aware comparison logic.
- Unit tests: pass / fail / wrong-direction / tolerance-boundary cases. Snapshot file fixture committed under `tests/test_backtest/fixtures/`.

### Phase 5 — pytest gate wiring

- Register `backtest` marker in `pyproject.toml`.
- Extend `tests/conftest.py` with `--run-backtest` flag + skip logic.
- Add `tests/backtest/test_backtest_gate.py` (opt-in) and `tests/backtest/test_backtest_smoke.py` (default-on).
- Add `scripts/backtest.py` with `--check / --update-snapshot / --report`.

### Phase 6 — First end-to-end run + commit snapshot

- Run `python scripts/refresh_features.py all --seasons 2018-2024` to populate `data/features/`.
- Run `python scripts/backtest.py --update-snapshot` to produce v1 `baseline_metrics.json`.
- Inspect the snapshot for sanity (per-position composite RMSE in the 5–8 range; Spearman 0.92–0.98).
- Commit the snapshot.
- Run `pytest -m backtest --run-backtest` — should pass (snapshot matches itself).
- Verify the gate fails when expected: temporarily perturb the snapshot (e.g., set a Spearman value to 0.99), rerun, confirm failure with a clear diff message; revert.
- Update `project_management.md` with the per-(position, year) metric table mirroring the 3a/3b sanity-check sections.
- Close TODO #4 in `TODO.md`.

### Phase 7 — Quality gate

Run the standard end-of-effort checks per `CLAUDE.md` § "Forced verification — end-of-effort checklist":
- `pytest -v` (default) — ~325 tests pass; backtest gate skipped.
- `pytest -m backtest --run-backtest` — gate passes.
- `mypy src tests` — zero violations.
- `ruff check src tests` — zero violations.
- `ruff format --check src tests` — no drift.
- `pytest -m network --run-network` — 8 smokes pass (verifying the existing opt-in path didn't regress).

---

## 7. Risks and mitigations

### 7.1 Initial snapshot is semi-arbitrary

The first `baseline_metrics.json` captures whatever the current Model A produces. If Model A has a latent bug that's surfaced only by walk-forward (e.g., pre-2024 calibration is much worse than 2024's), we'd be locking in that bug as the gate. Mitigation: Phase 6 explicitly inspects the snapshot for sanity (composite RMSE and Spearman in the 3a/3b range); anything wildly off is investigated before commit. The numbers don't have to be *good*; they have to be *expected given current Model A*.

### 7.2 Year-to-year noise on per-stat metrics

Per-stat RMSE for rare events (e.g., RB rushing_tds in any given season) can swing by 10%+ purely from Poisson noise on small samples. The 5% relative tolerance on RMSE will trigger false positives on these metrics. Mitigation: the `tolerances.json` overrides list. We add per-row overrides for cells that demonstrably swing on noise alone; first overrides land empirically as we observe them in Phases 6 and post-merge. If the override list grows past ~10 rows, that's a signal to revisit the per-metric-type defaults.

### 7.3 Feature cache staleness

A user touches a feature builder, forgets to re-run `refresh_features.py`, runs the gate, gets a passing result that's actually based on stale features. Mitigation: documented in CONTRIBUTING.md; the `--report` mode prints the modification time of the feature cache and the git head of the feature builder files (purely informational, not enforced). Auto-invalidation via code-hash is a TODO; deferred until we see real-world bites.

### 7.4 Walk-forward fits are noisier than full-history fits

`RidgeCV` on 3 seasons of training data (the 2021 hold-out case) has a smaller CV pool than 6 seasons (the 2024 case), so per-stat RMSE on the 2021 row will tend to be noisier than on the 2024 row. The fits themselves are deterministic given the same training data — `BaselineModel.fit` calls `RidgeCV(alphas=alphas)` with the default `cv=None` (efficient leave-one-out, no shuffling) and no other randomness source — so re-running the harness on unchanged data gives bit-identical metrics. The noise concern is therefore *across years within a single run*, not *across runs*. Mitigation lives in the per-row tolerance overrides: if a 2021 cell is empirically noisier than the default tolerance accommodates, we add an entry to `tolerances.json`'s `overrides` list with a documented rationale.

### 7.5 Runtime growing past acceptable

If Phase 6 reveals the full backtest takes >5 minutes, we have options: (a) parallelize the 16 fits across positions × years (joblib's `Parallel(n_jobs=...)`), (b) cache trained models on disk keyed by `(position, train_window_hash)` and reuse across runs, (c) reduce the held-out year set to 3 years. We'd take (a) first because it's the cheapest and the fits are embarrassingly parallel. Any reduction in the gate's coverage requires explicit user sign-off.

---

## 8. Out-of-scope items folded into TODO.md

Plan 3c does not modify the existing TODOs except to close #4 in Phase 6. New TODOs introduced by this design:

- **TODO #(next): walk-forward gate non-determinism check.** Phase 6 may surface tiny RMSE jitter from RidgeCV CV-fold randomness on 3-season train windows. If observed, add explicit `random_state` propagation throughout `BaselineModel.fit` and re-snapshot. Otherwise close.
- **TODO #(next+1): naive-baseline parquet output for trend tracking.** v1 writes naive metrics to a per-run JSON. If we ever want to track "how much value is Model A adding over naive *over time*", we'd persist naive metrics to a parquet table at `data/backtest/naive_history/...`. Not load-bearing for v1.
- **TODO #(next+2): feature cache code-hash auto-invalidation.** v1 is manual. Auto-invalidation reads the source files for the feature builder and refuses to read stale cache. Deferred until manual invalidation produces a real-world bug.

These are recorded by the implementation plan, not edited into `TODO.md` directly by this design.

---

## 9. Decision summary (from brainstorming Q&A)

| Q | Decision | Rationale |
|---|---|---|
| Q1 | Plan 3c = harness + gate; Plan 3d = real season-distribution aggregation | Ship gating quickly with predictable scope; don't bundle three TODO resolutions (#13, #14, score_distribution perf) with infrastructure work. |
| Q2 | Multi-year walk-forward, 4 held-out years, expanding train window, with feature parquet caching to make runtime tractable | Single-year is barely more than the existing sanity-check script; multi-year is what catches real model regressions. Feature caching is the runtime knob that makes this affordable. |
| Q3 | Snapshot-diff gating with per-metric-type tolerance | Directly answers "did this PR make the model worse?" Naive-relative ("within 2× naive") is too loose; absolute floors age badly. |
| Q4 | Snapshot at (position, year, metric) granularity; per-metric-type tolerances in `tolerances.json` | Per-year visibility is the whole point of multi-year backtest. 352 rows of JSON is small; tolerances grouped by metric type keeps maintenance low. |
| Q5 | Held-out years 2021, 2022, 2023, 2024 | Each held-out year has at least 3 prior seasons of training data. Skips 2019 (only 1 prior season) and 2020 (COVID-shortened structural outlier). |
