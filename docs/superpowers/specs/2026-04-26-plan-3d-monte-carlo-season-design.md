# Plan 3d — Real Monte Carlo season-distribution aggregation — Design

**Status:** approved (brainstorming)
**Date:** 2026-04-26
**Author:** alden + claude
**Builds on:** `2026-04-26-plan-3c-backtest-harness-design.md` (the walk-forward gate this plan extends with season-total calibration). Closes TODO #13 (per-row seed derivation in `BaselineModel.predict_distribution`), TODO #14 (`SAMPLED_SUMMARY` family decision), and TODO #19 (gate non-determinism check, by demonstration).

**Plan-3 series context:**

- **Plan 3a (merged at `598ab9c`):** Model A on WR only. Pinned the per-week interface, joblib persistence, first real-data ingest.
- **Plan 3b (merged at `c4a0401`):** Generalized Model A to QB / RB / TE. Four trained artifacts; per-position 2024 sanity checks; `POSITION_DISPATCH` registry.
- **Plan 3c (merged at `3db71a6`):** Walk-forward backtest harness + snapshot-diff CI gate. Used summed weekly means as season totals (degenerate aggregation).
- **Plan 3d (this design):** Real Monte Carlo season-distribution aggregation. Per-row deterministic seeds. Per-stat distribution params persisted in `ProjectionWeeklySchema.params`. Season-total calibration added to the gate.
- **Plan 3e (next, not in this design):** Calibration tightening — MLE-fit gamma α and/or per-stat residual-variance buckets. Plan 3e ratchets the snapshot tighter once weekly + season calibration coverage moves toward 0.80.

---

## 1. Overview

Plan 3c locked Model A as a regression gate but deliberately deferred two questions that converge here:

- **Per-row sample seeds.** `BaselineModel.predict_distribution` currently calls `score_distribution(..., seed=42)` for every row. The per-row Monte Carlo sample arrays are therefore correlated across rows. Per-row reductions (mean, quantiles) are unaffected, but any consumer that *combines* multiple rows' samples — DFS lineup variance, season-total simulation, joint correlations once TODO #1 lands — gets a wrong answer. Real Monte Carlo season aggregation is exactly that kind of consumer; it forces this fix.
- **`SAMPLED_SUMMARY` vs `SAMPLED`.** The persisted `params` blob currently encodes only `{samples_summary: {n, mean}}` while `family="SAMPLED"` falsely implies the full samples are there. The persisted distributional info actually lives in the `mean` / `p10` / `p50` / `p90` columns, with `params` as a breadcrumb. Plan 3d makes `params` truthful by encoding the per-stat distribution parameters needed to deterministically regenerate samples — a few hundred bytes per row, three orders of magnitude smaller than persisting full sample arrays, and decomposed in a way that's useful for downstream stat-level analysis (DFS exposures, "what's the distribution of receiving yards?").

With those two fixes, real season aggregation falls out: for each `(gsis_id, season)` group, regenerate per-week samples from the per-stat params + the per-row seed, sum the sample arrays positionally across weeks, summarize. Independence across weeks holds because per-row seeds are different (sha256 of `(gsis_id, season, week, ruleset.name)`); reproducibility holds because the seed is deterministic. The new season-total calibration metric (`season_calibration_p10p90`, `season_calibration_le_p90`) is the only new gated signal — it's the only metric real Monte Carlo aggregation enables that summed weekly means cannot derive.

Calibration tightening (currently 0.67–0.80 vs. 0.80 target across the 16 weekly cells of Plan 3c's snapshot) is **out of scope** for Plan 3d — that's a model-quality improvement (MoM gamma α → MLE, or per-stat variance bucketing) with its own validation surface. It moves to Plan 3e once the aggregation infrastructure is settled. Plan 3d's snapshot reflects current under-dispersed calibration as the regression floor.

### 1.1 Goals

- Add `DistributionFamily.SAMPLED_SUMMARY` enum value to `schemas.py`. The existing `SAMPLED` value stays — it remains valid for callers that persist real sample arrays (e.g., quantile regression in Plan 5).
- Add `pack_per_stat_params(per_stat_dists) -> bytes` and `unpack_per_stat_params(blob) -> dict[Stat, Distribution]` codec helpers to `distributions.py`. Symmetric encode/decode at the `Distribution` Protocol boundary.
- Add `derive_row_seed(gsis_id, season, week, ruleset_name) -> int` to `scoring/score_distribution.py`. Stable 32-bit seed via sha256; deterministic across processes; independent across rows.
- Rewire `BaselineModel.predict_distribution` to use per-row seeds, persist per-stat params via `pack_per_stat_params`, and write `family=SAMPLED_SUMMARY`. Delete the two `# v1 limitation` docstring blocks tracking TODO #13 and #14.
- New package `src/projections/aggregation/` with `season.py` exporting `aggregate_to_season(weekly, *, ruleset, n_samples=10_000) -> pd.DataFrame`.
- New `ProjectionSeasonSchema` in `schemas.py`. One row per `(gsis_id, season)`, columns: `gsis_id`, `season`, `position`, `ruleset`, `n_weeks`, `season_mean`, `season_p10`, `season_p50`, `season_p90`, `model_id`, `generated_at`.
- New `compute_season_calibration_metrics` in `backtest/metrics.py` returning `season_calibration_p10p90` and `season_calibration_le_p90`.
- Wire the season aggregator into `backtest/harness.py`. Each (position, year) cell now contributes 2 new rows to `metrics_df` (one per season-calibration metric).
- Extend `BacktestRun` with `per_player_results: pd.DataFrame` for per-`(gsis_id, season)` season-eval diagnostic output. `scripts/backtest.py` writes it to `data/backtest/run_<ts>/season_results.parquet` alongside the existing `results.parquet`.
- Phase 6 retrains all four BaselineModel artifacts (because `score_distribution.py` and `baseline.py` are both in `code_hash_files`, so `model_id` rotates), regenerates `tests/backtest/baseline_metrics.json` (32 new rows + bounded drift on existing rows), runs the gate, and updates `project_management.md` mirroring the 3c reporting style.
- TODO #19 closes by demonstration: a re-run of `--check` with no code changes produces bit-identical metrics because seeds are now deterministic.

### 1.2 Non-goals (deferred)

- **Calibration tightening (MLE gamma α / per-stat variance bucketing)** → Plan 3e. Plan 3d's gate snapshot reflects the current under-dispersed Plan 3c calibration as-is.
- **Persisting full sample arrays in `params`** → not planned. Per-stat params + deterministic seeds give us identical sample regeneration at three orders of magnitude less storage. Revisit only if a future consumer genuinely needs persisted samples.
- **Availability / injury modeling.** The aggregator sums whatever weeks are present; it does not model "expected number of games played." A player with 12 prediction rows produces a 12-week season distribution, not a 17-week one. `n_weeks` is reported on every row so consumers can filter.
- **Bye-week imputation.** Feature builders already filter out bye-week rows upstream, so the aggregator never sees them and doesn't need to model them.
- **Season-total naive baseline.** Naive uses point predictions (no quantiles), so its season-total calibration is 1.0 by definition. Not informative; not computed.
- **Catastrophic absolute floors** for season-calibration metrics → same posture as 3c. Defer until snapshot gating produces a real-world false negative.
- **K / DST positions** → TODO #10.
- **TODO #1 (joint correlations across players)** → DFS Engine. Plan 3d models per-player season aggregation under within-player cross-week independence; cross-player correlation is a separate design surface.
- **Snapshot tolerance overrides for season metrics** → not at v1. Inherit defaults; tighten empirically if Phase 6 re-run drift exceeds defaults.

---

## 2. Architecture

### 2.1 New packages and files

```
src/projections/
├── aggregation/                  # NEW PACKAGE
│   ├── __init__.py               # exports: aggregate_to_season
│   └── season.py                 # season aggregator (pure function over ProjectionWeeklySchema)
├── distributions.py              # extended: pack_per_stat_params / unpack_per_stat_params codec
├── scoring/
│   └── score_distribution.py     # extended: derive_row_seed
├── models/
│   └── baseline.py               # rewired: per-row seed + per-stat params blob + SAMPLED_SUMMARY family
├── backtest/
│   ├── harness.py                # extended: aggregate_to_season + season metrics per cell
│   └── metrics.py                # extended: compute_season_calibration_metrics
└── schemas.py                    # extended: DistributionFamily.SAMPLED_SUMMARY + ProjectionSeasonSchema

tests/
├── test_aggregation/             # NEW DIRECTORY
│   └── test_season.py            # season aggregator tests
├── test_distributions/
│   └── test_codec.py             # NEW: pack / unpack round-trip + error paths
├── test_scoring/
│   └── test_derive_row_seed.py   # NEW: determinism / independence / range tests
├── test_models/test_baseline.py  # extended: SAMPLED_SUMMARY family, codec round-trip, deterministic predict
├── test_backtest/test_metrics.py # extended: compute_season_calibration_metrics
├── test_backtest/test_harness.py # extended: BacktestRun.metrics contains season metrics; per_player_results populated
├── test_backtest/test_snapshot.py# extended: snapshot diff covers season metrics
├── test_schemas/                 # extended: ProjectionSeasonSchema validation
└── backtest/
    ├── baseline_metrics.json     # REGENERATED: 368 → 400 rows; existing rows drift within tolerance
    └── test_backtest_smoke.py    # extended: smoke asserts season metrics present and finite
```

No new scripts. No changes to `features/` builders, the feature-cache layer, the naive baseline, or the snapshot tolerance file (defaults inherit cleanly).

### 2.2 Data flow

```
predict_distribution (per-week)
   │  for each feature row:
   │    seed = derive_row_seed(gsis_id, season, week, ruleset.name)
   │    points = score_distribution(per_stat_dists, ruleset, n_samples=10_000, seed=seed)
   │    params_blob = pack_per_stat_params(per_stat_dists)
   │    family = SAMPLED_SUMMARY
   ▼
ProjectionWeeklySchema rows  (mean / p10 / p50 / p90 unchanged in shape)
   │
   ▼
aggregate_to_season(weekly, ruleset)
   │  validate ruleset and family on every row
   │  for each (gsis_id, season) group:
   │    for each week-row r:
   │      per_stat_dists = unpack_per_stat_params(r.params)
   │      seed = derive_row_seed(r.gsis_id, r.season, r.week, ruleset.name)
   │      week_samples = score_distribution(per_stat_dists, ruleset, 10_000, seed)
   │    season_samples = elementwise_sum(week_samples_i for i in range(n_weeks))
   │    yield row with n_weeks, season_mean, season_p10, season_p50, season_p90
   ▼
ProjectionSeasonSchema rows
   │
   ▼
backtest harness (per (position, year) cell)
   │  weekly metrics (existing) + 2 season metrics from compute_season_calibration_metrics
   │  per_row_results (existing) + per_player_results (new)
   ▼
baseline_metrics.json (368 → 400 rows; gate unchanged in shape; --check tolerance logic unchanged)
```

### 2.3 Module boundaries and responsibilities

- **`distributions.py`** — owns the `Distribution` Protocol, `ParametricNormal`, `ParametricGamma`, *and* now the persistent codec for per-stat distribution params. Symmetric encode/decode lives in one file because both operations cross the same boundary (in-memory `Distribution` ↔ persisted bytes).
- **`scoring/score_distribution.py`** — owns Monte Carlo composition of per-stat distributions into a points distribution under a `Ruleset`. The seed-derivation helper `derive_row_seed` lives here because seeds are an input to `score_distribution`; both consumers (`BaselineModel.predict_distribution` and `aggregate_to_season`) already import from `scoring`.
- **`models/baseline.py`** — `predict_distribution` is the only call site that produces `ProjectionWeeklySchema` rows. The codec call, seed derivation, and `family=SAMPLED_SUMMARY` write live here; no new module needed.
- **`aggregation/season.py`** — pure function over a `ProjectionWeeklySchema` frame. No model coupling, no parquet I/O. Testable end-to-end with synthetic frames; reusable from harness, future Plan 4 CLI verbs, and DFS / draft tooling.
- **`backtest/harness.py`** — calls `aggregate_to_season`, joins to actuals, calls `compute_season_calibration_metrics`, appends rows to `metrics_df`. Extends `BacktestRun` with `per_player_results`. No conceptual change to gating logic — it's still snapshot-diff over a long-form metrics frame.
- **`backtest/metrics.py`** — gains `compute_season_calibration_metrics(season_eval_df) -> dict[str, float]`. Mirrors `compute_calibration_metrics`'s shape and tolerance posture.
- **`schemas.py`** — gains `DistributionFamily.SAMPLED_SUMMARY` and `ProjectionSeasonSchema`. Both are additive; `_DIST_FAMILY_VALUES` updates by construction.

---

## 3. Detailed design

### 3.1 `params` blob format and codec

**Encoded format** (msgpack-packed bytes):

```python
{
  "schema_version": 1,
  "stats": {
    "passing_yards":  {"family": "NORMAL", "mean": 199.5, "std": 84.5},
    "passing_tds":    {"family": "GAMMA",  "shape": 4.2,  "scale": 0.29},
    "interceptions":  {"family": "GAMMA",  "shape": 1.6,  "scale": 0.43},
    "rushing_yards":  {"family": "NORMAL", "mean": 18.2,  "std": 17.9},
    "rushing_tds":    {"family": "GAMMA",  "shape": 0.8,  "scale": 0.24},
    "fumbles_lost":   {"family": "GAMMA",  "shape": 0.5,  "scale": 0.41},
  }
}
```

Each stat entry carries enough to fully reconstruct the per-row `Distribution`:
- `NORMAL` → `ParametricNormal(mean=…, std=…)`
- `GAMMA` → `ParametricGamma(shape=…, scale=…)` where `scale = mu_i / shape`

The Gamma `scale` already absorbs the per-row predicted mean (`mu_i`), so no separate `mu` field is needed. Per-row sample seed is *not* persisted — it is recomputed deterministically from the row's `(gsis_id, season, week, ruleset)` at consumption time.

**Codec helpers (new in `distributions.py`):**

```python
def pack_per_stat_params(per_stat_dists: Mapping[Stat, Distribution]) -> bytes:
    """Encode a per-row per-stat distribution dict for ProjectionWeeklySchema.params.

    The codec dispatches on Distribution type:
      - ParametricNormal -> {"family": "NORMAL", "mean": d.mean(), "std": d.std()}
      - ParametricGamma  -> {"family": "GAMMA",  "shape": d.shape, "scale": d.scale}

    Raises:
        ValueError: a Distribution type without a registered codec entry.
    """

def unpack_per_stat_params(blob: bytes) -> dict[Stat, Distribution]:
    """Decode the params blob into a {Stat -> Distribution} dict.

    Raises:
        ValueError: unknown schema_version, unknown family, or unknown stat name.
    """
```

Both helpers fail-fast with explicit error messages. `unpack` rejects unknown `schema_version` by name (today only `1` is valid); rejects unknown `family` by name (today only `NORMAL`/`GAMMA` are wired); rejects unknown `stat` values that aren't members of `Stat`.

**Why the codec lives in `distributions.py`:** symmetric encode/decode at the same module boundary minimizes the chance of one side drifting (e.g., a future `EMPIRICAL_QUANTILE` family wire-up would touch one file, not two).

**No backward-compatibility migration.** Current `predict_distribution` callers write to in-memory frames or `tmp/` parquet that gets deleted on test teardown; no persisted production projections exist. `aggregate_to_season` rejects `family != SAMPLED_SUMMARY` with an actionable error message; this is the only "migration" surface and it's a fail-fast.

### 3.2 Per-row seed derivation

**New function in `scoring/score_distribution.py`:**

```python
import hashlib

def derive_row_seed(gsis_id: str, season: int, week: int, ruleset_name: str) -> int:
    """Stable 32-bit seed from (gsis_id, season, week, ruleset.name).

    Used by BaselineModel.predict_distribution and aggregate_to_season to keep
    per-row Monte Carlo draws independent and reproducible.

    Properties:
      - Deterministic across processes. Python's built-in hash() is salt-randomized
        (PYTHONHASHSEED), so we use sha256 instead and truncate to 32 bits.
      - Independent: changes to any of the 4 inputs change the seed.
      - Reproducible: identical inputs always produce identical samples downstream.
    """
    h = hashlib.sha256(f"{gsis_id}|{season}|{week}|{ruleset_name}".encode()).digest()
    return int.from_bytes(h[:4], "big")  # 32-bit unsigned; fits np.random.default_rng
```

Re-exported from `scoring/__init__.py` so consumers can write `from projections.scoring import derive_row_seed`.

**Why `|` as separator and not e.g. `:`:** GSIS ids match `\d{2}-\d{7}` (per `GSIS_ID_PATTERN` in `schemas.py`), ruleset names are uppercase letters and underscores (e.g. `ESPN_PPR`, `ESPN_HALF`, `STANDARD`), season/week are integers — none contain `|`. The separator unambiguously delineates fields and prevents `("00-1234567", "1") == ("00-12345671", "")` style collisions.

**Why 32-bit truncation:** `np.random.default_rng(seed)` accepts any non-negative int up to `2**32 − 1` cleanly across NumPy versions. 32 bits give `~4.3 × 10⁹` distinct seeds — vastly more than the ~17 weeks × ~30 fantasy-relevant players per position × ~10 seasons of bounded scope, so birthday-collision probability is negligible (~10⁻⁵ over the largest plausible set).

**Why sha256:** stdlib, no extra dep, documented stable across Python versions and platforms. The first 4 bytes of sha256 are uniformly distributed by construction.

### 3.3 `BaselineModel.predict_distribution` rewire

The existing per-row loop in `models/baseline.py` is modified at three points:

```python
# inside the per-row loop, BEFORE:
points = score_distribution(stat_dists, ruleset, n_samples=10_000, seed=42)
family_blob = msgpack.packb(
    {"samples_summary": {"n": len(points.samples), "mean": float(points.mean())}},
    use_bin_type=True,
)
rows.append({..., "family": DistributionFamily.SAMPLED.value, "params": family_blob, ...})

# inside the per-row loop, AFTER:
seed = derive_row_seed(
    gsis_id=feat_row["gsis_id"],
    season=int(feat_row["season"]),
    week=int(feat_row["week"]),
    ruleset_name=ruleset.name,
)
points = score_distribution(stat_dists, ruleset, n_samples=10_000, seed=seed)
family_blob = pack_per_stat_params(stat_dists)
rows.append({..., "family": DistributionFamily.SAMPLED_SUMMARY.value, "params": family_blob, ...})
```

The two `# v1 limitation` blocks at the top of `predict_distribution` (cross-row sample correlation; params summary-only) are deleted — both are closed.

The `import msgpack` line in `baseline.py` is removed; msgpack is now used only inside `distributions.py`.

### 3.4 `aggregate_to_season`

**Signature:**

```python
def aggregate_to_season(
    weekly: pd.DataFrame,
    *,
    ruleset: Ruleset,
    n_samples: int = 10_000,
) -> pd.DataFrame:
    """Aggregate weekly per-player projections into season-total distributions.

    The input is validated against ProjectionWeeklySchema. Every row must have
    family == DistributionFamily.SAMPLED_SUMMARY and ruleset == ruleset.name —
    a mixed-ruleset frame or a row written before Plan 3d's codec swap raises
    ValueError immediately.

    For each (gsis_id, season) group, the function:
      - Decodes each week's per-stat distribution params via unpack_per_stat_params.
      - Re-derives each week's per-row seed via derive_row_seed.
      - Calls score_distribution(...) to regenerate that week's points samples.
      - Sums the per-week sample arrays positionally → n_samples season-total samples.
      - Summarizes mean / p10 / p50 / p90 from the season samples.

    Returns:
        pd.DataFrame validated against ProjectionSeasonSchema.

    Raises:
        ValueError: input has rows with family != SAMPLED_SUMMARY, or rows with
            ruleset string != ruleset.name, or empty input is fine and returns
            an empty validated frame.

    Performance:
        16 (position, year) cells × ~40-100 players × ~17 weeks × 10K samples ≈
        ~10^8 float64 operations per harness run, a few seconds. Not the bottleneck.
    """
```

**Implementation outline:**

```python
def aggregate_to_season(weekly, *, ruleset, n_samples=10_000):
    weekly = ProjectionWeeklySchema.validate(weekly)
    if weekly.empty:
        # Return empty validated frame with the right columns.
        empty_cols = list(ProjectionSeasonSchema.to_schema().columns.keys())
        return ProjectionSeasonSchema.validate(pd.DataFrame(columns=empty_cols))

    # Fail-fast on family mismatch.
    bad_family = weekly[weekly["family"] != DistributionFamily.SAMPLED_SUMMARY.value]
    if not bad_family.empty:
        raise ValueError(
            f"aggregate_to_season requires family={SAMPLED_SUMMARY}, "
            f"found {bad_family['family'].unique().tolist()}"
        )

    # Fail-fast on ruleset mismatch.
    bad_ruleset = weekly[weekly["ruleset"] != ruleset.name]
    if not bad_ruleset.empty:
        raise ValueError(
            f"Mixed-ruleset input: expected {ruleset.name}, "
            f"found {bad_ruleset['ruleset'].unique().tolist()}"
        )

    rows = []
    generated_at = datetime.now(UTC)
    for (gsis_id, season), group in weekly.groupby(["gsis_id", "season"]):
        # Sum per-week sample arrays positionally.
        season_samples = np.zeros(n_samples, dtype=np.float64)
        for _idx, week_row in group.iterrows():
            per_stat_dists = unpack_per_stat_params(week_row["params"])
            seed = derive_row_seed(
                gsis_id=str(gsis_id),
                season=int(season),
                week=int(week_row["week"]),
                ruleset_name=ruleset.name,
            )
            week_dist = score_distribution(per_stat_dists, ruleset, n_samples=n_samples, seed=seed)
            season_samples += week_dist.samples

        # Modal position handles the rare in-season position change.
        position = group["position"].mode().iloc[0]

        rows.append({
            "gsis_id": gsis_id,
            "season": int(season),
            "position": position,
            "ruleset": ruleset.name,
            "n_weeks": len(group),
            "season_mean": float(season_samples.mean()),
            "season_p10": float(np.quantile(season_samples, 0.1)),
            "season_p50": float(np.quantile(season_samples, 0.5)),
            "season_p90": float(np.quantile(season_samples, 0.9)),
            "model_id": group["model_id"].iloc[0],
            "generated_at": pd.Timestamp(generated_at).as_unit("us"),
        })

    out = pd.DataFrame(rows)
    # Coerce string columns per pyarrow convention.
    for col in ("gsis_id", "position", "ruleset", "model_id"):
        out[col] = out[col].astype(_PYARROW_STR)
    return ProjectionSeasonSchema.validate(out)
```

**Modal-position resolution.** A player whose position appears differently across weeks (Taysom-Hill-style multi-position) gets the modal value. Tie-breaking is `pd.Series.mode().iloc[0]`'s default (sorted-first), which for our `Position` string values is alphabetic. This is a v1 simplification documented in the function docstring; the rare tie case (player with equal counts of two positions across weeks) is not load-bearing for v1 metrics.

**Why regenerate, not persist samples.** Section 1's storage analysis: ~80 KB/row × ~10K rows/season ≈ 800 MB/year/ruleset persisted samples vs. ~300 bytes/row × same ≈ 3 MB persisted per-stat params. Regeneration cost is O(seconds) per harness run. The tradeoff is unambiguous; documented inline.

### 3.5 `ProjectionSeasonSchema`

Added to `schemas.py` next to `ProjectionWeeklySchema`:

```python
class ProjectionSeasonSchema(pa.DataFrameModel):
    """Published per-season projection (consumer-facing contract for season totals)."""

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=1999, le=2100)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    ruleset: Series[str]
    n_weeks: Series[int] = pa.Field(ge=1, le=22)
    season_mean: Series[float]
    season_p10: Series[float]
    season_p50: Series[float]
    season_p90: Series[float]
    model_id: Series[str]
    generated_at: Series[pd.DatetimeTZDtype] = pa.Field(dtype_kwargs={"tz": "UTC", "unit": "us"})

    class Config:
        strict = "filter"
        coerce = True  # mirror ProjectionWeeklySchema's empty-output fast path
```

No `team` / `opponent` columns: both vary across weeks (mid-season trades; opponent changes weekly), and no single value is meaningful at season scope. Downstream consumers that need per-week team/opponent should query the weekly schema directly.

### 3.6 Gate integration in `backtest/harness.py`

Inside the existing `(position, year)` loop, after `_model_metrics_for_cell` returns:

```python
# 1. Build season-total predictions.
season_predictions = aggregate_to_season(predictions, ruleset=ruleset)

# 2. Build season-total actuals from holdout.
holdout_pos["actual_ppr"] = _realized_ppr_points(holdout_pos, ruleset)
season_actuals = (
    holdout_pos.groupby("gsis_id", as_index=False)["actual_ppr"]
    .sum()
    .rename(columns={"actual_ppr": "actual_season_total"})
)

# 3. Inner-join to build season eval frame.
season_eval_df = season_predictions.merge(
    season_actuals, on="gsis_id", how="inner"
)

# 4. Compute and append season metrics.
season_metrics = compute_season_calibration_metrics(season_eval_df)
for metric_name, value in season_metrics.items():
    metrics_rows.append({
        "position": position.value,
        "year": year,
        "metric": metric_name,
        "value": float(value),
    })

# 5. Append per-player results for diagnostic output.
season_eval_df = season_eval_df.assign(position=position.value)
per_player_frames.append(season_eval_df)
```

`per_player_frames` is concatenated at the end of `run_backtest` into `BacktestRun.per_player_results`. The new field is added to the dataclass:

```python
@dataclass(frozen=True, slots=True)
class BacktestRun:
    timestamp: pd.Timestamp
    metrics: pd.DataFrame
    naive_metrics: pd.DataFrame
    per_row_results: pd.DataFrame
    per_player_results: pd.DataFrame  # NEW: per (position, year, gsis_id) season eval rows
```

`scripts/backtest.py` writes `per_player_results` to `data/backtest/run_<ts>/season_results.parquet`. The existing `data/backtest/` `.gitignore` rule covers it — no `.gitignore` change needed.

### 3.7 New metric: `compute_season_calibration_metrics`

In `backtest/metrics.py`:

```python
def compute_season_calibration_metrics(season_eval_df: pd.DataFrame) -> dict[str, float]:
    """Calibration coverage on season-total predictions.

    Expects columns season_p10, season_p90, actual_season_total. Returns
    season_calibration_p10p90 (fraction of (gsis_id, year) where actual is in
    [season_p10, season_p90]) and season_calibration_le_p90 (fraction where
    actual <= season_p90).

    Mirrors compute_calibration_metrics's shape; not gated for naive baseline
    (point predictor's quantiles are all equal to the point, calibration is 1.0
    by definition, not informative).

    Returns NaN for both metrics on an empty frame, matching the convention
    of compute_spearman_topN.
    """
    if season_eval_df.empty:
        return {"season_calibration_p10p90": float("nan"), "season_calibration_le_p90": float("nan")}
    a = season_eval_df["actual_season_total"]
    return {
        "season_calibration_p10p90": float(
            ((a >= season_eval_df["season_p10"]) & (a <= season_eval_df["season_p90"])).mean()
        ),
        "season_calibration_le_p90": float((a <= season_eval_df["season_p90"]).mean()),
    }
```

### 3.8 Tolerances

`tests/backtest/tolerances.json` is unchanged. The two new metric names match the existing `calibration_*` regex pattern in the snapshot diff logic — `season_calibration_p10p90` and `season_calibration_le_p90` both inherit `calibration_absolute=0.03` via the existing rule that maps any metric prefix `calibration_` or suffix `_calibration_*` to the calibration-absolute tolerance.

If Phase 6 surfaces season-calibration noise greater than 0.03 on a re-run (it should not — fully deterministic seeds make re-runs bit-identical), per-row overrides are added empirically. Plan 3c's `_METRIC_KIND_RULES` handles new metric names by prefix matching; verify in Phase 1 that the rule list correctly classifies `season_calibration_*` as calibration-kind. (If not, extend `_METRIC_KIND_RULES` in `backtest/snapshot.py` — one new line.)

---

## 4. Phasing

Seven phases (5 split into 5a + 5b to honor the ≤5-files-per-phase budget). Verification (`pytest -v`, `mypy src tests`, `ruff check src tests`, `ruff format --check src tests`) runs at each phase boundary; the PR summary captures all outputs.

### Phase 1 — Schemas and codec

**Files (4):** `src/projections/schemas.py`, `src/projections/distributions.py`, `tests/test_schemas/test_projection_season_schema.py` (new), `tests/test_distributions/test_codec.py` (new).

- Add `DistributionFamily.SAMPLED_SUMMARY` (one line, additive).
- Add `ProjectionSeasonSchema` next to `ProjectionWeeklySchema`.
- Add `pack_per_stat_params` and `unpack_per_stat_params` to `distributions.py`.
- Tests: codec round-trips for NORMAL+GAMMA, fail-fast on unknown `schema_version` / `family` / `Stat`. ProjectionSeasonSchema accepts canonical rows and rejects (gsis pattern violation, n_weeks=0, position not in `_POSITION_VALUES`, missing `model_id`).

### Phase 2 — `derive_row_seed`

**Files (2):** `src/projections/scoring/score_distribution.py` (extend), `src/projections/scoring/__init__.py` (re-export), `tests/test_scoring/test_derive_row_seed.py` (new).

- Add `derive_row_seed` with sha256 implementation.
- Re-export from `scoring/__init__.py`.
- Tests: determinism (two calls → same seed); independence (each of 4 inputs changes the seed); range (output in `[0, 2**32)`); cross-process stability (subprocess test confirms `PYTHONHASHSEED` env var has no effect on output).

### Phase 3 — `predict_distribution` rewire

**Files (2):** `src/projections/models/baseline.py` (rewire), `tests/test_models/test_baseline.py` (extend).

- Inside the per-row loop: replace `seed=42` with `derive_row_seed(...)`; replace summary blob with `pack_per_stat_params(stat_dists)`; switch family literal to `DistributionFamily.SAMPLED_SUMMARY`.
- Delete the two `# v1 limitation` docstring blocks (lines 446–459).
- Remove the `import msgpack` line (now unused in this module).
- Tests: predicted rows have `family == "SAMPLED_SUMMARY"`; `unpack_per_stat_params(row["params"])` returns the same per-stat distributions as `model.build_stat_distributions(...)`; two rows with identical features but different `(gsis_id, week)` produce different sample arrays; running `predict_distribution` twice with the same inputs produces bit-identical `mean` / `p10` / `p50` / `p90`.

### Phase 4 — `aggregate_to_season`

**Files (4):** `src/projections/aggregation/__init__.py` (new), `src/projections/aggregation/season.py` (new), `tests/test_aggregation/__init__.py` (new), `tests/test_aggregation/test_season.py` (new).

- New package; `aggregate_to_season` exported from `__init__.py`.
- Tests: empty input → empty validated frame; single-player single-week → `n_weeks=1`, `season_mean ≈ weekly_mean` (within 1% MC noise); single-player multi-week → quantile widening (variance increases as more weeks added); multi-player → one row per `(gsis_id, season)`; traded player gets modal position; mixed-ruleset frame raises `ValueError`; non-`SAMPLED_SUMMARY` family raises `ValueError`. Use small synthetic frames built via the codec to keep tests hermetic.

### Phase 5a — Harness wiring + season-calibration metric

**Files (5):** `src/projections/backtest/__init__.py` (re-export `compute_season_calibration_metrics` if exposed; otherwise no change), `src/projections/backtest/harness.py` (extend), `src/projections/backtest/metrics.py` (extend), `src/projections/backtest/snapshot.py` (verify metric-kind classifier; extend if needed), `scripts/backtest.py` (write `season_results.parquet`).

- Implement `compute_season_calibration_metrics` in `backtest/metrics.py`.
- Wire `aggregate_to_season` + `season_actuals` join + `compute_season_calibration_metrics` into `run_backtest`.
- Extend `BacktestRun` with `per_player_results`.
- `scripts/backtest.py` `--check` and `--update-snapshot` write `season_results.parquet` next to `results.parquet`.
- Verify `_METRIC_KIND_RULES` (or equivalent classifier) in `snapshot.py` correctly maps `season_calibration_*` to calibration-absolute tolerance; extend if needed.

### Phase 5b — Tests for harness + metrics

**Files (4):** `tests/test_backtest/test_metrics.py` (extend), `tests/test_backtest/test_harness.py` (extend), `tests/test_backtest/test_snapshot.py` (extend), `tests/backtest/test_backtest_smoke.py` (extend).

- `compute_season_calibration_metrics` on a known frame: 5/10 inside `[p10, p90]` → 0.5; empty frame → NaN.
- `BacktestRun.metrics` contains `season_calibration_p10p90`+`season_calibration_le_p90` for every cell.
- `BacktestRun.per_player_results` is non-empty and validates against `ProjectionSeasonSchema`-compatible columns + `actual_season_total` + `position`.
- Snapshot diff applies the calibration-absolute tolerance to season metrics.
- Default-on (WR, 2024) smoke asserts both season metrics are present and finite.

### Phase 6 — Retrain, re-snapshot, gate run

**Files: artifacts under `models/artifacts/`, `tests/backtest/baseline_metrics.json`, `project_management.md`, `TODO.md`.**

1. Retrain all four BaselineModel artifacts via `python scripts/train_baseline.py wr|qb|rb|te`. New `model_id`s; old artifacts deleted.
2. Run `python scripts/backtest.py --update-snapshot` to regenerate `tests/backtest/baseline_metrics.json` (32 new rows + bounded drift on existing rows).
3. Sanity-check: every existing weekly metric is within tolerance against the *old* snapshot. (We expect no false-positive gate failures; if any cell drifts outside tolerance, investigate before committing.)
4. Run `pytest -m backtest --run-backtest`; expect 0 failures.
5. **TODO #19 closure check:** re-run `--check` after `--update-snapshot` with no code changes; expect bit-identical metrics. Document this in the PR summary.
6. Update `project_management.md`'s "Plan 3d" section mirroring the 3c reporting style: composite RMSE / MAE / spearman / weekly calibration / season calibration tables, naive comparison still informational, decision log entries for: `params` codec format choice; per-row seed derivation; aggregator's modal-position resolution; calibration tightening explicitly deferred to Plan 3e.
7. Close TODO #13, #14, #19 in `TODO.md`. Add Plan 3e as the recommended next action.

---

## 5. Testing strategy

### 5.1 New / extended test files

| File | Coverage |
|---|---|
| `tests/test_distributions/test_codec.py` (new) | Pack/unpack round-trip for NORMAL+GAMMA. Unknown family → ValueError. Unknown schema_version → ValueError. Every `Stat` that any BaselineModel writes round-trips faithfully. |
| `tests/test_scoring/test_derive_row_seed.py` (new) | Determinism within process. Independence across input fields. Range fits in 32-bit unsigned. Cross-process stability via subprocess with explicit `PYTHONHASHSEED`. |
| `tests/test_models/test_baseline.py` (extend) | `family == SAMPLED_SUMMARY`. `params` round-trips to per-stat dists. Different (gsis_id, week) → different sample arrays. `predict_distribution` is bit-identical on re-run. |
| `tests/test_aggregation/test_season.py` (new) | Empty → empty validated frame. Single-player single-week. Single-player multi-week (quantile widening). Multi-player. Traded player → modal position. Mixed-ruleset → ValueError. Non-SAMPLED_SUMMARY family → ValueError. `n_weeks` field correct. |
| `tests/test_schemas/test_projection_season_schema.py` (new) | Required fields. dtype contracts. `n_weeks` bounds. GSIS regex. position in `_POSITION_VALUES`. tz-aware datetime. |
| `tests/test_backtest/test_metrics.py` (extend) | `compute_season_calibration_metrics` on a known frame: 5/10 inside `[p10, p90]` → 0.5; empty frame → NaN. |
| `tests/test_backtest/test_harness.py` (extend) | Smoke run produces `BacktestRun.metrics` with both season metrics in every cell; `per_player_results` non-empty; row count matches `n_players_in_holdout × n_cells`. |
| `tests/test_backtest/test_snapshot.py` (extend) | Snapshot diff classifies season-calibration metrics under calibration-absolute tolerance. |
| `tests/backtest/test_backtest_smoke.py` (extend) | Default-on (WR, 2024) smoke now asserts both season metrics are present and finite. |

### 5.2 Determinism contract

Plan 3d makes `BaselineModel.predict_distribution` and `aggregate_to_season` fully deterministic — same input frame + same `Ruleset` → bit-identical output frame across processes and machines. This is testable end-to-end (run twice, hash-compare the output frames) and is the substantive payoff for adopting sha256-based seeds.

The harness inherits this property: a Phase-6 re-run of `--check` (no code changes) should produce metrics that hash-match the snapshot exactly. This is the closure mechanism for TODO #19.

### 5.3 Performance / runtime budget

- Per-row codec encoding: <50 µs/row for `pack_per_stat_params` (msgpack on a small dict).
- Per-row codec decoding + sample regeneration: ~3 ms/row (`unpack_per_stat_params` + `score_distribution` at 10K samples).
- Aggregation: ~17 weeks × 3 ms = ~50 ms per (gsis_id, season). Across 16 cells × ~80 players = ~64 seconds total. Bounded by `score_distribution`; not new perf surface.
- Full gate (`pytest -m backtest --run-backtest`): expected total ~3 minutes (≈133 seconds for harness + ~50 seconds for season aggregation overhead).
- Default smoke (`pytest -v`): expected ~17 seconds, +~2 seconds for aggregation in the (WR, 2024) cell.

If Phase 6 measurement shows aggregation adds >30% to gate runtime, vectorize the inner-loop sample regeneration by batching all rows of a (gsis_id, season) group through a single `score_distribution` call. Not necessary for v1 but a concrete fallback.

### 5.4 Static checks

Per CLAUDE.md "Forced verification": at the end of each phase, run all four:

- `pytest -v` (or relevant subset, named in PR summary).
- `mypy src tests` (strict; zero violations).
- `ruff check src tests` (zero violations).
- `ruff format --check src tests` (no drift).

PR summary captures the output for each, mirroring the 3c PR.

---

## 6. Risks and open questions

### 6.1 Risks

- **Snapshot drift outside tolerance.** Per-row seeds change per-row sample arrays; per-row reductions (mean/p10/p50/p90) drift slightly; aggregated metrics across thousands of rows should drift well under tolerances, but this is empirically verified in Phase 6, not asserted up-front. *Mitigation:* Phase 6 step 3 sanity-checks every existing weekly metric against the old snapshot before committing the new one. If a cell drifts outside its tolerance, halt and investigate (likely a subtle correctness issue in seed derivation or codec).
- **`_METRIC_KIND_RULES` misclassifies season metrics.** If the snapshot's metric-kind classifier doesn't recognize `season_calibration_*` as calibration-kind, the gate would apply the wrong tolerance (likely defaulting to RMSE-relative). *Mitigation:* Phase 5 explicitly verifies the classifier output for the two new metric names before Phase 6 runs.
- **Modal-position resolution edge case.** A traded player with exactly equal counts of two positions across weeks gets the alphabetically-first position. v1 not load-bearing; documented in docstring; revisited only if a real player triggers it.
- **`schema_version` field never gets exercised.** If we never bump the codec format, `schema_version=1` becomes permanent decoration. *Acceptable cost:* one int per blob, costs nothing, and saves a future migration when the format does change. Standard pattern.

### 6.2 Open questions resolved during brainstorming

- **Storage strategy for `params`** → per-stat distribution params, not full samples. Three orders of magnitude smaller; decomposable; deterministic regeneration via seed.
- **Codec module location** → `distributions.py`. Symmetric encode/decode at the same boundary as the `Distribution` Protocol it serializes.
- **Aggregation API location** → `src/projections/aggregation/season.py`, pure function. Reusable from harness, CLI, DFS engine.
- **Bye / availability modeling** → not modeled. Aggregator sums whatever weeks are present; `n_weeks` reported on every row.
- **Season metrics in the gate** → calibration only. Composite RMSE/MAE and Spearman are redundant with weekly versions by linearity of expectation.
- **Calibration tightening** → out of scope (Plan 3e). Plan 3d's snapshot reflects current under-dispersed calibration as the regression floor.

### 6.3 Open questions for Plan 3e or beyond

- MLE-fit gamma α (replace `_gamma_alpha_from_residuals`'s method-of-moments).
- Per-stat residual-variance bucketing by predicted-mean tertile (heteroscedasticity).
- Joint correlations across players (TODO #1) — DFS engine.
- Persisting season projections to parquet for downstream consumption (Plan 4 CLI verbs).
- Real player availability modeling (injury distribution priors).
