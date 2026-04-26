# Plan 3d — Monte Carlo Season Aggregation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the degenerate season-total aggregation in Plan 3c (`groupby('gsis_id').sum()` over weekly mean predictions) with real Monte Carlo aggregation. Add `season_calibration_p10p90` / `season_calibration_le_p90` to the gate. Closes TODO #13 (per-row seeds), TODO #14 (`SAMPLED_SUMMARY` family), and TODO #19 (gate non-determinism check, by demonstration).

**Architecture:** Per-row deterministic seeds derived via sha256 of `(gsis_id, season, week, ruleset.name)` make `BaselineModel.predict_distribution` reproducible across processes. The persisted `params` blob switches from a `{samples_summary}` breadcrumb to encoded per-stat distribution parameters; family becomes `SAMPLED_SUMMARY`. A new `aggregation/season.py` module is a pure function over `ProjectionWeeklySchema` rows: it decodes per-stat params, regenerates per-week sample arrays via `score_distribution`, sums positionally across weeks per `(gsis_id, season)`, and emits a `ProjectionSeasonSchema`-validated frame. The backtest harness wires this into a per-cell season eval frame and adds two calibration metrics to the gated snapshot.

**Tech Stack:** pandas, pandera, numpy, msgpack, sha256, sklearn (existing RidgeCV, untouched), pytest, mypy strict, ruff.

**Spec:** `docs/superpowers/specs/2026-04-26-plan-3d-monte-carlo-season-design.md`.

**Branch:** `feat/plan-3d-monte-carlo-season` (already created; spec already committed).

**Pre-implementation note:** Phases 1–5 do not require running the full backtest harness or retraining models — the existing code path keeps working with the old `family=SAMPLED` rows produced by current artifacts in tests that don't go through `predict_distribution`. Phase 6 retrains all four artifacts and regenerates the snapshot.

---

## File Structure

### Files created in this plan

| Path | Responsibility |
|---|---|
| `src/projections/distributions/codec.py` | `pack_per_stat_params(per_stat_dists) -> bytes` and `unpack_per_stat_params(blob) -> dict[Stat, Distribution]`. Symmetric encode/decode for `ProjectionWeeklySchema.params`. |
| `src/projections/aggregation/__init__.py` | Re-exports `aggregate_to_season`. |
| `src/projections/aggregation/season.py` | `aggregate_to_season(weekly, *, ruleset, n_samples=10_000) -> pd.DataFrame`. Pure function over `ProjectionWeeklySchema` rows. |
| `tests/test_aggregation/__init__.py` | Test package marker. |
| `tests/test_aggregation/test_season.py` | Aggregator tests: empty input; single-player single-week; single-player multi-week; multi-player; traded-player modal-position; mixed-ruleset; non-`SAMPLED_SUMMARY` family. |
| `tests/test_distributions/test_codec.py` | Codec tests: round-trip NORMAL+GAMMA; unknown family/schema_version/stat. |
| `tests/test_scoring/test_derive_row_seed.py` | Seed determinism, independence, range, cross-process stability. |
| `tests/test_schemas/test_projection_season_schema.py` | `ProjectionSeasonSchema` validation. |

### Files modified in this plan

| Path | Change |
|---|---|
| `src/projections/schemas.py` | Add `DistributionFamily.SAMPLED_SUMMARY` enum value. Add `ProjectionSeasonSchema` class. |
| `src/projections/distributions/__init__.py` | Re-export `pack_per_stat_params`, `unpack_per_stat_params`. |
| `src/projections/scoring/score_distribution.py` | Add `derive_row_seed(gsis_id, season, week, ruleset_name) -> int`. |
| `src/projections/scoring/__init__.py` | Re-export `derive_row_seed`. |
| `src/projections/models/baseline.py` | Rewire `predict_distribution`: per-row seed via `derive_row_seed`; `params` via `pack_per_stat_params`; family `SAMPLED_SUMMARY`. Delete the two `# v1 limitation` docstring blocks (lines 446–459 in current source). Remove the now-unused `import msgpack` line. |
| `src/projections/backtest/metrics.py` | Add `compute_season_calibration_metrics(season_eval_df) -> dict[str, float]`. |
| `src/projections/backtest/harness.py` | After existing per-cell weekly metrics, call `aggregate_to_season`, build season actuals via inner-join on `gsis_id`, append two season-calibration rows to `metrics_rows`. Extend `BacktestRun` with `per_player_results: pd.DataFrame`; concat per-cell season eval frames into it. |
| `src/projections/backtest/snapshot.py` | Verify `_METRIC_KIND_RULES` already classifies `season_calibration_*` as `calibration_absolute` (the existing `("calibration_",)` substring rule covers it). No change expected; documented in Phase 5a verification step. |
| `src/projections/backtest/__init__.py` | No re-export change for `compute_season_calibration_metrics` (internal to harness). |
| `scripts/backtest.py` | Write `per_player_results` to `data/backtest/run_<ts>/season_results.parquet` alongside the existing `results.parquet` (only if results dir already exists per current behavior). |
| `tests/test_models/test_baseline.py` | Add tests: `family == SAMPLED_SUMMARY`; `params` round-trips via `unpack_per_stat_params` to the same per-stat distributions; different `(gsis_id, week)` produce different sample arrays; `predict_distribution` is bit-identical on re-run. |
| `tests/test_backtest/test_metrics.py` | Add tests: `compute_season_calibration_metrics` on a known frame; empty frame returns NaN. |
| `tests/test_backtest/test_harness.py` | Add tests: `BacktestRun.metrics` includes both season metrics for every cell; `per_player_results` is non-empty and contains expected columns. |
| `tests/test_backtest/test_snapshot.py` | Add test: `_classify_metric` routes `season_calibration_p10p90` and `season_calibration_le_p90` to `calibration_absolute`. |
| `tests/backtest/test_backtest_smoke.py` | Add assertion: smoke run includes both season metrics, both finite. |
| `tests/backtest/baseline_metrics.json` | Regenerated in Phase 6 by `scripts/backtest.py --update-snapshot`. |
| `models/artifacts/baseline-{wr,qb,rb,te}-2018-2023-*.joblib` | Regenerated in Phase 6 by `scripts/train_baseline.py`. Old artifacts deleted (their `code_hash` is no longer reachable from current code). |
| `project_management.md` | New "Plan 3d" section mirroring 3c reporting style: composite metrics, naive comparison, decision log entries; closes TODO #13/#14/#19. |
| `TODO.md` | Close #13, #14, #19. Add Plan 3e as next-action recommendation. |

---

## Phase 1 — Schemas and codec

### Task 1.1: Add `DistributionFamily.SAMPLED_SUMMARY`

**Files:**
- Modify: `src/projections/schemas.py:145-152`

- [ ] **Step 1: Re-read `src/projections/schemas.py` lines 140-160.** Confirm the current `DistributionFamily` enum is exactly:

```python
class DistributionFamily(StrEnum):
    """Backing representation of a `Distribution`."""

    NORMAL = "NORMAL"
    GAMMA = "GAMMA"
    EMPIRICAL_QUANTILE = "EMPIRICAL_QUANTILE"  # quantile-regression output
    SAMPLED = "SAMPLED"  # explicit sample array
```

- [ ] **Step 2: Add `SAMPLED_SUMMARY` value.** Edit `src/projections/schemas.py` to make the enum:

```python
class DistributionFamily(StrEnum):
    """Backing representation of a `Distribution`."""

    NORMAL = "NORMAL"
    GAMMA = "GAMMA"
    EMPIRICAL_QUANTILE = "EMPIRICAL_QUANTILE"  # quantile-regression output
    SAMPLED = "SAMPLED"  # explicit sample array
    SAMPLED_SUMMARY = "SAMPLED_SUMMARY"  # per-stat dist params + summary in mean/p10/p50/p90
```

- [ ] **Step 3: Verify `_DIST_FAMILY_VALUES` updates by construction.** Re-read `src/projections/schemas.py` line 252 to confirm:

```python
_DIST_FAMILY_VALUES = [f.value for f in DistributionFamily]
```

This list comprehension picks up the new value automatically — no change needed.

- [ ] **Step 4: Run schema-related tests.**

Run: `source .venv/Scripts/activate && pytest tests/test_schemas/ -v`
Expected: PASS, including any test that enumerates `DistributionFamily` members.

### Task 1.2: Add `ProjectionSeasonSchema`

**Files:**
- Modify: `src/projections/schemas.py` (add new class after `ProjectionWeeklySchema`)
- Create: `tests/test_schemas/test_projection_season_schema.py`

- [ ] **Step 1: Write the failing test.** Create `tests/test_schemas/test_projection_season_schema.py`:

```python
"""ProjectionSeasonSchema — validation behavior."""

from __future__ import annotations

import pandas as pd
import pandera.errors
import pytest

from projections.schemas import ProjectionSeasonSchema, _PYARROW_STR


def _canonical_season_row(**overrides: object) -> dict[str, object]:
    base = {
        "gsis_id": "00-0033873",
        "season": 2024,
        "position": "WR",
        "ruleset": "ESPN_PPR",
        "n_weeks": 17,
        "season_mean": 250.0,
        "season_p10": 180.0,
        "season_p50": 248.0,
        "season_p90": 320.0,
        "model_id": "baseline:wr:abcdef12:2018-2023",
        "generated_at": pd.Timestamp("2026-04-26", tz="UTC").as_unit("us"),
    }
    base.update(overrides)
    return base


def _to_validated_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ("gsis_id", "position", "ruleset", "model_id"):
        df[col] = df[col].astype(_PYARROW_STR)
    return ProjectionSeasonSchema.validate(df)


def test_canonical_row_validates() -> None:
    out = _to_validated_frame([_canonical_season_row()])
    assert len(out) == 1
    assert out["season_mean"].iloc[0] == 250.0


def test_invalid_gsis_id_rejected() -> None:
    with pytest.raises(pandera.errors.SchemaError):
        _to_validated_frame([_canonical_season_row(gsis_id="not-a-gsis-id")])


def test_n_weeks_zero_rejected() -> None:
    with pytest.raises(pandera.errors.SchemaError):
        _to_validated_frame([_canonical_season_row(n_weeks=0)])


def test_n_weeks_above_22_rejected() -> None:
    with pytest.raises(pandera.errors.SchemaError):
        _to_validated_frame([_canonical_season_row(n_weeks=23)])


def test_position_not_in_set_rejected() -> None:
    with pytest.raises(pandera.errors.SchemaError):
        _to_validated_frame([_canonical_season_row(position="ZZ")])


def test_season_below_1999_rejected() -> None:
    with pytest.raises(pandera.errors.SchemaError):
        _to_validated_frame([_canonical_season_row(season=1998)])


def test_naive_datetime_rejected() -> None:
    naive_ts = pd.Timestamp("2026-04-26").as_unit("us")
    with pytest.raises(pandera.errors.SchemaError):
        _to_validated_frame([_canonical_season_row(generated_at=naive_ts)])
```

- [ ] **Step 2: Run the test to verify it fails.**

Run: `source .venv/Scripts/activate && pytest tests/test_schemas/test_projection_season_schema.py -v`
Expected: FAIL with `ImportError` — `ProjectionSeasonSchema` doesn't exist.

- [ ] **Step 3: Add `ProjectionSeasonSchema` to `src/projections/schemas.py`.** Insert immediately after the `ProjectionWeeklySchema` class (after the `class Config:` block ending around line 662):

```python
class ProjectionSeasonSchema(pa.DataFrameModel):
    """Published per-season projection (consumer-facing contract for season totals).

    Aggregates per-week samples across the weeks a player has predictions for in a
    season. n_weeks reports how many weeks were aggregated; consumers may filter
    by it. position is the modal value from the input rows for the gsis_id (the
    rare in-season position change inherits the most-frequent value).
    """

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
        coerce = True
```

- [ ] **Step 4: Run the test to verify it passes.**

Run: `source .venv/Scripts/activate && pytest tests/test_schemas/test_projection_season_schema.py -v`
Expected: PASS — all six tests green.

- [ ] **Step 5: Run static checks.**

Run: `source .venv/Scripts/activate && mypy src tests && ruff check src tests && ruff format --check src tests`
Expected: zero violations.

- [ ] **Step 6: Commit.**

```bash
git add src/projections/schemas.py tests/test_schemas/test_projection_season_schema.py
git commit -m "feat(schemas): add SAMPLED_SUMMARY family + ProjectionSeasonSchema"
```

### Task 1.3: Add codec module — `pack_per_stat_params`

**Files:**
- Create: `src/projections/distributions/codec.py`
- Create: `tests/test_distributions/test_codec.py`
- Modify: `src/projections/distributions/__init__.py` (re-export)

- [ ] **Step 1: Write the failing test.** Create `tests/test_distributions/test_codec.py`:

```python
"""Codec tests — pack_per_stat_params / unpack_per_stat_params round-trip."""

from __future__ import annotations

import pytest

from projections.distributions import (
    ParametricGamma,
    ParametricNormal,
    pack_per_stat_params,
)
from projections.schemas import Stat


def test_pack_normal_dist_returns_bytes() -> None:
    dists = {Stat.RECEIVING_YARDS: ParametricNormal(mean=36.3, std=18.1)}
    blob = pack_per_stat_params(dists)
    assert isinstance(blob, bytes)
    assert len(blob) > 0


def test_pack_gamma_dist_returns_bytes() -> None:
    dists = {Stat.RECEPTIONS: ParametricGamma(shape=4.2, scale=0.7)}
    blob = pack_per_stat_params(dists)
    assert isinstance(blob, bytes)
    assert len(blob) > 0


def test_pack_mixed_families_returns_bytes() -> None:
    dists = {
        Stat.RECEIVING_YARDS: ParametricNormal(mean=36.3, std=18.1),
        Stat.RECEPTIONS: ParametricGamma(shape=4.2, scale=0.7),
    }
    blob = pack_per_stat_params(dists)
    assert isinstance(blob, bytes)
    assert len(blob) > 0


def test_pack_unknown_distribution_type_raises() -> None:
    class _NotADistribution:
        def mean(self) -> float:
            return 0.0

        def std(self) -> float:
            return 1.0

    with pytest.raises(ValueError, match="codec"):
        pack_per_stat_params({Stat.RECEIVING_YARDS: _NotADistribution()})  # type: ignore[dict-item]
```

- [ ] **Step 2: Run the test to verify it fails.**

Run: `source .venv/Scripts/activate && pytest tests/test_distributions/test_codec.py -v`
Expected: FAIL with `ImportError` — `pack_per_stat_params` not exported.

- [ ] **Step 3: Create `src/projections/distributions/codec.py`.**

```python
"""Symmetric codec for per-stat distribution params persisted in
ProjectionWeeklySchema.params.

The encoded blob is msgpack-packed with shape:

    {
        "schema_version": 1,
        "stats": {
            "<stat_value>": {
                "family": "NORMAL"|"GAMMA",
                ... family-specific params ...
            },
            ...
        }
    }

Currently registered families:
    NORMAL: {"family": "NORMAL", "mean": float, "std": float}
    GAMMA:  {"family": "GAMMA",  "shape": float, "scale": float}

Adding a new family means adding one branch each to pack_per_stat_params and
unpack_per_stat_params. schema_version=1 is the only supported version today.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import msgpack

from projections.distributions.base import Distribution
from projections.distributions.parametric import ParametricGamma, ParametricNormal
from projections.schemas import DistributionFamily, Stat

_SCHEMA_VERSION: Final[int] = 1


def pack_per_stat_params(per_stat_dists: Mapping[Stat, Distribution]) -> bytes:
    """Encode a per-row per-stat distribution dict for ProjectionWeeklySchema.params.

    Raises:
        ValueError: a Distribution type without a registered codec entry.
    """
    stats_blob: dict[str, dict[str, object]] = {}
    for stat, dist in per_stat_dists.items():
        if isinstance(dist, ParametricNormal):
            stats_blob[stat.value] = {
                "family": DistributionFamily.NORMAL.value,
                "mean": dist.mean(),
                "std": dist.std(),
            }
        elif isinstance(dist, ParametricGamma):
            stats_blob[stat.value] = {
                "family": DistributionFamily.GAMMA.value,
                "shape": dist.shape,
                "scale": dist.scale,
            }
        else:
            raise ValueError(
                f"No codec entry for Distribution type {type(dist).__name__}; "
                f"add a branch to pack_per_stat_params in distributions/codec.py."
            )
    payload = {"schema_version": _SCHEMA_VERSION, "stats": stats_blob}
    return bytes(msgpack.packb(payload, use_bin_type=True))


def unpack_per_stat_params(blob: bytes) -> dict[Stat, Distribution]:
    """Decode the params blob into a {Stat -> Distribution} dict.

    Raises:
        ValueError: unknown schema_version, unknown family, or unknown stat name.
    """
    payload = msgpack.unpackb(blob, raw=False)
    version = payload.get("schema_version")
    if version != _SCHEMA_VERSION:
        raise ValueError(
            f"Unknown per-stat params schema_version: {version!r} "
            f"(supported: {_SCHEMA_VERSION})"
        )
    stats_blob = payload["stats"]
    out: dict[Stat, Distribution] = {}
    for stat_name, entry in stats_blob.items():
        try:
            stat = Stat(stat_name)
        except ValueError as exc:
            raise ValueError(
                f"Unknown stat name in params blob: {stat_name!r}"
            ) from exc
        family_value = entry["family"]
        if family_value == DistributionFamily.NORMAL.value:
            out[stat] = ParametricNormal(mean=float(entry["mean"]), std=float(entry["std"]))
        elif family_value == DistributionFamily.GAMMA.value:
            out[stat] = ParametricGamma(shape=float(entry["shape"]), scale=float(entry["scale"]))
        else:
            raise ValueError(
                f"Unknown family in params blob: {family_value!r}; "
                f"add a branch to unpack_per_stat_params in distributions/codec.py."
            )
    return out
```

- [ ] **Step 4: Re-export from `src/projections/distributions/__init__.py`.** Replace the entire file with:

```python
"""Distribution layer — interface + parametric implementations + codec."""

from __future__ import annotations

from projections.distributions.base import Distribution
from projections.distributions.codec import pack_per_stat_params, unpack_per_stat_params
from projections.distributions.parametric import ParametricGamma, ParametricNormal

__all__ = [
    "Distribution",
    "ParametricGamma",
    "ParametricNormal",
    "pack_per_stat_params",
    "unpack_per_stat_params",
]
```

- [ ] **Step 5: Run the test to verify it passes.**

Run: `source .venv/Scripts/activate && pytest tests/test_distributions/test_codec.py -v`
Expected: PASS — all four tests green.

- [ ] **Step 6: Commit (incremental — codec exists; round-trip tests follow in 1.4).**

```bash
git add src/projections/distributions/codec.py src/projections/distributions/__init__.py tests/test_distributions/test_codec.py
git commit -m "feat(distributions): pack_per_stat_params codec for ProjectionWeeklySchema.params"
```

### Task 1.4: Add `unpack_per_stat_params` round-trip + error path tests

**Files:**
- Modify: `tests/test_distributions/test_codec.py`

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_distributions/test_codec.py`:

```python
import msgpack

from projections.distributions import unpack_per_stat_params


def test_round_trip_normal_preserves_params() -> None:
    original = {Stat.RECEIVING_YARDS: ParametricNormal(mean=36.3, std=18.1)}
    blob = pack_per_stat_params(original)
    decoded = unpack_per_stat_params(blob)
    assert set(decoded.keys()) == {Stat.RECEIVING_YARDS}
    d = decoded[Stat.RECEIVING_YARDS]
    assert isinstance(d, ParametricNormal)
    assert d.mean() == pytest.approx(36.3)
    assert d.std() == pytest.approx(18.1)


def test_round_trip_gamma_preserves_params() -> None:
    original = {Stat.RECEPTIONS: ParametricGamma(shape=4.2, scale=0.7)}
    blob = pack_per_stat_params(original)
    decoded = unpack_per_stat_params(blob)
    d = decoded[Stat.RECEPTIONS]
    assert isinstance(d, ParametricGamma)
    assert d.shape == pytest.approx(4.2)
    assert d.scale == pytest.approx(0.7)


def test_round_trip_six_stats_mixed_families_preserves_all() -> None:
    original = {
        Stat.PASSING_YARDS: ParametricNormal(mean=199.5, std=84.5),
        Stat.PASSING_TDS: ParametricGamma(shape=4.2, scale=0.29),
        Stat.INTERCEPTIONS: ParametricGamma(shape=1.6, scale=0.43),
        Stat.RUSHING_YARDS: ParametricNormal(mean=18.2, std=17.9),
        Stat.RUSHING_TDS: ParametricGamma(shape=0.8, scale=0.24),
        Stat.FUMBLES_LOST: ParametricGamma(shape=0.5, scale=0.41),
    }
    blob = pack_per_stat_params(original)
    decoded = unpack_per_stat_params(blob)
    assert set(decoded.keys()) == set(original.keys())
    for stat, original_dist in original.items():
        round_tripped = decoded[stat]
        assert type(round_tripped) is type(original_dist)
        assert round_tripped.mean() == pytest.approx(original_dist.mean())
        assert round_tripped.std() == pytest.approx(original_dist.std())


def test_unknown_schema_version_raises() -> None:
    bad = msgpack.packb({"schema_version": 999, "stats": {}}, use_bin_type=True)
    with pytest.raises(ValueError, match="schema_version"):
        unpack_per_stat_params(bytes(bad))


def test_unknown_family_raises() -> None:
    bad = msgpack.packb(
        {"schema_version": 1, "stats": {"receiving_yards": {"family": "WEIBULL", "k": 1.0}}},
        use_bin_type=True,
    )
    with pytest.raises(ValueError, match="WEIBULL"):
        unpack_per_stat_params(bytes(bad))


def test_unknown_stat_name_raises() -> None:
    bad = msgpack.packb(
        {
            "schema_version": 1,
            "stats": {"this_is_not_a_stat": {"family": "NORMAL", "mean": 0.0, "std": 1.0}},
        },
        use_bin_type=True,
    )
    with pytest.raises(ValueError, match="this_is_not_a_stat"):
        unpack_per_stat_params(bytes(bad))
```

- [ ] **Step 2: Run the tests to verify they pass.** Implementation is already in place from Task 1.3.

Run: `source .venv/Scripts/activate && pytest tests/test_distributions/test_codec.py -v`
Expected: PASS — 10 tests total (4 from 1.3 + 6 new).

- [ ] **Step 3: Run static checks.**

Run: `source .venv/Scripts/activate && mypy src tests && ruff check src tests && ruff format --check src tests`
Expected: zero violations.

- [ ] **Step 4: Commit.**

```bash
git add tests/test_distributions/test_codec.py
git commit -m "test(distributions): codec round-trip + error-path coverage"
```

---

## Phase 2 — `derive_row_seed`

### Task 2.1: Implement `derive_row_seed` and tests

**Files:**
- Create: `tests/test_scoring/test_derive_row_seed.py`
- Modify: `src/projections/scoring/score_distribution.py`
- Modify: `src/projections/scoring/__init__.py`

- [ ] **Step 1: Write the failing test.** Create `tests/test_scoring/test_derive_row_seed.py`:

```python
"""derive_row_seed — determinism, independence, range, cross-process stability."""

from __future__ import annotations

import subprocess
import sys

from projections.scoring import derive_row_seed


def test_seed_is_deterministic_within_process() -> None:
    s1 = derive_row_seed(gsis_id="00-0033873", season=2024, week=4, ruleset_name="ESPN_PPR")
    s2 = derive_row_seed(gsis_id="00-0033873", season=2024, week=4, ruleset_name="ESPN_PPR")
    assert s1 == s2


def test_seed_fits_uint32() -> None:
    s = derive_row_seed(gsis_id="00-0033873", season=2024, week=4, ruleset_name="ESPN_PPR")
    assert 0 <= s < 2**32


def test_seed_differs_when_gsis_id_differs() -> None:
    a = derive_row_seed(gsis_id="00-0033873", season=2024, week=4, ruleset_name="ESPN_PPR")
    b = derive_row_seed(gsis_id="00-0035640", season=2024, week=4, ruleset_name="ESPN_PPR")
    assert a != b


def test_seed_differs_when_season_differs() -> None:
    a = derive_row_seed(gsis_id="00-0033873", season=2023, week=4, ruleset_name="ESPN_PPR")
    b = derive_row_seed(gsis_id="00-0033873", season=2024, week=4, ruleset_name="ESPN_PPR")
    assert a != b


def test_seed_differs_when_week_differs() -> None:
    a = derive_row_seed(gsis_id="00-0033873", season=2024, week=3, ruleset_name="ESPN_PPR")
    b = derive_row_seed(gsis_id="00-0033873", season=2024, week=4, ruleset_name="ESPN_PPR")
    assert a != b


def test_seed_differs_when_ruleset_differs() -> None:
    a = derive_row_seed(gsis_id="00-0033873", season=2024, week=4, ruleset_name="ESPN_PPR")
    b = derive_row_seed(gsis_id="00-0033873", season=2024, week=4, ruleset_name="ESPN_HALF")
    assert a != b


def test_seed_stable_across_processes_with_different_pythonhashseed() -> None:
    """Python's built-in hash() is salt-randomized via PYTHONHASHSEED;
    derive_row_seed uses sha256 instead and must be invariant."""
    code = (
        "from projections.scoring import derive_row_seed; "
        "print(derive_row_seed("
        "gsis_id='00-0033873', season=2024, week=4, ruleset_name='ESPN_PPR'))"
    )
    env_a = {"PYTHONHASHSEED": "0"}
    env_b = {"PYTHONHASHSEED": "12345"}
    out_a = subprocess.run(
        [sys.executable, "-c", code], env=env_a, capture_output=True, text=True, check=True
    )
    out_b = subprocess.run(
        [sys.executable, "-c", code], env=env_b, capture_output=True, text=True, check=True
    )
    assert out_a.stdout.strip() == out_b.stdout.strip()
```

- [ ] **Step 2: Run the test to verify it fails.**

Run: `source .venv/Scripts/activate && pytest tests/test_scoring/test_derive_row_seed.py -v`
Expected: FAIL — `derive_row_seed` not importable from `projections.scoring`.

- [ ] **Step 3: Implement `derive_row_seed`.** Add to `src/projections/scoring/score_distribution.py`. Insert near the top of the file, after the existing imports (after the existing `from projections.scoring.score import StatLine` line) — both the new `import hashlib` line and the function:

```python
import hashlib


def derive_row_seed(*, gsis_id: str, season: int, week: int, ruleset_name: str) -> int:
    """Stable 32-bit seed from (gsis_id, season, week, ruleset_name).

    Used by BaselineModel.predict_distribution and aggregate_to_season to keep
    per-row Monte Carlo draws independent and reproducible.

    Properties:
      - Deterministic across processes. Python's built-in hash() is
        salt-randomized via PYTHONHASHSEED; this uses sha256 instead.
      - Independent: changes to any of the four inputs change the seed.
      - Reproducible: identical inputs always produce identical samples
        downstream.

    Returns:
        An int in [0, 2**32).
    """
    h = hashlib.sha256(f"{gsis_id}|{season}|{week}|{ruleset_name}".encode()).digest()
    return int.from_bytes(h[:4], "big")
```

Note: the function is keyword-only (`*` in signature) so that callers cannot accidentally pass arguments in the wrong order.

- [ ] **Step 4: Re-export from `src/projections/scoring/__init__.py`.** Replace the file contents with:

```python
"""Scoring engine -- pure stat -> points math, ruleset-parameterized."""

from __future__ import annotations

from projections.scoring.score import StatLine, score
from projections.scoring.score_distribution import (
    INTEGER_STATS,
    SampledDistribution,
    derive_row_seed,
    score_distribution,
)

__all__ = [
    "INTEGER_STATS",
    "SampledDistribution",
    "StatLine",
    "derive_row_seed",
    "score",
    "score_distribution",
]
```

- [ ] **Step 5: Run the tests to verify they pass.**

Run: `source .venv/Scripts/activate && pytest tests/test_scoring/test_derive_row_seed.py -v`
Expected: PASS — all 7 tests green.

- [ ] **Step 6: Run static checks.**

Run: `source .venv/Scripts/activate && mypy src tests && ruff check src tests && ruff format --check src tests`
Expected: zero violations.

- [ ] **Step 7: Commit.**

```bash
git add src/projections/scoring/score_distribution.py src/projections/scoring/__init__.py tests/test_scoring/test_derive_row_seed.py
git commit -m "feat(scoring): derive_row_seed — stable 32-bit per-row seed via sha256"
```

---

## Phase 3 — `predict_distribution` rewire

### Task 3.1: Add tests for new `predict_distribution` behavior

**Files:**
- Modify: `tests/test_models/test_baseline.py`

- [ ] **Step 1: Re-read `tests/test_models/test_baseline.py` lines 178-280.** Confirm the existing fixture name (`baseline_features_wr`, `baseline_weekly_stats_wr`) and the `wr_baseline()` factory pattern. The new tests use the same fixtures.

- [ ] **Step 2: Add new tests at the end of `tests/test_models/test_baseline.py`.** Append:

```python
def test_predict_distribution_writes_sampled_summary_family(
    baseline_features_wr: pd.DataFrame, baseline_weekly_stats_wr: pd.DataFrame
) -> None:
    from projections.schemas import DistributionFamily

    model = wr_baseline()
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)
    week_features = baseline_features_wr[
        (baseline_features_wr["season"] == 2025) & (baseline_features_wr["week"] == 4)
    ]
    out = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    assert (out["family"] == DistributionFamily.SAMPLED_SUMMARY.value).all()


def test_predict_distribution_params_round_trips_to_per_stat_dists(
    baseline_features_wr: pd.DataFrame, baseline_weekly_stats_wr: pd.DataFrame
) -> None:
    from projections.distributions import unpack_per_stat_params

    model = wr_baseline()
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)
    week_features = baseline_features_wr[
        (baseline_features_wr["season"] == 2025) & (baseline_features_wr["week"] == 4)
    ].head(3)
    expected_dists_per_row = model.build_stat_distributions(week_features)
    out = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())

    for row_idx, expected_dists in enumerate(expected_dists_per_row):
        decoded = unpack_per_stat_params(out["params"].iloc[row_idx])
        assert set(decoded.keys()) == set(expected_dists.keys())
        for stat, expected_dist in expected_dists.items():
            decoded_dist = decoded[stat]
            assert type(decoded_dist) is type(expected_dist)
            assert decoded_dist.mean() == pytest.approx(expected_dist.mean())
            assert decoded_dist.std() == pytest.approx(expected_dist.std())


def test_predict_distribution_uses_per_row_seeds(
    baseline_features_wr: pd.DataFrame, baseline_weekly_stats_wr: pd.DataFrame
) -> None:
    """Two rows for different (gsis_id, week) tuples must produce different
    samples even if their per-stat dists happen to coincide. We check that
    the persisted (mean, p10, p50, p90) summary is never identical across
    distinct rows of the same week (would happen under shared seed=42)."""
    model = wr_baseline()
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)
    week_features = baseline_features_wr[
        (baseline_features_wr["season"] == 2025) & (baseline_features_wr["week"] == 4)
    ]
    out = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    # Across all rows, every (mean, p10, p50, p90) tuple should be unique
    # because per-row seeds vary even when feature inputs are similar.
    summary_tuples = list(zip(out["mean"], out["p10"], out["p50"], out["p90"], strict=True))
    assert len(set(summary_tuples)) == len(summary_tuples)


def test_predict_distribution_is_deterministic_across_calls(
    baseline_features_wr: pd.DataFrame, baseline_weekly_stats_wr: pd.DataFrame
) -> None:
    """Two predict_distribution calls with the same fitted model and the same
    input frame must produce bit-identical mean/p10/p50/p90 columns — closes
    TODO #19's gate non-determinism check by demonstration."""
    model = wr_baseline()
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)
    week_features = baseline_features_wr[
        (baseline_features_wr["season"] == 2025) & (baseline_features_wr["week"] == 4)
    ]
    out_a = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    out_b = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    for col in ("mean", "p10", "p50", "p90"):
        assert (out_a[col].to_numpy() == out_b[col].to_numpy()).all(), (
            f"Determinism violated on column {col}"
        )
```

- [ ] **Step 3: Run the new tests to verify they fail.**

Run: `source .venv/Scripts/activate && pytest tests/test_models/test_baseline.py -v -k "sampled_summary or params_round_trips or per_row_seeds or deterministic_across_calls"`
Expected: FAIL on the family / params / determinism checks (current code writes `family=SAMPLED` with the summary blob and `seed=42` for every row).

### Task 3.2: Rewire `predict_distribution`

**Files:**
- Modify: `src/projections/models/baseline.py`

- [ ] **Step 1: Re-read `src/projections/models/baseline.py` lines 1-50** and **lines 439-512** to confirm the imports and the current `predict_distribution` body. The msgpack import is on line 18; the loop is around lines 470-503.

- [ ] **Step 2: Update the import block at the top of `src/projections/models/baseline.py`.** Replace:

```python
import msgpack
import numpy as np
import pandas as pd
import pandera.pandas as pa
from sklearn.linear_model import RidgeCV

from projections.distributions import Distribution, ParametricGamma, ParametricNormal
from projections.models.base import compute_code_hash
from projections.schemas import (
    _PYARROW_STR,
    DistributionFamily,
    Position,
    ProjectionWeeklySchema,
    QbFeaturesSchema,
    RbFeaturesSchema,
    Ruleset,
    Stat,
    TeFeaturesSchema,
    WeeklyStatsSchema,
    WrFeaturesSchema,
)
from projections.scoring import score_distribution
```

with (note: `msgpack` removed; `pack_per_stat_params` added; `derive_row_seed` added):

```python
import numpy as np
import pandas as pd
import pandera.pandas as pa
from sklearn.linear_model import RidgeCV

from projections.distributions import (
    Distribution,
    ParametricGamma,
    ParametricNormal,
    pack_per_stat_params,
)
from projections.models.base import compute_code_hash
from projections.schemas import (
    _PYARROW_STR,
    DistributionFamily,
    Position,
    ProjectionWeeklySchema,
    QbFeaturesSchema,
    RbFeaturesSchema,
    Ruleset,
    Stat,
    TeFeaturesSchema,
    WeeklyStatsSchema,
    WrFeaturesSchema,
)
from projections.scoring import derive_row_seed, score_distribution
```

- [ ] **Step 3: Rewrite the `predict_distribution` method body.** Replace the entire method (the docstring AND the body) so it becomes:

```python
    def predict_distribution(self, features: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame:
        """Predict per-player-week fantasy-points distributions under ``ruleset``.

        Returns a DataFrame validated against ``ProjectionWeeklySchema`` with one
        row per ``features`` row. Each row's persisted ``mean`` / ``p10`` / ``p50``
        / ``p90`` columns are the canonical per-row distributional summary; the
        ``params`` blob carries per-stat distribution parameters via
        ``pack_per_stat_params`` so a downstream consumer can rehydrate the
        per-stat distributions and regenerate samples deterministically.

        Per-row Monte Carlo seed is derived from ``(gsis_id, season, week,
        ruleset.name)`` via ``derive_row_seed``, giving cross-process reproducible
        and cross-row independent samples.
        """
        features = self.feature_schema.validate(features)
        if features.empty:
            empty_cols = list(ProjectionWeeklySchema.to_schema().columns.keys())
            return ProjectionWeeklySchema.validate(pd.DataFrame(columns=empty_cols))

        stat_dists_per_row = self.build_stat_distributions(features)

        rows: list[dict[str, object]] = []
        generated_at = datetime.now(UTC)
        for (_idx, feat_row), stat_dists in zip(
            features.reset_index(drop=True).iterrows(), stat_dists_per_row, strict=True
        ):
            seed = derive_row_seed(
                gsis_id=str(feat_row["gsis_id"]),
                season=int(feat_row["season"]),
                week=int(feat_row["week"]),
                ruleset_name=ruleset.name,
            )
            points = score_distribution(stat_dists, ruleset, n_samples=10_000, seed=seed)
            family_blob = pack_per_stat_params(stat_dists)
            rows.append(
                {
                    "gsis_id": feat_row["gsis_id"],
                    "season": int(feat_row["season"]),
                    "week": int(feat_row["week"]),
                    "position": self.position.value,
                    "team": feat_row["team"],
                    "opponent": feat_row["opponent"],
                    "ruleset": ruleset.name,
                    "family": DistributionFamily.SAMPLED_SUMMARY.value,
                    "params": family_blob,
                    "mean": points.mean(),
                    "p10": points.quantile(0.1),
                    "p50": points.quantile(0.5),
                    "p90": points.quantile(0.9),
                    "model_id": self.model_id,
                    "generated_at": pd.Timestamp(generated_at).as_unit("us"),
                }
            )

        out = pd.DataFrame(rows)
        for col in ("gsis_id", "team", "opponent", "ruleset", "family", "model_id"):
            out[col] = out[col].astype(_PYARROW_STR)
        out["position"] = out["position"].astype(_PYARROW_STR)
        return ProjectionWeeklySchema.validate(out)
```

The two `# v1 limitation` paragraphs (cross-row sample correlation; params summary-only) are gone — both closed.

- [ ] **Step 4: Run the targeted tests to verify they pass.**

Run: `source .venv/Scripts/activate && pytest tests/test_models/test_baseline.py -v -k "sampled_summary or params_round_trips or per_row_seeds or deterministic_across_calls"`
Expected: PASS — all 4 new tests green.

- [ ] **Step 5: Run the full baseline test file.**

Run: `source .venv/Scripts/activate && pytest tests/test_models/test_baseline.py -v`
Expected: PASS — all existing tests still green plus the 4 new ones.

- [ ] **Step 6: Run all per-position baseline tests.**

Run: `source .venv/Scripts/activate && pytest tests/test_models/ -v`
Expected: PASS — QB, RB, TE, WR all green. (All four positions go through the same `BaselineModel.predict_distribution` so they pick up the change automatically.)

- [ ] **Step 7: Run static checks.**

Run: `source .venv/Scripts/activate && mypy src tests && ruff check src tests && ruff format --check src tests`
Expected: zero violations.

- [ ] **Step 8: Commit.**

```bash
git add src/projections/models/baseline.py tests/test_models/test_baseline.py
git commit -m "feat(models): predict_distribution uses per-row seeds + per-stat params blob

- TODO #13 closure: derive_row_seed(gsis_id, season, week, ruleset.name)
  gives independent, reproducible samples per row.
- TODO #14 closure: family=SAMPLED_SUMMARY; params blob now carries per-stat
  distribution parameters via pack_per_stat_params, sufficient to rehydrate
  per-stat distributions and regenerate samples downstream."
```

---

## Phase 4 — `aggregate_to_season`

### Task 4.1: Create aggregation package skeleton

**Files:**
- Create: `src/projections/aggregation/__init__.py`
- Create: `src/projections/aggregation/season.py` (placeholder)
- Create: `tests/test_aggregation/__init__.py`
- Create: `tests/test_aggregation/test_season.py` (failing test)

- [ ] **Step 1: Create the package skeleton.** Create `src/projections/aggregation/__init__.py`:

```python
"""Aggregation layer -- weekly projections to season-total distributions."""

from __future__ import annotations

from projections.aggregation.season import aggregate_to_season

__all__ = ["aggregate_to_season"]
```

- [ ] **Step 2: Create the placeholder `src/projections/aggregation/season.py`.**

```python
"""Aggregate weekly per-player projections into season-total distributions."""

from __future__ import annotations

import pandas as pd

from projections.schemas import Ruleset


def aggregate_to_season(
    weekly: pd.DataFrame,
    *,
    ruleset: Ruleset,
    n_samples: int = 10_000,
) -> pd.DataFrame:
    """Stub — see Task 4.2 for the real implementation."""
    raise NotImplementedError("Implemented in Task 4.2")
```

- [ ] **Step 3: Create `tests/test_aggregation/__init__.py` (empty file).**

```python
```

(An empty file is fine; pytest just needs the package marker.)

- [ ] **Step 4: Create the first failing test in `tests/test_aggregation/test_season.py`.**

```python
"""aggregate_to_season — pure-function tests over ProjectionWeeklySchema rows."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from projections.aggregation import aggregate_to_season
from projections.distributions import (
    ParametricGamma,
    ParametricNormal,
    pack_per_stat_params,
)
from projections.schemas import (
    DistributionFamily,
    ProjectionSeasonSchema,
    Ruleset,
    Stat,
    _PYARROW_STR,
)
from projections.scoring import derive_row_seed, score_distribution


_RULESET = Ruleset.espn_ppr()


def _build_weekly_row(
    *,
    gsis_id: str = "00-0033873",
    season: int = 2024,
    week: int = 1,
    position: str = "WR",
    team: str = "KC",
    opponent: str = "BAL",
    family: str = DistributionFamily.SAMPLED_SUMMARY.value,
    ruleset_name: str | None = None,
    rec_yards_mean: float = 50.0,
    rec_yards_std: float = 18.0,
    rec_shape: float = 4.0,
    rec_scale: float = 0.7,
    model_id: str = "baseline:wr:abcdef12:2018-2023",
) -> dict[str, Any]:
    rs_name = ruleset_name if ruleset_name is not None else _RULESET.name
    per_stat_dists = {
        Stat.RECEIVING_YARDS: ParametricNormal(mean=rec_yards_mean, std=rec_yards_std),
        Stat.RECEPTIONS: ParametricGamma(shape=rec_shape, scale=rec_scale),
    }
    blob = pack_per_stat_params(per_stat_dists)
    seed = derive_row_seed(gsis_id=gsis_id, season=season, week=week, ruleset_name=rs_name)
    points = score_distribution(per_stat_dists, _RULESET, n_samples=10_000, seed=seed)
    return {
        "gsis_id": gsis_id,
        "season": season,
        "week": week,
        "position": position,
        "team": team,
        "opponent": opponent,
        "ruleset": rs_name,
        "family": family,
        "params": blob,
        "mean": points.mean(),
        "p10": points.quantile(0.1),
        "p50": points.quantile(0.5),
        "p90": points.quantile(0.9),
        "model_id": model_id,
        "generated_at": pd.Timestamp(datetime.now(UTC)).as_unit("us"),
    }


def _to_weekly_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ("gsis_id", "team", "opponent", "ruleset", "family", "model_id", "position"):
        df[col] = df[col].astype(_PYARROW_STR)
    return df


def test_empty_input_returns_empty_validated_frame() -> None:
    empty = pd.DataFrame(
        columns=[
            "gsis_id", "season", "week", "position", "team", "opponent", "ruleset",
            "family", "params", "mean", "p10", "p50", "p90", "model_id", "generated_at",
        ]
    )
    out = aggregate_to_season(empty, ruleset=_RULESET)
    assert out.empty
    ProjectionSeasonSchema.validate(out)
```

- [ ] **Step 5: Run the test to verify it fails.**

Run: `source .venv/Scripts/activate && pytest tests/test_aggregation/test_season.py -v`
Expected: FAIL with `NotImplementedError` raised by the placeholder.

- [ ] **Step 6: Commit the skeleton.**

```bash
git add src/projections/aggregation/__init__.py src/projections/aggregation/season.py tests/test_aggregation/__init__.py tests/test_aggregation/test_season.py
git commit -m "scaffold(aggregation): aggregate_to_season package skeleton + first test"
```

### Task 4.2: Implement `aggregate_to_season`

**Files:**
- Modify: `src/projections/aggregation/season.py`

- [ ] **Step 1: Replace `src/projections/aggregation/season.py` with the real implementation.**

```python
"""Aggregate weekly per-player projections into season-total distributions.

Pure function over ProjectionWeeklySchema rows (no model coupling, no parquet
I/O). For each (gsis_id, season) group the function:

  - Decodes each week's per-stat distribution params via unpack_per_stat_params.
  - Re-derives each week's per-row seed via derive_row_seed.
  - Calls score_distribution(...) to regenerate that week's points samples.
  - Sums per-week sample arrays positionally -> n_samples season-total samples.
  - Summarizes mean / p10 / p50 / p90.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from projections.distributions import unpack_per_stat_params
from projections.schemas import (
    DistributionFamily,
    ProjectionSeasonSchema,
    ProjectionWeeklySchema,
    Ruleset,
    _PYARROW_STR,
)
from projections.scoring import derive_row_seed, score_distribution


def aggregate_to_season(
    weekly: pd.DataFrame,
    *,
    ruleset: Ruleset,
    n_samples: int = 10_000,
) -> pd.DataFrame:
    """Aggregate weekly per-player projections into season-total distributions.

    The input is validated against ProjectionWeeklySchema. Every row must have
    family == DistributionFamily.SAMPLED_SUMMARY and ruleset == ruleset.name --
    a mixed-ruleset frame or a row written before Plan 3d's codec swap raises
    ValueError immediately.

    Returns a ProjectionSeasonSchema-validated DataFrame with one row per
    (gsis_id, season). position is the modal value across the input rows for
    that gsis_id (handles in-season position changes deterministically).

    Empty input returns an empty validated frame.
    """
    weekly = ProjectionWeeklySchema.validate(weekly)
    if weekly.empty:
        empty_cols = list(ProjectionSeasonSchema.to_schema().columns.keys())
        return ProjectionSeasonSchema.validate(pd.DataFrame(columns=empty_cols))

    bad_family = weekly[weekly["family"] != DistributionFamily.SAMPLED_SUMMARY.value]
    if not bad_family.empty:
        raise ValueError(
            f"aggregate_to_season requires family={DistributionFamily.SAMPLED_SUMMARY.value}, "
            f"found {bad_family['family'].unique().tolist()}"
        )

    bad_ruleset = weekly[weekly["ruleset"] != ruleset.name]
    if not bad_ruleset.empty:
        raise ValueError(
            f"Mixed-ruleset input: expected {ruleset.name}, "
            f"found {bad_ruleset['ruleset'].unique().tolist()}"
        )

    rows: list[dict[str, object]] = []
    generated_at = datetime.now(UTC)
    for (gsis_id, season), group in weekly.groupby(["gsis_id", "season"], sort=False):
        season_samples = np.zeros(n_samples, dtype=np.float64)
        for _idx, week_row in group.iterrows():
            per_stat_dists = unpack_per_stat_params(bytes(week_row["params"]))
            seed = derive_row_seed(
                gsis_id=str(gsis_id),
                season=int(season),
                week=int(week_row["week"]),
                ruleset_name=ruleset.name,
            )
            week_dist = score_distribution(
                per_stat_dists, ruleset, n_samples=n_samples, seed=seed
            )
            season_samples += week_dist.samples

        position = group["position"].mode().iloc[0]
        rows.append(
            {
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
            }
        )

    out = pd.DataFrame(rows)
    for col in ("gsis_id", "position", "ruleset", "model_id"):
        out[col] = out[col].astype(_PYARROW_STR)
    return ProjectionSeasonSchema.validate(out)
```

- [ ] **Step 2: Run the empty-input test to verify it passes.**

Run: `source .venv/Scripts/activate && pytest tests/test_aggregation/test_season.py::test_empty_input_returns_empty_validated_frame -v`
Expected: PASS.

- [ ] **Step 3: Run static checks (early).**

Run: `source .venv/Scripts/activate && mypy src tests && ruff check src tests && ruff format --check src tests`
Expected: zero violations.

- [ ] **Step 4: Commit.**

```bash
git add src/projections/aggregation/season.py
git commit -m "feat(aggregation): aggregate_to_season real Monte Carlo season aggregator"
```

### Task 4.3: Add the rest of the aggregator tests

**Files:**
- Modify: `tests/test_aggregation/test_season.py`

- [ ] **Step 1: Append the remaining tests** to `tests/test_aggregation/test_season.py`:

```python
def test_single_player_single_week_n_weeks_is_one() -> None:
    weekly = _to_weekly_frame([_build_weekly_row(week=1)])
    out = aggregate_to_season(weekly, ruleset=_RULESET)
    assert len(out) == 1
    assert out["n_weeks"].iloc[0] == 1
    # season_mean is the same week's mean (within MC noise from regeneration).
    assert out["season_mean"].iloc[0] == pytest.approx(weekly["mean"].iloc[0], rel=0.02)


def test_single_player_multi_week_quantiles_widen() -> None:
    """Sum of independent random variables has wider quantile spread than any
    single one. season_p90 - season_p10 should exceed any single week's
    p90 - p10 because variances add."""
    weekly = _to_weekly_frame(
        [_build_weekly_row(week=w) for w in range(1, 6)]
    )
    out = aggregate_to_season(weekly, ruleset=_RULESET)
    assert len(out) == 1
    assert out["n_weeks"].iloc[0] == 5
    season_spread = out["season_p90"].iloc[0] - out["season_p10"].iloc[0]
    weekly_spread = weekly["p90"].iloc[0] - weekly["p10"].iloc[0]
    # 5 independent weeks: variance scales 5x => std scales sqrt(5)x => spread
    # also scales sqrt(5) ~= 2.24x. Be conservative; require >= 1.5x.
    assert season_spread >= 1.5 * weekly_spread


def test_multi_player_one_row_per_player() -> None:
    weekly = _to_weekly_frame(
        [
            _build_weekly_row(gsis_id="00-0033873", week=w) for w in range(1, 4)
        ]
        + [
            _build_weekly_row(gsis_id="00-0035640", week=w) for w in range(1, 4)
        ]
    )
    out = aggregate_to_season(weekly, ruleset=_RULESET)
    assert len(out) == 2
    assert set(out["gsis_id"]) == {"00-0033873", "00-0035640"}
    assert (out["n_weeks"] == 3).all()


def test_traded_player_modal_position() -> None:
    """Same gsis_id appears with two positions across weeks; modal value wins."""
    rows = [
        _build_weekly_row(week=1, position="WR"),
        _build_weekly_row(week=2, position="WR"),
        _build_weekly_row(week=3, position="WR"),
        _build_weekly_row(week=4, position="RB"),
    ]
    weekly = _to_weekly_frame(rows)
    out = aggregate_to_season(weekly, ruleset=_RULESET)
    assert len(out) == 1
    assert out["position"].iloc[0] == "WR"


def test_non_sampled_summary_family_raises() -> None:
    rows = [
        _build_weekly_row(week=1, family=DistributionFamily.SAMPLED.value),
    ]
    weekly = _to_weekly_frame(rows)
    with pytest.raises(ValueError, match="SAMPLED_SUMMARY"):
        aggregate_to_season(weekly, ruleset=_RULESET)


def test_mixed_ruleset_input_raises() -> None:
    rows = [
        _build_weekly_row(week=1, ruleset_name="ESPN_PPR"),
        _build_weekly_row(week=2, ruleset_name="ESPN_HALF"),
    ]
    weekly = _to_weekly_frame(rows)
    with pytest.raises(ValueError, match="ruleset"):
        aggregate_to_season(weekly, ruleset=_RULESET)


def test_aggregate_is_deterministic_across_calls() -> None:
    weekly = _to_weekly_frame(
        [_build_weekly_row(week=w) for w in range(1, 6)]
    )
    out_a = aggregate_to_season(weekly, ruleset=_RULESET)
    out_b = aggregate_to_season(weekly, ruleset=_RULESET)
    for col in ("season_mean", "season_p10", "season_p50", "season_p90"):
        assert (out_a[col].to_numpy() == out_b[col].to_numpy()).all()


def test_output_validates_against_projection_season_schema() -> None:
    weekly = _to_weekly_frame(
        [_build_weekly_row(week=w) for w in range(1, 6)]
    )
    out = aggregate_to_season(weekly, ruleset=_RULESET)
    # Already validated inside the function, but assert explicitly here.
    ProjectionSeasonSchema.validate(out)
    assert "season_mean" in out.columns
    assert "n_weeks" in out.columns
    assert "position" in out.columns
```

- [ ] **Step 2: Run the full aggregation suite to verify all tests pass.**

Run: `source .venv/Scripts/activate && pytest tests/test_aggregation/test_season.py -v`
Expected: PASS — 9 tests total.

- [ ] **Step 3: Run static checks.**

Run: `source .venv/Scripts/activate && mypy src tests && ruff check src tests && ruff format --check src tests`
Expected: zero violations.

- [ ] **Step 4: Commit.**

```bash
git add tests/test_aggregation/test_season.py
git commit -m "test(aggregation): aggregate_to_season behavior + error-path coverage"
```

---

## Phase 5a — Harness wiring + season-calibration metric

### Task 5a.1: Verify `_METRIC_KIND_RULES` classifies season metrics correctly

**Files:**
- (No code change expected — this is a verification step that may extend `src/projections/backtest/snapshot.py` if the classifier doesn't already handle the new names.)

- [ ] **Step 1: Re-read `src/projections/backtest/snapshot.py:24-50`.** Confirm the existing rule list:

```python
_METRIC_KIND_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mean_pred_relative", ("_mean_pred",)),
    ("rmse_relative", ("_rmse",)),
    ("mae_relative", ("_mae",)),
    ("spearman_absolute", ("spearman_",)),
    ("calibration_absolute", ("calibration_",)),
)
```

`season_calibration_p10p90` and `season_calibration_le_p90` both contain the substring `calibration_`, so they route to `calibration_absolute` correctly under the existing rule. **No change to `_METRIC_KIND_RULES` is needed.**

- [ ] **Step 2: Add a test that pins this expectation.** Append to `tests/test_backtest/test_snapshot.py`:

```python
def test_classify_metric_routes_season_calibration_to_calibration_absolute() -> None:
    from projections.backtest.snapshot import _classify_metric

    assert _classify_metric("season_calibration_p10p90") == "calibration_absolute"
    assert _classify_metric("season_calibration_le_p90") == "calibration_absolute"
```

- [ ] **Step 3: Run the test to verify it passes.**

Run: `source .venv/Scripts/activate && pytest tests/test_backtest/test_snapshot.py::test_classify_metric_routes_season_calibration_to_calibration_absolute -v`
Expected: PASS — the existing classifier already handles this.

- [ ] **Step 4: Commit.**

```bash
git add tests/test_backtest/test_snapshot.py
git commit -m "test(backtest): pin season_calibration_* -> calibration_absolute classifier"
```

### Task 5a.2: Add `compute_season_calibration_metrics`

**Files:**
- Modify: `src/projections/backtest/metrics.py`
- Modify: `tests/test_backtest/test_metrics.py`

- [ ] **Step 1: Write the failing test.** Append to `tests/test_backtest/test_metrics.py`:

```python
def test_compute_season_calibration_metrics_known_frame() -> None:
    from projections.backtest.metrics import compute_season_calibration_metrics

    # 5 of 10 in [p10, p90]; 7 of 10 <= p90.
    df = pd.DataFrame(
        {
            "season_p10": [10.0] * 10,
            "season_p90": [50.0] * 10,
            "actual_season_total": [
                5.0,    # below p10 -> not in p10p90, not <= p90? actually 5 <= 50 yes
                15.0,   # in p10p90, <= p90
                25.0,   # in p10p90, <= p90
                40.0,   # in p10p90, <= p90
                45.0,   # in p10p90, <= p90
                49.0,   # in p10p90, <= p90
                51.0,   # > p90, not <= p90
                60.0,   # > p90, not <= p90
                70.0,   # > p90, not <= p90
                100.0,  # > p90, not <= p90
            ],
        }
    )
    out = compute_season_calibration_metrics(df)
    # 5 of 10 are in [10, 50] (15, 25, 40, 45, 49). The 5.0 is below p10 (10).
    # Wait: 5 < 10 -> not in [p10, p90].
    # 15, 25, 40, 45, 49 -> 5 in p10p90.
    # <= p90: 5, 15, 25, 40, 45, 49 -> 6 <= p90.
    assert out["season_calibration_p10p90"] == pytest.approx(0.5)
    assert out["season_calibration_le_p90"] == pytest.approx(0.6)


def test_compute_season_calibration_metrics_empty_frame_returns_nan() -> None:
    from math import isnan

    from projections.backtest.metrics import compute_season_calibration_metrics

    df = pd.DataFrame(columns=["season_p10", "season_p90", "actual_season_total"])
    out = compute_season_calibration_metrics(df)
    assert isnan(out["season_calibration_p10p90"])
    assert isnan(out["season_calibration_le_p90"])
```

- [ ] **Step 2: Run the test to verify it fails.**

Run: `source .venv/Scripts/activate && pytest tests/test_backtest/test_metrics.py -v -k "season_calibration"`
Expected: FAIL — `compute_season_calibration_metrics` not defined.

- [ ] **Step 3: Implement `compute_season_calibration_metrics`.** Append to `src/projections/backtest/metrics.py`:

```python
def compute_season_calibration_metrics(season_eval_df: pd.DataFrame) -> dict[str, float]:
    """Calibration coverage on season-total predictions.

    Expects columns ``season_p10``, ``season_p90``, ``actual_season_total``.
    Returns ``season_calibration_p10p90`` (fraction of (gsis_id, year) where
    actual is in [season_p10, season_p90]) and ``season_calibration_le_p90``
    (fraction where actual <= season_p90).

    Mirrors compute_calibration_metrics's shape; not gated for the naive
    baseline because point predictors have collapsed quantiles.

    Returns NaN for both metrics on an empty frame.
    """
    if season_eval_df.empty:
        return {
            "season_calibration_p10p90": float("nan"),
            "season_calibration_le_p90": float("nan"),
        }
    a = season_eval_df["actual_season_total"]
    in_p10p90 = ((a >= season_eval_df["season_p10"]) & (a <= season_eval_df["season_p90"])).mean()
    le_p90 = (a <= season_eval_df["season_p90"]).mean()
    return {
        "season_calibration_p10p90": float(in_p10p90),
        "season_calibration_le_p90": float(le_p90),
    }
```

- [ ] **Step 4: Run the tests to verify they pass.**

Run: `source .venv/Scripts/activate && pytest tests/test_backtest/test_metrics.py -v -k "season_calibration"`
Expected: PASS — 2 new tests green.

- [ ] **Step 5: Commit.**

```bash
git add src/projections/backtest/metrics.py tests/test_backtest/test_metrics.py
git commit -m "feat(backtest): compute_season_calibration_metrics"
```

### Task 5a.3: Wire season aggregation into the harness

**Files:**
- Modify: `src/projections/backtest/harness.py`

- [ ] **Step 1: Re-read `src/projections/backtest/harness.py` lines 1-50** for the imports and the dataclass; **lines 185-272** for `run_backtest`.

- [ ] **Step 2: Update the import block.** Replace:

```python
from projections.backtest.metrics import compute_all_metrics
from projections.backtest.naive import compute_naive_predictions
from projections.features.cache import read_features
from projections.models import POSITION_DISPATCH
from projections.schemas import Position, Ruleset, Stat
from projections.scoring import INTEGER_STATS, score
from projections.scoring.score import StatLine
from projections.store import read_partition
```

with:

```python
from projections.aggregation import aggregate_to_season
from projections.backtest.metrics import (
    compute_all_metrics,
    compute_season_calibration_metrics,
)
from projections.backtest.naive import compute_naive_predictions
from projections.features.cache import read_features
from projections.models import POSITION_DISPATCH
from projections.schemas import Position, Ruleset, Stat
from projections.scoring import INTEGER_STATS, score
from projections.scoring.score import StatLine
from projections.store import read_partition
```

- [ ] **Step 3: Extend the `BacktestRun` dataclass.** Replace the existing dataclass with:

```python
@dataclass(frozen=True, slots=True)
class BacktestRun:
    """Result of a single walk-forward backtest invocation.

    Attributes:
        timestamp: UTC time the run started; used to name diagnostic
            output directories under data/backtest/run_<ts>/.
        metrics: long-form DataFrame with columns
            (position, year, metric, value) -- the model's metrics across
            (position, year, metric) cells. Becomes the snapshot input.
        naive_metrics: same shape; computed alongside model metrics for
            informational reporting. Not gated.
        per_row_results: per-(position, year, week, gsis_id) row of
            actuals + model predictions for diagnosis. Plan 3c writes
            this to data/backtest/run_<ts>/results.parquet (gitignored).
        per_player_results: per-(position, year, gsis_id) season eval row
            for diagnosis. Plan 3d writes this to
            data/backtest/run_<ts>/season_results.parquet (gitignored).
    """

    timestamp: pd.Timestamp
    metrics: pd.DataFrame
    naive_metrics: pd.DataFrame
    per_row_results: pd.DataFrame
    per_player_results: pd.DataFrame
```

- [ ] **Step 4: Wire season aggregation into `run_backtest`.** Replace the existing `run_backtest` function body so that, immediately after the existing model-metrics loop and before the naive metrics, it builds and appends season metrics, and at the end concatenates `per_player_frames` into `BacktestRun.per_player_results`. Replace the `run_backtest` function (lines 185-272 in current source) with:

```python
def run_backtest(
    *,
    held_out_years: Iterable[int] = (2021, 2022, 2023, 2024),
    positions: Iterable[Position] | None = None,
    train_start: int = 2018,
    features_root: Path = Path("data/features"),
    raw_root: Path = Path("data/raw"),
    ruleset: Ruleset | None = None,
) -> BacktestRun:
    """Walk-forward backtest. Spec section 2.3."""
    if ruleset is None:
        ruleset = Ruleset.espn_ppr()
    if positions is None:
        positions = (Position.QB, Position.RB, Position.TE, Position.WR)

    timestamp = pd.Timestamp(datetime.now(UTC))
    positions_list = list(positions)
    years_list = list(held_out_years)

    metrics_rows: list[dict[str, object]] = []
    naive_rows: list[dict[str, object]] = []
    per_row_frames: list[pd.DataFrame] = []
    per_player_frames: list[pd.DataFrame] = []

    for position in positions_list:
        for year in years_list:
            train_seasons = list(range(train_start, year))
            train_features = pd.concat(
                [read_features(position, s, features_root=features_root) for s in train_seasons],
                ignore_index=True,
            )
            train_actuals = pd.concat(
                [read_partition(raw_root, "weekly_stats", season=s) for s in train_seasons],
                ignore_index=True,
            )
            predict_features = read_features(position, year, features_root=features_root)
            holdout_actuals = read_partition(raw_root, "weekly_stats", season=year)

            # Model: predict per-week, score weekly metrics.
            dispatch = POSITION_DISPATCH[position]
            model = dispatch.factory()
            model.fit(train_features, train_actuals)
            predictions = model.predict_distribution(predict_features, ruleset=ruleset)
            stat_dists_per_row = model.build_stat_distributions(predict_features)
            per_stat_pred_means = pd.DataFrame(
                {
                    stat.value: [d[stat].mean() for d in stat_dists_per_row]
                    for stat in model.target_stats
                }
            )
            per_stat_pred_means["gsis_id"] = predict_features["gsis_id"].values
            per_stat_pred_means["season"] = predict_features["season"].astype(int).values
            per_stat_pred_means["week"] = predict_features["week"].astype(int).values

            holdout_pos = holdout_actuals[holdout_actuals["position"] == position.value].copy()
            holdout_pos["actual_ppr"] = _realized_ppr_points(holdout_pos, ruleset)

            target_stats = tuple(model.target_stats)
            eval_df = _build_eval_df(
                predictions=predictions,
                per_stat_pred_means=per_stat_pred_means,
                held_out_pos=holdout_pos,
                target_stats=target_stats,
            )
            model_metrics = compute_all_metrics(eval_df, target_stats=target_stats)
            for metric_name, value in model_metrics.items():
                metrics_rows.append(
                    {
                        "position": position.value,
                        "year": year,
                        "metric": metric_name,
                        "value": float(value),
                    }
                )

            # Season aggregation: build season predictions, join to season actuals,
            # compute calibration metrics, append to metrics_rows + per_player_frames.
            season_predictions = aggregate_to_season(predictions, ruleset=ruleset)
            season_actuals = (
                holdout_pos.groupby("gsis_id", as_index=False)["actual_ppr"]
                .sum()
                .rename(columns={"actual_ppr": "actual_season_total"})
            )
            season_eval_df = season_predictions.merge(
                season_actuals, on="gsis_id", how="inner"
            )
            season_metrics = compute_season_calibration_metrics(season_eval_df)
            for metric_name, value in season_metrics.items():
                metrics_rows.append(
                    {
                        "position": position.value,
                        "year": year,
                        "metric": metric_name,
                        "value": float(value),
                    }
                )

            # Naive metrics (existing).
            naive_metrics = _naive_metrics_for_cell(
                train_actuals=train_actuals,
                holdout_actuals=holdout_actuals,
                position=position,
                target_stats=target_stats,
                held_out_year=year,
                ruleset=ruleset,
            )
            for metric_name, value in naive_metrics.items():
                naive_rows.append(
                    {
                        "position": position.value,
                        "year": year,
                        "metric": metric_name,
                        "value": float(value),
                    }
                )

            eval_df = eval_df.assign(position=position.value)
            per_row_frames.append(eval_df)
            season_eval_df = season_eval_df.assign(position=position.value)
            per_player_frames.append(season_eval_df)

    metrics_df = pd.DataFrame(metrics_rows, columns=list(_METRICS_COLUMNS))
    naive_metrics_df = pd.DataFrame(naive_rows, columns=list(_METRICS_COLUMNS))
    per_row_results = (
        pd.concat(per_row_frames, ignore_index=True) if per_row_frames else pd.DataFrame()
    )
    per_player_results = (
        pd.concat(per_player_frames, ignore_index=True) if per_player_frames else pd.DataFrame()
    )
    return BacktestRun(
        timestamp=timestamp,
        metrics=metrics_df,
        naive_metrics=naive_metrics_df,
        per_row_results=per_row_results,
        per_player_results=per_player_results,
    )
```

This inlines the model-metrics computation that was previously in `_model_metrics_for_cell` to keep the new season-aggregation step in the same scope. The existing `_model_metrics_for_cell` helper is no longer needed and can be removed; do that next.

- [ ] **Step 5: Remove the now-unused `_model_metrics_for_cell` helper.** Delete the function definition from `src/projections/backtest/harness.py` (lines 144-182 in the pre-3d source). The new inline body fully replaces it.

- [ ] **Step 6: Run static checks early.**

Run: `source .venv/Scripts/activate && mypy src tests && ruff check src tests && ruff format --check src tests`
Expected: zero violations.

- [ ] **Step 7: Commit.**

```bash
git add src/projections/backtest/harness.py
git commit -m "feat(backtest): wire aggregate_to_season + season-calibration metrics into harness"
```

### Task 5a.4: `scripts/backtest.py` writes `season_results.parquet`

**Files:**
- Modify: `scripts/backtest.py`

- [ ] **Step 1: Re-read `scripts/backtest.py`** to confirm there is no current `results.parquet` write call (the file just runs the harness and reports). The plan adds writing of both `results.parquet` and `season_results.parquet` to `data/backtest/run_<ts>/` when running `--update-snapshot` or `--report` (skipped on `--check` to keep the gate hermetic).

- [ ] **Step 2: Add diagnostic-output writing.** In `scripts/backtest.py`, add the following helper near the top (after the constants):

```python
def _write_diagnostic_outputs(run: object) -> None:
    """Write per-row + per-player diagnostic frames to
    data/backtest/run_<ts>/. The directory is gitignored. Skipped silently
    if either frame is empty (e.g., a synthetic test run with no positions)."""
    timestamp_str = (
        pd.Timestamp(run.timestamp).strftime("%Y%m%dT%H%M%SZ")  # type: ignore[attr-defined]
    )
    out_dir = Path("data/backtest") / f"run_{timestamp_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not run.per_row_results.empty:  # type: ignore[attr-defined]
        run.per_row_results.to_parquet(out_dir / "results.parquet")  # type: ignore[attr-defined]
    if not run.per_player_results.empty:  # type: ignore[attr-defined]
        run.per_player_results.to_parquet(out_dir / "season_results.parquet")  # type: ignore[attr-defined]
```

- [ ] **Step 3: Call the helper from `_update` and `_report` paths.** In the `main()` function, modify the `--update-snapshot` and `--report` branches so they call `_write_diagnostic_outputs(run)` after the run completes, **before** the snapshot write or report print. The `--check` branch does not write diagnostic outputs (the gate run is meant to be side-effect-free w.r.t. the data dir):

Replace:

```python
    if args.update_snapshot:
        sys.exit(_update(run))
    if args.report:
        _print_metrics_table("Backtest", run.metrics, run.naive_metrics)
        sys.exit(0)
    # Default: check.
    sys.exit(_check(run, tolerances))
```

with:

```python
    if args.update_snapshot:
        _write_diagnostic_outputs(run)
        sys.exit(_update(run))
    if args.report:
        _write_diagnostic_outputs(run)
        _print_metrics_table("Backtest", run.metrics, run.naive_metrics)
        sys.exit(0)
    # Default: check (no diagnostic output to keep the gate hermetic).
    sys.exit(_check(run, tolerances))
```

- [ ] **Step 4: Run static checks.**

Run: `source .venv/Scripts/activate && mypy src tests && ruff check src tests && ruff format --check src tests`
Expected: zero violations.

- [ ] **Step 5: Commit.**

```bash
git add scripts/backtest.py
git commit -m "feat(scripts): backtest.py writes per_row + per_player results to data/backtest/run_<ts>/"
```

---

## Phase 5b — Tests for harness + smoke

### Task 5b.1: Extend harness tests

**Files:**
- Modify: `tests/test_backtest/test_harness.py`

- [ ] **Step 1: Re-read `tests/test_backtest/test_harness.py`** to understand the existing fixture pattern (`tests/test_backtest/conftest.py` builds a synthetic feature parquet tree under tmp). Confirm what fixtures are available.

- [ ] **Step 2: Append the following tests** to `tests/test_backtest/test_harness.py`. (The fixture name `synthetic_features_root`, etc. is taken from existing tests in this file — re-use whatever the file already uses.)

```python
def test_backtest_run_includes_season_calibration_metrics_for_every_cell(
    synthetic_features_root: Path,
    synthetic_raw_root: Path,
) -> None:
    """Every (position, year) cell contributes both season_calibration_p10p90
    and season_calibration_le_p90 rows."""
    from projections.backtest import run_backtest

    run = run_backtest(
        held_out_years=(2024,),
        features_root=synthetic_features_root,
        raw_root=synthetic_raw_root,
    )
    metrics = run.metrics
    cells = metrics.groupby(["position", "year"])["metric"].apply(set)
    for (position, year), metric_set in cells.items():
        assert "season_calibration_p10p90" in metric_set, f"{position}/{year} missing season_calibration_p10p90"
        assert "season_calibration_le_p90" in metric_set, f"{position}/{year} missing season_calibration_le_p90"


def test_backtest_run_per_player_results_is_populated(
    synthetic_features_root: Path,
    synthetic_raw_root: Path,
) -> None:
    """per_player_results contains expected columns and at least one row per
    cell that produced predictions+actuals."""
    from projections.backtest import run_backtest

    run = run_backtest(
        held_out_years=(2024,),
        features_root=synthetic_features_root,
        raw_root=synthetic_raw_root,
    )
    assert not run.per_player_results.empty
    expected_columns = {
        "gsis_id", "season", "position", "ruleset", "n_weeks",
        "season_mean", "season_p10", "season_p50", "season_p90",
        "model_id", "generated_at", "actual_season_total",
    }
    actual_columns = set(run.per_player_results.columns)
    missing = expected_columns - actual_columns
    assert not missing, f"per_player_results missing columns: {missing}"
```

(If the existing test file uses different fixture names, substitute them — read the conftest and copy whatever pattern existing tests use for the synthetic feature/raw roots.)

- [ ] **Step 3: Run the tests to verify they pass.**

Run: `source .venv/Scripts/activate && pytest tests/test_backtest/test_harness.py -v`
Expected: PASS — existing harness tests + 2 new ones.

- [ ] **Step 4: Run static checks.**

Run: `source .venv/Scripts/activate && mypy src tests && ruff check src tests && ruff format --check src tests`
Expected: zero violations.

- [ ] **Step 5: Commit.**

```bash
git add tests/test_backtest/test_harness.py
git commit -m "test(backtest): harness produces season metrics + per_player_results"
```

### Task 5b.2: Extend default-on smoke test

**Files:**
- Modify: `tests/backtest/test_backtest_smoke.py`

- [ ] **Step 1: Re-read `tests/backtest/test_backtest_smoke.py`.** Note the existing assertions about which metrics are present; the new assertion is additive.

- [ ] **Step 2: Add an assertion** that the smoke run includes both season metrics, both finite. Append to the test (or extend the existing assertion block — match the existing style):

```python
    # Plan 3d: season-calibration metrics are present and finite for the
    # smoke cell (default-on smoke catches accidental regressions in the
    # season aggregation wiring before a full --run-backtest).
    season_metrics = metrics[
        metrics["metric"].isin(["season_calibration_p10p90", "season_calibration_le_p90"])
    ]
    assert len(season_metrics) == 2, (
        f"Expected 2 season_calibration rows, got {len(season_metrics)}: "
        f"{season_metrics['metric'].tolist()}"
    )
    assert season_metrics["value"].notna().all()
    assert ((season_metrics["value"] >= 0.0) & (season_metrics["value"] <= 1.0)).all()
```

(Replace `metrics` with whatever the existing test uses for the run's metrics frame.)

- [ ] **Step 3: Run the smoke test.**

Run: `source .venv/Scripts/activate && pytest tests/backtest/test_backtest_smoke.py -v`
Expected: PASS — assertion satisfied.

- [ ] **Step 4: Run static checks.**

Run: `source .venv/Scripts/activate && mypy src tests && ruff check src tests && ruff format --check src tests`
Expected: zero violations.

- [ ] **Step 5: Commit.**

```bash
git add tests/backtest/test_backtest_smoke.py
git commit -m "test(backtest): default-on smoke asserts season metrics present + finite"
```

### Task 5b.3: Full test sweep before Phase 6

**Files:**
- (no edits — verification only)

- [ ] **Step 1: Run the entire test suite in default mode (smoke gate runs, full gate skipped).**

Run: `source .venv/Scripts/activate && pytest -v`
Expected: PASS for every test. The full-gate test under `tests/backtest/test_backtest_gate.py` is skipped because `--run-backtest` is not passed.

- [ ] **Step 2: Run all four static checks.**

Run: `source .venv/Scripts/activate && pytest -v && mypy src tests && ruff check src tests && ruff format --check src tests`
Expected: zero violations across all four.

- [ ] **Step 3: No commit (verification only). Phase 6 begins next.**

---

## Phase 6 — Retrain, re-snapshot, gate run, docs

This phase is run-the-scripts mostly; no test code is added. It produces the re-snapshot that absorbs the per-row-seed and SAMPLED_SUMMARY changes.

### Task 6.1: Retrain WR baseline artifact

**Files:**
- Modify: `models/artifacts/baseline-wr-2018-2023-*.joblib` (regenerated)

- [ ] **Step 1: Delete old WR artifact** so the training script's output is unambiguous.

```bash
rm -f models/artifacts/baseline-wr-2018-2023-*.joblib
```

- [ ] **Step 2: Train.**

Run: `source .venv/Scripts/activate && python scripts/train_baseline.py wr`
Expected: prints `model_id: baseline:wr:<NEW_HASH>:2018-2023`, then `Saved artifact: models/artifacts/baseline-wr-2018-2023-<NEW_HASH>.joblib`. The new hash differs from the pre-3d hash because `score_distribution.py` and `models/baseline.py` are both in `code_hash_files`.

- [ ] **Step 3: Verify a new artifact exists.**

```bash
ls -la models/artifacts/baseline-wr-*.joblib
```

Expected: exactly one WR artifact, with the new hash.

### Task 6.2: Retrain QB baseline artifact

**Files:**
- Modify: `models/artifacts/baseline-qb-2018-2023-*.joblib` (regenerated)

- [ ] **Step 1: Delete + retrain QB.**

```bash
rm -f models/artifacts/baseline-qb-2018-2023-*.joblib && \
  source .venv/Scripts/activate && python scripts/train_baseline.py qb
```

Expected: `model_id: baseline:qb:<NEW_HASH>:2018-2023`.

### Task 6.3: Retrain RB baseline artifact

**Files:**
- Modify: `models/artifacts/baseline-rb-2018-2023-*.joblib` (regenerated)

- [ ] **Step 1: Delete + retrain RB.**

```bash
rm -f models/artifacts/baseline-rb-2018-2023-*.joblib && \
  source .venv/Scripts/activate && python scripts/train_baseline.py rb
```

Expected: `model_id: baseline:rb:<NEW_HASH>:2018-2023`.

### Task 6.4: Retrain TE baseline artifact

**Files:**
- Modify: `models/artifacts/baseline-te-2018-2023-*.joblib` (regenerated)

- [ ] **Step 1: Delete + retrain TE.**

```bash
rm -f models/artifacts/baseline-te-2018-2023-*.joblib && \
  source .venv/Scripts/activate && python scripts/train_baseline.py te
```

Expected: `model_id: baseline:te:<NEW_HASH>:2018-2023`.

### Task 6.5: Snapshot drift sanity check

**Files:**
- (no edits — diagnostic only)

- [ ] **Step 1: Run `--check` against the existing snapshot** to capture the size of the drift introduced by the per-row-seed change. This intentionally fails (because the snapshot doesn't yet contain the 32 new season-calibration rows AND because per-row sample regeneration produces slightly different summaries), but the failures should all be drift-within-tolerance regressions on existing metrics PLUS missing rows for the new metrics.

Run: `source .venv/Scripts/activate && python scripts/backtest.py --check 2>&1 | tee /tmp/3d-pre-snapshot-drift.txt`
Expected: FAIL — but failures should be either:
  (a) "missing-baseline" failures for `season_calibration_p10p90` / `season_calibration_le_p90` (the snapshot doesn't have them yet), OR
  (b) drift-within-tolerance failures on weekly metrics (drift comes from per-row seed change, expected to be small).

If any weekly metric drifts BEYOND tolerance, halt — that indicates a correctness issue in the seed derivation or codec, not a benign per-row noise effect. Investigate before proceeding.

- [ ] **Step 2: Capture and review** the drift output. If everything is benign, proceed.

### Task 6.6: Regenerate the snapshot

**Files:**
- Modify: `tests/backtest/baseline_metrics.json` (regenerated)

- [ ] **Step 1: Regenerate snapshot.**

Run: `source .venv/Scripts/activate && python scripts/backtest.py --update-snapshot`
Expected: prints "Previous snapshot: 368 rows" and "New snapshot: 400 rows" (or similar — 16 cells × 2 new metrics = 32 added). Writes `tests/backtest/baseline_metrics.json`.

- [ ] **Step 2: Verify row count.**

```bash
python -c "import json; print(len(json.load(open('tests/backtest/baseline_metrics.json'))))"
```

Expected: 400.

### Task 6.7: Run the full gate

**Files:**
- (no edits — gate verification)

- [ ] **Step 1: Run the opt-in gate against the new snapshot.**

Run: `source .venv/Scripts/activate && pytest -m backtest --run-backtest -v`
Expected: PASS — gate produces metrics that exactly match the just-written snapshot (zero diff because no code changed between `--update-snapshot` and this run).

### Task 6.8: TODO #19 closure check (gate non-determinism)

**Files:**
- (no edits — closure verification)

- [ ] **Step 1: Re-run `--check` immediately** with no code or data changes. With deterministic per-row seeds, this should pass with zero drift.

Run: `source .venv/Scripts/activate && python scripts/backtest.py --check`
Expected: `PASS — 400 metrics within tolerance.`

If any drift appears, the seed derivation has a non-determinism bug; investigate.

### Task 6.9: Update `project_management.md` and `TODO.md`; commit Phase 6

**Files:**
- Modify: `project_management.md` (new "Plan 3d" section at the top)
- Modify: `TODO.md` (close TODO #13, #14, #19; add Plan 3e as next-action)

- [ ] **Step 1: Add the Plan 3d section to `project_management.md`.** Insert at the top (above the existing "Plan 3c" section). Mirror the structure of the Plan 3c section: composite metrics table, naive comparison, decision-log entries, current-status update, next-action recommendation. Use the actual numbers from the `--update-snapshot` output. Skeleton:

```markdown
## Plan 3d — Real Monte Carlo season aggregation (run 2026-04-26, on branch `feat/plan-3d-monte-carlo-season`)

**Closes:** TODO #13 (per-row seeds), TODO #14 (SAMPLED_SUMMARY family), TODO #19 (gate non-determinism by demonstration).

Held-out years: 2021–2024 (same as Plan 3c). Snapshot at 400 rows
(368 weekly metrics from 3c + 32 new season-calibration rows from 3d).
Full gate runtime: <X> seconds.

### Composite metrics by (position, year) — drift from Plan 3c snapshot

| Position | Year | composite_rmse (3d) | composite_rmse (3c) | Δ |
| ... |

### Season-total calibration (new in Plan 3d)

| Position | Year | season_calibration_p10p90 | season_calibration_le_p90 |
| ... |

### Decision log (Plan 3d)

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-26 | params blob format = per-stat distribution params (Section 3.1 of spec) | Three orders of magnitude smaller than persisting full sample arrays; decomposable; deterministic regeneration via seed. |
| 2026-04-26 | Per-row seed = sha256 of `(gsis_id, season, week, ruleset.name)` truncated to 32 bits | Deterministic across processes (Python `hash()` is salt-randomized via PYTHONHASHSEED); independent across rows; reproducible. |
| 2026-04-26 | Aggregator regenerates per-week samples rather than persisting them | Storage 1000x smaller; regeneration is O(seconds); samples are deterministic given seed. |
| 2026-04-26 | Modal-position resolution for traded players | Deterministic; rare edge case; documented in docstring. |
| 2026-04-26 | Calibration tightening (MLE gamma alpha / variance buckets) explicitly deferred to Plan 3e | 3d's snapshot reflects under-dispersed calibration as the regression floor; tightening is a separable model-quality improvement. |

### Current status (as of 2026-04-26)

**Projections Core — Plan 3d (real Monte Carlo season aggregation) merged to `main` at commit `<TBD-after-merge>` (PR #<TBD>).**

### Next action

**Recommended: Plan 3e — calibration tightening.** Replace `_gamma_alpha_from_residuals`'s method-of-moments with an MLE fit, and/or add per-stat residual-variance bucketing by predicted-mean tertile, to move weekly + season calibration coverage toward 0.80.
```

- [ ] **Step 2: Update `TODO.md` — close TODO #13, #14, #19; add Plan 3e.**

Edit `TODO.md`:

- Replace the `### 13. Per-row seed derivation in BaselineModel.predict_distribution` heading body with: `**Closed in Plan 3d.** `derive_row_seed` in `scoring/score_distribution.py` produces a stable 32-bit seed from `(gsis_id, season, week, ruleset.name)` via sha256; `predict_distribution` and `aggregate_to_season` both consume it. Determinism verified by re-running `--check` immediately after `--update-snapshot` in Phase 6 (closes TODO #19 by demonstration).`
- Replace the `### 14. ProjectionWeeklySchema params blob carries summary, not samples` body with: `**Closed in Plan 3d.** New `DistributionFamily.SAMPLED_SUMMARY` enum value; `params` now encodes per-stat distribution parameters via `pack_per_stat_params` (codec in `distributions/codec.py`). Three orders of magnitude smaller than persisting full sample arrays; deterministic regeneration via the per-row seed makes samples available on demand.`
- Replace the `### 19. Walk-forward gate non-determinism check` body with: `**Closed in Plan 3d.** With deterministic per-row seeds, re-running `python scripts/backtest.py --check` immediately after `--update-snapshot` produces zero drift. No `random_state` propagation needed inside `RidgeCV` because the regression itself is deterministic; non-determinism only entered through `score_distribution`'s seed.`
- Add a new section after the closed entries:

```markdown
### 22. Plan 3e — calibration tightening

Plan 3c's snapshot showed weekly calibration coverage at 0.67–0.80 across
all 16 cells (target 0.80). Plan 3d's season-calibration metrics inherit
the same under-dispersion. Plan 3e replaces method-of-moments gamma alpha
with an MLE fit (likely scipy.optimize.minimize on the residual log-likelihood)
and/or adds per-stat residual-variance bucketing by predicted-mean tertile
to capture heteroscedasticity. Validation surface: weekly calibration metrics
move toward 0.80; season-calibration metrics widen accordingly. Re-snapshot
required after Plan 3e ships.
```

- [ ] **Step 3: Commit Phase 6 results.**

```bash
git add models/artifacts/ tests/backtest/baseline_metrics.json project_management.md TODO.md
git commit -m "$(cat <<'EOF'
feat(plan-3d): retrain artifacts + regenerate snapshot

- Retrained all four BaselineModel artifacts (model_id rotates because
  score_distribution.py and models/baseline.py are in code_hash_files).
- Regenerated tests/backtest/baseline_metrics.json: 368 -> 400 rows
  (+32 season-calibration rows; existing rows drift within tolerance).
- Verified gate passes: pytest -m backtest --run-backtest is green.
- Verified determinism: re-run of --check after --update-snapshot
  produces zero drift (closes TODO #19 by demonstration).

Closes TODO #13, #14, #19.
EOF
)"
```

### Task 6.10: Final verification + open PR

**Files:**
- (no edits — handoff)

- [ ] **Step 1: Run the full local verification suite.**

Run: `source .venv/Scripts/activate && pytest -v && mypy src tests && ruff check src tests && ruff format --check src tests`
Expected: zero violations.

- [ ] **Step 2: Run the opt-in gate one last time.**

Run: `source .venv/Scripts/activate && pytest -m backtest --run-backtest -v`
Expected: PASS.

- [ ] **Step 3: Push branch + open PR.**

```bash
git push -u origin feat/plan-3d-monte-carlo-season
gh pr create --title "feat: Plan 3d — real Monte Carlo season aggregation" --body "$(cat <<'EOF'
## Summary
- Closes TODO #13: per-row deterministic seed derivation in `BaselineModel.predict_distribution` via `derive_row_seed(gsis_id, season, week, ruleset.name)` (sha256, 32-bit, cross-process stable).
- Closes TODO #14: `ProjectionWeeklySchema.params` now encodes per-stat distribution parameters via the new `distributions/codec.py`. New `DistributionFamily.SAMPLED_SUMMARY` enum value.
- Closes TODO #19 by demonstration: re-running `--check` immediately after `--update-snapshot` produces zero drift, confirming determinism.
- Adds `aggregate_to_season` (new `src/projections/aggregation/season.py`) and `ProjectionSeasonSchema`. Wires season-calibration metrics into the gate.
- Snapshot grew 368 -> 400 rows (32 new season-calibration rows).
- Calibration tightening explicitly deferred to Plan 3e (filed as TODO #22).

## Test plan
- [x] `pytest -v` (full default suite, including 3d's new tests)
- [x] `mypy src tests` (zero violations, strict)
- [x] `ruff check src tests` (zero violations)
- [x] `ruff format --check src tests` (no drift)
- [x] `pytest -m backtest --run-backtest -v` (full gate, post-3d snapshot, passes)
- [x] Re-run of `--check` after `--update-snapshot` produces zero drift (TODO #19 closure)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR opened on GitHub; URL printed.

---

## Self-review

### Spec coverage check

Walking the spec section-by-section:

- §1.1 Goal "Add `DistributionFamily.SAMPLED_SUMMARY`" → Task 1.1.
- §1.1 Goal "codec helpers in `distributions.py`" → Tasks 1.3, 1.4. Note: implementation lives in `distributions/codec.py` (since `distributions` is a package, not a single file); re-exported through `__init__.py`. This is documented inline in the plan; spec text is consistent.
- §1.1 Goal "`derive_row_seed`" → Task 2.1.
- §1.1 Goal "Rewire `BaselineModel.predict_distribution`" → Tasks 3.1, 3.2.
- §1.1 Goal "New package `src/projections/aggregation/`" → Tasks 4.1, 4.2, 4.3.
- §1.1 Goal "`ProjectionSeasonSchema`" → Task 1.2.
- §1.1 Goal "`compute_season_calibration_metrics`" → Task 5a.2.
- §1.1 Goal "Wire the season aggregator into `backtest/harness.py`" → Task 5a.3.
- §1.1 Goal "Extend `BacktestRun` with `per_player_results`" → Task 5a.3.
- §1.1 Goal "`scripts/backtest.py` writes `season_results.parquet`" → Task 5a.4.
- §1.1 Goal "Phase 6 retrains all four artifacts + regenerates snapshot + updates docs" → Tasks 6.1–6.9.
- §1.1 Goal "TODO #19 closes by demonstration" → Task 6.8.
- §1.2 Non-goal "Calibration tightening deferred to Plan 3e" → Task 6.9 step 2 files TODO #22.
- §3 Detailed design — every numeric/format choice (msgpack with `schema_version=1`, sha256 first-4-bytes, modal position, etc.) is materialized in actual code in the relevant task.
- §4 Phasing — 7 phases, all ≤5 files; matches spec.
- §5 Testing — every test in the spec table is covered by a corresponding task step.
- §6 Risks — Phase 6 step 1 (Task 6.5) is the explicit drift-sanity check that the spec calls for.

No gaps identified.

### Placeholder scan

- No "TBD"/"TODO"/"implement later"/"fill in details" in any code step.
- "Add appropriate error handling" — not used. All error paths are spelled out as explicit `raise ValueError(...)` calls with their messages and matching test assertions.
- All test code is included in full; no "Write tests for the above" stubs.
- Tasks that share patterns (e.g., 6.1–6.4 retraining each position) repeat the commands rather than say "similar to Task 6.1".
- The Plan 3d section in `project_management.md` (Task 6.9 step 1) intentionally has `<TBD-after-merge>` and `PR #<TBD>` placeholders for the merge commit + PR number — these match the established pattern from Plan 3c's section, get filled in by the merge-time hygiene flow, and are not implementation-time placeholders. (Task 6.9 also has a `<X>` placeholder for full-gate runtime that gets filled with the actual measured time during Phase 6.) These are end-of-phase fill-ins, not unspecified work.
- The `project_management.md` table cell placeholder `| ... |` and the `Δ` column rely on the actual numbers being filled in from the `--update-snapshot` output during Phase 6. This is acceptable because the numbers don't exist until the run happens.

### Type / signature consistency

- `derive_row_seed` is keyword-only (`*` in signature) in both Task 2.1 and Task 3.2 (where `predict_distribution` calls it). Same for `aggregate_to_season` in Tasks 4.2, 5a.3.
- `pack_per_stat_params` accepts `Mapping[Stat, Distribution]` in Task 1.3; `predict_distribution` passes a `dict[Stat, Distribution]` in Task 3.2 (which satisfies the Mapping protocol). Consistent.
- `unpack_per_stat_params` returns `dict[Stat, Distribution]` in Task 1.3; `aggregate_to_season` consumes it the same way in Task 4.2. Consistent.
- `BacktestRun` gains `per_player_results: pd.DataFrame` in Task 5a.3 and is referenced by name in Task 5a.4 (`run.per_player_results`) and Task 5b.1 (`run.per_player_results`). Consistent.
- `compute_season_calibration_metrics` returns `dict[str, float]` in Task 5a.2 and is iterated as `(metric_name, value)` in Task 5a.3. Consistent.
- `ProjectionSeasonSchema` columns in Task 1.2 match the columns built in `aggregate_to_season` (Task 4.2): `gsis_id`, `season`, `position`, `ruleset`, `n_weeks`, `season_mean`, `season_p10`, `season_p50`, `season_p90`, `model_id`, `generated_at`. Consistent.

No type drift identified.
