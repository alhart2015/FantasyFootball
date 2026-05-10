# Weather Refined-Unit Family Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Probe three refined-unit weather features (`is_cold_weather`, multi-class surface one-hot, `is_primetime`) as a family at BaselineModel + lgb-nb composite, returning a SIGNAL/NULL family verdict that decides whether to scope a follow-up production-builder plan.

**Architecture:** Extends PR #28's `src/projections/features/weather_features.py` in-place with three new pure-compute helpers and a Phase-0-pinned `_SURFACE_CODES` tuple. Override generator script (`scripts/build_weather_override.py`) extends in place — same output path overwrites PR #28's narrower override. Probe CLI (`scripts/probe_feature_signal.py`) reused unchanged. The `--force-composite` flag is mandatory on lgb-nb runs (PR #22 spec-gap precedent). Production builders (`build_rb_features`, `build_wr_features` after PR #29) consume the existing 4 v1 weather cols and ignore the new ones — backward-compatible by superset.

**Tech Stack:** pandas (pure-pandas computes), pyarrow (parquet I/O via `projections.store.read_partition` + `df.to_parquet`), `zoneinfo` stdlib (UTC → America/New_York for `is_primetime`), pytest, mypy strict, ruff. No new schema, no new ingest.

**Spec:** `docs/superpowers/specs/2026-05-09-weather-refined-unit-probe-design.md`.

**Branch:** `feat/probe-weather-refined`.

---

## File Structure

**Modify:**
- `src/projections/features/weather_features.py` — add `_SURFACE_CODES`, `_SURFACE_COL_NAMES`, `_compute_is_cold_weather`, `_compute_surface_onehot`, `_compute_is_primetime`. Extend `compute_weather_features` to emit the union; extend `build_weather_overrides` return-column list.
- `tests/test_features/test_weather_features.py` — add ~10 unit tests for the 3 new axes (cold-weather boundary + dome + NaN; surface one-hot per code + sum + unseen-code raises + grass-vs-v1 parity; primetime EDT/EST switch + boundary + NaN).
- `scripts/build_weather_override.py` — extend `_print_audit` to print rates for the new features.
- `tests/test_scripts/test_build_weather_override_cli.py` — add a unit test for `_print_audit` against a small synthetic override DataFrame (the existing file is argparse-only; we add one new test rather than refactor existing ones).
- `CONTRIBUTING.md` — append to "Regenerating the weather override" subsection (refined-unit bundle is now ~12 cols; multi-class surface requires re-pinning if `nfl_data_py` introduces new codes).
- `TODO.md` — append a paragraph under #25 with the refined-unit probe verdict and date (Task 9).
- `project_management.md` — add a "Weather Refined-Unit Family Probe" decision-log entry at the top (Task 9).

**Create:**
- `reports/feature_probe_weather_refined_baseline_{augment,swap}.{md,csv}` — 2 baseline probe outputs.
- `reports/feature_probe_weather_refined_lgbnb_{augment,swap}.{md,csv}` — 2 lgb-nb probe outputs.
- `reports/feature_probe_weather_refined_override_audit.md` — hand-written audit from the build-script stdout.
- `reports/feature_probe_weather_refined_summary.md` — hand-written family summary.

**Untouched (deliberately):**
- `src/projections/schemas.py` — `SchedulesSchema` already declares `kickoff` (UTC), `temp`, `surface`, `roof`. No additions.
- `src/projections/features/_shared.py` — `build_game_environment` not extended; this is probe-only.
- `src/projections/backtest/feature_probe.py` — `family_verdict_from_reports` already handles this spec's verdict rule.
- `scripts/probe_feature_signal.py` — reused as-is.
- All per-position `*FeaturesSchema` and `BaselineModel._<POS>_FEATURE_COLUMNS` — probe-only; no production wiring.
- Production builders (`build_rb_features`, `build_wr_features`) — call `attach_weather_features` for the 4 v1 cols; they ignore the new override-only cols. Backward-compatible by superset.

---

## Task 1: Phase 0 — pin `_SURFACE_CODES` from real data

The exact set of distinct `surface` codes drives the multi-class one-hot column list. Read `data/raw/schedules` across 2018–2024 and pin the observed codes as a `Final` tuple. Add a regression test that protects against silent drift.

**Files:**
- Modify: `src/projections/features/weather_features.py:1-25` (imports + module-level constants section).
- Modify: `tests/test_features/test_weather_features.py` (append a new test).

**Precondition:** `data/raw/schedules/season={2018..2024}/part.parquet` must be present locally. If missing, refresh first:

```bash
PYTHONPATH=src python -c "
from pathlib import Path
from projections.ingest.refresh import refresh
refresh(data_root=Path('data'), seasons=range(2018, 2025), tables=['schedules'])
"
```

- [ ] **Step 1: Enumerate codes from data**

Run a one-shot enumeration:

```bash
PYTHONPATH=src python -c "
import pandas as pd
from pathlib import Path
parts = sorted(Path('data/raw/schedules').glob('season=*/part.parquet'))
df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
codes = df['surface'].dropna().unique()
print(sorted(c for c in codes))
print('NaN count:', df['surface'].isna().sum(), 'of', len(df))
"
```

Expected output: a sorted list of distinct codes, e.g. `['a_turf', 'astroplay', 'astroturf', 'dessograss', 'fieldturf', 'grass', 'matrixturf', 'sportturf']`. **Capture the exact list** — it goes verbatim into the `_SURFACE_CODES` tuple in Step 3.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_features/test_weather_features.py`:

```python
def test_surface_codes_tuple_well_formed() -> None:
    """Pinned _SURFACE_CODES tuple is non-empty, contains 'grass', and every
    code is a clean snake-case-able string. Protects against silent drift if a
    future refactor mangles the constant."""
    from projections.features.weather_features import _SURFACE_CODES, _SURFACE_COL_NAMES

    assert len(_SURFACE_CODES) >= 4, "should have at least grass + 3 turf variants"
    assert "grass" in _SURFACE_CODES
    for code in _SURFACE_CODES:
        assert isinstance(code, str)
        assert code == code.lower(), f"{code!r} should be lowercase"
        assert " " not in code, f"{code!r} should not contain spaces"

    # Column names mirror the codes via lower + replace('-', '_').
    assert len(_SURFACE_COL_NAMES) == len(_SURFACE_CODES)
    assert "is_grass" in _SURFACE_COL_NAMES
    for name in _SURFACE_COL_NAMES:
        assert name.startswith("is_")
        assert "-" not in name
```

- [ ] **Step 3: Run test to verify it fails**

```bash
PYTHONPATH=src pytest tests/test_features/test_weather_features.py::test_surface_codes_tuple_well_formed -v
```

Expected: FAIL with `ImportError: cannot import name '_SURFACE_CODES'`.

- [ ] **Step 4: Pin the tuple in `weather_features.py`**

Modify `src/projections/features/weather_features.py` near the top (after the existing module constants at lines 22-25). Replace the literal list below with the **exact codes captured in Step 1**, sorted alphabetically:

```python
# Pinned 2026-05-09 from data/raw/schedules across 2018-2024. Enumeration:
#   sorted(df['surface'].dropna().unique())
# An unseen code at compute time triggers ValueError in _compute_surface_onehot.
_SURFACE_CODES: Final[tuple[str, ...]] = (
    "a_turf",
    "astroplay",
    "astroturf",
    "dessograss",
    "fieldturf",
    "grass",
    "matrixturf",
    "sportturf",
)

_SURFACE_COL_NAMES: Final[tuple[str, ...]] = tuple(
    f"is_{c.lower().replace('-', '_')}" for c in _SURFACE_CODES
)
```

> If your Step 1 output differs from the literal above, **use your output** — the codes here are an example. The point of Phase 0 is that the tuple reflects the real data.

- [ ] **Step 5: Run test to verify it passes**

```bash
PYTHONPATH=src pytest tests/test_features/test_weather_features.py::test_surface_codes_tuple_well_formed -v
```

Expected: PASS.

- [ ] **Step 6: Verification gate**

```bash
mypy src/projections/features/weather_features.py
ruff check src/projections/features/weather_features.py tests/test_features/test_weather_features.py
ruff format --check src/projections/features/weather_features.py tests/test_features/test_weather_features.py
```

Expected: 0 errors on all three.

- [ ] **Step 7: Commit**

```bash
git add src/projections/features/weather_features.py tests/test_features/test_weather_features.py
git commit -m "feat(weather): pin _SURFACE_CODES from data/raw/schedules 2018-2024"
```

---

## Task 2: `_compute_is_cold_weather` + wire into `compute_weather_features`

Pure compute fn: `(temperature_f <= 32.0)` with NaN-preserving Float64 cast, mirroring `is_high_wind`'s pattern at `weather_features.py:73`. Dome rows already fill `temperature_f = 70.0`, so `is_cold_weather` is naturally `0.0` for indoors (semantically correct).

**Files:**
- Modify: `src/projections/features/weather_features.py` — add helper + extend `compute_weather_features` + extend `build_weather_overrides` return-list.
- Modify: `tests/test_features/test_weather_features.py` — add 3 tests.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_features/test_weather_features.py`:

```python
def test_is_cold_weather_boundary_inclusive_at_32() -> None:
    """temp == 32 → 1.0 (boundary inclusive). temp == 33 → 0.0."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows(
        [
            {"week": 1, "home_team": "GB", "away_team": "DET",
             "wind": 5, "temp": 32, "roof": "outdoors", "surface": "grass"},
            {"week": 2, "home_team": "GB", "away_team": "MIN",
             "wind": 5, "temp": 33, "roof": "outdoors", "surface": "grass"},
        ]
    )
    out = compute_weather_features(sch)

    week1 = out.loc[out["week"] == 1]
    week2 = out.loc[out["week"] == 2]
    assert week1["is_cold_weather"].tolist() == [1.0, 1.0]
    assert week2["is_cold_weather"].tolist() == [0.0, 0.0]


def test_is_cold_weather_dome_falls_out_to_zero() -> None:
    """Dome / closed roof fills temperature_f = 70.0, so is_cold_weather = 0.0."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows(
        [
            {"week": 1, "home_team": "MIN", "away_team": "DET",
             "wind": pd.NA, "temp": pd.NA, "roof": "dome", "surface": "fieldturf"},
            {"week": 2, "home_team": "DAL", "away_team": "NYG",
             "wind": pd.NA, "temp": pd.NA, "roof": "closed", "surface": "matrixturf"},
        ]
    )
    out = compute_weather_features(sch)

    assert out["is_cold_weather"].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_is_cold_weather_outdoor_nan_temp_propagates() -> None:
    """Outdoor game with NaN temp → is_cold_weather = NaN."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows(
        [
            {"week": 1, "home_team": "BUF", "away_team": "NYJ",
             "wind": 10, "temp": pd.NA, "roof": "outdoors", "surface": "grass"},
        ]
    )
    out = compute_weather_features(sch)
    assert out["is_cold_weather"].isna().all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src pytest tests/test_features/test_weather_features.py -v -k "cold_weather"
```

Expected: 3 failures with `KeyError: 'is_cold_weather'` (column doesn't exist yet on the output frame).

- [ ] **Step 3: Add the helper + integrate**

Modify `src/projections/features/weather_features.py`. Add the constant near other thresholds (around line 22):

```python
_COLD_WEATHER_TEMP_F = 32.0
```

Add the pure helper above `compute_weather_features` (around line 27):

```python
def _compute_is_cold_weather(temperature_f: pd.Series) -> pd.Series:
    """Float64 boolean: 1.0 if temperature_f <= 32.0, 0.0 if > 32.0, NaN if NaN.

    Mirrors `is_high_wind`'s NaN-preserving threshold pattern. Domes are
    already filled to `temperature_f=70.0` upstream, so this naturally
    produces 0.0 for indoor games.
    """
    return (temperature_f <= _COLD_WEATHER_TEMP_F).astype("Float64")
```

Now extend `compute_weather_features`. Find the section after `is_grass_surface` is computed (around line 75) and insert a single line before the return:

```python
    games["is_cold_weather"] = _compute_is_cold_weather(games["temperature_f"])
```

Update the return-list (around lines 77-87) to include the new column **just after `temperature_f`** so dependent code reads in a natural order:

```python
    return games[
        [
            "season",
            "week",
            "team",
            "wind_speed_mph",
            "is_high_wind",
            "temperature_f",
            "is_cold_weather",
            "is_grass_surface",
        ]
    ].reset_index(drop=True)
```

Update `build_weather_overrides`'s return-list (around lines 167-177) the same way:

```python
    return attached[
        [
            "gsis_id",
            "season",
            "week",
            "position",
            "wind_speed_mph",
            "is_high_wind",
            "temperature_f",
            "is_cold_weather",
            "is_grass_surface",
        ]
    ].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src pytest tests/test_features/test_weather_features.py -v -k "cold_weather"
```

Expected: 3 PASSes.

- [ ] **Step 5: Run the full module test file to verify no regressions**

```bash
PYTHONPATH=src pytest tests/test_features/test_weather_features.py -v
```

Expected: all tests pass (existing PR #28 tests + Task 1's surface-codes test + Task 2's 3 cold-weather tests).

- [ ] **Step 6: Verification gate**

```bash
mypy src/projections/features/weather_features.py tests/test_features/test_weather_features.py
ruff check src/projections/features/weather_features.py tests/test_features/test_weather_features.py
ruff format --check src/projections/features/weather_features.py tests/test_features/test_weather_features.py
```

Expected: 0 errors on all three.

- [ ] **Step 7: Commit**

```bash
git add src/projections/features/weather_features.py tests/test_features/test_weather_features.py
git commit -m "feat(weather): add _compute_is_cold_weather (temp <= 32F) + wire into compute_weather_features"
```

---

## Task 3: `_compute_surface_onehot` + wire into `compute_weather_features`

Multi-class one-hot encoding from `_SURFACE_CODES`. Per-row branches: `1.0` if match, `0.0` if known different code, `NaN` if `surface` is NaN. ValueError on unseen code.

**Files:**
- Modify: `src/projections/features/weather_features.py` — add helper + integrate.
- Modify: `tests/test_features/test_weather_features.py` — add 4 tests.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_features/test_weather_features.py`:

```python
def test_surface_onehot_each_code_produces_correct_column() -> None:
    """Each pinned surface code activates exactly one is_<code> column at 1.0."""
    from projections.features.weather_features import (
        _SURFACE_CODES,
        compute_weather_features,
    )

    rows = [
        {"week": i + 1, "home_team": "KC", "away_team": "BAL",
         "wind": 5, "temp": 70, "roof": "outdoors", "surface": code}
        for i, code in enumerate(_SURFACE_CODES)
    ]
    sch = _make_schedule_rows(rows)
    out = compute_weather_features(sch)

    # For each (week, code) row pair, the matching is_<code> col is 1.0
    # and all others are 0.0.
    for i, code in enumerate(_SURFACE_CODES):
        col = f"is_{code.lower().replace('-', '_')}"
        rows_for_week = out.loc[out["week"] == i + 1]
        assert (rows_for_week[col] == 1.0).all(), f"week {i+1}: {col} should be 1.0"
        for other_code in _SURFACE_CODES:
            if other_code == code:
                continue
            other_col = f"is_{other_code.lower().replace('-', '_')}"
            assert (rows_for_week[other_col] == 0.0).all(), (
                f"week {i+1}: {other_col} should be 0.0 (only {col} should fire)"
            )


def test_surface_onehot_sum_equals_one_on_known_codes_nan_on_unknown() -> None:
    """Sum across all is_<code> cols == 1.0 on rows with known code; == NaN on
    rows with NaN surface."""
    from projections.features.weather_features import (
        _SURFACE_COL_NAMES,
        compute_weather_features,
    )

    sch = _make_schedule_rows(
        [
            {"week": 1, "home_team": "KC", "away_team": "BAL",
             "wind": 5, "temp": 70, "roof": "outdoors", "surface": "grass"},
            {"week": 2, "home_team": "KC", "away_team": "BAL",
             "wind": 5, "temp": 70, "roof": "outdoors", "surface": pd.NA},
        ]
    )
    out = compute_weather_features(sch)

    week1 = out.loc[out["week"] == 1]
    week2 = out.loc[out["week"] == 2]

    # Week 1: sum of all surface bools == 1.0 (exactly one fires).
    surface_cols = list(_SURFACE_COL_NAMES)
    week1_sum = week1[surface_cols].sum(axis=1)
    assert (week1_sum == 1.0).all(), week1[surface_cols].to_string()

    # Week 2: every surface bool is NaN.
    for col in surface_cols:
        assert week2[col].isna().all()


def test_surface_onehot_unseen_code_raises_valueerror() -> None:
    """A surface code outside _SURFACE_CODES raises ValueError. Forces a
    deliberate spec amendment on nfl_data_py upstream changes."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows(
        [
            {"week": 1, "home_team": "KC", "away_team": "BAL",
             "wind": 5, "temp": 70, "roof": "outdoors",
             "surface": "moonrock"},
        ]
    )
    with pytest.raises(ValueError, match=r"unknown surface code\(s\).*moonrock"):
        compute_weather_features(sch)


def test_surface_onehot_is_grass_matches_v1_is_grass_surface_on_known_codes() -> None:
    """On rows where surface is non-NaN, refined `is_grass` equals v1
    `is_grass_surface` row-for-row. Differs only on NaN-surface rows: v1
    fills NaN to 0.0; refined preserves NaN."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows(
        [
            {"week": 1, "home_team": "KC", "away_team": "BAL",
             "wind": 5, "temp": 70, "roof": "outdoors", "surface": "grass"},
            {"week": 2, "home_team": "KC", "away_team": "BAL",
             "wind": 5, "temp": 70, "roof": "outdoors", "surface": "fieldturf"},
            {"week": 3, "home_team": "KC", "away_team": "BAL",
             "wind": 5, "temp": 70, "roof": "outdoors", "surface": pd.NA},
        ]
    )
    out = compute_weather_features(sch)

    # Weeks 1-2 (non-NaN surface): is_grass == is_grass_surface row-for-row.
    nonnan = out.loc[out["week"].isin([1, 2])]
    pd.testing.assert_series_equal(
        nonnan["is_grass"].astype("Float64"),
        nonnan["is_grass_surface"].astype("Float64"),
        check_names=False,
    )

    # Week 3 (NaN surface): is_grass is NaN; is_grass_surface is 0.0.
    week3 = out.loc[out["week"] == 3]
    assert week3["is_grass"].isna().all()
    assert (week3["is_grass_surface"] == 0.0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src pytest tests/test_features/test_weather_features.py -v -k "surface_onehot"
```

Expected: 4 failures (`KeyError: 'is_grass'` etc., or `ImportError` if helpers don't exist yet).

- [ ] **Step 3: Add the helper + integrate**

Add the helper above `compute_weather_features`:

```python
def _compute_surface_onehot(surface: pd.Series) -> pd.DataFrame:
    """Multi-class one-hot from `surface` against `_SURFACE_CODES`.

    Per-row encoding:
        - `1.0` if `surface == <code>` (the matching column)
        - `0.0` if `surface` is a different known code (the non-matching cols)
        - `NaN` if `surface` is NaN (all cols)

    Raises:
        ValueError: surface contains code(s) not in `_SURFACE_CODES`. Forces
            deliberate spec amendment on nfl_data_py upstream changes.
    """
    surface_known_or_nan = surface.isna() | surface.isin(_SURFACE_CODES)
    if not surface_known_or_nan.all():
        unknown_codes = sorted(set(surface.loc[~surface_known_or_nan].dropna()))
        raise ValueError(
            f"unknown surface code(s) {unknown_codes!r} not in _SURFACE_CODES "
            f"({list(_SURFACE_CODES)!r}); update the pinned tuple if upstream added "
            f"a new code, then re-run."
        )

    is_nan_row = surface.isna()
    out = pd.DataFrame(index=surface.index)
    for code, col_name in zip(_SURFACE_CODES, _SURFACE_COL_NAMES, strict=True):
        bool_col = (surface == code).astype("Float64")
        # Mask NaN-surface rows back to NaN so the one-hot preserves missingness.
        bool_col[is_nan_row] = pd.NA
        out[col_name] = bool_col
    return out
```

Extend `compute_weather_features`. Insert before the return statement (after the existing `is_cold_weather` line from Task 2):

```python
    surface_onehot = _compute_surface_onehot(games["surface"])
    for col_name in _SURFACE_COL_NAMES:
        games[col_name] = surface_onehot[col_name]
```

Extend the return-list to include the surface columns. They go **after `is_cold_weather`** and **before `is_grass_surface`** (v1's `is_grass_surface` stays for back-compat with PR #29 production builders):

```python
    return games[
        [
            "season",
            "week",
            "team",
            "wind_speed_mph",
            "is_high_wind",
            "temperature_f",
            "is_cold_weather",
            *_SURFACE_COL_NAMES,
            "is_grass_surface",
        ]
    ].reset_index(drop=True)
```

Apply the same change to `build_weather_overrides`'s return-list:

```python
    return attached[
        [
            "gsis_id",
            "season",
            "week",
            "position",
            "wind_speed_mph",
            "is_high_wind",
            "temperature_f",
            "is_cold_weather",
            *_SURFACE_COL_NAMES,
            "is_grass_surface",
        ]
    ].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src pytest tests/test_features/test_weather_features.py -v -k "surface_onehot"
```

Expected: 4 PASSes.

- [ ] **Step 5: Run the full module test file**

```bash
PYTHONPATH=src pytest tests/test_features/test_weather_features.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Verification gate**

```bash
mypy src/projections/features/weather_features.py tests/test_features/test_weather_features.py
ruff check src/projections/features/weather_features.py tests/test_features/test_weather_features.py
ruff format --check src/projections/features/weather_features.py tests/test_features/test_weather_features.py
```

Expected: 0 errors on all three.

- [ ] **Step 7: Commit**

```bash
git add src/projections/features/weather_features.py tests/test_features/test_weather_features.py
git commit -m "feat(weather): add _compute_surface_onehot multi-class encoding (preserves NaN)"
```

---

## Task 4: `_compute_is_primetime` + wire into `compute_weather_features`

UTC `kickoff` → ET local hour via `zoneinfo.ZoneInfo("America/New_York")` (handles EDT/EST switch automatically). Threshold at hour ≥ 18.0 captures TNF (8:15 ET) / SNF (8:20 ET) / MNF (8:15 ET).

**Files:**
- Modify: `src/projections/features/weather_features.py` — add helper + integrate.
- Modify: `tests/test_features/test_weather_features.py` — add 3 tests.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_features/test_weather_features.py`:

```python
def test_is_primetime_snf_kickoff_in_september() -> None:
    """SNF in September (EDT, UTC-4): 8:20pm ET = 00:20 UTC next day.
    is_primetime = 1.0."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows(
        [
            {"season": 2024, "week": 1, "home_team": "PHI", "away_team": "DAL",
             # Sun 9/8/2024 8:20pm ET (EDT) == Mon 9/9/2024 00:20 UTC.
             "kickoff": pd.Timestamp("2024-09-09 00:20:00", tz="UTC"),
             "wind": 5, "temp": 70, "roof": "outdoors", "surface": "grass"},
        ]
    )
    out = compute_weather_features(sch)
    assert out["is_primetime"].tolist() == [1.0, 1.0]


def test_is_primetime_early_window_in_november_not_primetime() -> None:
    """Sunday 1pm ET in November (EST, UTC-5): 18:00 UTC. is_primetime = 0.0.
    Same wall-clock 1pm ET in September (EDT, UTC-4) is also not primetime —
    test EST switch correctness."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows(
        [
            {"season": 2024, "week": 10, "home_team": "BUF", "away_team": "NYJ",
             # Sun 11/10/2024 1:00pm ET (EST) == 18:00 UTC.
             "kickoff": pd.Timestamp("2024-11-10 18:00:00", tz="UTC"),
             "wind": 5, "temp": 50, "roof": "outdoors", "surface": "grass"},
            {"season": 2024, "week": 1, "home_team": "MIA", "away_team": "JAX",
             # Sun 9/8/2024 1:00pm ET (EDT) == 17:00 UTC.
             "kickoff": pd.Timestamp("2024-09-08 17:00:00", tz="UTC"),
             "wind": 5, "temp": 80, "roof": "outdoors", "surface": "grass"},
        ]
    )
    out = compute_weather_features(sch)
    assert out["is_primetime"].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_is_primetime_nan_kickoff_propagates_nan() -> None:
    """NaN kickoff → NaN is_primetime."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows(
        [
            {"season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL",
             "kickoff": pd.NaT, "wind": 5, "temp": 70,
             "roof": "outdoors", "surface": "grass"},
        ]
    )
    out = compute_weather_features(sch)
    assert out["is_primetime"].isna().all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src pytest tests/test_features/test_weather_features.py -v -k "primetime"
```

Expected: 3 failures with `KeyError: 'is_primetime'`.

- [ ] **Step 3: Add the helper + integrate**

Add `from zoneinfo import ZoneInfo` to the imports section near the top of `weather_features.py`. Add the constants near other thresholds:

```python
_PRIMETIME_HOUR_ET = 18.0
_KICKOFF_TZ = ZoneInfo("America/New_York")
```

Add the pure helper above `compute_weather_features`:

```python
def _compute_is_primetime(kickoff_utc: pd.Series) -> pd.Series:
    """Float64 boolean: 1.0 if local-ET kickoff hour >= 18.0, 0.0 if <, NaN if NaT.

    Converts UTC to America/New_York via stdlib zoneinfo (handles EDT/EST
    switch automatically across the Sep-Feb season span). Uses the local
    hour + minute/60 to support fractional hours (e.g., 8:20pm = 20.333).
    """
    if not isinstance(kickoff_utc.dtype, pd.DatetimeTZDtype):
        # Already-naive timestamps would silently mis-convert; force the
        # caller to pass a UTC-aware Series (matches SchedulesSchema).
        raise TypeError(
            f"kickoff must be timezone-aware UTC Series, got dtype={kickoff_utc.dtype!r}"
        )
    local = kickoff_utc.dt.tz_convert(_KICKOFF_TZ)
    hour_frac = local.dt.hour + local.dt.minute / 60.0
    out = (hour_frac >= _PRIMETIME_HOUR_ET).astype("Float64")
    out[kickoff_utc.isna()] = pd.NA
    return out
```

Extend `compute_weather_features`. Insert before the return statement (after the surface_onehot loop from Task 3):

```python
    games["is_primetime"] = _compute_is_primetime(games["kickoff"])
```

You also need to make sure `kickoff` is in the `cols` list at the top of `compute_weather_features` (around line 54). Find:

```python
    cols = ["season", "week", "wind", "temp", "roof", "surface"]
```

Replace with:

```python
    cols = ["season", "week", "wind", "temp", "roof", "surface", "kickoff"]
```

Extend the return-list of `compute_weather_features` to put `is_primetime` at the end (after surface bools and `is_grass_surface`):

```python
    return games[
        [
            "season",
            "week",
            "team",
            "wind_speed_mph",
            "is_high_wind",
            "temperature_f",
            "is_cold_weather",
            *_SURFACE_COL_NAMES,
            "is_grass_surface",
            "is_primetime",
        ]
    ].reset_index(drop=True)
```

Apply the same change to `build_weather_overrides`'s return-list.

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src pytest tests/test_features/test_weather_features.py -v -k "primetime"
```

Expected: 3 PASSes.

- [ ] **Step 5: Run the full module test file**

```bash
PYTHONPATH=src pytest tests/test_features/test_weather_features.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Run schedules + ingest tests to confirm no upstream regression**

```bash
PYTHONPATH=src pytest -v -k "ingest or store or schemas or schedules"
```

Expected: all green. Per CLAUDE.md mandate: "For tasks that touch a pandera schema or any ingest/store path: run `pytest -v -k 'ingest or store or schemas'` even if your change is elsewhere."

- [ ] **Step 7: Verification gate**

```bash
mypy src/projections/features/weather_features.py tests/test_features/test_weather_features.py
ruff check src/projections/features/weather_features.py tests/test_features/test_weather_features.py
ruff format --check src/projections/features/weather_features.py tests/test_features/test_weather_features.py
```

Expected: 0 errors on all three.

- [ ] **Step 8: Commit**

```bash
git add src/projections/features/weather_features.py tests/test_features/test_weather_features.py
git commit -m "feat(weather): add _compute_is_primetime (kickoff_hour_et >= 18, EDT/EST aware)"
```

---

## Task 5: Update `scripts/build_weather_override.py` audit + CLI tests

The override generator needs two updates: extend `_print_audit` to print rates for the new features, and update existing CLI tests to assert the extended column set.

**Files:**
- Modify: `scripts/build_weather_override.py:131-163` (`_print_audit` body).
- Modify: `tests/test_scripts/test_build_weather_override_cli.py` (add `_print_audit` unit test).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scripts/test_build_weather_override_cli.py`:

```python
def test_print_audit_includes_refined_unit_rates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`_print_audit` prints rates for the refined-unit columns
    (`is_cold_weather`, per-surface bools, `is_primetime`) in addition
    to the v1 rates. Direct unit test on a small synthetic frame —
    avoids the wall-time of a full main() integration test."""
    from scripts.build_weather_override import _print_audit
    from projections.features.weather_features import _SURFACE_COL_NAMES

    n = 4
    overrides = pd.DataFrame(
        {
            "wind_speed_mph": [0.0, 5.0, 25.0, 0.0],
            "is_high_wind": [0.0, 0.0, 1.0, 0.0],
            "temperature_f": [70.0, 30.0, 50.0, 70.0],
            "is_cold_weather": [0.0, 1.0, 0.0, 0.0],
            "is_grass_surface": [0.0, 1.0, 1.0, 0.0],
            "is_primetime": [0.0, 0.0, 1.0, 0.0],
            **{col: [0.0, 0.0, 0.0, 0.0] for col in _SURFACE_COL_NAMES},
        }
    )
    overrides["is_grass"] = [0.0, 1.0, 1.0, 0.0]

    schedules = pd.DataFrame(
        {
            "roof": pd.array(
                ["dome", "outdoors", "outdoors", "closed"],
                dtype=pd.StringDtype("pyarrow"),
            ),
        }
    )

    _print_audit(overrides, schedules)
    captured = capsys.readouterr()

    assert "weather override audit (4 rows)" in captured.out
    assert "is_cold_weather" in captured.out
    assert "is_primetime" in captured.out
    assert "is_grass" in captured.out
    # v1 lines still printed.
    assert "is_high_wind=1.0 rate (v1)" in captured.out
    assert "is_grass_surface=1.0 rate (v1)" in captured.out
    # Refined lines printed.
    assert "is_cold_weather=1.0 rate (refined)" in captured.out
    assert "is_primetime=1.0 rate (refined)" in captured.out
```

You may need to add `import pandas as pd` and `import pytest` to the test file if not already present.

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src pytest tests/test_scripts/test_build_weather_override_cli.py::test_print_audit_includes_refined_unit_rates -v
```

Expected: FAIL — `_print_audit` doesn't yet print the new lines.

- [ ] **Step 3: Extend `_print_audit`**

Replace the body of `_print_audit` in `scripts/build_weather_override.py:131-163`:

```python
def _print_audit(overrides: pd.DataFrame, schedules: pd.DataFrame) -> None:
    """Print audit numbers for `reports/feature_probe_weather_refined_override_audit.md`.

    Numbers reported (extended for refined-unit bundle):
        - Pooled dome / closed-roof game share (% of games).
        - Outdoor-NaN rate per weather feature (% of override rows).
        - Pooled is_high_wind rate (% of override rows where True).
        - Pooled is_cold_weather rate (% of override rows where True).
        - Pooled is_grass_surface rate (v1, % where True).
        - Per-surface rate from _SURFACE_COL_NAMES (refined multi-class).
        - Pooled is_primetime rate (% of override rows where True).
    """
    from projections.features.weather_features import _SURFACE_COL_NAMES

    n = len(overrides)
    is_indoor = schedules["roof"].isin(["dome", "closed"]).fillna(False)
    n_indoor_games = int(is_indoor.sum())
    n_total_games = len(schedules)
    indoor_pct = (n_indoor_games / n_total_games * 100.0) if n_total_games else 0.0

    nan_cols = (
        "wind_speed_mph",
        "is_high_wind",
        "temperature_f",
        "is_cold_weather",
        "is_grass_surface",
        "is_primetime",
        *_SURFACE_COL_NAMES,
    )
    nan_rates = {col: overrides[col].isna().mean() * 100.0 for col in nan_cols}

    rate_cols_v1 = ("is_high_wind", "is_grass_surface")
    rate_cols_refined = ("is_cold_weather", "is_primetime", *_SURFACE_COL_NAMES)

    print(f"weather override audit ({n} rows):")
    print(f"  indoor games (dome+closed): {n_indoor_games}/{n_total_games} = {indoor_pct:.1f}%")
    for col, pct in nan_rates.items():
        print(f"  {col} NaN rate: {pct:.2f}%")
    for col in rate_cols_v1:
        rate = overrides[col].fillna(0.0).mean() * 100.0
        print(f"  {col}=1.0 rate (v1): {rate:.2f}%")
    for col in rate_cols_refined:
        rate = overrides[col].fillna(0.0).mean() * 100.0
        print(f"  {col}=1.0 rate (refined): {rate:.2f}%")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src pytest tests/test_scripts/test_build_weather_override_cli.py::test_print_audit_includes_refined_unit_rates -v
```

Expected: PASS.

- [ ] **Step 5: Run all CLI tests to verify no existing test regressed**

```bash
PYTHONPATH=src pytest tests/test_scripts/test_build_weather_override_cli.py -v
```

Expected: all green (4 existing argparse tests + 1 new audit test).

- [ ] **Step 6: Verification gate**

```bash
mypy scripts/build_weather_override.py tests/test_scripts/test_build_weather_override_cli.py
ruff check scripts/build_weather_override.py tests/test_scripts/test_build_weather_override_cli.py
ruff format --check scripts/build_weather_override.py tests/test_scripts/test_build_weather_override_cli.py
```

Expected: 0 errors on all three.

- [ ] **Step 7: Commit**

```bash
git add scripts/build_weather_override.py tests/test_scripts/test_build_weather_override_cli.py
git commit -m "feat(weather): extend build_weather_override audit for refined-unit bundle"
```

---

## Task 6: Update CONTRIBUTING.md "Regenerating the weather override" subsection

PR #28 added this subsection to document the override-generation workflow. The bundle is now ~12 cols (vs 4); document the multi-class surface re-pinning concern.

**Files:**
- Modify: `CONTRIBUTING.md` — locate "Regenerating the weather override" subsection.

- [ ] **Step 1: Find the existing subsection**

```bash
grep -n "Regenerating the weather override" CONTRIBUTING.md
```

Note the line number.

- [ ] **Step 2: Replace the subsection**

Replace the entire subsection content with:

```markdown
#### Regenerating the weather override

The weather refined-unit bundle (~12 columns: 4 v1 + 1 cold + ~7 surface multi-class + 1 primetime) is regenerated from `data/raw/schedules` via:

```
PYTHONPATH=src python -m scripts.build_weather_override --seasons 2018-2024 --force
```

Output goes to `data/features_probe/weather.parquet`. The script prints audit numbers (dome rate, per-column NaN rate, per-surface rates, primetime rate) to stdout.

**Multi-class surface re-pinning.** The `_SURFACE_CODES` tuple in `src/projections/features/weather_features.py` is pinned from observed codes in 2018–2024 data. If `nfl_data_py` introduces a new surface code in future seasons, the override generator raises `ValueError: unknown surface code(s) ...` rather than silently dropping it. Recover by:

1. Re-run the Phase 0 enumeration: `python -c "import pandas as pd; from pathlib import Path; df = pd.concat([pd.read_parquet(p) for p in sorted(Path('data/raw/schedules').glob('season=*/part.parquet'))], ignore_index=True); print(sorted(df['surface'].dropna().unique()))"`.
2. Update `_SURFACE_CODES` to add the new code in the right alphabetical position.
3. Re-run the test suite + override generator.
```

- [ ] **Step 3: Verify the diff**

```bash
git diff CONTRIBUTING.md
```

Confirm only the targeted subsection changed.

- [ ] **Step 4: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs(contributing): update weather override subsection for refined-unit bundle"
```

---

## Task 7: Generate override + write audit report

Run the build script against real data, capture audit numbers, hand-write the audit report. This is the "Phase 4: Real Execution" step — moves from synthetic-fixture verification to production-data exercise.

**Files:**
- Generate (not committed; reproducible): `data/features_probe/weather.parquet` (~56k rows × ~14 cols).
- Create: `reports/feature_probe_weather_refined_override_audit.md`.

- [ ] **Step 1: Confirm data is fresh**

```bash
ls data/raw/schedules/season=*/part.parquet | wc -l
```

Expected: 7 (2018–2024). If fewer, refresh first per Task 1's preconditions.

- [ ] **Step 2: Generate the override**

```bash
PYTHONPATH=src python -m scripts.build_weather_override --seasons 2018-2024 --force 2>&1 | tee /tmp/weather_refined_audit.txt
```

Expected output: `wrote N rows to data/features_probe/weather.parquet` followed by the audit block (~20 lines: indoor pct, per-col NaN rates, per-feature rates).

- [ ] **Step 3: Verify the parquet shape**

```bash
PYTHONPATH=src python -c "
import pandas as pd
df = pd.read_parquet('data/features_probe/weather.parquet')
print(df.shape)
print(sorted(df.columns.tolist()))
print(df.dtypes)
"
```

Expected: ~56652 rows; columns include `gsis_id`, `season`, `week`, `position`, `wind_speed_mph`, `is_high_wind`, `temperature_f`, `is_cold_weather`, all `_SURFACE_COL_NAMES`, `is_grass_surface`, `is_primetime`. All feature cols are nullable Float64.

- [ ] **Step 4: Write the audit report**

Create `reports/feature_probe_weather_refined_override_audit.md` with the captured numbers:

```markdown
# Weather Refined-Unit Override Audit — 2026-05-09

**Override path:** `data/features_probe/weather.parquet`
**Generator:** `scripts/build_weather_override.py` (PR `feat/probe-weather-refined`)
**Seasons:** 2018–2024
**Spec:** `docs/superpowers/specs/2026-05-09-weather-refined-unit-probe-design.md`

## Pooled audit (from generator stdout)

```
[paste the captured audit block from /tmp/weather_refined_audit.txt verbatim]
```

## Per-(season, position) coverage

[Compute manually with the snippet below or paste output]:

```python
import pandas as pd
df = pd.read_parquet('data/features_probe/weather.parquet')
for col in ['is_cold_weather', 'is_primetime']:
    print(f'\n=== {col} non-NaN rate ===')
    print(df.groupby(['season', 'position'])[col].apply(
        lambda s: 1 - s.isna().mean()
    ).unstack().round(3).to_markdown())
```

## Notes

- Indoor (dome + closed) rate ~30% — consistent with PR #28 audit.
- Outdoor `temp` / `wind` NaN rate inherited from PR #28 (~8% across the 2018–2024 span; concentrated in 2018–2019).
- `is_primetime` rate ~12–15% expected (TNF + SNF + MNF + occasional Saturday).
- Multi-class surface rate distribution: `is_grass` ~50% (matches v1 `is_grass_surface`); modern turfs (`is_fieldturf`, `is_matrixturf`, `is_sportturf`) split the remainder; legacy codes (`is_a_turf`, `is_astroturf`, `is_astroplay`, `is_dessograss`) appear only in older seasons.

## Coverage caveat

Per PR #29's coverage caveat (which corrected PR #28's overstated "uniformly ≥92%" claim), report per-(position, season) coverage separately rather than pooled. Trough seasons may dip below the pooled rate; audit confirms this is symmetric across baseline + candidate sides under the probe's left-merge join.
```

Replace the `[paste ...]` and `[Compute manually ...]` blocks with the actual captured / computed content.

- [ ] **Step 5: Commit**

```bash
git add reports/feature_probe_weather_refined_override_audit.md
git commit -m "report(weather-refined): override audit (X rows, dome rate Y%, primetime rate Z%)"
```

(Replace X, Y, Z with the real numbers from Step 4.)

---

## Task 8: Run the 4 probes + write summary report

Run augment + swap × baseline + lgb-nb × `--force-composite` against the generated override. Capture the 4 reports, then synthesize the family verdict per spec §1.2 decoding.

**Files:**
- Generate (committed): `reports/feature_probe_weather_refined_baseline_{augment,swap}.{md,csv}` (4 files).
- Generate (committed): `reports/feature_probe_weather_refined_lgbnb_{augment,swap}.{md,csv}` (4 files).
- Create: `reports/feature_probe_weather_refined_summary.md`.

**Probe CLI conventions** (per `scripts/probe_feature_signal.py:141-229`): markdown report goes to **stdout** — redirect with `>` to capture. CSV via `--csv-out`. Mode is determined by `--drop`: augment = `--override` only; swap = `--override` + `--drop "col1,col2,..."`. The probe reads the override parquet's candidate columns automatically (no `--override-cols` flag exists). Defaults: `--seasons 2018-2024`, `--holdout-years 2021-2024` — both match this spec, no override needed.

For **swap mode**, drop the v1 weather cols (4 cols are present in `RbFeaturesSchema` + `WrFeaturesSchema` after PR #29; `QbFeaturesSchema` + `TeFeaturesSchema` don't have them, so the drop is a no-op for QB/TE — same shape as PR #28's degenerate-swap precedent on those positions):

```
V1_WEATHER_COLS=wind_speed_mph,is_high_wind,temperature_f,is_grass_surface
```

- [ ] **Step 1: Run BaselineModel × augment**

```bash
PYTHONPATH=src python -m scripts.probe_feature_signal \
    --candidate-name weather_refined_baseline_augment \
    --override data/features_probe/weather.parquet \
    --model baseline \
    --coverage-threshold 0.90 \
    --csv-out reports/feature_probe_weather_refined_baseline_augment.csv \
    > reports/feature_probe_weather_refined_baseline_augment.md
```

Expected: probe completes (~5-10 min); both files written; final stdout/markdown ends with a verdict summary table.

- [ ] **Step 2: Run BaselineModel × swap**

```bash
PYTHONPATH=src python -m scripts.probe_feature_signal \
    --candidate-name weather_refined_baseline_swap \
    --override data/features_probe/weather.parquet \
    --drop "wind_speed_mph,is_high_wind,temperature_f,is_grass_surface" \
    --model baseline \
    --coverage-threshold 0.90 \
    --csv-out reports/feature_probe_weather_refined_baseline_swap.csv \
    > reports/feature_probe_weather_refined_baseline_swap.md
```

- [ ] **Step 3: Run lgb-nb × augment with `--force-composite`**

```bash
PYTHONPATH=src python -m scripts.probe_feature_signal \
    --candidate-name weather_refined_lgbnb_augment \
    --override data/features_probe/weather.parquet \
    --model lightgbm-nb \
    --force-composite \
    --coverage-threshold 0.90 \
    --csv-out reports/feature_probe_weather_refined_lgbnb_augment.csv \
    > reports/feature_probe_weather_refined_lgbnb_augment.md
```

Expected wall time: ~15–30 minutes. The `--model` value is `lightgbm-nb` (not `lgbnb` — see `_VALID_MODELS` in the probe script).

- [ ] **Step 4: Run lgb-nb × swap with `--force-composite`**

```bash
PYTHONPATH=src python -m scripts.probe_feature_signal \
    --candidate-name weather_refined_lgbnb_swap \
    --override data/features_probe/weather.parquet \
    --drop "wind_speed_mph,is_high_wind,temperature_f,is_grass_surface" \
    --model lightgbm-nb \
    --force-composite \
    --coverage-threshold 0.90 \
    --csv-out reports/feature_probe_weather_refined_lgbnb_swap.csv \
    > reports/feature_probe_weather_refined_lgbnb_swap.md
```

- [ ] **Step 5: Inspect each report's verdict**

For each of the 4 markdown reports, scan the per-stat verdict table (Phase 1) and the composite verdict block (Phase 2). Note for the summary report:

- Phase 1 SIGNAL count per (position, target_stat) cell.
- Phase 2 ADOPT / MARGINAL / DO_NOT_ADOPT verdict per position.
- Any cell with `REGRESSION` (CI strictly worse than zero).

Pay attention to the QB augment cells specifically — PRs #23, #24, #25, #28 each saw QB augment regress (or directionally regress) on context / team / trajectory adds. If the same pattern recurs here, document it.

The family verdict is **SIGNAL** if any Phase 2 cell is ADOPT (in any of the 4 reports); **NULL durable** otherwise. Refined-unit decoding (per spec §1.2):

- swap ADOPT (any model/position) → strict refinement available.
- augment-only ADOPT → additive refinement.
- All NULL → close at this cut.

Mirrors PR #28's manual verdict synthesis — the `family_verdict_from_reports` helper takes `ProbeReport` objects (not markdown paths), so we inspect the rendered reports directly.

- [ ] **Step 6: Write the summary report**

Create `reports/feature_probe_weather_refined_summary.md`. Use PR #28's `reports/feature_probe_weather_summary.md` as the template. The summary must include:

1. **Verdict line** — `SIGNAL` or `NULL`, plus the binding cell(s).
2. **Per-mode table** — 4 rows × 4 positions × Phase-1 (per-stat pooled) + Phase-2 (composite) verdicts.
3. **Refined-unit-specific decoding** per spec §1.2: which mode×model combination(s) bind; what integration shape the verdict greenlights:
   - swap ADOPT + augment ADOPT → strict refinement (replace v1 weather cols).
   - swap NULL + augment ADOPT → additive refinement.
   - swap ADOPT + augment NULL → replace v1 (signal lives in refined cols only).
   - All NULL → close refined-unit at this cut.
4. **Mechanism annotation:** did `is_cold_weather` fire on RB? Did multi-class surface fire on RB+WR (where binary `is_grass_surface` already did in PR #28)? Did `is_primetime` fire anywhere?
5. **Coverage note** — confirm per-(position, season) ≥ 0.90 on the eval window; relax to 0.80 if coverage drops unexpectedly (and document why).
6. **Recurring QB augment regression check** — per PR #23/#24/#25/#28 pattern, does the probe see QB augment regress (CI strictly above 0)? If yes, document.
7. **Refined-unit-of-refined-unit candidates left unexplored** — continuous kickoff hour, `is_london`, surface×position interactions, per-team weather acclimation, precipitation, wind direction.
8. **Predecessor pointers** — PR #28 broad-cut probe, PR #29 RB+WR integration, this spec.

- [ ] **Step 7: Commit reports**

```bash
git add reports/feature_probe_weather_refined_baseline_augment.md \
        reports/feature_probe_weather_refined_baseline_augment.csv \
        reports/feature_probe_weather_refined_baseline_swap.md \
        reports/feature_probe_weather_refined_baseline_swap.csv \
        reports/feature_probe_weather_refined_lgbnb_augment.md \
        reports/feature_probe_weather_refined_lgbnb_augment.csv \
        reports/feature_probe_weather_refined_lgbnb_swap.md \
        reports/feature_probe_weather_refined_lgbnb_swap.csv \
        reports/feature_probe_weather_refined_summary.md
git commit -m "report(weather-refined): family probe — verdict <SIGNAL|NULL>"
```

(Replace `<SIGNAL|NULL>` with the actual verdict from Step 6.)

---

## Task 9: PM/TODO update + final verification gates

Captures the family verdict in the running project log and runs the final end-of-effort verification per CLAUDE.md.

**Files:**
- Modify: `project_management.md` (top section).
- Modify: `TODO.md` (under TODO #25).

- [ ] **Step 1: Add PM entry**

Insert at the top of `project_management.md` (after the line `Running log of project status...`), above the most-recent existing entry:

```markdown
---

## Weather Refined-Unit Family Probe — verdict **<SIGNAL|NULL>** (2026-05-09, on branch `feat/probe-weather-refined`)

**Status:** Probe-only spec shipped per `docs/superpowers/specs/2026-05-09-weather-refined-unit-probe-design.md` and plan `docs/superpowers/plans/2026-05-09-weather-refined-unit-probe.md`. Implements three refined-unit weather features (`is_cold_weather`, multi-class surface one-hot, `is_primetime`) on top of PR #28's `weather_features.py` module. No new ingest, no schema changes; production builders unchanged (PR #29's RB+WR integration consumes the v1 4-col subset and ignores the new override-only cols).

**Verdict:** **<SIGNAL|NULL>** per spec §1.3 criterion 3 (BaselineModel + lgb-nb composite via `--force-composite`).

[If SIGNAL:]
- ADOPT cells: <list per (model, mode, position) cell — e.g. `(lgb-nb augment, RB) -0.0XXX fpts CI [..]`>
- Refined-unit decoding: <strict refinement | additive refinement | replace-only | closed-at-this-cut> per spec §1.2.
- Greenlights: <follow-up integration plan shape>.

[If NULL:]
- Closes the refined-unit family at this cut. Refined-unit-of-refined-unit candidates (continuous kickoff hour, `is_london`, surface×position interactions, per-team weather acclimation) remain open under TODO #25.

**Coverage:** `--coverage-threshold 0.90` (matching PR #28's relaxation). Per-(position, season) coverage <PASTE FROM AUDIT REPORT>.

**Recurring QB augment regression check:** <document yes/no, per PR #23/#24/#25/#28 pattern>.

**Reports:** `reports/feature_probe_weather_refined_summary.md`, `reports/feature_probe_weather_refined_override_audit.md`, 4 per-(model, mode) `.md`/`.csv` files.

**What this closes:** TODO #25's third refined-unit candidate (kickoff hour) plus the multi-class-surface and cold-weather candidates, all at the in-builder-bundle unit. Any remaining TODO #25 work is at the refined-unit-of-refined-unit level.

---
```

Fill in `<SIGNAL|NULL>` and the cell-level details from the summary report.

- [ ] **Step 2: Update TODO #25**

Append to TODO #25 in `TODO.md`, after the existing PR #29 paragraph:

```markdown
**Update 2026-05-09 (refined-unit family probe, branch `feat/probe-weather-refined`):** Refined-unit family probe shipped per `docs/superpowers/specs/2026-05-09-weather-refined-unit-probe-design.md`. Bundle: `is_cold_weather` (temp ≤ 32°F), multi-class surface one-hot (~7 cols pinned from data 2018-2024), `is_primetime` (kickoff_hour_et ≥ 18). **Family verdict: `<SIGNAL|NULL>`** — <one-line summary per the PM entry above>. Refined-unit-of-refined-unit candidates remain unexplored: continuous kickoff hour, `is_london`, surface×position interactions, per-team weather acclimation, precipitation (would require new ingest), wind direction (would require new ingest). None queued. See `reports/feature_probe_weather_refined_summary.md`.
```

- [ ] **Step 3: Final verification gates per CLAUDE.md**

Run all four:

```bash
PYTHONPATH=src pytest -v
```

Expected: all green (838+ tests). Note any new tests added by this plan are part of the count.

```bash
mypy src tests scripts
```

Expected: 0 violations across all three roots.

```bash
ruff check src tests scripts
```

Expected: 0 violations.

```bash
ruff format --check src tests scripts
```

Expected: 0 drift.

Plus the schema-seam mandate:

```bash
PYTHONPATH=src pytest -v -k "ingest or store or schemas"
```

Expected: all green.

- [ ] **Step 4: Commit PM/TODO updates**

```bash
git add project_management.md TODO.md
git commit -m "docs(pm,todo): record weather refined-unit family probe verdict — <SIGNAL|NULL>"
```

- [ ] **Step 5: Push branch + open PR**

```bash
git push -u origin feat/probe-weather-refined
gh pr create --title "feat(weather): refined-unit family probe — <SIGNAL|NULL>" --body "$(cat <<'EOF'
## Summary

- Refined-unit family probe extending PR #28's `weather_features.py` with three new feature axes: `is_cold_weather` (temp ≤ 32°F sibling to `is_high_wind`), multi-class surface one-hot (replaces v1 binary `is_grass_surface`), `is_primetime` (kickoff_hour_et ≥ 18, EDT/EST aware).
- Verdict: <SIGNAL|NULL> per spec §1.3.
- No production schema / builder changes; backward-compatible by superset (PR #29's RB+WR builders consume the v1 4-col subset, ignore new override-only cols).
- First refined-unit family probe in the project; verdict greenlights <follow-up integration plan | closes refined-unit at this cut>.

## Test plan

- [x] All weather feature unit tests (~10 new) pass against synthetic fixtures.
- [x] `pytest -v -k "ingest or store or schemas"` green.
- [x] mypy strict + ruff + ruff format clean.
- [x] 4 probe runs complete (augment + swap × baseline + lgb-nb composite).
- [x] Override audit confirms ~12-col bundle, expected coverage profile.
- [x] Family verdict computed via `family_verdict_from_reports`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Replace `<SIGNAL|NULL>` and the verdict-conditional bullet with the real verdict.

---

## End-of-plan checklist

After all 9 tasks complete, confirm:

- [ ] `_SURFACE_CODES` + `_SURFACE_COL_NAMES` in `weather_features.py` reflect real 2018-2024 data.
- [ ] All 10+ new unit tests pass on synthetic fixtures (Tasks 1–4).
- [ ] CLI test asserts the extended audit output (Task 5).
- [ ] `data/features_probe/weather.parquet` regenerated with ~12 cols (Task 7).
- [ ] 4 probe reports written under `reports/feature_probe_weather_refined_*` (Task 8).
- [ ] `reports/feature_probe_weather_refined_summary.md` documents the family verdict, refined-unit decoding (per spec §1.2), and recurring-QB-augment-regression check (Task 8).
- [ ] PM + TODO updated with the verdict (Task 9).
- [ ] Final `pytest -v` + mypy + ruff all green (Task 9).
- [ ] PR opened with the verdict in the title (Task 9).
