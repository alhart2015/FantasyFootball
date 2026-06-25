# RB feature lift — trajectory + Vegas-preseason probe → conditional integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cheaply determine whether RB can be lifted off the `baseline` model by adding the trajectory and/or Vegas-preseason feature families, and ship the lift only if a per-stat probe SIGNALs and the adoption gate ADOPTs.

**Architecture:** Repo's established **probe → integrate → adoption-gate** discipline. Phase 0 persists the baseline-to-beat. Phase 1 runs the cheap signal probe (no source changes) and hits decision gate **G1**. Only on SIGNAL do Phases 2–3 integrate the family, re-evaluate via a non-destructive dual-run adoption gate + model-class bake-off (**G2**), and — only on ADOPT — flip the default and re-run the DFS edge study.

**Tech Stack:** Python, pandas, pandera, scikit-learn (RidgeCV), lightgbm; existing `projections` package (`features/`, `models/`, `backtest/`); pytest + mypy strict + ruff.

## Global Constraints

- **`GsisId` canonical;** all joins/keys on `gsis_id`. Never join on names.
- **Reference enums, never raw strings** (`Position.RB`, `Stat.RUSHING_YARDS`, `DistributionFamily.*`).
- **`df = SCHEMA.validate(df)` with reassignment** at every module boundary producing a DataFrame.
- **Nullable dtypes:** `pd.StringDtype("pyarrow")` for nullable strings, `pd.Int64Dtype()` for nullable ints. New RB feature columns are `Series[float]` nullable per the WR precedent (`schemas.py:604-607`) — do **not** use bool/int for `age`/`is_rookie`.
- **`store.write_partition`/`read_partition`** are the only sanctioned parquet I/O for the cache.
- **End-of-effort checklist (CLAUDE.md §4)** on every task that touches source: `pytest -v` (relevant subset OK, state it), `mypy src tests` (0), `ruff check src tests` (0), `ruff format --check src tests` (0). For schema/ingest/store/cache touches also run `pytest -v -k "ingest or store or schemas"`.
- **No broad `# type: ignore` / `# noqa`.** Narrow + comment if unavoidable.
- **Venv quirk:** prepend `.venv/Scripts` to PATH before `git commit` (pre-commit mypy hook resolves to system Python otherwise).
- **Spec:** `docs/superpowers/specs/2026-06-25-rb-trajectory-vegas-features-design.md`.

## Execution model — decision gates (READ FIRST)

This plan is **gated**, not purely linear. Stop points:

- **Gate G1 (after Task 3).** If *both* families probe NULL → execute **Task 4 (STOP write-up)** and **END the plan**. Do not start Task 5. If *either* family SIGNALs → skip Task 4, proceed to Phase 2 for the signaling family(ies) only.
- **Gate G2 (after Task 9).** If the dual-run adoption gate does **not** ADOPT → execute **Task 10a (revert + STOP write-up)** and END. If it ADOPTs → proceed to Phase 3 (Task 10b onward).

A subagent executing a gated task must **report the measured verdict** in its summary so the orchestrator can branch. Tasks 5–13 are conditional; their checkboxes stay unchecked if the gate above them sent the plan to a STOP write-up.

---

## File structure

| File | Responsibility | Phase |
|---|---|---|
| `reports/rb_model_bakeoff_2026-06-25.md` | committed old-feature-set bake-off table (baseline-to-beat) | 0 |
| `data/features_probe/rb_trajectory.parquet`, `rb_vegas_preseason.parquet` | probe override inputs (gitignored) | 1 |
| `reports/feature_probe_rb_trajectory.csv` / `_vegas.csv` + a short `.md` verdict note | probe outputs + recorded coverage/threshold/verdict | 1 |
| `scripts/backtest.py` | add `--features-root` + `--position` passthrough | 2 |
| `scripts/refresh_features.py` | add `--features-root` passthrough | 2 |
| `src/projections/schemas.py` (`RbFeaturesSchema`) | declare the new RB feature columns | 2 |
| `src/projections/features/rb.py` (`build_rb_features`) | attach the signaling family | 2 |
| `src/projections/models/baseline.py` (`_RB_FEATURE_COLUMNS`) | add columns to the hardcoded baseline list | 2 |
| `src/projections/models/__init__.py` (`POSITION_DISPATCH`) | flip RB `default_model_class` (Phase 3, only if a non-baseline class wins) | 3 |
| `tests/backtest/model_metrics.json` | refreshed snapshot | 3 |
| `TODO.md`, `project_management.md` | verdict write-up | 1/2/3 (terminal) |

---

## Phase 0 — persist the baseline-to-beat

### Task 1: Capture the RB model bake-off table

**Files:**
- Create: `reports/rb_model_bakeoff_2026-06-25.md`
- Read-only driver (already present): `scripts/_rb_model_bakeoff.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a committed markdown table that §1 of the spec and the Phase 2(b) re-bake-off compare against.

- [ ] **Step 1: Run the driver, capture stdout to the report**

```bash
{
  echo "# RB model bake-off — old feature set (baseline-to-beat)"
  echo
  echo "RB-only walk-forward 2021-2024, ESPN-PPR composite. Driver: scripts/_rb_model_bakeoff.py."
  echo "Lower MAE/RMSE = better; higher spearman = better. baseline is the incumbent default."
  echo
  echo '```'
  .venv/Scripts/python.exe scripts/_rb_model_bakeoff.py
  echo '```'
} > reports/rb_model_bakeoff_2026-06-25.md
```

Expected: the file contains the per-year + pooled table with `baseline` lowest on composite_mae/rmse and highest on spearman_topN (matching spec §1).

- [ ] **Step 2: Sanity-check the file is non-empty and contains the pooled section**

Run: `grep -c "Pooled" reports/rb_model_bakeoff_2026-06-25.md`
Expected: `1` (or more).

- [ ] **Step 3: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add reports/rb_model_bakeoff_2026-06-25.md
PATH=".venv/Scripts:$PATH" git commit -m "report(dfs): persist RB old-feature bake-off (baseline-to-beat, #55)"
```

---

## Phase 1 — signal probe (cheap, no source changes)

### Task 2: Build the override parquets + measure per-column coverage

**Files:**
- Reuse (no edit): `scripts/build_trajectory_override.py`, `scripts/build_vegas_team_context_override.py`
- Create (gitignored): `data/features_probe/rb_trajectory.parquet`, `data/features_probe/rb_vegas_preseason.parquet`

**Interfaces:**
- Consumes: raw `draft_picks` (trajectory), raw `schedules` (Vegas), already present under `data/raw/`.
- Produces: override parquets keyed on `(gsis_id, season, week)`; the measured RB per-column non-null fractions that set Task 3's `--coverage-threshold`.

- [ ] **Step 1: Build both overrides (pass `--output` explicitly — the builders default to their own filenames)**

```bash
.venv/Scripts/python.exe -m scripts.build_trajectory_override \
    --output data/features_probe/rb_trajectory.parquet
.venv/Scripts/python.exe -m scripts.build_vegas_team_context_override \
    --output data/features_probe/rb_vegas_preseason.parquet
```

Expected: both files written; no error. (Both builders emit all four fantasy positions; RB rows are included.)

- [ ] **Step 2: Measure RB-restricted per-column coverage (this sets the thresholds)**

```bash
.venv/Scripts/python.exe - <<'PY'
import pandas as pd
from projections.backtest.cache import read_features
from projections.schemas import Position
# Baseline RB rows the probe will evaluate against (2021-2024, all weeks).
base = pd.concat([read_features(Position.RB, y) for y in (2021, 2022, 2023, 2024)], ignore_index=True)
keys = ["gsis_id", "season", "week"]
for name in ("rb_trajectory", "rb_vegas_preseason"):
    ov = pd.read_parquet(f"data/features_probe/{name}.parquet")
    merged = base[keys].merge(ov, on=keys, how="left")
    cand = [c for c in ov.columns if c not in keys]
    print(f"\n== {name} per-column non-null fraction over {len(base)} RB rows ==")
    for c in cand:
        print(f"  {c:42s} {merged[c].notna().mean():.4f}")
PY
```

Expected: prints a coverage fraction per candidate column. Record the **minimum** (binding) fraction for each family — this drives Task 3's threshold. (Anticipated: Vegas `season_avg_*` ≈0.90+; trajectory `age`/`is_rookie` ≈0.88–0.97; trajectory trend cols ≈0.45–0.55.)

- [ ] **Step 3: Decide the trajectory probe shape from the measured coverage**

- If the trajectory trend columns (`volume_trend_l4_minus_prior_l4`, `snap_pct_change_l4_vs_prior_l4`) are ≥ ~0.85 covered, probe the full trajectory override at a threshold just below the binding column.
- If they are ~0.50 (the likely case), **split**: write an `age`/`is_rookie`-only override (well-covered) and probe it at ~0.85; probe the sparse trend columns as a **separate** candidate at their measured floor, flagging in the report that their delta is imputation-sensitive. Build the split override with a column-select (no builder change):

```bash
.venv/Scripts/python.exe - <<'PY'
import pandas as pd
ov = pd.read_parquet("data/features_probe/rb_trajectory.parquet")
keep = ["gsis_id", "season", "week", "age", "is_rookie"]
ov[keep].to_parquet("data/features_probe/rb_trajectory_age_only.parquet")
print("wrote rb_trajectory_age_only.parquet", ov[keep].shape)
PY
```

- [ ] **Step 4: No commit** (override parquets are gitignored; coverage numbers are recorded in Task 3's report).

### Task 3: Run the probes and record the G1 verdict

**Files:**
- Create: `reports/feature_probe_rb_trajectory.csv`, `reports/feature_probe_rb_vegas.csv`, and a short `reports/feature_probe_rb_summary.md` (verdict + measured coverage + chosen thresholds).

**Interfaces:**
- Consumes: the override parquets + measured coverage from Task 2.
- Produces: **G1 verdict** (per-family SIGNAL/NULL) that branches the plan.

- [ ] **Step 1: Run the Vegas probe (threshold from measured coverage; ~0.90 expected)**

```bash
.venv/Scripts/python.exe -m scripts.probe_feature_signal \
    --candidate-name augment_rb_vegas --position RB \
    --override data/features_probe/rb_vegas_preseason.parquet \
    --coverage-threshold <measured_min_minus_epsilon> \
    --csv-out reports/feature_probe_rb_vegas.csv
```

Expected: per-stat Δ-RMSE table prints; no `OverrideCoverageError`. If it aborts on coverage, the threshold is above the binding column — lower it to the Task 2 measured minimum.

- [ ] **Step 2: Run the trajectory probe(s) per the Task 2 Step 3 decision**

Full-family form (if trend cols well-covered):
```bash
.venv/Scripts/python.exe -m scripts.probe_feature_signal \
    --candidate-name augment_rb_trajectory --position RB \
    --override data/features_probe/rb_trajectory.parquet \
    --coverage-threshold <measured_min_minus_epsilon> \
    --csv-out reports/feature_probe_rb_trajectory.csv
```
Split form (if trend cols ~0.50) — run the age-only override at ~0.85; optionally a second run on the trend-only columns at their measured floor, noted as imputation-sensitive.

- [ ] **Step 3: Record the verdict**

Write `reports/feature_probe_rb_summary.md` with: the measured per-column coverage, the chosen thresholds, the per-stat verdicts, and the **family-level G1 verdict** for each family. Family SIGNAL ⇔ `family_verdict_from_reports` would return SIGNAL — i.e. a **pooled** per-stat SIGNAL exists (CI strictly < 0 AND `|point| ≥ 0.05` fpts; `phase1_should_fire_phase2`/`feature_probe.py:102`). A lone single-year SIGNAL that is NULL pooled does **not** count.

- [ ] **Step 4: Commit the reports**

```bash
PATH=".venv/Scripts:$PATH" git add reports/feature_probe_rb_trajectory.csv reports/feature_probe_rb_vegas.csv reports/feature_probe_rb_summary.md
PATH=".venv/Scripts:$PATH" git commit -m "report(dfs): RB trajectory+vegas signal probe — G1 verdict (#55)"
```

- [ ] **Step 5: GATE G1 — branch**

- **Both families NULL → go to Task 4 (STOP write-up), then END the plan.**
- **Any family SIGNAL → skip Task 4; proceed to Task 5 for the signaling family(ies) only.**

### Task 4: STOP write-up (ONLY if G1 = both NULL)

**Files:**
- Modify: `TODO.md` (#55 entry), `project_management.md`

- [ ] **Step 1: Record the dead-end**

Update #55 in `TODO.md` and add a PM entry: RB feature lift attempted via the two unprobed families; both NULL on the signal probe (cite the recorded coverage + verdicts), so RB is confirmed signal-limited even after exhausting the available feature families — DFS STOP stands for RB, no integration shipped. Note the probe reports as the artifact.

- [ ] **Step 2: Commit and END**

```bash
PATH=".venv/Scripts:$PATH" git add TODO.md project_management.md
PATH=".venv/Scripts:$PATH" git commit -m "docs(dfs): RB feature lift NULL — both families no signal, STOP stands (#55)"
```

The plan is complete (negative result). Push + PR per superpowers-go; skip Tasks 5–13.

---

## Phase 2 — integrate the signaling family + re-test model classes (ONLY if G1 SIGNAL)

> Execute Tasks 5–9 for **each** family that SIGNALed. Column lists below cover both; use only the signaling family's columns.

### Task 5: Add `--features-root`/`--position` to `scripts/backtest.py` (TDD)

**Files:**
- Modify: `scripts/backtest.py` (arg parsing ~line 118-142; the `run_backtest(...)` call ~line 169-173)
- Test: `tests/backtest/test_backtest_cli.py` (create)

**Interfaces:**
- Consumes: `run_backtest(*, positions, features_root, ...)` (`harness.py:209,212`) — both already parameters.
- Produces: a CLI that can target a non-default feature root and a single position, enabling the non-destructive dual-run legs.

- [ ] **Step 1: Write the failing test**

```python
# tests/backtest/test_backtest_cli.py
import subprocess, sys

def test_backtest_help_exposes_features_root_and_position():
    out = subprocess.run(
        [sys.executable, "scripts/backtest.py", "--help"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "--features-root" in out
    assert "--position" in out
```

- [ ] **Step 2: Run it, verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/backtest/test_backtest_cli.py -v`
Expected: FAIL (flags absent from help).

- [ ] **Step 3: Add the flags and thread them through**

In `scripts/backtest.py` `main()` add:
```python
parser.add_argument(
    "--features-root", type=Path, default=Path("data/features"),
    help="feature cache root (default data/features); point at an alternate "
         "root to A/B two feature sets without overwriting the production cache.",
)
parser.add_argument(
    "--position", choices=["QB", "RB", "WR", "TE"], default=None,
    help="restrict the backtest to a single position (default: all four).",
)
```
Then, where `positions` is computed, honor `--position` (it currently only special-cases decomposed variants):
```python
if args.position is not None:
    positions = (Position[args.position],)
```
And pass `features_root` into both `run_backtest(...)` call sites:
```python
run = (
    run_backtest(model_classes=model_classes, positions=positions, features_root=args.features_root)
    if positions is not None
    else run_backtest(model_classes=model_classes, features_root=args.features_root)
)
```
(Keep the existing `decomposed-baseline`/`ensemble-decomposed` WR-only positions logic; `--position` overrides `None`, and an explicit `--position WR` is compatible.)

- [ ] **Step 4: Run the test, verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/backtest/test_backtest_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Checklist + commit**

```bash
.venv/Scripts/python.exe -m pytest tests/backtest/test_backtest_cli.py -v
.venv/Scripts/python.exe -m mypy src tests && .venv/Scripts/python.exe -m ruff check src tests scripts && .venv/Scripts/python.exe -m ruff format --check src tests scripts
PATH=".venv/Scripts:$PATH" git add scripts/backtest.py tests/backtest/test_backtest_cli.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(backtest): --features-root/--position passthrough for non-destructive A/B (#55)"
```

### Task 6: Add `--features-root` to `scripts/refresh_features.py` (TDD)

**Files:**
- Modify: `scripts/refresh_features.py` (parser ~line 132; `features_root` derivation ~line 137)
- Test: `tests/test_consensus/test_refresh.py` (extend) or `tests/backtest/test_backtest_cli.py`

**Interfaces:**
- Consumes: `_refresh_one(..., features_root: Path, ...)` (`refresh_features.py:64`) — already a parameter.
- Produces: ability to build the RB cache into an alternate root (`data/features_rb_aug`) without relocating `raw/`.

- [ ] **Step 1: Write the failing test**

```python
# tests/backtest/test_backtest_cli.py  (append)
def test_refresh_features_help_exposes_features_root():
    out = subprocess.run(
        [sys.executable, "scripts/refresh_features.py", "--help"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "--features-root" in out
```

- [ ] **Step 2: Run, verify fail**

Run: `.venv/Scripts/python.exe -m pytest tests/backtest/test_backtest_cli.py::test_refresh_features_help_exposes_features_root -v`
Expected: FAIL.

- [ ] **Step 3: Add the flag, default preserving current behavior**

```python
parser.add_argument(
    "--features-root", type=Path, default=None,
    help="output feature cache root (default: <data-root>/features). Set to "
         "build into an alternate root without relocating raw/.",
)
```
Then:
```python
raw_root = args.data_root / "raw"
features_root = args.features_root if args.features_root is not None else args.data_root / "features"
```

- [ ] **Step 4: Run, verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/backtest/test_backtest_cli.py::test_refresh_features_help_exposes_features_root -v`
Expected: PASS.

- [ ] **Step 5: Checklist + commit**

```bash
.venv/Scripts/python.exe -m mypy src tests && .venv/Scripts/python.exe -m ruff check scripts tests && .venv/Scripts/python.exe -m ruff format --check scripts tests
PATH=".venv/Scripts:$PATH" git add scripts/refresh_features.py tests/backtest/test_backtest_cli.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(refresh): --features-root output override (#55)"
```

### Task 7: Add the signaling family's columns to `RbFeaturesSchema` (TDD)

**Files:**
- Modify: `src/projections/schemas.py` (`RbFeaturesSchema`, ~line 687-751)
- Test: `tests/test_draft/test_backtest/test_schemas.py` (extend) — mirror the WR trajectory/Vegas schema tests already there.

**Interfaces:**
- Produces: `RbFeaturesSchema` accepting the new columns; consumed by `build_rb_features` (Task 8) and the lightgbm dynamic feature-list derivation.

- [ ] **Step 1: Write the failing test** (trajectory shown; for Vegas use the four `*_implied_team_total`/`*_spread`/`season_avg_*` columns)

```python
def test_rb_schema_accepts_trajectory_columns():
    import pandas as pd
    from projections.schemas import RbFeaturesSchema
    # minimal valid RB feature row + the new nullable float columns
    df = _minimal_rb_features_row()  # existing helper / fixture in this test module
    df["age"] = pd.Series([25.0], dtype="float64")
    df["is_rookie"] = pd.Series([0.0], dtype="float64")
    df["volume_trend_l4_minus_prior_l4"] = pd.Series([1.5], dtype="float64")
    df["snap_pct_change_l4_vs_prior_l4"] = pd.Series([0.1], dtype="float64")
    validated = RbFeaturesSchema.validate(df)  # must not raise
    assert "age" in validated.columns
```

- [ ] **Step 2: Run, verify fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_draft/test_backtest/test_schemas.py -k rb_schema_accepts -v`
Expected: FAIL (schema rejects the unexpected columns under `strict`).

- [ ] **Step 3: Add the columns to `RbFeaturesSchema`** (copy the WR declarations verbatim, `schemas.py:604-607`)

```python
    # Trajectory features (mirrors WrFeaturesSchema:604-607). float nullable.
    age: Series[float] = pa.Field(ge=15, le=50, nullable=True)
    is_rookie: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    volume_trend_l4_minus_prior_l4: Series[float] = pa.Field(nullable=True)
    snap_pct_change_l4_vs_prior_l4: Series[float] = pa.Field(ge=-1, le=1, nullable=True)
```
For the Vegas family (mirror QB `schemas.py:674-677`):
```python
    preseason_implied_team_total: Series[float] = pa.Field(nullable=True)
    preseason_spread: Series[float] = pa.Field(nullable=True)
    season_avg_implied_team_total: Series[float] = pa.Field(nullable=True)
    season_avg_spread: Series[float] = pa.Field(nullable=True)
```

- [ ] **Step 4: Run, verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_draft/test_backtest/test_schemas.py -k rb_schema -v`
Expected: PASS.

- [ ] **Step 5: Checklist + commit**

```bash
.venv/Scripts/python.exe -m pytest -v -k "schemas"
.venv/Scripts/python.exe -m mypy src tests && .venv/Scripts/python.exe -m ruff check src tests && .venv/Scripts/python.exe -m ruff format --check src tests
PATH=".venv/Scripts:$PATH" git add src/projections/schemas.py tests/test_draft/test_backtest/test_schemas.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(schemas): RB trajectory/vegas feature columns (#55)"
```

### Task 8: Wire the family into `build_rb_features` (TDD)

**Files:**
- Modify: `src/projections/features/rb.py` (`build_rb_features`)
- Test: `tests/test_features/test_rb.py` (extend) — mirror `tests/test_features/test_wr_trajectory_vegas_leakage.py` for the leakage assertion.

**Interfaces:**
- Consumes: `attach_trajectory_features(index, weekly_stats, snap_counts, build_draft_lookup(draft_picks), Position.RB)` and/or `attach_vegas_team_context_features(index, schedules)` (`features/trajectory_features.py:273`, `features/vegas_team_context_features.py:129`); `build_wr_features` (`features/wr.py:257-269`) is the working template.
- Produces: RB feature frame carrying the new columns, validated by `RbFeaturesSchema`.

- [ ] **Step 1: Write the failing test** (columns populated, not all-NaN, coverage ≥ recorded floor)

```python
def test_build_rb_features_attaches_trajectory(rb_feature_inputs):
    # rb_feature_inputs: existing fixture giving real-shaped raw frames incl. draft_picks
    feats = build_rb_features(**rb_feature_inputs)  # now passing draft_picks
    for col in ("age", "is_rookie", "volume_trend_l4_minus_prior_l4", "snap_pct_change_l4_vs_prior_l4"):
        assert col in feats.columns
    # age/is_rookie are well-covered; not all-NaN
    assert feats["age"].notna().mean() > 0.5
    assert feats["is_rookie"].notna().mean() > 0.5
```

- [ ] **Step 2: Run, verify fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_rb.py -k trajectory -v`
Expected: FAIL (columns absent).

- [ ] **Step 3: Wire the helper** (mirror WR: pass the **full** un-`prior_mask` `weekly_stats`/`snap_counts`)

In `build_rb_features`, after the existing feature assembly and before the final `RbFeaturesSchema.validate`, build the player-team-week index and attach:
```python
from projections.features.trajectory_features import attach_trajectory_features, build_draft_lookup
# ... (Vegas: attach_vegas_team_context_features)
draft_lookup = build_draft_lookup(draft_picks)
traj_idx = df[["gsis_id", "season", "week", "team", "opp"]].copy()  # match the index cols build_wr_features uses
traj = attach_trajectory_features(traj_idx, weekly_stats, snap_counts, draft_lookup, Position.RB)
df = df.merge(traj, on=["gsis_id", "season", "week", "team", "opp"], how="left")
```
Use `weekly_stats`/`snap_counts` (the full params), **not** the internally `prior_mask`-sliced `ws`/`sc` — the helper's trailing-window/age computation needs full history (see the WR comment at `features/wr.py:256-262`). Then `df = RbFeaturesSchema.validate(df)`.

- [ ] **Step 4: Run, verify pass; add the leakage test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_rb.py -k trajectory -v`
Expected: PASS. Then mirror `test_wr_trajectory_vegas_leakage.py` to assert no current-week leakage for RB and run it.

- [ ] **Step 5: Checklist + commit**

```bash
.venv/Scripts/python.exe -m pytest -v -k "rb or store or schemas"
.venv/Scripts/python.exe -m mypy src tests && .venv/Scripts/python.exe -m ruff check src tests && .venv/Scripts/python.exe -m ruff format --check src tests
PATH=".venv/Scripts:$PATH" git add src/projections/features/rb.py tests/test_features/test_rb.py tests/test_features/test_rb_leakage.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(features): attach trajectory/vegas to build_rb_features (#55)"
```

### Task 9: Add columns to `_RB_FEATURE_COLUMNS`, build augmented cache, run dual-run gate + bake-off (GATE G2)

**Files:**
- Modify: `src/projections/models/baseline.py` (`_RB_FEATURE_COLUMNS`, line 386)
- Create (gitignored): `data/features_rb_aug/rb/*`, `data/backtest/run_*/`
- Update: `reports/feature_probe_rb_summary.md` (append the gate + bake-off result)

**Interfaces:**
- Consumes: the augmented schema/builder (Tasks 7–8), the CLI passthroughs (Tasks 5–6), `adoption_gate.py` dual-run mode.
- Produces: **G2 verdict** (dual-run ADOPT? + which class wins the bake-off) that branches the plan.

- [ ] **Step 1: Add the signaling family's columns to `_RB_FEATURE_COLUMNS`**

Append the same column names added in Task 7 to the `_RB_FEATURE_COLUMNS` tuple (`baseline.py:386`), with a comment citing this plan. (lightgbm derives its list from the schema automatically; only the hardcoded baseline list needs the edit.)

- [ ] **Step 2: Run the model-backtest snapshot check to confirm baseline still trains** (sanity)

Run: `.venv/Scripts/python.exe scripts/backtest.py --report --model baseline --position RB`
Expected: completes, writes `data/backtest/run_<ts>/results.parquet`. **Record this run dir as the OLD-features baseline leg** (it ran against `data/features`).

- [ ] **Step 3: Build the augmented RB cache into a separate root**

```bash
.venv/Scripts/python.exe scripts/refresh_features.py rb --features-root data/features_rb_aug
```
Expected: `data/features_rb_aug/rb/season=YYYY/...` written for all cached seasons.

- [ ] **Step 4: Run the augmented baseline leg**

```bash
.venv/Scripts/python.exe scripts/backtest.py --report --model baseline --position RB --features-root data/features_rb_aug
```
Expected: writes a new `data/backtest/run_<ts>/results.parquet` (augmented leg).

- [ ] **Step 5: Dual-run adoption gate (old vs augmented)**

```bash
.venv/Scripts/python.exe scripts/adoption_gate.py \
    --baseline-run data/backtest/run_<OLD_ts> \
    --candidate-run data/backtest/run_<AUG_ts> \
    --csv-out reports/adoption_gate_rb_<family>_baseline.csv
```
(Do **not** pass `--candidate` — dual-run uses synthesized labels. Both legs are single-class/single-position so the one-to-one pairing holds; the two legs share identical row coverage since only feature *values* changed.) ADOPT ⇔ composite ΔRMSE `hi_95 < 0` AND spearman `lo_95 > -0.02`.

- [ ] **Step 6: Re-run the model-class bake-off on the augmented cache (class selection)**

```bash
.venv/Scripts/python.exe scripts/_rb_model_bakeoff.py --features-root data/features_rb_aug \
    > reports/rb_model_bakeoff_augmented.md
```
(Give `_rb_model_bakeoff.py` the same `--features-root` arg — a one-line `argparse` + pass to `run_backtest(..., features_root=...)`; it currently hardcodes the default. If preferred, run the three classes via the extended `backtest.py` instead.) Compare absolute composite_rmse/mae/spearman vs the Phase 0 table.

- [ ] **Step 7: Record the G2 verdict + commit reports/code**

Append to `reports/feature_probe_rb_summary.md`: the dual-run gate verdict and the augmented bake-off table; state which model class has the lowest composite_rmse (and that it does not regress spearman vs augmented baseline).
```bash
.venv/Scripts/python.exe -m mypy src tests && .venv/Scripts/python.exe -m ruff check src tests scripts && .venv/Scripts/python.exe -m ruff format --check src tests scripts
PATH=".venv/Scripts:$PATH" git add src/projections/models/baseline.py reports/feature_probe_rb_summary.md reports/rb_model_bakeoff_augmented.md scripts/_rb_model_bakeoff.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(models): RB augmented feature columns + G2 gate/bakeoff (#55)"
```

- [ ] **Step 8: GATE G2 — branch**

- **Dual-run gate does NOT ADOPT → Task 10a (revert + STOP write-up), END.**
- **Gate ADOPTs → Task 10b onward** (Phase 3). Note which class wins the bake-off (baseline vs a non-baseline class) — it decides whether the default flips.

### Task 10a: Revert + STOP write-up (ONLY if G2 not ADOPT)

**Files:**
- Revert: the Task 7/8/9 source edits (schema cols, builder wiring, `_RB_FEATURE_COLUMNS`) — do not ship inert feature columns. Keep the CLI passthroughs (Tasks 5–6) — they are independently useful, non-RB-specific infra.
- Modify: `TODO.md`, `project_management.md`

- [ ] **Step 1: Revert the integration commits** (keep Tasks 5–6)

```bash
git revert --no-edit <Task9_commit> <Task8_commit> <Task7_commit>
```
(Resolve any conflict by ensuring `RbFeaturesSchema`, `build_rb_features`, `_RB_FEATURE_COLUMNS` match `main`; rerun `pytest -k "rb or schemas or store"` to confirm green.)

- [ ] **Step 2: Document the result + commit**

Update #55/PM: the family SIGNALed on the cheap probe but did **not** ADOPT through the dual-run model gate (cite the gate ΔRMSE/CI), so the feature does not improve the model in practice — integration reverted, DFS STOP stands for RB. Commit, push + PR, END.

---

## Phase 3 — flip default + re-validate on DFS (ONLY if G2 ADOPT)

### Task 10b: Flip the default model class (ONLY if a non-baseline class won the bake-off)

**Files:**
- Modify: `src/projections/models/__init__.py` (`POSITION_DISPATCH[Position.RB].default_model_class`, line 194)

**Interfaces:**
- Consumes: the Task 9 bake-off winner.
- Produces: the new production RB default.

- [ ] **Step 1: If the bake-off winner is non-baseline, set the default**

Change `default_model_class="baseline"` → the winning class (`"lightgbm-nb"` or `"ensemble"`). If `baseline` still wins the augmented bake-off, **skip this task** (features ship but default stays `baseline`) and proceed to Task 11.

- [ ] **Step 2: Checklist + commit**

```bash
.venv/Scripts/python.exe -m pytest -v -k "models or dispatch"
.venv/Scripts/python.exe -m mypy src tests && .venv/Scripts/python.exe -m ruff check src tests
PATH=".venv/Scripts:$PATH" git add src/projections/models/__init__.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(models): flip RB default to <winner> on augmented features (#55)"
```

### Task 11: Rebuild the production RB cache + refresh the backtest snapshot

**Files:**
- Rebuild (gitignored): `data/features/rb/*`
- Modify: `tests/backtest/model_metrics.json`

- [ ] **Step 1: Rebuild the production cache (default root) with the augmented features**

```bash
.venv/Scripts/python.exe scripts/refresh_features.py rb
```

- [ ] **Step 2: Update + verify the snapshot**

```bash
.venv/Scripts/python.exe scripts/backtest.py --update-snapshot --model baseline
.venv/Scripts/python.exe scripts/backtest.py --check
```
Expected: `--check` prints PASS. (If the default flipped, also update the snapshot for the new default class as needed so `--check` passes.)

- [ ] **Step 3: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add tests/backtest/model_metrics.json
PATH=".venv/Scripts:$PATH" git commit -m "test(backtest): refresh snapshot for augmented RB features (#55)"
```

### Task 12: Re-run the DFS edge study + per-position significance

**Files:**
- Reuse: `scripts/_dfs_per_position_analysis.py`
- Create: `reports/dfs_per_position_significance_<date>.md` (or append a delta section)

**Interfaces:**
- Consumes: the new RB routing/features (the cache signature changes → a fresh `data/dfs_universe_2021-2024_<sig>.parquet` builds automatically).
- Produces: RB's new edge-study fraction + CI, and the pooled-verdict movement (reported, not gated).

- [ ] **Step 1: Run the per-position analysis**

```bash
.venv/Scripts/python.exe scripts/_dfs_per_position_analysis.py
```
Expected: rebuilds the RB cells (new signature), prints RB fraction + CI and the pooled fraction. (~75-min single-process; acceptable per spec — vectorization is out of scope.)

- [ ] **Step 2: Record before/after**

Write/append a report with RB's old (0.430, CI 0.397-0.463) vs new fraction + CI, and the old (0.476) vs new pooled fraction + verdict. State plainly whether RB crossed 0.50 and whether the pool moved off STOP — both outcomes are valid.

- [ ] **Step 3: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add reports/dfs_per_position_significance_*.md
PATH=".venv/Scripts:$PATH" git commit -m "report(dfs): RB edge-study re-check after feature lift (#55)"
```

### Task 13: Update TODO/PM with the final verdict

**Files:**
- Modify: `TODO.md` (#55), `project_management.md`

- [ ] **Step 1: Write the verdict**

Record: which family shipped, whether the default flipped, the model-backtest improvement (gate ΔRMSE), and the DFS edge-study before/after (RB fraction + pooled). Classify the outcome: **lift shipped** (RB fraction up, possibly pool off STOP) or **improved-but-still-STOP**. Link the spec, plan, and reports.

- [ ] **Step 2: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add TODO.md project_management.md
PATH=".venv/Scripts:$PATH" git commit -m "docs(dfs): RB feature lift verdict (#55)"
```

---

## Self-review notes

- **Spec coverage:** Phase 0 (§4 Phase 0) → Task 1; Phase 1 probe + G1 (§4 Phase 1, §6 G1) → Tasks 2–4; CLI prerequisite (§4 Phase 2 "CLI prerequisite") → Tasks 5–6; schema/builder/model-cols (§4 Phase 2 steps 1–3) → Tasks 7–9 step 1; dual-run gate + bake-off + G2 (§4 Phase 2 steps 4–5, §6 G2) → Task 9; revert-on-fail (§6 G2 third branch, §8) → Task 10a; default flip + prod rebuild + snapshot + edge study + docs (§4 Phase 3) → Tasks 10b–13.
- **Gates honored:** G1 (Task 3 step 5), G2 (Task 9 step 8) branch to STOP write-ups; the both-NULL early exit (the spec's likely outcome) costs only Tasks 1–4.
- **Coverage discipline:** thresholds are measured (Task 2), not guessed; the imputation-sensitive trend-col split is handled (Task 2 step 3 / Task 3 step 2).
- **No dead columns:** Task 10a reverts the integration if the gate fails.
- **Type consistency:** the four trajectory + four Vegas column names are identical across Tasks 7, 8, 9 and match the WR/QB schema precedents.
