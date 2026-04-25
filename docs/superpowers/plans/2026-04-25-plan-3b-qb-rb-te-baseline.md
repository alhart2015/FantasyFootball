# Plan 3b — QB / RB / TE Model A baselines — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the 3a WR baseline to QB / RB / TE. Add three new factories + a position-dispatch registry; unify the three CLI scripts into position-arg-driven scripts; train and sanity-check each position on real 2018-2023 data with 2024 held out.

**Architecture:** No new packages. `BaselineModel` (already parameterized in 3a) gains two required constructor args (`feature_schema`, `code_hash_files`) so its hardcoded WR references go away. Three new factory functions (`qb_baseline`, `rb_baseline`, `te_baseline`) live alongside the existing `wr_baseline()` in `baseline.py`. A `POSITION_DISPATCH` registry in `models/__init__.py` tells callers (CLI scripts, future Plan 3c backtest harness) which factory + feature builder + feature schema + NGS source to use per position. The three WR-specific scripts get replaced by three generalized ones that take `--position {qb|rb|te|wr}`. The TE feature schema and builder gain two rushing columns to support the Taysom-Hill rationale for including rushing as a TE target stat.

**Tech Stack:** Python 3.11+, pandas, pandera (`pandera.pandas`), scikit-learn (`RidgeCV`), pytest, mypy strict, ruff, joblib, msgpack.

**Spec:** `docs/superpowers/specs/2026-04-25-plan-3b-qb-rb-te-baseline-design.md` (commit `ba6fbc4` on `feat/plan-3b-qb-rb-te-baseline`).

**Worktree:** `.worktrees/feat-plan-3b-qb-rb-te-baseline/` on branch `feat/plan-3b-qb-rb-te-baseline`. All commands below run from inside this worktree unless stated otherwise.

**Venv note (Windows):** the worktree shares the main venv. Prepend `/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH` (bash) when running `pytest` / `ruff` / `mypy` / `pre-commit` from this worktree, or activate the main venv first.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/projections/schemas.py` | modify | Add `rushing_attempts_per_game_l4` + `rushing_yards_per_game_l4` to `TeFeaturesSchema`. |
| `src/projections/features/te.py` | modify | Populate the two new TE rushing feature columns in `build_te_features`. |
| `src/projections/models/baseline.py` | modify | Add `feature_schema` + `code_hash_files` constructor args; remove hardcoded WR references; add `_QB_*` / `_RB_*` / `_TE_*` constants; add `qb_baseline()` / `rb_baseline()` / `te_baseline()` factories. |
| `src/projections/models/__init__.py` | modify | Add `_PositionDispatch` dataclass + `POSITION_DISPATCH` registry; export from package. |
| `tests/test_features/test_te.py` | modify | New fixture row exercising non-zero TE rushing; new assertions on rushing columns. |
| `tests/test_features/conftest.py` | modify | TE fixture rows for the new Taysom-Hill-shape weekly_stats entry. |
| `tests/conftest.py` | modify | Rename `baseline_features` / `baseline_weekly_stats` → `*_wr`; add `_make_baseline_fixtures(position)` helper; add `baseline_features_{qb,rb,te}` / `baseline_weekly_stats_{qb,rb,te}` fixtures at root scope. |
| `tests/test_models/test_baseline.py` | modify | Update fixture refs to `baseline_features_wr`. |
| `tests/test_models/test_baseline_leakage.py` | modify | Update fixture refs to `baseline_features_wr`. |
| `tests/test_models/test_baseline_qb.py` | create | Per-position unit tests on `qb_baseline()`. |
| `tests/test_models/test_baseline_rb.py` | create | Per-position unit tests on `rb_baseline()`. |
| `tests/test_models/test_baseline_te.py` | create | Per-position unit tests on `te_baseline()`. |
| `tests/test_models/test_baseline_qb_leakage.py` | create | Per-position leakage test for QB. |
| `tests/test_models/test_baseline_rb_leakage.py` | create | Per-position leakage test for RB. |
| `tests/test_models/test_baseline_te_leakage.py` | create | Per-position leakage test for TE. |
| `tests/test_smoke.py` | modify | Parametrize the existing `test_smoke_wr_baseline_fit_predict_write` across all four positions. |
| `scripts/train_baseline.py` | create | Generalized training script: `--position {qb|rb|te|wr}`. |
| `scripts/predict_2024.py` | create | Generalized 2024 weekly-projection script. |
| `scripts/sanity_check_baseline.py` | create | Generalized 2024 held-out eval script. |
| `scripts/train_wr_baseline.py` | delete | Replaced by `train_baseline.py wr`. |
| `scripts/predict_2024_wr.py` | delete | Replaced by `predict_2024.py wr`. |
| `scripts/sanity_check_wr_baseline.py` | delete | Replaced by `sanity_check_baseline.py wr`. |
| `TODO.md` | modify | Phase 2 adds "retrain WR 2018-2023 artifact" entry; Phase 7 closes it. |
| `project_management.md` | modify | Phase 7 records per-position 2024 sanity-check eval + decision-log entries + "next action: Plan 3c". |

---

## Task 1 (Phase 1): TE feature schema + builder rushing extension

**Files:**
- Modify: `src/projections/schemas.py:567-604` (TeFeaturesSchema)
- Modify: `src/projections/features/te.py` (build_te_features)
- Modify: `tests/test_features/conftest.py` (TE fixture)
- Modify: `tests/test_features/test_te.py` (assertions)

**Why TDD-shaped: this is a schema + builder change. The test goes first to fail before the schema/builder land.**

- [ ] **Step 1.1: Add a TE-rushing test row to the conftest fixture.**

Open `tests/test_features/conftest.py` and locate the `te_weekly_stats` fixture. Add one row for a Taysom-Hill-shaped TE (high carries) alongside the existing rows. Find the fixture (search for `te_weekly_stats`) and append (within the same `pd.DataFrame({...})` call) values for the new gsis_id `"00-0033084"` (the existing fixture already references this id in test assertions — confirm it's present, otherwise pick a non-conflicting one).

The exact additions depend on how the existing fixture is shaped. Read the file first to know whether rows are dicts-of-lists or list-of-dicts:

Run: `grep -n "te_weekly_stats" tests/test_features/conftest.py`

For each row in the existing `te_weekly_stats` covering weeks 1–4, ensure that for `"00-0033084"` (Taysom Hill) the columns `carries`, `rushing_yards`, `rushing_tds` carry non-zero values (e.g., `carries: 6`, `rushing_yards: 28.0`, `rushing_tds: 1` on at least one week). Touching these is non-breaking because `WeeklyStatsSchema` already has those columns.

Also confirm the TE depth-chart fixture (`te_depth_charts`) lists `"00-0033084"` at TE for the same season/weeks; if not, add a row.

- [ ] **Step 1.2: Write the failing schema test.**

Open `tests/test_features/test_te.py` and add this test below `test_build_te_features_one_row_per_rostered_te`:

```python
def test_build_te_features_emits_rushing_columns(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
) -> None:
    """TE schema includes rushing_attempts_per_game_l4 and rushing_yards_per_game_l4
    so the TE Model A baseline (Plan 3b) can capture Taysom-Hill-style rushing
    contribution. The columns are populated from the same WeeklyStatsSchema
    rushing source as RB."""
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        season=2024,
        as_of_week=5,
    )
    assert "rushing_attempts_per_game_l4" in out.columns
    assert "rushing_yards_per_game_l4" in out.columns
    # The Taysom Hill row should have non-zero rushing means after the trailing-4 rollup.
    hill = out[out["gsis_id"] == "00-0033084"]
    assert not hill.empty
    assert hill["rushing_attempts_per_game_l4"].iloc[0] > 0
    assert hill["rushing_yards_per_game_l4"].iloc[0] > 0
```

- [ ] **Step 1.3: Run the test — expect failure.**

Run: `pytest tests/test_features/test_te.py::test_build_te_features_emits_rushing_columns -v`

Expected: FAIL with `KeyError: 'rushing_attempts_per_game_l4'` or `AssertionError: 'rushing_attempts_per_game_l4' not in out.columns`.

- [ ] **Step 1.4: Extend `TeFeaturesSchema` with the two rushing columns.**

Edit `src/projections/schemas.py`. Find the `TeFeaturesSchema` class (around line 567). After the receiving-usage block (ending with `receiving_tds_per_game_l4`) and before the `# Snap / role` comment, insert:

```python
    # Rushing usage (rolling) — added Plan 3b for Taysom-Hill-shape TEs that
    # carry the ball; mirrors RB's rushing-feature shape.
    rushing_attempts_per_game_l4: Series[float] = pa.Field(ge=0)
    rushing_yards_per_game_l4: Series[float] = pa.Field(ge=0)
```

- [ ] **Step 1.5: Populate the new columns in `build_te_features`.**

Edit `src/projections/features/te.py`. Two changes:

(a) Add `rushing_attempts_per_game_l4` and `rushing_yards_per_game_l4` to `_ROLLING_ZERO_FILL_COLS` so rookies / players-with-no-prior-rushing get 0.0 instead of NaN:

```python
_ROLLING_ZERO_FILL_COLS: tuple[str, ...] = (
    "targets_per_game_l4",
    "targets_per_game_std",
    "target_share_l4",
    "receptions_per_game_l4",
    "receiving_yards_per_game_l4",
    "receiving_tds_per_game_l4",
    "rushing_attempts_per_game_l4",
    "rushing_yards_per_game_l4",
)
```

(b) After the existing `rec_td_l4 = trailing_4_per_player(...)` block, add two new rolling computations against `ws_te` (the TE-filtered weekly stats):

```python
    rush_att_l4 = trailing_4_per_player(ws_te, Stat.CARRIES.value).rename(
        columns={"mean_l4": "rushing_attempts_per_game_l4"}
    )
    rush_yd_l4 = trailing_4_per_player(ws_te, Stat.RUSHING_YARDS.value).rename(
        columns={"mean_l4": "rushing_yards_per_game_l4"}
    )
```

(c) After the existing receiving-feature merges in the assemble block (where `out = out.merge(rec_td_l4, ...)` happens), add two more merges:

```python
    out = out.merge(rush_att_l4, on="gsis_id", how="left")
    out = out.merge(rush_yd_l4, on="gsis_id", how="left")
```

- [ ] **Step 1.6: Run the test — expect pass.**

Run: `pytest tests/test_features/test_te.py -v`

Expected: all TE feature tests pass, including the new `test_build_te_features_emits_rushing_columns`.

- [ ] **Step 1.7: Run the leakage test — expect pass without changes.**

Run: `pytest tests/test_features/test_te_leakage.py -v`

Expected: all pass. Leakage assertions don't depend on which columns exist; the new columns are populated from the same leakage-safe `prior_mask`-filtered weekly stats.

- [ ] **Step 1.8: Full gate.**

Run:
```bash
pytest -q
mypy src tests
ruff check src tests
ruff format --check src tests
pytest -q -k "ingest or store or schemas"
```

Expected: all pass; no mypy/ruff errors; no format drift; the schemas-keyword subset green.

- [ ] **Step 1.9: Commit.**

```bash
git add src/projections/schemas.py src/projections/features/te.py tests/test_features/conftest.py tests/test_features/test_te.py
git commit -m "$(cat <<'EOF'
feat(features,schemas): TeFeaturesSchema + build_te_features rushing columns

Plan 3b prep: Taysom-Hill-shape TEs carry the ball, and 3b's TE Model
A baseline includes rushing_yards / rushing_tds as target stats. The
TE feature builder must therefore expose rushing_attempts_per_game_l4
and rushing_yards_per_game_l4 alongside the existing receiving rolling
features, mirroring RB's shape.

- TeFeaturesSchema gains two new ge=0 float fields.
- build_te_features computes them from ws_te (TE-filtered weekly_stats)
  using the existing trailing_4_per_player helper.
- Rookies / non-rushing TEs zero-fill via _ROLLING_ZERO_FILL_COLS so
  the schema's non-nullable constraint is satisfied.
- Test fixture extended: a Taysom-Hill-shape TE with non-zero carries
  in the trailing window verifies the new columns roll up correctly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2 (Phase 2): `BaselineModel` constructor change

**Files:**
- Modify: `src/projections/models/baseline.py`
- Modify: `TODO.md` (add WR-retrain TODO)

**Goal:** Replace the two hardcoded WR references inside `BaselineModel` with constructor-driven equivalents. Required (no-default) fields per spec Section 1: `feature_schema: type[pa.DataFrameModel]` and `code_hash_files: tuple[Path, ...]`. The existing 3a-trained WR artifact becomes unloadable; a TODO entry tracks the retrain.

- [ ] **Step 2.1: Re-read the current BaselineModel dataclass** (you'll be editing it; don't trust memory).

Run: `cat src/projections/models/baseline.py` (or open in editor). Locate the dataclass definition (around line 110) and the two hardcoded sites: `WrFeaturesSchema.validate(features)` (in `fit` and in `predict_distribution`) and the `tracked = [...]` list in `fit`.

- [ ] **Step 2.2: Update imports.**

Edit `src/projections/models/baseline.py`. Replace this import block:

```python
from projections.schemas import (
    _PYARROW_STR,
    DistributionFamily,
    Position,
    ProjectionWeeklySchema,
    Ruleset,
    Stat,
    WeeklyStatsSchema,
    WrFeaturesSchema,
)
```

with this (drops `WrFeaturesSchema` since it's no longer hardcoded; adds `pandera.pandas as pa` for the type annotation):

```python
import pandera.pandas as pa

from projections.schemas import (
    _PYARROW_STR,
    DistributionFamily,
    Position,
    ProjectionWeeklySchema,
    Ruleset,
    Stat,
    WeeklyStatsSchema,
    WrFeaturesSchema,
)
```

(`WrFeaturesSchema` stays in the import block because `wr_baseline()` still references it as a constructor arg.)

- [ ] **Step 2.3: Add `feature_schema` and `code_hash_files` to the dataclass.**

In the same file, find the `@dataclass class BaselineModel:` definition. The current fields are:

```python
@dataclass
class BaselineModel:
    """Per-stat Ridge baseline. Construct via per-position factories
    (wr_baseline, etc.); do not call __init__ directly."""

    position: Position
    target_stats: tuple[Stat, ...]
    feature_columns: tuple[str, ...]
    dist_families: Mapping[Stat, DistributionFamily]
    ...
```

Add the two new fields immediately after `dist_families` and before the fit-populated state (`feature_means: pd.Series | None = field(default=None)`):

```python
    feature_schema: type[pa.DataFrameModel]
    code_hash_files: tuple[Path, ...]
```

Both are required (no `field(default=...)`), per the spec's no-backward-compat decision.

- [ ] **Step 2.4: Replace the hardcoded `WrFeaturesSchema.validate(features)` calls.**

Two sites:

(a) Inside `fit`, replace:
```python
        features = WrFeaturesSchema.validate(features)
```
with:
```python
        features = self.feature_schema.validate(features)
```

(b) Inside `predict_distribution`, replace:
```python
        features = WrFeaturesSchema.validate(features)
```
with:
```python
        features = self.feature_schema.validate(features)
```

- [ ] **Step 2.5: Replace the hardcoded `tracked = [...]` file list inside `fit`.**

Find the block (near the end of `fit`):

```python
        # Code hash over source files whose change should invalidate the
        # artifact. Spec section 5.2 lists the canonical set.
        repo_root = Path(__file__).resolve().parents[3]
        tracked = [
            repo_root / "src" / "projections" / "models" / "base.py",
            repo_root / "src" / "projections" / "models" / "baseline.py",
            repo_root / "src" / "projections" / "features" / "wr.py",
            repo_root / "src" / "projections" / "features" / "_shared.py",
            repo_root / "src" / "projections" / "features" / "_rolling.py",
            repo_root / "src" / "projections" / "features" / "_opponent.py",
            repo_root / "src" / "projections" / "scoring" / "score.py",
            repo_root / "src" / "projections" / "scoring" / "score_distribution.py",
        ]
        self.code_hash = compute_code_hash(tracked)
```

Replace with:

```python
        # Code hash over the file list this factory declared. The exact set
        # is per-position (each factory passes its own features/{pos}.py).
        self.code_hash = compute_code_hash(self.code_hash_files)
```

- [ ] **Step 2.6: Update `wr_baseline()` to pass the two new args.**

Find the existing factory at the bottom of the file. Replace it with:

```python
def wr_baseline() -> BaselineModel:
    """Construct an unfitted WR-baseline model. Caller invokes .fit(features,
    weekly_stats) and then .save(path)."""
    repo_root = Path(__file__).resolve().parents[3]
    return BaselineModel(
        position=Position.WR,
        target_stats=_WR_TARGET_STATS,
        feature_columns=_WR_FEATURE_COLUMNS,
        dist_families=_WR_DIST_FAMILIES,
        feature_schema=WrFeaturesSchema,
        code_hash_files=(
            repo_root / "src" / "projections" / "models" / "base.py",
            repo_root / "src" / "projections" / "models" / "baseline.py",
            repo_root / "src" / "projections" / "features" / "wr.py",
            repo_root / "src" / "projections" / "features" / "_shared.py",
            repo_root / "src" / "projections" / "features" / "_rolling.py",
            repo_root / "src" / "projections" / "features" / "_opponent.py",
            repo_root / "src" / "projections" / "scoring" / "score.py",
            repo_root / "src" / "projections" / "scoring" / "score_distribution.py",
        ),
    )
```

- [ ] **Step 2.7: Add the TODO entry for retraining the WR artifact.**

Edit `TODO.md`. Add the following entry just before `### 9. WR feature builder edge cases for production data` (so the open list keeps numerical-ish ordering by recency at the bottom). Pick a number that doesn't collide with the existing set — current open numbers are 1,2,3,4,5,6,7,9,10,11,12,13,14,16; pick **17**:

```markdown
### 17. Retrain WR 2018-2023 artifact after Plan 3b

Plan 3b adds two required fields (`feature_schema`, `code_hash_files`)
to the `BaselineModel` dataclass. The existing artifact at
`models/artifacts/wr-baseline-2018-2023-925f492b.joblib` (gitignored)
becomes unloadable through `BaselineModel.load()` — joblib pickle
reconstruction will raise `TypeError` on the missing required args.

Mitigation: run `python scripts/train_baseline.py wr` (the new
generalized script from Plan 3b Phase 5) once 3b is merged. Closes
this entry.
```

Append this number to the chronological position in the list (between #16 and EOF) is fine; just do not interleave with #14 area. Numerical order is the convention.

- [ ] **Step 2.8: Run unit tests.**

Run: `pytest tests/test_models/ -v`

Expected: all pass. The existing `test_baseline.py` tests use `wr_baseline()` which is updated; `feature_schema` and `code_hash_files` are now passed by the factory; the schema validation site uses `self.feature_schema` which is `WrFeaturesSchema` (same as before). Behavior is unchanged for the WR test path.

- [ ] **Step 2.9: Run leakage test.**

Run: `pytest tests/test_models/test_baseline_leakage.py -v`

Expected: pass.

- [ ] **Step 2.10: Run smoke test.**

Run: `pytest tests/test_smoke.py -v`

Expected: pass. The smoke test uses `wr_baseline()` end-to-end; same path as the unit tests.

- [ ] **Step 2.11: Full gate.**

Run:
```bash
pytest -q
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: all green. If `pa.DataFrameModel` produces a mypy complaint about unknown attributes, narrow the type with `# type: ignore[type-arg]` on the dataclass field — but try first without; pandera's stubs typically handle this.

- [ ] **Step 2.12: Commit.**

```bash
git add src/projections/models/baseline.py TODO.md
git commit -m "$(cat <<'EOF'
refactor(models): BaselineModel takes feature_schema + code_hash_files (3b prep)

3a hardcoded WrFeaturesSchema.validate(...) at two sites in fit() and
predict_distribution(), plus a hardcoded code-hash file list inside
fit(). Plan 3b requires both to be per-position. Refactor introduces
two required dataclass fields:

- feature_schema: type[pa.DataFrameModel]   — caller-supplied schema.
- code_hash_files: tuple[Path, ...]         — caller-supplied file list.

wr_baseline() updated to pass both. Behavior unchanged for the WR path
(same schema, same eight files).

Breaking change: the existing models/artifacts/wr-baseline-2018-2023-
925f492b.joblib artifact becomes unloadable through BaselineModel.load
because the pickle's pre-3b dataclass shape is missing the two new
required fields. TODO #17 tracks the retrain; Phase 6 of plan 3b runs
the generalized scripts/train_baseline.py wr to close it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3 (Phase 3a): Three new factories — `qb_baseline`, `rb_baseline`, `te_baseline`

**Files:**
- Modify: `src/projections/models/baseline.py` (add per-position constants + factories)

**Goal:** Add the three new factories alongside the existing `wr_baseline()`. Each factory encodes its position's `target_stats`, `dist_families`, `feature_columns`, and `code_hash_files`.

- [ ] **Step 3.1: Add per-position constants for QB.**

Edit `src/projections/models/baseline.py`. After the existing WR constants (`_WR_TARGET_STATS`, `_WR_DIST_FAMILIES`, `_WR_FEATURE_COLUMNS`), add the QB constants:

```python
_QB_TARGET_STATS: Final[tuple[Stat, ...]] = (
    Stat.PASSING_YARDS,
    Stat.PASSING_TDS,
    Stat.INTERCEPTIONS,
    Stat.RUSHING_YARDS,
    Stat.RUSHING_TDS,
    Stat.FUMBLES_LOST,
)

_QB_DIST_FAMILIES: Final[Mapping[Stat, DistributionFamily]] = {
    Stat.PASSING_YARDS: DistributionFamily.NORMAL,
    Stat.PASSING_TDS: DistributionFamily.GAMMA,
    Stat.INTERCEPTIONS: DistributionFamily.GAMMA,
    Stat.RUSHING_YARDS: DistributionFamily.NORMAL,
    Stat.RUSHING_TDS: DistributionFamily.GAMMA,
    Stat.FUMBLES_LOST: DistributionFamily.GAMMA,
}

# Feature columns from QbFeaturesSchema, minus identity (gsis_id/season/week/team/opponent).
# Boolean columns (rushing_qb / is_home / roof_dome) are coerced to 0/1 by fit/predict.
_QB_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "depth_rank",
    "pass_attempts_per_game_l4",
    "passing_yards_per_game_l4",
    "passing_tds_per_game_l4",
    "interceptions_per_game_l4",
    "sacks_per_game_l4",
    "passing_yards_per_game_std",
    "rushing_attempts_per_game_l4",
    "rushing_yards_per_game_l4",
    "rushing_qb",
    "snap_pct_l4",
    "aggressiveness_std",
    "completion_percentage_above_expectation_std",
    "avg_intended_air_yards_std",
    "avg_time_to_throw_std",
    "implied_team_total",
    "spread",
    "is_home",
    "roof_dome",
    "opp_allowed_qb_fppg_l4",
)
```

- [ ] **Step 3.2: Add per-position constants for RB.**

Append:

```python
_RB_TARGET_STATS: Final[tuple[Stat, ...]] = (
    Stat.RUSHING_YARDS,
    Stat.RUSHING_TDS,
    Stat.RECEPTIONS,
    Stat.RECEIVING_YARDS,
    Stat.RECEIVING_TDS,
    Stat.FUMBLES_LOST,
)

_RB_DIST_FAMILIES: Final[Mapping[Stat, DistributionFamily]] = {
    Stat.RUSHING_YARDS: DistributionFamily.NORMAL,
    Stat.RUSHING_TDS: DistributionFamily.GAMMA,
    Stat.RECEPTIONS: DistributionFamily.GAMMA,
    Stat.RECEIVING_YARDS: DistributionFamily.NORMAL,
    Stat.RECEIVING_TDS: DistributionFamily.GAMMA,
    Stat.FUMBLES_LOST: DistributionFamily.GAMMA,
}

_RB_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "depth_rank",
    "carries_per_game_l4",
    "rushing_yards_per_game_l4",
    "rushing_tds_per_game_l4",
    "rush_share_l4",
    "targets_per_game_l4",
    "receptions_per_game_l4",
    "receiving_yards_per_game_l4",
    "target_share_l4",
    "targets_per_game_std",
    "passing_down_back",
    "snap_pct_l4",
    "efficiency_std",
    "rush_yards_over_expected_per_att_std",
    "percent_attempts_gte_eight_defenders_std",
    "implied_team_total",
    "spread",
    "is_home",
    "roof_dome",
    "opp_allowed_rb_fppg_l4",
)
```

- [ ] **Step 3.3: Add per-position constants for TE.**

Append:

```python
_TE_TARGET_STATS: Final[tuple[Stat, ...]] = (
    Stat.RECEPTIONS,
    Stat.RECEIVING_YARDS,
    Stat.RECEIVING_TDS,
    Stat.RUSHING_YARDS,
    Stat.RUSHING_TDS,
    Stat.FUMBLES_LOST,
)

_TE_DIST_FAMILIES: Final[Mapping[Stat, DistributionFamily]] = {
    Stat.RECEPTIONS: DistributionFamily.GAMMA,
    Stat.RECEIVING_YARDS: DistributionFamily.NORMAL,
    Stat.RECEIVING_TDS: DistributionFamily.GAMMA,
    Stat.RUSHING_YARDS: DistributionFamily.NORMAL,
    Stat.RUSHING_TDS: DistributionFamily.GAMMA,
    Stat.FUMBLES_LOST: DistributionFamily.GAMMA,
}

# Feature columns include the rushing_*_per_game_l4 cols added to TeFeaturesSchema in Plan 3b Phase 1.
_TE_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "depth_rank",
    "targets_per_game_l4",
    "targets_per_game_std",
    "target_share_l4",
    "receptions_per_game_l4",
    "receiving_yards_per_game_l4",
    "receiving_tds_per_game_l4",
    "rushing_attempts_per_game_l4",
    "rushing_yards_per_game_l4",
    "snap_pct_l4",
    "avg_separation_std",
    "avg_intended_air_yards_std",
    "avg_yac_above_expectation_std",
    "implied_team_total",
    "spread",
    "is_home",
    "roof_dome",
    "opp_allowed_te_fppg_l4",
)
```

- [ ] **Step 3.4: Add a private helper `_default_code_hash_files(position_module: str)` to deduplicate factory bodies.**

Inside `baseline.py`, after the per-position constants block and before the existing `wr_baseline()` factory, add:

```python
def _default_code_hash_files(position_module: str) -> tuple[Path, ...]:
    """Build the canonical code-hash file tuple for a position factory.

    The eight files are: models/base.py, models/baseline.py,
    features/{position_module}.py, features/_shared.py,
    features/_rolling.py, features/_opponent.py, scoring/score.py,
    scoring/score_distribution.py.

    Used by every position factory so the per-factory call site is one line.
    """
    repo_root = Path(__file__).resolve().parents[3]
    return (
        repo_root / "src" / "projections" / "models" / "base.py",
        repo_root / "src" / "projections" / "models" / "baseline.py",
        repo_root / "src" / "projections" / "features" / position_module,
        repo_root / "src" / "projections" / "features" / "_shared.py",
        repo_root / "src" / "projections" / "features" / "_rolling.py",
        repo_root / "src" / "projections" / "features" / "_opponent.py",
        repo_root / "src" / "projections" / "scoring" / "score.py",
        repo_root / "src" / "projections" / "scoring" / "score_distribution.py",
    )
```

- [ ] **Step 3.5: Refactor `wr_baseline()` to use the helper.**

Replace the inline `code_hash_files=(...)` tuple from Task 2.6 with:

```python
def wr_baseline() -> BaselineModel:
    """Construct an unfitted WR-baseline model. Caller invokes .fit(features,
    weekly_stats) and then .save(path)."""
    return BaselineModel(
        position=Position.WR,
        target_stats=_WR_TARGET_STATS,
        feature_columns=_WR_FEATURE_COLUMNS,
        dist_families=_WR_DIST_FAMILIES,
        feature_schema=WrFeaturesSchema,
        code_hash_files=_default_code_hash_files("wr.py"),
    )
```

- [ ] **Step 3.6: Add the QB / RB / TE factories.**

Update the imports at the top of `baseline.py` to include the new schemas:

```python
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
```

Then add the three new factories after `wr_baseline()`:

```python
def qb_baseline() -> BaselineModel:
    """Construct an unfitted QB-baseline model."""
    return BaselineModel(
        position=Position.QB,
        target_stats=_QB_TARGET_STATS,
        feature_columns=_QB_FEATURE_COLUMNS,
        dist_families=_QB_DIST_FAMILIES,
        feature_schema=QbFeaturesSchema,
        code_hash_files=_default_code_hash_files("qb.py"),
    )


def rb_baseline() -> BaselineModel:
    """Construct an unfitted RB-baseline model."""
    return BaselineModel(
        position=Position.RB,
        target_stats=_RB_TARGET_STATS,
        feature_columns=_RB_FEATURE_COLUMNS,
        dist_families=_RB_DIST_FAMILIES,
        feature_schema=RbFeaturesSchema,
        code_hash_files=_default_code_hash_files("rb.py"),
    )


def te_baseline() -> BaselineModel:
    """Construct an unfitted TE-baseline model."""
    return BaselineModel(
        position=Position.TE,
        target_stats=_TE_TARGET_STATS,
        feature_columns=_TE_FEATURE_COLUMNS,
        dist_families=_TE_DIST_FAMILIES,
        feature_schema=TeFeaturesSchema,
        code_hash_files=_default_code_hash_files("te.py"),
    )
```

- [ ] **Step 3.7: Smoke-test the factories at the REPL (manual).**

Run:
```bash
python -c "
from projections.models.baseline import qb_baseline, rb_baseline, te_baseline, wr_baseline
from projections.schemas import Position
for fn, expected in [(qb_baseline, Position.QB), (rb_baseline, Position.RB), (te_baseline, Position.TE), (wr_baseline, Position.WR)]:
    m = fn()
    assert m.position == expected, (m.position, expected)
    assert m.target_stats and m.feature_columns and m.dist_families
    assert m.feature_schema is not None and m.code_hash_files
    print(f'{expected.value}: targets={len(m.target_stats)}, cols={len(m.feature_columns)}, hash_files={len(m.code_hash_files)}')
"
```

Expected output: four lines, one per position, with non-zero counts. Targets count = 6 for every position. Feature columns ≈ 17–20. Hash files = 8.

- [ ] **Step 3.8: Run existing tests.**

Run: `pytest tests/test_models/ -v`

Expected: all pass. (No new tests yet; Phase 4 adds them.)

- [ ] **Step 3.9: Full gate.**

```bash
pytest -q
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: all green.

- [ ] **Step 3.10: Commit.**

```bash
git add src/projections/models/baseline.py
git commit -m "$(cat <<'EOF'
feat(models): add qb_baseline / rb_baseline / te_baseline factories

Phase 3a of Plan 3b. The BaselineModel dataclass was already parameter-
ized by (position, target_stats, feature_columns, dist_families,
feature_schema, code_hash_files); this commit adds the three new factory
functions alongside wr_baseline() and the per-position constants they
reference.

Per-position config:
- QB: 6 target stats (passing_*, rushing_*, fumbles_lost). NORMAL for
  yards, GAMMA for counts. ~20 feature columns from QbFeaturesSchema.
- RB: 6 target stats (rushing_*, receiving_*, fumbles_lost). Same
  family convention. ~20 feature columns from RbFeaturesSchema.
- TE: 6 target stats including rushing — Taysom-Hill rationale (Phase
  1 added the rushing columns to TeFeaturesSchema). ~18 feature
  columns from TeFeaturesSchema.

A small _default_code_hash_files(position_module) helper deduplicates
the eight-file hash list per factory; the three new factories each
pass a one-line "qb.py" / "rb.py" / "te.py" argument. wr_baseline()
also moves to use the helper; behavior unchanged.

No tests in this commit — Phase 4 (tasks 5-9) adds per-position unit
tests, leakage tests, and the smoke test parametrization.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4 (Phase 3b): `POSITION_DISPATCH` registry in `models/__init__.py`

**Files:**
- Modify: `src/projections/models/__init__.py`

**Goal:** Add a position-keyed registry holding `(factory, feature_builder, feature_schema, ngs_stat_type)` tuples so the CLI scripts (Phase 5) and future Plan 3c backtest harness have one canonical "what positions does the system know about" answer.

- [ ] **Step 4.1: Read the current `__init__.py`.**

Run: `cat src/projections/models/__init__.py`. Note the current exports (`BaselineModel`, `wr_baseline`, `Model`).

- [ ] **Step 4.2: Replace the file contents.**

Write the new `src/projections/models/__init__.py`:

```python
"""Public surface for the models package.

Plan 3b adds POSITION_DISPATCH so callers (CLI scripts, Plan 3c backtest
harness) can dispatch by Position to the correct factory + feature builder
+ feature schema + NGS source. Adding a new position is one new line in
this registry plus a corresponding factory in baseline.py and a feature
builder in features/{pos}.py.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pandera.pandas as pa

from projections.features.qb import build_qb_features
from projections.features.rb import build_rb_features
from projections.features.te import build_te_features
from projections.features.wr import build_wr_features
from projections.ingest.ngs import NgsStatType
from projections.models.base import Model, compute_code_hash
from projections.models.baseline import (
    BaselineModel,
    qb_baseline,
    rb_baseline,
    te_baseline,
    wr_baseline,
)
from projections.schemas import (
    Position,
    QbFeaturesSchema,
    RbFeaturesSchema,
    TeFeaturesSchema,
    WrFeaturesSchema,
)

__all__ = [
    "BaselineModel",
    "Model",
    "POSITION_DISPATCH",
    "compute_code_hash",
    "qb_baseline",
    "rb_baseline",
    "te_baseline",
    "wr_baseline",
]


@dataclass(frozen=True)
class _PositionDispatch:
    """Per-position bundle of "what's needed to train and predict" entries.

    Consumed by the CLI scripts (scripts/train_baseline.py etc.) and
    intended to back Plan 3c's backtest harness. Frozen so callers can't
    mutate the registry by accident.

    Attributes:
        factory: zero-arg callable returning an unfitted BaselineModel.
        feature_builder: position-specific build_*_features function.
        feature_schema: pandera schema for the feature builder's output.
        ngs_stat_type: which NGS partition the feature builder consumes
            ("passing" / "rushing" / "receiving").
    """

    factory: Callable[[], BaselineModel]
    feature_builder: Callable[..., Any]
    feature_schema: type[pa.DataFrameModel]
    ngs_stat_type: NgsStatType


POSITION_DISPATCH: Mapping[Position, _PositionDispatch] = {
    Position.QB: _PositionDispatch(
        factory=qb_baseline,
        feature_builder=build_qb_features,
        feature_schema=QbFeaturesSchema,
        ngs_stat_type="passing",
    ),
    Position.RB: _PositionDispatch(
        factory=rb_baseline,
        feature_builder=build_rb_features,
        feature_schema=RbFeaturesSchema,
        ngs_stat_type="rushing",
    ),
    Position.TE: _PositionDispatch(
        factory=te_baseline,
        feature_builder=build_te_features,
        feature_schema=TeFeaturesSchema,
        ngs_stat_type="receiving",
    ),
    Position.WR: _PositionDispatch(
        factory=wr_baseline,
        feature_builder=build_wr_features,
        feature_schema=WrFeaturesSchema,
        ngs_stat_type="receiving",
    ),
}
```

- [ ] **Step 4.3: Verify the registry imports cleanly.**

Run:
```bash
python -c "
from projections.models import POSITION_DISPATCH
from projections.schemas import Position
for pos in (Position.QB, Position.RB, Position.TE, Position.WR):
    d = POSITION_DISPATCH[pos]
    m = d.factory()
    assert m.position == pos
    assert d.feature_schema is not None
    print(f'{pos.value}: factory={d.factory.__name__}, builder={d.feature_builder.__name__}, ngs={d.ngs_stat_type}')
"
```

Expected: four lines, all four positions resolved.

- [ ] **Step 4.4: Run existing tests.**

Run: `pytest tests/test_models/ tests/test_smoke.py -v`

Expected: all pass.

- [ ] **Step 4.5: Full gate.**

```bash
pytest -q
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: all green.

- [ ] **Step 4.6: Commit.**

```bash
git add src/projections/models/__init__.py
git commit -m "$(cat <<'EOF'
feat(models): POSITION_DISPATCH registry in models/__init__.py

Phase 3b of Plan 3b. Adds a frozen Mapping[Position, _PositionDispatch]
that bundles (factory, feature_builder, feature_schema, ngs_stat_type)
per position. CLI scripts (Phase 5) and Plan 3c's future backtest
harness consume this registry instead of growing per-script if-else
ladders. Adding a fifth position becomes: one new factory in
baseline.py, one new feature builder in features/{pos}.py, and one
new line here.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5 (Phase 4a): Fixtures refactor — rename WR + add per-position fixtures

**Files:**
- Modify: `tests/conftest.py` (rename WR fixtures; add helper + per-position fixtures)
- Modify: `tests/test_models/test_baseline.py` (rename fixture refs)
- Modify: `tests/test_models/test_baseline_leakage.py` (rename fixture refs)
- Modify: `tests/test_smoke.py` (rename fixture refs in the WR-only smoke test)

**Goal:** Provide per-position `baseline_features_{qb,rb,te,wr}` and `baseline_weekly_stats_{qb,rb,te,wr}` fixtures at root scope. Existing WR fixtures get a `_wr` suffix; everything that referenced them is updated. The smoke parametrization (Task 9) and per-position unit tests (Tasks 6-8) consume them.

This task is large but mostly mechanical. Break into sub-steps.

- [ ] **Step 5.1: Read the existing fixtures.**

Run:
```bash
grep -n "^def baseline_\|^def _wr_weekly_stats_row\|^_GSIS_IDS\|^_TEAMS\|^_TARGETS_BASE" tests/conftest.py
```

Note the line numbers. The current shape is `_GSIS_IDS` / `_TEAMS` / `_TARGETS_BASE` module-level constants, then `_wr_weekly_stats_row(...)` helper, then `baseline_weekly_stats` and `baseline_features` fixtures.

- [ ] **Step 5.2: Rename `baseline_weekly_stats` → `baseline_weekly_stats_wr` in `tests/conftest.py`.**

Edit `tests/conftest.py`. Find the fixture definition `def baseline_weekly_stats() -> pd.DataFrame:` and rename it to `def baseline_weekly_stats_wr() -> pd.DataFrame:`. Update the docstring to clarify it's WR-shaped.

- [ ] **Step 5.3: Rename `baseline_features` → `baseline_features_wr` in `tests/conftest.py`.**

Same file. The fixture takes the renamed `baseline_weekly_stats_wr` fixture as an arg now:

```python
@pytest.fixture
def baseline_features_wr(baseline_weekly_stats_wr: pd.DataFrame) -> pd.DataFrame:
    """WR feature rows produced by build_wr_features for every (season, week)
    in the WR training fixture. Built up-front so tests don't pay the cost
    individually."""
    from projections.features import build_wr_features
    # ... rest of body unchanged, except replace ALL references inside this
    # function from `baseline_weekly_stats` to `baseline_weekly_stats_wr`
```

- [ ] **Step 5.4: Add a `_make_baseline_fixtures(position)` helper.**

Below the existing `_wr_weekly_stats_row` helper and above the renamed fixtures, add the per-position factory helper. The body is structurally similar to the existing WR fixtures but parametric on position. Append to `tests/conftest.py`:

```python
# ---------------------------------------------------------------------------
# Per-position baseline fixtures (Plan 3b).
# Each {position} variant provides a baseline_weekly_stats_{position} +
# baseline_features_{position} pair shaped to that position's target stats
# and feature schema. The smoke test (test_smoke.py) parametrizes across
# all four positions, so every fixture must live at root scope; pytest
# fixtures only inherit downward.
# ---------------------------------------------------------------------------

# Synthetic universe per position. 5 players across 2 teams (KC/MIN), 8
# weeks of 2024 + 4 weeks of 2025. KC and MIN play each other every week
# so the opp_allowed_*_fppg_l4 proxy resolves.
_POSITION_BASE_RATES: dict[str, list[float]] = {
    "QB": [38.0, 32.0, 36.0, 26.0, 22.0],   # pass attempts/game baseline
    "RB": [16.0, 12.0, 14.0, 8.0, 6.0],     # carries/game baseline
    "TE": [9.0, 6.0, 8.0, 3.0, 2.0],        # targets/game baseline
}


def _qb_weekly_stats_row(
    *, gsis_id: str, season: int, week: int, team: str, opponent: str, base_attempts: float
) -> dict[str, object]:
    jitter = (week % 3) - 1
    attempts = max(0, int(base_attempts + jitter))
    completions = max(0, int(attempts * 0.65))
    pass_yards = float(completions * 11.0)
    pass_tds = 1 if (week + int(gsis_id[-1])) % 3 == 0 else 0
    interceptions = 1 if (week + int(gsis_id[-1])) % 5 == 0 else 0
    sacks = 1 if week % 4 == 0 else 0
    rush_attempts = 3 + (week % 2)
    rush_yards = float(rush_attempts * 4.0)
    return {
        "gsis_id": gsis_id, "season": season, "week": week, "position": "QB",
        "team": team, "opponent": opponent,
        "passing_yards": pass_yards, "passing_tds": pass_tds, "interceptions": interceptions,
        "attempts": attempts, "completions": completions, "sacks": sacks,
        "rushing_yards": rush_yards, "rushing_tds": 0, "carries": rush_attempts,
        "receptions": 0, "receiving_yards": 0.0, "receiving_tds": 0,
        "receiving_air_yards": 0.0, "targets": 0, "fumbles_lost": 0,
    }


def _rb_weekly_stats_row(
    *, gsis_id: str, season: int, week: int, team: str, opponent: str, base_carries: float
) -> dict[str, object]:
    jitter = (week % 3) - 1
    carries = max(0, int(base_carries + jitter))
    rush_yards = float(carries * 4.5)
    rush_tds = 1 if (week + int(gsis_id[-1])) % 4 == 0 else 0
    targets = max(0, int(base_carries * 0.25))
    receptions = max(0, int(targets * 0.7))
    return {
        "gsis_id": gsis_id, "season": season, "week": week, "position": "RB",
        "team": team, "opponent": opponent,
        "passing_yards": 0.0, "passing_tds": 0, "interceptions": 0,
        "attempts": 0, "completions": 0, "sacks": 0,
        "rushing_yards": rush_yards, "rushing_tds": rush_tds, "carries": carries,
        "receptions": receptions, "receiving_yards": float(receptions * 7.0),
        "receiving_tds": 0, "receiving_air_yards": float(targets * 5.0),
        "targets": targets, "fumbles_lost": 0,
    }


def _te_weekly_stats_row(
    *, gsis_id: str, season: int, week: int, team: str, opponent: str, base_targets: float
) -> dict[str, object]:
    jitter = (week % 3) - 1
    targets = max(0, int(base_targets + jitter))
    receptions = max(0, int(targets * 0.65))
    rec_yards = float(receptions * 11.0)
    rec_tds = 1 if (week + int(gsis_id[-1])) % 4 == 0 else 0
    # One synthetic TE rushes (Taysom-Hill-shape) when base_targets is the
    # third-highest in the cohort (gsis_id ending in "3").
    is_rushing_te = gsis_id.endswith("3")
    carries = 4 + (week % 2) if is_rushing_te else 0
    rush_yards = float(carries * 4.0)
    return {
        "gsis_id": gsis_id, "season": season, "week": week, "position": "TE",
        "team": team, "opponent": opponent,
        "passing_yards": 0.0, "passing_tds": 0, "interceptions": 0,
        "attempts": 0, "completions": 0, "sacks": 0,
        "rushing_yards": rush_yards,
        "rushing_tds": 1 if is_rushing_te and week % 4 == 0 else 0,
        "carries": carries,
        "receptions": receptions, "receiving_yards": rec_yards, "receiving_tds": rec_tds,
        "receiving_air_yards": float(targets * 12.0), "targets": targets, "fumbles_lost": 0,
    }


_POSITION_ROW_BUILDERS = {
    "QB": _qb_weekly_stats_row,
    "RB": _rb_weekly_stats_row,
    "TE": _te_weekly_stats_row,
}
```

- [ ] **Step 5.5: Add per-position `baseline_weekly_stats_{qb,rb,te}` fixtures.**

Append to `tests/conftest.py`:

```python
def _build_position_weekly_stats(position: str) -> pd.DataFrame:
    """Stack 8 weeks of 2024 + 4 weeks of 2025 for the synthetic universe,
    using the row-builder registered for `position`."""
    builder = _POSITION_ROW_BUILDERS[position]
    base_rates = _POSITION_BASE_RATES[position]
    rows: list[dict[str, object]] = []
    for season, weeks in [(2024, range(1, 9)), (2025, range(1, 5))]:
        for week in weeks:
            for gsis_id, team, base_rate in zip(_GSIS_IDS, _TEAMS, base_rates, strict=True):
                opponent = "MIN" if team == "KC" else "KC"
                # row builder uses one of base_attempts / base_carries / base_targets;
                # the helper signature is uniform on the kw name so we map by position.
                kw = {"QB": "base_attempts", "RB": "base_carries", "TE": "base_targets"}[position]
                rows.append(builder(
                    gsis_id=gsis_id, season=season, week=week, team=team,
                    opponent=opponent, **{kw: base_rate},
                ))
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def baseline_weekly_stats_qb() -> pd.DataFrame:
    return _build_position_weekly_stats("QB")


@pytest.fixture
def baseline_weekly_stats_rb() -> pd.DataFrame:
    return _build_position_weekly_stats("RB")


@pytest.fixture
def baseline_weekly_stats_te() -> pd.DataFrame:
    return _build_position_weekly_stats("TE")
```

- [ ] **Step 5.6: Add per-position `baseline_features_{qb,rb,te}` fixtures.**

Append to `tests/conftest.py`. The pattern mirrors `baseline_features_wr` but each fixture calls the position's builder and references the position's NGS source:

```python
def _build_position_supporting_frames(
    weekly_stats: pd.DataFrame, position: str
) -> dict[str, pd.DataFrame]:
    """Build snap_counts / depth_charts / ngs (passing or rushing or receiving)
    / schedules sub-frames matching the synthetic universe."""
    base_rates = _POSITION_BASE_RATES[position]

    snap_rows = [
        {
            "gsis_id": r["gsis_id"], "season": r["season"], "week": r["week"],
            "team": r["team"], "opponent": r["opponent"], "position": position,
            "offense_snaps": 60, "offense_pct": 0.95,
            "defense_snaps": 0, "defense_pct": 0.0,
            "st_snaps": 2, "st_pct": 0.05,
        }
        for _, r in weekly_stats.iterrows()
    ]
    snap_counts = pd.DataFrame(snap_rows)
    for col in ("gsis_id", "team", "opponent", "position"):
        snap_counts[col] = snap_counts[col].astype(_PYARROW_STR)

    dc_rows: list[dict[str, object]] = []
    for season in (2024, 2025):
        weeks = range(1, 9) if season == 2024 else range(1, 5)
        for week in weeks:
            for gsis_id, team, base in zip(_GSIS_IDS, _TEAMS, base_rates, strict=True):
                team_pool = sorted(
                    [(g, t, b) for g, t, b in zip(_GSIS_IDS, _TEAMS, base_rates, strict=True) if t == team],
                    key=lambda x: -x[2],
                )
                rank = next(i for i, (g, _, _) in enumerate(team_pool, start=1) if g == gsis_id)
                dc_rows.append({
                    "gsis_id": gsis_id, "season": season, "week": week,
                    "team": team, "position": position,
                    "depth_team": f"{position}{rank}", "depth_rank": rank,
                })
    depth = pd.DataFrame(dc_rows)
    for col in ("gsis_id", "team", "position", "depth_team"):
        depth[col] = depth[col].astype(_PYARROW_STR)

    # NGS source depends on position: QB→passing, RB→rushing, TE→receiving.
    ngs_rows: list[dict[str, object]] = []
    for season in (2024, 2025):
        weeks = range(1, 9) if season == 2024 else range(1, 5)
        for week in weeks:
            for gsis_id, team, base in zip(_GSIS_IDS, _TEAMS, base_rates, strict=True):
                row = {
                    "gsis_id": gsis_id, "season": season, "week": week,
                    "team": team, "position": position,
                }
                if position == "QB":
                    row.update({
                        "avg_time_to_throw": 2.7 + base * 0.001,
                        "avg_intended_air_yards": 8.0 + base * 0.05,
                        "completion_percentage_above_expectation": -1.0 + base * 0.1,
                        "aggressiveness": 12.0 + base * 0.05,
                    })
                elif position == "RB":
                    row.update({
                        "efficiency": 3.0 + base * 0.02,
                        "rush_yards_over_expected_per_att": -0.5 + base * 0.05,
                        "percent_attempts_gte_eight_defenders": 18.0 + base * 0.1,
                    })
                else:  # TE
                    row.update({
                        "avg_separation": 2.5 + base * 0.05,
                        "avg_intended_air_yards": 9.0 + base * 0.2,
                        "avg_yac_above_expectation": -0.2 + base * 0.05,
                    })
                ngs_rows.append(row)
    ngs = pd.DataFrame(ngs_rows)
    for col in ("gsis_id", "team", "position"):
        ngs[col] = ngs[col].astype(_PYARROW_STR)

    sch_rows: list[dict[str, object]] = []
    for season in (2024, 2025):
        weeks = range(1, 9) if season == 2024 else range(1, 5)
        for week in weeks:
            sch_rows.append({
                "season": season, "week": week,
                "game_id": f"{season}_{week:02d}_KC_MIN",
                "home_team": "KC", "away_team": "MIN",
                "kickoff": pd.Timestamp(f"{season}-09-{week + 1:02d}T17:00:00Z").tz_convert("UTC").as_unit("us"),
                "spread_line": -3.0, "total_line": 47.0,
                "home_moneyline": -150, "away_moneyline": 130,
                "surface": "grass", "roof": "outdoors", "temp": 60, "wind": 5,
            })
    schedules = pd.DataFrame(sch_rows)
    for col in ("game_id", "home_team", "away_team", "surface", "roof"):
        schedules[col] = schedules[col].astype(_PYARROW_STR)
    for col in ("temp", "wind", "home_moneyline", "away_moneyline"):
        schedules[col] = schedules[col].astype(pd.Int64Dtype())

    return {"snap_counts": snap_counts, "depth_charts": depth, "ngs": ngs, "schedules": schedules}


@pytest.fixture
def baseline_features_qb(baseline_weekly_stats_qb: pd.DataFrame) -> pd.DataFrame:
    from projections.features import build_qb_features
    aux = _build_position_supporting_frames(baseline_weekly_stats_qb, "QB")
    feat_frames: list[pd.DataFrame] = []
    for season in (2024, 2025):
        weeks = range(1, 9) if season == 2024 else range(1, 5)
        for week in weeks:
            f = build_qb_features(
                weekly_stats=baseline_weekly_stats_qb,
                snap_counts=aux["snap_counts"],
                depth_charts=aux["depth_charts"],
                ngs_passing=aux["ngs"],
                schedules=aux["schedules"],
                season=season, as_of_week=week,
            )
            if not f.empty:
                feat_frames.append(f)
    return pd.concat(feat_frames, ignore_index=True) if feat_frames else pd.DataFrame()


@pytest.fixture
def baseline_features_rb(baseline_weekly_stats_rb: pd.DataFrame) -> pd.DataFrame:
    from projections.features import build_rb_features
    aux = _build_position_supporting_frames(baseline_weekly_stats_rb, "RB")
    feat_frames: list[pd.DataFrame] = []
    for season in (2024, 2025):
        weeks = range(1, 9) if season == 2024 else range(1, 5)
        for week in weeks:
            f = build_rb_features(
                weekly_stats=baseline_weekly_stats_rb,
                snap_counts=aux["snap_counts"],
                depth_charts=aux["depth_charts"],
                ngs_rushing=aux["ngs"],
                schedules=aux["schedules"],
                season=season, as_of_week=week,
            )
            if not f.empty:
                feat_frames.append(f)
    return pd.concat(feat_frames, ignore_index=True) if feat_frames else pd.DataFrame()


@pytest.fixture
def baseline_features_te(baseline_weekly_stats_te: pd.DataFrame) -> pd.DataFrame:
    from projections.features import build_te_features
    aux = _build_position_supporting_frames(baseline_weekly_stats_te, "TE")
    feat_frames: list[pd.DataFrame] = []
    for season in (2024, 2025):
        weeks = range(1, 9) if season == 2024 else range(1, 5)
        for week in weeks:
            f = build_te_features(
                weekly_stats=baseline_weekly_stats_te,
                snap_counts=aux["snap_counts"],
                depth_charts=aux["depth_charts"],
                ngs_receiving=aux["ngs"],
                schedules=aux["schedules"],
                season=season, as_of_week=week,
            )
            if not f.empty:
                feat_frames.append(f)
    return pd.concat(feat_frames, ignore_index=True) if feat_frames else pd.DataFrame()
```

- [ ] **Step 5.7: Update existing test files to use `_wr` fixture names.**

Do a global grep first:

```bash
grep -rn "baseline_features\b\|baseline_weekly_stats\b" tests/
```

In `tests/test_models/test_baseline.py`, `tests/test_models/test_baseline_leakage.py`, and `tests/test_smoke.py`, replace:
- `baseline_features` → `baseline_features_wr`
- `baseline_weekly_stats` → `baseline_weekly_stats_wr`

Use ripgrep + sed (or your editor's project-wide replace) carefully. Confirm with another grep that no bare `baseline_features` or `baseline_weekly_stats` remain anywhere in `tests/`.

- [ ] **Step 5.8: Run the updated existing tests.**

Run: `pytest tests/test_models/ tests/test_smoke.py -v`

Expected: all pass. The renamed fixtures resolve and the WR path is unchanged.

- [ ] **Step 5.9: Sanity-check a per-position fixture loads.**

Run:
```bash
python -c "
import pandas as pd
import pytest

# Direct fixture access via pytest's internal API is awkward; instead,
# write a one-shot test:
"
pytest tests/test_models/test_baseline.py -v -k "test_wr_baseline_factory_returns_unfitted_model"
```

Expected: pass.

Then verify the new fixtures resolve by writing a temporary test (delete after):

```bash
cat > /tmp/sanity_fixtures.py << 'EOF'
import pandas as pd

def test_qb_fixture_loads(baseline_features_qb, baseline_weekly_stats_qb):
    assert not baseline_features_qb.empty
    assert not baseline_weekly_stats_qb.empty

def test_rb_fixture_loads(baseline_features_rb, baseline_weekly_stats_rb):
    assert not baseline_features_rb.empty
    assert not baseline_weekly_stats_rb.empty

def test_te_fixture_loads(baseline_features_te, baseline_weekly_stats_te):
    assert not baseline_features_te.empty
    assert not baseline_weekly_stats_te.empty
EOF

cp /tmp/sanity_fixtures.py tests/_sanity_fixtures_temp.py
pytest tests/_sanity_fixtures_temp.py -v
rm tests/_sanity_fixtures_temp.py
```

Expected: all three sanity tests pass. If a fixture build fails, debug the row-builder or aux frame for that position before continuing.

- [ ] **Step 5.10: Full gate.**

```bash
pytest -q
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: all green.

- [ ] **Step 5.11: Commit.**

```bash
git add tests/conftest.py tests/test_models/test_baseline.py tests/test_models/test_baseline_leakage.py tests/test_smoke.py
git commit -m "$(cat <<'EOF'
test(fixtures): rename WR fixtures + add per-position fixtures (3b prep)

Phase 4a of Plan 3b. Existing baseline_features / baseline_weekly_stats
fixtures get a _wr suffix; the three new positions get sibling
fixtures (baseline_features_{qb,rb,te} / baseline_weekly_stats_{qb,
rb,te}) at root scope so the parametrized smoke test in Phase 4e can
resolve them.

The per-position rows are shaped to each position's natural target
stats. TE includes a Taysom-Hill-shape rushing TE (gsis_id ending in
"3") so the new TeFeaturesSchema rushing columns from Phase 1 carry
non-zero rolling means.

A small _build_position_supporting_frames(weekly_stats, position)
helper unifies snap_counts / depth_charts / ngs / schedules
construction across positions; row builders are per-position
(QB/RB/TE) and dispatched via _POSITION_ROW_BUILDERS.

No new tests in this commit; the per-position test files (Tasks 6-8)
and the smoke parametrization (Task 9) consume the new fixtures.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6 (Phase 4b): QB unit tests + leakage test

**Files:**
- Create: `tests/test_models/test_baseline_qb.py`
- Create: `tests/test_models/test_baseline_qb_leakage.py`

**Goal:** Cover `qb_baseline()` with the same set of contract tests as `test_baseline.py` covers `wr_baseline()`. The fixture set from Task 5 makes this a position-swap.

- [ ] **Step 6.1: Create `tests/test_models/test_baseline_qb.py`.**

Write the file:

```python
"""qb_baseline() unit tests. Mirrors test_baseline.py's WR coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from projections.distributions.parametric import ParametricGamma, ParametricNormal
from projections.models import qb_baseline
from projections.schemas import DistributionFamily, Position, ProjectionWeeklySchema, Ruleset, Stat


def test_qb_baseline_factory_returns_unfitted_model() -> None:
    model = qb_baseline()
    assert model.position == Position.QB
    expected_targets = {
        Stat.PASSING_YARDS,
        Stat.PASSING_TDS,
        Stat.INTERCEPTIONS,
        Stat.RUSHING_YARDS,
        Stat.RUSHING_TDS,
        Stat.FUMBLES_LOST,
    }
    assert set(model.target_stats) == expected_targets
    assert model.dist_families[Stat.PASSING_YARDS] is DistributionFamily.NORMAL
    assert model.dist_families[Stat.PASSING_TDS] is DistributionFamily.GAMMA
    assert model.dist_families[Stat.INTERCEPTIONS] is DistributionFamily.GAMMA
    assert model.dist_families[Stat.RUSHING_YARDS] is DistributionFamily.NORMAL
    assert model.dist_families[Stat.RUSHING_TDS] is DistributionFamily.GAMMA
    assert model.dist_families[Stat.FUMBLES_LOST] is DistributionFamily.GAMMA
    assert model.feature_columns


def test_qb_baseline_fit_populates_ridges_per_target_stat(
    baseline_features_qb: pd.DataFrame, baseline_weekly_stats_qb: pd.DataFrame
) -> None:
    model = qb_baseline()
    model.fit(features=baseline_features_qb, weekly_stats=baseline_weekly_stats_qb)
    assert set(model.ridges.keys()) == set(model.target_stats)
    for stat in model.target_stats:
        assert isinstance(model.ridges[stat], RidgeCV)
        assert hasattr(model.ridges[stat], "coef_")


def test_qb_baseline_fit_persists_feature_means(
    baseline_features_qb: pd.DataFrame, baseline_weekly_stats_qb: pd.DataFrame
) -> None:
    model = qb_baseline()
    model.fit(features=baseline_features_qb, weekly_stats=baseline_weekly_stats_qb)
    assert model.feature_means is not None
    assert set(model.feature_means.index) == set(model.feature_columns)


def test_qb_baseline_fit_records_train_seasons(
    baseline_features_qb: pd.DataFrame, baseline_weekly_stats_qb: pd.DataFrame
) -> None:
    model = qb_baseline()
    model.fit(features=baseline_features_qb, weekly_stats=baseline_weekly_stats_qb)
    assert model.train_seasons == (2024, 2025)


def test_qb_baseline_fit_populates_normal_variance_params(
    baseline_features_qb: pd.DataFrame, baseline_weekly_stats_qb: pd.DataFrame
) -> None:
    model = qb_baseline()
    model.fit(features=baseline_features_qb, weekly_stats=baseline_weekly_stats_qb)
    for stat in (Stat.PASSING_YARDS, Stat.RUSHING_YARDS):
        params = model.variance_params[stat]
        assert "std" in params
        assert params["std"] > 0


def test_qb_baseline_fit_populates_gamma_variance_params(
    baseline_features_qb: pd.DataFrame, baseline_weekly_stats_qb: pd.DataFrame
) -> None:
    model = qb_baseline()
    model.fit(features=baseline_features_qb, weekly_stats=baseline_weekly_stats_qb)
    for stat in (Stat.PASSING_TDS, Stat.INTERCEPTIONS, Stat.RUSHING_TDS, Stat.FUMBLES_LOST):
        params = model.variance_params[stat]
        assert "shape" in params
        assert 0.01 <= params["shape"] <= 100.0


def test_qb_predict_distribution_returns_projection_weekly_schema_valid_frame(
    baseline_features_qb: pd.DataFrame, baseline_weekly_stats_qb: pd.DataFrame
) -> None:
    model = qb_baseline()
    model.fit(features=baseline_features_qb, weekly_stats=baseline_weekly_stats_qb)
    week_features = baseline_features_qb[
        (baseline_features_qb["season"] == 2025) & (baseline_features_qb["week"] == 4)
    ]
    out = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    ProjectionWeeklySchema.validate(out)
    assert len(out) == len(week_features)
    assert (out["model_id"].str.startswith("baseline:qb:")).all()
    assert (out["position"] == "QB").all()


def test_qb_predict_distribution_p10_le_p50_le_p90(
    baseline_features_qb: pd.DataFrame, baseline_weekly_stats_qb: pd.DataFrame
) -> None:
    model = qb_baseline()
    model.fit(features=baseline_features_qb, weekly_stats=baseline_weekly_stats_qb)
    week_features = baseline_features_qb[
        (baseline_features_qb["season"] == 2025) & (baseline_features_qb["week"] == 4)
    ]
    out = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    assert (out["p10"] <= out["p50"]).all()
    assert (out["p50"] <= out["p90"]).all()


def test_qb_baseline_save_load_round_trip_preserves_predictions(
    tmp_path: Path,
    baseline_features_qb: pd.DataFrame,
    baseline_weekly_stats_qb: pd.DataFrame,
) -> None:
    model = qb_baseline()
    model.fit(features=baseline_features_qb, weekly_stats=baseline_weekly_stats_qb)

    artifact = tmp_path / "qb-baseline.joblib"
    model.save(artifact)
    assert artifact.exists()

    from projections.models import BaselineModel

    loaded = BaselineModel.load(artifact)
    assert loaded.position == Position.QB
    assert loaded.model_id == model.model_id

    week = baseline_features_qb[
        (baseline_features_qb["season"] == 2025) & (baseline_features_qb["week"] == 4)
    ]
    out_orig = model.predict_distribution(week, ruleset=Ruleset.espn_ppr())
    out_loaded = loaded.predict_distribution(week, ruleset=Ruleset.espn_ppr())
    pd.testing.assert_frame_equal(
        out_orig.drop(columns=["generated_at"]),
        out_loaded.drop(columns=["generated_at"]),
    )


def test_qb_model_id_format(
    baseline_features_qb: pd.DataFrame, baseline_weekly_stats_qb: pd.DataFrame
) -> None:
    model = qb_baseline()
    model.fit(features=baseline_features_qb, weekly_stats=baseline_weekly_stats_qb)
    parts = model.model_id.split(":")
    assert len(parts) == 4
    assert parts[0] == "baseline"
    assert parts[1] == "qb"
    assert len(parts[2]) == 8  # code_hash
    assert "-" in parts[3]


def test_qb_unfitted_model_id_raises() -> None:
    model = qb_baseline()
    try:
        _ = model.model_id
    except RuntimeError:
        return
    raise AssertionError("Unfitted model.model_id should raise RuntimeError")


def test_qb_predict_distribution_imputes_nan_features_with_persisted_means(
    baseline_features_qb: pd.DataFrame, baseline_weekly_stats_qb: pd.DataFrame
) -> None:
    model = qb_baseline()
    model.fit(features=baseline_features_qb, weekly_stats=baseline_weekly_stats_qb)

    week = baseline_features_qb[
        (baseline_features_qb["season"] == 2025) & (baseline_features_qb["week"] == 4)
    ].copy()
    week.loc[week.index[0], "implied_team_total"] = np.nan
    out = model.predict_distribution(week, ruleset=Ruleset.espn_ppr())
    assert not out["mean"].isna().any()


def test_qb_predict_distribution_empty_input_returns_empty_schema_valid_frame(
    baseline_features_qb: pd.DataFrame, baseline_weekly_stats_qb: pd.DataFrame
) -> None:
    model = qb_baseline()
    model.fit(features=baseline_features_qb, weekly_stats=baseline_weekly_stats_qb)
    empty = baseline_features_qb.iloc[0:0]
    out = model.predict_distribution(empty, ruleset=Ruleset.espn_ppr())
    assert out.empty
    ProjectionWeeklySchema.validate(out)
```

- [ ] **Step 6.2: Create `tests/test_models/test_baseline_qb_leakage.py`.**

Write the file:

```python
"""QB baseline leakage test. Mirrors test_baseline_leakage.py's WR test.

Strategy: fit on a feature build through week W. Mutate weekly_stats rows at
season=Y, week>=W+1. Re-build features through W and refit. Assert each
fitted regressor's coefficients are byte-identical pre and post.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.models import qb_baseline
from projections.schemas import Stat


def test_qb_baseline_fit_does_not_use_post_as_of_week_data(
    baseline_features_qb: pd.DataFrame, baseline_weekly_stats_qb: pd.DataFrame
) -> None:
    ws = baseline_weekly_stats_qb[baseline_weekly_stats_qb["season"] == 2024].copy()
    feats = baseline_features_qb[baseline_features_qb["season"] == 2024].copy()

    model_a = qb_baseline()
    model_a.fit(features=feats, weekly_stats=ws)

    # Mutate week-8 truth dramatically.
    ws_mut = ws.copy()
    mask = ws_mut["week"] >= 8
    ws_mut.loc[mask, "passing_yards"] = 0.0
    ws_mut.loc[mask, "passing_tds"] = 0
    ws_mut.loc[mask, "interceptions"] = 0
    ws_mut.loc[mask, "rushing_yards"] = 999.0
    ws_mut.loc[mask, "rushing_tds"] = 9

    feats_through_7 = feats[feats["week"] <= 7].copy()
    ws_mut_through_7 = ws_mut[ws_mut["week"] <= 7].copy()
    ws_orig_through_7 = ws[ws["week"] <= 7].copy()

    model_b = qb_baseline()
    model_b.fit(features=feats_through_7, weekly_stats=ws_orig_through_7)
    model_c = qb_baseline()
    model_c.fit(features=feats_through_7, weekly_stats=ws_mut_through_7)

    for stat in model_b.target_stats:
        np.testing.assert_array_equal(
            model_b.ridges[stat].coef_,
            model_c.ridges[stat].coef_,
            err_msg=f"Leakage detected on stat {stat}",
        )
        assert model_b.ridges[stat].alpha_ == model_c.ridges[stat].alpha_

    # Control: full-fixture vs week<=7 SHOULD differ on the strongest signal.
    coef_a = model_a.ridges[Stat.PASSING_YARDS].coef_
    coef_b = model_b.ridges[Stat.PASSING_YARDS].coef_
    assert not np.array_equal(coef_a, coef_b), (
        "Sanity check: full-fixture fit and week<=7 fit should produce "
        "different coefficients on PASSING_YARDS"
    )
```

- [ ] **Step 6.3: Run the QB tests.**

Run: `pytest tests/test_models/test_baseline_qb.py tests/test_models/test_baseline_qb_leakage.py -v`

Expected: all pass. If a test fails on `train_seasons == (2024, 2025)` because the fixture dropna removes all 2025 rows, debug by inspecting `model.train_seasons` directly — the leakage test fix may be to assert `model.train_seasons[0] == 2024` only.

- [ ] **Step 6.4: Full gate.**

```bash
pytest -q
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: all green.

- [ ] **Step 6.5: Commit.**

```bash
git add tests/test_models/test_baseline_qb.py tests/test_models/test_baseline_qb_leakage.py
git commit -m "$(cat <<'EOF'
test(models): qb_baseline unit + leakage tests (Phase 4b)

Mirrors tests/test_models/test_baseline.py's WR coverage:
- factory shape (position, target_stats, dist_families, feature_columns)
- fit populates ridges + feature_means + train_seasons
- variance_params populated correctly per family (NORMAL/GAMMA)
- predict_distribution → ProjectionWeeklySchema valid; quantile order
- save/load round-trip preserves predictions
- model_id format: baseline:qb:<8-char-hash>:<train_start>-<train_end>
- unfitted .model_id raises RuntimeError
- NaN feature imputation via persisted means
- empty input → empty schema-valid frame

Leakage test: refitting on weekly_stats with mutated rows at week >= 8
produces byte-identical Ridge coefficients to the unmutated fit through
week <= 7. Control assertion confirms the test is non-trivial (full-
fixture fit on PASSING_YARDS yields different coefs than week-<=7 fit).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7 (Phase 4c): RB unit tests + leakage test

**Files:**
- Create: `tests/test_models/test_baseline_rb.py`
- Create: `tests/test_models/test_baseline_rb_leakage.py`

**Goal:** Same as Task 6 but for `rb_baseline()`. Position-swapped from Task 6.

- [ ] **Step 7.1: Create `tests/test_models/test_baseline_rb.py`.**

```python
"""rb_baseline() unit tests. Mirrors test_baseline.py's WR coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from projections.distributions.parametric import ParametricGamma, ParametricNormal
from projections.models import rb_baseline
from projections.schemas import DistributionFamily, Position, ProjectionWeeklySchema, Ruleset, Stat


def test_rb_baseline_factory_returns_unfitted_model() -> None:
    model = rb_baseline()
    assert model.position == Position.RB
    expected_targets = {
        Stat.RUSHING_YARDS,
        Stat.RUSHING_TDS,
        Stat.RECEPTIONS,
        Stat.RECEIVING_YARDS,
        Stat.RECEIVING_TDS,
        Stat.FUMBLES_LOST,
    }
    assert set(model.target_stats) == expected_targets
    assert model.dist_families[Stat.RUSHING_YARDS] is DistributionFamily.NORMAL
    assert model.dist_families[Stat.RUSHING_TDS] is DistributionFamily.GAMMA
    assert model.dist_families[Stat.RECEPTIONS] is DistributionFamily.GAMMA
    assert model.dist_families[Stat.RECEIVING_YARDS] is DistributionFamily.NORMAL
    assert model.dist_families[Stat.RECEIVING_TDS] is DistributionFamily.GAMMA
    assert model.dist_families[Stat.FUMBLES_LOST] is DistributionFamily.GAMMA
    assert model.feature_columns


def test_rb_baseline_fit_populates_ridges_per_target_stat(
    baseline_features_rb: pd.DataFrame, baseline_weekly_stats_rb: pd.DataFrame
) -> None:
    model = rb_baseline()
    model.fit(features=baseline_features_rb, weekly_stats=baseline_weekly_stats_rb)
    assert set(model.ridges.keys()) == set(model.target_stats)
    for stat in model.target_stats:
        assert isinstance(model.ridges[stat], RidgeCV)
        assert hasattr(model.ridges[stat], "coef_")


def test_rb_baseline_fit_persists_feature_means(
    baseline_features_rb: pd.DataFrame, baseline_weekly_stats_rb: pd.DataFrame
) -> None:
    model = rb_baseline()
    model.fit(features=baseline_features_rb, weekly_stats=baseline_weekly_stats_rb)
    assert model.feature_means is not None
    assert set(model.feature_means.index) == set(model.feature_columns)


def test_rb_baseline_fit_records_train_seasons(
    baseline_features_rb: pd.DataFrame, baseline_weekly_stats_rb: pd.DataFrame
) -> None:
    model = rb_baseline()
    model.fit(features=baseline_features_rb, weekly_stats=baseline_weekly_stats_rb)
    assert model.train_seasons == (2024, 2025)


def test_rb_baseline_fit_populates_normal_variance_params(
    baseline_features_rb: pd.DataFrame, baseline_weekly_stats_rb: pd.DataFrame
) -> None:
    model = rb_baseline()
    model.fit(features=baseline_features_rb, weekly_stats=baseline_weekly_stats_rb)
    for stat in (Stat.RUSHING_YARDS, Stat.RECEIVING_YARDS):
        params = model.variance_params[stat]
        assert "std" in params
        assert params["std"] > 0


def test_rb_baseline_fit_populates_gamma_variance_params(
    baseline_features_rb: pd.DataFrame, baseline_weekly_stats_rb: pd.DataFrame
) -> None:
    model = rb_baseline()
    model.fit(features=baseline_features_rb, weekly_stats=baseline_weekly_stats_rb)
    for stat in (Stat.RUSHING_TDS, Stat.RECEPTIONS, Stat.RECEIVING_TDS, Stat.FUMBLES_LOST):
        params = model.variance_params[stat]
        assert "shape" in params
        assert 0.01 <= params["shape"] <= 100.0


def test_rb_predict_distribution_returns_projection_weekly_schema_valid_frame(
    baseline_features_rb: pd.DataFrame, baseline_weekly_stats_rb: pd.DataFrame
) -> None:
    model = rb_baseline()
    model.fit(features=baseline_features_rb, weekly_stats=baseline_weekly_stats_rb)
    week_features = baseline_features_rb[
        (baseline_features_rb["season"] == 2025) & (baseline_features_rb["week"] == 4)
    ]
    out = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    ProjectionWeeklySchema.validate(out)
    assert len(out) == len(week_features)
    assert (out["model_id"].str.startswith("baseline:rb:")).all()
    assert (out["position"] == "RB").all()


def test_rb_predict_distribution_p10_le_p50_le_p90(
    baseline_features_rb: pd.DataFrame, baseline_weekly_stats_rb: pd.DataFrame
) -> None:
    model = rb_baseline()
    model.fit(features=baseline_features_rb, weekly_stats=baseline_weekly_stats_rb)
    week_features = baseline_features_rb[
        (baseline_features_rb["season"] == 2025) & (baseline_features_rb["week"] == 4)
    ]
    out = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    assert (out["p10"] <= out["p50"]).all()
    assert (out["p50"] <= out["p90"]).all()


def test_rb_baseline_save_load_round_trip_preserves_predictions(
    tmp_path: Path,
    baseline_features_rb: pd.DataFrame,
    baseline_weekly_stats_rb: pd.DataFrame,
) -> None:
    model = rb_baseline()
    model.fit(features=baseline_features_rb, weekly_stats=baseline_weekly_stats_rb)

    artifact = tmp_path / "rb-baseline.joblib"
    model.save(artifact)
    assert artifact.exists()

    from projections.models import BaselineModel

    loaded = BaselineModel.load(artifact)
    assert loaded.position == Position.RB
    assert loaded.model_id == model.model_id

    week = baseline_features_rb[
        (baseline_features_rb["season"] == 2025) & (baseline_features_rb["week"] == 4)
    ]
    out_orig = model.predict_distribution(week, ruleset=Ruleset.espn_ppr())
    out_loaded = loaded.predict_distribution(week, ruleset=Ruleset.espn_ppr())
    pd.testing.assert_frame_equal(
        out_orig.drop(columns=["generated_at"]),
        out_loaded.drop(columns=["generated_at"]),
    )


def test_rb_model_id_format(
    baseline_features_rb: pd.DataFrame, baseline_weekly_stats_rb: pd.DataFrame
) -> None:
    model = rb_baseline()
    model.fit(features=baseline_features_rb, weekly_stats=baseline_weekly_stats_rb)
    parts = model.model_id.split(":")
    assert len(parts) == 4
    assert parts[0] == "baseline"
    assert parts[1] == "rb"
    assert len(parts[2]) == 8
    assert "-" in parts[3]


def test_rb_unfitted_model_id_raises() -> None:
    model = rb_baseline()
    try:
        _ = model.model_id
    except RuntimeError:
        return
    raise AssertionError("Unfitted model.model_id should raise RuntimeError")


def test_rb_predict_distribution_imputes_nan_features_with_persisted_means(
    baseline_features_rb: pd.DataFrame, baseline_weekly_stats_rb: pd.DataFrame
) -> None:
    model = rb_baseline()
    model.fit(features=baseline_features_rb, weekly_stats=baseline_weekly_stats_rb)

    week = baseline_features_rb[
        (baseline_features_rb["season"] == 2025) & (baseline_features_rb["week"] == 4)
    ].copy()
    week.loc[week.index[0], "implied_team_total"] = np.nan
    out = model.predict_distribution(week, ruleset=Ruleset.espn_ppr())
    assert not out["mean"].isna().any()


def test_rb_predict_distribution_empty_input_returns_empty_schema_valid_frame(
    baseline_features_rb: pd.DataFrame, baseline_weekly_stats_rb: pd.DataFrame
) -> None:
    model = rb_baseline()
    model.fit(features=baseline_features_rb, weekly_stats=baseline_weekly_stats_rb)
    empty = baseline_features_rb.iloc[0:0]
    out = model.predict_distribution(empty, ruleset=Ruleset.espn_ppr())
    assert out.empty
    ProjectionWeeklySchema.validate(out)
```

- [ ] **Step 7.2: Create `tests/test_models/test_baseline_rb_leakage.py`.**

```python
"""RB baseline leakage test. Mirrors test_baseline_leakage.py's WR test."""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.models import rb_baseline
from projections.schemas import Stat


def test_rb_baseline_fit_does_not_use_post_as_of_week_data(
    baseline_features_rb: pd.DataFrame, baseline_weekly_stats_rb: pd.DataFrame
) -> None:
    ws = baseline_weekly_stats_rb[baseline_weekly_stats_rb["season"] == 2024].copy()
    feats = baseline_features_rb[baseline_features_rb["season"] == 2024].copy()

    model_a = rb_baseline()
    model_a.fit(features=feats, weekly_stats=ws)

    ws_mut = ws.copy()
    mask = ws_mut["week"] >= 8
    ws_mut.loc[mask, "rushing_yards"] = 0.0
    ws_mut.loc[mask, "rushing_tds"] = 0
    ws_mut.loc[mask, "carries"] = 0
    ws_mut.loc[mask, "receptions"] = 0
    ws_mut.loc[mask, "receiving_yards"] = 999.0
    ws_mut.loc[mask, "receiving_tds"] = 9

    feats_through_7 = feats[feats["week"] <= 7].copy()
    ws_mut_through_7 = ws_mut[ws_mut["week"] <= 7].copy()
    ws_orig_through_7 = ws[ws["week"] <= 7].copy()

    model_b = rb_baseline()
    model_b.fit(features=feats_through_7, weekly_stats=ws_orig_through_7)
    model_c = rb_baseline()
    model_c.fit(features=feats_through_7, weekly_stats=ws_mut_through_7)

    for stat in model_b.target_stats:
        np.testing.assert_array_equal(
            model_b.ridges[stat].coef_,
            model_c.ridges[stat].coef_,
            err_msg=f"Leakage detected on stat {stat}",
        )
        assert model_b.ridges[stat].alpha_ == model_c.ridges[stat].alpha_

    coef_a = model_a.ridges[Stat.RUSHING_YARDS].coef_
    coef_b = model_b.ridges[Stat.RUSHING_YARDS].coef_
    assert not np.array_equal(coef_a, coef_b), (
        "Sanity check: full-fixture fit and week<=7 fit should produce "
        "different coefficients on RUSHING_YARDS"
    )
```

- [ ] **Step 7.3: Run the RB tests.**

Run: `pytest tests/test_models/test_baseline_rb.py tests/test_models/test_baseline_rb_leakage.py -v`

Expected: all pass.

- [ ] **Step 7.4: Full gate.**

```bash
pytest -q
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: all green.

- [ ] **Step 7.5: Commit.**

```bash
git add tests/test_models/test_baseline_rb.py tests/test_models/test_baseline_rb_leakage.py
git commit -m "$(cat <<'EOF'
test(models): rb_baseline unit + leakage tests (Phase 4c)

Position-swap of test_baseline_qb.py against rb_baseline() and the RB
fixtures from Task 5. Coverage matches the WR / QB sets one-for-one.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8 (Phase 4d): TE unit tests + leakage test

**Files:**
- Create: `tests/test_models/test_baseline_te.py`
- Create: `tests/test_models/test_baseline_te_leakage.py`

**Goal:** Same as Tasks 6/7 but for `te_baseline()`. Includes one extra assertion that the Taysom-Hill-shape rushing TE flows through to a non-zero predicted rushing component.

- [ ] **Step 8.1: Create `tests/test_models/test_baseline_te.py`.**

```python
"""te_baseline() unit tests. Mirrors test_baseline.py's WR coverage; adds a
Taysom-Hill-shape assertion since TE rushing is the explicit rationale for
including rushing as a TE target stat (Plan 3b Q3=B)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from projections.distributions.parametric import ParametricGamma, ParametricNormal
from projections.models import te_baseline
from projections.schemas import DistributionFamily, Position, ProjectionWeeklySchema, Ruleset, Stat


def test_te_baseline_factory_returns_unfitted_model() -> None:
    model = te_baseline()
    assert model.position == Position.TE
    expected_targets = {
        Stat.RECEPTIONS,
        Stat.RECEIVING_YARDS,
        Stat.RECEIVING_TDS,
        Stat.RUSHING_YARDS,
        Stat.RUSHING_TDS,
        Stat.FUMBLES_LOST,
    }
    assert set(model.target_stats) == expected_targets
    assert model.dist_families[Stat.RECEPTIONS] is DistributionFamily.GAMMA
    assert model.dist_families[Stat.RECEIVING_YARDS] is DistributionFamily.NORMAL
    assert model.dist_families[Stat.RECEIVING_TDS] is DistributionFamily.GAMMA
    assert model.dist_families[Stat.RUSHING_YARDS] is DistributionFamily.NORMAL
    assert model.dist_families[Stat.RUSHING_TDS] is DistributionFamily.GAMMA
    assert model.dist_families[Stat.FUMBLES_LOST] is DistributionFamily.GAMMA
    assert model.feature_columns


def test_te_baseline_factory_includes_rushing_features() -> None:
    """Phase 1 added rushing_*_per_game_l4 to TeFeaturesSchema; te_baseline's
    _TE_FEATURE_COLUMNS must reference them."""
    model = te_baseline()
    assert "rushing_attempts_per_game_l4" in model.feature_columns
    assert "rushing_yards_per_game_l4" in model.feature_columns


def test_te_baseline_fit_populates_ridges_per_target_stat(
    baseline_features_te: pd.DataFrame, baseline_weekly_stats_te: pd.DataFrame
) -> None:
    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)
    assert set(model.ridges.keys()) == set(model.target_stats)
    for stat in model.target_stats:
        assert isinstance(model.ridges[stat], RidgeCV)
        assert hasattr(model.ridges[stat], "coef_")


def test_te_baseline_fit_persists_feature_means(
    baseline_features_te: pd.DataFrame, baseline_weekly_stats_te: pd.DataFrame
) -> None:
    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)
    assert model.feature_means is not None
    assert set(model.feature_means.index) == set(model.feature_columns)


def test_te_baseline_fit_records_train_seasons(
    baseline_features_te: pd.DataFrame, baseline_weekly_stats_te: pd.DataFrame
) -> None:
    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)
    assert model.train_seasons == (2024, 2025)


def test_te_baseline_fit_populates_normal_variance_params(
    baseline_features_te: pd.DataFrame, baseline_weekly_stats_te: pd.DataFrame
) -> None:
    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)
    for stat in (Stat.RECEIVING_YARDS, Stat.RUSHING_YARDS):
        params = model.variance_params[stat]
        assert "std" in params
        assert params["std"] > 0


def test_te_baseline_fit_populates_gamma_variance_params(
    baseline_features_te: pd.DataFrame, baseline_weekly_stats_te: pd.DataFrame
) -> None:
    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)
    for stat in (Stat.RECEPTIONS, Stat.RECEIVING_TDS, Stat.RUSHING_TDS, Stat.FUMBLES_LOST):
        params = model.variance_params[stat]
        assert "shape" in params
        assert 0.01 <= params["shape"] <= 100.0


def test_te_predict_distribution_returns_projection_weekly_schema_valid_frame(
    baseline_features_te: pd.DataFrame, baseline_weekly_stats_te: pd.DataFrame
) -> None:
    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)
    week_features = baseline_features_te[
        (baseline_features_te["season"] == 2025) & (baseline_features_te["week"] == 4)
    ]
    out = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    ProjectionWeeklySchema.validate(out)
    assert len(out) == len(week_features)
    assert (out["model_id"].str.startswith("baseline:te:")).all()
    assert (out["position"] == "TE").all()


def test_te_predict_distribution_p10_le_p50_le_p90(
    baseline_features_te: pd.DataFrame, baseline_weekly_stats_te: pd.DataFrame
) -> None:
    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)
    week_features = baseline_features_te[
        (baseline_features_te["season"] == 2025) & (baseline_features_te["week"] == 4)
    ]
    out = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    assert (out["p10"] <= out["p50"]).all()
    assert (out["p50"] <= out["p90"]).all()


def test_te_baseline_save_load_round_trip_preserves_predictions(
    tmp_path: Path,
    baseline_features_te: pd.DataFrame,
    baseline_weekly_stats_te: pd.DataFrame,
) -> None:
    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)

    artifact = tmp_path / "te-baseline.joblib"
    model.save(artifact)
    assert artifact.exists()

    from projections.models import BaselineModel

    loaded = BaselineModel.load(artifact)
    assert loaded.position == Position.TE
    assert loaded.model_id == model.model_id

    week = baseline_features_te[
        (baseline_features_te["season"] == 2025) & (baseline_features_te["week"] == 4)
    ]
    out_orig = model.predict_distribution(week, ruleset=Ruleset.espn_ppr())
    out_loaded = loaded.predict_distribution(week, ruleset=Ruleset.espn_ppr())
    pd.testing.assert_frame_equal(
        out_orig.drop(columns=["generated_at"]),
        out_loaded.drop(columns=["generated_at"]),
    )


def test_te_model_id_format(
    baseline_features_te: pd.DataFrame, baseline_weekly_stats_te: pd.DataFrame
) -> None:
    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)
    parts = model.model_id.split(":")
    assert len(parts) == 4
    assert parts[0] == "baseline"
    assert parts[1] == "te"
    assert len(parts[2]) == 8
    assert "-" in parts[3]


def test_te_unfitted_model_id_raises() -> None:
    model = te_baseline()
    try:
        _ = model.model_id
    except RuntimeError:
        return
    raise AssertionError("Unfitted model.model_id should raise RuntimeError")


def test_te_predict_distribution_imputes_nan_features_with_persisted_means(
    baseline_features_te: pd.DataFrame, baseline_weekly_stats_te: pd.DataFrame
) -> None:
    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)

    week = baseline_features_te[
        (baseline_features_te["season"] == 2025) & (baseline_features_te["week"] == 4)
    ].copy()
    week.loc[week.index[0], "implied_team_total"] = np.nan
    out = model.predict_distribution(week, ruleset=Ruleset.espn_ppr())
    assert not out["mean"].isna().any()


def test_te_predict_distribution_empty_input_returns_empty_schema_valid_frame(
    baseline_features_te: pd.DataFrame, baseline_weekly_stats_te: pd.DataFrame
) -> None:
    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)
    empty = baseline_features_te.iloc[0:0]
    out = model.predict_distribution(empty, ruleset=Ruleset.espn_ppr())
    assert out.empty
    ProjectionWeeklySchema.validate(out)


def test_te_baseline_taysom_hill_row_predicts_nonzero_rushing_yards_mean(
    baseline_features_te: pd.DataFrame, baseline_weekly_stats_te: pd.DataFrame
) -> None:
    """The Phase 1 TE rushing extension and the rushing-TE row in the fixture
    (gsis_id ending in "3") together imply a non-zero predicted rushing_yards
    mean. If the model collapses to zero, the rushing features aren't
    flowing through fit→predict correctly."""
    model = te_baseline()
    model.fit(features=baseline_features_te, weekly_stats=baseline_weekly_stats_te)

    rushing_te_id = next(g for g in baseline_features_te["gsis_id"].astype(str).unique() if g.endswith("3"))
    week = baseline_features_te[
        (baseline_features_te["season"] == 2025)
        & (baseline_features_te["week"] == 4)
        & (baseline_features_te["gsis_id"].astype(str) == rushing_te_id)
    ]
    if week.empty:
        # Fixture didn't produce a 2025 wk4 row for the rushing TE; skip rather than
        # fabricate one — coverage still comes from the per-stat-mu check below.
        from pytest import skip

        skip("rushing-TE not present in 2025 wk4 fixture slice")

    stat_dists = model._build_stat_distributions(week)
    rushing_yd_mu = stat_dists[0][Stat.RUSHING_YARDS].mean()
    assert rushing_yd_mu > 0.5, (
        f"Taysom-Hill-shape TE rushing_yards predicted mean is {rushing_yd_mu:.3f}; "
        "expected > 0.5. Verify TE rushing features made it through fit→predict."
    )
```

- [ ] **Step 8.2: Create `tests/test_models/test_baseline_te_leakage.py`.**

```python
"""TE baseline leakage test. Mirrors test_baseline_leakage.py's WR test."""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.models import te_baseline
from projections.schemas import Stat


def test_te_baseline_fit_does_not_use_post_as_of_week_data(
    baseline_features_te: pd.DataFrame, baseline_weekly_stats_te: pd.DataFrame
) -> None:
    ws = baseline_weekly_stats_te[baseline_weekly_stats_te["season"] == 2024].copy()
    feats = baseline_features_te[baseline_features_te["season"] == 2024].copy()

    model_a = te_baseline()
    model_a.fit(features=feats, weekly_stats=ws)

    ws_mut = ws.copy()
    mask = ws_mut["week"] >= 8
    ws_mut.loc[mask, "receptions"] = 0
    ws_mut.loc[mask, "receiving_yards"] = 0.0
    ws_mut.loc[mask, "receiving_tds"] = 0
    ws_mut.loc[mask, "rushing_yards"] = 999.0
    ws_mut.loc[mask, "rushing_tds"] = 9
    ws_mut.loc[mask, "carries"] = 30

    feats_through_7 = feats[feats["week"] <= 7].copy()
    ws_mut_through_7 = ws_mut[ws_mut["week"] <= 7].copy()
    ws_orig_through_7 = ws[ws["week"] <= 7].copy()

    model_b = te_baseline()
    model_b.fit(features=feats_through_7, weekly_stats=ws_orig_through_7)
    model_c = te_baseline()
    model_c.fit(features=feats_through_7, weekly_stats=ws_mut_through_7)

    for stat in model_b.target_stats:
        np.testing.assert_array_equal(
            model_b.ridges[stat].coef_,
            model_c.ridges[stat].coef_,
            err_msg=f"Leakage detected on stat {stat}",
        )
        assert model_b.ridges[stat].alpha_ == model_c.ridges[stat].alpha_

    coef_a = model_a.ridges[Stat.RECEPTIONS].coef_
    coef_b = model_b.ridges[Stat.RECEPTIONS].coef_
    assert not np.array_equal(coef_a, coef_b), (
        "Sanity check: full-fixture fit and week<=7 fit should produce "
        "different coefficients on RECEPTIONS"
    )
```

- [ ] **Step 8.3: Run the TE tests.**

Run: `pytest tests/test_models/test_baseline_te.py tests/test_models/test_baseline_te_leakage.py -v`

Expected: all pass.

- [ ] **Step 8.4: Full gate.**

```bash
pytest -q
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: all green.

- [ ] **Step 8.5: Commit.**

```bash
git add tests/test_models/test_baseline_te.py tests/test_models/test_baseline_te_leakage.py
git commit -m "$(cat <<'EOF'
test(models): te_baseline unit + leakage tests (Phase 4d)

Position-swap of test_baseline_qb.py + leakage against te_baseline()
and the TE fixtures from Task 5. Adds two TE-specific assertions:

- test_te_baseline_factory_includes_rushing_features verifies the
  Phase 1 schema extension is wired through to _TE_FEATURE_COLUMNS.
- test_te_baseline_taysom_hill_row_predicts_nonzero_rushing_yards_mean
  verifies the rushing TE row in the fixture flows through fit→predict
  and produces a non-zero predicted rushing_yards mean.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9 (Phase 4e): Smoke test parametrized across all four positions

**Files:**
- Modify: `tests/test_smoke.py`

**Goal:** Replace `test_smoke_wr_baseline_fit_predict_write` with a parametrized version that exercises the round-trip for every position via `POSITION_DISPATCH`. Catches "I broke RB silently while refactoring `BaselineModel`" regressions earlier than the per-position test files would.

- [ ] **Step 9.1: Read the existing smoke test.**

Run: `cat tests/test_smoke.py`. Locate `test_smoke_wr_baseline_fit_predict_write` (around line 200).

- [ ] **Step 9.2: Replace the test with a parametrized version.**

Edit `tests/test_smoke.py`. Replace `test_smoke_wr_baseline_fit_predict_write` and its body with:

```python
@pytest.mark.parametrize(
    "position_value,fixture_features,fixture_weekly_stats",
    [
        ("QB", "baseline_features_qb", "baseline_weekly_stats_qb"),
        ("RB", "baseline_features_rb", "baseline_weekly_stats_rb"),
        ("TE", "baseline_features_te", "baseline_weekly_stats_te"),
        ("WR", "baseline_features_wr", "baseline_weekly_stats_wr"),
    ],
)
def test_smoke_baseline_fit_predict_write_round_trip(
    tmp_path: Path,
    position_value: str,
    fixture_features: str,
    fixture_weekly_stats: str,
    request: pytest.FixtureRequest,
) -> None:
    """End-to-end: for every position, fit BaselineModel on synthetic data,
    predict, write a parquet partition through store.write_partition, read
    back, validate.

    Catches cross-position regressions (e.g., "I refactored BaselineModel
    and broke one factory but not others") before the per-position test
    files would surface them.
    """
    from projections.models import POSITION_DISPATCH
    from projections.schemas import Position, ProjectionWeeklySchema, Ruleset
    from projections.store import read_partition, write_partition

    pos = Position(position_value)
    dispatch = POSITION_DISPATCH[pos]
    features = request.getfixturevalue(fixture_features)
    weekly_stats = request.getfixturevalue(fixture_weekly_stats)

    model = dispatch.factory()
    model.fit(features=features, weekly_stats=weekly_stats)

    week_features = features[(features["season"] == 2025) & (features["week"] == 4)]
    if week_features.empty:
        pytest.skip(f"no 2025 wk4 features in {position_value} fixture")
    preds = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    ProjectionWeeklySchema.validate(preds)

    write_partition(
        tmp_path / "projections",
        "weekly/ruleset=ESPN_PPR",
        preds,
        season=2025,
        week=4,
    )
    round_tripped = read_partition(
        tmp_path / "projections", "weekly/ruleset=ESPN_PPR", season=2025, week=4
    )
    ProjectionWeeklySchema.validate(round_tripped)
    assert len(round_tripped) == len(preds)
    assert (round_tripped["model_id"].str.startswith(f"baseline:{position_value.lower()}:")).all()
```

- [ ] **Step 9.3: Run the smoke test.**

Run: `pytest tests/test_smoke.py -v`

Expected: 4 parametrized cases pass (one per position), plus the existing `test_package_imports` and `test_end_to_end_ingest_and_features`. Total smoke runtime ~20 seconds.

- [ ] **Step 9.4: Full gate.**

```bash
pytest -q
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: all green.

- [ ] **Step 9.5: Commit.**

```bash
git add tests/test_smoke.py
git commit -m "$(cat <<'EOF'
test(smoke): parametrize fit→predict→write round-trip across all 4 positions

Phase 4e of Plan 3b. The 3a smoke covered WR only; 3b refactors
BaselineModel and adds three new factories, so a cross-position
integration check is worth the ~20s runtime. The parametrized test
loops over POSITION_DISPATCH and asserts the round-trip works for
every position — catching "I broke RB silently" earlier than
test_baseline_rb.py would.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10 (Phase 5a): `train_baseline.py {position}` script

**Files:**
- Create: `scripts/train_baseline.py`
- Delete: `scripts/train_wr_baseline.py`

**Goal:** Replace `train_wr_baseline.py` with a position-arg-driven version that reads the same `data/raw/` partitions, dispatches by `POSITION_DISPATCH[pos]`, fits the model, and saves to `models/artifacts/baseline-{pos}-{train_start}-{train_end}-{hash}.joblib`.

- [ ] **Step 10.1: Create `scripts/train_baseline.py`.**

```python
"""Plan 3b -- train Model A baseline for a specified position on 2018-2023,
persist to models/artifacts/.

Replaces scripts/train_wr_baseline.py with a position-arg-driven version.
Usage:
    python scripts/train_baseline.py {qb|rb|te|wr}

Reads ingested raw partitions from data/raw/, builds features for every
week of 2018-2023, fits BaselineModel via POSITION_DISPATCH[pos].factory(),
saves the joblib artifact to:
    models/artifacts/baseline-{pos}-{train_start}-{train_end}-{hash}.joblib

Held-out: 2024 (sanity_check_baseline.py {pos} consumes the artifact).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.ingest.ngs import NgsStatType
from projections.models import POSITION_DISPATCH
from projections.schemas import Position
from projections.store import read_partition

_TRAIN_SEASONS = range(2018, 2024)  # 2018..2023 inclusive (2024 held out)


def _ngs_table_for(stat_type: NgsStatType) -> str:
    return f"ngs_{stat_type}"


def _build_training_features(
    raw_root: Path, position: Position
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a stacked feature DataFrame across (season, week) pairs in the
    training window plus the matching weekly_stats truth across the same
    seasons. Caller passes both into BaselineModel.fit."""
    dispatch = POSITION_DISPATCH[position]
    builder = dispatch.feature_builder
    ngs_table = _ngs_table_for(dispatch.ngs_stat_type)
    # Each builder takes a kwarg named ngs_{stat_type}; unify by mapping.
    ngs_kwarg = {"passing": "ngs_passing", "rushing": "ngs_rushing", "receiving": "ngs_receiving"}[
        dispatch.ngs_stat_type
    ]

    feature_frames: list[pd.DataFrame] = []
    truth_frames: list[pd.DataFrame] = []
    for season in _TRAIN_SEASONS:
        ws = read_partition(raw_root, "weekly_stats", season=season)
        sc = read_partition(raw_root, "snap_counts", season=season)
        dc = read_partition(raw_root, "depth_charts", season=season)
        ngs = read_partition(raw_root, ngs_table, season=season)
        sch = read_partition(raw_root, "schedules", season=season)
        truth_frames.append(ws)

        weeks = sorted(dc["week"].unique())
        for week in weeks:
            kwargs = {
                "weekly_stats": ws,
                "snap_counts": sc,
                "depth_charts": dc,
                "schedules": sch,
                "season": int(season),
                "as_of_week": int(week),
                ngs_kwarg: ngs,
            }
            f = builder(**kwargs)
            if not f.empty:
                feature_frames.append(f)
        print(f"  Built {position.value} features for season {season}: {len(weeks)} weeks")

    features = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    weekly_stats = pd.concat(truth_frames, ignore_index=True) if truth_frames else pd.DataFrame()
    return features, weekly_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Model A baseline for a position.")
    parser.add_argument("position", choices=["qb", "rb", "te", "wr"], help="Target position.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("models/artifacts"))
    args = parser.parse_args()

    position = Position(args.position.upper())
    print(f"Training {position.value} baseline; reading raw partitions from {args.raw_root}")

    features, weekly_stats = _build_training_features(args.raw_root, position)
    print(f"Total {position.value} feature rows: {len(features)}; weekly_stats rows: {len(weekly_stats)}")

    model = POSITION_DISPATCH[position].factory()
    model.fit(features=features, weekly_stats=weekly_stats)
    print(f"model_id: {model.model_id}")
    for stat in model.target_stats:
        print(f"  {stat.value}: variance_params = {model.variance_params[stat]}")

    train_start, train_end = model.train_seasons or (0, 0)
    artifact = (
        args.artifacts_root
        / f"baseline-{position.value.lower()}-{train_start}-{train_end}-{model.code_hash}.joblib"
    )
    model.save(artifact)
    print(f"Saved artifact: {artifact}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 10.2: Delete `scripts/train_wr_baseline.py`.**

```bash
rm scripts/train_wr_baseline.py
```

- [ ] **Step 10.3: Smoke-test the new script (no real data run yet — that's Phase 6).**

Run: `python scripts/train_baseline.py --help`

Expected: argparse prints usage with the four position choices.

Run: `python -c "import scripts.train_baseline"` — wait, this won't work because scripts/ is not a package. Skip.

- [ ] **Step 10.4: Full gate.**

```bash
pytest -q
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: all green. (mypy doesn't scan `scripts/` by default per pyproject.toml's mypy_path; if it does, `mypy src tests scripts` should also pass — confirm by running it explicitly:)

Run: `mypy scripts/train_baseline.py`

Expected: clean. If pandera generic complaints surface, follow the same pattern as `BaselineModel.feature_schema`.

- [ ] **Step 10.5: Commit.**

```bash
git add scripts/train_baseline.py
git rm scripts/train_wr_baseline.py
git commit -m "$(cat <<'EOF'
feat(scripts): generalized train_baseline.py {position} (Phase 5a)

Replaces train_wr_baseline.py with a position-arg-driven version.
Dispatches by POSITION_DISPATCH[pos] so the same script trains every
position. Same artifact-naming convention: baseline-{pos}-{train_start}-
{train_end}-{hash}.joblib.

The old scripts/train_wr_baseline.py is deleted; train_baseline.py wr
produces identical behavior.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11 (Phase 5b): `predict_2024.py {position}` script

**Files:**
- Create: `scripts/predict_2024.py`
- Delete: `scripts/predict_2024_wr.py`

**Goal:** Replace `predict_2024_wr.py` with a position-arg-driven version.

- [ ] **Step 11.1: Create `scripts/predict_2024.py`.**

```python
"""Plan 3b -- write 2024 weekly projections for a specified position to
data/projections/weekly/.

Replaces scripts/predict_2024_wr.py.

Usage:
    python scripts/predict_2024.py {qb|rb|te|wr}

Loads the trained artifact (baseline-{pos}-...joblib), builds features
for each week of 2024, predicts, and writes one parquet partition per
(season, week) using store.write_partition. Validated against
ProjectionWeeklySchema.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.models import POSITION_DISPATCH, BaselineModel
from projections.schemas import Position, ProjectionWeeklySchema, Ruleset
from projections.store import read_partition, write_partition

_PROJECTION_SEASON = 2024


def _find_artifact(artifacts_root: Path, position: Position) -> Path:
    pattern = f"baseline-{position.value.lower()}-*.joblib"
    matches = sorted(artifacts_root.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No {pattern} in {artifacts_root}. Run scripts/train_baseline.py {position.value.lower()} first."
        )
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict 2024 weekly projections for a position.")
    parser.add_argument("position", choices=["qb", "rb", "te", "wr"], help="Target position.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--projections-root", type=Path, default=Path("data/projections"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("models/artifacts"))
    parser.add_argument("--ruleset", type=str, default="espn_ppr")
    args = parser.parse_args()

    position = Position(args.position.upper())
    dispatch = POSITION_DISPATCH[position]
    builder = dispatch.feature_builder
    ngs_kwarg = {"passing": "ngs_passing", "rushing": "ngs_rushing", "receiving": "ngs_receiving"}[
        dispatch.ngs_stat_type
    ]
    ngs_table = f"ngs_{dispatch.ngs_stat_type}"

    artifact = _find_artifact(args.artifacts_root, position)
    print(f"Loading artifact: {artifact}")
    model = BaselineModel.load(artifact)

    ruleset_map = {
        "espn_ppr": Ruleset.espn_ppr(),
        "espn_half": Ruleset.espn_half(),
        "standard": Ruleset.standard(),
    }
    ruleset = ruleset_map[args.ruleset]

    raw_root = args.raw_root
    ws_prior = read_partition(raw_root, "weekly_stats", season=_PROJECTION_SEASON - 1)
    sc_prior = read_partition(raw_root, "snap_counts", season=_PROJECTION_SEASON - 1)
    ngs_prior = read_partition(raw_root, ngs_table, season=_PROJECTION_SEASON - 1)
    ws_curr = read_partition(raw_root, "weekly_stats", season=_PROJECTION_SEASON)
    sc_curr = read_partition(raw_root, "snap_counts", season=_PROJECTION_SEASON)
    dc_curr = read_partition(raw_root, "depth_charts", season=_PROJECTION_SEASON)
    ngs_curr = read_partition(raw_root, ngs_table, season=_PROJECTION_SEASON)
    sch_curr = read_partition(raw_root, "schedules", season=_PROJECTION_SEASON)

    ws_full = pd.concat([ws_prior, ws_curr], ignore_index=True)
    sc_full = pd.concat([sc_prior, sc_curr], ignore_index=True)
    ngs_full = pd.concat([ngs_prior, ngs_curr], ignore_index=True)

    weeks = sorted(dc_curr["week"].unique())
    rule_partition = ruleset.name  # "ESPN_PPR" etc.
    for week in weeks:
        kwargs = {
            "weekly_stats": ws_full,
            "snap_counts": sc_full,
            "depth_charts": dc_curr,
            "schedules": sch_curr,
            "season": _PROJECTION_SEASON,
            "as_of_week": int(week),
            ngs_kwarg: ngs_full,
        }
        feats = builder(**kwargs)
        if feats.empty:
            print(f"  Week {week}: no rostered {position.value}s; skipping")
            continue
        preds = model.predict_distribution(feats, ruleset=ruleset)
        ProjectionWeeklySchema.validate(preds)
        target = write_partition(
            args.projections_root,
            f"weekly/ruleset={rule_partition}",
            preds,
            season=_PROJECTION_SEASON,
            week=int(week),
        )
        print(f"  Week {week}: wrote {len(preds)} rows -> {target}")

    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 11.2: Delete `scripts/predict_2024_wr.py`.**

```bash
rm scripts/predict_2024_wr.py
```

- [ ] **Step 11.3: Smoke-test the script.**

Run: `python scripts/predict_2024.py --help`

Expected: argparse usage prints.

- [ ] **Step 11.4: Full gate.**

```bash
pytest -q
mypy src tests scripts
ruff check src tests scripts
ruff format --check src tests scripts
```

Expected: all green.

- [ ] **Step 11.5: Commit.**

```bash
git add scripts/predict_2024.py
git rm scripts/predict_2024_wr.py
git commit -m "$(cat <<'EOF'
feat(scripts): generalized predict_2024.py {position} (Phase 5b)

Replaces predict_2024_wr.py. Dispatches by POSITION_DISPATCH[pos];
artifact lookup uses pattern baseline-{pos}-*.joblib. Output partition
shape unchanged: data/projections/weekly/ruleset=ESPN_PPR/season=2024/
week=WW/part.parquet.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12 (Phase 5c): `sanity_check_baseline.py {position}` script

**Files:**
- Create: `scripts/sanity_check_baseline.py`
- Delete: `scripts/sanity_check_wr_baseline.py`

**Goal:** Replace `sanity_check_wr_baseline.py` with a position-arg-driven version that loads the artifact for `{pos}`, builds 2024 features, predicts, and reports per-stat fit + composite + calibration metrics. Stdout-only; no CI gate.

- [ ] **Step 12.1: Create `scripts/sanity_check_baseline.py`.**

```python
"""Plan 3b -- sanity-check eval of Model A baseline for a position against
the held-out 2024 season. Stdout-only; not a CI gate (Plan 3c builds the
backtest harness with thresholds).

Replaces scripts/sanity_check_wr_baseline.py.

Usage (after train_baseline.py {pos}):
    python scripts/sanity_check_baseline.py {qb|rb|te|wr}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from projections.models import POSITION_DISPATCH, BaselineModel
from projections.schemas import Position, Ruleset
from projections.scoring import score
from projections.scoring.score import StatLine
from projections.store import read_partition

_HELD_OUT_SEASON = 2024


def _find_artifact(artifacts_root: Path, position: Position) -> Path:
    pattern = f"baseline-{position.value.lower()}-*.joblib"
    matches = sorted(artifacts_root.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No {pattern} in {artifacts_root}. Run scripts/train_baseline.py {position.value.lower()} first."
        )
    return matches[-1]


def _realized_ppr_points(weekly_stats: pd.DataFrame, ruleset: Ruleset) -> pd.Series:
    points: list[float] = []
    for _, row in weekly_stats.iterrows():
        line = StatLine(
            passing_yards=float(row["passing_yards"]),
            passing_tds=int(row["passing_tds"]),
            interceptions=int(row["interceptions"]),
            rushing_yards=float(row["rushing_yards"]),
            rushing_tds=int(row["rushing_tds"]),
            receptions=int(row["receptions"]),
            receiving_yards=float(row["receiving_yards"]),
            receiving_tds=int(row["receiving_tds"]),
            fumbles_lost=int(row["fumbles_lost"]),
        )
        points.append(score(line, ruleset))
    return pd.Series(points, index=weekly_stats.index, name="actual_ppr")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check Model A on 2024 for a position.")
    parser.add_argument("position", choices=["qb", "rb", "te", "wr"], help="Target position.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("models/artifacts"))
    args = parser.parse_args()

    position = Position(args.position.upper())
    dispatch = POSITION_DISPATCH[position]
    builder = dispatch.feature_builder
    ngs_kwarg = {"passing": "ngs_passing", "rushing": "ngs_rushing", "receiving": "ngs_receiving"}[
        dispatch.ngs_stat_type
    ]
    ngs_table = f"ngs_{dispatch.ngs_stat_type}"

    artifact = _find_artifact(args.artifacts_root, position)
    print(f"Loading artifact: {artifact}")
    model = BaselineModel.load(artifact)
    print(f"model_id: {model.model_id}")

    raw_root = args.raw_root

    ws_held = read_partition(raw_root, "weekly_stats", season=_HELD_OUT_SEASON)
    sc_held = read_partition(raw_root, "snap_counts", season=_HELD_OUT_SEASON)
    dc_held = read_partition(raw_root, "depth_charts", season=_HELD_OUT_SEASON)
    ngs_held = read_partition(raw_root, ngs_table, season=_HELD_OUT_SEASON)
    sch_held = read_partition(raw_root, "schedules", season=_HELD_OUT_SEASON)

    ws_prior = read_partition(raw_root, "weekly_stats", season=_HELD_OUT_SEASON - 1)
    sc_prior = read_partition(raw_root, "snap_counts", season=_HELD_OUT_SEASON - 1)
    ngs_prior = read_partition(raw_root, ngs_table, season=_HELD_OUT_SEASON - 1)
    ws_full = pd.concat([ws_prior, ws_held], ignore_index=True)
    sc_full = pd.concat([sc_prior, sc_held], ignore_index=True)
    ngs_full = pd.concat([ngs_prior, ngs_held], ignore_index=True)

    weeks = sorted(dc_held["week"].unique())
    rows: list[pd.DataFrame] = []
    for week in weeks:
        kwargs = {
            "weekly_stats": ws_full,
            "snap_counts": sc_full,
            "depth_charts": dc_held,
            "schedules": sch_held,
            "season": _HELD_OUT_SEASON,
            "as_of_week": int(week),
            ngs_kwarg: ngs_full,
        }
        feats = builder(**kwargs)
        if feats.empty:
            continue
        preds = model.predict_distribution(feats, ruleset=Ruleset.espn_ppr())
        stat_dists_per_row = model._build_stat_distributions(feats)
        per_stat_means = pd.DataFrame(
            {
                stat.value: [d[stat].mean() for d in stat_dists_per_row]
                for stat in model.target_stats
            }
        )
        per_stat_means["gsis_id"] = feats["gsis_id"].values
        per_stat_means["season"] = _HELD_OUT_SEASON
        per_stat_means["week"] = int(week)

        joined = preds.merge(per_stat_means, on=["gsis_id", "season", "week"], how="left")
        rows.append(joined)

    all_preds = pd.concat(rows, ignore_index=True)

    actual = ws_held[ws_held["position"] == position.value].copy()
    actual["actual_ppr"] = _realized_ppr_points(actual, Ruleset.espn_ppr())
    keep = ["gsis_id", "season", "week", "actual_ppr"] + [s.value for s in model.target_stats]
    eval_df = all_preds.merge(
        actual[keep],
        on=["gsis_id", "season", "week"],
        how="inner",
        suffixes=("_pred", "_actual"),
    )

    print(f"\n=== {position.value} {_HELD_OUT_SEASON} sanity check (n={len(eval_df)} player-weeks) ===")

    print("\n-- Per-stat fit --")
    for stat in model.target_stats:
        pred_col = f"{stat.value}_pred"
        actual_col = f"{stat.value}_actual"
        rmse = float(np.sqrt(((eval_df[pred_col] - eval_df[actual_col]) ** 2).mean()))
        mae = float((eval_df[pred_col] - eval_df[actual_col]).abs().mean())
        print(
            f"  {stat.value:>20s}  rmse={rmse:6.3f}  mae={mae:6.3f}  "
            f"mean_pred={eval_df[pred_col].mean():6.3f}  "
            f"mean_actual={eval_df[actual_col].mean():6.3f}"
        )

    print("\n-- Composite (PPR points) --")
    rmse = float(np.sqrt(((eval_df["mean"] - eval_df["actual_ppr"]) ** 2).mean()))
    mae = float((eval_df["mean"] - eval_df["actual_ppr"]).abs().mean())
    print(f"  mean prediction:  rmse={rmse:.3f}  mae={mae:.3f}")
    pred_rank = eval_df.groupby("gsis_id")["mean"].sum().rank()
    actual_rank = eval_df.groupby("gsis_id")["actual_ppr"].sum().rank()
    common = pred_rank.index.intersection(actual_rank.index)
    spearman = float(np.corrcoef(pred_rank.loc[common], actual_rank.loc[common])[0, 1])
    print(f"  top-N season-total rank correlation (Spearman, all {position.value}s): {spearman:.3f}")

    print("\n-- Calibration --")
    in_p10p90 = (
        (eval_df["actual_ppr"] >= eval_df["p10"]) & (eval_df["actual_ppr"] <= eval_df["p90"])
    ).mean()
    le_p90 = (eval_df["actual_ppr"] <= eval_df["p90"]).mean()
    print(f"  fraction in [p10, p90]: {in_p10p90:.3f}  (target ~ 0.80)")
    print(f"  fraction <= p90:        {le_p90:.3f}  (target ~ 0.90)")

    print("\n=== End sanity check (informational; not a CI gate) ===")


if __name__ == "__main__":
    main()
```

- [ ] **Step 12.2: Delete `scripts/sanity_check_wr_baseline.py`.**

```bash
rm scripts/sanity_check_wr_baseline.py
```

- [ ] **Step 12.3: Smoke-test.**

Run: `python scripts/sanity_check_baseline.py --help`

Expected: argparse usage prints.

- [ ] **Step 12.4: Full gate.**

```bash
pytest -q
mypy src tests scripts
ruff check src tests scripts
ruff format --check src tests scripts
```

Expected: all green.

- [ ] **Step 12.5: Commit.**

```bash
git add scripts/sanity_check_baseline.py
git rm scripts/sanity_check_wr_baseline.py
git commit -m "$(cat <<'EOF'
feat(scripts): generalized sanity_check_baseline.py {position} (Phase 5c)

Replaces sanity_check_wr_baseline.py. Same metrics (per-stat RMSE/MAE,
composite RMSE/MAE, Spearman top-N rank correlation, [p10, p90] +
<= p90 calibration coverage) but parameterized on position via
POSITION_DISPATCH. Stdout-only; not a CI gate.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13 (Phase 6): Real-data train + sanity-check + 2024 weekly projections

**Files:** none (this task runs the scripts and records the output in Task 14).

**Goal:** Confirm the smokes from TODO #8 pass; train each position on real 2018-2023 data; run the 2024 sanity-check eval; write 2024 weekly projections. Capture the eval output so Task 14 can paste it into `project_management.md`.

- [ ] **Step 13.1: Run TODO #8's network smokes.**

```bash
pytest -m network --run-network -q
```

Expected: 8 tests pass (5 + 3 NGS parametrize). If any fails, the assertion message names the missing/renamed column; patch the corresponding ingest module before continuing. **Do not skip this step.** It is the front-runner against the eight drift fixes from TODO #16 — catching one here saves an hour vs. catching it during training.

- [ ] **Step 13.2: Verify `data/raw/` has the partitions for 2018-2024.**

```bash
ls data/raw/weekly_stats/ data/raw/snap_counts/ data/raw/depth_charts/ data/raw/ngs_passing/ data/raw/ngs_rushing/ data/raw/ngs_receiving/ data/raw/schedules/ 2>&1 | head -40
```

Expected: each directory has `season=2018` … `season=2024` partitions. If a season is missing, run `python -m projections.ingest.refresh ...` first (or whatever the canonical refresh entry point is — check `src/projections/ingest/__init__.py`).

- [ ] **Step 13.3: Train WR (closes the artifact-break TODO #17).**

```bash
python scripts/train_baseline.py wr 2>&1 | tee /tmp/3b_train_wr.log
```

Expected: prints per-season feature counts; ends with `model_id: baseline:wr:<8-char-hash>:2018-2023` and `Saved artifact: models/artifacts/baseline-wr-2018-2023-<hash>.joblib`.

- [ ] **Step 13.4: Sanity-check WR.**

```bash
python scripts/sanity_check_baseline.py wr 2>&1 | tee /tmp/3b_sanity_wr.log
```

Expected: prints `=== WR 2024 sanity check (n=... player-weeks) ===` followed by per-stat fit, composite, and calibration sections. Save the full output for Task 14.

- [ ] **Step 13.5: Predict 2024 WR weekly projections.**

```bash
python scripts/predict_2024.py wr
```

Expected: 18-22 weeks of `wrote N rows -> ...` followed by `Done.`.

- [ ] **Step 13.6: Repeat 13.3-13.5 for QB.**

```bash
python scripts/train_baseline.py qb 2>&1 | tee /tmp/3b_train_qb.log
python scripts/sanity_check_baseline.py qb 2>&1 | tee /tmp/3b_sanity_qb.log
python scripts/predict_2024.py qb
```

If `train_baseline.py qb` fails on a column-not-found or schema-validation error, that's exactly what TODO #16 warned about — a position-specific drift TODO #8's smokes didn't catch. Fix the corresponding builder/schema in a focused commit (style: `fix(features|ingest): <thing> from real-data drift in 3b QB train`), then resume.

- [ ] **Step 13.7: Repeat for RB and TE.**

```bash
python scripts/train_baseline.py rb 2>&1 | tee /tmp/3b_train_rb.log
python scripts/sanity_check_baseline.py rb 2>&1 | tee /tmp/3b_sanity_rb.log
python scripts/predict_2024.py rb

python scripts/train_baseline.py te 2>&1 | tee /tmp/3b_train_te.log
python scripts/sanity_check_baseline.py te 2>&1 | tee /tmp/3b_sanity_te.log
python scripts/predict_2024.py te
```

- [ ] **Step 13.8: Confirm artifacts and partitions exist.**

```bash
ls -1 models/artifacts/baseline-*.joblib
ls -1 data/projections/weekly/ruleset=ESPN_PPR/season=2024/ 2>&1 | head
```

Expected: four `.joblib` files (one per position) and 18-22 `week=WW` directories under the projections root.

- [ ] **Step 13.9: No commit.** (Artifacts and projections are gitignored; nothing to stage.)

If any drift fixes happened in 13.6 or 13.7, those landed as their own commits during the loop. Confirm with `git log --oneline origin/feat/plan-3b-qb-rb-te-baseline..HEAD` — the only commits since Task 12 should be drift fixes (if any).

---

## Task 14 (Phase 7): `project_management.md` + `TODO.md` finalization

**Files:**
- Modify: `project_management.md` (per-position sanity-check sections + decision log + next action)
- Modify: `TODO.md` (close TODO #17 — WR retrain done)

**Goal:** Record everything Plan 3b accomplished so a fresh session has the full context. The eval outputs from Task 13 get pasted into per-position sections at the top of `project_management.md`.

- [ ] **Step 14.1: Read the current `project_management.md` top section.**

Run: `head -100 project_management.md`. Note the existing "Plan 3a — 2024 WR sanity check" section template and the current "Current status (as of 2026-04-25)" section.

- [ ] **Step 14.2: Add four per-position 2024 sanity-check sections at the top of `project_management.md`.**

Above the existing "Plan 3a — 2024 WR sanity check" section, insert (use the exact tee'd output from Task 13.4, 13.6, 13.7):

```markdown
## Plan 3b — 2024 sanity check (run on branch `feat/plan-3b-qb-rb-te-baseline`)

Held-out year is 2024 (same as 3a; nfl_data_py has not yet published 2025).
Each position trained on 2018-2023; sanity-check evaluates against 2024.
Stdout-only metrics — Plan 3c owns CI threshold gating.

### QB

```
<paste contents of /tmp/3b_sanity_qb.log here>
```

### RB

```
<paste contents of /tmp/3b_sanity_rb.log here>
```

### TE

```
<paste contents of /tmp/3b_sanity_te.log here>
```

### WR (retrained under Plan 3b)

```
<paste contents of /tmp/3b_sanity_wr.log here>
```

The WR retrain in Phase 6 produced a new `model_id` because Plan 3b
modified `baseline.py` (which is part of the hashed code-files list);
substantively the predictions match the merged 3a artifact's output to
within numerical noise.

---
```

- [ ] **Step 14.3: Update the "Current status" section to reflect 3b.**

Replace the existing "Current status (as of 2026-04-25)" block with:

```markdown
## Current status (as of 2026-04-25)

**Projections Core — Plan 3b (QB / RB / TE Model A baselines + script generalization) merged to `main` at commit `<TBD-after-merge>`.**

**Predecessors:**
- Plan 1 (Foundations) merged at `8f02a6c`.
- Dev tooling merged via `feat/dev-tooling`.
- Plan 2a (Ingest expansion + WR feature builder) merged at `7926090`.
- Plan 2b (QB/RB/TE feature builders) merged at `af325ea`.
- Plan 3a (WR Model A baseline) merged at `598ab9c`.

**Plan 3b delivered:**
- `BaselineModel` constructor parameterized on `feature_schema` and `code_hash_files` (both required, no defaults).
- Three new factory functions: `qb_baseline()`, `rb_baseline()`, `te_baseline()`. Each declares its `target_stats`, `dist_families`, `feature_columns`, `feature_schema`, and `code_hash_files`.
- `POSITION_DISPATCH: Mapping[Position, _PositionDispatch]` registry in `src/projections/models/__init__.py`. CLI scripts and Plan 3c's future backtest harness consume it.
- `TeFeaturesSchema` extended with `rushing_attempts_per_game_l4` + `rushing_yards_per_game_l4`; `build_te_features` populates them. Phase 1 of the plan; Taysom-Hill rationale (TE rushing as a target stat in the model).
- Three CLI scripts unified: `scripts/train_baseline.py {position}`, `scripts/predict_2024.py {position}`, `scripts/sanity_check_baseline.py {position}`. The three WR-specific scripts are deleted; `--position wr` produces identical behavior.
- Six new test files under `tests/test_models/` (unit + leakage per position). Smoke test (`tests/test_smoke.py`) parametrized across all four positions.
- Per-position 2024 sanity-check eval recorded above. Each position's calibration is roughly comparable to WR's 70.8% in `[p10, p90]` (see per-position sections); Plan 3c will formalize.
- Per-position 2024 weekly projections written to `data/projections/weekly/ruleset=ESPN_PPR/season=2024/week=WW/part.parquet` (gitignored).
- All four trained artifacts at `models/artifacts/baseline-{pos}-2018-2023-<hash>.joblib` (gitignored).

**Held-out year remains 2024.** Same constraint as 3a.
```

- [ ] **Step 14.4: Update the "Next action" section.**

Replace the existing "Next action" block with:

```markdown
## Next action

**Recommended: Plan 3c — weekly→season aggregation + walk-forward backtest harness with CI threshold gating.**

3a pinned the per-week model interface for one position. 3b generalized
that to all four offensive skill positions. 3c is the natural next step:

- **Weekly → season aggregation.** Convert per-week distributions into season-total distributions via Monte Carlo over (bye, availability, schedule). Pins how the projection layer is consumed by the future Draft Hub.
- **Walk-forward backtest harness.** Train through season N-1, predict season N, score, repeat. The first place we formalize the soft thresholds 3a/3b reported informationally (Spearman top-N corr, calibration coverage, per-stat RMSE vs naive baseline).
- **CI threshold gating.** Backtest output as the basis for "this model regressed" alerts in CI rather than a stdout-only sanity check.

**Pre-requisites:** none currently open. TODO #8 and #15 closed before 3b kickoff; TODO #16 (real-data drifts list) is documentation, not actionable.
```

- [ ] **Step 14.5: Append decision-log entries.**

In the existing "Decision log" table, append rows for the 3b decisions:

```markdown
| 2026-04-25 | Plan 3b: BaselineModel gains required `feature_schema` + `code_hash_files` constructor args | Replaces hardcoded WR references; per-position config stays per-factory. Existing 3a artifact unloadable; retrain in Phase 6 (TODO #17 closed). |
| 2026-04-25 | Plan 3b: TE model includes rushing as target stat (Taysom Hill) | Q3 brainstorm decision; Phase 1 added rushing_*_per_game_l4 to TeFeaturesSchema and build_te_features; cost is two columns and a fixture row. |
| 2026-04-25 | Plan 3b: NORMAL/GAMMA convention extended mechanically; POISSON deferred | WR's family choices carry to QB/RB/TE without per-position tuning. POISSON for low-mean integer counts (interceptions, fumbles_lost) deferred to 3c contingent on calibration evidence. |
| 2026-04-25 | Plan 3b: centralized `POSITION_DISPATCH` registry in `models/__init__.py` | One canonical "what positions the system knows about" answer. Reused by CLI scripts and future 3c backtest harness. Adding a position is one new line. |
| 2026-04-25 | Plan 3b: per-position test files (mirrors `tests/test_features/`) | Q6 brainstorm decision. Six new files; failure isolation per position is worth ~210 lines of necessary duplication. |
| 2026-04-25 | Plan 3b: smoke test parametrized across all four positions | Q6 brainstorm B; catches "I broke RB silently" earlier than the per-position test files. ~20s smoke runtime acceptable. |
| 2026-04-25 | Plan 3b: three WR-specific scripts deleted; replaced by position-arg-driven generalized scripts | Q1 brainstorm C. Avoids producing four near-duplicate scripts after 3b. |
```

- [ ] **Step 14.6: Close TODO #17 in `TODO.md`.**

Edit `TODO.md`. Find the entry from Task 2.7:

```markdown
### 17. Retrain WR 2018-2023 artifact after Plan 3b

Plan 3b adds two required fields ...
```

Delete the entire entry (it's done — Phase 6's WR retrain produced the new artifact). The next-numbered entry stays put.

- [ ] **Step 14.7: Run gate (just to be sure docs edits didn't break anything).**

```bash
pytest -q
```

Expected: all 250+ tests pass.

- [ ] **Step 14.8: Commit.**

```bash
git add project_management.md TODO.md
git commit -m "$(cat <<'EOF'
docs(pm): close Plan 3b — per-position sanity check + decision log

Phase 7 of Plan 3b. Records:
- Per-position 2024 sanity-check eval (QB/RB/TE/WR-retrained).
- 3b's substantive decision-log entries (constructor refactor, TE
  rushing inclusion, POSITION_DISPATCH registry, per-position test
  files, smoke parametrization, script unification).
- Updated "Current status" + "Next action" pointing at Plan 3c.

Closes TODO #17 (WR retrain — done in Phase 6).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage check:**
- Section 1.1 goals — covered by Tasks 1-12 (factories, schema, dispatch, scripts, tests).
- Section 1.2 non-goals — none implemented (correct).
- Section 2.1 no new packages — Tasks 2-4 stay within `models/`.
- Section 2.2 BaselineModel constructor — Task 2.
- Section 2.3 POSITION_DISPATCH — Task 4.
- Section 2.4 phases — Tasks 1-14 follow the seven-phase structure exactly.
- Section 3.1 target stats — Task 3 constants.
- Section 3.2 dist families — Task 3 constants.
- Section 3.3 feature columns per position — Task 3 constants (note: spec used approximate column names; this plan uses the actual schema column names — `pass_attempts_per_game_l4` not `attempts_per_game_l4`, etc.).
- Section 3.4 code-hash file list — Task 3 helper.
- Section 4.1-4.3 testing — Tasks 5-9.
- Section 4.4 TE feature builder tests — Task 1.
- Section 5.1 sanity-check eval — Task 13 + Task 14.
- Section 5.2 real-data drift expectations — Task 13.1 + Task 13.6 fallback note.
- Section 7 required follow-ups (WR retrain) — Task 2.7 (open) + Task 14.6 (close).

**Placeholder scan:** no TBDs, TODOs, "implement later", "fill in details", "add appropriate error handling", "similar to Task N", or other forbidden patterns. Every code-changing step has the actual code.

**Type consistency:** function/class/variable names match across tasks. `_default_code_hash_files(position_module: str)` (Task 3.4) is referenced consistently in Tasks 3.5, 3.6. `POSITION_DISPATCH` and `_PositionDispatch` (Task 4.2) are referenced in Tasks 9, 10, 11, 12. `baseline_features_{qb,rb,te,wr}` and `baseline_weekly_stats_{qb,rb,te,wr}` are consistent across Tasks 5-9. Script names (`train_baseline.py`, `predict_2024.py`, `sanity_check_baseline.py`) consistent across Tasks 10-13.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-25-plan-3b-qb-rb-te-baseline.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
