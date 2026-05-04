# WR Trajectory Features Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the trajectory family probe (PR #25) into the production WR feature pipeline. Append 4 nullable-float trajectory columns to `WrFeaturesSchema`, wire `attach_trajectory_features` into `build_wr_features`, run the dual-run adoption gate on `(BaselineModel, WR)`, ship if `ADOPT` (probe predicted -0.0414 fpts).

**Architecture:** Schema delta + builder integration + caller plumbing, structured as a tight analog of PR #21's RB PBP integration. Promotes the script-private `_build_draft_lookup` to a public helper in `trajectory_features.py`. Updates `baseline.py:_WR_FEATURE_COLUMNS` (the spec gap PR #21 caught at `9895dee`) — lightgbm derives feature lists from the schema dynamically and auto-picks-up the new columns.

**Tech Stack:** Python 3.11, pandas (pyarrow-backed strings + nullable Int64/Float64), pandera (DataFrameModel + `strict="filter"`), pytest, scikit-learn (`RidgeCV` via `BaselineModel`), LightGBM (Quantile / NB-2 sub-models), pre-commit hooks (ruff, mypy strict, pre-commit hygiene).

**Spec:** `docs/superpowers/specs/2026-05-03-wr-trajectory-features-design.md`.

---

## File Structure (decomposition lock-in)

**Modify:**
- `src/projections/schemas.py` — extend `WrFeaturesSchema` with 4 nullable-float trajectory columns (Phase 1).
- `src/projections/features/trajectory_features.py` — add public `build_draft_lookup` helper (Phase 1).
- `scripts/build_trajectory_override.py` — replace private `_build_draft_lookup` with the imported public helper (Phase 1).
- `src/projections/features/wr.py` — wire `attach_trajectory_features` + new `draft_picks` kwarg (Phase 2).
- `src/projections/features/qb.py`, `rb.py`, `te.py` — add unused `draft_picks: pd.DataFrame = _EMPTY_DRAFT_PICKS` kwarg for signature symmetry (Phase 3).
- `src/projections/models/baseline.py` — extend `_WR_FEATURE_COLUMNS` tuple with 4 new column names (Phase 3).
- `scripts/refresh_features.py`, `train_baseline.py`, `predict_2024.py`, `sanity_check_baseline.py` — load `draft_picks` partition + thread through to all 4 builders (Phase 4).
- `project_management.md`, `TODO.md` — decision-log + close TODO #24's trailing-8-game-unit branch (Phase 6, conditional on gate verdict).

**Add (tests):**
- `tests/test_features/conftest.py` — gain `wr_draft_picks` fixture (Phase 1).
- `tests/test_features/test_trajectory_features.py` — add 3 tests covering `build_draft_lookup`'s empty / happy / NaN-`draft_age` paths (Phase 1).
- `tests/test_features/test_wr.py` — extend happy-path; 4 new trajectory join-side tests (Phase 2).

**Add (reports, Phase 5 — committed but not code):**
- `reports/adoption_gate_wr_trajectory_features.{md,csv}` — gate output.
- `reports/wr_trajectory_features_summary.md` — decision log + per-mode table + probe-vs-gate calibration + coverage stats.

---

## Phase 1 — Schema + helper promotion + override-script + helper tests

Goal: ship the `WrFeaturesSchema` column delta and the `build_draft_lookup` public-helper promotion. No builder wiring yet — the schema accepts NaN on the 4 new cols, so the existing WR cache will validate against an empty-trajectory build (which is fine; the feature cache will be regenerated in Phase 5 with real values). After this phase: existing 48+ tests pass; new helper tests pass.

### Task 1: Add `wr_draft_picks` fixture

**Files:**
- Modify: `tests/test_features/conftest.py`

- [ ] **Step 1: Read the existing fixture file**

Run: `head -160 tests/test_features/conftest.py`
Expected: shows `wr_weekly_stats`, `wr_snap_counts`, `wr_depth_charts` shapes — note the `gsis_id` values used (`00-0036322`, `00-0036323`, `00-0034950`, `00-0099777`).

- [ ] **Step 2: Append `wr_draft_picks` fixture to `tests/test_features/conftest.py`**

Add at end of file:

```python
@pytest.fixture
def wr_draft_picks() -> pd.DataFrame:
    """Synthetic draft_picks for the WRs used across the WR fixture set.

    Includes the four canonical WR gsis_ids from wr_weekly_stats /
    wr_depth_charts. The rookie 00-0099777 is intentionally drafted in
    2024 (the test season), so is_rookie computes to 1.0 from the
    primary draft_lookup path. The other three are veteran drafts.
    """
    return pd.DataFrame(
        [
            # Justin Jefferson: 2020 draft, 21yo, currently a 4-year vet in 2024.
            {"gsis_id": "00-0036322", "draft_year": 2020, "draft_round": 1,
             "draft_overall_pick": 22, "pfr_id": "JeffJu00", "draft_age": 21.0},
            # Jaylen Waddle: 2021 draft, 22yo at draft.
            {"gsis_id": "00-0036323", "draft_year": 2021, "draft_round": 1,
             "draft_overall_pick": 6, "pfr_id": "WaddJa00", "draft_age": 22.0},
            # CeeDee Lamb: 2020 draft, 21yo, NaN draft_age branch.
            {"gsis_id": "00-0034950", "draft_year": 2020, "draft_round": 1,
             "draft_overall_pick": 17, "pfr_id": "LambCe00", "draft_age": float("nan")},
            # Rookie WR 00-0099777: 2024 draft.
            {"gsis_id": "00-0099777", "draft_year": 2024, "draft_round": 2,
             "draft_overall_pick": 50, "pfr_id": "RookXX00", "draft_age": 22.0},
        ]
    ).astype(
        {
            "gsis_id": "string[pyarrow]",
            "draft_year": pd.Int64Dtype(),
            "draft_round": pd.Int64Dtype(),
            "draft_overall_pick": pd.Int64Dtype(),
            "pfr_id": "string[pyarrow]",
            "draft_age": pd.Float64Dtype(),
        }
    )
```

- [ ] **Step 3: Verify fixture loads**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_wr.py -k "returns_validated_frame" -v`
Expected: PASS (4 existing wr fixtures + the new unused one don't conflict).

- [ ] **Step 4: Commit**

```bash
git add tests/test_features/conftest.py
git commit -m "test(wr): add wr_draft_picks synthetic fixture"
```

---

### Task 2: Promote `build_draft_lookup` (TDD: tests first)

**Files:**
- Modify: `src/projections/features/trajectory_features.py`
- Modify: `tests/test_features/test_trajectory_features.py`

- [ ] **Step 1: Read current `trajectory_features.py` end-of-file region**

Run: `tail -30 src/projections/features/trajectory_features.py`
Expected: ends with `build_trajectory_overrides` definition; no public `build_draft_lookup` exists yet.

- [ ] **Step 2: Write 3 failing tests for `build_draft_lookup`**

Append to `tests/test_features/test_trajectory_features.py`:

```python
from projections.features.trajectory_features import build_draft_lookup


def test_build_draft_lookup_empty_returns_empty_dict() -> None:
    empty = pd.DataFrame(
        columns=["gsis_id", "draft_year", "draft_age"]
    ).astype(
        {
            "gsis_id": "string[pyarrow]",
            "draft_year": pd.Int64Dtype(),
            "draft_age": pd.Float64Dtype(),
        }
    )
    assert build_draft_lookup(empty) == {}


def test_build_draft_lookup_drafted_player_produces_year_age_tuple() -> None:
    df = pd.DataFrame(
        [{"gsis_id": "00-0036322", "draft_year": 2020, "draft_age": 21.0}]
    ).astype(
        {
            "gsis_id": "string[pyarrow]",
            "draft_year": pd.Int64Dtype(),
            "draft_age": pd.Float64Dtype(),
        }
    )
    lookup = build_draft_lookup(df)
    assert lookup == {"00-0036322": (2020, 21.0)}


def test_build_draft_lookup_nan_draft_age_preserved_as_nan() -> None:
    df = pd.DataFrame(
        [{"gsis_id": "00-0034950", "draft_year": 2020, "draft_age": float("nan")}]
    ).astype(
        {
            "gsis_id": "string[pyarrow]",
            "draft_year": pd.Int64Dtype(),
            "draft_age": pd.Float64Dtype(),
        }
    )
    lookup = build_draft_lookup(df)
    assert "00-0034950" in lookup
    year, age = lookup["00-0034950"]
    assert year == 2020
    assert pd.isna(age)
```

- [ ] **Step 3: Run tests, verify they fail with ImportError**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_trajectory_features.py -k "build_draft_lookup" -v`
Expected: 3 ERRORS / FAILS — `ImportError: cannot import name 'build_draft_lookup'`.

- [ ] **Step 4: Add public `build_draft_lookup` to `trajectory_features.py`**

Append to `src/projections/features/trajectory_features.py` (just before `_FANTASY_POSITIONS_ENUM`):

```python
def build_draft_lookup(draft_picks: pd.DataFrame) -> DraftLookup:
    """Convert a draft_picks DataFrame into a {gsis_id: (draft_year, draft_age)} lookup.

    Args:
        draft_picks: frame matching ``DraftPicksSchema``. Must include
            ``gsis_id``, ``draft_year``, ``draft_age``. ``draft_age`` may be
            NaN (rare; per ``nfl_data_py``, missing for players whose
            birthdate isn't in the source). Empty frame returns ``{}``.

    Returns:
        Lookup keyed by ``gsis_id``. Missing keys (UDFAs, pre-1980
        draftees) route to the inferred-draft-year fallback inside
        ``compute_age`` / ``compute_is_rookie``.
    """
    if draft_picks.empty:
        return {}
    return {
        str(row["gsis_id"]): (
            int(row["draft_year"]),
            float(row["draft_age"]) if pd.notna(row["draft_age"]) else float("nan"),
        )
        for _, row in draft_picks.iterrows()
    }
```

- [ ] **Step 5: Run tests, verify all 3 pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_trajectory_features.py -k "build_draft_lookup" -v`
Expected: 3 PASSED.

- [ ] **Step 6: Run full trajectory_features test file to confirm no regression**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_trajectory_features.py -v`
Expected: all existing tests + 3 new ones PASS.

- [ ] **Step 7: Commit**

```bash
git add src/projections/features/trajectory_features.py tests/test_features/test_trajectory_features.py
git commit -m "refactor(trajectory): promote build_draft_lookup to public helper"
```

---

### Task 3: Update `scripts/build_trajectory_override.py` to use public helper

**Files:**
- Modify: `scripts/build_trajectory_override.py`

- [ ] **Step 1: Read current import block + `_build_draft_lookup` definition**

Run: `sed -n '20,35p;105,116p' scripts/build_trajectory_override.py`
Expected: shows `from projections.features.trajectory_features import (DraftLookup, build_trajectory_overrides,)` and the local `_build_draft_lookup` definition at lines 106-114.

- [ ] **Step 2: Add `build_draft_lookup` to the import block, remove the private definition**

Edit the import (lines 26-29):

```python
from projections.features.trajectory_features import (
    build_draft_lookup,
    build_trajectory_overrides,
)
```

(The `DraftLookup` import is dropped — only the script needs the runtime helper now; the type alias stays in `trajectory_features.py` for `attach_trajectory_features`'s signature.)

Delete the entire `_build_draft_lookup` function (lines 106-114).

In the `main()` body, change `draft_lookup = _build_draft_lookup(draft_picks)` to `draft_lookup = build_draft_lookup(draft_picks)`.

- [ ] **Step 3: Run the script's CLI tests to verify the import refactor doesn't break the script**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scripts/test_build_trajectory_override_cli.py -v`
Expected: all 4 existing tests PASS.

- [ ] **Step 4: Run mypy + ruff to catch any unused-import warnings**

Run: `.venv/Scripts/python.exe -m mypy src tests scripts && .venv/Scripts/python.exe -m ruff check src tests scripts`
Expected: zero violations.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_trajectory_override.py
git commit -m "refactor(trajectory): use public build_draft_lookup in override script"
```

---

### Task 4: Add 4 trajectory columns to `WrFeaturesSchema`

**Files:**
- Modify: `src/projections/schemas.py:484-537`

- [ ] **Step 1: Read the current `WrFeaturesSchema` definition**

Run: `sed -n '484,540p' src/projections/schemas.py`
Expected: shows the schema ending with `opp_allowed_wr_fppg_l4` then the `Config` class.

- [ ] **Step 2: Append 4 columns + comment block before the `Config` class**

Edit `src/projections/schemas.py`, find the line:
```python
    # Opponent strength (proxy: opp's allowed WR fantasy points/game over trailing 4)
    opp_allowed_wr_fppg_l4: Series[float] = pa.Field(ge=0, nullable=True)

    class Config:
```

Replace with:
```python
    # Opponent strength (proxy: opp's allowed WR fantasy points/game over trailing 4)
    opp_allowed_wr_fppg_l4: Series[float] = pa.Field(ge=0, nullable=True)

    # Trajectory features (PR #25 family probe + 2026-05-03 WR integration
    # spec). All four are structurally sparse: age + is_rookie need a
    # draft_picks lookup hit (or the inferred fallback) and so cover ~88-97%
    # of player-weeks; the trend cols need 8 prior active games and so cover
    # ~50% of player-weeks. NaN where coverage is missing; BaselineModel
    # imputes with feature mean, lightgbm consumes NaN natively.
    age: Series[float] = pa.Field(ge=15, le=50, nullable=True)
    is_rookie: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    volume_trend_l4_minus_prior_l4: Series[float] = pa.Field(nullable=True)
    snap_pct_change_l4_vs_prior_l4: Series[float] = pa.Field(ge=-1, le=1, nullable=True)

    class Config:
```

- [ ] **Step 3: Confirm schema validation still parses**

Run: `.venv/Scripts/python.exe -c "from projections.schemas import WrFeaturesSchema; cols = list(WrFeaturesSchema.to_schema().columns.keys()); print(len(cols), 'columns'); assert 'age' in cols; assert 'is_rookie' in cols; assert 'volume_trend_l4_minus_prior_l4' in cols; assert 'snap_pct_change_l4_vs_prior_l4' in cols; print('schema OK')"`
Expected: `26 columns` (was 22, +4) and `schema OK`.

- [ ] **Step 4: Run schema test file to confirm no regression**

Run: `.venv/Scripts/python.exe -m pytest tests/test_schemas/ -v`
Expected: all PASS.

- [ ] **Step 5: Run existing WR builder tests — they should now FAIL because output frame is missing the 4 new columns**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_wr.py -v 2>&1 | tail -20`
Expected: FAIL on `test_build_wr_features_returns_validated_frame` with a SchemaError mentioning `age`, `is_rookie`, `volume_trend_l4_minus_prior_l4`, or `snap_pct_change_l4_vs_prior_l4` is required but missing. **This is expected** — Phase 2 wires the builder to produce them.

- [ ] **Step 6: Commit (with the WR test failures unfixed — Phase 2 closes them)**

```bash
git add src/projections/schemas.py
git commit -m "schema(wr): add 4 trajectory columns to WrFeaturesSchema

Schema-only change. WR builder tests will fail until Phase 2 wires
attach_trajectory_features into build_wr_features."
```

---

## Phase 2 — WR builder integration + tests

Goal: wire `attach_trajectory_features` into `build_wr_features` so the existing WR tests pass again. Add 4 new tests covering trajectory join-side correctness (drafted veteran, rookie, UDFA fallback, empty draft_picks).

### Task 5: Add `_EMPTY_DRAFT_PICKS` + integrate `attach_trajectory_features` into `build_wr_features`

**Files:**
- Modify: `src/projections/features/wr.py`

- [ ] **Step 1: Re-read `wr.py` to confirm current state**

Run: `head -50 src/projections/features/wr.py && echo "---" && tail -50 src/projections/features/wr.py`
Expected: imports include `Position`, `Ruleset`, `Stat`, `WrFeaturesSchema`; `_EMPTY_PBP` defined; `build_wr_features` signature has `pbp: pd.DataFrame = _EMPTY_PBP`; the assembly section ends with `return WrFeaturesSchema.validate(out)`.

- [ ] **Step 2: Add imports + `_EMPTY_DRAFT_PICKS` singleton + new kwarg + helper call**

Edit `src/projections/features/wr.py`. Replace the import block:

```python
from projections.features._opponent import opp_allowed_fppg
from projections.features._rolling import (
    latest_ngs_snapshot,
    trailing_4_per_player,
    trailing_n_share_in_group,
)
from projections.features._shared import build_game_environment, exact_week_mask, prior_mask
from projections.schemas import (
    _PYARROW_STR,
    Position,
    Ruleset,
    Stat,
    WrFeaturesSchema,
)
```

with:

```python
from projections.features._opponent import opp_allowed_fppg
from projections.features._rolling import (
    latest_ngs_snapshot,
    trailing_4_per_player,
    trailing_n_share_in_group,
)
from projections.features._shared import build_game_environment, exact_week_mask, prior_mask
from projections.features.trajectory_features import (
    attach_trajectory_features,
    build_draft_lookup,
)
from projections.schemas import (
    _PYARROW_STR,
    Position,
    Ruleset,
    Stat,
    WrFeaturesSchema,
)
```

Replace:

```python
# Module-level singleton for the unused `pbp` builder kwarg. Per ruff B008,
# we cannot put `pd.DataFrame()` directly in the function default.
_EMPTY_PBP: Final[pd.DataFrame] = pd.DataFrame()
```

with:

```python
# Module-level singletons for the optional builder kwargs. Per ruff B008,
# we cannot put `pd.DataFrame()` directly in function defaults.
_EMPTY_PBP: Final[pd.DataFrame] = pd.DataFrame()
_EMPTY_DRAFT_PICKS: Final[pd.DataFrame] = pd.DataFrame()
```

In the `build_wr_features` signature, add `draft_picks` after `pbp`:

```python
def build_wr_features(
    *,
    weekly_stats: pd.DataFrame,
    snap_counts: pd.DataFrame,
    depth_charts: pd.DataFrame,
    ngs_receiving: pd.DataFrame,
    schedules: pd.DataFrame,
    season: int,
    as_of_week: int,
    # pbp: reserved plumbing for future PBP-driven features (Plan 9 Phase 6
    # negative result, kept threaded for future plans). Currently unused.
    pbp: pd.DataFrame = _EMPTY_PBP,
    # draft_picks: consumed by the trajectory features (PR #25 + 2026-05-03
    # WR integration). Empty -> all rows route to inferred-draft-year fallback.
    draft_picks: pd.DataFrame = _EMPTY_DRAFT_PICKS,
) -> pd.DataFrame:
```

In the assembly section, immediately before `return WrFeaturesSchema.validate(out)`, insert the trajectory wiring:

```python
    # --- Trajectory features (PR #25 + 2026-05-03 WR integration) ---------
    # Helper does its own position filter + rolling, so pass the unfiltered
    # (but prior-mask-filtered) weekly_stats / snap_counts.
    draft_lookup = build_draft_lookup(draft_picks)
    traj_idx = out[["gsis_id", "season", "week", "team", "opponent"]].rename(
        columns={"opponent": "opp"}
    )
    traj = attach_trajectory_features(traj_idx, ws, sc, draft_lookup, Position.WR)
    out = out.merge(
        traj[
            [
                "gsis_id",
                "season",
                "week",
                "age",
                "is_rookie",
                "volume_trend_l4_minus_prior_l4",
                "snap_pct_change_l4_vs_prior_l4",
            ]
        ],
        on=["gsis_id", "season", "week"],
        how="left",
    )

    return WrFeaturesSchema.validate(out)
```

- [ ] **Step 3: Run the existing WR happy-path test to confirm trajectory wiring closes the schema gap**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_wr.py::test_build_wr_features_returns_validated_frame -v`
Expected: PASS — the 4 new schema columns are now produced (all-NaN since the test passes no `draft_picks` and no player has 8 prior games in the synthetic fixture, but that satisfies `nullable=True`).

- [ ] **Step 4: Run the full WR builder test file**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_wr.py tests/test_features/test_wr_leakage.py -v`
Expected: all existing tests PASS — trajectory cols are NaN throughout (no draft_picks passed), schema accepts NaN.

- [ ] **Step 5: Commit**

```bash
git add src/projections/features/wr.py
git commit -m "feat(wr): wire attach_trajectory_features into build_wr_features

Adds draft_picks kwarg (default empty) and computes the 4 trajectory
features via the existing attach_trajectory_features helper. Existing
WR tests pass with all-NaN trajectory cols (no draft_picks fixture);
4 new join-side tests follow."
```

---

### Task 6: Test — drafted veteran has computed `age` and `is_rookie=0.0`

**Files:**
- Modify: `tests/test_features/test_wr.py`

- [ ] **Step 1: Append the new test, importing the `wr_draft_picks` fixture from conftest**

Add at end of `tests/test_features/test_wr.py`:

```python
def test_build_wr_features_attach_trajectory_drafted_veteran(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    wr_draft_picks: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Justin Jefferson: drafted 2020 at age 21; in 2024 should be age 25, is_rookie=0."""
    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        pbp=fake_pbp_df,
        draft_picks=wr_draft_picks,
        season=2024,
        as_of_week=5,
    )
    jef = out[out["gsis_id"] == "00-0036322"].iloc[0]
    assert jef["age"] == 25.0  # 21 + (2024 - 2020)
    assert jef["is_rookie"] == 0.0
```

- [ ] **Step 2: Run the new test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_wr.py::test_build_wr_features_attach_trajectory_drafted_veteran -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_features/test_wr.py
git commit -m "test(wr): drafted-veteran age + is_rookie computed from draft_picks"
```

---

### Task 7: Test — rookie has `is_rookie=1.0` and `age == draft_age`

**Files:**
- Modify: `tests/test_features/test_wr.py`

- [ ] **Step 1: Append the rookie test**

Add to `tests/test_features/test_wr.py`:

```python
def test_build_wr_features_attach_trajectory_rookie(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    wr_draft_picks: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Inject rookie WR 00-0099777 (drafted 2024 in fixture); is_rookie=1.0,
    age=22.0, no prior 8 games so volume_trend / snap_pct_change are NaN."""
    extra_dc = pd.concat(
        [
            wr_depth_charts,
            pd.DataFrame(
                [
                    {
                        "gsis_id": "00-0099777",
                        "season": 2024,
                        "week": 5,
                        "team": "KC",
                        "position": "WR",
                        "depth_team": "WR2",
                        "depth_rank": 2,
                    }
                ]
            ).astype({"gsis_id": "string[pyarrow]"}),
        ],
        ignore_index=True,
    )

    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=extra_dc,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        pbp=fake_pbp_df,
        draft_picks=wr_draft_picks,
        season=2024,
        as_of_week=5,
    )
    rookie = out[out["gsis_id"] == "00-0099777"].iloc[0]
    assert rookie["is_rookie"] == 1.0
    assert rookie["age"] == 22.0
    assert pd.isna(rookie["volume_trend_l4_minus_prior_l4"])
    assert pd.isna(rookie["snap_pct_change_l4_vs_prior_l4"])
```

- [ ] **Step 2: Run the new test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_wr.py::test_build_wr_features_attach_trajectory_rookie -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_features/test_wr.py
git commit -m "test(wr): rookie age + is_rookie + NaN trends from primary draft path"
```

---

### Task 8: Test — UDFA (no draft_lookup entry) uses inferred-draft-year fallback

**Files:**
- Modify: `tests/test_features/test_wr.py`

- [ ] **Step 1: Append the UDFA test**

Add to `tests/test_features/test_wr.py`:

```python
def test_build_wr_features_attach_trajectory_udfa_fallback(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """UDFA WR (not in draft_picks fixture) falls back to inferred draft year
    = earliest weekly_stats appearance. Jefferson appears in fixture from
    week 1 of 2024, so inferred_year = 2024 -> is_rookie = 1.0,
    age = 2024 - 2024 + 22.0 = 22.0."""
    udfa_picks = pd.DataFrame(
        columns=["gsis_id", "draft_year", "draft_round",
                 "draft_overall_pick", "pfr_id", "draft_age"]
    ).astype(
        {
            "gsis_id": "string[pyarrow]",
            "draft_year": pd.Int64Dtype(),
            "draft_round": pd.Int64Dtype(),
            "draft_overall_pick": pd.Int64Dtype(),
            "pfr_id": "string[pyarrow]",
            "draft_age": pd.Float64Dtype(),
        }
    )
    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        pbp=fake_pbp_df,
        draft_picks=udfa_picks,
        season=2024,
        as_of_week=5,
    )
    jef = out[out["gsis_id"] == "00-0036322"].iloc[0]
    # Jefferson's earliest appearance in synthetic fixture is 2024 week 1
    # (no historical seasons in the fixture), so inferred_year = 2024.
    assert jef["age"] == 22.0  # 2024 - 2024 + 22.0
    assert jef["is_rookie"] == 1.0
```

- [ ] **Step 2: Run the new test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_wr.py::test_build_wr_features_attach_trajectory_udfa_fallback -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_features/test_wr.py
git commit -m "test(wr): UDFA fallback uses inferred-draft-year + 22.0 offset"
```

---

### Task 9: Test — empty `draft_picks` default routes every row to fallback without errors

**Files:**
- Modify: `tests/test_features/test_wr.py`

- [ ] **Step 1: Append the empty-draft-picks test**

Add to `tests/test_features/test_wr.py`:

```python
def test_build_wr_features_empty_draft_picks_default(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Calling build_wr_features without draft_picks (default empty) must
    not raise. Every row falls through inferred-draft-year and the schema
    validates with non-NaN age (~22.0 from the offset) for everyone."""
    # Note: NO draft_picks kwarg passed.
    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    WrFeaturesSchema.validate(out)
    assert "age" in out.columns
    # All rows should have age = inferred-fallback (22.0) since the
    # synthetic fixture's earliest-week is the target season.
    assert (out["age"] == 22.0).all()
```

- [ ] **Step 2: Run the new test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_wr.py::test_build_wr_features_empty_draft_picks_default -v`
Expected: PASS.

- [ ] **Step 3: Run the full WR test files end-to-end as a regression check**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_wr.py tests/test_features/test_wr_leakage.py tests/test_features/test_trajectory_features.py -v 2>&1 | tail -10`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_features/test_wr.py
git commit -m "test(wr): empty draft_picks default routes all rows to fallback"
```

---

## Phase 3 — Per-position builder kwarg additions + baseline.py

Goal: extend `build_qb_features`, `build_rb_features`, `build_te_features` to accept the unused-but-plumbed `draft_picks` kwarg (mirror the existing `pbp` precedent). Extend `_WR_FEATURE_COLUMNS` in `baseline.py`. After this phase: per-position smoke tests still pass; lightgbm dynamic feature lists include the 4 new WR columns; baseline's hardcoded WR list includes them too.

### Task 10: Add `_EMPTY_DRAFT_PICKS` + `draft_picks` kwarg to `build_qb_features`

**Files:**
- Modify: `src/projections/features/qb.py`

- [ ] **Step 1: Re-read `qb.py` for the `_EMPTY_PBP` definition + `build_qb_features` signature**

Run: `head -50 src/projections/features/qb.py`
Expected: shows `_EMPTY_PBP: Final[pd.DataFrame] = pd.DataFrame()` and the function signature with `pbp: pd.DataFrame = _EMPTY_PBP`.

- [ ] **Step 2: Add the singleton + kwarg**

Edit `src/projections/features/qb.py`. Replace:

```python
# Module-level singleton for the unused `pbp` builder kwarg. Per ruff B008,
# we cannot put `pd.DataFrame()` directly in the function default.
_EMPTY_PBP: Final[pd.DataFrame] = pd.DataFrame()
```

with:

```python
# Module-level singletons for the optional builder kwargs. Per ruff B008,
# we cannot put `pd.DataFrame()` directly in function defaults.
_EMPTY_PBP: Final[pd.DataFrame] = pd.DataFrame()
_EMPTY_DRAFT_PICKS: Final[pd.DataFrame] = pd.DataFrame()
```

In the `build_qb_features` signature, add `draft_picks` after `pbp`:

```python
def build_qb_features(
    *,
    # ... existing args ...
    pbp: pd.DataFrame = _EMPTY_PBP,
    # draft_picks: reserved plumbing for the trajectory feature family
    # (PR #25); accept-and-ignore until the QB-specific integration plan
    # fires. Mirror the `pbp` precedent.
    draft_picks: pd.DataFrame = _EMPTY_DRAFT_PICKS,
) -> pd.DataFrame:
```

(Use `# noqa: ARG001` on the parameter line if ruff flags it as unused. Check first by running ruff.)

- [ ] **Step 3: Run QB tests + lint to verify**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_qb.py -v && .venv/Scripts/python.exe -m ruff check src/projections/features/qb.py`
Expected: all tests PASS; ruff zero violations. (If ruff flags the unused arg, add `# noqa: ARG001` on the same line as `draft_picks: pd.DataFrame = _EMPTY_DRAFT_PICKS,` — note: check first whether `_EMPTY_PBP` has the same suppression; if not, ruff is configured to allow unused kwargs in this codebase.)

- [ ] **Step 4: Commit**

```bash
git add src/projections/features/qb.py
git commit -m "feat(qb): add draft_picks kwarg for plumbing symmetry (unused)"
```

---

### Task 11: Add `_EMPTY_DRAFT_PICKS` + `draft_picks` kwarg to `build_rb_features`

**Files:**
- Modify: `src/projections/features/rb.py`

- [ ] **Step 1: Re-read `rb.py` for the singleton + signature**

Run: `head -50 src/projections/features/rb.py`
Expected: shows `_EMPTY_PBP` and `build_rb_features` signature with `pbp: pd.DataFrame = _EMPTY_PBP`. Note `pbp` IS used in RB (PR #21 wires `attach_pbp_family_features`).

- [ ] **Step 2: Add the singleton + kwarg analogous to Task 10**

Edit `src/projections/features/rb.py`. Replace the `_EMPTY_PBP` block with the dual-singleton block (same as Task 10 step 2). Add `draft_picks` to the `build_rb_features` signature with the same comment/default as Task 10. **Do not wire it in the body** — it stays unused for RB until a future spec.

- [ ] **Step 3: Run RB tests + lint**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_rb.py -v && .venv/Scripts/python.exe -m ruff check src/projections/features/rb.py`
Expected: all RB tests PASS; ruff clean.

- [ ] **Step 4: Commit**

```bash
git add src/projections/features/rb.py
git commit -m "feat(rb): add draft_picks kwarg for plumbing symmetry (unused)"
```

---

### Task 12: Add `_EMPTY_DRAFT_PICKS` + `draft_picks` kwarg to `build_te_features`

**Files:**
- Modify: `src/projections/features/te.py`

- [ ] **Step 1: Re-read `te.py` for the singleton + signature**

Run: `head -50 src/projections/features/te.py`
Expected: same shape as QB.

- [ ] **Step 2: Add the singleton + kwarg analogous to Task 10**

Edit `src/projections/features/te.py`. Same pattern as QB.

- [ ] **Step 3: Run TE tests + lint**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_te.py -v && .venv/Scripts/python.exe -m ruff check src/projections/features/te.py`
Expected: all TE tests PASS; ruff clean.

- [ ] **Step 4: Commit**

```bash
git add src/projections/features/te.py
git commit -m "feat(te): add draft_picks kwarg for plumbing symmetry (unused)"
```

---

### Task 13: Extend `_WR_FEATURE_COLUMNS` in `baseline.py`

**Files:**
- Modify: `src/projections/models/baseline.py:266-288`

- [ ] **Step 1: Re-read the current tuple**

Run: `sed -n '266,290p' src/projections/models/baseline.py`
Expected: shows the 21-element tuple ending with `"opp_allowed_wr_fppg_l4",`.

- [ ] **Step 2: Append 4 new entries with a comment**

Edit `src/projections/models/baseline.py`, replace:

```python
    "opp_allowed_wr_fppg_l4",
)
```

with:

```python
    "opp_allowed_wr_fppg_l4",
    # Trajectory features (PR #25 family probe + 2026-05-03 WR integration).
    # lightgbm derives feature lists from WrFeaturesSchema dynamically and
    # auto-picks-up; baseline.py is hardcoded so must be updated explicitly.
    "age",
    "is_rookie",
    "volume_trend_l4_minus_prior_l4",
    "snap_pct_change_l4_vs_prior_l4",
)
```

- [ ] **Step 3: Smoke-verify both baseline + lightgbm see the 4 new columns**

Run: `.venv/Scripts/python.exe -c "
from projections.models.baseline import _WR_FEATURE_COLUMNS as bl
from projections.models.lightgbm import _filter_features, _WR_FEATURE_COLUMNS as lg
TRAJ = ('age', 'is_rookie', 'volume_trend_l4_minus_prior_l4', 'snap_pct_change_l4_vs_prior_l4')
for c in TRAJ:
    assert c in bl, f'{c} missing from baseline _WR_FEATURE_COLUMNS'
    assert c in _filter_features(lg), f'{c} missing from lightgbm _filter_features(_WR_FEATURE_COLUMNS)'
print('both lists include the 4 trajectory cols')
"`
Expected: `both lists include the 4 trajectory cols`.

- [ ] **Step 4: Run baseline + lightgbm test files**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models/ -v 2>&1 | tail -20`
Expected: all PASS.

- [ ] **Step 5: Run the parametrized cross-position smoke test (catches signature regressions)**

Run: `.venv/Scripts/python.exe -m pytest -k "smoke" -v 2>&1 | tail -10`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/projections/models/baseline.py
git commit -m "feat(baseline): extend _WR_FEATURE_COLUMNS with 4 trajectory cols

Spec gap caught for RB at PR #21 commit 9895dee — baseline.py
hardcodes the per-position feature tuple while lightgbm derives from
the schema dynamically. WR mirror fix."
```

---

## Phase 4 — Caller-script plumbing

Goal: thread `draft_picks` through the 4 caller scripts so the production WR builder receives real draft data when invoked end-to-end. Each script: read `draft_picks` partition, pass to all 4 `build_*_features` calls. Verified by each script's existing CLI test.

### Task 14: Plumb `draft_picks` through `scripts/refresh_features.py`

**Files:**
- Modify: `scripts/refresh_features.py`

- [ ] **Step 1: Re-read the script structure**

Run: `wc -l scripts/refresh_features.py && grep -n "build_wr_features\|build_qb_features\|build_rb_features\|build_te_features\|read_partition\|pbp" scripts/refresh_features.py | head -30`
Expected: shows where pbp is loaded + threaded; locates the `build_*_features` call sites.

- [ ] **Step 2: Add `draft_picks` loading + pass to all 4 builders**

Locate the section where `pbp` is loaded (likely a `read_partition(raw_root, "pbp", ...)` or `_read_concat`-style block). Immediately after, add:

```python
try:
    draft_picks_frames = []
    for s in range(1980, max(seasons) + 1):
        try:
            draft_picks_frames.append(read_partition(raw_root, "draft_picks", season=s))
        except FileNotFoundError:
            continue
    draft_picks = (
        pd.concat(draft_picks_frames, ignore_index=True)
        if draft_picks_frames
        else pd.DataFrame()
    )
except Exception:
    draft_picks = pd.DataFrame()  # graceful degradation
```

(Match the script's existing seasons-iteration style; the snippet above is illustrative — adapt to whether `seasons` is a `range` or a list, and use the script's existing `_read_concat` helper if defined.)

In each `build_*_features(...)` call site, add `draft_picks=draft_picks` to the kwargs.

- [ ] **Step 3: Re-read your edits to confirm**

Run: `grep -n "draft_picks" scripts/refresh_features.py`
Expected: at least 1 load + 4 pass-throughs (one per position).

- [ ] **Step 4: Run the script's CLI tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scripts/ -k "refresh_features" -v`
Expected: all PASS. (If a CLI test fails because the synthetic test fixture doesn't have a `draft_picks` partition, the graceful-degradation `except FileNotFoundError` in step 2 handles it.)

- [ ] **Step 5: Commit**

```bash
git add scripts/refresh_features.py
git commit -m "feat(refresh): thread draft_picks through all 4 builders"
```

---

### Task 15: Plumb `draft_picks` through `scripts/train_baseline.py`

**Files:**
- Modify: `scripts/train_baseline.py`

- [ ] **Step 1: Re-read the script's `pbp` plumbing**

Run: `grep -n "pbp\|build_.*_features\|read_partition" scripts/train_baseline.py | head -20`
Expected: shows the `pbp` load + pass-through pattern.

- [ ] **Step 2: Mirror the pattern for `draft_picks`**

Same shape as Task 14 step 2 — load `draft_picks` partition for `range(1980, max_season + 1)` with try/except graceful degradation; pass to every `build_*_features` call.

- [ ] **Step 3: Run the script's CLI test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scripts/ -k "train_baseline" -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_baseline.py
git commit -m "feat(train-baseline): thread draft_picks through all 4 builders"
```

---

### Task 16: Plumb `draft_picks` through `scripts/predict_2024.py`

**Files:**
- Modify: `scripts/predict_2024.py`

- [ ] **Step 1: Re-read the script's `pbp` plumbing**

Run: `grep -n "pbp\|build_.*_features\|read_partition" scripts/predict_2024.py | head -20`
Expected: shows the pattern.

- [ ] **Step 2: Mirror the pattern for `draft_picks`**

Same shape as Task 14 / 15.

- [ ] **Step 3: Run the script's CLI test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scripts/ -k "predict_2024" -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/predict_2024.py
git commit -m "feat(predict): thread draft_picks through all 4 builders"
```

---

### Task 17: Plumb `draft_picks` through `scripts/sanity_check_baseline.py`

**Files:**
- Modify: `scripts/sanity_check_baseline.py`

- [ ] **Step 1: Re-read the script's `pbp` plumbing**

Run: `grep -n "pbp\|build_.*_features\|read_partition" scripts/sanity_check_baseline.py | head -20`
Expected: shows the pattern.

- [ ] **Step 2: Mirror the pattern for `draft_picks`**

Same shape as Task 14 / 15 / 16.

- [ ] **Step 3: Run the script's CLI test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scripts/ -k "sanity_check" -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/sanity_check_baseline.py
git commit -m "feat(sanity-check): thread draft_picks through all 4 builders"
```

---

### Task 18: End-of-Phase-4 verification — full suite + all gates

**Files:** none (verification only)

- [ ] **Step 1: Full pytest suite**

Run: `.venv/Scripts/python.exe -m pytest -v 2>&1 | tail -10`
Expected: all PASS.

- [ ] **Step 2: mypy strict**

Run: `.venv/Scripts/python.exe -m mypy src tests 2>&1 | tail -3`
Expected: `Success: no issues found in N source files`.

- [ ] **Step 3: ruff check**

Run: `.venv/Scripts/python.exe -m ruff check src tests scripts 2>&1 | tail -3`
Expected: `All checks passed!`.

- [ ] **Step 4: ruff format check**

Run: `.venv/Scripts/python.exe -m ruff format --check src tests scripts 2>&1 | tail -3`
Expected: zero formatting drift.

- [ ] **Step 5: Schema-touching integration smoke (per CLAUDE.md mechanical override #4)**

Run: `.venv/Scripts/python.exe -m pytest -v -k "ingest or store or schemas" 2>&1 | tail -10`
Expected: all PASS — confirms the schema delta hasn't introduced dtype regressions in the ingest/store seam.

If any step fails, fix and re-run before proceeding to Phase 5.

---

## Phase 5 — Real-data execution + reports

Goal: regenerate the WR feature cache against real `draft_picks` + `weekly_stats` + `snap_counts` data; run the walk-forward backtest; run the adoption gate with `--coverage-threshold 0.35` (structural sparsity precedent); write the summary report. The gate's `(BaselineModel, WR)` verdict determines Phase 6's "ship" or "revert" branch.

### Task 19: Verify `data/raw/draft_picks/` partition exists

**Files:** none (data verification only)

- [ ] **Step 1: List partition years**

Run: `ls data/raw/draft_picks/ 2>/dev/null | head -10`
Expected: `season=1980/`, `season=1981/`, ..., `season=2024/` (or similar). At minimum 1980-2024 needs coverage.

- [ ] **Step 2: If empty or missing, refresh draft_picks ingest first**

If step 1 returns nothing or only partial coverage:

Run: `.venv/Scripts/python.exe -m scripts.refresh_draft_picks --seasons 1980-2024`
Expected: writes ~45 partition files; no error.

(If `refresh_draft_picks` doesn't exist as a `__main__` entry point, invoke via `python -c "from projections.ingest.draft_picks import refresh_draft_picks; refresh_draft_picks(data_root=Path('data'), seasons=range(1980, 2025))"`.)

- [ ] **Step 3: Confirm**

Run: `ls data/raw/draft_picks/ | wc -l`
Expected: ≥45 (1980 through 2024 inclusive).

---

### Task 20: Refresh WR feature cache

**Files:** none (data regeneration only — output not committed)

- [ ] **Step 1: Refresh WR cache**

Run: `.venv/Scripts/python.exe scripts/refresh_features.py wr --seasons 2018-2024 2>&1 | tail -20`
Expected: each (season, week) partition validates against the extended `WrFeaturesSchema`; no SchemaError raised.

- [ ] **Step 2: Spot-check a partition**

Run: `.venv/Scripts/python.exe -c "
import pandas as pd
from projections.store import read_partition
from pathlib import Path
df = read_partition(Path('data/features'), 'wr', season=2024, week=5)
TRAJ = ['age', 'is_rookie', 'volume_trend_l4_minus_prior_l4', 'snap_pct_change_l4_vs_prior_l4']
for c in TRAJ:
    assert c in df.columns, f'{c} missing'
print('rows:', len(df))
print(df[TRAJ].describe())
"`
Expected: 4 trajectory cols present; non-NaN values for `age` (most rows); some NaN expected for `volume_trend_*` / `snap_pct_change_*` (early-season WRs without 8 prior active games).

- [ ] **Step 3: Coverage check across the 2021-2024 eval window**

Run: `.venv/Scripts/python.exe -c "
import pandas as pd
from pathlib import Path
from projections.store import read_partition
TRAJ = ['age', 'is_rookie', 'volume_trend_l4_minus_prior_l4', 'snap_pct_change_l4_vs_prior_l4']
frames = []
for s in range(2021, 2025):
    for w in range(1, 19):
        try:
            frames.append(read_partition(Path('data/features'), 'wr', season=s, week=w))
        except FileNotFoundError:
            continue
df = pd.concat(frames, ignore_index=True)
print(f'total rows: {len(df)}')
for c in TRAJ:
    cov = df[c].notna().mean()
    print(f'  {c}: {cov:.1%} coverage')
"`
Expected: `age` ~96-97%; `is_rookie` ~96-97%; `volume_trend_l4_minus_prior_l4` ~50-55%; `snap_pct_change_l4_vs_prior_l4` ~65-70% — should match the probe's measured coverage (PR #25's PM entry: "WR (96.7 / 53.6 / 68.4)"). If materially different (e.g., volume_trend is 5% or 95%), the builder wiring is wrong — investigate before proceeding. Save the printed coverage stats for inclusion in the summary report.

---

### Task 21: Capture pre-PR baseline run sha for the dual-run gate

**Files:** none (git inspection only)

- [ ] **Step 1: Identify the most recent backtest run on `main` pre-this-branch**

Run: `git log --all --oneline -- tests/backtest/model_metrics.json | head -5`
Expected: shows the most recent commits that touched the backtest snapshot. The baseline run sha is the most recent one on `main` (pre-`feat/wr-trajectory-features`).

- [ ] **Step 2: Note the pre-branch baseline sha for use in Task 23**

Capture the commit sha (e.g., `d045eb8` if main HEAD is unchanged) — this is the `--baseline-run` argument for `adoption_gate.py`.

---

### Task 22: Run the walk-forward backtest with the new WR features

**Files:**
- Modify: `tests/backtest/model_metrics.json` (snapshot update)

- [ ] **Step 1: Run backtest for WR, all 4 model classes**

Run: `.venv/Scripts/python.exe scripts/backtest.py --position WR --update-snapshot 2>&1 | tail -20`
Expected: walk-forward over 2021-2024 holdout years × 4 model classes; updates `tests/backtest/model_metrics.json`. Runtime: ~5-15 minutes depending on machine.

- [ ] **Step 2: Diff the snapshot to confirm only WR rows changed**

Run: `git diff tests/backtest/model_metrics.json | head -40`
Expected: changes confined to `position: "WR"` rows; QB/RB/TE rows unchanged (they didn't get new features).

- [ ] **Step 3: Commit the snapshot update**

```bash
git add tests/backtest/model_metrics.json
git commit -m "snapshot: backtest update for WR trajectory features

Walk-forward over 2021-2024, 4 model classes. WR rows reflect the
new trajectory feature inputs; QB/RB/TE unchanged."
```

---

### Task 23: Run the dual-run adoption gate

**Files:**
- Add: `reports/adoption_gate_wr_trajectory_features.md`
- Add: `reports/adoption_gate_wr_trajectory_features.csv`

- [ ] **Step 1: Capture the candidate (current branch) sha**

Run: `git rev-parse HEAD`
Expected: 40-char sha (the candidate run sha).

- [ ] **Step 2: Run the adoption gate with the structural-sparsity coverage threshold**

Run:
```bash
.venv/Scripts/python.exe scripts/adoption_gate.py \
  --position WR \
  --baseline-run <baseline-sha-from-Task-21> \
  --candidate-run <candidate-sha-from-step-1> \
  --coverage-threshold 0.35 \
  --output-md reports/adoption_gate_wr_trajectory_features.md \
  --output-csv reports/adoption_gate_wr_trajectory_features.csv
```
Expected: produces per-(model_class, WR) verdicts. **Must use `--coverage-threshold 0.35`** — without it, the pooled coverage check fails on `volume_trend_*` and `snap_pct_change_*` (~50% NaN by design).

- [ ] **Step 3: Read the gate's BaselineModel verdict**

Run: `grep -A 2 "baseline.*WR\|WR.*baseline" reports/adoption_gate_wr_trajectory_features.md | head -10`
Expected: a single verdict line — `ADOPT`, `MARGINAL`, or `DO_NOT_ADOPT` — with the composite RMSE delta + 95% CI. **This is the binding decision** (spec §1.3.5).

- [ ] **Step 4: Commit the gate reports**

```bash
git add reports/adoption_gate_wr_trajectory_features.md reports/adoption_gate_wr_trajectory_features.csv
git commit -m "report(wr-trajectory): adoption gate output

Dual-run gate, --coverage-threshold 0.35, all 4 model classes on WR
2021-2024 holdout. Binding cell: (BaselineModel, WR)."
```

---

### Task 24: Write the summary report

**Files:**
- Add: `reports/wr_trajectory_features_summary.md`

- [ ] **Step 1: Draft the summary report**

Create `reports/wr_trajectory_features_summary.md` with the following structure (fill in numbers from the gate output + Task 20 step 3 coverage stats):

```markdown
# WR Trajectory Features Integration — Summary Report

**Status:** [ADOPT / MARGINAL / DO_NOT_ADOPT] on (BaselineModel, WR)
**Branch:** `feat/wr-trajectory-features`
**Spec:** `docs/superpowers/specs/2026-05-03-wr-trajectory-features-design.md`
**Plan:** `docs/superpowers/plans/2026-05-03-wr-trajectory-features.md`
**Date:** 2026-05-XX

## Decision

[Single sentence — ship or revert, with the binding magnitude + CI.]

## Probe-vs-gate calibration

| Source | Composite RMSE Δ on (BaselineModel, WR) augment | 95% CI |
|---|---:|---|
| PR #25 probe (predicted) | -0.0414 | [-0.0606, -0.0230] |
| This PR's gate (measured) | [filled] | [filled] |

[2-3 sentences on whether probe and gate agree to 4 decimals like PR #21 did, or diverge — what direction, by how much, what it means.]

## Per-(model_class, WR) verdicts

| Model class | RMSE Δ | 95% CI | Verdict |
|---|---:|---|:---:|
| baseline | [filled] | [filled] | [filled] |
| lightgbm-tuned | [filled] | [filled] | [filled] |
| lightgbm-nb | [filled] | [filled] | [filled] |
| ensemble | [filled] | [filled] | [filled] |

[Cross-check: the lgb-nb cell is the cross-check on the probe's second WR ADOPT cell at -0.0194 fpts.]

## Coverage statistics (2021-2024 eval window)

| Column | Coverage | Probe coverage (PR #25) | Match? |
|---|---:|---:|:---:|
| age | [from Task 20 step 3] | 96.7% | [yes/no] |
| is_rookie | [from Task 20 step 3] | 96.7% | [yes/no] |
| volume_trend_l4_minus_prior_l4 | [from Task 20 step 3] | 53.6% | [yes/no] |
| snap_pct_change_l4_vs_prior_l4 | [from Task 20 step 3] | 68.4% | [yes/no] |

[If coverage matches the probe within ~1pp, the builder wiring is correct. If divergent, document the cause.]

## Threshold relaxation rationale

`--coverage-threshold 0.35` used (spec §1.3.3). Trajectory's trend features are structurally sparse — they require 8 prior active games per player, which excludes ~50% of player-weeks across all years. Same precedent as PR #25's probe (which used the same threshold).

## What this closes

[If ADOPT:] TODO #24's "trailing-8-game unit" branch — the bundled trajectory probe carried clear signal and integrates production. Refined-unit candidates (`age²`, `is_2nd_year` flags, longer windows, `has_trajectory_history` indicator) remain unexplored under the same TODO.

[If DO_NOT_ADOPT:] The probe-vs-gate divergence is documented; the spec gap (if any — e.g., builder NaN handling differs from probe override) is captured for future probe-tuning.

## Next track

[Either: TE trajectory integration (own spec, must address per-position-routing question — TE only ADOPT'd under lgb-nb), or: refined-unit candidates re-test, or: pivot to other Track 2 candidates (TODO #25 weather features).]
```

- [ ] **Step 2: Fill in numbers from `reports/adoption_gate_wr_trajectory_features.md` and Task 20 step 3 output**

Open the gate report; copy the per-cell numbers into the summary table.

- [ ] **Step 3: Commit**

```bash
git add reports/wr_trajectory_features_summary.md
git commit -m "report(wr-trajectory): family summary — verdict [ADOPT/...]"
```

---

## Phase 6 — Documentation + decision-log update (conditional on Phase 5 verdict)

Goal: append the decision-log entry to `project_management.md` and update `TODO.md` #24. **Branch follows the gate verdict from Task 23.**

### Task 25 — IF `(BaselineModel, WR)` verdict is `ADOPT`

**Files:**
- Modify: `project_management.md` (top of file)
- Modify: `TODO.md` #24

- [ ] **Step 1: Append the top-of-file decision-log entry to `project_management.md`**

Format matches PR #21's entry. Insert immediately after the line `---` at line 5 (or before the existing top-most entry):

```markdown
## WR Trajectory Features Integration — verdict ADOPT on (BaselineModel, WR); shipped (2026-05-XX, on branch `feat/wr-trajectory-features`)

**Status:** Production integration of the 4 trajectory features into `WrFeaturesSchema` + `build_wr_features` per `docs/superpowers/specs/2026-05-03-wr-trajectory-features-design.md`. Promoted `_build_draft_lookup` from override-script-private to public `build_draft_lookup` in `src/projections/features/trajectory_features.py`. Wired `attach_trajectory_features` into `build_wr_features` with the new `draft_picks` kwarg; added the same kwarg to QB/RB/TE builders for plumbing symmetry (unused there, mirroring the existing `pbp` precedent). Updated `baseline.py:_WR_FEATURE_COLUMNS` (the spec gap PR #21 caught at `9895dee`). All caller scripts (`refresh_features.py`, `train_baseline.py`, `predict_2024.py`, `sanity_check_baseline.py`) load + thread `draft_picks`.

**Dual-run gate verdict on `(BaselineModel, WR)`:** `ADOPT` (composite RMSE delta [filled] fpts, CI [filled]). Probe predicted -0.0414 fpts; gate matched the point estimate to [N] decimal places.

**Coverage relaxation:** `--coverage-threshold 0.35` (matches PR #25's probe precedent for structurally-sparse trajectory features).

**Other model classes** (informational per spec §1.3.5):
- lgb-nb: [filled] — cross-check on the probe's second WR ADOPT cell at -0.0194 fpts.
- lightgbm-tuned: [filled].
- ensemble: [filled].

**What this closes:** TODO #24's trailing-8-game-unit branch of the trajectory candidate. Refined-unit candidates (`age²`, `is_2nd_year` flags, longer trailing windows, `has_trajectory_history` indicator) remain unexplored under the same TODO.

**Spec gaps caught + fixed:** [list any — at minimum the `_WR_FEATURE_COLUMNS` hardcoded list update was anticipated by the spec; if anything else surfaced during execution, document it here].

**TE follow-up status:** Per the trajectory probe, TE adopted only under lgb-nb (not baseline). Per-position-routing decision required for any TE integration (precedent: Plan 6's QB-only ensemble suggestion). Not queued.

See `reports/wr_trajectory_features_summary.md` for the full decision log + per-mode table + probe-vs-gate calibration.

---
```

- [ ] **Step 2: Update `TODO.md` #24**

Open `TODO.md`, find the section header `### 24. Player-trajectory features (age curves, career arc, trend gradients)`. Append at the end of that section:

```markdown
**Update 2026-05-XX (WR trajectory features integration, branch `feat/wr-trajectory-features`):** Production integration of the 4 trajectory features into `WrFeaturesSchema` + `build_wr_features` per `docs/superpowers/specs/2026-05-03-wr-trajectory-features-design.md`. Dual-run gate verdict on `(BaselineModel, WR)`: **`ADOPT`** (composite RMSE delta [filled] fpts, CI [filled]). Probe predicted -0.0414 fpts; gate matched [calibration commentary]. Shipped. Other model classes informational per spec §1.3.5. **TE remains open for a separate refined-unit spec — must address per-position routing (TE only ADOPT'd under lgb-nb, not baseline).** Refined-unit candidates beyond trailing-8-game unit remain unexplored: per-position aging-curve interactions (`age²`), `is_2nd_year` / `is_3rd_year` flags, depth-chart-rank trends, longer trailing windows (l8 vs l16), `has_trajectory_history` indicator. None queued. See `reports/wr_trajectory_features_summary.md`.
```

- [ ] **Step 3: Commit**

```bash
git add project_management.md TODO.md
git commit -m "docs(pm): record WR trajectory features integration ADOPT verdict"
```

- [ ] **Step 4: Push branch + open PR**

```bash
git push -u origin feat/wr-trajectory-features
gh pr create --title "feat(wr): trajectory features integration — ADOPT" --body "$(cat <<'EOF'
## Summary

Promotes the trajectory family probe (PR #25) into the production WR feature pipeline. Adds 4 nullable-float trajectory columns to `WrFeaturesSchema`; wires `attach_trajectory_features` into `build_wr_features`. Dual-run gate verdict on `(BaselineModel, WR)` is `ADOPT` ([filled] fpts, CI [filled]).

- Probe predicted -0.0414 fpts; gate measured [filled].
- Coverage threshold: 0.35 (structural-sparsity precedent matching PR #25).
- Other model classes: lgb-nb [filled]; informational per spec §1.3.5.
- Spec gap from PR #21 explicit: `baseline.py:_WR_FEATURE_COLUMNS` updated; lightgbm derives dynamically from schema and auto-picks-up.

## Test plan

- [x] Full pytest suite passes
- [x] mypy strict zero violations
- [x] ruff check + format clean
- [x] WR feature cache regenerates against extended schema
- [x] Backtest snapshot diff confined to WR rows
- [x] Adoption gate `--coverage-threshold 0.35` runs on all 4 model classes

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

### Task 26 — IF `(BaselineModel, WR)` verdict is `MARGINAL` or `DO_NOT_ADOPT`

**Files:**
- Modify: `src/projections/schemas.py` (revert 4 columns)
- Modify: `src/projections/features/wr.py` (revert builder integration; keep `_EMPTY_DRAFT_PICKS` singleton + kwarg, since QB/RB/TE also gained it)
- Modify: `src/projections/models/baseline.py` (revert `_WR_FEATURE_COLUMNS` extension)
- Modify: `tests/test_features/test_wr.py` (revert 4 new tests + happy-path extension; keep `wr_draft_picks` fixture for TE follow-up)
- Modify: `project_management.md` + `TODO.md`

**Note:** The `build_draft_lookup` promotion (Phase 1 Task 2) and the QB/RB/TE kwarg additions (Phase 3 Tasks 10-12) and the caller-script plumbing (Phase 4) all stay — they're useful infrastructure for the TE follow-up regardless of WR's outcome.

- [ ] **Step 1: Revert WR-specific changes**

Use `git revert <commit-sha>` for each commit that touched: `WrFeaturesSchema` (Task 4), `build_wr_features` integration (Task 5), the 4 join-side tests (Tasks 6-9), the `_WR_FEATURE_COLUMNS` extension (Task 13). Resolve any conflicts from later changes.

- [ ] **Step 2: Re-run full test suite + lint to confirm clean revert**

Run: `.venv/Scripts/python.exe -m pytest -v 2>&1 | tail -10 && .venv/Scripts/python.exe -m mypy src tests && .venv/Scripts/python.exe -m ruff check src tests scripts`
Expected: all PASS, zero violations.

- [ ] **Step 3: Append decision-log entry to `project_management.md` documenting the divergence**

Same shape as Task 25 step 1, but with a `verdict DO_NOT_ADOPT (or MARGINAL)` header and a "What this closes" note pointing to the probe-vs-gate divergence.

- [ ] **Step 4: Update `TODO.md` #24** — same shape as Task 25 step 2 with `DO_NOT_ADOPT` framing.

- [ ] **Step 5: Commit + push + open documentation-only PR**

```bash
git add project_management.md TODO.md
git commit -m "docs(pm): record WR trajectory features [MARGINAL/DO_NOT_ADOPT] verdict + revert"
git push -u origin feat/wr-trajectory-features
gh pr create --title "spec/report(wr-trajectory): family verdict [MARGINAL/DO_NOT_ADOPT]" --body "$(cat <<'EOF'
## Summary

Trajectory family integration probed and gated; gate verdict on `(BaselineModel, WR)` was [MARGINAL/DO_NOT_ADOPT]. Schema + builder changes reverted per spec §1.3.5; `build_draft_lookup` promotion + QB/RB/TE kwarg plumbing kept (infrastructure for TE follow-up).

- Probe predicted -0.0414 fpts; gate measured [filled].
- See `reports/wr_trajectory_features_summary.md` for the full divergence analysis.

## Test plan

- [x] Full pytest suite passes (post-revert)
- [x] mypy strict zero violations
- [x] ruff check + format clean

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review checklist (run before handoff)

Quick scan after Phase 6 completes:

- [ ] All 4 trajectory columns appear in `WrFeaturesSchema` (or are reverted on DO_NOT_ADOPT path).
- [ ] `build_draft_lookup` is publicly exported from `trajectory_features.py` regardless of the gate verdict.
- [ ] All 4 caller scripts load + thread `draft_picks`.
- [ ] `baseline.py:_WR_FEATURE_COLUMNS` includes the 4 new names (or is reverted).
- [ ] Adoption gate report exists at `reports/adoption_gate_wr_trajectory_features.{md,csv}`.
- [ ] Summary report exists at `reports/wr_trajectory_features_summary.md`.
- [ ] PM + TODO #24 reflect the verdict.
- [ ] Coverage check from Task 20 step 3 was within ~1pp of the probe's measured coverage (not, by itself, a blocker — but a divergence flagged in the summary).

If all green, the spec's success criteria (§1.3) are satisfied.
