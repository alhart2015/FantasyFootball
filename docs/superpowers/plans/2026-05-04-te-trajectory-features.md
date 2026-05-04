# TE Trajectory Features Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote PR #25's trajectory family probe TE-cell signal into the production TE feature pipeline. Append the same 4 nullable-float trajectory columns to `TeFeaturesSchema` that PR #26 added to `WrFeaturesSchema`, wire `attach_trajectory_features` into `build_te_features`, run the dual-run adoption gate, ship if `(LightGBMNbModel, TE)` returns `ADOPT` (probe predicted -0.0107 fpts).

**Architecture:** Schema delta + builder integration only — all the heavy plumbing (public `build_draft_lookup`, `attach_trajectory_features` helper, `Position.TE` branch in helper, `draft_picks` kwarg on `build_te_features`, all 4 caller scripts loading `draft_picks`) already shipped in PR #25 + PR #26. **First integration spec in the project to bind on a non-default model class** (lgb-nb, not baseline) — TE production routing remains on `baseline` per Plan 8 (cross-class flip is a deferred follow-up). Spec §1.3.5 pre-decides a 3-way ship outcome (ship-as-designed / ship-modified-shape / revert) based on the gate verdict on `(lgb-nb, TE)` AND `(baseline, TE)`.

**Tech Stack:** Python 3.11, pandas (pyarrow-backed strings + nullable Int64/Float64), pandera (DataFrameModel + `strict="filter"`), pytest, scikit-learn (`RidgeCV` via `BaselineModel`), LightGBM (Quantile / NB-2 sub-models), pre-commit hooks (ruff, mypy strict, pre-commit hygiene).

**Spec:** `docs/superpowers/specs/2026-05-04-te-trajectory-features-design.md`.

---

## File Structure (decomposition lock-in)

**Modify:**
- `src/projections/schemas.py` — extend `TeFeaturesSchema` with 4 nullable-float trajectory columns (Phase 1).
- `src/projections/features/te.py` — wire `attach_trajectory_features` into `build_te_features` using the existing `draft_picks` kwarg (Phase 1).
- `src/projections/models/baseline.py` — extend `_TE_FEATURE_COLUMNS` tuple with 4 new column names (Phase 3, conditionally reverted in Phase 5 if the modified-shape ship path fires).
- `tests/conftest.py` — extend the shared `_build_position_weekly_stats` (or analogous helper) so `baseline_weekly_stats_te` covers 17 weeks of 2023 + 17 weeks of 2024 + 4 weeks of 2025 (Phase 2). Trajectory trends require 8+ prior active games per player; 2024-only fixtures yield NaN trends and `BaselineModel.fit`'s dropna empties the training set.
- `tests/test_features/conftest.py` — gain `te_draft_picks` fixture (Phase 2).
- `tests/test_features/test_te.py` — extend happy-path; 4 new trajectory join-side tests (Phase 2).
- `tests/test_features/test_cache.py`, `tests/test_models/test_lightgbm_te.py`, `tests/test_models/test_baseline_te.py`, and any other site discovered via the `opp_allowed_te_fppg_l4` defensive grep — special-case `age` / `is_rookie` / trend cols on synthetic TE feature rows (Phase 3).
- `project_management.md`, `TODO.md` — decision-log + close TODO #24's TE branch (Phase 5, conditional on gate verdict).

**Add (reports, Phase 4 — committed but not code):**
- `reports/adoption_gate_te_trajectory_features.{md,csv}` — gate output across all 5 model classes.
- `reports/te_trajectory_features_summary.md` — decision log + per-mode table + probe-vs-gate calibration + coverage stats + binding-cell-shift rationale + cross-class deferred-follow-up note.

**Already in place (no changes needed):**
- `src/projections/features/trajectory_features.py` — public `build_draft_lookup`, public `attach_trajectory_features` with `Position.TE` branch (PR #25 + PR #26).
- `src/projections/features/te.py` signature — `draft_picks: pd.DataFrame = _EMPTY_DRAFT_PICKS` kwarg already plumbed (PR #26 commit `36313d9`-style addition).
- `scripts/refresh_features.py`, `scripts/train_baseline.py`, `scripts/predict_2024.py`, `scripts/sanity_check_baseline.py` — all load + thread `draft_picks` to all 4 builders (PR #26).
- `src/projections/ingest/refresh_draft_picks.py` + `DraftPicksSchema` (PR #25).

---

## Phase 1 — Schema + builder integration

Goal: ship the `TeFeaturesSchema` column delta and wire `attach_trajectory_features` into `build_te_features`. After this phase: existing TE feature builder tests will FAIL (fixtures don't yet have trajectory cols, and `baseline_weekly_stats_te` doesn't have the 8-game prior history needed for non-NaN trends to fully exercise the path) — Phase 2 closes them.

### Task 1: Add 4 trajectory columns to `TeFeaturesSchema`

**Files:**
- Modify: `src/projections/schemas.py:658-700`

- [ ] **Step 1: Read the current `TeFeaturesSchema` definition**

Run: `sed -n '658,705p' src/projections/schemas.py`
Expected: shows the schema ending with `opp_allowed_te_fppg_l4: Series[float] = pa.Field(ge=0, nullable=True)` then the `Config` class.

- [ ] **Step 2: Append 4 columns + comment block before the `Config` class**

Edit `src/projections/schemas.py`. Find the line:
```python
    # Opponent strength (proxy: opp's allowed TE fantasy points/game over trailing 4)
    opp_allowed_te_fppg_l4: Series[float] = pa.Field(ge=0, nullable=True)

    class Config:
```

Replace with:
```python
    # Opponent strength (proxy: opp's allowed TE fantasy points/game over trailing 4)
    opp_allowed_te_fppg_l4: Series[float] = pa.Field(ge=0, nullable=True)

    # Trajectory features (PR #25 family probe + 2026-05-04 TE integration
    # spec). All four are structurally sparse: age + is_rookie need a
    # draft_picks lookup hit (or the inferred fallback) and so cover ~95%
    # of TE player-weeks per the probe; the trend cols need 8 prior active
    # games and so cover ~45-71% of TE player-weeks. NaN where coverage
    # is missing; BaselineModel imputes with feature mean, lightgbm
    # consumes NaN natively.
    age: Series[float] = pa.Field(ge=15, le=50, nullable=True)
    is_rookie: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    volume_trend_l4_minus_prior_l4: Series[float] = pa.Field(nullable=True)
    snap_pct_change_l4_vs_prior_l4: Series[float] = pa.Field(ge=-1, le=1, nullable=True)

    class Config:
```

(Note the column shape exactly mirrors PR #26's `WrFeaturesSchema` extension. Use the same field types, bounds, comment style.)

- [ ] **Step 3: Confirm schema parses + new column count**

Run: `.venv/Scripts/python.exe -c "from projections.schemas import TeFeaturesSchema; cols = list(TeFeaturesSchema.to_schema().columns.keys()); print(len(cols), 'columns'); TRAJ = ['age', 'is_rookie', 'volume_trend_l4_minus_prior_l4', 'snap_pct_change_l4_vs_prior_l4']; [print(' ', c, 'OK' if c in cols else 'MISSING') for c in TRAJ]"`
Expected: prints column count (was 21, now 25) and `OK` for each of the 4 trajectory cols.

- [ ] **Step 4: Run schema tests to confirm no regression**

Run: `.venv/Scripts/python.exe -m pytest tests/test_schemas/ -v`
Expected: all PASS (existing schema tests don't enforce TE column count, just shape).

- [ ] **Step 5: Run existing TE builder tests — they should now FAIL because output frame is missing the 4 new columns**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_te.py -v 2>&1 | tail -20`
Expected: FAIL on `test_build_te_features_returns_validated_frame` with a SchemaError mentioning one of `age`, `is_rookie`, `volume_trend_l4_minus_prior_l4`, or `snap_pct_change_l4_vs_prior_l4` is required but missing. **This is expected** — Task 2 wires the builder to produce them.

- [ ] **Step 6: Commit (with the TE test failures unfixed — Task 2 closes them)**

```bash
git add src/projections/schemas.py
git commit -m "schema(te): add 4 trajectory columns to TeFeaturesSchema

Schema-only change. TE builder tests will fail until Task 2 wires
attach_trajectory_features into build_te_features. Mirrors PR #26's
WrFeaturesSchema extension."
```

---

### Task 2: Wire `attach_trajectory_features` into `build_te_features`

**Files:**
- Modify: `src/projections/features/te.py`

- [ ] **Step 1: Re-read `te.py` to confirm current state**

Run: `head -30 src/projections/features/te.py && echo "---" && tail -40 src/projections/features/te.py`
Expected: imports include `Position`, `Ruleset`, `Stat`, `TeFeaturesSchema`; both `_EMPTY_PBP` and `_EMPTY_DRAFT_PICKS` already defined; `build_te_features` signature already has `draft_picks: pd.DataFrame = _EMPTY_DRAFT_PICKS`; the assembly section ends with `return TeFeaturesSchema.validate(out)`.

- [ ] **Step 2: Add trajectory_features imports**

Edit `src/projections/features/te.py`. Find the import block ending with the `projections.schemas` import and replace:

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
    TeFeaturesSchema,
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
    TeFeaturesSchema,
)
```

- [ ] **Step 3: Wire trajectory features into `build_te_features` body**

Edit `src/projections/features/te.py`. Find the line `return TeFeaturesSchema.validate(out)` (currently the last line of the function). Insert the trajectory wiring immediately before it:

```python
    # --- Trajectory features (PR #25 + 2026-05-04 TE integration) ---------
    # Helper does its own position filter (WR + TE share compute_wr_te_volume_trend)
    # + rolling, so pass the unfiltered (but prior-mask-filtered) weekly_stats /
    # snap_counts. PR #26's spec gap fix at commit d1b3092: the helper does its
    # own .shift(1) / .shift(5) leakage shifting; an external prior-mask-filter
    # would double-shift to 100% NaN.
    draft_lookup = build_draft_lookup(draft_picks)
    traj_idx = out[["gsis_id", "season", "week", "team", "opponent"]].rename(
        columns={"opponent": "opp"}
    )
    traj = attach_trajectory_features(traj_idx, ws, sc, draft_lookup, Position.TE)
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

    return TeFeaturesSchema.validate(out)
```

- [ ] **Step 4: Run the TE happy-path test to confirm trajectory wiring closes the schema gap**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_te.py::test_build_te_features_returns_validated_frame -v`
Expected: PASS — the 4 new schema columns are now produced. They'll be NaN for the synthetic TE fixture (no `draft_picks` passed → all rows route to inferred fallback for age, and the 8-game prior window isn't satisfied so trends are NaN), but `nullable=True` on all 4 fields accepts NaN.

- [ ] **Step 5: Run the full TE builder test files**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_te.py tests/test_features/test_te_leakage.py -v 2>&1 | tail -10`
Expected: all PASS — trajectory cols are NaN throughout (no draft_picks fixture), schema accepts NaN.

- [ ] **Step 6: Commit**

```bash
git add src/projections/features/te.py
git commit -m "feat(te): wire attach_trajectory_features into build_te_features

Consumes the existing draft_picks kwarg (plumbed in PR #26) and
computes the 4 trajectory features via the existing
attach_trajectory_features helper. Existing TE tests pass with all-NaN
trajectory cols (no draft_picks fixture); 4 new join-side tests come
in Phase 2."
```

---

## Phase 2 — Test fixtures + new TE tests

Goal: extend `baseline_weekly_stats_te` to cover 17 weeks of 2023 + 17 weeks of 2024 + 4 weeks of 2025 so the 8-game prior window is satisfied for trajectory trends; add the `te_draft_picks` fixture; add 4 new TE tests covering trajectory join-side correctness (drafted veteran, rookie, UDFA fallback, empty draft_picks).

### Task 3: Add `te_draft_picks` fixture

**Files:**
- Modify: `tests/test_features/conftest.py`

- [ ] **Step 1: Read the existing TE fixture gsis_ids**

Run: `grep -n '"00-00305\|"00-00330\|def te_' tests/test_features/conftest.py | head -10`
Expected: confirms TE fixtures use `00-0030506` (Kelce, KC) and `00-0033084` (synthetic TE, MIN). Note the wr_draft_picks fixture for shape reference.

- [ ] **Step 2: Append `te_draft_picks` fixture at end of file**

Add at end of `tests/test_features/conftest.py`:

```python
@pytest.fixture
def te_draft_picks() -> pd.DataFrame:
    """Synthetic draft_picks for the TEs used across the TE fixture set.

    Includes the two canonical TE gsis_ids from te_weekly_stats /
    te_depth_charts plus a NaN-draft_age branch and a rookie. The
    rookie 00-0099778 is intentionally drafted in 2024 (the test
    season), so is_rookie computes to 1.0 from the primary
    draft_lookup path. The other three are veteran drafts. Mirrors
    PR #26's wr_draft_picks fixture shape.
    """
    return pd.DataFrame(
        [
            # Travis Kelce: 2013 draft, 23yo at draft.
            {
                "gsis_id": "00-0030506",
                "draft_year": 2013,
                "draft_round": 3,
                "draft_overall_pick": 63,
                "pfr_id": "KelcTr00",
                "draft_age": 23.0,
            },
            # Synthetic TE 00-0033084: 2017 draft, 24yo at draft.
            {
                "gsis_id": "00-0033084",
                "draft_year": 2017,
                "draft_round": 1,
                "draft_overall_pick": 19,
                "pfr_id": "SyntTe00",
                "draft_age": 24.0,
            },
            # Synthetic veteran TE with NaN draft_age branch (drafted but missing age).
            {
                "gsis_id": "00-0035555",
                "draft_year": 2019,
                "draft_round": 4,
                "draft_overall_pick": 110,
                "pfr_id": "NaNTe000",
                "draft_age": float("nan"),
            },
            # Rookie TE 00-0099778: 2024 draft.
            {
                "gsis_id": "00-0099778",
                "draft_year": 2024,
                "draft_round": 2,
                "draft_overall_pick": 51,
                "pfr_id": "RookTE00",
                "draft_age": 22.0,
            },
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

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_te.py -k "returns_validated_frame" -v`
Expected: PASS (existing TE fixture set + the new unused te_draft_picks fixture don't conflict).

- [ ] **Step 4: Commit**

```bash
git add tests/test_features/conftest.py
git commit -m "test(te): add te_draft_picks synthetic fixture"
```

---

### Task 4: Extend `baseline_weekly_stats_te` to cover trajectory's 8-game prior window

**Files:**
- Modify: `tests/conftest.py:757-765` (and the shared `_build_position_weekly_stats` helper if needed)

- [ ] **Step 1: Re-read the current `baseline_weekly_stats_te` fixture and the `_build_position_weekly_stats` helper**

Run: `sed -n '750,800p' tests/conftest.py`
Expected: shows the fixture wraps `_build_position_weekly_stats("TE")`. Find the `_build_position_weekly_stats` definition (likely above line 750) — it currently constructs 8 weeks of 2024 + 4 weeks of 2025. Note the `_GSIS_IDS` and `_TEAMS` module-level constants used.

- [ ] **Step 2: Read PR #26's `baseline_weekly_stats_wr` extension for shape reference**

Run: `git log --all -p -- tests/conftest.py | grep -A 50 "17 weeks of 2023" | head -80`
Expected: shows the WR extension pattern — `for season, weeks in [(2023, range(1, 18)), (2024, range(1, 18)), (2025, range(1, 5))]:` replacing the original `[(2024, range(1, 9)), (2025, range(1, 5))]:`.

- [ ] **Step 3: Extend `_build_position_weekly_stats` to accept a `season_weeks` parameter**

Edit `tests/conftest.py`. Find the `_build_position_weekly_stats` function definition. Add a parameter with the existing 8/4 default:

```python
def _build_position_weekly_stats(
    position: str,
    *,
    season_weeks: list[tuple[int, range]] | None = None,
) -> pd.DataFrame:
    """... existing docstring ..."""
    if season_weeks is None:
        season_weeks = [(2024, range(1, 9)), (2025, range(1, 5))]
    # ... existing body, replacing the hardcoded `for season, weeks in [(2024, range(1, 9)), (2025, range(1, 5))]:` with `for season, weeks in season_weeks:`
```

(QB and RB fixtures keep the default, so no other call site needs updating.)

- [ ] **Step 4: Update `baseline_weekly_stats_te` to pass the trajectory-friendly window**

Replace:

```python
@pytest.fixture
def baseline_weekly_stats_te() -> pd.DataFrame:
    """8 weeks of 2024 + 4 weeks of 2025 TE-shaped stats for 5 synthetic TEs.

    The TE whose gsis_id ends in "3" rushes (Taysom-Hill-shape) so the new
    TeFeaturesSchema rushing columns from Phase 1 carry non-zero rolling
    means.
    """
    return _build_position_weekly_stats("TE")
```

with:

```python
@pytest.fixture
def baseline_weekly_stats_te() -> pd.DataFrame:
    """17 weeks of 2023 + 17 weeks of 2024 + 4 weeks of 2025 TE-shaped stats
    for 5 synthetic TEs.

    The TE whose gsis_id ends in "3" rushes (Taysom-Hill-shape) so the
    TeFeaturesSchema rushing columns from Plan 3b carry non-zero rolling
    means.

    Trajectory-features note (2026-05-04 TE integration): the trailing-4
    minus prior-4 trends require 8+ active games of history per player.
    Including 2023 weeks 1-17 ensures every 2024 row has a full 8-game
    prior window, so the trend cols are non-NaN even on early 2024
    training rows. Without 2023, every (l4 - prior_l4) row would be NaN
    and BaselineModel.fit's dropna would empty the TE training set.
    Mirrors PR #26's baseline_weekly_stats_wr extension.
    """
    return _build_position_weekly_stats(
        "TE",
        season_weeks=[
            (2023, range(1, 18)),
            (2024, range(1, 18)),
            (2025, range(1, 5)),
        ],
    )
```

- [ ] **Step 5: Inspect `baseline_features_te` (line ~1011) to ensure its supporting frames also cover the extended seasons**

Run: `sed -n '1005,1080p' tests/conftest.py`
Expected: shows `baseline_features_te` calls `_build_position_supporting_frames(baseline_weekly_stats_te, "TE")` then iterates over (season, week) tuples to call `build_te_features`. The supporting-frames helper iterates over `weekly_stats.iterrows()`, so it already adapts to the longer history. The (season, week) iteration in `baseline_features_te` may have a hardcoded `[(2024, range(1, 9)), (2025, range(1, 5))]` — find and update it to match the extended fixture if so. (Reference: PR #26's `baseline_features_wr` was extended in the same PR.)

If `baseline_features_te` has a hardcoded season-weeks loop, replace it with:

```python
    for season, weeks in [(2024, range(1, 18)), (2025, range(1, 5))]:
```

(2023 isn't included in the iteration because the fixture only computes features for the eval window — but the underlying weekly_stats covers 2023 so trailing-window rolls have full history.)

Similarly check `_build_position_supporting_frames` (line ~767) — its `for season in (2024, 2025): weeks = range(1, 9) if season == 2024 else range(1, 5)` block needs the same extension. **Conditional:** if extending `_build_position_supporting_frames` would unintentionally extend QB/RB depth charts / NGS / schedules too (creating asymmetry between QB/RB and TE fixtures), branch on position OR pass an `extra_season_weeks` parameter that defaults to empty.

The simplest path: add a `season_weeks_for_supporting_frames` param to `_build_position_supporting_frames` defaulted to the original `[(2024, range(1, 9)), (2025, range(1, 5))]`; TE calls it with `[(2023, range(1, 18)), (2024, range(1, 18)), (2025, range(1, 5))]`.

- [ ] **Step 6: Run the baseline TE model tests to confirm fixture extension doesn't break anything**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models/test_baseline_te.py tests/test_models/test_baseline_te_leakage.py -v 2>&1 | tail -15`
Expected: all PASS. Some tests may need fixture adjustments — if any fail, read the error and check whether the test was assuming a fixed fixture row count; widen the assertion if so.

- [ ] **Step 7: Commit**

```bash
git add tests/conftest.py
git commit -m "test(te): extend baseline_weekly_stats_te to 17/17/4 weeks for trajectory

Trajectory's trailing-4 minus prior-4 trends need 8+ active games per
player. Without 2023 history, all 2024 rows have NaN trends and
BaselineModel.fit's dropna empties the training set. Mirrors PR #26's
baseline_weekly_stats_wr extension."
```

---

### Task 5: Test — drafted veteran has computed `age` and `is_rookie=0.0`

**Files:**
- Modify: `tests/test_features/test_te.py`

- [ ] **Step 1: Append the new test, importing the `te_draft_picks` fixture from conftest**

Add at end of `tests/test_features/test_te.py`:

```python
def test_build_te_features_attach_trajectory_drafted_veteran(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
    te_draft_picks: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Travis Kelce: drafted 2013 at age 23; in 2024 should be age 34, is_rookie=0."""
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        pbp=fake_pbp_df,
        draft_picks=te_draft_picks,
        season=2024,
        as_of_week=5,
    )
    kelce = out[out["gsis_id"] == "00-0030506"].iloc[0]
    assert kelce["age"] == 34.0  # 23 + (2024 - 2013)
    assert kelce["is_rookie"] == 0.0
```

- [ ] **Step 2: Run the new test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_te.py::test_build_te_features_attach_trajectory_drafted_veteran -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_features/test_te.py
git commit -m "test(te): drafted-veteran age + is_rookie computed from draft_picks"
```

---

### Task 6: Test — rookie has `is_rookie=1.0` and `age == draft_age`

**Files:**
- Modify: `tests/test_features/test_te.py`

- [ ] **Step 1: Append the rookie test**

Add to `tests/test_features/test_te.py`:

```python
def test_build_te_features_attach_trajectory_rookie(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
    te_draft_picks: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Inject rookie TE 00-0099778 (drafted 2024 in fixture); is_rookie=1.0,
    age=22.0, no prior 8 games so volume_trend / snap_pct_change are NaN."""
    extra_dc = pd.concat(
        [
            te_depth_charts,
            pd.DataFrame(
                [
                    {
                        "gsis_id": "00-0099778",
                        "season": 2024,
                        "week": 5,
                        "team": "KC",
                        "position": "TE",
                        "depth_team": "TE2",
                        "depth_rank": 2,
                    }
                ]
            ).astype({"gsis_id": "string[pyarrow]"}),
        ],
        ignore_index=True,
    )

    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=extra_dc,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        pbp=fake_pbp_df,
        draft_picks=te_draft_picks,
        season=2024,
        as_of_week=5,
    )
    rookie = out[out["gsis_id"] == "00-0099778"].iloc[0]
    assert rookie["is_rookie"] == 1.0
    assert rookie["age"] == 22.0
    assert pd.isna(rookie["volume_trend_l4_minus_prior_l4"])
    assert pd.isna(rookie["snap_pct_change_l4_vs_prior_l4"])
```

- [ ] **Step 2: Run the new test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_te.py::test_build_te_features_attach_trajectory_rookie -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_features/test_te.py
git commit -m "test(te): rookie age + is_rookie + NaN trends from primary draft path"
```

---

### Task 7: Test — UDFA (no draft_lookup entry) uses inferred-draft-year fallback

**Files:**
- Modify: `tests/test_features/test_te.py`

- [ ] **Step 1: Append the UDFA test**

Add to `tests/test_features/test_te.py`:

```python
def test_build_te_features_attach_trajectory_udfa_fallback(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """UDFA TE (not in draft_picks fixture) falls back to inferred draft year
    = earliest weekly_stats appearance. Kelce appears in fixture from week 1
    of 2024, so inferred_year = 2024 -> is_rookie = 1.0, age = 2024 - 2024 +
    22.0 = 22.0."""
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
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        pbp=fake_pbp_df,
        draft_picks=udfa_picks,
        season=2024,
        as_of_week=5,
    )
    kelce = out[out["gsis_id"] == "00-0030506"].iloc[0]
    # Kelce's earliest appearance in synthetic fixture is 2024 week 1
    # (no historical seasons in te_weekly_stats), so inferred_year = 2024.
    assert kelce["age"] == 22.0  # 2024 - 2024 + 22.0
    assert kelce["is_rookie"] == 1.0
```

- [ ] **Step 2: Run the new test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_te.py::test_build_te_features_attach_trajectory_udfa_fallback -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_features/test_te.py
git commit -m "test(te): UDFA fallback uses inferred-draft-year + 22.0 offset"
```

---

### Task 8: Test — empty `draft_picks` default routes every row to fallback without errors

**Files:**
- Modify: `tests/test_features/test_te.py`

- [ ] **Step 1: Append the empty-draft-picks test**

Add to `tests/test_features/test_te.py`:

```python
def test_build_te_features_empty_draft_picks_default(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Calling build_te_features without draft_picks (default empty) must
    not raise. Every row falls through inferred-draft-year and the schema
    validates with non-NaN age (~22.0 from the offset) for everyone."""
    # Note: NO draft_picks kwarg passed.
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    TeFeaturesSchema.validate(out)
    assert "age" in out.columns
    # All rows should have age = inferred-fallback (22.0) since the
    # synthetic fixture's earliest-week is the target season.
    assert (out["age"] == 22.0).all()
```

- [ ] **Step 2: Run the new test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_te.py::test_build_te_features_empty_draft_picks_default -v`
Expected: PASS.

- [ ] **Step 3: Run the full TE test files end-to-end as a regression check**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features/test_te.py tests/test_features/test_te_leakage.py tests/test_features/test_trajectory_features.py -v 2>&1 | tail -15`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_features/test_te.py
git commit -m "test(te): empty draft_picks default routes all rows to fallback"
```

---

## Phase 3 — `_TE_FEATURE_COLUMNS` + cluster-A leftover fixtures

Goal: extend `_TE_FEATURE_COLUMNS` in `baseline.py` so `BaselineModel.fit` for TE consumes the 4 new schema columns (the lightgbm family auto-picks-up via dynamic schema derivation). Defensive grep for `opp_allowed_te_fppg_l4` to find every site building a synthetic minimal-TE-features row; add `age` / `is_rookie` / trend cols defaults to each. After this phase: parametrized cross-position smoke tests pass; baseline + lightgbm TE smoke fixtures construct schema-valid rows.

### Task 9: Extend `_TE_FEATURE_COLUMNS` in `baseline.py`

**Files:**
- Modify: `src/projections/models/baseline.py:408-427`

- [ ] **Step 1: Re-read the current `_TE_FEATURE_COLUMNS` tuple**

Run: `sed -n '405,430p' src/projections/models/baseline.py`
Expected: shows the 18-element tuple ending with `"opp_allowed_te_fppg_l4",`.

- [ ] **Step 2: Append 4 new entries with a comment**

Edit `src/projections/models/baseline.py`. Replace:

```python
    "opp_allowed_te_fppg_l4",
)
```

with:

```python
    "opp_allowed_te_fppg_l4",
    # Trajectory features (PR #25 family probe + 2026-05-04 TE integration).
    # lightgbm derives feature lists from TeFeaturesSchema dynamically and
    # auto-picks-up; baseline.py is hardcoded so must be updated explicitly.
    # Same spec gap class as PR #21 (RB, commit 9895dee) and PR #26 (WR).
    "age",
    "is_rookie",
    "volume_trend_l4_minus_prior_l4",
    "snap_pct_change_l4_vs_prior_l4",
)
```

- [ ] **Step 3: Smoke-verify both baseline + lightgbm see the 4 new columns**

Run: `.venv/Scripts/python.exe -c "
from projections.models.baseline import _TE_FEATURE_COLUMNS as bl
from projections.models.lightgbm import _filter_features
from projections.schemas import TeFeaturesSchema
schema_cols = list(TeFeaturesSchema.to_schema().columns.keys())
TRAJ = ('age', 'is_rookie', 'volume_trend_l4_minus_prior_l4', 'snap_pct_change_l4_vs_prior_l4')
for c in TRAJ:
    assert c in bl, f'{c} missing from baseline _TE_FEATURE_COLUMNS'
    assert c in _filter_features(tuple(schema_cols)), f'{c} missing from lightgbm _filter_features(schema_cols)'
print('both baseline + lightgbm see the 4 trajectory cols')
"`
Expected: `both baseline + lightgbm see the 4 trajectory cols`. (If `_filter_features` has a different signature in this codebase, adapt — the important thing is both lists end up containing the 4 new column names.)

- [ ] **Step 4: Run baseline + lightgbm test files**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models/test_baseline_te.py tests/test_models/test_lightgbm_te.py -v 2>&1 | tail -20`
Expected: tests likely fail because synthetic minimal-TE-feature rows in the test fixtures don't have the 4 new columns. **This is expected** — Task 10 closes the cluster-A leftovers.

- [ ] **Step 5: Commit (with the cluster-A failures unfixed — Task 10 closes them)**

```bash
git add src/projections/models/baseline.py
git commit -m "feat(baseline): extend _TE_FEATURE_COLUMNS with 4 trajectory cols

Same spec-gap class as PR #21 (RB, 9895dee) and PR #26 (WR) — baseline.py
hardcodes the per-position feature tuple while lightgbm derives from the
schema dynamically. Cluster-A synthetic TE fixtures need the new cols
added; Task 10 closes the failures."
```

---

### Task 10: Defensive grep + cluster-A leftover fixture special-casing

**Files:**
- Modify: `tests/test_features/test_cache.py` (likely; confirm via grep)
- Modify: `tests/test_models/test_lightgbm_te.py` (likely)
- Modify: `tests/test_models/test_baseline_te.py` (likely)
- Modify: any other site discovered

- [ ] **Step 1: Defensive grep for `opp_allowed_te_fppg_l4` to find every minimal-TE-feature-row construction site**

Run: `grep -rn "opp_allowed_te_fppg_l4" tests/`
Expected: shows every test file that builds a synthetic TE features row. Compare with PR #26's grep result for `opp_allowed_wr_fppg_l4` (3 cluster-A sites: `test_cache.py`, lightgbm/ensemble fixtures, tune_lightgbm test). Note each TE-side site for editing.

- [ ] **Step 2: For each site found, add the 4 trajectory column defaults to the synthetic row**

For every minimal-TE-features-row dict that currently ends with something like:
```python
{
    # ... existing TE feature columns ...
    "opp_allowed_te_fppg_l4": 12.5,  # or whatever value
}
```

extend to:
```python
{
    # ... existing TE feature columns ...
    "opp_allowed_te_fppg_l4": 12.5,
    "age": 26.0,
    "is_rookie": 0.0,
    "volume_trend_l4_minus_prior_l4": 0.5,
    "snap_pct_change_l4_vs_prior_l4": 0.05,
}
```

(Use sane finite values — `age` ∈ [22, 30], `is_rookie` ∈ {0, 1}, both trend cols at small finite numbers like 0.5 / 0.05.)

If the grep finds a helper like `_minimal_te_features_row()` in `test_cache.py`, modify it once at the helper level to add the 4 cols — that's better than per-call-site edits.

- [ ] **Step 3: Re-run baseline + lightgbm TE test files**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models/test_baseline_te.py tests/test_models/test_lightgbm_te.py tests/test_features/test_cache.py -v 2>&1 | tail -20`
Expected: all PASS — the cluster-A fixture rows now include the 4 trajectory cols.

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test(te): special-case trajectory cols on cluster-A synthetic TE fixtures

Defensive grep on opp_allowed_te_fppg_l4 found N cluster-A sites
constructing minimal TE features rows; each gets age/is_rookie/trend
cols at sane finite defaults. Same pattern as PR #26's WR cluster-A
edits at commits 1f1f415 / 33eea57 / 807f046."
```

---

### Task 11: End-of-Phase-3 verification — full suite + all gates

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

- [ ] **Step 6: Cross-position parametrized smoke test (catches signature regressions)**

Run: `.venv/Scripts/python.exe -m pytest -k "smoke" -v 2>&1 | tail -10`
Expected: all PASS.

If any step fails, fix and re-run before proceeding to Phase 4.

---

## Phase 4 — Real-data execution + reports

Goal: regenerate the TE feature cache against real `draft_picks` + `weekly_stats` + `snap_counts` data; run the walk-forward backtest for TE × all 5 model classes; run the dual-run adoption gate with `--coverage-threshold 0.35` (matches PR #26's structural-sparsity precedent); write the summary report. The gate's verdict on `(LightGBMNbModel, TE)` AND `(BaselineModel, TE)` determines which Phase 5 branch fires.

### Task 12: Verify `data/raw/draft_picks/` partition exists

**Files:** none (data verification only)

- [ ] **Step 1: List partition years**

Run: `ls data/raw/draft_picks/ 2>/dev/null | head -10`
Expected: `season=1980/`, `season=1981/`, ..., `season=2024/` (or similar). At minimum 1980-2024 needs coverage.

- [ ] **Step 2: If empty or missing, refresh draft_picks ingest first**

If step 1 returns nothing or only partial coverage:

Run: `.venv/Scripts/python.exe -c "from pathlib import Path; from projections.ingest.refresh_draft_picks import refresh_draft_picks; refresh_draft_picks(data_root=Path('data'), seasons=range(1980, 2025))"`
Expected: writes ~45 partition files; no error.

- [ ] **Step 3: Confirm**

Run: `ls data/raw/draft_picks/ | wc -l`
Expected: ≥45 (1980 through 2024 inclusive).

---

### Task 13: Refresh TE feature cache

**Files:** none (data regeneration only — output not committed)

- [ ] **Step 1: Refresh TE cache**

Run: `.venv/Scripts/python.exe scripts/refresh_features.py te --seasons 2018-2024 2>&1 | tail -20`
Expected: each (season, week) partition validates against the extended `TeFeaturesSchema`; no SchemaError raised.

- [ ] **Step 2: Spot-check a partition**

Run: `.venv/Scripts/python.exe -c "
import pandas as pd
from projections.store import read_partition
from pathlib import Path
df = read_partition(Path('data/features'), 'te', season=2024, week=5)
TRAJ = ['age', 'is_rookie', 'volume_trend_l4_minus_prior_l4', 'snap_pct_change_l4_vs_prior_l4']
for c in TRAJ:
    assert c in df.columns, f'{c} missing'
print('rows:', len(df))
print(df[TRAJ].describe())
"`
Expected: 4 trajectory cols present; non-NaN values for `age` (most rows); some NaN expected for `volume_trend_*` / `snap_pct_change_*` (early-season TEs without 8 prior active games).

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
            frames.append(read_partition(Path('data/features'), 'te', season=s, week=w))
        except FileNotFoundError:
            continue
df = pd.concat(frames, ignore_index=True)
print(f'total rows: {len(df)}')
for c in TRAJ:
    cov = df[c].notna().mean()
    print(f'  {c}: {cov:.1%} coverage')
"`
Expected: `age` ~95%; `is_rookie` ~95%; `volume_trend_l4_minus_prior_l4` ~44-45%; `snap_pct_change_l4_vs_prior_l4` ~70-71% — should approximate the probe's measured TE coverage (PR #25's PM entry: "TE (95.4 / 44.7 / 71.1)"). If materially different (e.g., volume_trend is 10% or 90%), the builder wiring is wrong — investigate before proceeding. Save the printed coverage stats for inclusion in the summary report.

---

### Task 14: Capture pre-PR baseline run sha for the dual-run gate

**Files:** none (git inspection only)

- [ ] **Step 1: Identify the most recent backtest run on `main` pre-this-branch**

Run: `git log --all --oneline -- tests/backtest/model_metrics.json | head -5`
Expected: shows the most recent commits that touched the backtest snapshot. The baseline run sha is the most recent one on `main` (pre-`feat/te-trajectory-features`).

- [ ] **Step 2: Note the pre-branch baseline sha for use in Task 16**

Capture the commit sha (e.g., `884d025` if main HEAD is unchanged from PR #26 merge) — this is the `--baseline-run` argument for `adoption_gate.py`.

---

### Task 15: Run the walk-forward backtest with the new TE features

**Files:**
- Modify: `tests/backtest/model_metrics.json` (snapshot update)
- Modify: `data/ensemble_weights/ensemble_te_*.json` (regen — 4 files, one per training-window cohort)

- [ ] **Step 1: Run backtest for TE, all 5 model classes**

Run: `.venv/Scripts/python.exe scripts/backtest.py --position TE --update-snapshot 2>&1 | tail -20`
Expected: walk-forward over 2021-2024 holdout years × 5 model classes (baseline / lightgbm / lightgbm-tuned / lightgbm-nb / ensemble); updates `tests/backtest/model_metrics.json` and `data/ensemble_weights/ensemble_te_*.json`. Runtime: ~10-20 minutes (TE × 5 classes × 4 holdout years × walk-forward; ensemble adds the per-stat pinball weight optimization).

- [ ] **Step 2: Diff the snapshot to confirm only TE rows changed**

Run: `git diff tests/backtest/model_metrics.json | head -40`
Expected: changes confined to `position: "TE"` rows; QB/RB/WR rows unchanged (their schemas weren't touched).

- [ ] **Step 3: Diff the ensemble weights to confirm only TE files changed**

Run: `git status data/ensemble_weights/ | head -10`
Expected: `ensemble_te_*.json` files modified; `ensemble_qb_*.json`, `ensemble_rb_*.json`, `ensemble_wr_*.json` unchanged.

- [ ] **Step 4: Commit the snapshot + ensemble weight updates**

```bash
git add tests/backtest/model_metrics.json data/ensemble_weights/ensemble_te_*.json
git commit -m "snapshot: backtest update for TE trajectory features

Walk-forward over 2021-2024, 5 model classes. TE rows reflect the
new trajectory feature inputs; QB/RB/WR unchanged. Ensemble weight
JSONs for TE regenerated since per-stat sub-model fits change."
```

---

### Task 16: Run the dual-run adoption gate

**Files:**
- Add: `reports/adoption_gate_te_trajectory_features.md`
- Add: `reports/adoption_gate_te_trajectory_features.csv`

- [ ] **Step 1: Capture the candidate (current branch) sha**

Run: `git rev-parse HEAD`
Expected: 40-char sha (the candidate run sha).

- [ ] **Step 2: Run the adoption gate with the structural-sparsity coverage threshold**

Run:
```bash
.venv/Scripts/python.exe scripts/adoption_gate.py \
  --position TE \
  --baseline-run <baseline-sha-from-Task-14> \
  --candidate-run <candidate-sha-from-step-1> \
  --coverage-threshold 0.35 \
  --output-md reports/adoption_gate_te_trajectory_features.md \
  --output-csv reports/adoption_gate_te_trajectory_features.csv
```
Expected: produces per-(model_class, TE) verdicts across all 5 classes. **Must use `--coverage-threshold 0.35`** — without it, the pooled coverage check fails on `volume_trend_*` and `snap_pct_change_*` (~50% NaN by design).

- [ ] **Step 3: Read the gate's `(LightGBMNbModel, TE)` verdict — this is the binding cell**

Run: `grep -A 2 "lightgbm-nb" reports/adoption_gate_te_trajectory_features.md | head -10`
Expected: a single verdict line — `ADOPT`, `MARGINAL`, or `DO_NOT_ADOPT` — with the composite RMSE delta + 95% CI. **This is the binding decision** (spec §1.3.5).

- [ ] **Step 4: Read the gate's `(BaselineModel, TE)` verdict — needed for the modified-shape branch decision**

Run: `grep -A 2 "baseline" reports/adoption_gate_te_trajectory_features.md | head -10`
Expected: a single verdict line. If RMSE delta CI is **strictly above zero**, this is REGRESSION — Phase 5 fires the modified-shape branch. Otherwise (DO_NOT_ADOPT or ADOPT), Phase 5 ships as designed.

- [ ] **Step 5: Commit the gate reports**

```bash
git add reports/adoption_gate_te_trajectory_features.md reports/adoption_gate_te_trajectory_features.csv
git commit -m "report(te-trajectory): adoption gate output

Dual-run gate, --coverage-threshold 0.35, all 5 model classes on TE
2021-2024 holdout. Binding cell: (LightGBMNbModel, TE) per spec
§1.3.5 — first integration to bind on a non-default class.
(BaselineModel, TE) cell is informational + drives the modified-shape
ship contingency."
```

---

### Task 17: Write the summary report

**Files:**
- Add: `reports/te_trajectory_features_summary.md`

- [ ] **Step 1: Draft the summary report**

Create `reports/te_trajectory_features_summary.md` with the following structure (fill in numbers from the gate output + Task 13 step 3 coverage stats):

```markdown
# TE Trajectory Features Integration — Summary Report

**Status:** [ADOPT (ship-as-designed) / ADOPT (ship-modified-shape) / MARGINAL / DO_NOT_ADOPT] on (LightGBMNbModel, TE)
**Branch:** `feat/te-trajectory-features`
**Spec:** `docs/superpowers/specs/2026-05-04-te-trajectory-features-design.md`
**Plan:** `docs/superpowers/plans/2026-05-04-te-trajectory-features.md`
**Date:** 2026-05-XX

## Decision

[Single sentence — ship-as-designed / ship-modified-shape / revert, with the binding magnitude + CI on (lgb-nb, TE).]

## Binding-cell shift from PR #21 / PR #26 — first non-default-class integration

PR #21 (RB) and PR #26 (WR) bound the ship decision on `(BaselineModel, position)` because baseline was each position's `default_model_class`. For TE, baseline is still the production default but the PR #25 trajectory probe ADOPT'd TE only under `lightgbm-nb` (-0.0107 fpts), not under baseline. Per spec §1, this PR binds on `(LightGBMNbModel, TE)` — the cell where the probe's signal lives. TE production routing stays on `baseline`; flipping `_PositionDispatch[TE].default_model_class` is a deferred cross-class follow-up (see "Cross-class deferred follow-up" below).

## Probe-vs-gate calibration

| Source | Composite RMSE Δ on (LightGBMNbModel, TE) augment | 95% CI |
|---|---:|---|
| PR #25 probe (predicted) | -0.0107 | [-0.0191, -0.0028] |
| This PR's gate (measured) | [filled] | [filled] |

[2-3 sentences on whether probe and gate agree (PR #20→#21 matched to 4 decimals; PR #25→#26 matched within CI at ~10% smaller magnitude). For TE at the smaller -0.0107 magnitude, ~10% calibration error is ~0.001 fpts — within the per-cell noise floor.]

## Per-(model_class, TE) verdicts

| Model class | RMSE Δ | 95% CI | Verdict |
|---|---:|---|:---:|
| **baseline** | [filled] | [filled] | [filled] (informational; drives modified-shape contingency) |
| lightgbm | [filled] | [filled] | [filled] |
| lightgbm-tuned | [filled] | [filled] | [filled] (TODO #29 pruning candidate) |
| **lightgbm-nb** | [filled] | [filled] | **[filled] (binding)** |
| ensemble | [filled] | [filled] | [filled] |

[Cross-check: the baseline cell tests whether the probe's TE baseline DO_NOT_ADOPT prediction holds. If baseline measures REGRESSION (CI strictly above zero), the modified-shape ship path fires per spec §1.3.5 — `_TE_FEATURE_COLUMNS` extension reverted; lightgbm family still picks up cols via dynamic schema derivation.]

## Coverage statistics (2021-2024 eval window)

| Column | Coverage | Probe coverage (PR #25) | Match? |
|---|---:|---:|:---:|
| age | [from Task 13 step 3] | 95.4% | [yes/no] |
| is_rookie | [from Task 13 step 3] | 95.4% | [yes/no] |
| volume_trend_l4_minus_prior_l4 | [from Task 13 step 3] | 44.7% | [yes/no] |
| snap_pct_change_l4_vs_prior_l4 | [from Task 13 step 3] | 71.1% | [yes/no] |

[If coverage matches the probe within ~1pp, the builder wiring is correct. If divergent, document the cause.]

## Threshold relaxation rationale

`--coverage-threshold 0.35` used (spec §1.3.3). Trajectory's trend features are structurally sparse — they require 8 prior active games per player, which excludes ~50% of player-weeks across all years. Same precedent as PR #25's probe and PR #26's WR integration.

## Cross-class deferred follow-up

TE production routes to `baseline` per Plan 8 (2026-04-29). With trajectory cols now in `TeFeaturesSchema`, a separate cross-class re-eval (`scripts/adoption_gate.py --position TE` comparing `lightgbm-nb` candidate to `baseline` baseline at the position-level — not the within-class with-vs-without comparison this PR ran) could justify flipping `_PositionDispatch[TE].default_model_class` to `lightgbm-nb`. The Plan 8 numbers from 2026-04-29 said `lgb-nb` for TE was +0.0028 fpts (point estimate slightly worse, CI bracketed zero); naively stacking with this PR's measured trajectory lift might bring lgb-nb-with-trajectory close to or below baseline-without-trajectory. Not load-bearing for any current consumer; queue alongside the next TE-related work.

## What this closes

[If ADOPT in either form:] TODO #24's "trailing-8-game unit" branch for TE — the bundled trajectory probe carried clear signal at the lgb-nb cell and integrates production. WR was closed by PR #26; TE is closed by this PR. Refined-unit candidates (`age²`, `is_2nd_year` flags, longer windows, `has_trajectory_history` indicator) remain unexplored under the same TODO.

[If DO_NOT_ADOPT:] The probe-vs-gate divergence for the TE lgb-nb cell is documented; spec gap (if any) captured for future probe-tuning.

## Next track

[Either: TODO #25 weather-features sibling probe (unblocked, independent of trajectory follow-up), or: cross-class TE production routing re-eval, or: Plan 4 public API + CLI verbs.]
```

- [ ] **Step 2: Fill in numbers from `reports/adoption_gate_te_trajectory_features.md` and Task 13 step 3 output**

Open the gate report; copy the per-cell numbers into the summary table.

- [ ] **Step 3: Commit**

```bash
git add reports/te_trajectory_features_summary.md
git commit -m "report(te-trajectory): family summary — verdict [ADOPT/...]"
```

---

## Phase 5 — Conditional code adjustments + documentation

Goal: append the decision-log entry to `project_management.md`, update `TODO.md` #24, and conditionally adjust code based on the gate verdict from Task 16. **Three branches; only one fires.**

### Task 18a — IF `(lgb-nb, TE)` verdict is `ADOPT` AND `(baseline, TE)` is NOT `REGRESSION` (ship-as-designed)

**Files:**
- Modify: `project_management.md` (top of file)
- Modify: `TODO.md` #24

- [ ] **Step 1: Append the top-of-file decision-log entry to `project_management.md`**

Format matches PR #26's entry. Insert immediately after the line `---` at line 5 (before the existing top-most entry):

```markdown
## TE Trajectory Features Integration — verdict ADOPT on (LightGBMNbModel, TE); shipped (2026-05-XX, on branch `feat/te-trajectory-features`)

**Status:** Production integration of the 4 trajectory features into `TeFeaturesSchema` + `build_te_features` per `docs/superpowers/specs/2026-05-04-te-trajectory-features-design.md`. Wired `attach_trajectory_features` into `build_te_features` via the existing `draft_picks` kwarg (plumbed PR #26). Updated `baseline.py:_TE_FEATURE_COLUMNS` (the spec gap PR #21 / PR #26 caught for RB / WR — same recurring class). Extended `baseline_weekly_stats_te` to 17/17/4 weeks for trajectory's 8-game prior window.

**Dual-run gate verdict on `(LightGBMNbModel, TE)`:** `ADOPT` (composite RMSE delta [filled] fpts, CI [filled]). Probe predicted -0.0107 fpts; gate matched the point estimate within [N] decimal places. **First production integration in the project to bind on a non-default model class** (TE production routes to `baseline`; lgb-nb is where the probe's signal lived).

**Per-(model_class, TE) verdicts:**

| Model class | RMSE Δ | 95% CI | Verdict |
|---|---:|---|:---:|
| baseline | [filled] | [filled] | [filled] (informational — confirmed probe's DO_NOT_ADOPT prediction) |
| lightgbm | [filled] | [filled] | [filled] |
| lightgbm-tuned | [filled] | [filled] | [filled] (TODO #29 pruning candidate) |
| **lightgbm-nb** | [filled] | [filled] | **ADOPT (binding)** |
| ensemble | [filled] | [filled] | [filled] |

**Coverage relaxation:** `--coverage-threshold 0.35` (matches PR #25's probe + PR #26's WR integration precedent for structurally-sparse trajectory features).

**What this closes:** TODO #24's TE branch of the trajectory candidate (trailing-8-game unit). Combined with PR #26's WR integration, the trailing-8-game-unit branch is now closed at all three of PR #25's ADOPT cells (WR baseline, WR lgb-nb, TE lgb-nb). Refined-unit candidates (`age²`, `is_2nd_year` flags, longer trailing windows, `has_trajectory_history` indicator) remain unexplored under the same TODO.

**Cross-class deferred follow-up:** TE production routing remains on `baseline`. A cross-class re-eval (lgb-nb-with-trajectory vs baseline-without-trajectory at position level) could justify flipping `_PositionDispatch[TE].default_model_class` to `lightgbm-nb`. Not load-bearing for any current consumer; queued under TODO #24.

See `reports/te_trajectory_features_summary.md` for the full decision log + per-mode table + probe-vs-gate calibration.

---
```

- [ ] **Step 2: Update `TODO.md` #24**

Open `TODO.md`, find the section header `### 24. Player-trajectory features (age curves, career arc, trend gradients)`. Append at the end of that section:

```markdown
**Update 2026-05-XX (TE trajectory features integration, branch `feat/te-trajectory-features`):** Production integration of the 4 trajectory features into `TeFeaturesSchema` + `build_te_features` per `docs/superpowers/specs/2026-05-04-te-trajectory-features-design.md`. Dual-run gate verdict on `(LightGBMNbModel, TE)`: **`ADOPT`** (composite RMSE delta [filled] fpts, CI [filled]). Probe predicted -0.0107 fpts; gate matched [calibration commentary]. **First production integration to bind on a non-default model class** (TE production stays on `baseline` per Plan 8). Cross-class TE production routing flip queued separately. Trailing-8-game-unit branch now closed at all three PR #25 ADOPT cells (WR via PR #26, TE via this PR). Refined-unit candidates remain unexplored. See `reports/te_trajectory_features_summary.md`.
```

- [ ] **Step 3: Commit**

```bash
git add project_management.md TODO.md
git commit -m "docs(pm): record TE trajectory features integration ADOPT verdict"
```

- [ ] **Step 4: Push branch + open PR**

```bash
git push -u origin feat/te-trajectory-features
gh pr create --title "feat(te): trajectory features integration — ADOPT (lgb-nb)" --body "$(cat <<'EOF'
## Summary

Promotes PR #25's trajectory family probe TE-cell signal into the production TE feature pipeline. Adds 4 nullable-float trajectory columns to `TeFeaturesSchema`; wires `attach_trajectory_features` into `build_te_features`. **First integration to bind on a non-default model class** — the binding cell is `(LightGBMNbModel, TE)`, not `(BaselineModel, TE)`, because the probe's TE signal lived on lgb-nb. Dual-run gate verdict on `(lgb-nb, TE)` is `ADOPT` ([filled] fpts, CI [filled]).

- Probe predicted -0.0107 fpts; gate measured [filled].
- Coverage threshold: 0.35 (structural-sparsity precedent matching PR #25 + PR #26).
- TE production routing unchanged (still `baseline`); cross-class flip is a deferred follow-up.
- Spec gap from PR #21 / PR #26 explicit: `baseline.py:_TE_FEATURE_COLUMNS` updated; lightgbm derives dynamically from schema and auto-picks-up.

## Test plan

- [x] Full pytest suite passes
- [x] mypy strict zero violations
- [x] ruff check + format clean
- [x] TE feature cache regenerates against extended schema
- [x] Backtest snapshot diff confined to TE rows
- [x] Adoption gate `--coverage-threshold 0.35` runs on all 5 model classes

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

### Task 18b — IF `(lgb-nb, TE)` verdict is `ADOPT` AND `(baseline, TE)` is `REGRESSION` (ship-modified-shape)

**Files:**
- Modify: `src/projections/models/baseline.py` (revert `_TE_FEATURE_COLUMNS` extension only)
- Modify: `tests/backtest/model_metrics.json` (re-run baseline TE rows only)
- Modify: `project_management.md`, `TODO.md`
- Modify: `reports/te_trajectory_features_summary.md` (add modified-shape addendum)

- [ ] **Step 1: Revert the `_TE_FEATURE_COLUMNS` extension only**

Edit `src/projections/models/baseline.py` and remove the 4 trajectory col entries (and the comment block) from `_TE_FEATURE_COLUMNS`, restoring it to the 18-element pre-Task-9 state. **Do not** revert the schema edit (Task 1) or the builder edit (Task 2) — the lightgbm family auto-picks-up the cols via dynamic schema derivation, and we want them to keep doing that.

- [ ] **Step 2: Re-run baseline-only TE backtest to refresh `model_metrics.json`**

Run: `.venv/Scripts/python.exe scripts/backtest.py --position TE --model baseline --update-snapshot 2>&1 | tail -10`
Expected: only baseline TE rows in `tests/backtest/model_metrics.json` change. lightgbm family rows stay as-is from Task 15 (they correctly reflect with-trajectory-cols since the schema still has them).

- [ ] **Step 3: Verify the lightgbm family didn't lose the trajectory cols by re-checking dynamic feature derivation**

Run: `.venv/Scripts/python.exe -c "
from projections.models.lightgbm import _filter_features
from projections.schemas import TeFeaturesSchema
schema_cols = list(TeFeaturesSchema.to_schema().columns.keys())
TRAJ = ('age', 'is_rookie', 'volume_trend_l4_minus_prior_l4', 'snap_pct_change_l4_vs_prior_l4')
features = _filter_features(tuple(schema_cols))
for c in TRAJ:
    assert c in features, f'{c} missing from lightgbm derived features'
print('lightgbm family still sees the 4 trajectory cols (good)')
"`
Expected: `lightgbm family still sees the 4 trajectory cols (good)`.

- [ ] **Step 4: Append modified-shape addendum to summary report**

Edit `reports/te_trajectory_features_summary.md`. Add at the bottom:

```markdown
## Modified-shape ship branch fired (spec §1.3.5)

`(BaselineModel, TE)` returned REGRESSION ([filled] fpts, CI [filled] — strictly above zero). Per spec §1.3.5, the modified-shape ship path fired:

- **Schema edit** (4 cols added to `TeFeaturesSchema`) — kept; lightgbm family auto-picks-up via dynamic schema derivation.
- **Builder edit** (`build_te_features` wires `attach_trajectory_features`) — kept; cols are computed at refresh-features time and persisted in the cache.
- **`_TE_FEATURE_COLUMNS` extension in `baseline.py`** — REVERTED. Baseline TE production no longer sees the 4 cols.
- **`tests/backtest/model_metrics.json` baseline TE rows** — re-run after the revert. Lightgbm family rows kept from the with-trajectory-cols backtest.

Effect: lightgbm-family TE classes consume trajectory features in production (via `production_model_for(Position.TE)` if routing ever flips); baseline TE production output unchanged from pre-PR.

This is the project's first time the modified-shape branch has fired. Pre-decided in spec §1.3.5; documented here for record-of-decision.
```

- [ ] **Step 5: Append decision-log entry to `project_management.md`** — same shape as Task 18a step 1, but with `verdict ADOPT (modified-shape) on (LightGBMNbModel, TE)` header. Note prominently that the modified-shape branch fired.

- [ ] **Step 6: Update `TODO.md` #24** — same shape as Task 18a step 2 with the modified-shape outcome documented.

- [ ] **Step 7: Commit + push + open PR**

```bash
git add src/projections/models/baseline.py tests/backtest/model_metrics.json reports/te_trajectory_features_summary.md project_management.md TODO.md
git commit -m "docs(pm): record TE trajectory features ADOPT (modified-shape) verdict

Spec §1.3.5 modified-shape branch fired: baseline TE returned
REGRESSION ([filled] fpts) while lgb-nb TE ADOPTed ([filled] fpts).
_TE_FEATURE_COLUMNS reverted to keep baseline TE production unchanged;
schema + builder edits kept so the lightgbm family (which derives
features dynamically) still consumes the cols. First time the
modified-shape branch has fired in the project."
git push -u origin feat/te-trajectory-features
gh pr create --title "feat(te): trajectory features — ADOPT modified-shape (lgb-nb)" --body "$(cat <<'EOF'
## Summary

PR #25 trajectory probe's TE lgb-nb cell ADOPT'd in the gate ([filled] fpts) — but the baseline TE cell REGRESSED ([filled] fpts, CI strictly above zero). Spec §1.3.5 pre-decided this branch: ship the schema + builder edits (lightgbm family auto-picks-up the cols dynamically), but revert the `_TE_FEATURE_COLUMNS` extension so baseline TE production output stays unchanged.

- (lgb-nb, TE): ADOPT [filled] fpts.
- (baseline, TE): REGRESSION [filled] fpts — the trigger for modified-shape.
- TE production routing unchanged (still `baseline`).
- Lightgbm family TE classes now have access to trajectory cols if production routing ever flips (deferred follow-up).

## Test plan

- [x] Full pytest suite passes (post-revert)
- [x] mypy strict zero violations
- [x] ruff check + format clean
- [x] Lightgbm family dynamic feature derivation still includes the 4 trajectory cols

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

### Task 18c — IF `(lgb-nb, TE)` verdict is `MARGINAL` or `DO_NOT_ADOPT` (revert)

**Files:**
- Modify: `src/projections/schemas.py` (revert 4 columns)
- Modify: `src/projections/features/te.py` (revert builder integration; keep `_EMPTY_DRAFT_PICKS` singleton + kwarg)
- Modify: `src/projections/models/baseline.py` (revert `_TE_FEATURE_COLUMNS` extension)
- Modify: `tests/test_features/test_te.py` (revert 4 new tests + happy-path extension)
- Modify: cluster-A leftover sites (revert trajectory col additions)
- Modify: `tests/backtest/model_metrics.json` (re-run TE × all 5 classes post-revert)
- Modify: `data/ensemble_weights/ensemble_te_*.json` (re-run TE post-revert)
- Modify: `project_management.md` + `TODO.md`

**Note:** Keep the `te_draft_picks` fixture (Task 3) and the `baseline_weekly_stats_te` extension (Task 4) — they're useful infrastructure for any future TE-trajectory revisit.

- [ ] **Step 1: Revert TE-specific changes**

Use `git revert <commit-sha>` for each commit that touched: `TeFeaturesSchema` (Task 1), `build_te_features` integration (Task 2), the 4 join-side tests (Tasks 5-8), the `_TE_FEATURE_COLUMNS` extension (Task 9), the cluster-A leftovers (Task 10). Resolve any conflicts from later changes.

- [ ] **Step 2: Re-run TE backtest snapshot post-revert**

Run: `.venv/Scripts/python.exe scripts/backtest.py --position TE --update-snapshot 2>&1 | tail -10`
Expected: TE rows in `model_metrics.json` and `ensemble_te_*.json` revert to pre-PR values.

- [ ] **Step 3: Re-run full test suite + lint to confirm clean revert**

Run: `.venv/Scripts/python.exe -m pytest -v 2>&1 | tail -10 && .venv/Scripts/python.exe -m mypy src tests && .venv/Scripts/python.exe -m ruff check src tests scripts`
Expected: all PASS, zero violations.

- [ ] **Step 4: Append decision-log entry to `project_management.md` documenting the divergence**

Same shape as Task 18a step 1, but with a `verdict MARGINAL (or DO_NOT_ADOPT) on (LightGBMNbModel, TE)` header and a "What this closes" note pointing to the probe-vs-gate divergence on the binding cell.

- [ ] **Step 5: Update `TODO.md` #24** — same shape as Task 18a step 2 with the revert outcome documented. Close the TE branch as "probe SIGNAL did not reproduce in production gate."

- [ ] **Step 6: Commit + push + open documentation-only PR**

```bash
git add .
git commit -m "docs(pm): record TE trajectory features [MARGINAL/DO_NOT_ADOPT] verdict + revert"
git push -u origin feat/te-trajectory-features
gh pr create --title "spec/report(te-trajectory): family verdict [MARGINAL/DO_NOT_ADOPT]" --body "$(cat <<'EOF'
## Summary

Trajectory family integration for TE probed and gated; gate verdict on `(LightGBMNbModel, TE)` was [MARGINAL/DO_NOT_ADOPT]. Schema + builder + baseline feature list changes reverted per spec §1.3.5; `te_draft_picks` fixture and `baseline_weekly_stats_te` extension kept (infrastructure for any TE-trajectory revisit).

- Probe predicted -0.0107 fpts; gate measured [filled].
- See `reports/te_trajectory_features_summary.md` for the full divergence analysis.

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

Quick scan after Phase 5 completes:

- [ ] All 4 trajectory columns appear in `TeFeaturesSchema` (or are reverted on the DO_NOT_ADOPT path).
- [ ] `build_te_features` wires `attach_trajectory_features` (or is reverted on DO_NOT_ADOPT).
- [ ] `te_draft_picks` fixture is in `tests/test_features/conftest.py` regardless of gate verdict (kept for future revisit).
- [ ] `baseline_weekly_stats_te` covers 17/17/4 weeks regardless of gate verdict (kept for future revisit).
- [ ] `baseline.py:_TE_FEATURE_COLUMNS` includes the 4 new names ONLY if ship-as-designed branch fired; reverted on ship-modified-shape or revert.
- [ ] Adoption gate report exists at `reports/adoption_gate_te_trajectory_features.{md,csv}`.
- [ ] Summary report exists at `reports/te_trajectory_features_summary.md` with binding-cell-shift rationale + cross-class deferred-follow-up note + (if applicable) modified-shape addendum.
- [ ] PM + TODO #24 reflect the verdict (one of the three branches).
- [ ] Coverage check from Task 13 step 3 was within ~1pp of the probe's measured TE coverage (95.4 / 44.7 / 71.1) — if divergent, documented in the summary.
- [ ] PR title + body explicitly note the binding-cell shift from PR #21 / PR #26's `(BaselineModel, position)` pattern.

If all green, the spec's success criteria (§1.3) are satisfied.
