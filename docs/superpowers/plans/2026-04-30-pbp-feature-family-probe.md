# PBP Feature Family Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the override-generation infrastructure for the PBP feature family probe (4 pure compute functions + assembler + script + family-verdict helper), run the probe in 2–4 modes, commit reports + summary, and update docs with the family verdict.

**Architecture:** Pure pandas on the existing 27-column `PbpSchema` (Plan 9). Trailing-4 windows via `groupby('team').rolling(4, min_periods=4).mean().shift(1)` over `(team, season, week)`-sorted frames; cross-season backfill handled by feeding prior-season PBP concat'd with current. Override parquet (4 uniform team-level columns) lives at `data/features_probe/pbp_family.parquet`; consumed by the existing probe CLI (PR #18) with no new flags. A new `family_verdict_from_reports` helper collapses per-report verdicts into a one-bit family verdict.

**Tech Stack:** Python 3.13, pandas, pyarrow, pandera, pytest (no network), mypy strict, ruff. PBP / weekly_stats / schedules read via `projections.store.read_partition`.

**Spec:** `docs/superpowers/specs/2026-04-30-pbp-feature-family-probe-design.md`.

**Branch:** `feat/probe-pbp-family` (already cut from `origin/main` at `98c2088`; spec committed at `c1813e0` + `31d9a1b`).

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `src/projections/features/pbp_team_features.py` | Create | Four pure compute fns + `build_pbp_family_overrides` assembler. |
| `tests/test_features/test_pbp_team_features.py` | Create | Synthetic-PBP unit tests per compute + assembler. |
| `src/projections/backtest/feature_probe.py` | Modify | Add `family_verdict_from_reports` helper next to `phase1_should_fire_phase2`. |
| `tests/test_backtest/test_feature_probe.py` | Modify | Add 3 truth-table tests for the new helper. |
| `scripts/build_pbp_family_override.py` | Create | Argparse + I/O glue: load partitions, call assembler, write parquet. |
| `data/features_probe/pbp_family.parquet` | Created at run-time | Override input for the probe. **Not committed** (regenerable). |
| `reports/feature_probe_pbp_family_*_{QB,RB,WR,TE}.{md,csv}` | Committed at run-time | 16–32 probe report files. |
| `reports/feature_probe_pbp_family_summary.md` | Committed at run-time | Hand-written narrative + verdict. |
| `TODO.md` | Modify | Append to #3c. |
| `project_management.md` | Modify | Append decision-log entry. |
| `CONTRIBUTING.md` | Modify | Add "Regenerating the PBP family override" subsection. |

Tasks 1–7 are pure-code (TDD, fully testable in CI). Task 8 generates the override parquet against real PBP. Tasks 9–13 run probes and commit reports. Tasks 14–16 close out (summary + docs + verification gates).

---

### Task 1: Add `family_verdict_from_reports` helper (TDD)

**Files:**
- Modify: `src/projections/backtest/feature_probe.py` (append helper next to `phase1_should_fire_phase2`)
- Test: `tests/test_backtest/test_feature_probe.py` (append three tests)

Builds the helper before the four compute functions because the helper has no dependencies on the override pipeline — it operates purely on `ProbeReport` instances and is the smallest unit to land first.

- [ ] **Step 1: Write the three failing tests**

Append to `tests/test_backtest/test_feature_probe.py`:

```python
from projections.backtest.feature_probe import family_verdict_from_reports
from projections.backtest.adoption_gate import PositionVerdict


def _stub_pooled_psv(verdict: str) -> PerStatVerdict:
    return PerStatVerdict(
        position=Position.QB,
        stat=Stat.PASSING_YARDS,
        year_or_pooled="pooled",
        n_paired=2676,
        rmse_delta=BootstrapDelta(
            point=-0.5 if verdict == "SIGNAL" else 0.0,
            lo_95=-0.9 if verdict == "SIGNAL" else -0.1,
            hi_95=-0.1 if verdict == "SIGNAL" else 0.1,
            n_paired_rows=2676,
            n_bootstrap=1000,
        ),
        r_squared_delta=0.0,
        verdict=verdict,  # type: ignore[arg-type]
    )


def _stub_phase2_verdict(verdict: str) -> PositionVerdict:
    """Minimal PositionVerdict with the right `verdict` field; numbers
    don't matter for the family-verdict helper."""
    bd_zero = BootstrapDelta(point=0.0, lo_95=-0.1, hi_95=0.1, n_paired_rows=100, n_bootstrap=1000)
    return PositionVerdict(
        position=Position.QB,
        n_paired=100,
        rmse_delta=bd_zero,
        spearman_delta=bd_zero,
        verdict=verdict,  # type: ignore[arg-type]
        per_year=[],
    )


def _stub_report(*, phase1_verdict: str, phase2_verdict: str | None) -> ProbeReport:
    """Build a one-row Phase 1 report; Phase 2 either present or skipped."""
    phase2 = [_stub_phase2_verdict(phase2_verdict)] if phase2_verdict is not None else None
    return ProbeReport(
        candidate_name="stub",
        model_class="baseline",
        baseline_features_path="data/features",
        override_paths=("data/features_probe/x.parquet",),
        drop_columns=(),
        phase1=[_stub_pooled_psv(phase1_verdict)],
        phase2=phase2,
        phase2_skip_reason=None if phase2 is not None else "no_signal",
    )


def test_family_verdict_signal_via_phase1() -> None:
    """Pooled Phase 1 SIGNAL on any report flips family to SIGNAL."""
    reports = [
        _stub_report(phase1_verdict="NULL", phase2_verdict=None),
        _stub_report(phase1_verdict="SIGNAL", phase2_verdict="DO_NOT_ADOPT"),
    ]
    assert family_verdict_from_reports(reports) == "SIGNAL"


def test_family_verdict_signal_via_phase2() -> None:
    """Phase 2 ADOPT or MARGINAL anywhere flips family to SIGNAL even if all
    Phase 1 cells are NULL."""
    reports = [
        _stub_report(phase1_verdict="NULL", phase2_verdict=None),
        _stub_report(phase1_verdict="NULL", phase2_verdict="MARGINAL"),
    ]
    assert family_verdict_from_reports(reports) == "SIGNAL"


def test_family_verdict_null_when_all_null() -> None:
    """Family is NULL only when every pooled Phase 1 cell is NULL/REGRESSION
    AND every Phase 2 cell is DO_NOT_ADOPT (or absent)."""
    reports = [
        _stub_report(phase1_verdict="NULL", phase2_verdict=None),
        _stub_report(phase1_verdict="REGRESSION", phase2_verdict="DO_NOT_ADOPT"),
    ]
    assert family_verdict_from_reports(reports) == "NULL"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backtest/test_feature_probe.py -k "family_verdict" -v`
Expected: 3 tests FAIL with `ImportError: cannot import name 'family_verdict_from_reports'`.

- [ ] **Step 3: Implement helper**

Append to `src/projections/backtest/feature_probe.py` directly below `phase1_should_fire_phase2`:

```python
def family_verdict_from_reports(reports: list[ProbeReport]) -> Literal["SIGNAL", "NULL"]:
    """Family-level verdict across executed probe reports.

    SIGNAL iff any pooled Phase-1 PerStatVerdict has verdict == "SIGNAL"
    OR any Phase-2 PositionVerdict has verdict in ("ADOPT", "MARGINAL").
    NULL otherwise — including when every Phase-1 cell is REGRESSION
    (the family-level question is "is there orthogonal signal?", not
    "does this hurt").

    Spec: docs/superpowers/specs/2026-04-30-pbp-feature-family-probe-design.md §4.
    """
    for report in reports:
        for psv in report.phase1:
            if psv.year_or_pooled == "pooled" and psv.verdict == "SIGNAL":
                return "SIGNAL"
        if report.phase2 is not None:
            for pv in report.phase2:
                if pv.verdict in ("ADOPT", "MARGINAL"):
                    return "SIGNAL"
    return "NULL"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backtest/test_feature_probe.py -k "family_verdict" -v`
Expected: 3 PASS.

Also run: `pytest tests/test_backtest/test_feature_probe.py -v`
Expected: full file passes (no regressions in pre-existing tests).

- [ ] **Step 5: Lint + type-check**

Run: `ruff check src/projections/backtest/feature_probe.py tests/test_backtest/test_feature_probe.py`
Expected: zero violations.

Run: `mypy src/projections/backtest/feature_probe.py tests/test_backtest/test_feature_probe.py`
Expected: zero violations.

Run: `ruff format --check src/projections/backtest/feature_probe.py tests/test_backtest/test_feature_probe.py`
Expected: no drift.

- [ ] **Step 6: Commit**

```bash
git add src/projections/backtest/feature_probe.py tests/test_backtest/test_feature_probe.py
git commit -m "feat(probe): family_verdict_from_reports helper

Collapses one-or-more probe reports into a one-bit family verdict
(SIGNAL/NULL) per spec §4. Pooled Phase-1 SIGNAL OR Phase-2
ADOPT/MARGINAL anywhere flips to SIGNAL; otherwise NULL. REGRESSION
cells do not flip the family verdict.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `compute_team_pace` + test (TDD)

**Files:**
- Create: `src/projections/features/pbp_team_features.py` (new module — this task creates the file with imports and the first compute fn)
- Create: `tests/test_features/test_pbp_team_features.py` (new test file with the first test)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_features/test_pbp_team_features.py`:

```python
"""PBP team-level feature computes — tests."""

from __future__ import annotations

import pandas as pd
import pytest


def _make_pbp_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a synthetic PBP frame with sane defaults for unspecified columns.

    The compute fns only read a subset of PbpSchema columns; tests fill in
    only the columns the function under test uses, defaults the rest."""
    defaults = {
        "play_id": 1,
        "game_id": "2024_01_KC_BAL",
        "season": 2024,
        "week": 1,
        "posteam": "KC",
        "defteam": "BAL",
        "play_type": "pass",
        "qb_dropback": 1.0,
        "qb_scramble": 0.0,
        "sack": 0.0,
        "rush_attempt": 0.0,
        "pass_attempt": 1.0,
        "epa": 0.0,
        "wpa": 0.0,
        "success": 0.0,
        "air_yards": 8.0,
        "yards_after_catch": 0.0,
        "complete_pass": 1.0,
        "xpass": 0.5,
        "pass_oe": 0.0,
        "down": 1.0,
        "ydstogo": 10,
        "yardline_100": 50.0,
        "half_seconds_remaining": 600.0,
        "passer_player_id": "00-0011111",
        "rusher_player_id": None,
        "receiver_player_id": "00-0022222",
    }
    out = []
    for r in rows:
        merged = {**defaults, **r}
        out.append(merged)
    return pd.DataFrame(out)


def test_pace_counts_pass_and_run_only() -> None:
    """Kickoffs / punts / FGs / no_play do not count toward pace."""
    from projections.features.pbp_team_features import compute_team_pace

    # Build 4 prior weeks for KC so trailing-4 has a window, then test week 5.
    # Each prior week: 50 pass+run plays + 10 special-teams plays.
    rows: list[dict[str, object]] = []
    for wk in range(1, 6):
        for i in range(50):
            rows.append({
                "season": 2024, "week": wk, "posteam": "KC",
                "play_type": "pass" if i % 2 == 0 else "run",
                "play_id": 100 * wk + i,
            })
        for i in range(10):
            rows.append({
                "season": 2024, "week": wk, "posteam": "KC",
                "play_type": "kickoff",
                "play_id": 100 * wk + 50 + i,
            })
    pbp = _make_pbp_rows(rows)
    out = compute_team_pace(pbp)
    # Week 5's pace_l4 should be 50 (mean of 50 plays/game over wk1-4),
    # not 60 (which would include kickoffs).
    wk5_kc = out.query("team == 'KC' and season == 2024 and week == 5")
    assert len(wk5_kc) == 1
    assert wk5_kc["pace_l4"].iloc[0] == pytest.approx(50.0)


def test_pace_trailing_window_excludes_current_week() -> None:
    """pace_l4 at week W is computed from weeks 1..W-1, not 1..W."""
    from projections.features.pbp_team_features import compute_team_pace

    rows: list[dict[str, object]] = []
    # KC plays 60 plays in wk1, 60 in wk2, 60 in wk3, 60 in wk4, then 100 in wk5.
    for wk, count in [(1, 60), (2, 60), (3, 60), (4, 60), (5, 100)]:
        for i in range(count):
            rows.append({
                "season": 2024, "week": wk, "posteam": "KC",
                "play_type": "pass", "play_id": 1000 * wk + i,
            })
    pbp = _make_pbp_rows(rows)
    out = compute_team_pace(pbp)
    wk5 = out.query("team == 'KC' and season == 2024 and week == 5")
    # Trailing-4 of wk5 = mean(wk1..wk4) = 60, NOT mean(wk2..wk5) = 70.
    assert wk5["pace_l4"].iloc[0] == pytest.approx(60.0)


def test_pace_returns_nan_for_first_4_weeks_when_no_prior_history() -> None:
    """min_periods=4: weeks with fewer than 4 prior games emit NaN."""
    from projections.features.pbp_team_features import compute_team_pace

    rows: list[dict[str, object]] = []
    for wk in range(1, 4):
        for i in range(50):
            rows.append({
                "season": 2024, "week": wk, "posteam": "KC",
                "play_type": "pass", "play_id": 100 * wk + i,
            })
    pbp = _make_pbp_rows(rows)
    out = compute_team_pace(pbp)
    # Wk1 has 0 prior games, wk2 has 1, wk3 has 2 — all NaN under min_periods=4.
    for wk in (1, 2, 3):
        row = out.query(f"team == 'KC' and season == 2024 and week == {wk}")
        assert row["pace_l4"].iloc[0] != row["pace_l4"].iloc[0]  # NaN check
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features/test_pbp_team_features.py -v`
Expected: 3 tests FAIL with `ModuleNotFoundError: No module named 'projections.features.pbp_team_features'`.

- [ ] **Step 3: Implement compute_team_pace**

Create `src/projections/features/pbp_team_features.py`:

```python
"""PBP-derived team-level features for the PBP family probe.

Pure-pandas computes consumed by build_pbp_family_overrides (this module's
public assembler). Each compute returns a (team, season, week, <metric>_l4)
frame with one row per (team, season, week) where the team has a scheduled
game, computed as the rolling mean over the trailing 4 prior games (min 4).

The trailing-4 backfill across season boundaries is handled by the caller
feeding multiple seasons of PBP concat'd together — see
scripts/build_pbp_family_override.py.

Spec: docs/superpowers/specs/2026-04-30-pbp-feature-family-probe-design.md §6.1.
"""

from __future__ import annotations

import pandas as pd

_OFFENSIVE_PLAY_TYPES: frozenset[str] = frozenset({"pass", "run"})


def _trailing_4_mean(per_game: pd.DataFrame, *, value_col: str, out_col: str) -> pd.DataFrame:
    """Rolling-4 mean of value_col per team, shifted so row at week W
    reflects the mean over W-4..W-1 (NOT W). min_periods=4 → fewer than 4
    prior games yield NaN. Input must have columns (team, season, week,
    value_col); output is (team, season, week, out_col).

    Both the rolling AND the shift are within-team (via groupby+transform);
    a global .shift(1) would leak the last row of one team into the first
    row of the next.
    """
    sorted_df = per_game.sort_values(["team", "season", "week"]).reset_index(drop=True)
    rolled = sorted_df.groupby("team", sort=False)[value_col].transform(
        lambda s: s.rolling(window=4, min_periods=4).mean().shift(1)
    )
    sorted_df[out_col] = rolled
    return sorted_df[["team", "season", "week", out_col]]


def compute_team_pace(pbp: pd.DataFrame) -> pd.DataFrame:
    """Team-level offensive plays per game, trailing 4 prior games.

    Plays counted: rows where ``play_type in {'pass', 'run'}``. Excludes
    kickoff / punt / field_goal / no_play. No neutral-script filter — the
    curated PbpSchema lacks ``wp`` / ``qtr`` / ``score_differential``.
    """
    offensive = pbp[pbp["play_type"].isin(_OFFENSIVE_PLAY_TYPES)]
    per_game = (
        offensive.groupby(["posteam", "season", "week"], as_index=False)
        .size()
        .rename(columns={"posteam": "team", "size": "plays"})
    )
    return _trailing_4_mean(per_game, value_col="plays", out_col="pace_l4")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features/test_pbp_team_features.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Lint + type-check**

Run: `ruff check src/projections/features/pbp_team_features.py tests/test_features/test_pbp_team_features.py`
Expected: zero violations.

Run: `mypy src/projections/features/pbp_team_features.py tests/test_features/test_pbp_team_features.py`
Expected: zero violations.

Run: `ruff format --check src/projections/features/pbp_team_features.py tests/test_features/test_pbp_team_features.py`
Expected: no drift.

- [ ] **Step 6: Commit**

```bash
git add src/projections/features/pbp_team_features.py tests/test_features/test_pbp_team_features.py
git commit -m "feat(probe): compute_team_pace + _trailing_4_mean helper

Team offensive plays per game (trailing 4 prior games), no neutral-
script filter (curated PbpSchema lacks wp/qtr/score_differential).
Shared _trailing_4_mean helper used by all four compute fns.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `compute_team_proe` + test (TDD)

**Files:**
- Modify: `src/projections/features/pbp_team_features.py` (append fn)
- Modify: `tests/test_features/test_pbp_team_features.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_features/test_pbp_team_features.py`:

```python
def test_proe_uses_pass_oe_mean_directly() -> None:
    """proe_l4 is the per-team rolling-4 mean of nflfastR's pass_oe column."""
    from projections.features.pbp_team_features import compute_team_proe

    # KC: 4 prior weeks with mean pass_oe = +5.0; week 5 should report +5.0.
    # BAL: 4 prior weeks with mean pass_oe = -3.0; week 5 should report -3.0.
    rows: list[dict[str, object]] = []
    for team, oe in [("KC", 5.0), ("BAL", -3.0)]:
        for wk in range(1, 6):
            for i in range(20):
                rows.append({
                    "season": 2024, "week": wk, "posteam": team,
                    "pass_oe": oe, "play_type": "pass", "play_id": 1000 * wk + i,
                })
    pbp = _make_pbp_rows(rows)
    out = compute_team_proe(pbp)

    kc_wk5 = out.query("team == 'KC' and season == 2024 and week == 5")
    bal_wk5 = out.query("team == 'BAL' and season == 2024 and week == 5")
    assert kc_wk5["proe_l4"].iloc[0] == pytest.approx(5.0)
    assert bal_wk5["proe_l4"].iloc[0] == pytest.approx(-3.0)


def test_proe_drops_nan_pass_oe_rows() -> None:
    """pass_oe NaN (e.g., kickoffs, no-plays) are excluded from the mean."""
    from projections.features.pbp_team_features import compute_team_proe

    rows: list[dict[str, object]] = []
    # KC wk1-4: 10 plays with pass_oe=10.0, plus 90 plays with pass_oe=NaN.
    # Mean over non-NaN should be 10.0.
    for wk in range(1, 6):
        for i in range(10):
            rows.append({
                "season": 2024, "week": wk, "posteam": "KC",
                "pass_oe": 10.0, "play_type": "pass", "play_id": 1000 * wk + i,
            })
        for i in range(90):
            rows.append({
                "season": 2024, "week": wk, "posteam": "KC",
                "pass_oe": float("nan"), "play_type": "kickoff", "play_id": 2000 * wk + i,
            })
    pbp = _make_pbp_rows(rows)
    out = compute_team_proe(pbp)
    wk5 = out.query("team == 'KC' and season == 2024 and week == 5")
    assert wk5["proe_l4"].iloc[0] == pytest.approx(10.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features/test_pbp_team_features.py -k "proe" -v`
Expected: 2 FAIL with `ImportError: cannot import name 'compute_team_proe'`.

- [ ] **Step 3: Implement compute_team_proe**

Append to `src/projections/features/pbp_team_features.py`:

```python
def compute_team_proe(pbp: pd.DataFrame) -> pd.DataFrame:
    """Team-level pass rate over expected, trailing 4 prior games.

    Mean of ``pass_oe`` (nflfastR's pass-over-expected, percentage points)
    across rows where ``posteam == team`` and ``pass_oe`` is non-NaN.
    Upstream's xpass model already game-state-controls the per-play
    pass_oe value, so the per-play mean is itself a properly-controlled
    PROE — no further bucketing required here.
    """
    plays = pbp[pbp["pass_oe"].notna()]
    per_game = (
        plays.groupby(["posteam", "season", "week"], as_index=False)["pass_oe"]
        .mean()
        .rename(columns={"posteam": "team", "pass_oe": "proe"})
    )
    return _trailing_4_mean(per_game, value_col="proe", out_col="proe_l4")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features/test_pbp_team_features.py -k "proe" -v`
Expected: 2 PASS.

Run: `pytest tests/test_features/test_pbp_team_features.py -v`
Expected: full file passes (5 tests so far).

- [ ] **Step 5: Lint + type-check**

Run: `ruff check src/projections/features/pbp_team_features.py tests/test_features/test_pbp_team_features.py`
Run: `mypy src/projections/features/pbp_team_features.py tests/test_features/test_pbp_team_features.py`
Run: `ruff format --check src/projections/features/pbp_team_features.py tests/test_features/test_pbp_team_features.py`
Expected: zero violations on each.

- [ ] **Step 6: Commit**

```bash
git add src/projections/features/pbp_team_features.py tests/test_features/test_pbp_team_features.py
git commit -m "feat(probe): compute_team_proe via pass_oe mean

Team PROE = mean of nflfastR's pass_oe across non-NaN plays, trailing
4 prior games. Upstream xpass model already game-state-controls
pass_oe, so no further bucketing needed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `compute_team_ayps` + test (TDD)

**Files:**
- Modify: `src/projections/features/pbp_team_features.py` (append fn)
- Modify: `tests/test_features/test_pbp_team_features.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_features/test_pbp_team_features.py`:

```python
def test_ayps_only_counts_pass_attempts() -> None:
    """team_ayps_l4 averages air_yards over plays where pass_attempt == 1.0."""
    from projections.features.pbp_team_features import compute_team_ayps

    rows: list[dict[str, object]] = []
    # KC wk1-5: each week, 10 pass attempts at air_yards=15.0 (mean=15.0)
    #          + 50 rushing plays at air_yards=NaN, pass_attempt=0.0.
    for wk in range(1, 6):
        for i in range(10):
            rows.append({
                "season": 2024, "week": wk, "posteam": "KC",
                "pass_attempt": 1.0, "air_yards": 15.0,
                "play_type": "pass", "play_id": 1000 * wk + i,
            })
        for i in range(50):
            rows.append({
                "season": 2024, "week": wk, "posteam": "KC",
                "pass_attempt": 0.0, "air_yards": float("nan"),
                "play_type": "run", "play_id": 2000 * wk + i,
            })
    pbp = _make_pbp_rows(rows)
    out = compute_team_ayps(pbp)
    wk5 = out.query("team == 'KC' and season == 2024 and week == 5")
    assert wk5["team_ayps_l4"].iloc[0] == pytest.approx(15.0)


def test_ayps_drops_nan_air_yards_pass_attempts() -> None:
    """A pass attempt with air_yards=NaN (sack, throw-away) is excluded."""
    from projections.features.pbp_team_features import compute_team_ayps

    rows: list[dict[str, object]] = []
    # 10 valid pass attempts (air_yards=10.0) + 5 NaN-air-yards pass attempts
    # per week. Mean should be 10.0, not (10*10 + 5*0) / 15 = 6.67.
    for wk in range(1, 6):
        for i in range(10):
            rows.append({
                "season": 2024, "week": wk, "posteam": "KC",
                "pass_attempt": 1.0, "air_yards": 10.0,
                "play_type": "pass", "play_id": 1000 * wk + i,
            })
        for i in range(5):
            rows.append({
                "season": 2024, "week": wk, "posteam": "KC",
                "pass_attempt": 1.0, "air_yards": float("nan"),
                "play_type": "pass", "play_id": 2000 * wk + i,
            })
    pbp = _make_pbp_rows(rows)
    out = compute_team_ayps(pbp)
    wk5 = out.query("team == 'KC' and season == 2024 and week == 5")
    assert wk5["team_ayps_l4"].iloc[0] == pytest.approx(10.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features/test_pbp_team_features.py -k "ayps" -v`
Expected: 2 FAIL.

- [ ] **Step 3: Implement compute_team_ayps**

Append to `src/projections/features/pbp_team_features.py`:

```python
def compute_team_ayps(pbp: pd.DataFrame) -> pd.DataFrame:
    """Team-level mean air yards per pass attempt, trailing 4 prior games.

    Plays counted: rows where ``posteam == team``, ``pass_attempt == 1.0``,
    and ``air_yards`` is non-NaN. Sacks and throw-aways have NaN air_yards
    upstream and are excluded from the per-game mean.
    """
    plays = pbp[(pbp["pass_attempt"] == 1.0) & (pbp["air_yards"].notna())]
    per_game = (
        plays.groupby(["posteam", "season", "week"], as_index=False)["air_yards"]
        .mean()
        .rename(columns={"posteam": "team", "air_yards": "ayps"})
    )
    return _trailing_4_mean(per_game, value_col="ayps", out_col="team_ayps_l4")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features/test_pbp_team_features.py -k "ayps" -v`
Expected: 2 PASS.

Run: `pytest tests/test_features/test_pbp_team_features.py -v`
Expected: 7 passing.

- [ ] **Step 5: Lint + type-check**

Run: `ruff check src/projections/features/pbp_team_features.py tests/test_features/test_pbp_team_features.py`
Run: `mypy src/projections/features/pbp_team_features.py tests/test_features/test_pbp_team_features.py`
Run: `ruff format --check src/projections/features/pbp_team_features.py tests/test_features/test_pbp_team_features.py`
Expected: zero violations on each.

- [ ] **Step 6: Commit**

```bash
git add src/projections/features/pbp_team_features.py tests/test_features/test_pbp_team_features.py
git commit -m "feat(probe): compute_team_ayps

Team mean air yards per pass attempt, trailing 4 prior games. Drops
NaN-air-yards pass attempts (sacks, throw-aways) so the mean reflects
only completed downfield throws.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `compute_team_def_epa_residual` + test (TDD)

**Files:**
- Modify: `src/projections/features/pbp_team_features.py` (append fn)
- Modify: `tests/test_features/test_pbp_team_features.py` (append tests)

The residual is: per (defteam, season, game) compute mean defensive EPA-allowed-per-play; then subtract the offensive-opponent's season-average mean-EPA-on-offense (schedule strength). Trailing-4 mean over the residual series per team.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features/test_pbp_team_features.py`:

```python
def test_def_epa_residual_subtracts_schedule_strength() -> None:
    """Two defenses allow identical raw EPA, but BAL faces top offenses
    while CIN faces bottom offenses; BAL's residual should be NEGATIVE
    (allowing less than expected against tough offenses) and CIN's
    POSITIVE."""
    from projections.features.pbp_team_features import compute_team_def_epa_residual

    rows: list[dict[str, object]] = []
    # Construct a 2024 season where:
    #   - KC offense: mean EPA = +0.3 (top tier).
    #   - JAX offense: mean EPA = -0.1 (bottom tier).
    #   - BAL defense: faces KC every week, allows EPA = +0.1 each game (5 games).
    #     Schedule-strength expectation = +0.3; residual = +0.1 - +0.3 = -0.2.
    #   - CIN defense: faces JAX every week, allows EPA = +0.1 each game (5 games).
    #     Schedule-strength expectation = -0.1; residual = +0.1 - -0.1 = +0.2.
    def add_game(season: int, week: int, off: str, defn: str, off_epa: float, def_epa: float) -> None:
        # 10 offensive plays for `off` against `defn` at off_epa; 10 defensive
        # plays for `defn` (which is the same row set) at def_epa.
        # Use TWO distinct sets of plays so off_epa != def_epa is meaningful.
        for i in range(10):
            rows.append({
                "season": season, "week": week,
                "posteam": off, "defteam": defn,
                "epa": off_epa, "play_id": 10000 * week + i,
            })

    # KC offense vs BAL defense, 5 weeks.
    for wk in range(1, 6):
        # KC's actual offensive EPA on these plays = +0.3 (their season tier).
        # But against BAL, plays go for +0.1 (BAL is a tough defense).
        # Resolve by treating "EPA" as DEFENSIVE-perspective EPA-allowed:
        # for the residual fn, we sum/mean EPA per (defteam, season, week),
        # so the rows above are already the def-allowed values. We separately
        # need each offense's *own* season-average EPA — built from rows
        # against a "league-avg" opponent.
        add_game(2024, wk, off="KC", defn="BAL", off_epa=0.0, def_epa=0.1)
    # KC offense vs LV (league-average filler) for season-strength signal.
    for wk in range(11, 16):
        add_game(2024, wk, off="KC", defn="LV", off_epa=0.3, def_epa=0.3)

    # JAX offense vs CIN defense, 5 weeks (CIN allows +0.1 each).
    for wk in range(1, 6):
        add_game(2024, wk, off="JAX", defn="CIN", off_epa=0.0, def_epa=0.1)
    # JAX offense vs LV for season-strength signal.
    for wk in range(11, 16):
        add_game(2024, wk, off="JAX", defn="LV", off_epa=-0.1, def_epa=-0.1)

    # Filler: enough LV games so its own residual isn't tested.
    pbp = _make_pbp_rows(rows)
    out = compute_team_def_epa_residual(pbp)

    bal_late = out.query("team == 'BAL' and season == 2024 and week >= 5")
    cin_late = out.query("team == 'CIN' and season == 2024 and week >= 5")
    # BAL's trailing-4 residual: allowed +0.1 against KC (season EPA +0.3)
    # → residual = +0.1 - +0.3 = -0.2 (4 games' mean).
    assert bal_late["team_def_epa_resid_l4"].iloc[0] == pytest.approx(-0.2, abs=0.01)
    # CIN's: allowed +0.1 against JAX (season EPA -0.1) → +0.2.
    assert cin_late["team_def_epa_resid_l4"].iloc[0] == pytest.approx(+0.2, abs=0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features/test_pbp_team_features.py -k "def_epa" -v`
Expected: 1 FAIL.

- [ ] **Step 3: Implement compute_team_def_epa_residual**

Append to `src/projections/features/pbp_team_features.py`:

```python
def compute_team_def_epa_residual(pbp: pd.DataFrame) -> pd.DataFrame:
    """Defensive EPA-allowed-per-play residual vs offensive-opponent
    season-average EPA, trailing 4 prior games.

    Per (defteam, season, week): mean of ``epa`` across rows where
    ``defteam == team`` and ``epa`` is non-NaN. Per (posteam, season):
    season-average mean of ``epa`` on offense (the opponent's strength
    signal). Per-game residual = (mean def-allowed EPA) - (offensive
    opponent's season-average EPA-on-offense). Then rolling-4 mean of
    the residual series per team, shifted so row at week W reflects the
    last 4 prior games.

    Plain (non-regression) residual: subtracting the opp's season-avg
    EPA from each game's def-allowed EPA gives the "above-or-below
    expected" residual for that game. Same shape as Plan 9's per-position
    EPA-residual but pooled across all plays.

    Output schema: (team, season, week, team_def_epa_resid_l4) where
    `team` is the DEFENSE's team code.
    """
    epa_plays = pbp[pbp["epa"].notna()]

    # Per (defteam, season, week): mean EPA allowed.
    def_per_game = (
        epa_plays.groupby(["defteam", "season", "week"], as_index=False)
        .agg(def_epa_mean=("epa", "mean"), opp=("posteam", "first"))
        .rename(columns={"defteam": "team"})
    )

    # Per (posteam, season): season-average offensive EPA. The
    # opponent's "strength expectation" for any given week.
    off_season_avg = (
        epa_plays.groupby(["posteam", "season"], as_index=False)["epa"]
        .mean()
        .rename(columns={"posteam": "opp", "epa": "opp_season_off_epa"})
    )

    merged = def_per_game.merge(off_season_avg, on=["opp", "season"], how="left")
    merged["resid"] = merged["def_epa_mean"] - merged["opp_season_off_epa"]

    return _trailing_4_mean(
        merged[["team", "season", "week", "resid"]],
        value_col="resid",
        out_col="team_def_epa_resid_l4",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features/test_pbp_team_features.py -k "def_epa" -v`
Expected: 1 PASS.

Run: `pytest tests/test_features/test_pbp_team_features.py -v`
Expected: 8 passing.

- [ ] **Step 5: Lint + type-check**

Run: `ruff check src/projections/features/pbp_team_features.py tests/test_features/test_pbp_team_features.py`
Run: `mypy src/projections/features/pbp_team_features.py tests/test_features/test_pbp_team_features.py`
Run: `ruff format --check src/projections/features/pbp_team_features.py tests/test_features/test_pbp_team_features.py`
Expected: zero violations on each.

- [ ] **Step 6: Commit**

```bash
git add src/projections/features/pbp_team_features.py tests/test_features/test_pbp_team_features.py
git commit -m "feat(probe): compute_team_def_epa_residual

Defensive EPA-allowed residual vs offensive-opponent season-average
EPA, trailing 4 prior games. Plain (non-regression) residual: subtract
opp's season-avg offensive EPA from each game's def-allowed EPA, then
roll the residual.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `build_pbp_family_overrides` assembler + tests

**Files:**
- Modify: `src/projections/features/pbp_team_features.py` (append assembler)
- Modify: `tests/test_features/test_pbp_team_features.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_features/test_pbp_team_features.py`:

```python
def _make_player_team_week_index(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a (gsis_id, season, week, team, opp) frame with GSIS-format IDs."""
    out = pd.DataFrame(rows)
    out["gsis_id"] = out["gsis_id"].astype(pd.StringDtype("pyarrow"))
    out["team"] = out["team"].astype(pd.StringDtype("pyarrow"))
    out["opp"] = out["opp"].astype(pd.StringDtype("pyarrow"))
    out["season"] = out["season"].astype("Int64")
    out["week"] = out["week"].astype("Int64")
    return out


def test_assembler_emits_4_columns_with_correct_join_sides() -> None:
    """pace/proe/team_ayps join on the player's TEAM; team_def_epa_resid
    joins on the player's OPPONENT."""
    from projections.features.pbp_team_features import build_pbp_family_overrides

    # KC plays, BAL plays, with PBP for trailing-4 history.
    pbp_rows: list[dict[str, object]] = []
    for team, oe, ay in [("KC", 5.0, 8.0), ("BAL", -3.0, 6.0)]:
        for wk in range(1, 6):
            for i in range(20):
                pbp_rows.append({
                    "season": 2024, "week": wk, "posteam": team,
                    "defteam": "BAL" if team == "KC" else "KC",
                    "play_type": "pass", "pass_attempt": 1.0,
                    "air_yards": ay, "pass_oe": oe, "epa": 0.1,
                    "play_id": 1000 * wk + i + (50000 if team == "BAL" else 0),
                })

    pbp = _make_pbp_rows(pbp_rows)

    # One player on KC, one on BAL — both at week 5.
    idx = _make_player_team_week_index([
        {"gsis_id": "00-0011111", "season": 2024, "week": 5, "team": "KC", "opp": "BAL"},
        {"gsis_id": "00-0022222", "season": 2024, "week": 5, "team": "BAL", "opp": "KC"},
    ])

    out = build_pbp_family_overrides(pbp, idx)

    assert set(out.columns) == {
        "gsis_id", "season", "week", "pace_l4", "proe_l4",
        "team_ayps_l4", "team_def_epa_resid_l4",
    }
    assert len(out) == 2

    kc_player = out.query("gsis_id == '00-0011111'")
    bal_player = out.query("gsis_id == '00-0022222'")

    # KC player gets KC's offensive features.
    assert kc_player["proe_l4"].iloc[0] == pytest.approx(5.0)
    assert kc_player["team_ayps_l4"].iloc[0] == pytest.approx(8.0)

    # BAL player gets BAL's offensive features.
    assert bal_player["proe_l4"].iloc[0] == pytest.approx(-3.0)
    assert bal_player["team_ayps_l4"].iloc[0] == pytest.approx(6.0)


def test_assembler_rejects_invalid_gsis_id() -> None:
    """build_pbp_family_overrides raises if any input gsis_id violates the
    GSIS_ID_PATTERN."""
    from projections.features.pbp_team_features import build_pbp_family_overrides

    pbp = _make_pbp_rows([{"season": 2024, "week": 1, "posteam": "KC"}])
    idx = _make_player_team_week_index([
        {"gsis_id": "BOGUS_ID", "season": 2024, "week": 1, "team": "KC", "opp": "BAL"},
    ])

    with pytest.raises(ValueError, match="gsis_id"):
        build_pbp_family_overrides(pbp, idx)


def test_assembler_rejects_duplicate_keys() -> None:
    """Two rows with the same (gsis_id, season, week) is a programmer
    error in the index — assembler refuses."""
    from projections.features.pbp_team_features import build_pbp_family_overrides

    pbp = _make_pbp_rows([{"season": 2024, "week": 1, "posteam": "KC"}])
    idx = _make_player_team_week_index([
        {"gsis_id": "00-0011111", "season": 2024, "week": 5, "team": "KC", "opp": "BAL"},
        {"gsis_id": "00-0011111", "season": 2024, "week": 5, "team": "BAL", "opp": "KC"},
    ])

    with pytest.raises(ValueError, match="duplicate"):
        build_pbp_family_overrides(pbp, idx)


def test_assembler_normalizes_team_codes() -> None:
    """Index using JAC (legacy) joins to PBP using JAX (canonical)."""
    from projections.features.pbp_team_features import build_pbp_family_overrides

    # Build PBP with canonical JAX team rows for trailing-4.
    pbp_rows: list[dict[str, object]] = []
    for wk in range(1, 6):
        for i in range(20):
            pbp_rows.append({
                "season": 2024, "week": wk, "posteam": "JAX",
                "defteam": "HOU", "play_type": "pass", "pass_attempt": 1.0,
                "air_yards": 7.0, "pass_oe": 2.0, "epa": 0.05,
                "play_id": 1000 * wk + i,
            })
    pbp = _make_pbp_rows(pbp_rows)

    # Index uses JAC (legacy alias).
    idx = _make_player_team_week_index([
        {"gsis_id": "00-0011111", "season": 2024, "week": 5, "team": "JAC", "opp": "HOU"},
    ])

    out = build_pbp_family_overrides(pbp, idx)
    # Player should pick up JAX's pass_oe mean (2.0) — not NaN.
    assert out["proe_l4"].iloc[0] == pytest.approx(2.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features/test_pbp_team_features.py -k "assembler" -v`
Expected: 4 FAIL with `ImportError: cannot import name 'build_pbp_family_overrides'`.

- [ ] **Step 3: Implement build_pbp_family_overrides**

Append to `src/projections/features/pbp_team_features.py`:

```python
import re

from projections.schemas import GSIS_ID_PATTERN, normalize_team_code

_GSIS_RE = re.compile(rf"^{GSIS_ID_PATTERN}$")


def build_pbp_family_overrides(
    pbp: pd.DataFrame,
    player_team_week_index: pd.DataFrame,
) -> pd.DataFrame:
    """Public assembler. Returns the 4-column override frame ready to write.

    Args:
        pbp: PBP frame matching ``PbpSchema``. Must include the seasons
            spanning the index plus one prior season for trailing-4 backfill.
        player_team_week_index: ``(gsis_id, season, week, team, opp)`` —
            one row per player-week.

    Returns:
        ``(gsis_id, season, week, pace_l4, proe_l4, team_ayps_l4,
        team_def_epa_resid_l4)`` — one row per input index row.

    Raises:
        ValueError: gsis_id format violations, duplicate
            (gsis_id, season, week) keys, unknown team codes after
            ``normalize_team_code``.

    Per-position coverage validation is the probe's responsibility (the
    assembler has no access to the per-position feature parquets); see
    spec §1.3 criterion 1 + §3.3 step 2.
    """
    bad_ids = [g for g in player_team_week_index["gsis_id"].dropna() if not _GSIS_RE.match(str(g))]
    if bad_ids:
        raise ValueError(f"invalid gsis_id format(s): {bad_ids[:3]} (and {max(0, len(bad_ids) - 3)} more)")

    dup_mask = player_team_week_index.duplicated(subset=["gsis_id", "season", "week"], keep=False)
    if dup_mask.any():
        n_dup = int(dup_mask.sum())
        raise ValueError(f"duplicate (gsis_id, season, week) keys in index: {n_dup} rows")

    idx = player_team_week_index.copy()
    # normalize_team_code returns a Team enum; take .value for the string
    # form that PBP's posteam/defteam columns are already keyed by.
    idx["team"] = idx["team"].map(lambda c: normalize_team_code(c).value).astype(pd.StringDtype("pyarrow"))
    idx["opp"] = idx["opp"].map(lambda c: normalize_team_code(c).value).astype(pd.StringDtype("pyarrow"))

    pace = compute_team_pace(pbp)
    proe = compute_team_proe(pbp)
    ayps = compute_team_ayps(pbp)
    def_resid = compute_team_def_epa_residual(pbp)

    out = idx.merge(pace, on=["team", "season", "week"], how="left")
    out = out.merge(proe, on=["team", "season", "week"], how="left")
    out = out.merge(ayps, on=["team", "season", "week"], how="left")
    out = out.merge(
        def_resid.rename(columns={"team": "opp"}),
        on=["opp", "season", "week"],
        how="left",
    )

    return out[
        ["gsis_id", "season", "week", "pace_l4", "proe_l4", "team_ayps_l4", "team_def_epa_resid_l4"]
    ].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features/test_pbp_team_features.py -k "assembler" -v`
Expected: 4 PASS.

Run: `pytest tests/test_features/test_pbp_team_features.py -v`
Expected: 12 passing.

- [ ] **Step 5: Lint + type-check**

Run: `ruff check src/projections/features/pbp_team_features.py tests/test_features/test_pbp_team_features.py`
Run: `mypy src/projections/features/pbp_team_features.py tests/test_features/test_pbp_team_features.py`
Run: `ruff format --check src/projections/features/pbp_team_features.py tests/test_features/test_pbp_team_features.py`
Expected: zero violations on each.

- [ ] **Step 6: Commit**

```bash
git add src/projections/features/pbp_team_features.py tests/test_features/test_pbp_team_features.py
git commit -m "feat(probe): build_pbp_family_overrides assembler

Public assembler for the PBP family override parquet. Validates GSIS-id
format and uniqueness of (gsis_id, season, week) keys; normalizes team
codes via the canonical mapping before joining. Pace/PROE/AYPS join on
player's team; def-EPA-residual joins on opponent.

Per-position coverage validation is the probe's job (this layer has no
access to per-position feature parquets) — see spec §1.3 + §3.3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: `scripts/build_pbp_family_override.py` CLI

**Files:**
- Create: `scripts/build_pbp_family_override.py`

No new tests for the CLI itself — it's argparse glue around already-tested code. Real-data correctness is verified at run-time in Task 8.

- [ ] **Step 1: Write the script**

Create `scripts/build_pbp_family_override.py`:

```python
"""Build the PBP family override parquet for the family probe.

One-shot CLI. Loads PBP / weekly_stats / schedules across the requested
season range (plus the prior season for trailing-4 backfill at week 1-4),
calls build_pbp_family_overrides, writes the resulting frame to a parquet.

Output is NOT committed — it's regenerable from the live raw partitions.

Usage:
    python -m scripts.build_pbp_family_override --seasons 2018-2024
    python -m scripts.build_pbp_family_override --seasons 2018-2024 --force
    python -m scripts.build_pbp_family_override --output data/features_probe/x.parquet

Spec: docs/superpowers/specs/2026-04-30-pbp-feature-family-probe-design.md §6.2.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from projections.features.pbp_team_features import build_pbp_family_overrides
from projections.store import read_partition

_DEFAULT_OUTPUT = Path("data/features_probe/pbp_family.parquet")


def _parse_season_range(s: str) -> range:
    """`'2018-2024'` -> `range(2018, 2025)`; `'2024'` -> `range(2024, 2025)`."""
    if "-" in s:
        lo_s, hi_s = s.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
        return range(lo, hi + 1)
    n = int(s)
    return range(n, n + 1)


def _read_concat(raw_root: Path, table: str, seasons: Sequence[int]) -> pd.DataFrame:
    """Read one partition per season and concat. Skip seasons without a partition."""
    frames: list[pd.DataFrame] = []
    for s in seasons:
        try:
            frames.append(read_partition(raw_root, table, season=s))
        except FileNotFoundError:
            pass
    if not frames:
        raise FileNotFoundError(f"no partitions found for table={table!r} in seasons={list(seasons)}")
    return pd.concat(frames, ignore_index=True)


def _build_player_team_week_index(
    weekly_stats: pd.DataFrame, schedules: pd.DataFrame, seasons: range
) -> pd.DataFrame:
    """Inner-join weekly_stats and schedules to produce
    (gsis_id, season, week, team, opp). Restrict to the target season range
    so the override is keyed only to the seasons being probed."""
    ws = weekly_stats[weekly_stats["season"].isin(seasons)][
        ["gsis_id", "season", "week", "team"]
    ]
    sch = schedules[schedules["season"].isin(seasons)][
        ["season", "week", "home_team", "away_team"]
    ]
    home = sch.rename(columns={"home_team": "team", "away_team": "opp"})
    away = sch.rename(columns={"away_team": "team", "home_team": "opp"})
    team_opp = pd.concat([home, away], ignore_index=True)[["season", "week", "team", "opp"]]
    return ws.merge(team_opp, on=["season", "week", "team"], how="inner")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--seasons", type=_parse_season_range, default=range(2018, 2025),
        help="Season range, e.g. '2018-2024' or '2024'. Default: 2018-2024.",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="Root for raw and features partitions. Default: data.",
    )
    parser.add_argument(
        "--output", type=Path, default=_DEFAULT_OUTPUT,
        help=f"Override output parquet path. Default: {_DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite the output if it already exists.",
    )
    args = parser.parse_args(argv)

    if args.output.exists() and not args.force:
        parser.error(f"{args.output} exists; pass --force to overwrite.")

    seasons: range = args.seasons
    raw_root = args.data_root / "raw"
    pbp_seasons = range(seasons.start - 1, seasons.stop)  # +1 prior for backfill

    pbp = _read_concat(raw_root, "pbp", list(pbp_seasons))
    weekly_stats = _read_concat(raw_root, "weekly_stats", list(seasons))
    schedules = _read_concat(raw_root, "schedules", list(seasons))

    idx = _build_player_team_week_index(weekly_stats, schedules, seasons)
    overrides = build_pbp_family_overrides(pbp, idx)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    overrides.to_parquet(args.output, index=False)
    print(f"wrote {len(overrides)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-test the CLI's `--help`**

Run: `python -m scripts.build_pbp_family_override --help`
Expected: usage block prints; exit code 0.

- [ ] **Step 3: Lint + type-check**

Run: `ruff check scripts/build_pbp_family_override.py`
Run: `mypy scripts/build_pbp_family_override.py`
Run: `ruff format --check scripts/build_pbp_family_override.py`
Expected: zero violations on each.

- [ ] **Step 4: Run the full test suite to confirm no regressions**

Run: `pytest -v`
Expected: all passing.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_pbp_family_override.py
git commit -m "feat(probe): scripts/build_pbp_family_override CLI

Argparse glue around build_pbp_family_overrides. Loads pbp/weekly_stats/
schedules via read_partition (matches scripts/refresh_features.py:88
pattern), builds player-team-week index from weekly_stats+schedules,
calls assembler, writes parquet. Output not committed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Generate the override parquet (run-time, not committed)

**Files:**
- Run-time output: `data/features_probe/pbp_family.parquet` (NOT committed)

This is a real-data step. The output parquet is regenerable; it lives under `data/features_probe/` and is `.gitignore`'d implicitly (the `data/` tree is per probe spec convention).

- [ ] **Step 1: Verify raw partitions exist**

Run: `ls data/raw/pbp/season=2024/ data/raw/weekly_stats/season=2024/ data/raw/schedules/season=2024/`
Expected: each shows `part.parquet`. If any is missing, run `python -c "from projections.ingest.refresh import refresh; from pathlib import Path; refresh(data_root=Path('data'), seasons=range(2017, 2025))"` (per TODO #18 — no `python -m` entry point yet).

- [ ] **Step 2: Run the override builder**

Run: `python -m scripts.build_pbp_family_override --seasons 2018-2024`
Expected: prints `wrote N rows to data/features_probe/pbp_family.parquet` for some N in roughly the 60k-100k range. Exit code 0.

- [ ] **Step 3: Spot-check the parquet**

Run:
```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/features_probe/pbp_family.parquet')
print('rows:', len(df))
print('seasons:', sorted(df['season'].unique()))
print('cols:', list(df.columns))
print('non-null fractions:')
print((1 - df[['pace_l4','proe_l4','team_ayps_l4','team_def_epa_resid_l4']].isna().mean()).round(3))
"
```
Expected: 4 columns; seasons 2018-2024; non-null fractions ≥ 0.95 on each of the 4 columns. If any < 0.95 for the full 2018-2024 range — look at per-season breakdown:
```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/features_probe/pbp_family.parquet')
print(df.groupby('season')[['pace_l4','proe_l4','team_ayps_l4','team_def_epa_resid_l4']].apply(lambda g: (1 - g.isna().mean()).round(3)))
"
```
Expected: 2018 may show < 0.95 on early-season weeks (no 2017 PBP partition, per spec §2.3 step 3). Other seasons should be ≥ 0.95.

If 2019-2024 are < 0.95 across all weeks: investigate before continuing — either a backfill bug or an upstream data gap. The probe coverage check will reject the probe run otherwise.

- [ ] **Step 4: NO commit (output not committed)**

Skip — output is regenerable, not in git.

---

### Task 9: Run baseline augment probe + commit reports

**Files:**
- Run-time output: `reports/feature_probe_pbp_family_augment_{QB,RB,WR,TE}.{md,csv}` — 8 committed files.

- [ ] **Step 1: Run the probe (~ minutes)**

Run:
```bash
python -m scripts.probe_feature_signal \
  --candidate-name pbp_family_augment \
  --model baseline \
  --override data/features_probe/pbp_family.parquet \
  --csv-out reports/feature_probe_pbp_family_augment.csv
```
Expected: 4 markdown blocks printed to stdout (one per position) + `reports/feature_probe_pbp_family_augment_{QB,RB,WR,TE}.{md,csv}` written. Exit code 0. If the probe rejects with "OverrideCoverageError," return to Task 8 — the override has < 95% coverage on at least one (position, season) pair.

- [ ] **Step 2: Commit the reports**

```bash
git add reports/feature_probe_pbp_family_augment_*.md reports/feature_probe_pbp_family_augment_*.csv
git commit -m "chore(probe): pbp-family baseline augment probe — 4 positions

Reports for the augment-mode (no drops) baseline probe of the four PBP
features. Per-position markdown + CSV under reports/.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Run baseline swap probe + commit reports

**Files:**
- Run-time output: `reports/feature_probe_pbp_family_swap_{QB,RB,WR,TE}.{md,csv}` — 8 committed files.

- [ ] **Step 1: Run the probe (~ minutes)**

Run:
```bash
python -m scripts.probe_feature_signal \
  --candidate-name pbp_family_swap \
  --model baseline \
  --override data/features_probe/pbp_family.parquet \
  --drop opp_allowed_qb_fppg_l4,opp_allowed_rb_fppg_l4,opp_allowed_wr_fppg_l4,opp_allowed_te_fppg_l4 \
  --csv-out reports/feature_probe_pbp_family_swap.csv
```
Expected: 4 markdown blocks + 8 files under `reports/`. Exit code 0.

- [ ] **Step 2: Commit the reports**

```bash
git add reports/feature_probe_pbp_family_swap_*.md reports/feature_probe_pbp_family_swap_*.csv
git commit -m "chore(probe): pbp-family baseline swap probe — 4 positions

Reports for the swap-mode (drops opp_allowed_*_fppg_l4) baseline probe
of the four PBP features. Per-position markdown + CSV under reports/.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Compute baseline-only family verdict; branch SIGNAL or NULL

This is a decision step, not a code change.

- [ ] **Step 1: Inspect both reports' Phase-1 + Phase-2 verdicts**

Run:
```bash
python -c "
import pandas as pd
for cand in ['pbp_family_augment', 'pbp_family_swap']:
    print(f'=== {cand} ===')
    for pos in ['QB','RB','WR','TE']:
        df = pd.read_csv(f'reports/feature_probe_{cand}_{pos}.csv')
        pooled = df[df['year_or_pooled'] == 'pooled']
        signal = (pooled['verdict'] == 'SIGNAL').sum()
        regression = (pooled['verdict'] == 'REGRESSION').sum()
        composite = df[df['phase'] == 'composite']
        c_verdict = composite['verdict'].iloc[0] if len(composite) else 'skipped'
        print(f'  {pos}: pooled SIGNAL={signal} REGRESSION={regression} composite={c_verdict}')
"
```

- [ ] **Step 2: Determine the verdict**

Apply the §4 rule manually from the printout above:
- Family is `SIGNAL` if any line shows `pooled SIGNAL >= 1` OR `composite=ADOPT/MARGINAL`.
- Family is `NULL` (so far) if every line shows `pooled SIGNAL=0` AND `composite=DO_NOT_ADOPT/skipped`.

- [ ] **Step 3: Decide the branch**

- **If SIGNAL:** skip Tasks 12 + 13. Go to Task 14 (write summary report); the lgb-nb conditional runs are not required.
- **If NULL:** continue to Task 12 + 13 (lgb-nb conditional probes); then Task 14.

---

### Task 12 (conditional, NULL path only): Run lgb-nb augment probe

**Skip this task if Task 11 returned SIGNAL.**

- [ ] **Step 1: Run the probe (~ 1-2 hr)**

Run:
```bash
python -m scripts.probe_feature_signal \
  --candidate-name pbp_family_lgbnb_augment \
  --model lightgbm-nb \
  --override data/features_probe/pbp_family.parquet \
  --csv-out reports/feature_probe_pbp_family_lgbnb_augment.csv
```
Expected: 4 markdown blocks + 8 files under `reports/`. Slow — runtime ≈ 1-2 hr per probe spec §8.

- [ ] **Step 2: Commit the reports**

```bash
git add reports/feature_probe_pbp_family_lgbnb_augment_*.md reports/feature_probe_pbp_family_lgbnb_augment_*.csv
git commit -m "chore(probe): pbp-family lgb-nb augment probe — 4 positions

Conditional run: baseline returned NULL × all, so lgb-nb augment fires
to test the model-class hypothesis at the family level (TODO #3c).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 13 (conditional, NULL path only): Run lgb-nb swap probe

**Skip this task if Task 11 returned SIGNAL.**

- [ ] **Step 1: Run the probe (~ 1-2 hr)**

Run:
```bash
python -m scripts.probe_feature_signal \
  --candidate-name pbp_family_lgbnb_swap \
  --model lightgbm-nb \
  --override data/features_probe/pbp_family.parquet \
  --drop opp_allowed_qb_fppg_l4,opp_allowed_rb_fppg_l4,opp_allowed_wr_fppg_l4,opp_allowed_te_fppg_l4 \
  --csv-out reports/feature_probe_pbp_family_lgbnb_swap.csv
```
Expected: 4 markdown blocks + 8 files under `reports/`. Slow.

- [ ] **Step 2: Re-run the verdict inspection across all 4 reports**

Run the script from Task 11 step 1, but extend the candidate list:

```bash
python -c "
import pandas as pd
for cand in ['pbp_family_augment', 'pbp_family_swap', 'pbp_family_lgbnb_augment', 'pbp_family_lgbnb_swap']:
    print(f'=== {cand} ===')
    for pos in ['QB','RB','WR','TE']:
        df = pd.read_csv(f'reports/feature_probe_{cand}_{pos}.csv')
        pooled = df[df['year_or_pooled'] == 'pooled']
        signal = (pooled['verdict'] == 'SIGNAL').sum()
        regression = (pooled['verdict'] == 'REGRESSION').sum()
        composite = df[df['phase'] == 'composite']
        c_verdict = composite['verdict'].iloc[0] if len(composite) else 'skipped'
        print(f'  {pos}: pooled SIGNAL={signal} REGRESSION={regression} composite={c_verdict}')
"
```

- [ ] **Step 3: Commit the reports**

```bash
git add reports/feature_probe_pbp_family_lgbnb_swap_*.md reports/feature_probe_pbp_family_lgbnb_swap_*.csv
git commit -m "chore(probe): pbp-family lgb-nb swap probe — 4 positions

Conditional run: lgb-nb swap closes out the conditional matrix per
spec §3.2 / §1.3 criterion 3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: Write the family summary report

**Files:**
- Create: `reports/feature_probe_pbp_family_summary.md`

This is hand-written narrative + a machine-rendered table. The verdict line is populated from the printout in Task 11 (or Task 13 step 2).

- [ ] **Step 1: Generate the per-mode summary table**

Run:
```bash
python -c "
import pandas as pd

candidates = ['pbp_family_augment', 'pbp_family_swap']
# Add lgb-nb only if those reports exist:
import os
for c in ['pbp_family_lgbnb_augment', 'pbp_family_lgbnb_swap']:
    if os.path.exists(f'reports/feature_probe_{c}_QB.csv'):
        candidates.append(c)

print('| Model | Mode | Pos | Pooled SIGNAL | Pooled REGRESSION | Composite | Best ΔRMSE (fpts) |')
print('|---|---|---|---:|---:|---|---:|')
for cand in candidates:
    model = 'lightgbm-nb' if 'lgbnb' in cand else 'baseline'
    mode = 'swap' if 'swap' in cand else 'augment'
    for pos in ['QB','RB','WR','TE']:
        df = pd.read_csv(f'reports/feature_probe_{cand}_{pos}.csv')
        pooled = df[df['year_or_pooled'] == 'pooled']
        signal = int((pooled['verdict'] == 'SIGNAL').sum())
        regression = int((pooled['verdict'] == 'REGRESSION').sum())
        composite = df[df['phase'] == 'composite']
        c_verdict = composite['verdict'].iloc[0] if len(composite) else 'skipped'
        best = pooled['rmse_delta_point'].min() if len(pooled) else float('nan')
        print(f'| {model} | {mode} | {pos} | {signal} | {regression} | {c_verdict} | {best:+.4f} |')
" > /tmp/summary_table.md
cat /tmp/summary_table.md
```

- [ ] **Step 2: Hand-write `reports/feature_probe_pbp_family_summary.md`**

Use this template. **Fill in the `<verdict>` and the conclusion paragraph based on the actual reports**:

```markdown
# PBP Feature Family Probe — Summary

**Date:** 2026-04-30
**Branch:** feat/probe-pbp-family
**Spec:** docs/superpowers/specs/2026-04-30-pbp-feature-family-probe-design.md
**Override:** data/features_probe/pbp_family.parquet (regenerable; not committed)

The four PBP-derived features `pace_l4`, `proe_l4`, `team_ayps_l4`,
`team_def_epa_resid_l4` were bundled into a single override and probed
in {2 or 4} configurations against the v1 baseline + (in swap mode)
with the v1 `opp_allowed_*_fppg_l4` columns dropped. Family verdict is
the §4 rule applied across the executed reports.

## Per-mode summary

<paste the table from Step 1 here>

## Family verdict

**<SIGNAL or NULL>** (computed by `family_verdict_from_reports`).

<If SIGNAL: name the (model, mode, position, stat) tuples that lit up
and what the candidate production-builder plan would scope. Cite the
specific SIGNAL cells.>

<If NULL: state the family is closed across {baseline / baseline + lgb-nb}
at the family level for the four-feature bundle. Note any REGRESSION
cells for the future production-builder spec to consider.>

## Decision

<Two-to-three sentences:
- What this verdict greenlights or closes.
- What the next plan is (or "no next plan; family closed").
- Cross-reference to TODO #3c update.>
```

- [ ] **Step 3: Commit the summary**

```bash
git add reports/feature_probe_pbp_family_summary.md
git commit -m "docs(probe): pbp-family probe summary — verdict <SIGNAL|NULL>

Family-level verdict across the {2|4} probe runs. <One-line conclusion
per the spec §4 rule.>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 15: Update `TODO.md` #3c + `project_management.md`

**Files:**
- Modify: `TODO.md` (append to entry #3c)
- Modify: `project_management.md` (append a decision-log entry)

- [ ] **Step 1: Append to TODO #3c**

Open `TODO.md` and locate the `### 3c. Remaining PBP-derived feature plans (open)` heading. Append a paragraph immediately after the existing **Update 2026-04-30 (option C re-evaluation)** paragraph:

```markdown
**Update 2026-04-30 (PBP family probe, branch `feat/probe-pbp-family`):**
The four-feature PBP family (`pace_l4`, `proe_l4`, `team_ayps_l4`,
`team_def_epa_resid_l4`) was probed in {augment + swap}{ × baseline ± lgb-nb}
modes per spec `docs/superpowers/specs/2026-04-30-pbp-feature-family-probe-design.md`.
Family verdict: **<SIGNAL or NULL>**. <Brief consequence — either name
the production-builder follow-up plan or note the family closure.> See
`reports/feature_probe_pbp_family_summary.md` for the per-mode table
and decision log.
```

- [ ] **Step 2: Append to project_management.md**

Append to the decision-log section of `project_management.md` (matching the format of prior Plan-N entries):

```markdown
## Plan: PBP Feature Family Probe — 2026-04-30

**Branch:** feat/probe-pbp-family
**Spec:** docs/superpowers/specs/2026-04-30-pbp-feature-family-probe-design.md
**Plan:** docs/superpowers/plans/2026-04-30-pbp-feature-family-probe.md
**Verdict:** <SIGNAL or NULL>
**What this greenlights/closes:** <One paragraph. If SIGNAL: production
builder follow-up scoping per-position units (player-aDOT for receivers,
per-position EPA-residual à la Plan 9); cite the SIGNAL cells. If NULL:
the four-feature PBP family at the BaselineModel{ + lgb-nb} level is
closed; cross-reference to spec §1.2 deferred PROE-model-fitting and
neutral-script pace as remaining open extensions if independent
evidence emerges.>

Reports: `reports/feature_probe_pbp_family_summary.md` + 16-32
per-position .md/.csv files.
```

- [ ] **Step 3: Commit**

```bash
git add TODO.md project_management.md
git commit -m "docs(probe): TODO #3c + project_management decision-log

Records the pbp-family probe verdict and what it greenlights/closes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 16: Update `CONTRIBUTING.md`

**Files:**
- Modify: `CONTRIBUTING.md` — add "Regenerating the PBP family override" subsection.

- [ ] **Step 1: Locate the existing feature-plan workflow section**

Run: `grep -n "feature plan workflow" CONTRIBUTING.md`
Expected: at least one match (the section added in PR #18 per spec §9).

- [ ] **Step 2: Append a subsection**

Below the existing feature-plan workflow paragraph, append:

```markdown
#### Regenerating the PBP family override

The PBP family probe (spec `docs/superpowers/specs/2026-04-30-pbp-feature-family-probe-design.md`)
consumes a four-column override parquet at `data/features_probe/pbp_family.parquet`.
The output is regenerable from the live PBP partitions and is not
committed. To regenerate:

```bash
python -m scripts.build_pbp_family_override --seasons 2018-2024
```

Pass `--force` to overwrite an existing parquet. The script needs
`data/raw/pbp/`, `data/raw/weekly_stats/`, and `data/raw/schedules/`
populated (run the standard ingest refresh first if not).
```

- [ ] **Step 3: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs(probe): CONTRIBUTING — regenerating the PBP family override

One-paragraph reproducibility note for the not-committed override
parquet that the family probe consumes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 17: End-of-effort verification gates

**No file changes** — runs the CLAUDE.md checklist. Fix any failure before opening the PR.

- [ ] **Step 1: Full pytest**

Run: `pytest -v`
Expected: all passing (no skips, no errors).

- [ ] **Step 2: Mypy strict**

Run: `mypy src tests`
Expected: zero violations.

- [ ] **Step 3: Ruff lint**

Run: `ruff check src tests scripts`
Expected: zero violations.

- [ ] **Step 4: Ruff format**

Run: `ruff format --check src tests scripts`
Expected: no drift.

- [ ] **Step 5: Ingest/store/schemas integration gate**

Run: `pytest -v -k "ingest or store or schemas"`
Expected: all passing. (This spec touches `src/projections/features/pbp_team_features.py` which is downstream of ingest, not ingest itself, but the gate is cheap and runs anyway per CLAUDE.md.)

- [ ] **Step 6: Final commit if any drift was fixed**

If any of steps 1-5 surfaced fixable issues, fix them and commit:

```bash
git add <files>
git commit -m "chore(probe): end-of-effort verification gate fixes

<one-line description of what was fixed>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

If all 5 steps were green from the start: nothing to commit.

- [ ] **Step 7: Push the branch**

```bash
git push -u origin feat/probe-pbp-family
```

- [ ] **Step 8: Open PR**

```bash
gh pr create --title "feat(probe): PBP feature family probe (pace + PROE + air-yards + EPA-residual)" --body "$(cat <<'EOF'
## Summary
- Bundles 4 PBP-derived team-level features into a single override parquet and screens them at family-level granularity per spec `2026-04-30-pbp-feature-family-probe-design.md`.
- Adds 4 pure compute fns + assembler in `src/projections/features/pbp_team_features.py`, a new `family_verdict_from_reports` helper, and a `scripts/build_pbp_family_override.py` CLI.
- Family verdict: **<SIGNAL or NULL>** — see `reports/feature_probe_pbp_family_summary.md`.

## Test plan
- [x] All synthetic-fixture tests pass (`pytest -v`)
- [x] mypy strict clean
- [x] ruff lint + format clean
- [x] Real-data probe runs committed under `reports/`
- [x] Decision log updated in `TODO.md` #3c + `project_management.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review

(Done inline by the planner before handoff. Spec coverage: each spec section is implemented by Tasks 1–17. Placeholder scan: no TBDs remain. Type consistency: helper signature in Task 1 matches §6.3; assembler signature in Task 6 matches §6.1; CLI shape in Task 7 matches §6.2.)
