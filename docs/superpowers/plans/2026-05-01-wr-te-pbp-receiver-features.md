# WR/TE PBP Receiver Family Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the receiver-level PBP feature module + override generator + run the WR/TE family probe matrix; emit the family verdict (`SIGNAL` or `NULL`) per `family_verdict_from_reports`; commit the verdict-line summary report and decision-log entry. Probe-only — production builder is a follow-up plan if `SIGNAL`.

**Architecture:** Mirror PR #20's team-level pattern (`pbp_team_features.py` + `scripts/build_pbp_family_override.py`) one-for-one in a new `pbp_receiver_features.py` module + `scripts/build_pbp_receiver_override.py`. Four pure compute functions (one per candidate feature) + an as-of trailing-4 helper + an assembler + a validating wrapper. The receiver index comes from `depth_charts` (matching the WR/TE baseline-feature row coverage). The probe CLI is reused unchanged with `--position WR --position TE` to scope to receivers; `family_verdict_from_reports` is reused unchanged.

**Tech Stack:** Python 3.12, pandas (with pyarrow string dtype), pandera (schema validation), pyarrow (parquet), pytest, mypy strict, ruff. Existing infrastructure: `projections.store.read_partition`, `projections.schemas.PbpSchema`, `projections.backtest.feature_probe.family_verdict_from_reports`, `scripts.probe_feature_signal`.

---

## File Structure

**Created:**

- `src/projections/features/pbp_receiver_features.py` — 4 pure compute fns (`compute_receiver_adot`, `compute_receiver_deep_target_share`, `compute_receiver_yac_per_reception`, `compute_receiver_red_zone_target_share`) + `_trailing_4_per_player_asof` helper + `attach_pbp_receiver_features` assembler + `build_pbp_receiver_overrides` validating wrapper.
- `tests/test_features/test_pbp_receiver_features.py` — 11 synthetic-PBP tests covering each compute, the assembler, and the validating wrapper.
- `scripts/build_pbp_receiver_override.py` — argparse + I/O glue that loads PBP + depth_charts, builds the receiver index, calls `build_pbp_receiver_overrides`, writes `data/features_probe/pbp_receiver.parquet`.
- `reports/feature_probe_pbp_receiver_augment_{WR,TE}.{md,csv}` — baseline augment-mode probe reports (4 files).
- `reports/feature_probe_pbp_receiver_swap_{WR,TE}.{md,csv}` — baseline swap-mode probe reports (4 files).
- `reports/feature_probe_pbp_receiver_lgbnb_augment_{WR,TE}.{md,csv}` — conditional lgb-nb augment reports (4 files; only if both baseline modes return NULL).
- `reports/feature_probe_pbp_receiver_lgbnb_swap_{WR,TE}.{md,csv}` — conditional lgb-nb swap reports (4 files; same trigger).
- `reports/feature_probe_pbp_receiver_summary.md` — family-level decision log (always).

**Modified:**

- `CONTRIBUTING.md` — add a one-paragraph "Regenerating the PBP receiver override" subsection.
- `TODO.md` — append family verdict outcome to entry #3c.
- `project_management.md` — append top-of-file "PBP Receiver Family Probe" decision-log entry.

**Generated, NOT committed:**

- `data/features_probe/pbp_receiver.parquet` — regenerable from PBP partitions.

---

## Phase 1 — Pure-function module (TDD per compute)

### Task 1: Module skeleton + as-of trailing-4 helper

**Files:**
- Create: `src/projections/features/pbp_receiver_features.py`
- Test: `tests/test_features/test_pbp_receiver_features.py`

**Why an as-of helper, not a copy of `_trailing_4_mean`?** The team-level helper assumes `per_game` has a row at every `(team, season, week)` the team played — which is true (teams play every week modulo bye). For receivers, `per_game` only has rows at *receiver-active* games (weeks where the player had at least 1 target). The override index, however, is keyed on `(gsis_id, season, week)` from `depth_charts` — which has rows for rostered-but-inactive players too. So we need to attach values from `per_game` to a wider index via an as-of merge: for each index row `(g, s, w)`, take the trailing-4 value at the most recent receiver-active game strictly before `(s, w)`.

- [ ] **Step 1: Write the failing test for the as-of helper**

Add to `tests/test_features/test_pbp_receiver_features.py`:

```python
"""PBP receiver-level feature computes — tests."""

from __future__ import annotations

import pandas as pd
import pytest


def _make_pbp_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a synthetic PBP frame with sane defaults for unspecified columns.

    The compute fns only read a subset of PbpSchema columns; tests fill in
    only the columns the function under test uses, default the rest. Mirrors
    `tests/test_features/test_pbp_team_features.py::_make_pbp_rows`.
    """
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
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_trailing_4_asof_within_player_no_leakage() -> None:
    """Rolling-4 stays within gsis_id; row N for player A doesn't see player B."""
    from projections.features.pbp_receiver_features import _trailing_4_per_player_asof

    # per_game: 5 rows for A (values 10..14), 5 rows for B (values 100..104).
    per_game = pd.DataFrame(
        [
            {"gsis_id": "00-0000001", "season": 2024, "week": w, "val": 9 + w}
            for w in range(1, 6)
        ]
        + [
            {"gsis_id": "00-0000002", "season": 2024, "week": w, "val": 99 + w}
            for w in range(1, 6)
        ]
    )
    # index: one row per (player, season, week) for both players, weeks 1-6.
    index = pd.DataFrame(
        [
            {"gsis_id": gid, "season": 2024, "week": w}
            for gid in ("00-0000001", "00-0000002")
            for w in range(1, 7)
        ]
    )
    out = _trailing_4_per_player_asof(per_game, index, value_col="val", out_col="val_l4")

    # Player A, week 5: trailing-4 at most-recent receiver-active game strictly
    # before (2024, w5) is row at (2024, w4) with rolling = mean(w1..w4 vals
    # 10,11,12,13) = 11.5.
    a_w5 = out.query("gsis_id == '00-0000001' and season == 2024 and week == 5")
    assert a_w5["val_l4"].iloc[0] == pytest.approx(11.5)

    # Player B, week 5: rolling-4 at (2024, w4) for B is mean(100,101,102,103)
    # = 101.5. NOT contaminated by A's values.
    b_w5 = out.query("gsis_id == '00-0000002' and season == 2024 and week == 5")
    assert b_w5["val_l4"].iloc[0] == pytest.approx(101.5)


def test_trailing_4_asof_inactive_week_uses_last_active() -> None:
    """For an index row where the player wasn't receiver-active, value comes
    from the last receiver-active game strictly before (season, week)."""
    from projections.features.pbp_receiver_features import _trailing_4_per_player_asof

    # Player active in weeks 1, 2, 3, 4, 5, 7 (skipped week 6).
    per_game = pd.DataFrame(
        [
            {"gsis_id": "00-0000001", "season": 2024, "week": w, "val": float(w)}
            for w in (1, 2, 3, 4, 5, 7)
        ]
    )
    # Index has rostered weeks 1-8; week 6 is a non-active week for the player.
    index = pd.DataFrame(
        [{"gsis_id": "00-0000001", "season": 2024, "week": w} for w in range(1, 9)]
    )
    out = _trailing_4_per_player_asof(per_game, index, value_col="val", out_col="val_l4")

    # Week 6 (inactive): last active game strictly before w6 is w5.
    # rolling-4 at w5 = mean(2, 3, 4, 5) = 3.5. So index w6 gets 3.5.
    w6 = out.query("week == 6")
    assert w6["val_l4"].iloc[0] == pytest.approx(3.5)

    # Week 7 (active again): last active game strictly before w7 is w5.
    # Same value as w6: 3.5.
    w7 = out.query("week == 7")
    assert w7["val_l4"].iloc[0] == pytest.approx(3.5)

    # Week 8: last active game strictly before w8 is w7. rolling-4 at w7 =
    # mean of 4 most recent receiver-active games up to and including w7 =
    # mean(3, 4, 5, 7) = 4.75.
    w8 = out.query("week == 8")
    assert w8["val_l4"].iloc[0] == pytest.approx(4.75)


def test_trailing_4_asof_min_periods_4() -> None:
    """Fewer than 4 prior receiver-active games yield NaN."""
    from projections.features.pbp_receiver_features import _trailing_4_per_player_asof

    # Player active in only weeks 1, 2, 3 (3 games, not 4).
    per_game = pd.DataFrame(
        [
            {"gsis_id": "00-0000001", "season": 2024, "week": w, "val": float(w)}
            for w in (1, 2, 3)
        ]
    )
    index = pd.DataFrame(
        [{"gsis_id": "00-0000001", "season": 2024, "week": w} for w in range(1, 6)]
    )
    out = _trailing_4_per_player_asof(per_game, index, value_col="val", out_col="val_l4")

    # All output values should be NaN — never enough prior receiver-active games.
    assert out["val_l4"].isna().all()


def test_trailing_4_asof_cross_season() -> None:
    """Trailing-4 wraps across season boundary if Y has fewer than 4 active games."""
    from projections.features.pbp_receiver_features import _trailing_4_per_player_asof

    # Player active in 2023 weeks 16, 17 and 2024 weeks 1, 2, 3.
    per_game = pd.DataFrame(
        [
            {"gsis_id": "00-0000001", "season": 2023, "week": 16, "val": 16.0},
            {"gsis_id": "00-0000001", "season": 2023, "week": 17, "val": 17.0},
            {"gsis_id": "00-0000001", "season": 2024, "week": 1, "val": 1.0},
            {"gsis_id": "00-0000001", "season": 2024, "week": 2, "val": 2.0},
            {"gsis_id": "00-0000001", "season": 2024, "week": 3, "val": 3.0},
        ]
    )
    index = pd.DataFrame(
        [{"gsis_id": "00-0000001", "season": 2024, "week": w} for w in range(1, 5)]
    )
    out = _trailing_4_per_player_asof(per_game, index, value_col="val", out_col="val_l4")

    # 2024 w4: latest active game strictly before is 2024 w3. rolling-4 at w3 =
    # mean of 4 most recent up-to-and-including w3 = mean(2023 w17, 2024 w1,
    # 2024 w2, 2024 w3) = mean(17, 1, 2, 3) = 5.75.
    w4 = out.query("season == 2024 and week == 4")
    assert w4["val_l4"].iloc[0] == pytest.approx(5.75)
```

- [ ] **Step 2: Run test to verify failures (module/helper don't exist yet)**

Run: `pytest tests/test_features/test_pbp_receiver_features.py -v`
Expected: 4 ImportError or AttributeError failures referencing `pbp_receiver_features` or `_trailing_4_per_player_asof`.

- [ ] **Step 3: Write the module skeleton + helper**

Create `src/projections/features/pbp_receiver_features.py`:

```python
"""PBP-derived player-level (receiver) features for the WR/TE PBP family probe.

Pure-pandas computes consumed by build_pbp_receiver_overrides (this module's
public assembler). Each compute returns a (gsis_id, season, week, <metric>_l4)
frame attached to a depth-chart-derived (gsis_id, season, week) index via an
as-of trailing-4 lookup over the player's prior receiver-active games.

The trailing-4 backfill across season boundaries is handled by the caller
feeding multiple seasons of PBP concat'd together — see
scripts/build_pbp_receiver_override.py.

Spec: docs/superpowers/specs/2026-05-01-wr-te-pbp-receiver-features-design.md §6.1.
"""

from __future__ import annotations

import re
from typing import Final

import pandas as pd

from projections.schemas import GSIS_ID_PATTERN

_GSIS_RE: Final[re.Pattern[str]] = re.compile(rf"^{GSIS_ID_PATTERN}$")
_DEEP_AIR_YARDS: Final[float] = 20.0
_RED_ZONE_YARDLINE: Final[int] = 20
_PBP_COLUMNS_USED: Final[tuple[str, ...]] = (
    "receiver_player_id",
    "season",
    "week",
    "pass_attempt",
    "complete_pass",
    "air_yards",
    "yards_after_catch",
    "yardline_100",
)


def _trailing_4_per_player_asof(
    per_game: pd.DataFrame,
    index: pd.DataFrame,
    *,
    value_col: str,
    out_col: str,
) -> pd.DataFrame:
    """Attach a trailing-4-receiver-active-game mean to a wider index via as-of join.

    Args:
        per_game: (gsis_id, season, week, value_col) — one row per
            receiver-active game with the per-game stat.
        index: (gsis_id, season, week) — one row per (rostered) player-week
            in the override; may include weeks where the player was not
            receiver-active.
        value_col: the per-game stat column name in ``per_game``.
        out_col: the output column name to attach to ``index``.

    Returns:
        A (gsis_id, season, week, out_col) frame with one row per input
        ``index`` row. ``out_col`` is the mean of the player's last 4
        receiver-active games strictly before (season, week); NaN if the
        player has fewer than 4 prior receiver-active games.

    Implementation: rolling-4 mean at per_game (no shift; "value at row N =
    mean of N-3 through N inclusive"), then merge_asof with
    direction='backward' and allow_exact_matches=False. The strict
    less-than semantics ensure (s, w) for an index row sees only games
    chronologically before it.
    """
    # 1. Sort per_game and compute rolling-4 (no shift) within player.
    pg = per_game.sort_values(["gsis_id", "season", "week"]).reset_index(drop=True).copy()
    pg["_rolled"] = pg.groupby("gsis_id", sort=False)[value_col].transform(
        lambda s: s.rolling(window=4, min_periods=4).mean()
    )

    # 2. Build a sortable timestamp from (season, week) — season*100+week is
    # monotonic provided week <= 22 (PbpSchema enforces).
    pg["_t"] = pg["season"].astype("int64") * 100 + pg["week"].astype("int64")

    # 3. As-of merge; preserve the original index row order via reset_index.
    idx = index.copy()
    idx["_t"] = idx["season"].astype("int64") * 100 + idx["week"].astype("int64")
    idx_sorted = idx.sort_values("_t").reset_index().rename(columns={"index": "_orig_idx"})
    pg_sorted = pg[["gsis_id", "_t", "_rolled"]].sort_values("_t")

    merged = pd.merge_asof(
        idx_sorted,
        pg_sorted,
        on="_t",
        by="gsis_id",
        direction="backward",
        allow_exact_matches=False,
    )

    out = merged.sort_values("_orig_idx").drop(columns=["_t", "_orig_idx"])
    out = out.rename(columns={"_rolled": out_col})
    return out[["gsis_id", "season", "week", out_col]].reset_index(drop=True)
```

- [ ] **Step 4: Run helper tests to verify they pass**

Run: `pytest tests/test_features/test_pbp_receiver_features.py -v -k trailing_4_asof`
Expected: 4 PASS (within_player_no_leakage, inactive_week_uses_last_active, min_periods_4, cross_season).

- [ ] **Step 5: Commit**

```bash
git add src/projections/features/pbp_receiver_features.py tests/test_features/test_pbp_receiver_features.py
git commit -m "feat(pbp-receiver): module skeleton + as-of trailing-4 helper"
```

---

### Task 2: `compute_receiver_adot`

**Files:**
- Modify: `src/projections/features/pbp_receiver_features.py` (add compute fn at module bottom, before `_trailing_4_per_player_asof`'s consumers)
- Modify: `tests/test_features/test_pbp_receiver_features.py` (append 3 tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_features/test_pbp_receiver_features.py`:

```python
def test_adot_air_yards_only() -> None:
    """aDOT averages only over rows with non-NaN air_yards (excludes sacks /
    throwaways with NaN air_yards upstream)."""
    from projections.features.pbp_receiver_features import compute_receiver_adot

    rows: list[dict[str, object]] = []
    # Build 4 prior weeks of receiver-active games for player A so trailing-4
    # has a window. Each week: 5 targets with air_yards = 10, plus 1 sack
    # (NaN air_yards). Mean over weeks 1-4 should be 10.0 (sacks excluded).
    gid = "00-0000001"
    for wk in range(1, 5):
        for i in range(5):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "play_id": 1000 * wk + i,
                    "receiver_player_id": gid,
                    "pass_attempt": 1.0,
                    "air_yards": 10.0,
                }
            )
        # 1 sack on the same player (would not actually credit air_yards to
        # receiver, but defensive: NaN air_yards excluded from mean).
        rows.append(
            {
                "season": 2024,
                "week": wk,
                "play_id": 1000 * wk + 99,
                "receiver_player_id": gid,
                "pass_attempt": 0.0,  # not a target — sack
                "air_yards": float("nan"),
            }
        )
    # Add a week 5 row in the index but with no PBP rows (test as-of lookup).
    index = pd.DataFrame(
        [{"gsis_id": gid, "season": 2024, "week": w} for w in range(1, 6)]
    )

    pbp = _make_pbp_rows(rows)
    per_game = compute_receiver_adot(pbp)
    # per_game has rows for receiver-active games (weeks 1-4) but no rolling
    # column attached — that's the helper's job. compute_receiver_adot returns
    # the per-game-mean frame ready for _trailing_4_per_player_asof.
    assert set(per_game.columns) == {"gsis_id", "season", "week", "aDOT_l4"}
    # Per-game mean for week 1: 5 targets at 10 yards = mean 10.0.
    wk1 = per_game.query("week == 1")
    assert wk1["aDOT_l4"].iloc[0] == pytest.approx(10.0)


def test_adot_trailing_4_within_player() -> None:
    """6 receiver-active games for A and B; trailing-4 stays within gsis_id."""
    from projections.features.pbp_receiver_features import compute_receiver_adot

    rows: list[dict[str, object]] = []
    for gid, base_yards in [("00-0000001", 5.0), ("00-0000002", 15.0)]:
        for wk in range(1, 7):
            for i in range(3):  # 3 targets per game
                rows.append(
                    {
                        "season": 2024,
                        "week": wk,
                        "play_id": 1000 * wk + (0 if gid == "00-0000001" else 500) + i,
                        "receiver_player_id": gid,
                        "pass_attempt": 1.0,
                        "air_yards": base_yards + wk,  # A: 6,7,...; B: 16,17,...
                    }
                )
    index = pd.DataFrame(
        [
            {"gsis_id": gid, "season": 2024, "week": w}
            for gid in ("00-0000001", "00-0000002")
            for w in range(1, 8)
        ]
    )

    pbp = _make_pbp_rows(rows)
    per_game = compute_receiver_adot(pbp)
    # per_game has 12 rows (6 weeks × 2 players).
    assert len(per_game) == 12
    # Per-game mean for player A, week 1: 5+1 = 6.0.
    a_wk1 = per_game.query("gsis_id == '00-0000001' and week == 1")
    assert a_wk1["aDOT_l4"].iloc[0] == pytest.approx(6.0)


def test_adot_zero_air_yards_targets_no_per_game_row() -> None:
    """If a receiver has zero non-NaN air_yards targets in a week, no per-game
    row is emitted (the player has no per-game value for that week)."""
    from projections.features.pbp_receiver_features import compute_receiver_adot

    gid = "00-0000001"
    # Week 1: 0 valid targets (only sacks). Week 2: 3 valid targets.
    rows: list[dict[str, object]] = [
        {
            "season": 2024,
            "week": 1,
            "play_id": 100,
            "receiver_player_id": gid,
            "pass_attempt": 0.0,
            "air_yards": float("nan"),
        },
        {
            "season": 2024,
            "week": 2,
            "play_id": 200,
            "receiver_player_id": gid,
            "pass_attempt": 1.0,
            "air_yards": 12.0,
        },
        {
            "season": 2024,
            "week": 2,
            "play_id": 201,
            "receiver_player_id": gid,
            "pass_attempt": 1.0,
            "air_yards": 18.0,
        },
    ]
    pbp = _make_pbp_rows(rows)
    per_game = compute_receiver_adot(pbp)
    # No row for week 1 (no valid targets); one row for week 2.
    assert len(per_game) == 1
    assert per_game.iloc[0]["week"] == 2
    assert per_game.iloc[0]["aDOT_l4"] == pytest.approx(15.0)
```

- [ ] **Step 2: Run tests to verify failures**

Run: `pytest tests/test_features/test_pbp_receiver_features.py -v -k adot`
Expected: 3 ImportError or AttributeError on `compute_receiver_adot`.

- [ ] **Step 3: Write the implementation**

Append to `src/projections/features/pbp_receiver_features.py` (after the helper):

```python
def compute_receiver_adot(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-receiver mean depth of target, per receiver-active game.

    Per (gsis_id, season, week): mean of ``air_yards`` across rows where
    ``receiver_player_id == gsis_id`` AND ``pass_attempt == 1.0`` AND
    ``air_yards.notna()``. NaN ``air_yards`` (sacks, throw-aways, no-plays)
    excluded.

    Output is the per-game mean frame, NOT the trailing-4 lookup. The
    assembler ``attach_pbp_receiver_features`` calls
    ``_trailing_4_per_player_asof`` to compute the trailing-4 against an
    index. The output column is named ``aDOT_l4`` to match the final
    override-column name; the helper preserves the name.

    Output: (gsis_id, season, week, aDOT_l4) — one row per receiver-active
    game (player had at least 1 valid target).
    """
    plays = pbp[
        (pbp["receiver_player_id"].notna())
        & (pbp["pass_attempt"] == 1.0)
        & (pbp["air_yards"].notna())
    ]
    per_game = (
        plays.groupby(["receiver_player_id", "season", "week"], as_index=False)["air_yards"]
        .mean()
        .rename(columns={"receiver_player_id": "gsis_id", "air_yards": "aDOT_l4"})
    )
    return per_game[["gsis_id", "season", "week", "aDOT_l4"]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features/test_pbp_receiver_features.py -v -k adot`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/features/pbp_receiver_features.py tests/test_features/test_pbp_receiver_features.py
git commit -m "feat(pbp-receiver): compute_receiver_adot"
```

---

### Task 3: `compute_receiver_deep_target_share`

**Files:**
- Modify: `src/projections/features/pbp_receiver_features.py`
- Modify: `tests/test_features/test_pbp_receiver_features.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_features/test_pbp_receiver_features.py`:

```python
def test_deep_target_share_threshold() -> None:
    """Targets with air_yards >= 20 count as deep; < 20 do not."""
    from projections.features.pbp_receiver_features import compute_receiver_deep_target_share

    gid = "00-0000001"
    depths = [5.0, 15.0, 19.0, 20.0, 25.0, 35.0]  # 3 deep (20, 25, 35), 3 shallow.
    rows = [
        {
            "season": 2024,
            "week": 1,
            "play_id": 100 + i,
            "receiver_player_id": gid,
            "pass_attempt": 1.0,
            "air_yards": d,
        }
        for i, d in enumerate(depths)
    ]
    pbp = _make_pbp_rows(rows)
    per_game = compute_receiver_deep_target_share(pbp)
    assert len(per_game) == 1
    assert per_game.iloc[0]["deep_target_share_l4"] == pytest.approx(3 / 6)


def test_deep_target_share_zero_targets_no_per_game_row() -> None:
    """A receiver with 0 valid targets in a week contributes no per-game row."""
    from projections.features.pbp_receiver_features import compute_receiver_deep_target_share

    gid = "00-0000001"
    # Only a sack (no valid targets).
    rows = [
        {
            "season": 2024,
            "week": 1,
            "play_id": 100,
            "receiver_player_id": gid,
            "pass_attempt": 0.0,
            "air_yards": float("nan"),
        }
    ]
    pbp = _make_pbp_rows(rows)
    per_game = compute_receiver_deep_target_share(pbp)
    assert len(per_game) == 0
```

- [ ] **Step 2: Run tests to verify failures**

Run: `pytest tests/test_features/test_pbp_receiver_features.py -v -k deep_target_share`
Expected: 2 ImportError on `compute_receiver_deep_target_share`.

- [ ] **Step 3: Write the implementation**

Append to `src/projections/features/pbp_receiver_features.py`:

```python
def compute_receiver_deep_target_share(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-receiver share of targets with air_yards >= 20, per
    receiver-active game.

    Per (gsis_id, season, week):
      total_valid_targets = count rows where receiver_player_id == gsis_id
                            AND pass_attempt == 1.0 AND air_yards.notna()
      deep_targets        = count rows where receiver_player_id == gsis_id
                            AND pass_attempt == 1.0 AND air_yards >= 20.0
      share               = deep_targets / total_valid_targets

    A receiver with zero valid targets in a week contributes no per-game row
    (no division-by-zero; the player simply doesn't appear in per_game for
    that week). The 20-yard cutoff is the conventional "deep" threshold.

    Output: (gsis_id, season, week, deep_target_share_l4) — one row per
    receiver-active game.
    """
    valid = pbp[
        (pbp["receiver_player_id"].notna())
        & (pbp["pass_attempt"] == 1.0)
        & (pbp["air_yards"].notna())
    ].copy()
    valid["_is_deep"] = (valid["air_yards"] >= _DEEP_AIR_YARDS).astype("float64")
    per_game = (
        valid.groupby(["receiver_player_id", "season", "week"], as_index=False)["_is_deep"]
        .mean()
        .rename(columns={"receiver_player_id": "gsis_id", "_is_deep": "deep_target_share_l4"})
    )
    return per_game[["gsis_id", "season", "week", "deep_target_share_l4"]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features/test_pbp_receiver_features.py -v -k deep_target_share`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/features/pbp_receiver_features.py tests/test_features/test_pbp_receiver_features.py
git commit -m "feat(pbp-receiver): compute_receiver_deep_target_share"
```

---

### Task 4: `compute_receiver_yac_per_reception`

**Files:**
- Modify: `src/projections/features/pbp_receiver_features.py`
- Modify: `tests/test_features/test_pbp_receiver_features.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features/test_pbp_receiver_features.py`:

```python
def test_yac_completions_only() -> None:
    """YAC averages only over completions (complete_pass == 1.0); incompletions
    excluded even when receiver_player_id is set."""
    from projections.features.pbp_receiver_features import compute_receiver_yac_per_reception

    gid = "00-0000001"
    rows = [
        # 3 completions with YAC 5, 10, 15.
        {
            "season": 2024,
            "week": 1,
            "play_id": 100,
            "receiver_player_id": gid,
            "pass_attempt": 1.0,
            "complete_pass": 1.0,
            "yards_after_catch": 5.0,
        },
        {
            "season": 2024,
            "week": 1,
            "play_id": 101,
            "receiver_player_id": gid,
            "pass_attempt": 1.0,
            "complete_pass": 1.0,
            "yards_after_catch": 10.0,
        },
        {
            "season": 2024,
            "week": 1,
            "play_id": 102,
            "receiver_player_id": gid,
            "pass_attempt": 1.0,
            "complete_pass": 1.0,
            "yards_after_catch": 15.0,
        },
        # 2 incompletions (NaN YAC). Should NOT contribute.
        {
            "season": 2024,
            "week": 1,
            "play_id": 103,
            "receiver_player_id": gid,
            "pass_attempt": 1.0,
            "complete_pass": 0.0,
            "yards_after_catch": float("nan"),
        },
        {
            "season": 2024,
            "week": 1,
            "play_id": 104,
            "receiver_player_id": gid,
            "pass_attempt": 1.0,
            "complete_pass": 0.0,
            "yards_after_catch": float("nan"),
        },
    ]
    pbp = _make_pbp_rows(rows)
    per_game = compute_receiver_yac_per_reception(pbp)
    # Mean of (5, 10, 15) = 10.0.
    assert len(per_game) == 1
    assert per_game.iloc[0]["yac_per_reception_l4"] == pytest.approx(10.0)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_features/test_pbp_receiver_features.py -v -k yac`
Expected: ImportError on `compute_receiver_yac_per_reception`.

- [ ] **Step 3: Write the implementation**

Append to `src/projections/features/pbp_receiver_features.py`:

```python
def compute_receiver_yac_per_reception(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-receiver mean yards-after-catch per completion, per receiver-active
    game.

    Per (gsis_id, season, week): mean of ``yards_after_catch`` across rows
    where ``receiver_player_id == gsis_id`` AND ``complete_pass == 1.0`` AND
    ``yards_after_catch.notna()``. Filtered to completions — YAC only exists
    when the ball is caught.

    Receivers with no catches in a week contribute no per-game row.

    Output: (gsis_id, season, week, yac_per_reception_l4) — one row per
    receiver-active game.
    """
    completions = pbp[
        (pbp["receiver_player_id"].notna())
        & (pbp["complete_pass"] == 1.0)
        & (pbp["yards_after_catch"].notna())
    ]
    per_game = (
        completions.groupby(["receiver_player_id", "season", "week"], as_index=False)[
            "yards_after_catch"
        ]
        .mean()
        .rename(
            columns={
                "receiver_player_id": "gsis_id",
                "yards_after_catch": "yac_per_reception_l4",
            }
        )
    )
    return per_game[["gsis_id", "season", "week", "yac_per_reception_l4"]]
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_features/test_pbp_receiver_features.py -v -k yac`
Expected: 1 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/features/pbp_receiver_features.py tests/test_features/test_pbp_receiver_features.py
git commit -m "feat(pbp-receiver): compute_receiver_yac_per_reception"
```

---

### Task 5: `compute_receiver_red_zone_target_share`

**Files:**
- Modify: `src/projections/features/pbp_receiver_features.py`
- Modify: `tests/test_features/test_pbp_receiver_features.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features/test_pbp_receiver_features.py`:

```python
def test_red_zone_target_share_yardline_threshold() -> None:
    """yardline_100 <= 20 counts as red-zone; > 20 does not."""
    from projections.features.pbp_receiver_features import (
        compute_receiver_red_zone_target_share,
    )

    gid = "00-0000001"
    yardlines = [5.0, 15.0, 20.0, 21.0, 50.0]  # 3 RZ (5, 15, 20), 2 non-RZ (21, 50).
    rows = [
        {
            "season": 2024,
            "week": 1,
            "play_id": 100 + i,
            "receiver_player_id": gid,
            "pass_attempt": 1.0,
            "yardline_100": yl,
        }
        for i, yl in enumerate(yardlines)
    ]
    pbp = _make_pbp_rows(rows)
    per_game = compute_receiver_red_zone_target_share(pbp)
    assert len(per_game) == 1
    assert per_game.iloc[0]["red_zone_target_share_l4"] == pytest.approx(3 / 5)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_features/test_pbp_receiver_features.py -v -k red_zone`
Expected: ImportError on `compute_receiver_red_zone_target_share`.

- [ ] **Step 3: Write the implementation**

Append to `src/projections/features/pbp_receiver_features.py`:

```python
def compute_receiver_red_zone_target_share(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-receiver share of targets at yardline_100 <= 20, per
    receiver-active game.

    Per (gsis_id, season, week):
      total_targets = count rows where receiver_player_id == gsis_id AND
                      pass_attempt == 1.0 AND yardline_100.notna()
      rz_targets    = count rows where receiver_player_id == gsis_id AND
                      pass_attempt == 1.0 AND yardline_100 <= 20
      share         = rz_targets / total_targets

    yardline_100 = 20 is the standard NFL red-zone definition (yards from
    the opponent's goal line). This is the receiver's RZ target share, not
    the team's RZ target rate; captures whether the player is the offense's
    preferred end-zone target.

    Output: (gsis_id, season, week, red_zone_target_share_l4) — one row per
    receiver-active game.
    """
    targets = pbp[
        (pbp["receiver_player_id"].notna())
        & (pbp["pass_attempt"] == 1.0)
        & (pbp["yardline_100"].notna())
    ].copy()
    targets["_is_rz"] = (targets["yardline_100"] <= _RED_ZONE_YARDLINE).astype("float64")
    per_game = (
        targets.groupby(["receiver_player_id", "season", "week"], as_index=False)["_is_rz"]
        .mean()
        .rename(
            columns={
                "receiver_player_id": "gsis_id",
                "_is_rz": "red_zone_target_share_l4",
            }
        )
    )
    return per_game[["gsis_id", "season", "week", "red_zone_target_share_l4"]]
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_features/test_pbp_receiver_features.py -v -k red_zone`
Expected: 1 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/features/pbp_receiver_features.py tests/test_features/test_pbp_receiver_features.py
git commit -m "feat(pbp-receiver): compute_receiver_red_zone_target_share"
```

---

### Task 6: `attach_pbp_receiver_features` (assembler)

**Files:**
- Modify: `src/projections/features/pbp_receiver_features.py`
- Modify: `tests/test_features/test_pbp_receiver_features.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_features/test_pbp_receiver_features.py`:

```python
def test_attach_receiver_features_schema() -> None:
    """Assembler output has the 4 new columns appended in spec order; row
    count matches the input index."""
    from projections.features.pbp_receiver_features import attach_pbp_receiver_features

    gid = "00-0000001"
    # 4 receiver-active games for the player so trailing-4 fires at week 5.
    rows: list[dict[str, object]] = []
    for wk in range(1, 5):
        rows.append(
            {
                "season": 2024,
                "week": wk,
                "play_id": 1000 * wk,
                "receiver_player_id": gid,
                "pass_attempt": 1.0,
                "complete_pass": 1.0,
                "air_yards": 10.0,
                "yards_after_catch": 5.0,
                "yardline_100": 30.0,
            }
        )
    pbp = _make_pbp_rows(rows)
    index = pd.DataFrame(
        [{"gsis_id": gid, "season": 2024, "week": w} for w in range(1, 7)]
    )
    out = attach_pbp_receiver_features(index, pbp)
    assert list(out.columns) == [
        "gsis_id",
        "season",
        "week",
        "aDOT_l4",
        "deep_target_share_l4",
        "yac_per_reception_l4",
        "red_zone_target_share_l4",
    ]
    assert len(out) == len(index)


def test_attach_receiver_features_left_join_semantics() -> None:
    """Index row for a receiver with no PBP rows yields NaN on all 4 columns."""
    from projections.features.pbp_receiver_features import attach_pbp_receiver_features

    # Player A has 4 weeks of PBP; player B has none.
    rows: list[dict[str, object]] = []
    for wk in range(1, 5):
        rows.append(
            {
                "season": 2024,
                "week": wk,
                "play_id": 1000 * wk,
                "receiver_player_id": "00-0000001",
                "pass_attempt": 1.0,
                "complete_pass": 1.0,
                "air_yards": 10.0,
                "yards_after_catch": 5.0,
                "yardline_100": 30.0,
            }
        )
    pbp = _make_pbp_rows(rows)
    index = pd.DataFrame(
        [
            {"gsis_id": "00-0000001", "season": 2024, "week": 5},
            {"gsis_id": "00-0000002", "season": 2024, "week": 5},
        ]
    )
    out = attach_pbp_receiver_features(index, pbp)
    b_row = out.query("gsis_id == '00-0000002'")
    assert len(b_row) == 1
    for col in ("aDOT_l4", "deep_target_share_l4", "yac_per_reception_l4", "red_zone_target_share_l4"):
        assert pd.isna(b_row[col].iloc[0])


def test_attach_receiver_features_empty_pbp() -> None:
    """Empty PBP short-circuits to all-NaN columns (matches team-level fast
    path)."""
    from projections.features.pbp_receiver_features import attach_pbp_receiver_features

    pbp = pd.DataFrame()
    index = pd.DataFrame(
        [{"gsis_id": "00-0000001", "season": 2024, "week": 5}]
    )
    out = attach_pbp_receiver_features(index, pbp)
    assert len(out) == 1
    for col in ("aDOT_l4", "deep_target_share_l4", "yac_per_reception_l4", "red_zone_target_share_l4"):
        assert pd.isna(out[col].iloc[0])
```

- [ ] **Step 2: Run tests to verify failures**

Run: `pytest tests/test_features/test_pbp_receiver_features.py -v -k attach_receiver_features`
Expected: 3 ImportError on `attach_pbp_receiver_features`.

- [ ] **Step 3: Write the implementation**

Append to `src/projections/features/pbp_receiver_features.py`:

```python
_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "aDOT_l4",
    "deep_target_share_l4",
    "yac_per_reception_l4",
    "red_zone_target_share_l4",
)


def attach_pbp_receiver_features(
    index: pd.DataFrame,
    pbp: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the 4 PBP receiver features to a (gsis_id, season, week) index.

    Args:
        index: ``(gsis_id, season, week)`` — one row per receiver-week.
            Built from ``depth_charts`` filtered to position in {WR, TE}.
        pbp: PBP frame matching ``PbpSchema``, projected to or wider than
            the receiver-features column set. Must include the seasons
            spanning the index plus one prior season for trailing-4 backfill.

    Returns:
        A copy of ``index`` with 4 columns appended in order:
        ``aDOT_l4``, ``deep_target_share_l4``, ``yac_per_reception_l4``,
        ``red_zone_target_share_l4``. Row count equals ``len(index)``.
        All 4 columns are float64 (NaN where trailing-4 has fewer than 4
        prior receiver-active games or the player has no PBP rows at all).

    All four computes key on ``receiver_player_id``; no team / opponent
    join required.

    Empty ``pbp`` short-circuits to all-NaN columns — same shape as a
    successful call where every row's trailing-4 has fewer than 4 prior
    receiver-active games. Schema ``nullable=True`` covers this.
    """
    if pbp.empty:
        out = index.copy()
        for col in _OUTPUT_COLUMNS:
            out[col] = float("nan")
        return out.reset_index(drop=True)

    pbp_proj = pbp[list(_PBP_COLUMNS_USED)]
    adot_pg = compute_receiver_adot(pbp_proj)
    deep_pg = compute_receiver_deep_target_share(pbp_proj)
    yac_pg = compute_receiver_yac_per_reception(pbp_proj)
    rz_pg = compute_receiver_red_zone_target_share(pbp_proj)

    out = index.copy()
    for per_game, col in (
        (adot_pg, "aDOT_l4"),
        (deep_pg, "deep_target_share_l4"),
        (yac_pg, "yac_per_reception_l4"),
        (rz_pg, "red_zone_target_share_l4"),
    ):
        attached = _trailing_4_per_player_asof(
            per_game, index, value_col=col, out_col=col
        )
        # attached has len == len(index); merge it onto out 1:1 by position.
        out = out.merge(attached, on=["gsis_id", "season", "week"], how="left")

    return out.reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features/test_pbp_receiver_features.py -v -k attach_receiver_features`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/features/pbp_receiver_features.py tests/test_features/test_pbp_receiver_features.py
git commit -m "feat(pbp-receiver): attach_pbp_receiver_features assembler"
```

---

### Task 7: `build_pbp_receiver_overrides` (validating wrapper)

**Files:**
- Modify: `src/projections/features/pbp_receiver_features.py`
- Modify: `tests/test_features/test_pbp_receiver_features.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_features/test_pbp_receiver_features.py`:

```python
def test_build_receiver_overrides_canonical_gsis() -> None:
    """Wrapper raises ValueError on malformed GSIS in the index."""
    from projections.features.pbp_receiver_features import build_pbp_receiver_overrides

    pbp = _make_pbp_rows([])
    bad_index = pd.DataFrame(
        [
            {"gsis_id": "00-0000001", "season": 2024, "week": 5},  # valid
            {"gsis_id": "garbage-id", "season": 2024, "week": 5},  # invalid
        ]
    )
    with pytest.raises(ValueError, match="invalid gsis_id"):
        build_pbp_receiver_overrides(pbp, bad_index)


def test_build_receiver_overrides_dup_key() -> None:
    """Wrapper raises ValueError on duplicate (gsis_id, season, week) keys."""
    from projections.features.pbp_receiver_features import build_pbp_receiver_overrides

    pbp = _make_pbp_rows([])
    dup_index = pd.DataFrame(
        [
            {"gsis_id": "00-0000001", "season": 2024, "week": 5},
            {"gsis_id": "00-0000001", "season": 2024, "week": 5},  # duplicate
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        build_pbp_receiver_overrides(pbp, dup_index)


def test_build_receiver_overrides_row_count_invariant() -> None:
    """Wrapper output has exactly len(index) rows; column projection matches
    spec §2.4."""
    from projections.features.pbp_receiver_features import build_pbp_receiver_overrides

    pbp = _make_pbp_rows([])  # empty PBP -> all-NaN values
    index = pd.DataFrame(
        [
            {"gsis_id": "00-0000001", "season": 2024, "week": w}
            for w in range(1, 6)
        ]
    )
    out = build_pbp_receiver_overrides(pbp, index)
    assert len(out) == len(index)
    assert list(out.columns) == [
        "gsis_id",
        "season",
        "week",
        "aDOT_l4",
        "deep_target_share_l4",
        "yac_per_reception_l4",
        "red_zone_target_share_l4",
    ]
```

- [ ] **Step 2: Run tests to verify failures**

Run: `pytest tests/test_features/test_pbp_receiver_features.py -v -k build_receiver_overrides`
Expected: 3 ImportError on `build_pbp_receiver_overrides`.

- [ ] **Step 3: Write the implementation**

Append to `src/projections/features/pbp_receiver_features.py`:

```python
def build_pbp_receiver_overrides(
    pbp: pd.DataFrame,
    receiver_index: pd.DataFrame,
) -> pd.DataFrame:
    """Public assembler. Returns the 4-column override frame ready to write.

    Args:
        pbp: PBP frame matching ``PbpSchema``. Must include the seasons
            spanning the index plus one prior season for trailing-4 backfill.
        receiver_index: ``(gsis_id, season, week)`` — one row per
            receiver-week. Built by the override script from ``depth_charts``
            filtered to ``position in {WR, TE}``.

    Returns:
        ``(gsis_id, season, week, aDOT_l4, deep_target_share_l4,
        yac_per_reception_l4, red_zone_target_share_l4)`` — one row per
        input index row.

    Raises:
        ValueError: gsis_id format violations or duplicate
            (gsis_id, season, week) keys in the index.
        AssertionError: row-count mismatch after merges (internal-invariant
            violation; a future compute regression that introduces duplicate
            (gsis_id, season, week) keys would trigger this).

    Per-position coverage validation is the probe's responsibility; see
    spec §1.3 criterion 1 + §3.3 step 2.
    """
    bad_ids = [
        g for g in receiver_index["gsis_id"].dropna() if not _GSIS_RE.match(str(g))
    ]
    if bad_ids:
        raise ValueError(
            f"invalid gsis_id format(s): {bad_ids[:3]} (and {max(0, len(bad_ids) - 3)} more)"
        )

    dup_mask = receiver_index.duplicated(subset=["gsis_id", "season", "week"], keep=False)
    if dup_mask.any():
        n_dup = int(dup_mask.sum())
        raise ValueError(f"duplicate (gsis_id, season, week) keys in index: {n_dup} rows")

    out = attach_pbp_receiver_features(receiver_index, pbp)

    if len(out) != len(receiver_index):
        raise AssertionError(
            f"row count mismatch: input index had {len(receiver_index)} rows, "
            f"output has {len(out)}; suggests a many-to-many merge regression"
        )

    return out[["gsis_id", "season", "week", *_OUTPUT_COLUMNS]].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features/test_pbp_receiver_features.py -v -k build_receiver_overrides`
Expected: 3 PASS.

- [ ] **Step 5: Run the full module test suite**

Run: `pytest tests/test_features/test_pbp_receiver_features.py -v`
Expected: 17 PASS (4 trailing_4_asof + 3 adot + 2 deep_target_share + 1 yac + 1 red_zone + 3 attach + 3 build).

- [ ] **Step 6: Commit**

```bash
git add src/projections/features/pbp_receiver_features.py tests/test_features/test_pbp_receiver_features.py
git commit -m "feat(pbp-receiver): build_pbp_receiver_overrides validating wrapper"
```

---

## Phase 2 — Override script

### Task 8: `scripts/build_pbp_receiver_override.py` + CONTRIBUTING.md update

**Files:**
- Create: `scripts/build_pbp_receiver_override.py`
- Create: `tests/test_scripts/test_build_pbp_receiver_override_cli.py`
- Modify: `CONTRIBUTING.md` (append "Regenerating the PBP receiver override" subsection in same commit as the script)

- [ ] **Step 1: Write the failing CLI tests**

Create `tests/test_scripts/test_build_pbp_receiver_override_cli.py`:

```python
"""scripts/build_pbp_receiver_override.py — argparse + main smoke tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.build_pbp_receiver_override import (
    _build_receiver_index,
    _parse_season_range,
    main,
)


def test_parse_season_range_dash() -> None:
    assert _parse_season_range("2018-2024") == range(2018, 2025)


def test_parse_season_range_single() -> None:
    assert _parse_season_range("2024") == range(2024, 2025)


def test_build_receiver_index_filters_to_wr_te() -> None:
    """Index includes WR + TE rows only; QB / RB filtered out; deduped on
    (gsis_id, season, week)."""
    depth_charts = pd.DataFrame(
        [
            {"gsis_id": "00-0000001", "season": 2024, "week": 1, "position": "WR"},
            {"gsis_id": "00-0000001", "season": 2024, "week": 1, "position": "WR"},  # dup
            {"gsis_id": "00-0000002", "season": 2024, "week": 1, "position": "TE"},
            {"gsis_id": "00-0000003", "season": 2024, "week": 1, "position": "QB"},  # filtered
            {"gsis_id": "00-0000004", "season": 2024, "week": 1, "position": "RB"},  # filtered
        ]
    )
    idx = _build_receiver_index(depth_charts, range(2024, 2025))
    assert set(idx["gsis_id"]) == {"00-0000001", "00-0000002"}
    assert len(idx) == 2  # dedupe of player A's duplicate row
    assert list(idx.columns) == ["gsis_id", "season", "week"]


def test_main_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    """If output exists and --force is not passed, main exits with code 2."""
    output = tmp_path / "pbp_receiver.parquet"
    output.write_text("placeholder")  # exists
    with pytest.raises(SystemExit) as excinfo:
        main(["--output", str(output), "--data-root", str(tmp_path)])
    assert excinfo.value.code == 2  # argparse.error -> exit 2
```

- [ ] **Step 2: Run tests to verify failures**

Run: `pytest tests/test_scripts/test_build_pbp_receiver_override_cli.py -v`
Expected: 4 ImportError on `scripts.build_pbp_receiver_override`.

- [ ] **Step 3: Write the implementation**

Create `scripts/build_pbp_receiver_override.py`:

```python
"""Build the PBP receiver override parquet for the WR/TE family probe.

One-shot CLI. Loads PBP across the requested season range (plus the prior
season for trailing-4 backfill) and depth_charts filtered to WR + TE, calls
build_pbp_receiver_overrides, writes the resulting frame to a parquet.

Output is NOT committed — it's regenerable from the live raw partitions.

Usage:
    python -m scripts.build_pbp_receiver_override --seasons 2018-2024
    python -m scripts.build_pbp_receiver_override --seasons 2018-2024 --force
    python -m scripts.build_pbp_receiver_override --output data/features_probe/x.parquet

Spec: docs/superpowers/specs/2026-05-01-wr-te-pbp-receiver-features-design.md §6.2.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from projections.features.pbp_receiver_features import build_pbp_receiver_overrides
from projections.schemas import Position
from projections.store import read_partition

_DEFAULT_OUTPUT = Path("data/features_probe/pbp_receiver.parquet")
_RECEIVER_POSITIONS: tuple[str, ...] = (Position.WR.value, Position.TE.value)


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
        raise FileNotFoundError(
            f"no partitions found for table={table!r} in seasons={list(seasons)}"
        )
    return pd.concat(frames, ignore_index=True)


def _build_receiver_index(depth_charts: pd.DataFrame, seasons: range) -> pd.DataFrame:
    """Filter depth_charts to WR + TE in the requested season range; dedupe
    on (gsis_id, season, week).

    Mirrors the team-level pattern (scripts/build_pbp_family_override.py:78-80)
    which uses depth_charts as the index source so the override's coverage
    matches the per-position baseline-feature parquet's coverage.
    """
    return (
        depth_charts[
            depth_charts["season"].isin(seasons)
            & depth_charts["position"].isin(_RECEIVER_POSITIONS)
        ][["gsis_id", "season", "week"]]
        .drop_duplicates(subset=["gsis_id", "season", "week"])
        .reset_index(drop=True)
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else "")
    parser.add_argument(
        "--seasons",
        type=_parse_season_range,
        default=range(2018, 2025),
        help="Season range, e.g. '2018-2024' or '2024'. Default: 2018-2024.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root for raw and features partitions. Default: data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Override output parquet path. Default: {_DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output if it already exists.",
    )
    args = parser.parse_args(argv)

    if args.output.exists() and not args.force:
        parser.error(f"{args.output} exists; pass --force to overwrite.")

    seasons: range = args.seasons
    raw_root = args.data_root / "raw"
    pbp_seasons = range(seasons.start - 1, seasons.stop)  # +1 prior for backfill

    pbp = _read_concat(raw_root, "pbp", list(pbp_seasons))
    depth_charts = _read_concat(raw_root, "depth_charts", list(seasons))

    receiver_index = _build_receiver_index(depth_charts, seasons)
    overrides = build_pbp_receiver_overrides(pbp, receiver_index)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    overrides.to_parquet(args.output, index=False)
    print(f"wrote {len(overrides)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scripts/test_build_pbp_receiver_override_cli.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Append CONTRIBUTING.md subsection**

Open `CONTRIBUTING.md`. Find the existing "Regenerating the PBP family override" subsection (added in PR #20). Immediately after it, append:

```markdown
### Regenerating the PBP receiver override

The receiver-level override parquet (`data/features_probe/pbp_receiver.parquet`)
is regenerable from the live `data/raw/pbp` and `data/raw/depth_charts`
partitions. It is NOT committed.

```bash
python -m scripts.build_pbp_receiver_override --seasons 2018-2024
```

To overwrite an existing file, pass `--force`. The script reads PBP for
`[seasons.start - 1, seasons.stop)` (one prior season for trailing-4 backfill
at week 1–4 of each season) and writes one row per `(gsis_id, season, week)`
for every WR / TE rostered per `depth_charts`.
```

- [ ] **Step 6: Commit**

```bash
git add scripts/build_pbp_receiver_override.py tests/test_scripts/test_build_pbp_receiver_override_cli.py CONTRIBUTING.md
git commit -m "feat(pbp-receiver): override generator script + CONTRIBUTING note"
```

---

## Phase 3 — Verification gate

### Task 9: Full pytest + mypy + ruff + format pass

This is the CLAUDE.md "end-of-effort checklist" applied at the natural Phase 1+2 boundary, before any real-data work. Catches typing / lint drift before the long-running real-data tasks.

- [ ] **Step 1: Run the full test suite**

Run: `pytest -v`
Expected: ALL tests pass (existing 1000+ tests + 17 new tests in `test_pbp_receiver_features.py` + 4 new CLI tests). If any prior test fails, investigate — the new module's helpers should be additive only.

- [ ] **Step 2: Run mypy strict**

Run: `mypy src tests`
Expected: zero violations. If `_trailing_4_per_player_asof`'s `pd.merge_asof` return type triggers a mypy complaint, narrow with `cast(pd.DataFrame, ...)` and explain in a comment.

- [ ] **Step 3: Run ruff lint**

Run: `ruff check src tests scripts`
Expected: zero violations.

- [ ] **Step 4: Run ruff format check**

Run: `ruff format --check src tests scripts`
Expected: no drift.

- [ ] **Step 5: Run the ingest/store/schemas guard suite**

Run: `pytest -v -k "ingest or store or schemas"`
Expected: ALL pass. Nothing in this spec touches ingest/store/schemas, but the gate is cheap and runs anyway per CLAUDE.md.

- [ ] **Step 6: If anything failed, fix and re-run, then commit any cleanup**

If a fix is needed:

```bash
git add -p   # carefully add only the cleanup
git commit -m "chore(pbp-receiver): fix <category> drift surfaced by verification gate"
```

If everything passed, no commit needed for this task — proceed to Phase 4.

---

## Phase 4 — Real-data execution (manual; reports committed)

These tasks run actual real-data probes against the live RB/WR/TE/QB feature parquets and emit reports. They are not scriptable as TDD tasks because they consume real data and have ~minutes-to-hours runtime. Each step's acceptance criterion is the produced file + a one-line check.

### Task 10: Generate override + inspect coverage

- [ ] **Step 1: Generate the override**

Run: `python -m scripts.build_pbp_receiver_override --seasons 2018-2024`
Expected output (last line): `wrote N rows to data/features_probe/pbp_receiver.parquet` for some `N` in the range ~25,000–35,000 (across 7 seasons × ~22 weeks × ~50 rostered WR/TE per team × 32 teams ≈ 247k worst-case, but deduped + only-with-prior-history filters bring it down).

- [ ] **Step 2: Inspect per-season per-position coverage**

The probe's coverage check fires per (position, season). Inspect the override's NaN rate manually before invoking the probe to get ahead of any coverage failure:

```python
python - <<'PY'
import pandas as pd
df = pd.read_parquet("data/features_probe/pbp_receiver.parquet")
print(f"total rows: {len(df)}")
for col in ("aDOT_l4", "deep_target_share_l4", "yac_per_reception_l4", "red_zone_target_share_l4"):
    nonnull = df[col].notna().sum()
    pct = nonnull / len(df) * 100
    print(f"  {col}: {nonnull}/{len(df)} non-null ({pct:.1f}%)")
print()
for season in sorted(df["season"].unique()):
    sub = df[df["season"] == season]
    print(f"season {season}: {len(sub)} rows; non-null rates by feature:")
    for col in ("aDOT_l4", "deep_target_share_l4", "yac_per_reception_l4", "red_zone_target_share_l4"):
        nonnull = sub[col].notna().sum()
        pct = nonnull / len(sub) * 100
        print(f"  {col}: {pct:.1f}%")
PY
```

Expected: 2018 rates lowest (no Y-1 backfill available; PR #20 saw 96.6% coverage on the team-level override). 2019+ rates near-100% on `aDOT_l4` and `deep_target_share_l4`; `yac_per_reception_l4` slightly lower (filtered to completions only); `red_zone_target_share_l4` lower (RZ target coverage requires `yardline_100.notna()` plus a target — small fraction of plays).

- [ ] **Step 3: Decide coverage threshold**

If 2018 rates dip below 95% on any feature, this is expected for the WR position (early-season cold-start). Plan to pass `--coverage-threshold 0.90` to the probe in Task 11. If TE rates are below 95% in any season, also plan `0.90`. If any rate is below 0.90, escalate — review the spec's §1.3 criterion 1 risk note before proceeding.

- [ ] **Step 4: No commit (override is not committed per spec §5.3)**

The override parquet stays under `data/features_probe/` which is not committed. Proceed to Task 11.

---

### Task 11: Run baseline augment + swap probes; commit reports

- [ ] **Step 1: Run baseline augment-mode probe**

Run (one line, broken for readability):

```bash
python -m scripts.probe_feature_signal \
  --candidate-name pbp_receiver_augment \
  --model baseline \
  --position WR --position TE \
  --override data/features_probe/pbp_receiver.parquet \
  --csv-out reports/feature_probe_pbp_receiver_augment.csv
```

If Task 10 step 3 indicated coverage below 95%, append `--coverage-threshold 0.90`.

Expected: 4 files produced under `reports/`:
- `feature_probe_pbp_receiver_augment_WR.md`
- `feature_probe_pbp_receiver_augment_WR.csv`
- `feature_probe_pbp_receiver_augment_TE.md`
- `feature_probe_pbp_receiver_augment_TE.csv`

Runtime: 2–10 minutes (baseline probe; per probe spec §8).

- [ ] **Step 2: Run baseline swap-mode probe**

Run:

```bash
python -m scripts.probe_feature_signal \
  --candidate-name pbp_receiver_swap \
  --model baseline \
  --position WR --position TE \
  --override data/features_probe/pbp_receiver.parquet \
  --drop avg_intended_air_yards_std,avg_yac_above_expectation_std \
  --csv-out reports/feature_probe_pbp_receiver_swap.csv
```

(Same `--coverage-threshold` adjustment if needed from Task 10.)

Expected: 4 more files: `feature_probe_pbp_receiver_swap_{WR,TE}.{md,csv}`.

- [ ] **Step 3: Commit the 8 baseline reports**

```bash
git add reports/feature_probe_pbp_receiver_augment_WR.md \
        reports/feature_probe_pbp_receiver_augment_WR.csv \
        reports/feature_probe_pbp_receiver_augment_TE.md \
        reports/feature_probe_pbp_receiver_augment_TE.csv \
        reports/feature_probe_pbp_receiver_swap_WR.md \
        reports/feature_probe_pbp_receiver_swap_WR.csv \
        reports/feature_probe_pbp_receiver_swap_TE.md \
        reports/feature_probe_pbp_receiver_swap_TE.csv
git commit -m "report(pbp-receiver): baseline augment + swap probe outputs"
```

---

### Task 12: Compute family verdict; conditional lgb-nb runs

- [ ] **Step 1: Compute the family verdict from the two baseline reports**

Run:

```python
python - <<'PY'
from pathlib import Path
from projections.backtest.feature_probe import (
    family_verdict_from_reports,
    load_probe_report_from_csv,
)

reports = [
    load_probe_report_from_csv(Path(f"reports/feature_probe_pbp_receiver_augment_{pos}.csv"))
    for pos in ("WR", "TE")
] + [
    load_probe_report_from_csv(Path(f"reports/feature_probe_pbp_receiver_swap_{pos}.csv"))
    for pos in ("WR", "TE")
]
print(f"Family verdict (baseline only): {family_verdict_from_reports(reports)}")
PY
```

Expected output: either `SIGNAL` or `NULL`. Note: `load_probe_report_from_csv` is the existing helper in `feature_probe.py` (added in PR #18). If the helper has a different name in the codebase, locate it via `grep -n "ProbeReport" src/projections/backtest/feature_probe.py` and adapt the snippet.

- [ ] **Step 2a: If verdict is SIGNAL — skip lgb-nb, proceed to Task 13**

Note this in your terminal: "Baseline returned SIGNAL; skipping conditional lgb-nb per spec §3.2." Proceed directly to Task 13.

- [ ] **Step 2b: If verdict is NULL — run lgb-nb augment probe**

Per spec §1.3 criterion 3 + §3.2, NULL is not durable until both lgb-nb modes have run. Run:

```bash
python -m scripts.probe_feature_signal \
  --candidate-name pbp_receiver_lgbnb_augment \
  --model lightgbm-nb \
  --position WR --position TE \
  --override data/features_probe/pbp_receiver.parquet \
  --csv-out reports/feature_probe_pbp_receiver_lgbnb_augment.csv
```

(Same `--coverage-threshold` adjustment if needed.)

Expected: 4 new files: `feature_probe_pbp_receiver_lgbnb_augment_{WR,TE}.{md,csv}`. Runtime: ~1 hr.

- [ ] **Step 2c: If verdict is NULL — run lgb-nb swap probe**

```bash
python -m scripts.probe_feature_signal \
  --candidate-name pbp_receiver_lgbnb_swap \
  --model lightgbm-nb \
  --position WR --position TE \
  --override data/features_probe/pbp_receiver.parquet \
  --drop avg_intended_air_yards_std,avg_yac_above_expectation_std \
  --csv-out reports/feature_probe_pbp_receiver_lgbnb_swap.csv
```

Runtime: ~1 hr.

- [ ] **Step 2d: If lgb-nb ran — recompute the family verdict over all 4 reports**

```python
python - <<'PY'
from pathlib import Path
from projections.backtest.feature_probe import (
    family_verdict_from_reports,
    load_probe_report_from_csv,
)

reports = []
for prefix in (
    "feature_probe_pbp_receiver_augment",
    "feature_probe_pbp_receiver_swap",
    "feature_probe_pbp_receiver_lgbnb_augment",
    "feature_probe_pbp_receiver_lgbnb_swap",
):
    for pos in ("WR", "TE"):
        reports.append(load_probe_report_from_csv(Path(f"reports/{prefix}_{pos}.csv")))
print(f"Family verdict (baseline + lgb-nb): {family_verdict_from_reports(reports)}")
PY
```

This is the durable verdict per spec §1.3 criterion 3.

- [ ] **Step 3: If lgb-nb ran — commit the 8 lgb-nb reports**

```bash
git add reports/feature_probe_pbp_receiver_lgbnb_augment_WR.md \
        reports/feature_probe_pbp_receiver_lgbnb_augment_WR.csv \
        reports/feature_probe_pbp_receiver_lgbnb_augment_TE.md \
        reports/feature_probe_pbp_receiver_lgbnb_augment_TE.csv \
        reports/feature_probe_pbp_receiver_lgbnb_swap_WR.md \
        reports/feature_probe_pbp_receiver_lgbnb_swap_WR.csv \
        reports/feature_probe_pbp_receiver_lgbnb_swap_TE.md \
        reports/feature_probe_pbp_receiver_lgbnb_swap_TE.csv
git commit -m "report(pbp-receiver): lgb-nb augment + swap probe outputs (baseline NULL trigger)"
```

If lgb-nb did not run (baseline returned SIGNAL), no commit needed for this task.

---

### Task 13: Write summary report; commit

**Files:**
- Create: `reports/feature_probe_pbp_receiver_summary.md`

- [ ] **Step 1: Use PR #20's summary as the structural template**

Open `reports/feature_probe_pbp_family_summary.md` for reference. The receiver summary mirrors its structure.

- [ ] **Step 2: Write the summary report**

Create `reports/feature_probe_pbp_receiver_summary.md` with these sections (filling in the actual numbers from the committed CSVs):

```markdown
# PBP Receiver Family Probe — Summary

**Date:** 2026-05-01
**Branch:** `feat/wr-te-pbp-features`
**Spec:** `docs/superpowers/specs/2026-05-01-wr-te-pbp-receiver-features-design.md`
**Plan:** `docs/superpowers/plans/2026-05-01-wr-te-pbp-receiver-features.md`
**Override:** `data/features_probe/pbp_receiver.parquet` (regenerable; not committed)
**Override generator:** `scripts/build_pbp_receiver_override.py`

The four PBP-derived player-level (receiver) features `aDOT_l4`,
`deep_target_share_l4`, `yac_per_reception_l4`, `red_zone_target_share_l4`
were bundled into a single override and probed in two modes (augment, swap)
at the BaselineModel level against the v1 baseline features for WR + TE.
Swap mode dropped the NGS season-snapshot analogs `avg_intended_air_yards_std`
and `avg_yac_above_expectation_std`. Family verdict applies the §4 rule
across the executed reports.

## Per-mode summary

| Model | Mode | Pos | Pooled SIGNAL cells | Pooled REGRESSION cells | Composite Phase 2 verdict | RMSE delta (fpts) | RMSE 95% CI |
|---|---|---|---:|---:|---|---:|---|
| baseline | augment | WR | <fill> | <fill> | <fill> | <fill> | <fill> |
| baseline | augment | TE | <fill> | <fill> | <fill> | <fill> | <fill> |
| baseline | swap    | WR | <fill> | <fill> | <fill> | <fill> | <fill> |
| baseline | swap    | TE | <fill> | <fill> | <fill> | <fill> | <fill> |
<!-- Add lgb-nb rows here only if Task 12 step 2b-c ran. -->

### Phase 1 SIGNAL cells (pooled across years)

<!-- Fill from CSVs; one row per (mode, position, stat) where verdict == SIGNAL pooled -->

### Phase 1 REGRESSION cells (pooled)

<!-- Fill from CSVs; one row per (mode, position, stat) where verdict == REGRESSION pooled -->

## Family verdict

**`<SIGNAL or NULL>`** (computed by the spec §4 rule via `family_verdict_from_reports`).

<!-- If SIGNAL: state the (position, stat, mode, model) tuples that lit up. -->
<!-- If NULL after lgb-nb also ran: state the family is closed across BaselineModel + lgb-nb for receivers. -->

## Decision

<!-- One paragraph. -->
<!-- If SIGNAL: name the candidate production-builder follow-up plan and what it scopes (WR-only, TE-only, or both). -->
<!-- If NULL: name the family closed; cross-reference TODO #3c; note that other refined-unit candidates (per-route-concept distributions, target-quality residuals) remain unexplored. -->

## Cross-references

- Per-position augment reports: `reports/feature_probe_pbp_receiver_augment_{WR,TE}.{md,csv}`
- Per-position swap reports: `reports/feature_probe_pbp_receiver_swap_{WR,TE}.{md,csv}`
<!-- If lgb-nb ran, add the corresponding lgbnb report cross-references. -->
- Spec: `docs/superpowers/specs/2026-05-01-wr-te-pbp-receiver-features-design.md`
- Plan: `docs/superpowers/plans/2026-05-01-wr-te-pbp-receiver-features.md`
- TODO #3c: cross-reference will be appended in the same commit cluster.
- Predecessor (team-level family): `reports/feature_probe_pbp_family_summary.md` (RB SIGNAL + WR/TE NULL at team-level granularity).
```

Replace every `<fill>` and `<...>` placeholder with values from the committed CSVs. The CSVs are read by the existing probe-report parser; if a small Python helper script is needed to render the table, write it inline:

```python
python - <<'PY'
import csv
from pathlib import Path

for pos in ("WR", "TE"):
    for mode in ("augment", "swap"):
        path = Path(f"reports/feature_probe_pbp_receiver_{mode}_{pos}.csv")
        with path.open() as f:
            rows = list(csv.DictReader(f))
        # ... extract pooled SIGNAL count, REGRESSION count, Phase 2 verdict, etc.
        # Use the row schema documented in src/projections/backtest/feature_probe.py.
        print(pos, mode, len(rows))  # placeholder; replace with the table-row print
PY
```

- [ ] **Step 3: Commit the summary report**

```bash
git add reports/feature_probe_pbp_receiver_summary.md
git commit -m "report(pbp-receiver): family summary — verdict $(VERDICT)"
```

Replace `$(VERDICT)` with the actual verdict (`SIGNAL` or `NULL`) from Task 12.

---

### Task 14: Update TODO #3c + project_management.md decision log

**Files:**
- Modify: `TODO.md` (entry #3c)
- Modify: `project_management.md` (top-of-file decision-log entry)

- [ ] **Step 1: Update TODO.md #3c**

Open `TODO.md`. Find entry `### 3c. Remaining PBP-derived feature plans (open)`. Append a paragraph at the bottom of that entry:

```markdown
**Update 2026-05-01 (WR/TE PBP receiver-level family probe, branch `feat/wr-te-pbp-features`):** Player-level air-yards / aDOT family probe shipped per `docs/superpowers/specs/2026-05-01-wr-te-pbp-receiver-features-design.md`. Bundled four player-level PBP features (`aDOT_l4`, `deep_target_share_l4`, `yac_per_reception_l4`, `red_zone_target_share_l4`) into a single override and probed in two modes (augment + swap) at the BaselineModel level for WR + TE only (QB excluded per PR #20's regression; RB has team-level shipped). **Family verdict: `<SIGNAL or NULL>`** — <one-line elaboration: where the signal lives, or what's closed>. <If SIGNAL: greenlights a follow-up production-builder plan for [WR / TE / both]. If NULL: family closed at BaselineModel + lgb-nb for receivers; refined-unit candidates beyond air-yards / aDOT (per-route-concept, target-quality residuals) remain unexplored.> See `reports/feature_probe_pbp_receiver_summary.md`.
```

Replace `<SIGNAL or NULL>` and the elaborations with the actual verdict + content.

- [ ] **Step 2: Update project_management.md top-of-file decision log**

Open `project_management.md`. Insert a new section at the top (immediately after the "# Project Management" header and the first paragraph), pushing the current top entry ("RB PBP Features Integration") down:

```markdown
## PBP Receiver Family Probe — verdict <SIGNAL or NULL> for WR/TE; <greenlight or closure narrative> (2026-05-01, on branch `feat/wr-te-pbp-features`)

**Status:** Probe-only spec shipped per `docs/superpowers/specs/2026-05-01-wr-te-pbp-receiver-features-design.md` and plan `docs/superpowers/plans/2026-05-01-wr-te-pbp-receiver-features.md`. Implements 4 pure compute fns + as-of trailing-4 helper + assembler + validating wrapper in `src/projections/features/pbp_receiver_features.py`, the override-generator script `scripts/build_pbp_receiver_override.py`, and 17 synthetic-fixture tests (4 helper + 3 adot + 2 deep_target_share + 1 yac + 1 red_zone + 3 attach + 3 build) plus 4 CLI tests. mypy strict + ruff + ruff format clean on touched files.

**Verdict:** `<SIGNAL or NULL>`. <Per-position breakdown: WR result, TE result. If SIGNAL, name the cells that lit up (position × stat × mode × model). If NULL after lgb-nb also ran, name the family as closed across BaselineModel + lgb-nb for receivers.>

**What this <greenlights or closes>:** <One paragraph. If SIGNAL: name the candidate production-builder follow-up plan and what it scopes (WR-only, TE-only, or both). If NULL: name TODO #3c receiver-level path closed; refined-unit candidates beyond air-yards / aDOT (per-route-concept distributions, target-quality residuals) remain unexplored under TODO #3c.>

**Reports:** `reports/feature_probe_pbp_receiver_summary.md` + 4-or-8 per-(mode, position) .md/.csv files under `reports/feature_probe_pbp_receiver_*`.

---
```

Replace `<SIGNAL or NULL>` and all `<...>` placeholders with the actual content.

- [ ] **Step 3: Update the "Next action" section in project_management.md**

Find the existing "## Next action" section. Replace its content (Track 2A WR/TE refined-unit PBP) with a current-state version reflecting the probe outcome:

If verdict is SIGNAL:

```markdown
## Next action

**Track 2A — WR/TE PBP receiver production-builder plan (next plan).** The PBP receiver family probe (PR #<TBD>) returned `SIGNAL` on <which positions / cells>. Scope a follow-up production-builder spec on a new branch (`feat/<wr|te|wr-te>-pbp-receiver-integration`) that promotes the 4 receiver-level PBP features into the corresponding `<Wr|Te>FeaturesSchema` and feature builder. Mirror PR #21's RB integration shape (helper extraction, schema append, builder wiring, refresh + adoption gate, ship/revert decision).

**Possible parallel / follow-on tracks (queued, not blocking the WR/TE production-builder work):**

1. **Other PBP feature families** (TODO #3c) — pressure rate allowed by O-line, red-zone usage shares (separate from the receiver-level RZ target share already tested here), team pace alone vs the bundled probe. Bundle 3-4 candidates per probe per the family-level prior.
2. **RB PBP × other model classes** — gate the 4 RB PBP cols against `lightgbm-tuned`, `lightgbm-nb`, and `ensemble`. Informational, not gating.

**Followup housekeeping (low-priority):** Model C-tuned strictly dominated by Model C-NB on RMSE — TODO #29 captures the pruning when ready.

After WR/TE PBP production-builder + the above tracks: Plan 4 (public API + CLI verbs + free-tier hosting), then Draft Hub.
```

If verdict is NULL:

```markdown
## Next action

**WR/TE PBP receiver-level family closed.** The PBP receiver family probe (PR #<TBD>) returned `NULL` across BaselineModel + lgb-nb for both WR and TE. The team-level cut (PR #20) was already null for these positions. **Refined unit candidates beyond air-yards / aDOT remain unexplored**: per-route-concept distributions (data not in current PBP subset), target-quality residuals (would need per-throw difficulty modeling), in-line vs flexed routes for TE (data not ingested). None of these are queued; revisit only if independent evidence suggests the unit choice was the binding constraint.

**Pivot to other PBP feature families** (TODO #3c) — pressure rate allowed by O-line, red-zone usage shares at the team level, team pace alone vs the bundled probe. Bundle 3-4 candidates per probe. This is the next active feature-class track.

**Possible parallel / follow-on tracks (queued):**

1. **RB PBP × other model classes** — gate the 4 RB PBP cols against `lightgbm-tuned`, `lightgbm-nb`, and `ensemble`. Informational, not gating.

**Followup housekeeping (low-priority):** Model C-tuned strictly dominated by Model C-NB on RMSE — TODO #29 captures the pruning when ready.

After the next PBP family probe + the above tracks: Plan 4 (public API + CLI verbs + free-tier hosting), then Draft Hub.
```

- [ ] **Step 4: Commit the docs**

```bash
git add TODO.md project_management.md
git commit -m "docs: log WR/TE PBP receiver family probe verdict — <SIGNAL or NULL>"
```

---

## Self-review summary

**Spec coverage check.** Each spec section has a corresponding task or note:

- §1.1 module + 6 functions → Tasks 1-7.
- §1.1 override script → Task 8.
- §1.1 tests (11 listed in spec §6.3) → Tasks 1-7 cover all 11 (4 trailing_4_asof + 3 adot + 2 deep_target_share + 1 yac + 1 red_zone + 3 attach + 3 build = 17 tests; spec listed 11 because it didn't enumerate the 4 trailing_4_asof helper tests separately).
- §1.1 probe runs (4-8 reports) → Tasks 11 (baseline) + 12 (conditional lgb-nb).
- §1.1 summary report → Task 13.
- §1.1 TODO #3c + project_management.md updates → Task 14.
- §1.3 success criteria → Task 10 (coverage), Task 11 (both baseline modes), Task 12 (lgb-nb if NULL).
- §6.2 script with `--seasons` / `--data-root` / `--output` / `--force` → Task 8.
- §6.4 no probe-CLI changes → Tasks 11-12 use existing CLI flags only.
- §9 CONTRIBUTING.md note → Task 8 step 5.

**Placeholder scan.** No "TBD" / "TODO" / "implement later" in the implementation tasks. Real-data tasks (10-14) intentionally have placeholders for verdict-dependent content (`<SIGNAL or NULL>`, `<greenlight or closure narrative>`); these are filled in at execution time from observable data, not at planning time.

**Type / name consistency.** `_trailing_4_per_player_asof` is consistently named across Task 1 (definition), Task 6 (assembler call site), and helper docstring. `_OUTPUT_COLUMNS` constant introduced in Task 6 is reused in Task 7. The `aDOT_l4` / `deep_target_share_l4` / `yac_per_reception_l4` / `red_zone_target_share_l4` column names appear consistently in spec §2.4, compute fn outputs (Tasks 2-5), assembler output (Task 6), and the override schema (Task 7).
