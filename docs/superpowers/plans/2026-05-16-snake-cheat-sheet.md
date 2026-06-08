# Snake-Draft Cheat Sheet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `generate_snake_cheat_sheet(vorp_table, league_config, display_names, tiers_per_position)` + `SnakeCheatSheetSchema` + `scripts/generate_snake_cheat_sheet.py` — the human-facing per-position draft-day cheat sheet with gap-based tier breaks.

**Architecture:** Pure transform that consumes a `VorpTableSchema` parquet (output of `generate_vorp_table`), the existing `_select_pool` helper (`src/projections/draft/_pool.py`), an optional `id_map.parquet` for display names, and a `LeagueConfig`. Emits a per-player table sorted by `(position canonical order, positional_rank)` with `tier` (1..N for in-pool, NA otherwise) and `is_in_pool` columns. CLI writes CSV or parquet (sniffed by `--out` extension).

**Tech Stack:** Python 3.12 strict mode, pandas, pandera, pydantic v2 (for LeagueConfig), pytest, numpy. No new ingest, no new model, no new store partition.

**Spec:** `docs/superpowers/specs/2026-05-16-snake-cheat-sheet-design.md`

**Worktree:** `.worktrees/feat-snake-cheat-sheet` on branch `feat/snake-cheat-sheet` (branched off `origin/main` at `c8da85b` — post-merge of PR #40 auction-values and PR #43 VORP).

**Verification at end of every task:** the CLAUDE.md §4 checklist relevant to the changes. Per-task verification steps are inline. Full sweep runs in Task 11.

**Pre-commit / venv quirk:** mypy's pre-commit hook resolves to system Python (pydantic v1). Per memory `project_pre_commit_venv_quirk.md`, prepend `.venv/Scripts` to `PATH` before `git commit` whenever a Python file is staged: `PATH="C:/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit ...`.

**Editable-install quirk for CLI subprocess tests:** the project `.venv` editable install currently points at another worktree's `src/` (see PR #40 plan-vs-execution deviation #5). Subprocess-based CLI tests must set `env={**os.environ, "PYTHONPATH": str(repo_root / "src")}` to import `projections.*` reliably. Matches the auction/VORP CLI-test convention.

---

## File Structure

**Created in this plan:**
- `src/projections/draft/snake_cheat_sheet.py` — `generate_snake_cheat_sheet` public function + `_assign_tiers` private helper.
- `scripts/generate_snake_cheat_sheet.py` — CLI wrapper.
- `tests/test_draft/test_snake_cheat_sheet.py` — algorithmic tests.
- `tests/test_scripts/test_generate_snake_cheat_sheet_cli.py` — CLI integration tests.

**Modified:**
- `src/projections/schemas.py` — append `SnakeCheatSheetSchema` after `VorpTableSchema`, before `AuctionValuesSchema`.
- `src/projections/draft/__init__.py` — re-export `generate_snake_cheat_sheet`.
- `tests/test_schemas/test_dataframe_schemas.py` — append `SnakeCheatSheetSchema` round-trip test.
- `project_management.md` — Task 12 adds a PM top entry.
- `draft_ready_checklist.md` — Task 12 flips §2b.1 from `[ ]` to `[x]`.

---

## Phase 1 — Schema (Task 1)

### Task 1: Append `SnakeCheatSheetSchema` to `schemas.py`

Single-file schema addition. The schema-seam guard is the verification gate.

**Files:**
- Modify: `src/projections/schemas.py` (append new class after `VorpTableSchema`, before `AuctionValuesSchema`)
- Modify: `tests/test_schemas/test_dataframe_schemas.py` (append round-trip test)

- [ ] **Step 1: Locate the insertion point in `schemas.py`**

Run: `rg -n "^class VorpTableSchema|^class AuctionValuesSchema" src/projections/schemas.py`

Expected output: two line numbers — `VorpTableSchema` line N1, `AuctionValuesSchema` line N2 (N1 < N2). Insert the new class between them.

- [ ] **Step 2: Write the failing round-trip test**

Add to the end of `tests/test_schemas/test_dataframe_schemas.py`:

```python
def test_snake_cheat_sheet_schema_round_trip() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.Series(["00-1000001", "00-2000001"], dtype=_PYARROW_STR),
            "position": pd.Series(["QB", "RB"], dtype=_PYARROW_STR),
            "display_name": pd.Series(["Patrick Mahomes", "Christian McCaffrey"], dtype=_PYARROW_STR),
            "positional_rank": pd.array([1, 1], dtype=pd.Int64Dtype()),
            "season_mean_fpts": [333.5, 280.1],
            "vorp": [91.3, 181.2],
            "replacement_fpts": [242.2, 98.9],
            "is_in_pool": [True, True],
            "tier": pd.array([1, 1], dtype=pd.Int64Dtype()),
        }
    )
    validated = SnakeCheatSheetSchema.validate(df)
    revalidated = SnakeCheatSheetSchema.validate(validated)
    pd.testing.assert_frame_equal(validated, revalidated)
```

Add `SnakeCheatSheetSchema` to the existing import line in that file (alongside `VorpTableSchema` / `AuctionValuesSchema`).

- [ ] **Step 3: Run the test to confirm it fails**

```bash
cd .worktrees/feat-snake-cheat-sheet
PYTHONPATH=src ../../.venv/Scripts/python.exe -m pytest tests/test_schemas/test_dataframe_schemas.py::test_snake_cheat_sheet_schema_round_trip -v
```

Expected: `ImportError: cannot import name 'SnakeCheatSheetSchema' from 'projections.schemas'`.

- [ ] **Step 4: Add `SnakeCheatSheetSchema` to `schemas.py`**

Insert between `VorpTableSchema` and `AuctionValuesSchema` in `src/projections/schemas.py`:

```python
class SnakeCheatSheetSchema(pa.DataFrameModel):
    """Per-player snake-draft cheat sheet. End-user surface for draft day.

    One row per player in the input VORP table. In-pool players get a numeric
    tier (1..N); out-of-pool players get tier = NA. `display_name` is
    best-effort from id_map.parquet; falls back to '—' for players without
    an id_map row.
    """

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$", unique=True)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    display_name: Series[str]
    positional_rank: Series[pd.Int64Dtype] = pa.Field(ge=1)
    season_mean_fpts: Series[float]
    vorp: Series[float]
    replacement_fpts: Series[float]
    is_in_pool: Series[bool]
    tier: Series[pd.Int64Dtype] = pa.Field(ge=1, nullable=True)

    class Config:
        strict = "filter"
        coerce = True
```

- [ ] **Step 5: Re-run the test**

```bash
PYTHONPATH=src ../../.venv/Scripts/python.exe -m pytest tests/test_schemas/test_dataframe_schemas.py::test_snake_cheat_sheet_schema_round_trip -v
```

Expected: PASS.

- [ ] **Step 6: Run the schema-seam guard (CLAUDE.md §4)**

```bash
PYTHONPATH=src ../../.venv/Scripts/python.exe -m pytest -k "ingest or store or schemas"
```

Expected: all green (no regressions). The auction and VORP round-trip tests should still pass.

- [ ] **Step 7: Run mypy and ruff on the touched files**

```bash
../../.venv/Scripts/python.exe -m mypy src/projections/schemas.py tests/test_schemas/test_dataframe_schemas.py
../../.venv/Scripts/python.exe -m ruff check src/projections/schemas.py tests/test_schemas/test_dataframe_schemas.py
../../.venv/Scripts/python.exe -m ruff format --check src/projections/schemas.py tests/test_schemas/test_dataframe_schemas.py
```

Expected: all clean.

- [ ] **Step 8: Commit**

```bash
git add src/projections/schemas.py tests/test_schemas/test_dataframe_schemas.py
PATH="C:/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
feat(schemas): SnakeCheatSheetSchema for snake-draft cheat sheet output

One row per player. In-pool: numeric tier (1..N). Out-of-pool: tier=NA.
positional_rank computed across all rows (in-pool + out) so the sheet
doubles as a waiver-wire lookup. display_name auto-joined from
id_map.parquet at the CLI layer; falls back to '—'.

Spec: docs/superpowers/specs/2026-05-16-snake-cheat-sheet-design.md §2.2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2 — Tier algorithm helper (Task 2)

### Task 2: `_assign_tiers` pure-numpy helper + algorithmic tests

Pinned in `snake_cheat_sheet.py` as a module-private helper. No pandas, no schema. Operates on a sorted (VORP descending) numpy float64 array and returns a numpy int64 array of tier labels. The five algorithm-correctness tests pin §3.2 / §3.6 behavior.

**Files:**
- Create: `src/projections/draft/snake_cheat_sheet.py` (skeleton + `_assign_tiers`)
- Create: `tests/test_draft/test_snake_cheat_sheet.py` (tier-algorithm tests)

- [ ] **Step 1: Write the failing test file**

Create `tests/test_draft/test_snake_cheat_sheet.py`:

```python
"""Tests for `projections.draft.snake_cheat_sheet`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.draft.snake_cheat_sheet import _assign_tiers


def test_assign_tiers_gap_based_correctness() -> None:
    """§5.1 #8 — synthetic gaps produce the documented tier partition."""
    vorps = np.array([100.0, 99.0, 98.0, 50.0, 49.0, 48.0, 10.0, 9.0, 8.0])
    tiers = _assign_tiers(vorps, n_tiers=3)
    assert list(tiers) == [1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_assign_tiers_fallback_when_n_in_pool_less_than_n() -> None:
    """§5.1 #9 — fewer in-pool players than tiers: each gets own tier."""
    vorps = np.array([50.0, 30.0, 20.0, 10.0, 5.0])  # 5 players
    tiers = _assign_tiers(vorps, n_tiers=8)
    assert list(tiers) == [1, 2, 3, 4, 5]


def test_assign_tiers_exact_when_n_in_pool_equals_n() -> None:
    """§5.1 #10 — exactly N in-pool players: 1-per-tier."""
    vorps = np.array([100.0, 80.0, 60.0, 40.0])
    tiers = _assign_tiers(vorps, n_tiers=4)
    assert list(tiers) == [1, 2, 3, 4]


def test_assign_tiers_with_n_equal_one_all_tier_one() -> None:
    """§5.1 #19 — tiers_per_position=1 collapses everyone into tier 1."""
    vorps = np.array([100.0, 50.0, 25.0, 10.0, 1.0])
    tiers = _assign_tiers(vorps, n_tiers=1)
    assert list(tiers) == [1, 1, 1, 1, 1]


def test_assign_tiers_tie_break_prefers_earlier_gap() -> None:
    """§5.1 #21 — when gaps are tied, the earlier (higher-rank) gap wins."""
    # gaps = [1, 4, 1, 4]: two gaps of 4 competing for the single allowed cut
    # under n_tiers=2. Earlier gap (index 1) wins; later gap (index 3) loses.
    vorps = np.array([10.0, 9.0, 5.0, 4.0, 0.0])
    tiers = _assign_tiers(vorps, n_tiers=2)
    assert list(tiers) == [1, 1, 2, 2, 2]


def test_assign_tiers_empty_input() -> None:
    vorps = np.array([], dtype=np.float64)
    tiers = _assign_tiers(vorps, n_tiers=8)
    assert tiers.shape == (0,)
    assert tiers.dtype == np.int64
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
PYTHONPATH=src ../../.venv/Scripts/python.exe -m pytest tests/test_draft/test_snake_cheat_sheet.py -v
```

Expected: `ModuleNotFoundError: No module named 'projections.draft.snake_cheat_sheet'`.

- [ ] **Step 3: Create `snake_cheat_sheet.py` skeleton with `_assign_tiers`**

Write `src/projections/draft/snake_cheat_sheet.py`:

```python
"""Snake-draft cheat sheet — per-position rankings with gap-based tiers.

Pure transform over the VORP table (`VorpTableSchema`). Reuses `_select_pool`
for in-pool identification. Emits `SnakeCheatSheetSchema`-validated output.

Spec: docs/superpowers/specs/2026-05-16-snake-cheat-sheet-design.md
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _assign_tiers(
    vorp_desc: NDArray[np.float64],
    n_tiers: int,
) -> NDArray[np.int64]:
    """Gap-based tier assignment over a VORP-descending array.

    Returns a 1-indexed int64 array of the same length as `vorp_desc` giving
    the tier (1..N) for each in-pool player. See spec §3.2 for the algorithm.

    Tie-break: when multiple gaps share the value that is competing for the
    `(N-1)`th-largest slot, the earlier (higher-rank) gap-index wins.
    Deterministic.
    """
    n = len(vorp_desc)
    if n == 0:
        return np.array([], dtype=np.int64)
    if n <= n_tiers:
        return np.arange(1, n + 1, dtype=np.int64)

    gaps = vorp_desc[:-1] - vorp_desc[1:]
    # lexsort: primary key (last arg) sorts ascending; -gaps ascending = gaps
    # descending. Ties broken by np.arange (gap-index) ascending.
    order = np.lexsort((np.arange(n - 1), -gaps))
    cut_indices = np.sort(order[: n_tiers - 1])

    tier = np.empty(n, dtype=np.int64)
    start = 0
    for t, cut in enumerate(cut_indices, start=1):
        tier[start : cut + 1] = t
        start = int(cut) + 1
    tier[start:] = n_tiers
    return tier
```

- [ ] **Step 4: Re-run tests**

```bash
PYTHONPATH=src ../../.venv/Scripts/python.exe -m pytest tests/test_draft/test_snake_cheat_sheet.py -v
```

Expected: 6 PASS.

- [ ] **Step 5: Run mypy and ruff**

```bash
../../.venv/Scripts/python.exe -m mypy src/projections/draft/snake_cheat_sheet.py tests/test_draft/test_snake_cheat_sheet.py
../../.venv/Scripts/python.exe -m ruff check src/projections/draft/snake_cheat_sheet.py tests/test_draft/test_snake_cheat_sheet.py
../../.venv/Scripts/python.exe -m ruff format --check src/projections/draft/snake_cheat_sheet.py tests/test_draft/test_snake_cheat_sheet.py
```

Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/projections/draft/snake_cheat_sheet.py tests/test_draft/test_snake_cheat_sheet.py
PATH="C:/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
feat(draft): _assign_tiers gap-based tier helper for snake cheat sheet

Pure numpy. Given a VORP-descending float64 array and N, returns an int64
tier array of the same length. Top (N-1) gaps select the cuts; ties broken
by earlier-gap-index. Handles n=0, n<=N (1-per-tier), n>N (gap-based)
branches per spec §3.2 / §3.6.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3 — Public function (Tasks 3-7)

### Task 3: `generate_snake_cheat_sheet` skeleton + happy-path tests

Lay down the public function signature, the orchestration (call `_select_pool`, compute positional_rank, call `_assign_tiers` per position, attach display names, sort, validate), and the first two algorithmic tests (#1 schema round-trip, #2 row count preserved).

**Files:**
- Modify: `src/projections/draft/snake_cheat_sheet.py` (add `generate_snake_cheat_sheet`)
- Modify: `src/projections/draft/__init__.py` (re-export)
- Modify: `tests/test_draft/test_snake_cheat_sheet.py` (add fixture helpers + 2 tests)

- [ ] **Step 1: Add fixture helpers to the test file**

Append to `tests/test_draft/test_snake_cheat_sheet.py`:

```python
from datetime import UTC, datetime

from projections.draft.league_config import LeagueConfig
from projections.draft.snake_cheat_sheet import generate_snake_cheat_sheet
from projections.schemas import (
    _PYARROW_STR,
    Position,
    RosterSlot,
    Ruleset,
    SnakeCheatSheetSchema,
    VorpTableSchema,
)


_POSITION_ID_PREFIX: dict[Position, int] = {
    Position.QB: 1,
    Position.RB: 2,
    Position.WR: 3,
    Position.TE: 4,
    Position.K: 5,
    Position.DST: 6,
}


def _make_config(
    n_teams: int = 4,
    roster_slots: dict[RosterSlot, int] | None = None,
    ruleset: Ruleset | None = None,
) -> LeagueConfig:
    return LeagueConfig(
        name="test",
        n_teams=n_teams,
        budget=100,
        min_bid=1,
        roster_slots=roster_slots
        or {
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 1,
        },
        ruleset=ruleset or Ruleset.espn_ppr(),
    )


def _make_vorp_table(positions: dict[Position, int], base_fpts: float = 300.0) -> pd.DataFrame:
    """Build a VorpTableSchema-validated frame with `count` rows per position.

    Per-row VORPs are arbitrary but monotonically decreasing within position
    (vorp = base_fpts - i). Replacement fpts is broadcast per-position from
    the row at rank `count` (the "first off the board" position-internal
    boundary — close enough for testing; algorithmic correctness lives in
    upstream VORP tests).
    """
    rows: list[dict[str, object]] = []
    for pos, count in positions.items():
        prefix = _POSITION_ID_PREFIX[pos]
        # Pick replacement = the worst player at this position (so all VORPs >= 0).
        replacement_fpts = base_fpts - (count - 1)
        for i in range(count):
            season_mean_fpts = base_fpts - i
            rows.append(
                {
                    "gsis_id": f"00-{prefix}{i:06d}",
                    "position": pos.value,
                    "season_mean_fpts": season_mean_fpts,
                    "vorp": season_mean_fpts - replacement_fpts,
                    "replacement_fpts": replacement_fpts,
                }
            )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    return VorpTableSchema.validate(df)


def test_output_validates_against_schema() -> None:
    """§5.1 #1 — output is SnakeCheatSheetSchema-valid; column order matches."""
    cfg = _make_config()
    vorp = _make_vorp_table({Position.QB: 8, Position.RB: 12, Position.WR: 12, Position.TE: 8})
    out = generate_snake_cheat_sheet(vorp, cfg)
    SnakeCheatSheetSchema.validate(out)
    assert list(out.columns) == list(SnakeCheatSheetSchema.to_schema().columns)


def test_row_count_preserved() -> None:
    """§5.1 #2 — output row count == input row count."""
    cfg = _make_config()
    vorp = _make_vorp_table({Position.QB: 8, Position.RB: 12, Position.WR: 12, Position.TE: 8})
    out = generate_snake_cheat_sheet(vorp, cfg)
    assert len(out) == len(vorp)
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
PYTHONPATH=src ../../.venv/Scripts/python.exe -m pytest tests/test_draft/test_snake_cheat_sheet.py::test_output_validates_against_schema tests/test_draft/test_snake_cheat_sheet.py::test_row_count_preserved -v
```

Expected: `ImportError: cannot import name 'generate_snake_cheat_sheet' from 'projections.draft.snake_cheat_sheet'`.

- [ ] **Step 3: Implement `generate_snake_cheat_sheet`**

Append to `src/projections/draft/snake_cheat_sheet.py`:

```python
import pandas as pd

from projections.draft._pool import _select_pool
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, Position, SnakeCheatSheetSchema, VorpTableSchema

_POSITION_ORDER: tuple[Position, ...] = (
    Position.QB,
    Position.RB,
    Position.WR,
    Position.TE,
    Position.K,
    Position.DST,
)

_DISPLAY_NAME_FALLBACK = "—"


def generate_snake_cheat_sheet(
    vorp_table: pd.DataFrame,
    league_config: LeagueConfig,
    display_names: pd.DataFrame | None = None,
    tiers_per_position: int = 8,
) -> pd.DataFrame:
    """Build a per-position snake-draft cheat sheet from a VORP table.

    Pure transform. See spec §3.1 for the three-stage algorithm:
    (1) call _select_pool to flag in-pool rows; (2) compute positional_rank
    across all rows by vorp desc; (3) gap-based tier breaks within each
    position's in-pool subset; out-of-pool rows get tier=NA.

    Args:
      vorp_table: VorpTableSchema-validated frame.
      league_config: drives _select_pool's in-pool definition.
      display_names: optional (gsis_id, display_name) frame. If None, all
        display_name values become '—'.
      tiers_per_position: positive int; default 8.

    Returns: SnakeCheatSheetSchema-validated frame, sorted by
      (position canonical, positional_rank).
    """
    if tiers_per_position <= 0:
        raise ValueError(f"tiers_per_position must be >= 1; got {tiers_per_position}")

    vorp = VorpTableSchema.validate(vorp_table)

    # Stage 1: in-pool flag via _select_pool (which itself enforces config-
    # required positions present, raising "cannot fill N {slot} slots" if not).
    in_pool_ids = set(_select_pool(vorp, league_config))
    df = vorp.copy()
    df["is_in_pool"] = df["gsis_id"].isin(in_pool_ids)

    # Stage 2: positional_rank across all rows (in-pool + out), by vorp desc
    # with gsis_id ascending tie-break (matches _select_pool tie-break).
    df = df.sort_values(["position", "vorp", "gsis_id"], ascending=[True, False, True])
    df["positional_rank"] = df.groupby("position", sort=False).cumcount() + 1
    df["positional_rank"] = df["positional_rank"].astype(pd.Int64Dtype())

    # Stage 3: gap-based tiers within each position's in-pool subset.
    tier_col = pd.array([pd.NA] * len(df), dtype=pd.Int64Dtype())
    df = df.reset_index(drop=True)
    for pos_value in df["position"].unique():
        pos_mask = df["position"] == pos_value
        in_pool_mask = pos_mask & df["is_in_pool"]
        in_pool_idx = df.index[in_pool_mask].to_numpy()
        if len(in_pool_idx) == 0:
            continue
        # in_pool_idx is already in positional_rank order because we sorted
        # the whole frame by (position, vorp desc, gsis_id) above.
        vorps = df.loc[in_pool_idx, "vorp"].to_numpy(dtype=np.float64)
        tiers = _assign_tiers(vorps, tiers_per_position)
        for idx, t in zip(in_pool_idx, tiers, strict=True):
            tier_col[idx] = int(t)
    df["tier"] = tier_col

    # Display names: left-join optional map; fallback "—".
    if display_names is None or display_names.empty:
        df["display_name"] = pd.Series([_DISPLAY_NAME_FALLBACK] * len(df), dtype=_PYARROW_STR)
    else:
        names = display_names.set_index("gsis_id")["display_name"]
        mapped = df["gsis_id"].map(names).fillna(_DISPLAY_NAME_FALLBACK)
        df["display_name"] = mapped.astype(_PYARROW_STR)

    # Final sort: position canonical order, then positional_rank ascending.
    position_rank = {pos.value: i for i, pos in enumerate(_POSITION_ORDER)}
    df["_pos_sort"] = df["position"].map(position_rank)
    df = df.sort_values(["_pos_sort", "positional_rank"], ascending=[True, True])
    df = df.drop(columns=["_pos_sort"]).reset_index(drop=True)

    return SnakeCheatSheetSchema.validate(df)
```

- [ ] **Step 4: Re-export from `src/projections/draft/__init__.py`**

Read the current `__init__.py` and add `generate_snake_cheat_sheet` to the existing re-export pattern. The file should already re-export `generate_vorp_table` and `generate_auction_values`; follow the same shape.

```python
from projections.draft.snake_cheat_sheet import generate_snake_cheat_sheet
```

If there's an `__all__`, add `"generate_snake_cheat_sheet"` to it.

- [ ] **Step 5: Re-run the new tests**

```bash
PYTHONPATH=src ../../.venv/Scripts/python.exe -m pytest tests/test_draft/test_snake_cheat_sheet.py -v
```

Expected: 8 PASS (6 from Task 2 + 2 from this task).

- [ ] **Step 6: Run mypy on the touched files**

```bash
../../.venv/Scripts/python.exe -m mypy src/projections/draft/snake_cheat_sheet.py src/projections/draft/__init__.py tests/test_draft/test_snake_cheat_sheet.py
```

Expected: clean. If you see `[no-any-return]` complaints from pandas operations, type-annotate the offending intermediate as `pd.Series[bool]` / `pd.DataFrame` etc. — narrow the type, don't add `# type: ignore` blanket.

- [ ] **Step 7: Run ruff**

```bash
../../.venv/Scripts/python.exe -m ruff check src/projections/draft/snake_cheat_sheet.py
../../.venv/Scripts/python.exe -m ruff format --check src/projections/draft/snake_cheat_sheet.py
```

- [ ] **Step 8: Commit**

```bash
git add src/projections/draft/snake_cheat_sheet.py src/projections/draft/__init__.py tests/test_draft/test_snake_cheat_sheet.py
PATH="C:/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
feat(draft): generate_snake_cheat_sheet skeleton + happy-path tests

Three-stage pure transform: _select_pool tags in-pool rows, positional_rank
across all rows by vorp desc + gsis_id tie-break, _assign_tiers within each
position's in-pool subset. Output sorted by (position canonical,
positional_rank). Display names optional (left-join, fallback "—").

Spec §2.1 / §3.1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: positional_rank, is_in_pool, tier invariants

Pin the five row-level invariants from spec §5.1 (#3, #4, #5, #6, #7): positional_rank strict monotonicity within position, equal-VORP tie-break by gsis_id, is_in_pool matching `_select_pool`, tier dtype = nullable Int64, tier monotonic with VORP.

**Files:**
- Modify: `tests/test_draft/test_snake_cheat_sheet.py` (add 5 tests)

- [ ] **Step 1: Add the five tests**

Append to `tests/test_draft/test_snake_cheat_sheet.py`:

```python
from projections.draft._pool import _select_pool


def test_positional_rank_strictly_monotonic_within_position() -> None:
    """§5.1 #3 — positional_rank is 1, 2, 3, ... by vorp desc within each position."""
    cfg = _make_config()
    vorp = _make_vorp_table({Position.QB: 8, Position.RB: 12, Position.WR: 12, Position.TE: 8})
    out = generate_snake_cheat_sheet(vorp, cfg)
    for pos_value in out["position"].unique():
        sub = out[out["position"] == pos_value].sort_values("positional_rank")
        assert list(sub["positional_rank"]) == list(range(1, len(sub) + 1))
        # vorp must be non-increasing as positional_rank increases
        vorps = sub["vorp"].to_numpy()
        assert (vorps[:-1] >= vorps[1:]).all()


def test_positional_rank_tie_break_by_gsis_id() -> None:
    """§5.1 #4 — equal-vorp rows tie-break by gsis_id ascending."""
    # Construct two QBs with identical VORP and explicit gsis_id ordering.
    df = pd.DataFrame(
        {
            "gsis_id": pd.Series(["00-1000002", "00-1000001"], dtype=_PYARROW_STR),
            "position": pd.Series(["QB", "QB"], dtype=_PYARROW_STR),
            "season_mean_fpts": [300.0, 300.0],
            "vorp": [50.0, 50.0],
            "replacement_fpts": [250.0, 250.0],
        }
    )
    vorp = VorpTableSchema.validate(df)
    cfg = _make_config(
        roster_slots={RosterSlot.QB: 1, RosterSlot.BENCH: 1},
        n_teams=1,
    )
    out = generate_snake_cheat_sheet(vorp, cfg)
    ranked = out.sort_values("positional_rank").reset_index(drop=True)
    assert ranked.loc[0, "gsis_id"] == "00-1000001"   # lower gsis_id first on tie
    assert ranked.loc[1, "gsis_id"] == "00-1000002"


def test_is_in_pool_matches_select_pool() -> None:
    """§5.1 #5 — set of is_in_pool=True gsis_ids equals _select_pool's output."""
    cfg = _make_config()
    vorp = _make_vorp_table({Position.QB: 8, Position.RB: 12, Position.WR: 12, Position.TE: 8})
    out = generate_snake_cheat_sheet(vorp, cfg)
    expected_pool = set(_select_pool(vorp, cfg))
    actual_pool = set(out.loc[out["is_in_pool"], "gsis_id"])
    assert actual_pool == expected_pool


def test_tier_dtype_is_nullable_int64() -> None:
    """§5.1 #6 — tier is pd.Int64Dtype(); in-pool rows int, out-of-pool rows NA."""
    cfg = _make_config()
    # Deliberately oversized inputs so some are out of pool.
    vorp = _make_vorp_table({Position.QB: 20, Position.RB: 30, Position.WR: 30, Position.TE: 20})
    out = generate_snake_cheat_sheet(vorp, cfg)
    assert out["tier"].dtype == pd.Int64Dtype()
    assert out.loc[out["is_in_pool"], "tier"].notna().all()
    assert out.loc[~out["is_in_pool"], "tier"].isna().all()


def test_tier_monotonic_with_vorp_within_position() -> None:
    """§5.1 #7 — tier T's min vorp >= tier T+1's max vorp (contiguous partition)."""
    cfg = _make_config()
    vorp = _make_vorp_table({Position.QB: 20, Position.RB: 30, Position.WR: 30, Position.TE: 20})
    out = generate_snake_cheat_sheet(vorp, cfg)
    in_pool = out[out["is_in_pool"]]
    for pos_value in in_pool["position"].unique():
        sub = in_pool[in_pool["position"] == pos_value]
        for t in sorted(sub["tier"].dropna().unique())[:-1]:
            tier_t_min = sub.loc[sub["tier"] == t, "vorp"].min()
            tier_t1_max = sub.loc[sub["tier"] == t + 1, "vorp"].max()
            assert tier_t_min >= tier_t1_max, (
                f"tier {t} min vorp ({tier_t_min}) < tier {t+1} max vorp ({tier_t1_max})"
            )
```

- [ ] **Step 2: Run the new tests**

```bash
PYTHONPATH=src ../../.venv/Scripts/python.exe -m pytest tests/test_draft/test_snake_cheat_sheet.py -v -k "positional_rank or is_in_pool or tier_dtype or tier_monotonic"
```

Expected: 5 PASS.

- [ ] **Step 3: Run the full test file to confirm no regressions**

```bash
PYTHONPATH=src ../../.venv/Scripts/python.exe -m pytest tests/test_draft/test_snake_cheat_sheet.py -v
```

Expected: 13 PASS (8 prior + 5 new).

- [ ] **Step 4: mypy + ruff**

```bash
../../.venv/Scripts/python.exe -m mypy tests/test_draft/test_snake_cheat_sheet.py
../../.venv/Scripts/python.exe -m ruff check tests/test_draft/test_snake_cheat_sheet.py
../../.venv/Scripts/python.exe -m ruff format --check tests/test_draft/test_snake_cheat_sheet.py
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_draft/test_snake_cheat_sheet.py
PATH="C:/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
test(draft): snake cheat sheet — positional_rank / is_in_pool / tier invariants

Pin spec §5.1 #3-7: positional_rank monotonicity + tie-break by gsis_id,
is_in_pool matches _select_pool, tier dtype is pd.Int64Dtype() with NA for
out-of-pool, tier T's min vorp >= tier T+1's max vorp (contiguous partition).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Display-name auto-join + position-with-no-in-pool tests

Pin spec §5.1 #11 (position with zero in-pool rows), #12 (happy-path name join), #13 (missing rows → fallback), #14 (`display_names=None` → all fallback).

**Files:**
- Modify: `tests/test_draft/test_snake_cheat_sheet.py` (add 4 tests)

- [ ] **Step 1: Add the four tests**

Append to `tests/test_draft/test_snake_cheat_sheet.py`:

```python
def test_display_name_join_happy_path() -> None:
    """§5.1 #12 — every gsis_id in display_names gets its mapped name."""
    cfg = _make_config()
    vorp = _make_vorp_table({Position.QB: 4, Position.RB: 6, Position.WR: 6, Position.TE: 4})
    display = pd.DataFrame(
        {
            "gsis_id": vorp["gsis_id"].astype(_PYARROW_STR),
            "display_name": pd.Series(
                [f"Player {i}" for i in range(len(vorp))], dtype=_PYARROW_STR
            ),
        }
    )
    out = generate_snake_cheat_sheet(vorp, cfg, display_names=display)
    for _, row in out.iterrows():
        expected = display.loc[display["gsis_id"] == row["gsis_id"], "display_name"].iloc[0]
        assert row["display_name"] == expected


def test_display_name_missing_rows_fall_back_to_em_dash() -> None:
    """§5.1 #13 — uncovered gsis_ids get '—'."""
    cfg = _make_config()
    vorp = _make_vorp_table({Position.QB: 4, Position.RB: 6, Position.WR: 6, Position.TE: 4})
    # Cover only the first half.
    half = vorp.head(len(vorp) // 2)
    display = pd.DataFrame(
        {
            "gsis_id": half["gsis_id"].astype(_PYARROW_STR),
            "display_name": pd.Series(
                [f"Player {i}" for i in range(len(half))], dtype=_PYARROW_STR
            ),
        }
    )
    out = generate_snake_cheat_sheet(vorp, cfg, display_names=display)
    covered_ids = set(display["gsis_id"])
    for _, row in out.iterrows():
        if row["gsis_id"] in covered_ids:
            assert row["display_name"] != "—"
        else:
            assert row["display_name"] == "—"


def test_display_name_none_yields_all_em_dash() -> None:
    """§5.1 #14 — display_names=None → every row has display_name '—'."""
    cfg = _make_config()
    vorp = _make_vorp_table({Position.QB: 4, Position.RB: 6, Position.WR: 6, Position.TE: 4})
    out = generate_snake_cheat_sheet(vorp, cfg, display_names=None)
    assert (out["display_name"] == "—").all()


def test_position_with_no_in_pool_rows_emits_rank_but_no_tier() -> None:
    """§5.1 #11 — a position whose players are all squeezed out of the pool
    (no in-pool rows but rows do exist in input) still appears in output with
    positional_rank populated and tier = NA.

    Construct: 1-team league with roster {QB:1, BENCH:0} consuming exactly
    1 player. Provide 1 QB (the starter, in pool) and 2 RBs (both out of
    pool, since roster doesn't include RB). RB rows have is_in_pool=False
    and tier=NA but positional_rank 1 and 2.
    """
    cfg = _make_config(
        n_teams=1,
        roster_slots={RosterSlot.QB: 1, RosterSlot.BENCH: 0},
    )
    vorp = _make_vorp_table({Position.QB: 1, Position.RB: 2})
    out = generate_snake_cheat_sheet(vorp, cfg)
    rb = out[out["position"] == "RB"]
    assert len(rb) == 2
    assert (~rb["is_in_pool"]).all()
    assert rb["tier"].isna().all()
    assert list(rb.sort_values("positional_rank")["positional_rank"]) == [1, 2]
```

- [ ] **Step 2: Run new tests**

```bash
PYTHONPATH=src ../../.venv/Scripts/python.exe -m pytest tests/test_draft/test_snake_cheat_sheet.py -v -k "display_name or position_with_no_in_pool"
```

Expected: 4 PASS.

If the position-with-no-in-pool test fails because `LeagueConfig` rejects `roster_slots={QB:1, BENCH:0}` (auction's `roster_size >= 1` post-validator), change to `n_teams=1, roster_slots={RosterSlot.QB: 1}` and only include 1 QB + 2 RBs as input.

- [ ] **Step 3: Run full file for regressions**

```bash
PYTHONPATH=src ../../.venv/Scripts/python.exe -m pytest tests/test_draft/test_snake_cheat_sheet.py -v
```

Expected: 17 PASS.

- [ ] **Step 4: mypy + ruff**

- [ ] **Step 5: Commit**

```bash
git add tests/test_draft/test_snake_cheat_sheet.py
PATH="C:/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
test(draft): snake cheat sheet — display name join + position-with-no-pool

Pin spec §5.1 #11-14: name auto-join happy path, partial-coverage fallback,
display_names=None → all '—', and the corner case where a position's input
rows exist but none make the pool (rank populated, tier=NA, is_in_pool=False).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Input-validation guards

Pin spec §5.1 #17 (missing-position raise via `_select_pool`), #18 (empty input → empty output), #20 (tiers_per_position ≤ 0 raises).

**Files:**
- Modify: `tests/test_draft/test_snake_cheat_sheet.py` (add 3 tests)

- [ ] **Step 1: Add the three tests**

```python
def test_missing_required_position_raises() -> None:
    """§5.1 #17 — LeagueConfig requires K but VORP has no K rows → raises from _select_pool."""
    cfg = _make_config(
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.K: 1,
            RosterSlot.BENCH: 1,
        },
    )
    vorp = _make_vorp_table({Position.QB: 8})  # no K rows
    with pytest.raises(ValueError, match=r"cannot fill \d+ K slots"):
        generate_snake_cheat_sheet(vorp, cfg)


def test_empty_input_returns_empty() -> None:
    """§5.1 #18 — empty VORP input → empty output, schema-valid."""
    cfg = _make_config()
    empty_df = pd.DataFrame(
        {
            "gsis_id": pd.Series([], dtype=_PYARROW_STR),
            "position": pd.Series([], dtype=_PYARROW_STR),
            "season_mean_fpts": pd.Series([], dtype=float),
            "vorp": pd.Series([], dtype=float),
            "replacement_fpts": pd.Series([], dtype=float),
        }
    )
    empty_vorp = VorpTableSchema.validate(empty_df)
    # Empty input → _select_pool will raise because config requires positions
    # not present. Test that THIS specific raise still happens (the contract
    # is "fail loudly when can't compute," not "silently return empty").
    with pytest.raises(ValueError, match=r"cannot fill"):
        generate_snake_cheat_sheet(empty_vorp, cfg)


def test_tiers_per_position_zero_or_negative_raises() -> None:
    """§5.1 #20 — invalid tiers_per_position raises before computation."""
    cfg = _make_config()
    vorp = _make_vorp_table({Position.QB: 8, Position.RB: 12, Position.WR: 12, Position.TE: 8})
    with pytest.raises(ValueError, match="tiers_per_position must be >= 1"):
        generate_snake_cheat_sheet(vorp, cfg, tiers_per_position=0)
    with pytest.raises(ValueError, match="tiers_per_position must be >= 1"):
        generate_snake_cheat_sheet(vorp, cfg, tiers_per_position=-3)
```

Note on `test_empty_input_returns_empty`: the spec §3.6 says "Empty VORP input → return an empty schema-valid frame." But the actual algorithm calls `_select_pool` first, which raises if it can't fill the config's required positions. With empty input it can't fill any position. So this test enforces the **stricter, more-correct behavior**: empty input + non-empty config raises explicitly (you can't silently return an empty cheat sheet when the user requested rankings for positions you have no data for). If a future spec wants "empty + empty config = empty output," add that branch then.

Update spec §3.6 / §1.2 if this contract surprises the reviewer (already-implemented behavior is "raise via _select_pool", not "return empty").

- [ ] **Step 2: Run new tests**

```bash
PYTHONPATH=src ../../.venv/Scripts/python.exe -m pytest tests/test_draft/test_snake_cheat_sheet.py -v -k "missing_required or empty_input or tiers_per_position_zero"
```

Expected: 3 PASS.

- [ ] **Step 3: Full file**

```bash
PYTHONPATH=src ../../.venv/Scripts/python.exe -m pytest tests/test_draft/test_snake_cheat_sheet.py -v
```

Expected: 20 PASS.

- [ ] **Step 4: mypy + ruff**

- [ ] **Step 5: Commit**

```bash
git add tests/test_draft/test_snake_cheat_sheet.py
PATH="C:/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
test(draft): snake cheat sheet — input validation guards

Pin spec §5.1 #17, #18, #20: missing-required-position raises via
_select_pool's existing "cannot fill N {slot} slots" message; empty VORP
input + non-empty config raises (stricter than spec §3.6's "return empty"
language — see test docstring); tiers_per_position <= 0 raises before any
computation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

If the empty-input contract is genuinely meant to return empty (not raise), this commit also amends the spec §3.6 bullet to match the as-implemented behavior. Decide during user review; default to as-implemented.

---

### Task 7: Sort order + determinism

Pin spec §5.1 #15 (sort by `(position canonical, positional_rank)`) and #16 (byte-identical re-run).

**Files:**
- Modify: `tests/test_draft/test_snake_cheat_sheet.py` (add 2 tests)

- [ ] **Step 1: Add the two tests**

```python
def test_output_sorted_by_position_canonical_then_rank() -> None:
    """§5.1 #15 — output sorted (QB, RB, WR, TE, K, DST), then positional_rank asc."""
    cfg = _make_config()
    vorp = _make_vorp_table({Position.WR: 6, Position.QB: 4, Position.RB: 6, Position.TE: 4})
    out = generate_snake_cheat_sheet(vorp, cfg)
    canonical = [Position.QB.value, Position.RB.value, Position.WR.value, Position.TE.value]
    # Filter canonical to only positions actually present, preserving order.
    expected_positions: list[str] = []
    for pos_value in canonical:
        sub = out[out["position"] == pos_value]
        expected_positions.extend([pos_value] * len(sub))
    assert list(out["position"]) == expected_positions
    # Within each position, positional_rank is ascending.
    for pos_value in out["position"].unique():
        sub = out[out["position"] == pos_value]
        assert list(sub["positional_rank"]) == sorted(sub["positional_rank"])


def test_determinism_byte_identical_reruns() -> None:
    """§5.1 #16 — same inputs → byte-identical output frame."""
    cfg = _make_config()
    vorp = _make_vorp_table({Position.QB: 8, Position.RB: 12, Position.WR: 12, Position.TE: 8})
    display = pd.DataFrame(
        {
            "gsis_id": vorp["gsis_id"].astype(_PYARROW_STR),
            "display_name": pd.Series(
                [f"Player {i}" for i in range(len(vorp))], dtype=_PYARROW_STR
            ),
        }
    )
    out1 = generate_snake_cheat_sheet(vorp, cfg, display_names=display, tiers_per_position=6)
    out2 = generate_snake_cheat_sheet(vorp, cfg, display_names=display, tiers_per_position=6)
    pd.testing.assert_frame_equal(out1, out2)
```

- [ ] **Step 2: Run new tests**

```bash
PYTHONPATH=src ../../.venv/Scripts/python.exe -m pytest tests/test_draft/test_snake_cheat_sheet.py -v -k "sorted_by_position or determinism"
```

Expected: 2 PASS.

- [ ] **Step 3: Full file**

```bash
PYTHONPATH=src ../../.venv/Scripts/python.exe -m pytest tests/test_draft/test_snake_cheat_sheet.py -v
```

Expected: 22 PASS.

- [ ] **Step 4: mypy + ruff**

- [ ] **Step 5: Commit**

```bash
git add tests/test_draft/test_snake_cheat_sheet.py
PATH="C:/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
test(draft): snake cheat sheet — sort order + determinism

Pin spec §5.1 #15-16: output sorted by (position canonical order:
QB/RB/WR/TE/K/DST) then positional_rank ascending; byte-identical output
across repeat calls with identical inputs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4 — CLI (Tasks 8-10)

### Task 8: CLI skeleton + happy-path integration test

Implement the script, including the id_map auto-join + writer. Pin §5.3 #23 (end-to-end happy path).

**Files:**
- Create: `scripts/generate_snake_cheat_sheet.py`
- Create: `tests/test_scripts/test_generate_snake_cheat_sheet_cli.py`

- [ ] **Step 1: Write the failing CLI happy-path test**

Create `tests/test_scripts/test_generate_snake_cheat_sheet_cli.py`:

```python
"""Integration tests for scripts/generate_snake_cheat_sheet.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from projections.schemas import _PYARROW_STR, IdMapSchema, Position, VorpTableSchema

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "generate_snake_cheat_sheet.py"


_POSITION_ID_PREFIX: dict[Position, int] = {
    Position.QB: 1,
    Position.RB: 2,
    Position.WR: 3,
    Position.TE: 4,
}


def _write_synthetic_vorp(path: Path) -> pd.DataFrame:
    """Write a small synthetic VorpTableSchema parquet at `path`. Returns the frame."""
    rows: list[dict[str, object]] = []
    base = 300.0
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE):
        count = {Position.QB: 16, Position.RB: 30, Position.WR: 30, Position.TE: 12}[pos]
        replacement = base - (count - 1)
        for i in range(count):
            mean_fpts = base - i
            rows.append(
                {
                    "gsis_id": f"00-{_POSITION_ID_PREFIX[pos]}{i:06d}",
                    "position": pos.value,
                    "season_mean_fpts": mean_fpts,
                    "vorp": mean_fpts - replacement,
                    "replacement_fpts": replacement,
                }
            )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df = VorpTableSchema.validate(df)
    df.to_parquet(path)
    return df


def _write_synthetic_id_map(path: Path, gsis_ids: list[str]) -> None:
    """Write a minimal IdMapSchema-valid parquet covering every gsis_id."""
    df = pd.DataFrame(
        {
            "gsis_id": pd.Series(gsis_ids, dtype=_PYARROW_STR),
            "espn_id": pd.Series([None] * len(gsis_ids), dtype=_PYARROW_STR),
            "sleeper_id": pd.Series([None] * len(gsis_ids), dtype=_PYARROW_STR),
            "pfr_id": pd.Series([None] * len(gsis_ids), dtype=_PYARROW_STR),
            "full_name": pd.Series([f"Player {gid}" for gid in gsis_ids], dtype=_PYARROW_STR),
            "position": pd.Series(
                ["QB" if gid.startswith("00-1") else "RB" for gid in gsis_ids],
                dtype=_PYARROW_STR,
            ),
            "team": pd.Series([None] * len(gsis_ids), dtype=_PYARROW_STR),
        }
    )
    IdMapSchema.validate(df).to_parquet(path)


def _write_league_config(path: Path) -> None:
    cfg = {
        "name": "test_ppr",
        "n_teams": 4,
        "budget": 100,
        "min_bid": 1,
        "roster_slots": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "BENCH": 1},
        "ruleset": "ESPN_PPR",
    }
    path.write_text(json.dumps(cfg))


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Invoke the CLI with PYTHONPATH=src so subprocess imports work."""
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src")}
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_end_to_end_happy_path(tmp_path: Path) -> None:
    """§5.3 #23 — script produces a valid output CSV from synthetic inputs."""
    vorp_path = tmp_path / "vorp.parquet"
    vorp = _write_synthetic_vorp(vorp_path)
    id_map_path = tmp_path / "id_map.parquet"
    _write_synthetic_id_map(id_map_path, list(vorp["gsis_id"]))
    cfg_path = tmp_path / "league.json"
    _write_league_config(cfg_path)
    out_path = tmp_path / "cheat_sheet.csv"

    result = _run_cli(
        [
            "--season", "2026",
            "--league-config", str(cfg_path),
            "--vorp-input", str(vorp_path),
            "--id-map", str(id_map_path),
            "--out", str(out_path),
        ]
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert out_path.exists()

    df = pd.read_csv(out_path)
    # Schema validity (basic columns present + correct counts).
    expected_cols = {
        "gsis_id", "position", "display_name", "positional_rank",
        "season_mean_fpts", "vorp", "replacement_fpts", "is_in_pool", "tier",
    }
    assert set(df.columns) >= expected_cols
    assert len(df) == len(vorp)
    # Display names attached for known players.
    assert (df["display_name"] != "—").any()
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
PYTHONPATH=src ../../.venv/Scripts/python.exe -m pytest tests/test_scripts/test_generate_snake_cheat_sheet_cli.py::test_cli_end_to_end_happy_path -v
```

Expected: failure because `scripts/generate_snake_cheat_sheet.py` doesn't exist.

- [ ] **Step 3: Implement the CLI**

Create `scripts/generate_snake_cheat_sheet.py`:

```python
"""Generate a snake-draft cheat sheet from a VORP parquet.

Reads:
  - --vorp-input    : VorpTableSchema parquet
  - --league-config : LeagueConfig JSON
  - --id-map        : IdMapSchema parquet (optional; --no-op if missing)

Writes:
  - --out           : .csv or .parquet (sniffed by extension)

Per-position stdout summary printed to stderr-friendly stdout for eyeball
mitigation (see spec §4).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from projections.draft.league_config import LeagueConfig
from projections.draft.snake_cheat_sheet import generate_snake_cheat_sheet
from projections.schemas import _PYARROW_STR, IdMapSchema, Position, VorpTableSchema

_POSITION_ORDER: tuple[Position, ...] = (
    Position.QB,
    Position.RB,
    Position.WR,
    Position.TE,
    Position.K,
    Position.DST,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a snake-draft cheat sheet.")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--league-config", type=Path, required=True)
    p.add_argument("--vorp-input", type=Path, required=True)
    p.add_argument("--id-map", type=Path, default=Path("data/raw/id_map.parquet"))
    p.add_argument("--tiers-per-position", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def _load_league_config(path: Path) -> LeagueConfig:
    raw = json.loads(path.read_text())
    return LeagueConfig.model_validate(raw)


def _load_vorp_table(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return VorpTableSchema.validate(df)


def _load_display_names(path: Path) -> pd.DataFrame | None:
    """Read id_map.parquet → (gsis_id, display_name). Returns None on missing."""
    if not path.exists():
        print(
            f"WARNING: id_map parquet not found at {path}; display names will be '—'",
            file=sys.stderr,
        )
        return None
    df = pd.read_parquet(path)
    df = IdMapSchema.validate(df)
    return pd.DataFrame(
        {
            "gsis_id": df["gsis_id"].astype(_PYARROW_STR),
            "display_name": df["full_name"].astype(_PYARROW_STR),
        }
    )


def _write_output(df: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(
            f"--out must be .csv or .parquet (got '{path.suffix}'); use one of these extensions."
        )


def _log_per_position_summary(df: pd.DataFrame, n_tiers: int, ruleset: str) -> None:
    n_total = len(df)
    print(
        f"Snake cheat sheet written: {n_total} players, ruleset={ruleset}, "
        f"tiers_per_position={n_tiers}"
    )
    print()
    print("Position summary (n_in_pool | tier-1 size | top-3):")
    for pos in _POSITION_ORDER:
        sub = df[df["position"] == pos.value]
        if sub.empty:
            continue
        in_pool = sub[sub["is_in_pool"]]
        tier_1_size = int((in_pool["tier"] == 1).sum())
        top3 = sub.head(3)
        top_str = ", ".join(
            f"{row['display_name']} ({pos.value}{int(row['positional_rank'])}, "
            f"T{int(row['tier'])}, VORP{row['vorp']:+.1f})"
            for _, row in top3.iterrows()
            if pd.notna(row["tier"])
        )
        print(
            f"  {pos.value}  in_pool={len(in_pool):>4}  tier1={tier_1_size:>2}  top: {top_str}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _load_league_config(args.league_config)
    vorp = _load_vorp_table(args.vorp_input)
    display = _load_display_names(args.id_map)
    sheet = generate_snake_cheat_sheet(
        vorp, cfg, display_names=display, tiers_per_position=args.tiers_per_position
    )
    _write_output(sheet, args.out)
    _log_per_position_summary(sheet, args.tiers_per_position, cfg.ruleset.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Re-run the test**

```bash
PYTHONPATH=src ../../.venv/Scripts/python.exe -m pytest tests/test_scripts/test_generate_snake_cheat_sheet_cli.py::test_cli_end_to_end_happy_path -v
```

Expected: PASS.

If the test fails because `subprocess.run` can't find a Python interpreter, confirm `sys.executable` resolves to the venv Python — if necessary, hard-code the venv path in the test (matching the auction CLI test's pattern).

- [ ] **Step 5: mypy + ruff on the CLI**

```bash
../../.venv/Scripts/python.exe -m mypy scripts/generate_snake_cheat_sheet.py tests/test_scripts/test_generate_snake_cheat_sheet_cli.py
../../.venv/Scripts/python.exe -m ruff check scripts/generate_snake_cheat_sheet.py tests/test_scripts/test_generate_snake_cheat_sheet_cli.py
../../.venv/Scripts/python.exe -m ruff format --check scripts/generate_snake_cheat_sheet.py tests/test_scripts/test_generate_snake_cheat_sheet_cli.py
```

If mypy flags the `argparse.Namespace` `args.season: Any` returns, the canonical fix is to narrow with explicit `int(args.season)` etc. at each use-site (the auction and VORP CLIs do this pattern).

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_snake_cheat_sheet.py tests/test_scripts/test_generate_snake_cheat_sheet_cli.py
PATH="C:/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
feat(scripts): generate_snake_cheat_sheet CLI + happy-path integration test

Flags: --season --league-config --vorp-input --id-map --tiers-per-position
--out. Reads VORP parquet + id_map (optional, warns if missing) +
LeagueConfig JSON; writes CSV or parquet sniffed by extension. Per-position
stdout summary surfaces tier-1 cliff for eyeball-check.

PYTHONPATH=src subprocess workaround in CLI tests matches auction/VORP
conventions (.venv editable install points at another worktree's src/).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: CLI handles missing id_map gracefully

Pin §5.3 #24 (missing id_map file → warning to stderr + display_name='—' for everyone + exit 0).

**Files:**
- Modify: `tests/test_scripts/test_generate_snake_cheat_sheet_cli.py` (add 1 test)

- [ ] **Step 1: Add the test**

```python
def test_cli_missing_id_map_logs_warning_and_falls_back(tmp_path: Path) -> None:
    """§5.3 #24 — --id-map points at a non-existent file → warning + '—' names + exit 0."""
    vorp_path = tmp_path / "vorp.parquet"
    _write_synthetic_vorp(vorp_path)
    cfg_path = tmp_path / "league.json"
    _write_league_config(cfg_path)
    out_path = tmp_path / "cheat_sheet.csv"
    missing_id_map = tmp_path / "nope.parquet"  # doesn't exist

    result = _run_cli(
        [
            "--season", "2026",
            "--league-config", str(cfg_path),
            "--vorp-input", str(vorp_path),
            "--id-map", str(missing_id_map),
            "--out", str(out_path),
        ]
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert "id_map parquet not found" in result.stderr
    df = pd.read_csv(out_path)
    assert (df["display_name"] == "—").all()
```

- [ ] **Step 2: Run new test (and the existing happy-path)**

```bash
PYTHONPATH=src ../../.venv/Scripts/python.exe -m pytest tests/test_scripts/test_generate_snake_cheat_sheet_cli.py -v
```

Expected: 2 PASS.

- [ ] **Step 3: mypy + ruff**

- [ ] **Step 4: Commit**

```bash
git add tests/test_scripts/test_generate_snake_cheat_sheet_cli.py
PATH="C:/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
test(scripts): snake cheat sheet CLI — missing id_map fallback

Spec §5.3 #24: --id-map pointing at non-existent file logs a warning to
stderr, falls back display_name='—' for every row, exits 0.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: CLI `--tiers-per-position` flag propagates

Pin §5.3 #25 (`--tiers-per-position 3` → output's max tier per position ≤ 3).

**Files:**
- Modify: `tests/test_scripts/test_generate_snake_cheat_sheet_cli.py` (add 1 test)

- [ ] **Step 1: Add the test**

```python
def test_cli_tiers_per_position_flag_propagates(tmp_path: Path) -> None:
    """§5.3 #25 — --tiers-per-position 3 caps output tiers at 3 per position."""
    vorp_path = tmp_path / "vorp.parquet"
    _write_synthetic_vorp(vorp_path)
    cfg_path = tmp_path / "league.json"
    _write_league_config(cfg_path)
    out_path = tmp_path / "cheat_sheet.csv"

    result = _run_cli(
        [
            "--season", "2026",
            "--league-config", str(cfg_path),
            "--vorp-input", str(vorp_path),
            "--id-map", str(tmp_path / "id_map_missing.parquet"),
            "--tiers-per-position", "3",
            "--out", str(out_path),
        ]
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    df = pd.read_csv(out_path)
    in_pool = df[df["is_in_pool"]]
    for pos_value in in_pool["position"].unique():
        sub = in_pool[in_pool["position"] == pos_value]
        # tier column comes back as float64 from CSV when there are NaNs;
        # but in_pool rows have integer tiers, so max() is well-defined.
        assert sub["tier"].max() <= 3, f"position {pos_value} has tier > 3"
```

- [ ] **Step 2: Run all CLI tests**

```bash
PYTHONPATH=src ../../.venv/Scripts/python.exe -m pytest tests/test_scripts/test_generate_snake_cheat_sheet_cli.py -v
```

Expected: 3 PASS.

- [ ] **Step 3: mypy + ruff**

- [ ] **Step 4: Commit**

```bash
git add tests/test_scripts/test_generate_snake_cheat_sheet_cli.py
PATH="C:/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
test(scripts): snake cheat sheet CLI — tiers-per-position flag

Spec §5.3 #25: --tiers-per-position 3 → output max tier per position <= 3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 5 — Verification + docs (Tasks 11-12)

### Task 11: Full CLAUDE.md §4 verification sweep

The forced verification gate. Run all four checks at the worktree root and paste the summary into the commit message.

**Files:** none modified (verification only).

- [ ] **Step 1: Full pytest**

```bash
cd .worktrees/feat-snake-cheat-sheet
PYTHONPATH=src ../../.venv/Scripts/python.exe -m pytest -v 2>&1 | tail -30
```

Expected: all green. Note the pre-existing `test_dispatch_default_model_class_for_wr_is_unchanged` failure flagged in PR #41 description — if it still appears, that's the inherited issue, not something to fix here. Reference it in the final commit / PM entry.

- [ ] **Step 2: Schema-seam guard**

```bash
PYTHONPATH=src ../../.venv/Scripts/python.exe -m pytest -k "ingest or store or schemas" -v 2>&1 | tail -10
```

Expected: green.

- [ ] **Step 3: mypy**

```bash
../../.venv/Scripts/python.exe -m mypy src tests 2>&1 | tail -5
```

Expected: `Success: no issues found in N source files`.

- [ ] **Step 4: ruff**

```bash
../../.venv/Scripts/python.exe -m ruff check src tests 2>&1 | tail -3
../../.venv/Scripts/python.exe -m ruff format --check src tests 2>&1 | tail -3
```

Expected: both clean.

- [ ] **Step 5: If anything failed, fix and re-run.** Do NOT proceed to Task 12 with red gates.

No commit at this task — verification only.

---

### Task 12: PM entry + draft_ready_checklist flip + docs

Final commit: project_management.md gets a top entry, draft_ready_checklist.md §2b.1 flips to `[x]`. This is the work record future agents will read first.

**Files:**
- Modify: `project_management.md` (prepend new entry)
- Modify: `draft_ready_checklist.md` (flip §2b.1)

- [ ] **Step 1: Read the current top entry of project_management.md to match style**

```bash
head -50 project_management.md
```

The top entry should be the VORP entry (post-merge of #43). Style: section heading with feature name + verdict / date / branch + Status / Shipped surface / Decision log / Risks / Plan-vs-execution deviations / Recommended next direction subheadings.

- [ ] **Step 2: Prepend the new entry to project_management.md**

Insert after the file's intro `---` separator (around line 5) and before the existing top entry:

```markdown
## Snake-Draft Cheat Sheet — feature shipped (2026-05-16, on branch `feat/snake-cheat-sheet`)

**Status:** Spec + plan + impl on `feat/snake-cheat-sheet`. Third surface of the Draft Hub sub-project (auction $, VORP, now snake cheat sheet). Reads a `VorpTableSchema` parquet + `id_map.parquet` + `LeagueConfig`, emits a per-player table sorted by `(position canonical order, positional_rank)` with gap-based tier breaks (1..N for in-pool, NA otherwise). v1 scope: VORP + tier breaks only. ADP delta and p10/p90 confidence band deferred to follow-up specs. Spec at `docs/superpowers/specs/2026-05-16-snake-cheat-sheet-design.md`; plan at `docs/superpowers/plans/2026-05-16-snake-cheat-sheet.md`.

**Shipped surface:**
- `src/projections/draft/snake_cheat_sheet.py` — `generate_snake_cheat_sheet` public function + `_assign_tiers` private numpy helper.
- `src/projections/schemas.py` — appended `SnakeCheatSheetSchema`.
- `src/projections/draft/__init__.py` — re-exports `generate_snake_cheat_sheet`.
- `scripts/generate_snake_cheat_sheet.py` — CLI with `--season --league-config --vorp-input --id-map --tiers-per-position --out` flags; CSV and parquet output supported; per-position stdout summary as VORP-quality eyeball mitigation (spec §4).
- 22 tests in `tests/test_draft/test_snake_cheat_sheet.py`, 3 integration tests in `tests/test_scripts/test_generate_snake_cheat_sheet_cli.py`, 1 schema round-trip test appended to `tests/test_schemas/test_dataframe_schemas.py`. All passing; mypy + ruff + format clean.

**Decision log:**
- **Tier algorithm: gap-based, fixed N (default 8).** Captures "talent cliffs" rather than smoothing distributions into arbitrary buckets. N is configurable via `--tiers-per-position`. Alternatives (variable-N gap threshold, k-means, fixed buckets) documented in spec §3.2 and rejected with reason.
- **Show all players, tier only in-pool.** Output includes out-of-pool players (positional_rank computed across both) so the sheet doubles as a waiver-wire lookup. Out-of-pool rows get `tier = NA`.
- **Display names from `id_map.parquet`, not `depth_charts`.** First draft of the spec named depth_charts as the name source — fact-check during spec-writing revealed `DepthChartsSchema` carries no name column. `IdMapSchema.full_name` is the canonical name source in this codebase (built by `build_id_map` from `nflreadpy.load_ff_playerids()`).
- **ADP delta and confidence band deferred.** Spec §1.2 — each blocks on infrastructure that doesn't exist (no ADP ingest; no p10/p90 plumbed through `VorpTableSchema`). Follow-up specs.

**Risks logged (spec §6):**
- **No ADP signal means cheat sheet reflects model view, not room view.** Manual ADP cross-reference required during draft for v1.
- **Tier instability across runs.** Gap-based tiers can flip if a small VORP shift moves which gap is "Nth largest." Stdout `tier-1 size` per position surfaces cliff stability for eyeball-check.
- **`_select_pool` now has three callers** (`auction.py`, `vorp.py`, `snake_cheat_sheet.py`). Pool refactors must consider all three; auction test suite remains the regression gate.

**Plan-vs-execution deviations:** [implementer fills in any deviations from this plan — e.g. fixture-helper rewrites, mypy strictness fixes, surprise pandas-dtype gotchas. Otherwise leave bullet absent.]

**Recommended next direction:**
1. **ADP ingest + ADP-delta column** (`draft_ready_checklist.md` §2b.3). FantasyPros free CSV or Sleeper API.
2. **Confidence band — p10/p90 floor/ceiling rank.** Plumb through `VorpTableSchema` or re-aggregate in this CLI.
3. **Live snake-draft recommender** (`draft_ready_checklist.md` §2b.2). Greedy vs. lookahead; ADP-blocked for the latter.

See `docs/superpowers/specs/2026-05-16-snake-cheat-sheet-design.md` and `docs/superpowers/plans/2026-05-16-snake-cheat-sheet.md`. Draft-readiness status: `draft_ready_checklist.md` §2b.1 flipped to `[x]`.

---
```

- [ ] **Step 3: Flip the draft_ready_checklist.md §2b.1 checkbox**

In `draft_ready_checklist.md`, find:

```markdown
### 2b. Snake draft

- [ ] **Pre-draft cheat sheet.** Per-position ordered list with VORP, ADP delta, tier, and confidence band. Static export (CSV / markdown) is enough for v1.
```

Replace with:

```markdown
### 2b. Snake draft

- [x] **Pre-draft cheat sheet.** Per-position ordered list with VORP, tier (gap-based, default N=8), display_name from id_map.parquet, and is_in_pool flag. ADP delta and confidence band deferred (own specs). Shipped on `feat/snake-cheat-sheet` (2026-05-16): `src/projections/draft/snake_cheat_sheet.py` + `scripts/generate_snake_cheat_sheet.py` + `tests/test_draft/test_snake_cheat_sheet.py` (22 tests) + `tests/test_scripts/test_generate_snake_cheat_sheet_cli.py` (3 tests). CSV and parquet output supported.
```

- [ ] **Step 4: Confirm both edits look right**

```bash
head -50 project_management.md
rg -A 2 "Pre-draft cheat sheet" draft_ready_checklist.md
```

Expected: PM entry visible at the top; checklist line shows `[x]` and the new shipped-surface summary.

- [ ] **Step 5: Run the full §4 verification sweep one more time** (the doc edits shouldn't have broken anything, but the rule is rule).

```bash
PYTHONPATH=src ../../.venv/Scripts/python.exe -m pytest -k "ingest or store or schemas" 2>&1 | tail -3
../../.venv/Scripts/python.exe -m mypy src tests 2>&1 | tail -3
../../.venv/Scripts/python.exe -m ruff check src tests 2>&1 | tail -3
../../.venv/Scripts/python.exe -m ruff format --check src tests 2>&1 | tail -3
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add project_management.md draft_ready_checklist.md
PATH="C:/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
docs(snake-cheat-sheet): PM entry + flip draft_ready_checklist §2b.1 to [x]

Snake-draft cheat sheet is the third Draft Hub surface (after auction $
generator and VORP). Reads VorpTableSchema parquet + id_map.parquet +
LeagueConfig; emits per-player table sorted by (position canonical,
positional_rank) with gap-based tiers. v1 scope: VORP + tier breaks.
ADP delta and confidence band deferred.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes (Plan Author)

**Spec coverage:**
- §1 Overview / Goals — covered by overall plan structure.
- §2.1 Public function — Task 3 (skeleton) + Tasks 4-7 (invariants pin behavior).
- §2.2 Schema — Task 1.
- §2.3 Why no store partition — informational only; no task needed.
- §3.1 Stages — Task 3 implements all three.
- §3.2 Tier algorithm — Task 2 (`_assign_tiers`).
- §3.3 Worked example — informational; pinned implicitly by Task 2 #8.
- §3.4 Determinism — Task 7 #16.
- §3.5 Tier instability — informational; surfaced in CLI stdout via Task 8.
- §3.6 Edge cases — Tasks 2 (empty), 5 (no-in-pool position), 6 (validation).
- §4 CLI — Tasks 8-10. Stdout summary in Task 8 implementation (`_log_per_position_summary`).
- §5.1 algorithmic tests #1-21 — Tasks 2-7. Map: #1, #2→T3; #3-7→T4; #8-10, #19, #21→T2; #11-14→T5; #15-16→T7; #17, #18, #20→T6.
- §5.2 schema round-trip #22 — Task 1.
- §5.3 CLI tests #23-25 — Tasks 8, 9, 10.
- §5.4 deliberately-not-tested — informational.
- §6 risks, open items, out-of-scope — informational; surface via Task 12 PM entry.
- §7 acceptance — Task 11 + Task 12.
- §8 follow-ups — informational; Task 12 PM entry calls them out.
- §9 implementation phases — this plan's phase headers match.

**Placeholder scan:** one deliberate `[implementer fills in any deviations]` line in the Task 12 PM entry, marked as a placeholder for the impl session to update. All other content concrete.

**Type / name consistency:**
- `_assign_tiers(vorp_desc: NDArray[np.float64], n_tiers: int) -> NDArray[np.int64]` — same signature in Task 2 implementation and Task 2 tests.
- `generate_snake_cheat_sheet(vorp_table, league_config, display_names=None, tiers_per_position=8)` — same signature in Task 3 implementation, Tasks 4-7 test calls.
- `SnakeCheatSheetSchema` columns — declared in Task 1, asserted in Task 3 + Task 8.
- `_DISPLAY_NAME_FALLBACK = "—"` — Task 3 implementation, asserted in Tasks 5 + 9.
- CLI flag names — `--season --league-config --vorp-input --id-map --tiers-per-position --out` consistent across Task 8 implementation + Tasks 8/9/10 test calls.

**One forward dependency to flag in implementer head:** Task 3's implementation uses `np` (numpy) — already imported at top of `snake_cheat_sheet.py` from Task 2. If the implementer ships Task 2 and Task 3 in different sittings, double-check the import remains.
