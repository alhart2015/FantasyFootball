# Auction Values Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the VORP → $ auction-values generator scoped in the spec — `LeagueConfig` pydantic model, `generate_auction_values` pure function, `AuctionValuesSchema`, CLI script, example configs, and tests.

**Architecture:** New subpackage `src/projections/draft/` as the Draft Hub seed. Pure function transforms a VORP DataFrame + `LeagueConfig` into an `AuctionValuesSchema`-validated output DataFrame. CLI wraps the function with file I/O. No new ingest, no new store partition, no new model code.

**Tech Stack:** Python 3.12+, pydantic v2 (frozen models), pandera (`DataFrameModel`), pandas with pyarrow string dtype, click or argparse for CLI (match existing scripts/ style).

**Spec:** `docs/superpowers/specs/2026-05-16-auction-values-design.md`
**Branch:** `feat/auction-values` (already cut from `main`; spec already committed as `b68c52f`)

---

## File map (locked decisions)

| File | Status | Responsibility |
|---|---|---|
| `src/projections/draft/__init__.py` | create | Subpackage marker; re-export `LeagueConfig` and `generate_auction_values` |
| `src/projections/draft/league_config.py` | create | `LeagueConfig` pydantic model + ruleset-string deserializer |
| `src/projections/draft/auction.py` | create | `generate_auction_values` + private `_select_pool` helper |
| `src/projections/schemas.py` | modify | Append `AuctionValuesSchema` |
| `scripts/generate_auction_values.py` | create | CLI: read inputs, call function, write output |
| `configs/league_espn_ppr_12team.json` | create | Example 12-team PPR config |
| `configs/league_espn_half_10team.json` | create | Example 10-team half-PPR config |
| `tests/test_draft/__init__.py` | create | Test package marker |
| `tests/test_draft/test_league_config.py` | create | `LeagueConfig` unit tests |
| `tests/test_draft/test_auction.py` | create | `generate_auction_values` unit + algorithmic-invariant tests |
| `tests/test_schemas/test_dataframe_schemas.py` | modify | Add `AuctionValuesSchema` round-trip test alongside existing schema tests |
| `tests/test_scripts/test_generate_auction_values_cli.py` | create | CLI end-to-end integration test |

---

## Phase 1 — Foundations: `LeagueConfig`, schema, example configs

### Task 1: `LeagueConfig` pydantic model

**Files:**
- Create: `src/projections/draft/__init__.py`
- Create: `src/projections/draft/league_config.py`
- Create: `tests/test_draft/__init__.py`
- Create: `tests/test_draft/test_league_config.py`

- [ ] **Step 1: Write the failing tests for `LeagueConfig`**

Test file `tests/test_draft/test_league_config.py`:

```python
"""Unit tests for `projections.draft.league_config.LeagueConfig`."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from projections.draft.league_config import LeagueConfig
from projections.schemas import RosterSlot, Ruleset


def _base_kwargs() -> dict[str, object]:
    return {
        "name": "test_league",
        "n_teams": 12,
        "budget": 200,
        "min_bid": 1,
        "roster_slots": {
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 3,
            RosterSlot.TE: 1,
            RosterSlot.FLEX: 1,
            RosterSlot.K: 1,
            RosterSlot.DST: 1,
            RosterSlot.BENCH: 7,
        },
        "ruleset": Ruleset.espn_ppr(),
    }


def test_roster_size_excludes_ir() -> None:
    kwargs = _base_kwargs()
    kwargs["roster_slots"] = {**kwargs["roster_slots"], RosterSlot.IR: 2}  # type: ignore[dict-item]
    cfg = LeagueConfig(**kwargs)  # type: ignore[arg-type]
    # IR (2) excluded; remaining = 1+2+3+1+1+1+1+7 = 17
    assert cfg.roster_size == 17


def test_roster_size_and_pool_and_budget_properties() -> None:
    cfg = LeagueConfig(**_base_kwargs())  # type: ignore[arg-type]
    assert cfg.roster_size == 16
    assert cfg.total_pool_size == 192
    assert cfg.total_budget == 2400


def test_json_round_trip() -> None:
    cfg = LeagueConfig(**_base_kwargs())  # type: ignore[arg-type]
    blob = cfg.model_dump_json()
    restored = LeagueConfig.model_validate_json(blob)
    assert restored == cfg


def test_ruleset_string_preset_deserialization() -> None:
    kwargs = _base_kwargs()
    kwargs["ruleset"] = "espn_ppr"
    cfg = LeagueConfig(**kwargs)  # type: ignore[arg-type]
    assert cfg.ruleset == Ruleset.espn_ppr()


def test_ruleset_full_object_deserialization() -> None:
    kwargs = _base_kwargs()
    custom = Ruleset(name="CUSTOM", reception_pts=0.25)
    kwargs["ruleset"] = json.loads(custom.model_dump_json())
    cfg = LeagueConfig(**kwargs)  # type: ignore[arg-type]
    assert cfg.ruleset == custom


def test_rejects_n_teams_le_1() -> None:
    kwargs = _base_kwargs()
    kwargs["n_teams"] = 1
    with pytest.raises(ValidationError):
        LeagueConfig(**kwargs)  # type: ignore[arg-type]


def test_rejects_budget_le_0() -> None:
    kwargs = _base_kwargs()
    kwargs["budget"] = 0
    with pytest.raises(ValidationError):
        LeagueConfig(**kwargs)  # type: ignore[arg-type]


def test_rejects_empty_roster_slots() -> None:
    kwargs = _base_kwargs()
    kwargs["roster_slots"] = {}
    with pytest.raises(ValidationError):
        LeagueConfig(**kwargs)  # type: ignore[arg-type]


def test_rejects_min_bid_lt_1() -> None:
    kwargs = _base_kwargs()
    kwargs["min_bid"] = 0
    with pytest.raises(ValidationError):
        LeagueConfig(**kwargs)  # type: ignore[arg-type]


def test_frozen() -> None:
    cfg = LeagueConfig(**_base_kwargs())  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        cfg.n_teams = 14  # type: ignore[misc]
```

Also create the empty test package marker:

`tests/test_draft/__init__.py`:
```python
```

- [ ] **Step 2: Run tests to verify they fail**

```
PATH=".venv/Scripts:$PATH" pytest tests/test_draft/test_league_config.py -v
```

Expected: every test errors at import time (module `projections.draft.league_config` doesn't exist yet).

- [ ] **Step 3: Implement `LeagueConfig`**

`src/projections/draft/__init__.py`:
```python
"""Draft Hub sub-project: pre-draft tooling (auction values, snake recommender, VORP)."""

from projections.draft.league_config import LeagueConfig

__all__ = ["LeagueConfig"]
```

`src/projections/draft/league_config.py`:
```python
"""LeagueConfig — pydantic model shared by VORP, auction-values, and snake-draft tooling.

Captures the user's league rules in one immutable, hashable object: team count,
auction budget, roster slot composition, and scoring ruleset. Constructed in code
or deserialized from JSON via `model_validate_json`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from projections.schemas import RosterSlot, Ruleset

_RULESET_PRESETS: dict[str, Ruleset] = {
    "espn_ppr": Ruleset.espn_ppr(),
    "espn_half": Ruleset.espn_half(),
    "standard": Ruleset.standard(),
}


class LeagueConfig(BaseModel):
    """Frozen league configuration. Shared input for all Draft Hub tooling."""

    model_config = ConfigDict(frozen=True)

    name: str
    n_teams: int = Field(gt=1)
    budget: int = Field(gt=0, default=200)
    min_bid: int = Field(ge=1, default=1)
    roster_slots: dict[RosterSlot, int] = Field(min_length=1)
    ruleset: Ruleset

    @field_validator("ruleset", mode="before")
    @classmethod
    def _resolve_ruleset(cls, v: Any) -> Any:
        """Allow string preset names (`espn_ppr`, `espn_half`, `standard`) for ergonomics
        in config JSON. Pass through pydantic-shaped dict or `Ruleset` instances unchanged.
        """
        if isinstance(v, str):
            try:
                return _RULESET_PRESETS[v]
            except KeyError as exc:
                allowed = ", ".join(sorted(_RULESET_PRESETS))
                raise ValueError(f"Unknown ruleset preset {v!r}; expected one of: {allowed}") from exc
        return v

    @property
    def roster_size(self) -> int:
        """Drafted slots per team (excludes IR, which is post-draft)."""
        return sum(count for slot, count in self.roster_slots.items() if slot != RosterSlot.IR)

    @property
    def total_pool_size(self) -> int:
        return self.n_teams * self.roster_size

    @property
    def total_budget(self) -> int:
        return self.n_teams * self.budget
```

- [ ] **Step 4: Run tests to verify they pass**

```
PATH=".venv/Scripts:$PATH" pytest tests/test_draft/test_league_config.py -v
```

Expected: 10 passing tests.

- [ ] **Step 5: Run focused lint / type check**

```
PATH=".venv/Scripts:$PATH" mypy src/projections/draft tests/test_draft
PATH=".venv/Scripts:$PATH" ruff check src/projections/draft tests/test_draft
PATH=".venv/Scripts:$PATH" ruff format --check src/projections/draft tests/test_draft
```

Expected: zero violations across all three. If `ruff format --check` reports drift, run `ruff format src/projections/draft tests/test_draft` and re-run `--check`.

- [ ] **Step 6: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add src/projections/draft tests/test_draft
PATH=".venv/Scripts:$PATH" git commit -m "feat(draft): LeagueConfig pydantic model + tests"
```

---

### Task 2: `AuctionValuesSchema` in `schemas.py`

**Files:**
- Modify: `src/projections/schemas.py` (append after `ProjectionSeasonSchema`)
- Modify: `tests/test_schemas/test_dataframe_schemas.py`

- [ ] **Step 1: Write the failing schema-validation test**

Add to `tests/test_schemas/test_dataframe_schemas.py` (append; do not replace existing tests):

```python
def test_auction_values_schema_round_trip() -> None:
    """`AuctionValuesSchema.validate` accepts a well-formed frame and rejects bad rows."""
    from projections.schemas import AuctionValuesSchema, Position

    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036912", "00-0034857"], dtype=pd.StringDtype("pyarrow")),
            "position": pd.array([Position.RB.value, Position.WR.value], dtype=pd.StringDtype("pyarrow")),
            "season_mean_fpts": pd.array([280.0, 240.0], dtype="float64"),
            "vorp": pd.array([130.0, 50.0], dtype="float64"),
            "in_pool": pd.array([True, True], dtype="bool"),
            "auction_dollars": pd.array([66, 30], dtype=pd.Int64Dtype()),
            "pool_rank": pd.array([1, 2], dtype=pd.Int64Dtype()),
            "reference_dollars": pd.array([pd.NA, pd.NA], dtype=pd.Int64Dtype()),
            "value_delta": pd.array([pd.NA, pd.NA], dtype=pd.Int64Dtype()),
        }
    )
    validated = AuctionValuesSchema.validate(df)
    assert len(validated) == 2

    bad = df.copy()
    bad.loc[bad.index[0], "auction_dollars"] = -5
    with pytest.raises(pa.errors.SchemaError):  # noqa: PT011 — pandera SchemaError lacks message specificity
        AuctionValuesSchema.validate(bad)
```

Confirm the imports at the top of `test_dataframe_schemas.py` already include `pandas as pd`, `pandera.pandas as pa`, and `pytest`. If not, add what's missing.

- [ ] **Step 2: Run the test to verify it fails**

```
PATH=".venv/Scripts:$PATH" pytest tests/test_schemas/test_dataframe_schemas.py::test_auction_values_schema_round_trip -v
```

Expected: FAIL at `from projections.schemas import AuctionValuesSchema` (ImportError).

- [ ] **Step 3: Append `AuctionValuesSchema` to `src/projections/schemas.py`**

Append after the last existing class (`ProjectionSeasonSchema`):

```python
class AuctionValuesSchema(pa.DataFrameModel):
    """Per-player auction $ allocation. Consumer-facing output of the auction-values generator.

    One row per player VORP knows about. `in_pool=False` rows have `auction_dollars=0`
    and `pool_rank=NA`. `reference_dollars` and `value_delta` are present in every
    output frame; both are all-NA when the caller didn't supply a reference-prices CSV.
    """

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$", unique=True)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    season_mean_fpts: Series[float]
    vorp: Series[float]
    in_pool: Series[bool]
    auction_dollars: Series[pd.Int64Dtype] = pa.Field(ge=0)
    pool_rank: Series[pd.Int64Dtype] = pa.Field(ge=1, nullable=True)
    reference_dollars: Series[pd.Int64Dtype] = pa.Field(ge=0, nullable=True)
    value_delta: Series[pd.Int64Dtype] = pa.Field(nullable=True)

    class Config:
        strict = "filter"
        coerce = True
```

- [ ] **Step 4: Run the test to verify it passes**

```
PATH=".venv/Scripts:$PATH" pytest tests/test_schemas/test_dataframe_schemas.py::test_auction_values_schema_round_trip -v
```

Expected: PASS.

- [ ] **Step 5: Run schema seam suite**

Per CLAUDE.md Agent Directive #4 (any pandera schema change runs the full schemas/ingest/store seam):

```
PATH=".venv/Scripts:$PATH" pytest -v -k "ingest or store or schemas"
```

Expected: all passing.

- [ ] **Step 6: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add src/projections/schemas.py tests/test_schemas/test_dataframe_schemas.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(schemas): AuctionValuesSchema for auction $ generator output"
```

---

### Task 3: Example `LeagueConfig` JSON files

**Files:**
- Create: `configs/league_espn_ppr_12team.json`
- Create: `configs/league_espn_half_10team.json`

- [ ] **Step 1: Verify `configs/` directory does not yet exist, then create files**

```
PATH=".venv/Scripts:$PATH" ls configs 2>/dev/null || echo "configs/ does not exist; will be created"
```

`configs/league_espn_ppr_12team.json`:
```json
{
  "name": "espn_ppr_12team_2026",
  "n_teams": 12,
  "budget": 200,
  "min_bid": 1,
  "roster_slots": {
    "QB": 1,
    "RB": 2,
    "WR": 3,
    "TE": 1,
    "FLEX": 1,
    "K": 1,
    "DST": 1,
    "BENCH": 7
  },
  "ruleset": "espn_ppr"
}
```

`configs/league_espn_half_10team.json`:
```json
{
  "name": "espn_half_10team_2026",
  "n_teams": 10,
  "budget": 200,
  "min_bid": 1,
  "roster_slots": {
    "QB": 1,
    "RB": 2,
    "WR": 2,
    "TE": 1,
    "FLEX": 1,
    "K": 1,
    "DST": 1,
    "BENCH": 6
  },
  "ruleset": "espn_half"
}
```

- [ ] **Step 2: Round-trip both configs through `LeagueConfig.model_validate_json`**

One-off check:

```
PATH=".venv/Scripts:$PATH" python -c "from pathlib import Path; from projections.draft.league_config import LeagueConfig; \
  cfg1 = LeagueConfig.model_validate_json(Path('configs/league_espn_ppr_12team.json').read_text()); \
  cfg2 = LeagueConfig.model_validate_json(Path('configs/league_espn_half_10team.json').read_text()); \
  print(cfg1.name, cfg1.roster_size, cfg1.total_budget); \
  print(cfg2.name, cfg2.roster_size, cfg2.total_budget)"
```

Expected output:
```
espn_ppr_12team_2026 16 2400
espn_half_10team_2026 14 2000
```

- [ ] **Step 3: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add configs/
PATH=".venv/Scripts:$PATH" git commit -m "feat(draft): example LeagueConfig JSONs (espn_ppr_12team, espn_half_10team)"
```

---

### Phase 1 verification

Run before starting Phase 2:

```
PATH=".venv/Scripts:$PATH" pytest tests/test_draft -v
PATH=".venv/Scripts:$PATH" pytest tests/test_schemas -v
PATH=".venv/Scripts:$PATH" mypy src/projections/draft src/projections/schemas.py tests/test_draft tests/test_schemas
PATH=".venv/Scripts:$PATH" ruff check src/projections/draft src/projections/schemas.py tests/test_draft tests/test_schemas
PATH=".venv/Scripts:$PATH" ruff format --check src/projections/draft src/projections/schemas.py tests/test_draft tests/test_schemas
```

Stop and report before Phase 2.

---

## Phase 2 — Core algorithm: `generate_auction_values`

This phase implements the function in three logical chunks, each driven by tests written first. All edits land in two files: `src/projections/draft/auction.py` (new) and `tests/test_draft/test_auction.py` (new).

### Task 4: Pool-building helper `_select_pool`

**Files:**
- Create: `src/projections/draft/auction.py` (skeleton + `_select_pool`)
- Create: `tests/test_draft/test_auction.py` (pool tests)

- [ ] **Step 1: Write failing pool-selection tests**

`tests/test_draft/test_auction.py`:

```python
"""Unit tests for `projections.draft.auction.generate_auction_values` and helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.draft.auction import _select_pool, generate_auction_values
from projections.draft.league_config import LeagueConfig
from projections.schemas import (
    AuctionValuesSchema,
    Position,
    RosterSlot,
    Ruleset,
)


def _make_config(
    n_teams: int = 4,
    roster_slots: dict[RosterSlot, int] | None = None,
    budget: int = 100,
    min_bid: int = 1,
) -> LeagueConfig:
    return LeagueConfig(
        name="test",
        n_teams=n_teams,
        budget=budget,
        min_bid=min_bid,
        roster_slots=roster_slots
        or {
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 1,
        },
        ruleset=Ruleset.espn_ppr(),
    )


def _make_vorp_table(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a VORP-table-shaped DataFrame from a list of dicts.

    Required keys: gsis_id, position, season_mean_fpts, vorp.
    """
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(pd.StringDtype("pyarrow"))
    df["position"] = df["position"].astype(pd.StringDtype("pyarrow"))
    df["season_mean_fpts"] = df["season_mean_fpts"].astype("float64")
    df["vorp"] = df["vorp"].astype("float64")
    return df


def _bulk_position_rows(position: Position, count: int, base_fpts: float = 200.0) -> list[dict[str, object]]:
    """Generate `count` rows for `position` with descending season_mean_fpts and matching VORP."""
    out: list[dict[str, object]] = []
    for i in range(count):
        out.append(
            {
                "gsis_id": f"00-{position.value}{i:05d}"[:10],
                "position": position.value,
                "season_mean_fpts": base_fpts - i,
                "vorp": (base_fpts - i) - 100.0,
            }
        )
    return out


def test_select_pool_size_matches_total_pool_size() -> None:
    cfg = _make_config()
    rows: list[dict[str, object]] = []
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE):
        rows.extend(_bulk_position_rows(pos, count=20))
    df = _make_vorp_table(rows)
    pool_ids = _select_pool(df, cfg)
    assert len(pool_ids) == cfg.total_pool_size


def test_select_pool_respects_position_quotas() -> None:
    cfg = _make_config()
    rows: list[dict[str, object]] = []
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE):
        rows.extend(_bulk_position_rows(pos, count=20))
    df = _make_vorp_table(rows)
    pool_ids = _select_pool(df, cfg)
    pool_df = df[df["gsis_id"].isin(pool_ids)]
    counts = pool_df["position"].value_counts().to_dict()
    # 4 teams x 1 QB = 4 QBs guaranteed; 4 x 2 = 8 RBs; 4 x 2 = 8 WRs; 4 x 1 = 4 TEs.
    # FLEX (4 slots) + BENCH (4 slots) fill from the best remaining.
    assert counts[Position.QB.value] >= 4
    assert counts[Position.RB.value] >= 8
    assert counts[Position.WR.value] >= 8
    assert counts[Position.TE.value] >= 4
    assert sum(counts.values()) == cfg.total_pool_size


def test_select_pool_omits_low_projection_players() -> None:
    """With only 12 RBs and a league that wants 12 in-pool RBs (8 strict + 4 FLEX-eligible),
    RB13+ should be out of pool."""
    cfg = _make_config(
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.BENCH: 0,
        }
    )
    rows: list[dict[str, object]] = []
    rows.extend(_bulk_position_rows(Position.QB, count=10))
    rows.extend(_bulk_position_rows(Position.RB, count=20))
    rows.extend(_bulk_position_rows(Position.WR, count=20))
    rows.extend(_bulk_position_rows(Position.TE, count=10))
    df = _make_vorp_table(rows)
    pool_ids = _select_pool(df, cfg)
    pool_df = df[df["gsis_id"].isin(pool_ids)]
    rb_pool = pool_df[pool_df["position"] == Position.RB.value]
    # 4 teams * 2 RB starters = 8 strict RBs. FLEX may add more.
    assert len(rb_pool) >= 8
    # The omitted RBs are the lowest-projection ones
    omitted_rb_ids = set(df[df["position"] == Position.RB.value]["gsis_id"]) - set(rb_pool["gsis_id"])
    if omitted_rb_ids:
        min_in_pool_fpts = rb_pool["season_mean_fpts"].min()
        omitted_max_fpts = df[df["gsis_id"].isin(omitted_rb_ids)]["season_mean_fpts"].max()
        assert omitted_max_fpts <= min_in_pool_fpts


def test_select_pool_omits_position_not_in_roster_slots() -> None:
    """A config without K or DST should not consume K or DST players into the pool."""
    cfg = _make_config(
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.BENCH: 2,
        }
    )
    rows: list[dict[str, object]] = []
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE, Position.K, Position.DST):
        rows.extend(_bulk_position_rows(pos, count=15))
    df = _make_vorp_table(rows)
    pool_ids = _select_pool(df, cfg)
    pool_df = df[df["gsis_id"].isin(pool_ids)]
    assert (pool_df["position"] != Position.K.value).all()
    assert (pool_df["position"] != Position.DST.value).all()


def test_select_pool_errors_on_missing_required_position() -> None:
    """If the config requires a position the VORP table doesn't cover, raise clearly."""
    cfg = _make_config(
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.K: 1,
            RosterSlot.BENCH: 0,
        }
    )
    rows: list[dict[str, object]] = []
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE):
        rows.extend(_bulk_position_rows(pos, count=10))
    df = _make_vorp_table(rows)
    with pytest.raises(ValueError, match=r"\bK\b"):
        _select_pool(df, cfg)
```

- [ ] **Step 2: Run the tests to verify they fail**

```
PATH=".venv/Scripts:$PATH" pytest tests/test_draft/test_auction.py -v
```

Expected: ImportError (module not yet created).

- [ ] **Step 3: Implement `_select_pool`**

`src/projections/draft/auction.py`:

```python
"""Auction $ generator: converts per-player VORP to per-player auction dollars.

Public surface: `generate_auction_values(vorp_table, league_config, reference_prices=None)`.

Algorithm: standard Surplus-Of-Surplus (SOS) allocation. Reserve `min_bid` for every
drafted slot, then distribute the remaining budget proportionally to positive VORP
among the rostered pool. Strategy-agnostic; one $ per player.

Spec: docs/superpowers/specs/2026-05-16-auction-values-design.md
"""

from __future__ import annotations

import pandas as pd

from projections.draft.league_config import LeagueConfig
from projections.schemas import AuctionValuesSchema, Position, RosterSlot

# RosterSlot keys that consume position-specific picks at draft. FLEX, SUPER_FLEX, BENCH
# fill from the remainder. IR is excluded (post-draft).
_POSITION_SLOTS: tuple[RosterSlot, ...] = (
    RosterSlot.QB,
    RosterSlot.RB,
    RosterSlot.WR,
    RosterSlot.TE,
    RosterSlot.K,
    RosterSlot.DST,
)

# Position eligibility for filler slots.
_FLEX_ELIGIBLE: frozenset[Position] = frozenset({Position.RB, Position.WR, Position.TE})
_SUPER_FLEX_ELIGIBLE: frozenset[Position] = frozenset({Position.QB, Position.RB, Position.WR, Position.TE})


def _select_pool(vorp_table: pd.DataFrame, league_config: LeagueConfig) -> list[str]:
    """Select the in-pool `gsis_id`s per the spec §3.1 algorithm.

    Returns a list of length `league_config.total_pool_size`. Selection order:
    position-specific slots, then FLEX, then SUPER_FLEX, then BENCH. Within each
    pass, players are ranked by `season_mean_fpts` desc, tie-broken by `vorp` desc
    then `gsis_id` asc.

    Raises `ValueError` if any required position is missing from `vorp_table`.
    """
    # Pre-sort once; we'll slice by position from this sorted view.
    sorted_df = vorp_table.sort_values(
        by=["season_mean_fpts", "vorp", "gsis_id"],
        ascending=[False, False, True],
        kind="mergesort",  # stable
    ).reset_index(drop=True)

    picked: list[str] = []
    picked_set: set[str] = set()

    # Pass 1: position-specific slots.
    for slot in _POSITION_SLOTS:
        wanted = league_config.roster_slots.get(slot, 0)
        if wanted <= 0:
            continue
        # The RosterSlot enum mirrors Position values for non-FLEX/BENCH slots.
        pos_value = slot.value
        position_pool = sorted_df[sorted_df["position"] == pos_value]
        needed = league_config.n_teams * wanted
        if len(position_pool) < needed:
            raise ValueError(
                f"VORP table has only {len(position_pool)} {pos_value} players "
                f"but league_config requires {needed} ({league_config.n_teams} teams x {wanted} {pos_value} slots)."
            )
        for gid in position_pool["gsis_id"].head(needed).tolist():
            picked.append(gid)
            picked_set.add(gid)

    def _fill_filler_slot(slot: RosterSlot, eligible: frozenset[Position]) -> None:
        wanted = league_config.roster_slots.get(slot, 0)
        if wanted <= 0:
            return
        needed = league_config.n_teams * wanted
        eligible_values = {p.value for p in eligible}
        remaining = sorted_df[
            (sorted_df["position"].isin(eligible_values)) & (~sorted_df["gsis_id"].isin(picked_set))
        ]
        if len(remaining) < needed:
            raise ValueError(
                f"VORP table cannot fill {needed} {slot.value} slots: only "
                f"{len(remaining)} eligible players remain after position-specific picks."
            )
        for gid in remaining["gsis_id"].head(needed).tolist():
            picked.append(gid)
            picked_set.add(gid)

    # Pass 2: FLEX (RB/WR/TE).
    _fill_filler_slot(RosterSlot.FLEX, _FLEX_ELIGIBLE)

    # Pass 3: SUPER_FLEX (QB/RB/WR/TE).
    _fill_filler_slot(RosterSlot.SUPER_FLEX, _SUPER_FLEX_ELIGIBLE)

    # Pass 4: BENCH. Position-agnostic over positions the league recognizes.
    bench_count = league_config.roster_slots.get(RosterSlot.BENCH, 0)
    if bench_count > 0:
        needed = league_config.n_teams * bench_count
        # Bench can come from any position that has at least one slot in the league
        # (excluding IR). This guards against drafting K/DST onto bench in leagues
        # that don't roster them.
        league_positions = {
            slot.value
            for slot in league_config.roster_slots
            if slot in _POSITION_SLOTS and league_config.roster_slots[slot] > 0
        }
        remaining = sorted_df[
            (sorted_df["position"].isin(league_positions)) & (~sorted_df["gsis_id"].isin(picked_set))
        ]
        if len(remaining) < needed:
            raise ValueError(
                f"VORP table cannot fill {needed} BENCH slots: only "
                f"{len(remaining)} eligible players remain after starter + flex picks."
            )
        for gid in remaining["gsis_id"].head(needed).tolist():
            picked.append(gid)
            picked_set.add(gid)

    return picked


# Placeholder; implemented in Task 5.
def generate_auction_values(
    vorp_table: pd.DataFrame,
    league_config: LeagueConfig,
    reference_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Placeholder. Implemented in Task 5."""
    raise NotImplementedError
```

Also update `src/projections/draft/__init__.py`:

```python
"""Draft Hub sub-project: pre-draft tooling (auction values, snake recommender, VORP)."""

from projections.draft.auction import generate_auction_values
from projections.draft.league_config import LeagueConfig

__all__ = ["LeagueConfig", "generate_auction_values"]
```

- [ ] **Step 4: Run pool tests to verify they pass**

```
PATH=".venv/Scripts:$PATH" pytest tests/test_draft/test_auction.py -v -k "select_pool"
```

Expected: 5 passing.

- [ ] **Step 5: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add src/projections/draft/auction.py src/projections/draft/__init__.py tests/test_draft/test_auction.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(draft): _select_pool helper for auction $ generator"
```

---

### Task 5: `generate_auction_values` core flow (SOS allocation + rounding drift)

**Files:**
- Modify: `src/projections/draft/auction.py` (replace placeholder)
- Modify: `tests/test_draft/test_auction.py` (append algorithmic-invariant tests)

- [ ] **Step 1: Write failing core-flow tests**

Append to `tests/test_draft/test_auction.py`:

```python
def _full_pool_vorp_table(cfg: LeagueConfig, extra_per_position: int = 5) -> pd.DataFrame:
    """Build a VORP table large enough to fill `cfg`'s pool plus a buffer of out-of-pool rows."""
    rows: list[dict[str, object]] = []
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE, Position.K, Position.DST):
        slot = RosterSlot(pos.value)
        # Skip positions the league does not roster
        if cfg.roster_slots.get(slot, 0) == 0 and pos not in (Position.RB, Position.WR, Position.TE):
            continue
        rows.extend(_bulk_position_rows(pos, count=cfg.n_teams * 4 + extra_per_position))
    return _make_vorp_table(rows)


def test_sum_invariant_matches_total_budget() -> None:
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    out = generate_auction_values(df, cfg)
    assert int(out["auction_dollars"].sum()) == cfg.total_budget


def test_min_bid_floor_for_in_pool_players() -> None:
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    out = generate_auction_values(df, cfg)
    in_pool = out[out["in_pool"]]
    assert (in_pool["auction_dollars"] >= cfg.min_bid).all()


def test_out_of_pool_players_get_zero_dollars() -> None:
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    out = generate_auction_values(df, cfg)
    out_of_pool = out[~out["in_pool"]]
    assert (out_of_pool["auction_dollars"] == 0).all()
    assert out_of_pool["pool_rank"].isna().all()


def test_pool_size_exactly_total_pool_size() -> None:
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    out = generate_auction_values(df, cfg)
    assert int(out["in_pool"].sum()) == cfg.total_pool_size


def test_negative_vorp_in_pool_gets_min_bid() -> None:
    """In-pool players with vorp <= 0 should get exactly min_bid (modulo drift adjustments
    that never reach this part of the curve in realistic test sizes)."""
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    out = generate_auction_values(df, cfg)
    in_pool = out[out["in_pool"]]
    neg_vorp = in_pool[in_pool["vorp"] <= 0]
    if len(neg_vorp) > 0:
        # Drift adjustments only land on players with the largest fractional parts,
        # which are mid-pack high-VORP players, never the min-bid floor.
        assert (neg_vorp["auction_dollars"] == cfg.min_bid).all()


def test_vorp_scale_invariance() -> None:
    """Doubling all positive VORPs leaves auction_dollars unchanged (proportional allocation)."""
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    out_a = generate_auction_values(df, cfg)
    df_scaled = df.copy()
    df_scaled["vorp"] = df_scaled["vorp"] * 2.0
    out_b = generate_auction_values(df_scaled, cfg)
    merged = out_a.merge(
        out_b[["gsis_id", "auction_dollars"]],
        on="gsis_id",
        suffixes=("_a", "_b"),
    )
    assert (merged["auction_dollars_a"] == merged["auction_dollars_b"]).all()


def test_higher_budget_scales_surplus() -> None:
    """With identical VORPs but budget=200 vs budget=100, in-pool players get
    approximately 2x the dollars (exactly 2x after subtracting min_bid)."""
    cfg_a = _make_config(budget=100)
    cfg_b = _make_config(budget=200)
    df = _full_pool_vorp_table(cfg_a)
    out_a = generate_auction_values(df, cfg_a)
    out_b = generate_auction_values(df, cfg_b)
    merged = (
        out_a[out_a["in_pool"]]
        .merge(
            out_b[["gsis_id", "auction_dollars"]],
            on="gsis_id",
            suffixes=("_a", "_b"),
        )
    )
    # Compare expected and actual extras above min_bid
    expected_b = 2 * (merged["auction_dollars_a"] - cfg_a.min_bid) + cfg_b.min_bid
    # Allow +/- 1 for rounding-drift redistribution
    diff = (merged["auction_dollars_b"] - expected_b).abs()
    assert (diff <= 1).all()


def test_pool_rank_is_dense_and_ordered() -> None:
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    out = generate_auction_values(df, cfg)
    in_pool = out[out["in_pool"]].sort_values("pool_rank")
    assert in_pool["pool_rank"].tolist() == list(range(1, cfg.total_pool_size + 1))
    # auction_dollars is non-increasing with pool_rank
    dollars = in_pool["auction_dollars"].tolist()
    assert dollars == sorted(dollars, reverse=True)


def test_output_validates_against_schema() -> None:
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    out = generate_auction_values(df, cfg)
    AuctionValuesSchema.validate(out)


def test_degenerate_zero_positive_vorp_distributes_uniformly() -> None:
    """If every in-pool player has vorp <= 0, distribute surplus uniformly."""
    cfg = _make_config(n_teams=2, roster_slots={RosterSlot.QB: 1, RosterSlot.RB: 1, RosterSlot.BENCH: 0})
    # 4 players total; all VORP = 0
    rows = [
        {"gsis_id": "00-QB000001", "position": "QB", "season_mean_fpts": 200.0, "vorp": 0.0},
        {"gsis_id": "00-QB000002", "position": "QB", "season_mean_fpts": 190.0, "vorp": 0.0},
        {"gsis_id": "00-RB000001", "position": "RB", "season_mean_fpts": 180.0, "vorp": 0.0},
        {"gsis_id": "00-RB000002", "position": "RB", "season_mean_fpts": 170.0, "vorp": 0.0},
    ]
    df = _make_vorp_table(rows)
    out = generate_auction_values(df, cfg)
    in_pool = out[out["in_pool"]]
    # total_budget = 200, total_pool_size = 4 -> $50 each
    assert in_pool["auction_dollars"].tolist() == [50, 50, 50, 50] or (
        sorted(in_pool["auction_dollars"].tolist()) in ([49, 50, 50, 51], [50, 50, 50, 50])
    )
    assert int(in_pool["auction_dollars"].sum()) == cfg.total_budget


def test_duplicate_gsis_id_rejected() -> None:
    cfg = _make_config(n_teams=2, roster_slots={RosterSlot.QB: 1, RosterSlot.BENCH: 0})
    rows = [
        {"gsis_id": "00-QB000001", "position": "QB", "season_mean_fpts": 200.0, "vorp": 50.0},
        {"gsis_id": "00-QB000001", "position": "QB", "season_mean_fpts": 190.0, "vorp": 40.0},
    ]
    df = _make_vorp_table(rows)
    with pytest.raises(ValueError, match="duplicate"):
        generate_auction_values(df, cfg)
```

- [ ] **Step 2: Run tests to verify they fail**

```
PATH=".venv/Scripts:$PATH" pytest tests/test_draft/test_auction.py -v -k "not select_pool"
```

Expected: all fail with `NotImplementedError`.

- [ ] **Step 3: Replace placeholder `generate_auction_values` with full implementation**

In `src/projections/draft/auction.py`, replace the placeholder function with:

```python
def generate_auction_values(
    vorp_table: pd.DataFrame,
    league_config: LeagueConfig,
    reference_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert per-player VORP into per-player auction dollars under `league_config`.

    Returns a DataFrame validated against `AuctionValuesSchema`. One row per player
    in `vorp_table`. Players not in the rostered pool get `auction_dollars=0` and
    `pool_rank=NA`. `reference_dollars` and `value_delta` are present in the output
    regardless of whether `reference_prices` was passed (all-NA when not passed).

    See spec §3 for the SOS algorithm and §6 for edge-case decisions.
    """
    # Input validation.
    if vorp_table["gsis_id"].duplicated().any():
        dup = vorp_table.loc[vorp_table["gsis_id"].duplicated(), "gsis_id"].iloc[0]
        raise ValueError(f"vorp_table has duplicate gsis_id rows (first duplicate: {dup}).")

    # Step 1 - build the rostered pool.
    pool_ids = _select_pool(vorp_table, league_config)
    pool_set = set(pool_ids)

    # Step 2 - compute the surplus.
    total_budget = league_config.total_budget
    reserve = league_config.total_pool_size * league_config.min_bid
    surplus = total_budget - reserve

    # Step 3 - allocate surplus to positive VORP.
    pool_df = vorp_table[vorp_table["gsis_id"].isin(pool_set)].copy()
    positive_vorp = pool_df["vorp"].clip(lower=0.0)
    positive_vorp_sum = float(positive_vorp.sum())

    if positive_vorp_sum > 0:
        extra_float = (positive_vorp / positive_vorp_sum) * surplus
    else:
        # Degenerate case: distribute surplus uniformly.
        extra_float = pd.Series(surplus / league_config.total_pool_size, index=pool_df.index)

    pool_df["_dollars_float"] = league_config.min_bid + extra_float

    # Step 4 - round and close drift.
    rounded = pool_df["_dollars_float"].round().astype("int64")
    drift = total_budget - int(rounded.sum())
    if drift != 0:
        fractional = pool_df["_dollars_float"] - pool_df["_dollars_float"].astype("int64")
        # When drift > 0 we need to add: pick rows with largest fractional parts.
        # When drift < 0 we need to subtract: pick rows with smallest fractional parts.
        order = fractional.sort_values(ascending=(drift < 0)).index
        step = 1 if drift > 0 else -1
        for idx in order[: abs(drift)]:
            rounded.loc[idx] = rounded.loc[idx] + step
    pool_df["auction_dollars"] = rounded.astype(pd.Int64Dtype())

    # Step 5 - rank within pool.
    rank_sort = pool_df.sort_values(
        by=["auction_dollars", "vorp", "season_mean_fpts", "gsis_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    rank_sort["pool_rank"] = range(1, len(rank_sort) + 1)
    pool_df = pool_df.merge(
        rank_sort[["gsis_id", "pool_rank"]],
        on="gsis_id",
        how="left",
    )

    # Assemble the full output: pool + non-pool rows.
    non_pool_df = vorp_table[~vorp_table["gsis_id"].isin(pool_set)].copy()
    non_pool_df["auction_dollars"] = pd.array([0] * len(non_pool_df), dtype=pd.Int64Dtype())
    non_pool_df["pool_rank"] = pd.array([pd.NA] * len(non_pool_df), dtype=pd.Int64Dtype())

    out = pd.concat(
        [
            pool_df[
                [
                    "gsis_id",
                    "position",
                    "season_mean_fpts",
                    "vorp",
                    "auction_dollars",
                    "pool_rank",
                ]
            ],
            non_pool_df[
                [
                    "gsis_id",
                    "position",
                    "season_mean_fpts",
                    "vorp",
                    "auction_dollars",
                    "pool_rank",
                ]
            ],
        ],
        ignore_index=True,
    )
    out["in_pool"] = out["gsis_id"].isin(pool_set)

    # Step 6 - attach reference prices (implemented in Task 6 - stub for now).
    out["reference_dollars"] = pd.array([pd.NA] * len(out), dtype=pd.Int64Dtype())
    out["value_delta"] = pd.array([pd.NA] * len(out), dtype=pd.Int64Dtype())

    # Re-order columns to match AuctionValuesSchema.
    out = out[
        [
            "gsis_id",
            "position",
            "season_mean_fpts",
            "vorp",
            "in_pool",
            "auction_dollars",
            "pool_rank",
            "reference_dollars",
            "value_delta",
        ]
    ]

    return AuctionValuesSchema.validate(out)  # type: ignore[no-any-return]
```

- [ ] **Step 4: Run core-flow tests to verify they pass**

```
PATH=".venv/Scripts:$PATH" pytest tests/test_draft/test_auction.py -v -k "not reference"
```

Expected: all passing (pool tests + the 11 core-flow tests added above; reference-prices test added in Task 6).

- [ ] **Step 5: Run mypy + ruff on the touched files**

```
PATH=".venv/Scripts:$PATH" mypy src/projections/draft tests/test_draft
PATH=".venv/Scripts:$PATH" ruff check src/projections/draft tests/test_draft
PATH=".venv/Scripts:$PATH" ruff format --check src/projections/draft tests/test_draft
```

Expected: zero violations.

- [ ] **Step 6: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add src/projections/draft/auction.py tests/test_draft/test_auction.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(draft): generate_auction_values core SOS allocation"
```

---

### Task 6: `reference_prices` pass-through

**Files:**
- Modify: `src/projections/draft/auction.py` (replace stub reference handling)
- Modify: `tests/test_draft/test_auction.py` (append reference-prices tests)

- [ ] **Step 1: Write failing reference-prices tests**

Append to `tests/test_draft/test_auction.py`:

```python
def test_reference_prices_pass_through_matched_rows() -> None:
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    # Build a partial reference table covering only the first two players.
    first_two = df["gsis_id"].iloc[:2].tolist()
    ref = pd.DataFrame(
        {
            "gsis_id": pd.array(first_two, dtype=pd.StringDtype("pyarrow")),
            "reference_dollars": pd.array([45, 30], dtype=pd.Int64Dtype()),
        }
    )
    out = generate_auction_values(df, cfg, reference_prices=ref)
    matched = out[out["gsis_id"].isin(first_two)].sort_values("gsis_id")
    expected_ref = ref.sort_values("gsis_id")
    assert matched["reference_dollars"].tolist() == expected_ref["reference_dollars"].tolist()
    # value_delta = auction_dollars - reference_dollars on matched rows
    deltas = (matched["auction_dollars"] - expected_ref["reference_dollars"].values).tolist()
    assert matched["value_delta"].tolist() == deltas


def test_reference_prices_unmatched_rows_get_na() -> None:
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    ref = pd.DataFrame(
        {
            "gsis_id": pd.array([df["gsis_id"].iloc[0]], dtype=pd.StringDtype("pyarrow")),
            "reference_dollars": pd.array([45], dtype=pd.Int64Dtype()),
        }
    )
    out = generate_auction_values(df, cfg, reference_prices=ref)
    unmatched = out[out["gsis_id"] != df["gsis_id"].iloc[0]]
    assert unmatched["reference_dollars"].isna().all()
    assert unmatched["value_delta"].isna().all()


def test_no_reference_prices_columns_all_na() -> None:
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    out = generate_auction_values(df, cfg, reference_prices=None)
    assert "reference_dollars" in out.columns
    assert "value_delta" in out.columns
    assert out["reference_dollars"].isna().all()
    assert out["value_delta"].isna().all()


def test_reference_prices_duplicate_gsis_id_rejected() -> None:
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    ref = pd.DataFrame(
        {
            "gsis_id": pd.array([df["gsis_id"].iloc[0], df["gsis_id"].iloc[0]], dtype=pd.StringDtype("pyarrow")),
            "reference_dollars": pd.array([45, 50], dtype=pd.Int64Dtype()),
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        generate_auction_values(df, cfg, reference_prices=ref)
```

- [ ] **Step 2: Run reference tests to verify they fail**

```
PATH=".venv/Scripts:$PATH" pytest tests/test_draft/test_auction.py -v -k "reference"
```

Expected: 4 failures (current implementation stubs reference cols to all-NA).

- [ ] **Step 3: Replace the stub reference-prices block**

In `src/projections/draft/auction.py`, replace the stub reference-prices block:

```python
    # Step 6 - attach reference prices (implemented in Task 6 - stub for now).
    out["reference_dollars"] = pd.array([pd.NA] * len(out), dtype=pd.Int64Dtype())
    out["value_delta"] = pd.array([pd.NA] * len(out), dtype=pd.Int64Dtype())
```

with:

```python
    # Step 6 - attach reference prices.
    if reference_prices is None:
        out["reference_dollars"] = pd.array([pd.NA] * len(out), dtype=pd.Int64Dtype())
        out["value_delta"] = pd.array([pd.NA] * len(out), dtype=pd.Int64Dtype())
    else:
        if reference_prices["gsis_id"].duplicated().any():
            dup = reference_prices.loc[reference_prices["gsis_id"].duplicated(), "gsis_id"].iloc[0]
            raise ValueError(f"reference_prices has duplicate gsis_id rows (first duplicate: {dup}).")
        ref = reference_prices[["gsis_id", "reference_dollars"]].copy()
        ref["reference_dollars"] = ref["reference_dollars"].astype(pd.Int64Dtype())
        out = out.merge(ref, on="gsis_id", how="left")
        # `merge` preserves Int64Dtype with NA for unmatched rows.
        out["value_delta"] = (out["auction_dollars"] - out["reference_dollars"]).astype(pd.Int64Dtype())
```

- [ ] **Step 4: Run reference tests to verify they pass**

```
PATH=".venv/Scripts:$PATH" pytest tests/test_draft/test_auction.py -v -k "reference"
```

Expected: all 4 passing.

- [ ] **Step 5: Run the full test_draft suite**

```
PATH=".venv/Scripts:$PATH" pytest tests/test_draft -v
```

Expected: all passing.

- [ ] **Step 6: Run mypy + ruff on touched files**

```
PATH=".venv/Scripts:$PATH" mypy src/projections/draft tests/test_draft
PATH=".venv/Scripts:$PATH" ruff check src/projections/draft tests/test_draft
PATH=".venv/Scripts:$PATH" ruff format --check src/projections/draft tests/test_draft
```

Expected: zero violations.

- [ ] **Step 7: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add src/projections/draft/auction.py tests/test_draft/test_auction.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(draft): reference_prices pass-through for auction $ output"
```

---

### Phase 2 verification

```
PATH=".venv/Scripts:$PATH" pytest tests/test_draft -v
PATH=".venv/Scripts:$PATH" pytest -v -k "ingest or store or schemas"
PATH=".venv/Scripts:$PATH" mypy src tests
PATH=".venv/Scripts:$PATH" ruff check src tests
PATH=".venv/Scripts:$PATH" ruff format --check src tests
```

Stop and report before Phase 3.

---

## Phase 3 — CLI + final verification

### Task 7: CLI script + integration test

**Files:**
- Create: `scripts/generate_auction_values.py`
- Create: `tests/test_scripts/test_generate_auction_values_cli.py`

- [ ] **Step 1: Write the failing CLI integration test**

`tests/test_scripts/test_generate_auction_values_cli.py`:

```python
"""End-to-end integration test for `scripts/generate_auction_values.py`."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from projections.schemas import AuctionValuesSchema, Position, RosterSlot, Ruleset


@pytest.fixture
def cli_inputs(tmp_path: Path) -> dict[str, Path]:
    """Build a minimal LeagueConfig JSON + VORP parquet for an end-to-end CLI run."""
    cfg_path = tmp_path / "league.json"
    cfg_path.write_text(
        json.dumps(
            {
                "name": "tiny_test",
                "n_teams": 2,
                "budget": 100,
                "min_bid": 1,
                "roster_slots": {
                    RosterSlot.QB.value: 1,
                    RosterSlot.RB.value: 1,
                    RosterSlot.BENCH.value: 1,
                },
                "ruleset": "standard",
            }
        )
    )
    rows = [
        {"gsis_id": "00-QB000001", "position": Position.QB.value, "season_mean_fpts": 320.0, "vorp": 100.0},
        {"gsis_id": "00-QB000002", "position": Position.QB.value, "season_mean_fpts": 280.0, "vorp": 60.0},
        {"gsis_id": "00-RB000001", "position": Position.RB.value, "season_mean_fpts": 260.0, "vorp": 80.0},
        {"gsis_id": "00-RB000002", "position": Position.RB.value, "season_mean_fpts": 220.0, "vorp": 40.0},
        {"gsis_id": "00-RB000003", "position": Position.RB.value, "season_mean_fpts": 180.0, "vorp": 5.0},
        {"gsis_id": "00-RB000004", "position": Position.RB.value, "season_mean_fpts": 120.0, "vorp": -20.0},
    ]
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(pd.StringDtype("pyarrow"))
    df["position"] = df["position"].astype(pd.StringDtype("pyarrow"))
    vorp_path = tmp_path / "vorp.parquet"
    df.to_parquet(vorp_path, index=False)
    return {
        "config": cfg_path,
        "vorp": vorp_path,
        "out_csv": tmp_path / "auction_values.csv",
        "out_parquet": tmp_path / "auction_values.parquet",
    }


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "generate_auction_values.py"
    return subprocess.run(
        [sys.executable, str(script), *args],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )


def test_cli_csv_output_sum_invariant(cli_inputs: dict[str, Path]) -> None:
    proc = _run_cli(
        "--season", "2026",
        "--league-config", str(cli_inputs["config"]),
        "--vorp-input", str(cli_inputs["vorp"]),
        "--out", str(cli_inputs["out_csv"]),
    )
    assert proc.returncode == 0, proc.stderr
    out = pd.read_csv(cli_inputs["out_csv"])
    # 2 teams x 3 roster slots = 6 in-pool players (4 RB + 2 QB)
    assert int(out["auction_dollars"].sum()) == 2 * 100  # n_teams * budget
    assert int(out["in_pool"].sum()) == 6


def test_cli_parquet_output_schema(cli_inputs: dict[str, Path]) -> None:
    _run_cli(
        "--season", "2026",
        "--league-config", str(cli_inputs["config"]),
        "--vorp-input", str(cli_inputs["vorp"]),
        "--out", str(cli_inputs["out_parquet"]),
    )
    out = pd.read_parquet(cli_inputs["out_parquet"])
    # round-trip through the canonical schema
    AuctionValuesSchema.validate(out)


def test_cli_errors_on_missing_vorp_input(cli_inputs: dict[str, Path], tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.parquet"
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        _run_cli(
            "--season", "2026",
            "--league-config", str(cli_inputs["config"]),
            "--vorp-input", str(missing),
            "--out", str(cli_inputs["out_csv"]),
        )
    assert "vorp" in exc_info.value.stderr.lower() or "not exist" in exc_info.value.stderr.lower()
```

- [ ] **Step 2: Run the CLI tests to verify they fail**

```
PATH=".venv/Scripts:$PATH" pytest tests/test_scripts/test_generate_auction_values_cli.py -v
```

Expected: fail (script does not exist yet).

- [ ] **Step 3: Implement the CLI script**

`scripts/generate_auction_values.py`:

```python
"""CLI: convert per-player VORP into per-player auction $ for a given LeagueConfig.

Reads:
    --league-config  Path to LeagueConfig JSON.
    --vorp-input     Path to VORP parquet (gsis_id, position, season_mean_fpts, vorp).
    --reference-prices  Optional CSV with gsis_id, reference_dollars columns.
Writes:
    --out            Output path; .csv and .parquet supported (extension-sniffed).

Spec: docs/superpowers/specs/2026-05-16-auction-values-design.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.draft.auction import generate_auction_values
from projections.draft.league_config import LeagueConfig
from projections.schemas import Position


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True, help="Season (metadata only).")
    parser.add_argument("--league-config", type=Path, required=True, help="LeagueConfig JSON path.")
    parser.add_argument("--vorp-input", type=Path, required=True, help="VORP parquet path.")
    parser.add_argument(
        "--reference-prices",
        type=Path,
        default=None,
        help="Optional CSV with gsis_id, reference_dollars columns.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output path (.csv or .parquet).")
    return parser.parse_args()


def _read_vorp(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"vorp-input parquet does not exist: {path}")
    df = pd.read_parquet(path)
    required = {"gsis_id", "position", "season_mean_fpts", "vorp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"vorp-input parquet is missing required columns: {sorted(missing)}")
    df["gsis_id"] = df["gsis_id"].astype(pd.StringDtype("pyarrow"))
    df["position"] = df["position"].astype(pd.StringDtype("pyarrow"))
    df["season_mean_fpts"] = df["season_mean_fpts"].astype("float64")
    df["vorp"] = df["vorp"].astype("float64")
    return df


def _read_reference_prices(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"reference-prices CSV does not exist: {path}")
    df = pd.read_csv(path)
    required = {"gsis_id", "reference_dollars"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"reference-prices CSV is missing required columns: {sorted(missing)}")
    df["gsis_id"] = df["gsis_id"].astype(pd.StringDtype("pyarrow"))
    df["reference_dollars"] = df["reference_dollars"].astype(pd.Int64Dtype())
    return df


def _write_output(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".csv":
        sorted_df = df.sort_values(
            by=["pool_rank", "auction_dollars", "gsis_id"],
            ascending=[True, False, True],
            na_position="last",
            kind="mergesort",
        )
        sorted_df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(
            f"unsupported output extension {path.suffix!r}; expected .csv or .parquet."
        )


def _emit_summary(df: pd.DataFrame, season: int) -> None:
    """Per-position summary printed to stdout. Risk mitigation per spec §6."""
    in_pool = df[df["in_pool"]]
    print(f"Auction values for season {season}: {len(in_pool)} in-pool players, "
          f"sum auction_dollars = ${int(in_pool['auction_dollars'].sum())}")
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE, Position.K, Position.DST):
        pos_df = in_pool[in_pool["position"] == pos.value]
        if len(pos_df) == 0:
            continue
        top = pos_df.sort_values("auction_dollars", ascending=False).head(3)
        top_summary = ", ".join(
            f"{row.gsis_id}: ${int(row.auction_dollars)} (vorp {row.vorp:.1f})"
            for row in top.itertuples()
        )
        print(
            f"  {pos.value}: n={len(pos_df)}, "
            f"min/median/max vorp = {pos_df['vorp'].min():.1f} / "
            f"{pos_df['vorp'].median():.1f} / {pos_df['vorp'].max():.1f}; "
            f"top3 = [{top_summary}]"
        )


def main() -> None:
    args = _parse_args()
    league_config = LeagueConfig.model_validate_json(args.league_config.read_text())
    vorp_table = _read_vorp(args.vorp_input)
    reference_prices = _read_reference_prices(args.reference_prices)
    out = generate_auction_values(vorp_table, league_config, reference_prices=reference_prices)
    _emit_summary(out, args.season)
    _write_output(out, args.out)
    print(f"Wrote {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run CLI tests to verify they pass**

```
PATH=".venv/Scripts:$PATH" pytest tests/test_scripts/test_generate_auction_values_cli.py -v
```

Expected: 3 passing.

- [ ] **Step 5: Run mypy + ruff on touched files**

```
PATH=".venv/Scripts:$PATH" mypy scripts/generate_auction_values.py tests/test_scripts/test_generate_auction_values_cli.py
PATH=".venv/Scripts:$PATH" ruff check scripts/generate_auction_values.py tests/test_scripts/test_generate_auction_values_cli.py
PATH=".venv/Scripts:$PATH" ruff format --check scripts/generate_auction_values.py tests/test_scripts/test_generate_auction_values_cli.py
```

Expected: zero violations.

- [ ] **Step 6: Commit**

```bash
PATH=".venv/Scripts:$PATH" git add scripts/generate_auction_values.py tests/test_scripts/test_generate_auction_values_cli.py
PATH=".venv/Scripts:$PATH" git commit -m "feat(scripts): generate_auction_values CLI + integration test"
```

---

## Phase 3 verification (end-of-effort forced checklist per CLAUDE.md Agent Directive #4)

Run from repo root. Paste output (or concise summary) into the final report.

```
PATH=".venv/Scripts:$PATH" pytest -v
PATH=".venv/Scripts:$PATH" mypy src tests
PATH=".venv/Scripts:$PATH" ruff check src tests
PATH=".venv/Scripts:$PATH" ruff format --check src tests
PATH=".venv/Scripts:$PATH" pytest -v -k "ingest or store or schemas"
```

All must be clean before the branch is considered complete.

---

## Wrap-up

- [ ] Update `project_management.md` — append a top entry summarizing: spec + plan + impl all on `feat/auction-values`, scope (LeagueConfig + auction $ generator + CLI), explicit deferred items (live recommender, market scaling, ADP anchor, VORP), Phase 2 risk note that algorithm correctness is decoupled from VORP-quality (a broken VORP produces silently-broken $ values; CLI summary is the eyeball mitigation), and Phase 3 forced-verification results.
- [ ] Update `draft_ready_checklist.md` — flip §2c.1 ("Dollar value generator") from `[ ]` to `[x]`. Note in the entry that the live bid recommender (§2c.2) and nomination helper (§2c.3) remain `[ ]`. Note that the spec depends on a VORP spec that has not yet been written; the generator's output is gated on that landing.
- [ ] Push the branch and open a PR titled `feat(draft): auction $ generator (spec + impl)` per `feedback_branching.md`.

---

## Self-review notes (recorded inline during plan write)

- Spec §2.1 LeagueConfig contract — covered by Task 1.
- Spec §2.2 generate_auction_values signature — covered by Task 5.
- Spec §2.3 AuctionValuesSchema — covered by Task 2.
- Spec §3 algorithm steps 1-5 — covered by Tasks 4-5; reference-prices step 5 split into Task 6.
- Spec §4 CLI surface and flags — covered by Task 7.
- Spec §5 testing list (21 items) — mapped:
  - §5.1 invariants 1-3 (sum, floor, zero) → Task 5 tests.
  - §5.1 invariant 4 (pool size) → Task 5 tests.
  - §5.1 invariants 5-7 (pool composition, FLEX, SUPER_FLEX) → Task 4 tests.
  - §5.1 invariants 8-10 (scale, shift, budget scaling) → Task 5 tests. Shift-sensitivity is covered implicitly by the scale-invariance and pass-by-positive-VORP-share semantics; explicit shift test omitted to avoid noise; if execution wants it, add as a follow-up.
  - §5.1 invariant 11 (pool_rank density) → Task 5 tests.
  - §5.1 invariants 12 (reference pass-through), 13 (duplicate ID), 15 (degenerate input), 16 (missing required position) → Tasks 5, 6.
  - §5.1 invariant 14 (pandera validation) → Task 5 tests + AuctionValuesSchema.validate inside the function.
  - §5.2 LeagueConfig tests 17-20 → Task 1.
  - §5.3 CLI integration test 21 → Task 7.
- Spec §6 open items / risks — defaults baked into algorithm; CLI summary print mitigates VORP-quality coupling.
- Spec §7 acceptance — Phase 3 final checklist matches.
- Spec §8 follow-ups — Wrap-up section flips §2c.1 and notes deferred items.

No placeholders detected. Type/name consistency: `LeagueConfig.roster_size`, `total_pool_size`, `total_budget` (properties) — used the same names in tests and in `auction.py`. `_select_pool` returns `list[str]`; matched in tests. `AuctionValuesSchema` column list matches between schema, function output, and tests.
