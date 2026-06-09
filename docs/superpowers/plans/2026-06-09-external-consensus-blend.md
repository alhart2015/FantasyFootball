# External Consensus Projection Layer — v1 blend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive a single per-player published preseason projection (`consensus_projections`) from the already-ingested ESPN+Sleeper `external_projections` snapshots: consensus ADP/ranking + ESPN-derived point estimates, as the contract Draft Hub consumes.

**Architecture:** A pure `build_consensus` groups one `external_projections` snapshot by `gsis_id` (mean ADP across sources → ordinal `consensus_rank`; mean stat line → fantasy points via a new fractional-aware scorer). A `refresh_consensus` orchestrator reads the raw snapshot, builds, validates against a new `ConsensusProjectionSchema`, and writes a derived snapshot under `data/processed/consensus_projections/season=YYYY/asof=YYYY-MM-DD/` via sanctioned store I/O. Point estimates only; distributions/scraping/weighting are later slices.

**Tech Stack:** Python 3.12, pandas, pandera (DataFrameModel schemas), pydantic (Ruleset/StatLine), pytest, mypy strict, ruff.

**Spec:** `docs/superpowers/specs/2026-06-09-external-consensus-blend-design.md`

**Conventions (CLAUDE.md):** `GsisId` canonical; reference enums not strings; `df = SCHEMA.validate(df)` with reassignment at module boundaries; `pd.Int64Dtype()` for nullable ints, `pd.StringDtype("pyarrow")` for nullable strings; store I/O only via `write_partition`/`read_partition`/`read_latest_partition`; scoring math lives only in `src/projections/scoring/`.

**Run tests with `PYTHONPATH=src`** (the editable install points at the main checkout, not this worktree). On Windows PowerShell: `$env:PYTHONPATH="src"; python -m pytest ...`.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `src/projections/ingest/identity.py` | **New.** Shared `placeholder_name_key` + `NAME_SUFFIXES` (moved from `external_projections.py`). | 1 |
| `src/projections/ingest/external_projections.py` | **Modify.** Import the key from `identity` instead of defining it. | 1 |
| `src/projections/scoring/score.py` | **Modify.** Factor shared arithmetic into `_score_fields`; add `expected_points(Mapping, Ruleset)`. | 2 |
| `src/projections/schemas.py` | **Modify.** Add `ConsensusProjectionSchema`. | 3 |
| `src/projections/consensus/__init__.py` | **New.** Package marker + exports. | 4 |
| `src/projections/consensus/blend.py` | **New.** Pure `build_consensus(external, ruleset)`. | 4 |
| `src/projections/consensus/refresh.py` | **New.** `refresh_consensus(...)` orchestrator + `ConsensusError` + CLI. | 5 |
| `tests/test_ingest/test_identity.py` | **New.** Unit tests for the moved key. | 1 |
| `tests/test_scoring/test_expected_points.py` | **New.** Fractional scoring + int-equivalence to `score()`. | 2 |
| `tests/test_schemas/test_consensus_projection_schema.py` | **New.** Schema accept/reject + nullability. | 3 |
| `tests/test_consensus/__init__.py` | **New.** Package marker. | 4 |
| `tests/test_consensus/test_blend.py` | **New.** `build_consensus` behavior. | 4 |
| `tests/test_consensus/test_refresh.py` | **New.** Orchestrator store round-trip + error. | 5 |

---

## Task 1: Move `placeholder_name_key` to a shared identity util

Pure refactor (TODO #38 directive). The key currently lives privately in `external_projections.py`; move it so the blend and ingest share one identity source. Ingest behavior and existing placeholder ids must be unchanged.

**Files:**
- Create: `src/projections/ingest/identity.py`
- Modify: `src/projections/ingest/external_projections.py` (the `_NAME_SUFFIXES` constant + `_placeholder_name_key` function, and the 2 call sites at `_make_placeholder_gsis` and the placeholder-collision check)
- Test: `tests/test_ingest/test_identity.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingest/test_identity.py`:

```python
from __future__ import annotations

from projections.ingest.identity import placeholder_name_key


def test_folds_accents_and_lowercases() -> None:
    # 'José'/'Jose' must agree across sources
    assert placeholder_name_key("José Hernández", "RB") == placeholder_name_key("Jose Hernandez", "RB")


def test_strips_generational_suffixes_and_punctuation() -> None:
    assert placeholder_name_key("Marvin Harrison Jr.", "WR") == placeholder_name_key("Marvin Harrison", "WR")
    # hyphen/punctuation removed
    assert placeholder_name_key("Amon-Ra St. Brown", "WR") == "amonrastbrown|wr"


def test_position_is_part_of_key() -> None:
    assert placeholder_name_key("Taysom Hill", "QB") != placeholder_name_key("Taysom Hill", "TE")


def test_degenerate_name_falls_back_to_raw_lower() -> None:
    # all-suffix/punctuation name keys on the raw name, not the empty '|pos'
    assert placeholder_name_key("Jr.", "WR") == "jr.|wr"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/test_ingest/test_identity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'projections.ingest.identity'`.

- [ ] **Step 3: Create the identity module**

Create `src/projections/ingest/identity.py` (move the body verbatim from `external_projections.py`, rename to public `placeholder_name_key`, export `NAME_SUFFIXES`):

```python
"""Shared player-identity helpers for external-projection ingest and the consensus blend.

`placeholder_name_key` is the single source of truth for the normalized (name, position)
key that reconciles the same rookie across sources (and seeds the deterministic placeholder
gsis_id when a player is not yet in id_map). Ingest and any downstream cross-source matching
import it from here so they agree by construction rather than re-deriving the rule.
"""

from __future__ import annotations

import re
import unicodedata

# Generational suffixes dropped from the identity key (Jr/Sr/II/III/IV/V).
NAME_SUFFIXES: frozenset[str] = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})


def placeholder_name_key(full_name: str, position: str) -> str:
    """Normalize (full_name, position) into a stable cross-source key: accents folded to ASCII
    (so 'José'/'Jose' agree across sources), lowercased, punctuation/whitespace removed, common
    generational suffixes (Jr/Sr/II…) dropped. ESPN and Sleeper spell the same rookie nearly
    identically, so this lets both sources' rows reconcile."""
    folded = (
        unicodedata.normalize("NFKD", full_name).encode("ascii", "ignore").decode("ascii").lower()
    )
    tokens = [t for t in re.split(r"[^a-z0-9]+", folded) if t and t not in NAME_SUFFIXES]
    if tokens:
        return "".join(tokens) + "|" + position.lower()
    # Degenerate name (all suffix/punctuation, or non-ASCII that folded to nothing): key on the raw
    # name instead, so two such distinct players don't both collapse to the position-only key
    # '|<pos>' and collide into one placeholder gsis.
    return full_name.strip().lower() + "|" + position.lower()
```

> Note: confirm the moved `NAME_SUFFIXES` set matches the original `_NAME_SUFFIXES` in `external_projections.py` exactly. Read the original constant before copying — if it differs from the set above, use the original's contents.

- [ ] **Step 4: Rewire `external_projections.py` to import from `identity`**

In `src/projections/ingest/external_projections.py`:
1. Delete the local `_NAME_SUFFIXES` constant and the `_placeholder_name_key` function (lines around 199–213).
2. Add to the imports block: `from projections.ingest.identity import placeholder_name_key`.
3. Replace the two call sites — `_make_placeholder_gsis` (was `_placeholder_name_key(full_name, position)`) and the collision check (was `_placeholder_name_key(name, pos)`) — with `placeholder_name_key(...)`.
4. Remove now-unused imports if `re`/`unicodedata` are no longer referenced elsewhere in the file (run ruff to confirm; `re` is likely still used — only remove what ruff flags).

- [ ] **Step 5: Run identity + external_projections tests to verify they pass**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/test_ingest/test_identity.py tests/test_ingest/test_external_projections.py -v`
Expected: PASS (identity tests green; external_projections tests unchanged-green — proves the move preserved behavior).

- [ ] **Step 6: Lint + types**

Run: `$env:PYTHONPATH="src"; ruff check src/projections/ingest tests/test_ingest; python -m mypy src/projections/ingest`
Expected: clean. Fix any unused-import (`F401`) flagged in `external_projections.py`.

- [ ] **Step 7: Commit**

```bash
git add src/projections/ingest/identity.py src/projections/ingest/external_projections.py tests/test_ingest/test_identity.py
git commit -m "refactor(ingest): extract placeholder_name_key to shared identity util (TODO #38)"
```

---

## Task 2: Fractional-aware `expected_points` in the scoring layer

`score()` requires integer count stats (its `StatLine` types TDs/receptions/etc. as `int`), so it cannot score ESPN's fractional preseason projections (e.g. 8.4 receiving TDs). Add a mapping-based scorer that shares the exact same coefficient arithmetic.

**Files:**
- Modify: `src/projections/scoring/score.py`
- Test: `tests/test_scoring/test_expected_points.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_scoring/test_expected_points.py`:

```python
from __future__ import annotations

from projections.schemas import Ruleset
from projections.scoring.score import StatLine, expected_points, score


def test_fractional_counts_are_scored() -> None:
    r = Ruleset()  # ESPN_PPR
    line = {"receptions": 105.0, "receiving_yards": 1335.0, "receiving_tds": 8.4}
    # 105*1 + 1335/10 + 8.4*6 = 105 + 133.5 + 50.4 = 288.9
    assert expected_points(line, r) == 288.9


def test_absent_keys_treated_as_zero() -> None:
    r = Ruleset()
    assert expected_points({"rushing_yards": 100.0}, r) == 10.0
    assert expected_points({}, r) == 0.0


def test_equivalent_to_score_on_integer_lines() -> None:
    r = Ruleset()
    fields = {
        "passing_yards": 4000.0,
        "passing_tds": 30.0,
        "interceptions": 10.0,
        "rushing_yards": 200.0,
        "rushing_tds": 2.0,
        "receptions": 0.0,
        "receiving_yards": 0.0,
        "receiving_tds": 0.0,
        "fumbles_lost": 3.0,
    }
    line = StatLine(
        passing_yards=4000.0, passing_tds=30, interceptions=10,
        rushing_yards=200.0, rushing_tds=2, fumbles_lost=3,
    )
    assert expected_points(fields, r) == score(line, r)


def test_half_ppr_ruleset_applies() -> None:
    r = Ruleset.espn_half()
    assert expected_points({"receptions": 100.0}, r) == 50.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/test_scoring/test_expected_points.py -v`
Expected: FAIL — `ImportError: cannot import name 'expected_points'`.

- [ ] **Step 3: Refactor `score.py` to a shared helper + add `expected_points`**

In `src/projections/scoring/score.py`, add `from collections.abc import Mapping` to imports, then replace the `score` function with the factored version and add `expected_points`:

```python
def _score_fields(f: Mapping[str, float], ruleset: Ruleset) -> float:
    """Shared scoring arithmetic over a stat-field -> value mapping. Absent keys count as 0.0.
    The single source of fantasy-points truth; both score() (realized int lines) and
    expected_points() (fractional projections) delegate here."""
    pts = 0.0
    pts += f.get("passing_yards", 0.0) / ruleset.passing_yds_per_pt
    pts += f.get("passing_tds", 0.0) * ruleset.passing_td_pts
    pts += f.get("interceptions", 0.0) * ruleset.interception_pts
    pts += f.get("passing_2pt_conversions", 0.0) * ruleset.two_pt_pts

    pts += f.get("rushing_yards", 0.0) / ruleset.rushing_yds_per_pt
    pts += f.get("rushing_tds", 0.0) * ruleset.rushing_td_pts
    pts += f.get("rushing_2pt_conversions", 0.0) * ruleset.two_pt_pts

    pts += f.get("receptions", 0.0) * ruleset.reception_pts
    pts += f.get("receiving_yards", 0.0) / ruleset.receiving_yds_per_pt
    pts += f.get("receiving_tds", 0.0) * ruleset.receiving_td_pts
    pts += f.get("receiving_2pt_conversions", 0.0) * ruleset.two_pt_pts

    pts += f.get("fumbles_lost", 0.0) * ruleset.fumble_lost_pts
    pts += f.get("return_tds", 0.0) * ruleset.return_td_pts
    return pts


def score(line: StatLine, ruleset: Ruleset) -> float:
    """Convert a `StatLine` to fantasy points under `ruleset`. Pure function."""
    return _score_fields(line.model_dump(), ruleset)


def expected_points(line: Mapping[str, float], ruleset: Ruleset) -> float:
    """Score a fractional *expected* stat line (e.g. a preseason projection's 8.4 receiving TDs)
    under `ruleset`, using the same coefficients as `score()`. Absent fields count as 0.0."""
    return _score_fields(line, ruleset)
```

- [ ] **Step 4: Run scoring tests to verify they pass**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/test_scoring/ -v`
Expected: PASS — new `test_expected_points.py` green AND existing `test_score.py` / `test_actuals.py` still green (proves the `score()` refactor is behavior-preserving).

- [ ] **Step 5: Lint + types**

Run: `$env:PYTHONPATH="src"; ruff check src/projections/scoring tests/test_scoring; python -m mypy src/projections/scoring`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/projections/scoring/score.py tests/test_scoring/test_expected_points.py
git commit -m "feat(scoring): add fractional expected_points(); factor shared scoring arithmetic"
```

---

## Task 3: `ConsensusProjectionSchema` published contract

**Files:**
- Modify: `src/projections/schemas.py` (add the class immediately after `ExternalProjectionSchema`, before `ProjectionWeeklySchema`)
- Test: `tests/test_schemas/test_consensus_projection_schema.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_schemas/test_consensus_projection_schema.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from projections.schemas import ConsensusProjectionSchema


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": ["00-0036900", "99-0001234"],
            "season": pd.array([2026, 2026], dtype="Int64"),
            "asof": ["2026-06-09", "2026-06-09"],
            "full_name": ["Ja'Marr Chase", "Some Rookie"],
            "position": ["WR", "RB"],
            "consensus_adp": [4.1, pd.NA],
            "consensus_rank": pd.array([1, pd.NA], dtype="Int64"),
            "n_adp_sources": pd.array([2, 0], dtype="Int64"),
            "has_points": [True, False],
            "projected_points_ppr": [288.9, pd.NA],
            "passing_yards": [0.0, pd.NA],
            "passing_tds": [0.0, pd.NA],
            "interceptions": [0.0, pd.NA],
            "rushing_yards": [21.0, pd.NA],
            "rushing_tds": [0.0, pd.NA],
            "receptions": [119.0, pd.NA],
            "receiving_yards": [1506.0, pd.NA],
            "receiving_tds": [8.4, pd.NA],
            "fumbles_lost": [0.0, pd.NA],
            "is_placeholder_gsis": [False, True],
            "ruleset": ["ESPN_PPR", "ESPN_PPR"],
        }
    )


def test_accepts_wellformed_frame() -> None:
    out = ConsensusProjectionSchema.validate(_valid_frame())
    assert len(out) == 2
    # nullable cols carry pd.NA through
    assert pd.isna(out.loc[1, "consensus_adp"])
    assert pd.isna(out.loc[1, "consensus_rank"])


def test_rejects_bad_gsis_id() -> None:
    df = _valid_frame()
    df.loc[0, "gsis_id"] = "not-a-gsis"
    with pytest.raises(Exception):
        ConsensusProjectionSchema.validate(df)


def test_rejects_unknown_position() -> None:
    df = _valid_frame()
    df.loc[0, "position"] = "DST"
    with pytest.raises(Exception):
        ConsensusProjectionSchema.validate(df)


def test_gsis_id_must_be_unique() -> None:
    df = _valid_frame()
    df.loc[1, "gsis_id"] = "00-0036900"  # duplicate
    with pytest.raises(Exception):
        ConsensusProjectionSchema.validate(df)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/test_schemas/test_consensus_projection_schema.py -v`
Expected: FAIL — `ImportError: cannot import name 'ConsensusProjectionSchema'`.

- [ ] **Step 3: Add the schema**

In `src/projections/schemas.py`, immediately after the `ExternalProjectionSchema` class (and before `ProjectionWeeklySchema`), add:

```python
class ConsensusProjectionSchema(pa.DataFrameModel):
    """Published preseason consensus projection: one row per (gsis_id, season, asof).

    The consumer-facing contract downstream draft tooling reads. `consensus_adp` is the mean of
    available source ADPs (nullable — a stat-line-only / unranked player can have none);
    `consensus_rank` is the ordinal over non-null `consensus_adp` (null when adp is null). The
    stat line + `projected_points_ppr` are present only for players a stat-line source covers
    (`has_points`). v1 sources: ESPN (stat line + ADP) + Sleeper (ADP only).
    """

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$", unique=True)
    season: Series[pd.Int64Dtype] = pa.Field(ge=1999, le=2100)
    # ISO YYYY-MM-DD; mirrors the raw external_projections snapshot this was derived from
    asof: Series[str] = pa.Field(str_matches=r"^\d{4}-\d{2}-\d{2}$")
    full_name: Series[str]
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    consensus_adp: Series[float] = pa.Field(gt=0, nullable=True)
    consensus_rank: Series[pd.Int64Dtype] = pa.Field(ge=1, nullable=True)
    n_adp_sources: Series[pd.Int64Dtype] = pa.Field(ge=0)
    has_points: Series[bool]
    projected_points_ppr: Series[float] = pa.Field(nullable=True)
    passing_yards: Series[float] = pa.Field(nullable=True)
    passing_tds: Series[float] = pa.Field(nullable=True)
    interceptions: Series[float] = pa.Field(nullable=True)
    rushing_yards: Series[float] = pa.Field(nullable=True)
    rushing_tds: Series[float] = pa.Field(nullable=True)
    receptions: Series[float] = pa.Field(nullable=True)
    receiving_yards: Series[float] = pa.Field(nullable=True)
    receiving_tds: Series[float] = pa.Field(nullable=True)
    fumbles_lost: Series[float] = pa.Field(nullable=True)
    is_placeholder_gsis: Series[bool]
    ruleset: Series[str]

    class Config:
        strict = "filter"
        coerce = True
```

- [ ] **Step 4: Run schema tests to verify they pass**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/test_schemas/test_consensus_projection_schema.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Lint + types**

Run: `$env:PYTHONPATH="src"; ruff check src/projections/schemas.py tests/test_schemas/test_consensus_projection_schema.py; python -m mypy src/projections/schemas.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/projections/schemas.py tests/test_schemas/test_consensus_projection_schema.py
git commit -m "feat(schemas): add ConsensusProjectionSchema published contract"
```

---

## Task 4: `build_consensus` pure blend

**Files:**
- Create: `src/projections/consensus/__init__.py`
- Create: `src/projections/consensus/blend.py`
- Create: `tests/test_consensus/__init__.py`
- Test: `tests/test_consensus/test_blend.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_consensus/__init__.py` (empty file).

Create `tests/test_consensus/test_blend.py`:

```python
from __future__ import annotations

import pandas as pd

from projections.consensus.blend import build_consensus
from projections.schemas import ConsensusProjectionSchema, Ruleset

_STAT_COLS = [
    "passing_yards", "passing_tds", "interceptions", "rushing_yards", "rushing_tds",
    "receptions", "receiving_yards", "receiving_tds", "fumbles_lost",
]


def _row(source, gsis_id, *, adp, full_name, position, placeholder, stats=None):
    r = {
        "source": source,
        "source_player_id": f"{source}-{gsis_id}",
        "gsis_id": gsis_id,
        "is_placeholder_gsis": placeholder,
        "full_name": full_name,
        "position": position,
        "season": 2026,
        "asof": "2026-06-09",
        "adp": adp,
        "espn_draft_rank": pd.NA,
    }
    for c in _STAT_COLS:
        r[c] = (stats or {}).get(c, pd.NA)
    return r


def _external(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_two_source_veteran_blends_adp_and_scores_points() -> None:
    chase_stats = {"receptions": 119.0, "receiving_yards": 1506.0, "receiving_tds": 8.4, "rushing_yards": 21.0,
                   "passing_yards": 0.0, "passing_tds": 0.0, "interceptions": 0.0, "rushing_tds": 0.0, "fumbles_lost": 0.0}
    ext = _external([
        _row("ESPN", "00-0036900", adp=4.8, full_name="Ja'Marr Chase", position="WR", placeholder=False, stats=chase_stats),
        _row("SLEEPER", "00-0036900", adp=3.4, full_name="Ja'Marr Chase", position="WR", placeholder=False),
    ])
    out = build_consensus(ext, Ruleset())
    out = ConsensusProjectionSchema.validate(out)  # self-conforming
    assert len(out) == 1
    r = out.iloc[0]
    assert r["consensus_adp"] == 4.1  # mean(4.8, 3.4)
    assert r["n_adp_sources"] == 2
    assert bool(r["has_points"]) is True
    # 119 + 1506/10 + 8.4*6 + 21/10 = 119 + 150.6 + 50.4 + 2.1 = 322.1
    assert round(float(r["projected_points_ppr"]), 1) == 322.1
    assert r["receiving_tds"] == 8.4
    assert r["consensus_rank"] == 1


def test_sleeper_only_player_has_adp_no_points() -> None:
    ext = _external([
        _row("SLEEPER", "00-0011111", adp=50.0, full_name="Deep Sleeper", position="RB", placeholder=False),
    ])
    out = build_consensus(ext, Ruleset())
    r = out[out["gsis_id"] == "00-0011111"].iloc[0]
    assert r["n_adp_sources"] == 1
    assert bool(r["has_points"]) is False
    assert pd.isna(r["projected_points_ppr"])
    assert all(pd.isna(r[c]) for c in _STAT_COLS)


def test_union_coverage_and_deterministic_rank() -> None:
    ext = _external([
        _row("SLEEPER", "00-0000002", adp=10.0, full_name="B Player", position="WR", placeholder=False),
        _row("ESPN", "00-0000001", adp=10.0, full_name="A Player", position="RB", placeholder=False,
             stats={c: 0.0 for c in _STAT_COLS}),
        _row("SLEEPER", "00-0000003", adp=5.0, full_name="C Player", position="TE", placeholder=False),
    ])
    out = build_consensus(ext, Ruleset())
    assert len(out) == 3  # union
    by_id = out.set_index("gsis_id")
    assert by_id.loc["00-0000003", "consensus_rank"] == 1  # adp 5 first
    # tie at adp 10: ordered by gsis_id, so ...0001 (rank 2) before ...0002 (rank 3)
    assert by_id.loc["00-0000001", "consensus_rank"] == 2
    assert by_id.loc["00-0000002", "consensus_rank"] == 3


def test_player_with_points_but_no_adp_gets_null_rank() -> None:
    ext = _external([
        _row("ESPN", "00-0000009", adp=pd.NA, full_name="No ADP", position="WR", placeholder=False,
             stats={c: 0.0 for c in _STAT_COLS}),
    ])
    out = build_consensus(ext, Ruleset())
    r = out[out["gsis_id"] == "00-0000009"].iloc[0]
    assert pd.isna(r["consensus_adp"])
    assert pd.isna(r["consensus_rank"])
    assert r["n_adp_sources"] == 0
    assert bool(r["has_points"]) is True


def test_placeholder_rookie_carried_through() -> None:
    ext = _external([
        _row("ESPN", "99-0001234", adp=17.6, full_name="Jeremiyah Love", position="RB", placeholder=True,
             stats={c: 0.0 for c in _STAT_COLS}),
        _row("SLEEPER", "99-0001234", adp=18.0, full_name="Jeremiyah Love", position="RB", placeholder=True),
    ])
    out = build_consensus(ext, Ruleset())
    r = out[out["gsis_id"] == "99-0001234"].iloc[0]
    assert bool(r["is_placeholder_gsis"]) is True
    assert r["n_adp_sources"] == 2


def test_empty_input_returns_empty_conforming_frame() -> None:
    cols = ["source", "source_player_id", "gsis_id", "is_placeholder_gsis", "full_name", "position",
            "season", "asof", "adp", "espn_draft_rank", *_STAT_COLS]
    out = build_consensus(pd.DataFrame(columns=cols), Ruleset())
    assert out.empty
    ConsensusProjectionSchema.validate(out)  # empty frame still conforms
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/test_consensus/test_blend.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'projections.consensus'`.

- [ ] **Step 3: Create the package marker**

Create `src/projections/consensus/__init__.py`:

```python
"""Consensus projection layer: blend external sources into one published preseason projection."""

from projections.consensus.blend import build_consensus

__all__ = ["build_consensus"]
```

- [ ] **Step 4: Implement `build_consensus`**

Create `src/projections/consensus/blend.py`:

```python
"""Pure blend of one external_projections snapshot into per-player consensus rows.

Groups by gsis_id: mean ADP across sources (-> ordinal consensus_rank), mean stat line across
the sources that carry one (-> fantasy points via scoring.expected_points). Union coverage: every
player ranked by >=1 source appears. No I/O; the orchestrator (refresh.py) handles read/validate/write.
"""

from __future__ import annotations

import pandas as pd

from projections.schemas import Ruleset
from projections.scoring.score import expected_points

# The 9 canonical preseason stat-line fields the ExternalProjectionSchema carries.
STAT_FIELDS: tuple[str, ...] = (
    "passing_yards",
    "passing_tds",
    "interceptions",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "fumbles_lost",
)

_OUTPUT_COLUMNS: tuple[str, ...] = (
    "gsis_id",
    "season",
    "asof",
    "full_name",
    "position",
    "consensus_adp",
    "consensus_rank",
    "n_adp_sources",
    "has_points",
    "projected_points_ppr",
    *STAT_FIELDS,
    "is_placeholder_gsis",
    "ruleset",
)


def _empty_output() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in _OUTPUT_COLUMNS})


def build_consensus(external: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame:
    """Blend one validated external_projections snapshot into ConsensusProjectionSchema-shaped rows.

    `external` carries one row per (source, gsis_id), all sharing one season + asof. Returns one row
    per gsis_id; pass the result through ConsensusProjectionSchema.validate at the call site.
    """
    if external.empty:
        return _empty_output()

    season = int(external["season"].iloc[0])
    asof = str(external["asof"].iloc[0])

    records: list[dict[str, object]] = []
    for gsis_id, grp in external.groupby("gsis_id", sort=False):
        adp_vals = grp["adp"].dropna()
        n_adp_sources = int(adp_vals.shape[0])
        consensus_adp: float | None = float(adp_vals.mean()) if n_adp_sources > 0 else None

        # Prefer a stat-bearing row for identity (full_name/position); fall back to the first row.
        stat_mask = grp[list(STAT_FIELDS)].notna().any(axis=1)
        identity_row = grp[stat_mask].iloc[0] if stat_mask.any() else grp.iloc[0]

        statline: dict[str, float] = {}
        has_points = False
        for field in STAT_FIELDS:
            vals = grp[field].dropna()
            if not vals.empty:
                statline[field] = float(vals.mean())
                has_points = True

        projected = expected_points(statline, ruleset) if has_points else None

        rec: dict[str, object] = {
            "gsis_id": gsis_id,
            "season": season,
            "asof": asof,
            "full_name": str(identity_row["full_name"]),
            "position": str(identity_row["position"]),
            "consensus_adp": consensus_adp,
            "consensus_rank": pd.NA,  # filled after the full-group ranking below
            "n_adp_sources": n_adp_sources,
            "has_points": has_points,
            "projected_points_ppr": projected,
            "is_placeholder_gsis": bool(identity_row["is_placeholder_gsis"]),
            "ruleset": ruleset.name,
        }
        for field in STAT_FIELDS:
            rec[field] = statline.get(field, pd.NA)
        records.append(rec)

    df = pd.DataFrame.from_records(records)

    # Ordinal rank over non-null consensus_adp, deterministic tie-break on gsis_id.
    df = df.sort_values(["consensus_adp", "gsis_id"], na_position="last").reset_index(drop=True)
    ranked = df["consensus_adp"].notna()
    df["consensus_rank"] = pd.NA
    df.loc[ranked, "consensus_rank"] = range(1, int(ranked.sum()) + 1)

    # Nullable-aware dtypes (schema coerces, but be explicit so pd.NA survives).
    df["season"] = df["season"].astype("Int64")
    df["n_adp_sources"] = df["n_adp_sources"].astype("Int64")
    df["consensus_rank"] = df["consensus_rank"].astype("Int64")

    return df[list(_OUTPUT_COLUMNS)]
```

- [ ] **Step 5: Run blend tests to verify they pass**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/test_consensus/test_blend.py -v`
Expected: PASS (6 tests).

- [ ] **Step 6: Lint + types**

Run: `$env:PYTHONPATH="src"; ruff check src/projections/consensus tests/test_consensus; python -m mypy src/projections/consensus`
Expected: clean. (If mypy flags the `groupby` key type, annotate `gsis_id` via `str(gsis_id)` when assigning to `rec["gsis_id"]`.)

- [ ] **Step 7: Commit**

```bash
git add src/projections/consensus/__init__.py src/projections/consensus/blend.py tests/test_consensus/__init__.py tests/test_consensus/test_blend.py
git commit -m "feat(consensus): build_consensus pure blend over external_projections"
```

---

## Task 5: `refresh_consensus` orchestrator + CLI

**Files:**
- Create: `src/projections/consensus/refresh.py`
- Test: `tests/test_consensus/test_refresh.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_consensus/test_refresh.py`:

```python
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from projections.consensus.refresh import ConsensusError, refresh_consensus
from projections.schemas import ConsensusProjectionSchema
from projections.store import read_partition, write_partition

_STAT_COLS = [
    "passing_yards", "passing_tds", "interceptions", "rushing_yards", "rushing_tds",
    "receptions", "receiving_yards", "receiving_tds", "fumbles_lost",
]


def _raw_external() -> pd.DataFrame:
    """A minimal validated-shape external_projections snapshot (ESPN + Sleeper, one veteran)."""
    base = {
        "source_player_id": "x",
        "is_placeholder_gsis": False,
        "full_name": "Ja'Marr Chase",
        "position": "WR",
        "season": 2026,
        "asof": "2026-06-09",
        "espn_draft_rank": pd.NA,
    }
    espn_stats = {c: 0.0 for c in _STAT_COLS} | {"receptions": 119.0, "receiving_yards": 1506.0, "receiving_tds": 8.0}
    espn = {**base, "source": "ESPN", "gsis_id": "00-0036900", "adp": 4.8, **espn_stats}
    sleeper = {**base, "source": "SLEEPER", "gsis_id": "00-0036900", "adp": 3.4, **{c: pd.NA for c in _STAT_COLS}}
    return pd.DataFrame([espn, sleeper])


def _seed_raw(data_root: Path) -> None:
    write_partition(
        data_root / "raw", "external_projections", _raw_external(),
        season=2026, asof=date(2026, 6, 9),
    )


def test_refresh_writes_validated_consensus_snapshot(tmp_path: Path) -> None:
    _seed_raw(tmp_path)
    out_path = refresh_consensus(tmp_path, season=2026)
    assert out_path.exists()
    # consensus asof mirrors the raw snapshot's asof
    assert "asof=2026-06-09" in str(out_path)
    df = read_partition(tmp_path / "processed", "consensus_projections", season=2026, asof=date(2026, 6, 9))
    ConsensusProjectionSchema.validate(df)
    assert len(df) == 1
    r = df.iloc[0]
    assert r["gsis_id"] == "00-0036900"
    assert r["consensus_adp"] == 4.1
    assert r["n_adp_sources"] == 2
    assert bool(r["has_points"]) is True


def test_explicit_asof_reads_that_snapshot(tmp_path: Path) -> None:
    _seed_raw(tmp_path)
    out_path = refresh_consensus(tmp_path, season=2026, asof=date(2026, 6, 9))
    assert "asof=2026-06-09" in str(out_path)


def test_missing_raw_snapshot_raises_consensus_error(tmp_path: Path) -> None:
    with pytest.raises(ConsensusError):
        refresh_consensus(tmp_path, season=2026)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/test_consensus/test_refresh.py -v`
Expected: FAIL — `ImportError: cannot import name 'ConsensusError'`.

- [ ] **Step 3: Implement `refresh_consensus` + CLI**

Create `src/projections/consensus/refresh.py`:

```python
"""Orchestrator + CLI: read one external_projections snapshot, blend, validate, write the
derived consensus_projections snapshot.

Usage:
    python -m projections.consensus.refresh --season 2026 [--asof YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from projections.consensus.blend import build_consensus
from projections.ingest.manifest import record as record_manifest
from projections.schemas import ConsensusProjectionSchema, Ruleset
from projections.store import read_latest_partition, read_partition, write_partition

_log = logging.getLogger(__name__)


class ConsensusError(RuntimeError):
    """Raised when the raw external_projections snapshot needed to build a consensus is missing.
    The CLI converts it to SystemExit; programmatic callers can catch it."""


def refresh_consensus(data_root: Path, *, season: int, asof: date | None = None) -> Path:
    """Build and write one consensus_projections snapshot from an external_projections snapshot.

    `asof=None` uses the latest raw snapshot for the season; otherwise the named one. The written
    consensus snapshot's asof MIRRORS the raw snapshot it was derived from (reproducible from input).
    """
    raw_root = data_root / "raw"
    try:
        if asof is not None:
            external = read_partition(raw_root, "external_projections", season=season, asof=asof)
            snapshot_asof = asof
        else:
            external = read_latest_partition(raw_root, "external_projections", season=season)
            snapshot_asof = date.fromisoformat(str(external["asof"].iloc[0]))
    except FileNotFoundError as exc:
        raise ConsensusError(
            f"No external_projections snapshot for season={season}"
            f"{f' asof={asof.isoformat()}' if asof else ''}; run the ingest first."
        ) from exc

    if external.empty:
        raise ConsensusError(
            f"external_projections snapshot for season={season} asof={snapshot_asof.isoformat()} "
            f"is empty; refusing to write an empty consensus snapshot."
        )

    frame = build_consensus(external, ruleset=Ruleset())
    frame = ConsensusProjectionSchema.validate(frame)
    _log.info(
        "consensus_projections season=%s asof=%s: wrote %d players (with_points=%d, placeholders=%d).",
        season,
        snapshot_asof.isoformat(),
        len(frame),
        int(frame["has_points"].sum()),
        int(frame["is_placeholder_gsis"].sum()),
    )
    out = write_partition(
        data_root / "processed", "consensus_projections", frame, season=season, asof=snapshot_asof
    )
    record_manifest(data_root, table="consensus_projections", season=season, df=frame)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the consensus preseason projection from an external_projections snapshot."
    )
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument(
        "--asof",
        type=date.fromisoformat,
        default=None,
        help="Raw snapshot date YYYY-MM-DD; defaults to the latest snapshot for the season.",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        out = refresh_consensus(args.data_root, season=args.season, asof=args.asof)
    except ConsensusError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Wrote consensus snapshot: {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run refresh tests to verify they pass**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/test_consensus/test_refresh.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint + types**

Run: `$env:PYTHONPATH="src"; ruff check src/projections/consensus tests/test_consensus; python -m mypy src/projections/consensus`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/projections/consensus/refresh.py tests/test_consensus/test_refresh.py
git commit -m "feat(consensus): refresh_consensus orchestrator + CLI"
```

---

## Task 6: Full-suite verification + live 2026 smoke + docs

**Files:**
- Modify: `TODO.md` (mark #2b slice 1 done), `project_management.md` (new top entry)

- [ ] **Step 1: Full end-of-effort gate (CLAUDE.md checklist)**

Run each at repo root and fix every failure:

```
$env:PYTHONPATH="src"; python -m pytest -q
$env:PYTHONPATH="src"; python -m mypy src tests
$env:PYTHONPATH="src"; ruff check src tests
$env:PYTHONPATH="src"; ruff format --check src tests
$env:PYTHONPATH="src"; python -m pytest -q -k "ingest or store or schemas"
```

Expected: all green. Paste a concise summary of each into the task completion notes.

- [ ] **Step 2: Live 2026 smoke (network — optional but recommended)**

The raw 2026 snapshot lives in the main checkout (gitignored, not in the worktree). Point `--data-root` at it:

```
$env:PYTHONPATH="src"; python -m projections.consensus.refresh --season 2026 --data-root C:\Users\HartAlden\FantasyFootball\data
```

Expected: prints `Wrote consensus snapshot: …\data\processed\consensus_projections\season=2026\asof=2026-06-09\part.parquet`. Spot-check: read it back, confirm ~3,042 rows (distinct gsis), Bijan Robinson `consensus_rank == 1`, the ~448 two-source veterans have `n_adp_sources == 2`, and `projected_points_ppr` is populated for `has_points` players only.

- [ ] **Step 3: Update TODO.md**

In `TODO.md` under #38 "Remaining scope (#2b+, still open)", mark the consensus-blend item done with a pointer to this slice's spec/plan and the `data/processed/consensus_projections` table, and note that the scraped-source + distribution-wrapping items remain open (now unblocked — the published contract exists for them to extend).

- [ ] **Step 4: Update project_management.md**

Add a new top entry summarizing this slice: the published `ConsensusProjectionSchema` contract, `build_consensus` + `refresh_consensus`, the `expected_points` scoring addition, union coverage / point-estimates-now decisions, and the next slice (scraped source → ≥2-source points consensus → distribution-wrapping).

- [ ] **Step 5: Commit docs**

```bash
git add TODO.md project_management.md
git commit -m "docs(pm,todo): consensus-blend v1 shipped (#2b slice 1)"
```

---

## Self-Review

**Spec coverage** — every §2 in-scope item maps to a task:
- §2.1 `expected_points` in scoring layer → Task 2. ✅
- §2.2 pure `build_consensus` → Task 4. ✅
- §2.3 `ConsensusProjectionSchema` → Task 3. ✅
- §2.4 `refresh_consensus` orchestrator + CLI → Task 5. ✅
- §2.5 `placeholder_name_key` util move → Task 1. ✅
- §3.2 union coverage, mean ADP, null-adp handling, identity preference → Task 4 tests. ✅
- §3.4 `data/processed/…`, asof-mirrors-raw, missing-snapshot error → Task 5 tests. ✅
- §5 testing matrix → Tasks 2–5 tests; §5 gates → Task 6. ✅

**Type/name consistency:** `build_consensus(external, ruleset)`, `expected_points(line, ruleset)`, `refresh_consensus(data_root, *, season, asof)`, `ConsensusError`, `ConsensusProjectionSchema`, `placeholder_name_key`, `STAT_FIELDS`, `_OUTPUT_COLUMNS` — used identically across tasks and tests. Output columns in `blend.py:_OUTPUT_COLUMNS` match `ConsensusProjectionSchema` field-for-field (strict="filter" would drop extras, but there are none). `write_partition(root, table, df, *, season, asof)` and `read_latest_partition(root, table, *, season)` match the store signatures verified in `parquet.py`.

**Placeholder scan:** no TBD/TODO/"add error handling"-style gaps; every code step shows complete, copy-ready code with exact run commands and expected output.

**Decisions deferred (not gaps):** scraped sources, ≥2-source points consensus, Distribution-wrapping, accuracy-weighting, rookie-matching refinement — all explicitly out of scope per spec §2 / §7.
