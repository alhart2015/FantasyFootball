# ESPN auction-value ingest (Slice 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture ESPN's real human-facing auction values through the ingest → consensus → VORP-table pipeline and land them on an Optional `espn_auction_dollars` pool column, with no behavior change.

**Architecture:** Extend the existing `external_projections → consensus → vorp-table` flow (Approach A). The ESPN value comes from the `kona_player_info` payload we already fetch. All four new columns are **Optional** (pandera `Series[...] | None`) so stale partitions and the weekly VORP path still validate. Resolution (crowd-now / expert-fallback-by-ruleset, NA-safe `Int64`) happens at the pool stage where the ruleset is known.

**Tech Stack:** Python 3.11, pandas (nullable `Float64`/`Int64` extension dtypes), pandera DataFrameModel schemas (`strict="filter"`, `coerce=True`), pytest.

**Spec:** `docs/superpowers/specs/2026-06-18-espn-auction-values-ingest-design.md` (hardened via 3-iteration superpowers-spec-review).

---

## File structure

| File | Change | Responsibility |
|---|---|---|
| `src/projections/schemas.py` | modify | +3 Optional `Float64` cols on `ExternalProjectionSchema` & `ConsensusProjectionSchema`; +1 Optional `Int64` col on `VorpTableSchema` |
| `src/projections/ingest/external_projections.py` | modify | `parse_espn_players` extracts the 3 ESPN fields; `_to_canonical` emits them (presence-guarded, Sleeper→NA) |
| `src/projections/consensus/blend.py` | modify | `build_consensus` carries them via first-non-null-across-group, absence-guarded; `_OUTPUT_COLUMNS` + dtype casts |
| `scripts/generate_preset_vorp_tables.py` | modify | `resolve_espn_auction_dollars` helper; `build_preset_table` resolves from `consensus` + merges onto the table |
| `tests/test_ingest/test_external_projections.py` | modify | parser + `_to_canonical` tests |
| `tests/test_consensus/test_blend.py` | modify | blend carry-through + absence-guard tests |
| `tests/test_schemas/test_consensus_projection_schema.py` | modify | Optional-column validation |
| `tests/test_scripts/test_generate_preset_vorp_tables.py` | modify | resolver + merge + weekly-path validation tests |

**Canonical column names (use verbatim everywhere):** `espn_auction_value_avg`, `espn_auction_value_ppr`, `espn_auction_value_std` (external + consensus, `Float64`), `espn_auction_dollars` (vorp/pool, `Int64`).

**Commit cadence:** one commit per task (after its tests pass). Prepend `.venv/Scripts` to PATH before `git commit` so the pre-commit mypy hook resolves to the project venv.

---

## Task 1: Ingest the three ESPN auction fields

**Files:**
- Modify: `src/projections/ingest/external_projections.py` (`parse_espn_players`, `_to_canonical`)
- Modify: `src/projections/schemas.py` (`ExternalProjectionSchema`)
- Test: `tests/test_ingest/test_external_projections.py`

- [ ] **Step 1: Write the failing parser test**

Add to `tests/test_ingest/test_external_projections.py`:

```python
from projections.ingest.external_projections import parse_espn_players


def _espn_player(pid, name, pos_id, *, adp=5.0, auction_avg=None, ppr_av=None, std_av=None):
    """Minimal kona_player_info entry for one player with a projection."""
    ownership = {"averageDraftPosition": adp}
    if auction_avg is not None:
        ownership["auctionValueAverage"] = auction_avg
    ranks = {"PPR": {"rank": 1}, "STANDARD": {"rank": 1}}
    if ppr_av is not None:
        ranks["PPR"]["auctionValue"] = ppr_av
    if std_av is not None:
        ranks["STANDARD"]["auctionValue"] = std_av
    return {
        "player": {
            "id": pid,
            "fullName": name,
            "defaultPositionId": pos_id,  # 2 == RB
            "stats": [{"seasonId": 2026, "statSplitTypeId": 0, "statSourceId": 1, "stats": {"24": 1000.0}}],
            "ownership": ownership,
            "draftRanksByRankType": ranks,
        }
    }


def test_parse_espn_extracts_auction_values():
    payload = {"players": [_espn_player(1, "Crowd Guy", 2, auction_avg=58.67, ppr_av=57, std_av=55)]}
    df = parse_espn_players(payload, 2026)
    row = df.iloc[0]
    assert row["espn_auction_value_avg"] == 58.67
    assert row["espn_auction_value_ppr"] == 57
    assert row["espn_auction_value_std"] == 55


def test_parse_espn_auction_values_non_positive_and_missing_become_none():
    payload = {
        "players": [
            _espn_player(1, "Zero Crowd", 2, auction_avg=0, ppr_av=0, std_av=0),
            _espn_player(2, "No Auction Keys", 2),  # no auction_avg / auctionValue at all
        ]
    }
    df = parse_espn_players(payload, 2026).set_index("espn_id")
    for col in ("espn_auction_value_avg", "espn_auction_value_ppr", "espn_auction_value_std"):
        assert pd.isna(df.loc["1", col])  # <=0 normalized to None  (pd.isna: robust to dtype)
        assert pd.isna(df.loc["2", col])  # missing key -> None, no crash
```

- [ ] **Step 2: Run the tests, verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ingest/test_external_projections.py -k auction -q -n0`
Expected: FAIL with `KeyError: 'espn_auction_value_avg'` (the parser doesn't produce the columns yet).

- [ ] **Step 3: Implement the parser extraction**

First, add a module-level helper near `round_count` (around line 81) — keep it at module scope, not inside the loop, mirroring `round_count`:

```python
def _pos_auction(value: float | None) -> float | None:
    """ESPN encodes 'no auction value' as 0; normalize non-positive to None (same rule as ADP)."""
    return None if value is None or value <= 0 else float(value)
```

Then, inside `parse_espn_players`, locate the block that builds `ppr_rank` and `espn_adp` (currently around lines 158-164) and the `row` dict that follows. Replace from `ppr_rank = ...` through the `row = {...}` literal with:

```python
        draft_ranks = pl.get("draftRanksByRankType") or {}
        ppr_ranks = draft_ranks.get("PPR") or {}
        std_ranks = draft_ranks.get("STANDARD") or {}
        ppr_rank = ppr_ranks.get("rank")

        # ESPN encodes "undrafted / no draft data" as ADP 0; normalize non-positive to None so the
        # raw table stores honest null (adp is nullable) rather than an in-band sentinel that every
        # downstream consumer would have to re-discover.
        espn_adp = ownership.get("averageDraftPosition")
        if espn_adp is not None and espn_adp <= 0:
            espn_adp = None
        row: dict[str, object] = {
            "espn_id": str(espn_id),
            "full_name": full_name,
            "position": position,
            "espn_adp": espn_adp,
            "espn_pos_rank": ppr_rank,
            "espn_auction_value_avg": _pos_auction(ownership.get("auctionValueAverage")),
            "espn_auction_value_ppr": _pos_auction(ppr_ranks.get("auctionValue")),
            "espn_auction_value_std": _pos_auction(std_ranks.get("auctionValue")),
        }
```

(The `ownership = pl.get("ownership") or {}` line just above this block is unchanged and still in scope.)

- [ ] **Step 4: Run the parser tests, verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ingest/test_external_projections.py -k auction -q -n0`
Expected: PASS (2 tests).

- [ ] **Step 5: Write the failing `_to_canonical` / schema test**

Add to `tests/test_ingest/test_external_projections.py`:

```python
import pandas as pd
from projections.schemas import ExternalProjectionSchema


def test_external_schema_validates_without_auction_columns():
    # A stale-style frame lacking the new columns must still validate (Optional).
    df = pd.DataFrame(
        {
            "source": pd.array(["SLEEPER"], dtype="string[pyarrow]"),  # isin(['ESPN','SLEEPER'])
            "source_player_id": pd.array(["x"], dtype="string[pyarrow]"),
            "gsis_id": pd.array(["00-0011111"], dtype="string[pyarrow]"),
            "is_placeholder_gsis": [False],
            "full_name": pd.array(["Old Row"], dtype="string[pyarrow]"),
            "position": pd.array(["RB"], dtype="string[pyarrow]"),
            "season": [2026],
            "asof": pd.array(["2026-06-09"], dtype="string[pyarrow]"),
            "adp": pd.array([5.0], dtype="Float64"),
            "espn_draft_rank": pd.array([pd.NA], dtype="Float64"),
            **{f: pd.array([pd.NA], dtype="Float64") for f in (
                "passing_yards", "passing_tds", "interceptions", "rushing_yards", "rushing_tds",
                "receptions", "receiving_yards", "receiving_tds", "fumbles_lost")},
        }
    )
    out = ExternalProjectionSchema.validate(df)  # must not raise
    assert "espn_auction_value_avg" not in out.columns  # absent -> stays absent, no fabricate


def test_external_schema_auction_columns_are_float64():
    df = pd.DataFrame(
        {
            "source": pd.array(["ESPN"], dtype="string[pyarrow]"),  # isin(['ESPN','SLEEPER'])
            "source_player_id": pd.array(["1"], dtype="string[pyarrow]"),
            "gsis_id": pd.array(["00-0011111"], dtype="string[pyarrow]"),
            "is_placeholder_gsis": [False],
            "full_name": pd.array(["E"], dtype="string[pyarrow]"),
            "position": pd.array(["RB"], dtype="string[pyarrow]"),
            "season": [2026],
            "asof": pd.array(["2026-06-09"], dtype="string[pyarrow]"),
            "adp": pd.array([5.0], dtype="Float64"),
            "espn_draft_rank": pd.array([pd.NA], dtype="Float64"),
            "espn_auction_value_avg": pd.array([58.67], dtype="Float64"),
            "espn_auction_value_ppr": pd.array([57.0], dtype="Float64"),
            "espn_auction_value_std": pd.array([55.0], dtype="Float64"),
            **{f: pd.array([pd.NA], dtype="Float64") for f in (
                "passing_yards", "passing_tds", "interceptions", "rushing_yards", "rushing_tds",
                "receptions", "receiving_yards", "receiving_tds", "fumbles_lost")},
        }
    )
    out = ExternalProjectionSchema.validate(df)
    assert str(out["espn_auction_value_avg"].dtype) == "Float64"


def test_to_canonical_sleeper_auction_columns_are_float64_na():
    # _to_canonical null-fills the ESPN-only auction columns for a Sleeper frame; they must land
    # as Float64/pd.NA (not float64/NaN — the CLAUDE.md dtype-regression trap, spec Testing).
    from datetime import date
    from projections.ingest.external_projections import _to_canonical
    from projections.schemas import STAT_FIELDS, ProjectionSource

    sleeper = pd.DataFrame(
        {
            "sleeper_id": ["s1"], "full_name": ["Sleeper Guy"], "position": ["RB"],
            "sleeper_adp": [12.0], **{f: [100.0] for f in STAT_FIELDS},
        }
    )
    id_map = pd.DataFrame({"gsis_id": ["00-0099999"], "sleeper_id": ["s1"]})
    out = _to_canonical(
        sleeper, source=ProjectionSource.SLEEPER, id_col="sleeper_id", adp_col="sleeper_adp",
        rank_col=None, has_stats=True, season=2026, asof=date(2026, 6, 9), id_map=id_map,
    )
    for col in ("espn_auction_value_avg", "espn_auction_value_ppr", "espn_auction_value_std"):
        assert str(out[col].dtype) == "Float64"
        assert pd.isna(out[col].iloc[0])
```

- [ ] **Step 6: Run the schema tests, verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ingest/test_external_projections.py -k schema -q -n0`
Expected: the `_float64` test FAILS — `strict="filter"` drops the undeclared `espn_auction_value_*` columns, so the assertion `... in out.columns` is false (the columns are filtered out because they're not yet declared on the schema).

- [ ] **Step 7: Add the Optional columns to `ExternalProjectionSchema`**

In `src/projections/schemas.py`, in `ExternalProjectionSchema`, immediately after the `espn_draft_rank` field (line 830), add:

```python
    # Optional (not-required): ESPN-only auction values; absent on the Sleeper path and on
    # partitions written before this column existed. Float64 to avoid the NaN dtype-regression.
    espn_auction_value_avg: Series[pd.Float64Dtype] | None = pa.Field(nullable=True)
    espn_auction_value_ppr: Series[pd.Float64Dtype] | None = pa.Field(nullable=True)
    espn_auction_value_std: Series[pd.Float64Dtype] | None = pa.Field(nullable=True)
```

- [ ] **Step 8: Implement the `_to_canonical` emission (presence-guarded)**

In `src/projections/ingest/external_projections.py`, define a module-level constant near `_CANONICAL_STR_COLS` (around line 268):

```python
_ESPN_AUCTION_COLS = ("espn_auction_value_avg", "espn_auction_value_ppr", "espn_auction_value_std")
```

Then in `_to_canonical`, after the `for f in STAT_FIELDS:` loop that fills stat columns (around line 311), add:

```python
    # ESPN-only: present in the ESPN parsed frame, absent in the Sleeper one (fixed column list).
    # Guard on presence -> null_col for Sleeper, exactly mirroring espn_draft_rank's null fallback.
    for col in _ESPN_AUCTION_COLS:
        out[col] = keyed[col] if col in keyed.columns else null_col
```

And extend the dtype-cast loop (currently `for col in ("adp", "espn_draft_rank", *STAT_FIELDS):`) to include them:

```python
    for col in ("adp", "espn_draft_rank", *_ESPN_AUCTION_COLS, *STAT_FIELDS):
        out[col] = out[col].astype("Float64")
```

- [ ] **Step 9: Run the schema tests, verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ingest/test_external_projections.py -q -n0`
Expected: PASS (all, including the pre-existing parser tests — no regression).

- [ ] **Step 10: Commit**

```bash
export PATH="$(pwd)/.venv/Scripts:$PATH"
git add src/projections/ingest/external_projections.py src/projections/schemas.py tests/test_ingest/test_external_projections.py
git commit -m "feat(ingest): capture ESPN auction values (crowd + expert PPR/STD) in external_projections"
```

---

## Task 2: Carry the auction values through `build_consensus`

**Files:**
- Modify: `src/projections/consensus/blend.py`
- Modify: `src/projections/schemas.py` (`ConsensusProjectionSchema`)
- Test: `tests/test_consensus/test_blend.py`, `tests/test_schemas/test_consensus_projection_schema.py`

- [ ] **Step 1: Write the failing blend tests**

Add to `tests/test_consensus/test_blend.py` (reuse the file's existing helpers for building an `external` frame if present; otherwise this self-contained builder works):

```python
import pandas as pd
from projections.consensus.blend import build_consensus
from projections.schemas import Ruleset

_STATS = ("passing_yards", "passing_tds", "interceptions", "rushing_yards", "rushing_tds",
          "receptions", "receiving_yards", "receiving_tds", "fumbles_lost")


def _ext_row(source, gsis_id, *, adp=5.0, av_avg=pd.NA, stats=True, **extra):
    row = {
        "source": source, "source_player_id": f"{source}-{gsis_id}", "gsis_id": gsis_id,
        "is_placeholder_gsis": False, "full_name": "Player X", "position": "RB",
        "season": 2026, "asof": "2026-06-09", "adp": adp, "espn_draft_rank": pd.NA,
        "espn_auction_value_avg": av_avg, "espn_auction_value_ppr": pd.NA, "espn_auction_value_std": pd.NA,
    }
    for s in _STATS:
        row[s] = (100.0 if stats else pd.NA)
    row.update(extra)
    return row


def test_blend_carries_espn_auction_value_first_non_null():
    # ESPN row has the value, Sleeper row does not -> consensus keeps the value.
    external = pd.DataFrame([
        _ext_row("espn", "00-0011111", av_avg=58.67),
        _ext_row("sleeper", "00-0011111", av_avg=pd.NA, stats=False),
    ])
    out = build_consensus(external, Ruleset.espn_half()).set_index("gsis_id")
    assert out.loc["00-0011111", "espn_auction_value_avg"] == 58.67


def test_blend_sleeper_only_player_has_na_auction():
    external = pd.DataFrame([_ext_row("sleeper", "00-0022222", av_avg=pd.NA, stats=False)])
    out = build_consensus(external, Ruleset.espn_half()).set_index("gsis_id")
    assert pd.isna(out.loc["00-0022222", "espn_auction_value_avg"])


def test_blend_does_not_crash_when_auction_columns_absent():
    # A frame lacking the columns entirely (stale snapshot / existing tests) must not KeyError.
    external = pd.DataFrame([_ext_row("espn", "00-0033333", av_avg=58.0)]).drop(
        columns=["espn_auction_value_avg", "espn_auction_value_ppr", "espn_auction_value_std"]
    )
    out = build_consensus(external, Ruleset.espn_half()).set_index("gsis_id")
    assert pd.isna(out.loc["00-0033333", "espn_auction_value_avg"])  # seeded to NA


def test_blend_keeps_espn_value_when_sleeper_is_identity_row():
    # The distinguishing test (spec R3): ESPN row is NOT stat-bearing, so the Sleeper row becomes
    # the identity_row — but ESPN carries the auction value. First-non-null must still surface it;
    # an identity_row[col] pick would wrongly return the Sleeper row's NA.
    external = pd.DataFrame([
        _ext_row("espn", "00-0044444", av_avg=58.0, stats=False),   # ESPN: value, no stat line
        _ext_row("sleeper", "00-0044444", av_avg=pd.NA, stats=True),  # Sleeper: stat-bearing -> identity
    ])
    out = build_consensus(external, Ruleset.espn_half()).set_index("gsis_id")
    assert out.loc["00-0044444", "espn_auction_value_avg"] == 58.0


def test_blend_empty_input_carries_auction_columns():
    # _empty_output() builds from _OUTPUT_COLUMNS; the new names must be present on the empty path.
    empty = pd.DataFrame(
        columns=["source", "source_player_id", "gsis_id", "is_placeholder_gsis", "full_name",
                 "position", "season", "asof", "adp", "espn_draft_rank",
                 "espn_auction_value_avg", "espn_auction_value_ppr", "espn_auction_value_std", *_STATS]
    )
    out = build_consensus(empty, Ruleset.espn_half())
    for col in ("espn_auction_value_avg", "espn_auction_value_ppr", "espn_auction_value_std"):
        assert col in out.columns
```

- [ ] **Step 2: Run them, verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_consensus/test_blend.py -k "auction or absent or identity or empty" -q -n0`
Expected: FAIL — `KeyError: 'espn_auction_value_avg'` on the carry/identity tests (the column isn't in `_OUTPUT_COLUMNS`), the absent test KeyErrors in the loop, and the empty test fails the `col in out.columns` assertion.

- [ ] **Step 3: Implement the blend carry-through + absence guard**

In `src/projections/consensus/blend.py`:

(a) Add the three names to `_OUTPUT_COLUMNS` (the tuple at lines 28-42) — insert before `"is_placeholder_gsis"`:

```python
    "espn_auction_value_avg",
    "espn_auction_value_ppr",
    "espn_auction_value_std",
```

(`_empty_output` builds from this tuple, so it's covered automatically.)

(b) Add a module constant after `_OUTPUT_COLUMNS`:

```python
_ESPN_AUCTION_COLS = ("espn_auction_value_avg", "espn_auction_value_ppr", "espn_auction_value_std")
```

(c) Just after `external = external.reset_index(drop=True)` (line 62), seed absent columns to NA:

```python
    # ESPN-only auction columns are Optional; a stale snapshot or an existing caller's frame may
    # omit them. Seed to NA so the per-group reads below never KeyError (R3/R6/R8).
    for col in _ESPN_AUCTION_COLS:
        if col not in external.columns:
            external[col] = pd.NA
```

(d) Inside the `for gsis_id, grp in external.groupby("gsis_id", sort=False):` loop, after the `statline`/`projected` block and before `rec` is appended (i.e. add to the `rec` dict construction, after the existing fields), add a first-non-null reduction. Place this right after `rec: dict[str, object] = { ... }` is built and before the `for field in STAT_FIELDS:` loop that adds stat fields:

```python
        for col in _ESPN_AUCTION_COLS:
            non_null = grp[col].dropna()
            rec[col] = non_null.iloc[0] if not non_null.empty else pd.NA
```

(e) In the dtype block (after line 125, `df["consensus_rank"] = ...astype("Int64")`), add:

```python
    for col in _ESPN_AUCTION_COLS:
        df[col] = df[col].astype("Float64")
```

- [ ] **Step 4: Run the blend tests, verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_consensus/test_blend.py -q -n0`
Expected: PASS — including all pre-existing blend tests (the absence guard keeps them green).

- [ ] **Step 5: Write the failing consensus-schema test**

Add to `tests/test_schemas/test_consensus_projection_schema.py`:

```python
def test_consensus_schema_auction_columns_optional_and_float64():
    import pandas as pd
    from projections.schemas import ConsensusProjectionSchema
    base = {
        "gsis_id": pd.array(["00-0011111"], dtype="string[pyarrow]"),
        "season": pd.array([2026], dtype="Int64"),
        "asof": pd.array(["2026-06-09"], dtype="string[pyarrow]"),
        "full_name": pd.array(["P"], dtype="string[pyarrow]"),
        "position": pd.array(["RB"], dtype="string[pyarrow]"),
        "consensus_adp": pd.array([5.0], dtype="Float64"),
        "consensus_rank": pd.array([1], dtype="Int64"),
        "n_adp_sources": pd.array([1], dtype="Int64"),
        "has_points": [True],
        "projected_points_ppr": pd.array([200.0], dtype="Float64"),
        **{f: pd.array([pd.NA], dtype="Float64") for f in (
            "passing_yards", "passing_tds", "interceptions", "rushing_yards", "rushing_tds",
            "receptions", "receiving_yards", "receiving_tds", "fumbles_lost")},
        "is_placeholder_gsis": [False],
        "ruleset": pd.array(["ESPN_HALF"], dtype="string[pyarrow]"),
    }
    # Without the auction columns: must validate (Optional).
    ConsensusProjectionSchema.validate(pd.DataFrame(base))
    # With them: validates and stays Float64.
    withcols = pd.DataFrame({**base,
        "espn_auction_value_avg": pd.array([58.67], dtype="Float64"),
        "espn_auction_value_ppr": pd.array([57.0], dtype="Float64"),
        "espn_auction_value_std": pd.array([55.0], dtype="Float64")})
    out = ConsensusProjectionSchema.validate(withcols)
    assert str(out["espn_auction_value_avg"].dtype) == "Float64"
```

- [ ] **Step 6: Run it, verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_schemas/test_consensus_projection_schema.py -k auction -q -n0`
Expected: FAIL — `strict="filter"` drops the undeclared columns, so the `dtype == "Float64"` assertion fails (column absent).

- [ ] **Step 7: Add the Optional columns to `ConsensusProjectionSchema`**

In `src/projections/schemas.py`, in `ConsensusProjectionSchema`, immediately after `fumbles_lost` (line 879), add:

```python
    # Optional (not-required): ESPN-only auction values carried from external_projections.
    espn_auction_value_avg: Series[pd.Float64Dtype] | None = pa.Field(nullable=True)
    espn_auction_value_ppr: Series[pd.Float64Dtype] | None = pa.Field(nullable=True)
    espn_auction_value_std: Series[pd.Float64Dtype] | None = pa.Field(nullable=True)
```

- [ ] **Step 8: Run the schema test, verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_schemas/test_consensus_projection_schema.py -q -n0`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
export PATH="$(pwd)/.venv/Scripts:$PATH"
git add src/projections/consensus/blend.py src/projections/schemas.py tests/test_consensus/test_blend.py tests/test_schemas/test_consensus_projection_schema.py
git commit -m "feat(consensus): carry ESPN auction values through build_consensus (first-non-null, absence-guarded)"
```

---

## Task 3: Resolve `espn_auction_dollars` and land it on the pool

**Files:**
- Modify: `scripts/generate_preset_vorp_tables.py` (`resolve_espn_auction_dollars`, `build_preset_table`)
- Modify: `src/projections/schemas.py` (`VorpTableSchema`)
- Test: `tests/test_scripts/test_generate_preset_vorp_tables.py`

- [ ] **Step 1: Write the failing resolver tests**

Add to `tests/test_scripts/test_generate_preset_vorp_tables.py`:

```python
import pandas as pd
from projections.schemas import Ruleset
from generate_preset_vorp_tables import resolve_espn_auction_dollars  # script import; see Step 3 note


def _frame(**cols):
    return pd.DataFrame({k: pd.array(v, dtype="Float64") for k, v in cols.items()})


def test_resolve_prefers_crowd_when_positive():
    frame = _frame(espn_auction_value_avg=[58.67], espn_auction_value_ppr=[40.0], espn_auction_value_std=[30.0])
    out = resolve_espn_auction_dollars(frame, Ruleset.espn_half())
    assert out.iloc[0] == 59  # 58.67 rounded
    assert str(out.dtype) == "Int64"


def test_resolve_falls_back_to_ppr_expert_for_half_when_crowd_zero():
    frame = _frame(espn_auction_value_avg=[pd.NA], espn_auction_value_ppr=[40.0], espn_auction_value_std=[30.0])
    out = resolve_espn_auction_dollars(frame, Ruleset.espn_half())
    assert out.iloc[0] == 40


def test_resolve_uses_std_expert_for_standard():
    frame = _frame(espn_auction_value_avg=[pd.NA], espn_auction_value_ppr=[40.0], espn_auction_value_std=[30.0])
    out = resolve_espn_auction_dollars(frame, Ruleset.standard())
    assert out.iloc[0] == 30


def test_resolve_na_when_no_value():
    frame = _frame(espn_auction_value_avg=[pd.NA], espn_auction_value_ppr=[pd.NA], espn_auction_value_std=[pd.NA])
    out = resolve_espn_auction_dollars(frame, Ruleset.espn_half())
    assert pd.isna(out.iloc[0])


def test_resolve_all_na_when_columns_absent():
    frame = pd.DataFrame({"gsis_id": pd.array(["00-0011111"], dtype="string[pyarrow]")})
    out = resolve_espn_auction_dollars(frame, Ruleset.espn_half())
    assert pd.isna(out.iloc[0])
    assert str(out.dtype) == "Int64"
```

- [ ] **Step 2: Run them, verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scripts/test_generate_preset_vorp_tables.py -k resolve -q -n0`
Expected: FAIL with `ImportError: cannot import name 'resolve_espn_auction_dollars'`.

(Note: the top-of-module `from generate_preset_vorp_tables import ...` resolves because `tests/test_scripts/conftest.py` inserts `scripts/` onto `sys.path` at collection time — verified. So the import fails only on the missing symbol, exactly as the expected error states.)

- [ ] **Step 3: Implement the resolver**

In `scripts/generate_preset_vorp_tables.py`, add this import and helper above `build_preset_table`:

```python
from projections.schemas import Ruleset  # add to the existing import block

_ESPN_AUCTION_COLS = ("espn_auction_value_avg", "espn_auction_value_ppr", "espn_auction_value_std")


def resolve_espn_auction_dollars(frame: pd.DataFrame, ruleset: Ruleset) -> pd.Series:
    """Resolve one per-player ESPN auction dollar from the consensus frame: crowd average when
    present and >0, else the ruleset's expert value (PPR for ESPN_PPR/ESPN_HALF, STANDARD for
    STANDARD), else NA. Vectorized; NA-safe Int64 (rounds the fractional crowd average). An
    absent input column is treated as all-NA. `frame` must be the consensus frame — the columns
    do not survive consensus_to_season_projections onto the VORP table."""
    n = len(frame)

    def _col(name: str) -> pd.Series:
        if name in frame.columns:
            return frame[name].astype("Float64")
        return pd.Series([pd.NA] * n, dtype="Float64", index=frame.index)

    crowd = _col("espn_auction_value_avg")
    if ruleset.name in ("ESPN_PPR", "ESPN_HALF"):
        expert = _col("espn_auction_value_ppr")
    elif ruleset.name == "STANDARD":
        expert = _col("espn_auction_value_std")
    else:
        expert = pd.Series([pd.NA] * n, dtype="Float64", index=frame.index)

    use_crowd = crowd.notna() & (crowd > 0)
    value = expert.where(~use_crowd, crowd)  # crowd where use_crowd, else expert
    return value.round().astype("Int64")
```

- [ ] **Step 4: Run the resolver tests, verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scripts/test_generate_preset_vorp_tables.py -k resolve -q -n0`
Expected: PASS (5 tests).

- [ ] **Step 5: Write the failing VorpTableSchema test**

Add to `tests/test_scripts/test_generate_preset_vorp_tables.py`:

```python
def test_vorp_schema_espn_auction_dollars_optional():
    from projections.schemas import VorpTableSchema
    base = {
        "gsis_id": pd.array(["00-0011111"], dtype="string[pyarrow]"),
        "position": pd.array(["RB"], dtype="string[pyarrow]"),
        "season_mean_fpts": [200.0], "vorp": [50.0], "replacement_fpts": [150.0],
    }
    VorpTableSchema.validate(pd.DataFrame(base))  # weekly-path frame, no column -> validates
    withcol = pd.DataFrame({**base, "espn_auction_dollars": pd.array([57], dtype="Int64")})
    out = VorpTableSchema.validate(withcol)
    assert str(out["espn_auction_dollars"].dtype) == "Int64"
```

- [ ] **Step 6: Run it, verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scripts/test_generate_preset_vorp_tables.py -k optional -q -n0`
Expected: FAIL — `espn_auction_dollars` is filtered out (undeclared), so the dtype assertion fails.

- [ ] **Step 7: Add the Optional column to `VorpTableSchema`**

In `src/projections/schemas.py`, in `VorpTableSchema`, immediately after the `full_name` field (line 986), add:

```python
    # Optional (not-required): the resolved ESPN human auction value, populated only on the
    # consensus-fed preset path. Weekly-path VORP tables omit it and still validate. Slice 1
    # lands it here; Slice 2 feeds it to generate_auction_values as reference_prices.
    espn_auction_dollars: Series[pd.Int64Dtype] | None = pa.Field(ge=0, nullable=True)
```

- [ ] **Step 8: Implement the merge in `build_preset_table`**

In `scripts/generate_preset_vorp_tables.py`, in `build_preset_table`, replace the `cols`/merge block (currently `cols = consensus[["gsis_id", "consensus_adp", "full_name"]]` and the `table.merge(...)` line) with:

```python
    consensus = consensus.copy()
    consensus["espn_auction_dollars"] = resolve_espn_auction_dollars(
        consensus, preset.league_config.ruleset
    )
    cols = consensus[["gsis_id", "consensus_adp", "full_name", "espn_auction_dollars"]]
    table = table.merge(cols, on="gsis_id", how="left")
```

- [ ] **Step 9: Run the script tests, verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scripts/test_generate_preset_vorp_tables.py -q -n0`
Expected: PASS (all, including pre-existing).

- [ ] **Step 10: Commit**

```bash
export PATH="$(pwd)/.venv/Scripts:$PATH"
git add scripts/generate_preset_vorp_tables.py src/projections/schemas.py tests/test_scripts/test_generate_preset_vorp_tables.py
git commit -m "feat(auction): resolve espn_auction_dollars onto the preset VORP table (Optional)"
```

---

## Task 4: Integration + no-regression sweep

**Files:** none new — verification only.

- [ ] **Step 1: Schema-seam + ingest subset**

Run: `.venv/Scripts/python.exe -m pytest -k "ingest or store or schemas or blend or consensus" -q`
Expected: PASS (the CLAUDE.md-mandated seam gate for schema/ingest touches).

- [ ] **Step 2: No-regression on the auction subsystem**

Run: `.venv/Scripts/python.exe -m pytest tests/test_draft -q`
Expected: PASS — the bid-strategy / market / simulation / tournament / vorp tests are unchanged (R6). Any failure here means the new column leaked into a behavior path; investigate before proceeding.

- [ ] **Step 3: Full gate sweep**

Run:
```
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m mypy src tests
.venv/Scripts/python.exe -m ruff check src tests scripts
.venv/Scripts/python.exe -m ruff format --check src tests scripts
```
Expected: full suite passes except the known pre-existing `test_backtest_smoke_one_cell` (TODO #40, fails on main too); mypy/ruff clean. State explicitly which (if any) failures are the known pre-existing one.

- [ ] **Step 4: Real-data smoke (optional — requires network + populates the column)**

Per R8, the on-disk `external_projections` snapshot predates these columns (and may be absent on this machine). To populate `espn_auction_dollars` end-to-end:

```
.venv/Scripts/python.exe -m projections.ingest.external_projections --season 2026
.venv/Scripts/python.exe scripts/generate_preset_vorp_tables.py --season 2026
.venv/Scripts/python.exe -c "import pandas as pd; df=pd.read_parquet('data/vorp_2026/half_12team.parquet'); print(df[['full_name','espn_auction_dollars']].sort_values('espn_auction_dollars', ascending=False).head())"
```
Expected: the top players show real ESPN dollars (e.g. ~$58 for Gibbs). If the network is unavailable, skip — the unit tests already prove the plumbing; note the skip.

- [ ] **Step 5: Commit any artifacts / final state**

No source changes in this task. If Step 4 regenerated `data/vorp_2026/*.parquet`, leave them untracked (they are untracked artifacts, consistent with the other preset tables). Nothing to commit unless gates required a fix (commit that fix separately with a descriptive message).

---

## Notes for the implementer

- **The four new columns are Optional on purpose.** Never make them required — the weekly `generate_vorp_table` path and every pre-this-PR partition omit them and must still validate. The `test_*_optional`/`test_*_without_*` tests guard this; if one fails, you made a column required.
- **`frame` for the resolver is the `consensus` frame, never the VORP `table`** — `consensus_to_season_projections` drops the auction columns. Resolving off `table` silently yields all-NA.
- **Use `pd.Int64Dtype()` / `astype("Int64")`, never `astype("int64")`** for `espn_auction_dollars` — the column carries NA and the non-nullable cast crashes.
- **Slice 1 is inert.** `generate_auction_values`, `market.py`, `bid_strategy.py`, the engine, and the tournament are not touched. The column lands on the pool/VORP table only and is dropped at the `generate_auction_values` boundary. If you find yourself editing any of those files, you've left Slice 1's scope.
