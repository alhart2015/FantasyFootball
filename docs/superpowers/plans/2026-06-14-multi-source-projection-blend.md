# Multi-source (ESPN + Sleeper) projection blend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-06-14-multi-source-projection-blend-design.md`

**Goal:** Make Sleeper a real second projection source (parse its stat line into `STAT_FIELDS`) and blend ESPN+Sleeper in `build_consensus` behind a non-zero "stat-bearing" gate, so the draft basis is robust to ESPN's degenerate historical season projections (2023) and is a true two-source consensus where both exist.

**Architecture:** Two surgical changes — (1) `ingest/external_projections.py` parses Sleeper's raw stat line; (2) `consensus/blend.py` averages only stat-bearing source rows (≥2 non-null, non-zero `STAT_FIELDS`). No schema change, no `build_draft_basis`/backtest change. Verification re-ingests 2021–2025 and confirms healthy pools (R7) + cross-source value sanity (R8).

**Tech Stack:** Python 3.12, pandas (pyarrow/Float64 nullable dtypes), pandera schemas, pytest, mypy strict, ruff.

**Key facts the implementer must not re-derive:**
- `STAT_FIELDS` (in `projections.schemas`) is the canonical 9-tuple of stat column-name strings: `passing_yards, passing_tds, interceptions, rushing_yards, rushing_tds, receptions, receiving_yards, receiving_tds, fumbles_lost`.
- `_espn_stats_to_statline` (`external_projections.py:87`) stores RAW fractional values, **no rounding** (deliberate — see its comment). `round_count`/`COUNT_FIELDS` exist but are NOT used by the ingest stat-line path. Mirror the raw behavior for Sleeper.
- `build_consensus` (`consensus/blend.py`) already takes the **per-field mean across a player's source rows**, then scores the mean line once via `expected_points(statline, ruleset)`. The only change is restricting the rows that contribute to stat-bearing ones.
- Run the OpenMP-safe env for anything that imports numpy-heavy code on this Windows box (Task 4): PowerShell with `$env:KMP_DUPLICATE_LIB_OK="TRUE"; $env:OMP_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"`. The intermittent `-1073741819` (0xC0000005) crash is a known flake — retry.

---

## File Structure

- `src/projections/ingest/external_projections.py` — add `SLEEPER_STAT_FIELDS`, `_sleeper_stats_to_statline`; populate Sleeper stat line in `parse_sleeper_projections`; flip the Sleeper `source_specs` `has_stats` to `True`; cast numeric columns to `Float64` in `_to_canonical` (R6). (Tasks 1, 2)
- `src/projections/consensus/blend.py` — add `MIN_STAT_FIELDS`, `_is_stat_bearing`; restrict the per-field mean + `has_points` + identity-row to stat-bearing rows. (Task 3)
- `tests/test_ingest/test_external_projections.py` — Sleeper statline + parse + canonical + no-FutureWarning tests. (Tasks 1, 2)
- `tests/test_consensus/test_blend.py` — stat-bearing gate / blend / regression / ADP-unaffected tests. (Task 3)
- Verification only (Task 4): re-ingest 2021–2025, check R7 (pool health) + R8 (cross-source value sanity); fix the stale `parse_sleeper_projections` docstring; note the TODO #38 / FutureWarning closure.

---

## Task 1: Sleeper stat-line mapping (`_sleeper_stats_to_statline`)

**Files:**
- Modify: `src/projections/ingest/external_projections.py` (add `SLEEPER_STAT_FIELDS` + `_sleeper_stats_to_statline` near `_espn_stats_to_statline`, ~line 96)
- Test: `tests/test_ingest/test_external_projections.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ingest/test_external_projections.py`:

```python
def test_sleeper_stats_to_statline_maps_raw_no_rounding() -> None:
    # Real Sleeper WR line (fractional rec_td); maps to canonical fields, raw, unrounded.
    stats = {
        "rec": 105.0, "rec_yd": 1501.0, "rec_td": 8.4, "rush_yd": 44.0,
        "fum_lost": 1.0, "gp": 18.0, "cmp_pct": 0.0, "bonus_rec_wr": 105.0,
        "adp_ppr": 3.4, "rec_fd": 150.1, "rec_0_4": 21.0,
    }
    out = ext._sleeper_stats_to_statline(stats)
    assert out is not None
    assert out["receptions"] == 105.0
    assert out["receiving_yards"] == 1501.0
    assert out["receiving_tds"] == 8.4  # raw, not rounded to 8
    assert out["rushing_yards"] == 44.0
    assert out["fumbles_lost"] == 1.0
    # unmapped/absent canonical fields default to 0.0
    assert out["passing_yards"] == 0.0 and out["rushing_tds"] == 0.0
    # non-mapped Sleeper keys are ignored (no stray keys)
    from projections.schemas import STAT_FIELDS
    assert set(out) == set(STAT_FIELDS)


def test_sleeper_stats_to_statline_qb_fields() -> None:
    stats = {"pass_yd": 4193.0, "pass_td": 32.0, "pass_int": 14.0, "rush_yd": 599.0, "rush_td": 6.0}
    out = ext._sleeper_stats_to_statline(stats)
    assert out is not None
    assert out["passing_yards"] == 4193.0 and out["passing_tds"] == 32.0
    assert out["interceptions"] == 14.0 and out["rushing_yards"] == 599.0 and out["rushing_tds"] == 6.0


def test_sleeper_stats_to_statline_none_when_adp_only() -> None:
    # A Sleeper row with only ADP (no mapped stat keys) has no projection -> None.
    assert ext._sleeper_stats_to_statline({"adp_ppr": 14.5, "adp_std": 20.0, "gp": 0.0}) is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_ingest/test_external_projections.py -k sleeper_stats_to_statline -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_sleeper_stats_to_statline'`.

- [ ] **Step 3: Implement**

In `src/projections/ingest/external_projections.py`, immediately after `_espn_stats_to_statline` (after line 96), add:

```python
# Sleeper's raw projected stat keys -> canonical STAT_FIELDS. Verified live against the Sleeper
# projections API. Values are STAT_FIELDS members (same convention as ESPN_STAT_IDS).
SLEEPER_STAT_FIELDS: dict[str, str] = {
    "pass_yd": "passing_yards",
    "pass_td": "passing_tds",
    "pass_int": "interceptions",
    "rush_yd": "rushing_yards",
    "rush_td": "rushing_tds",
    "rec": "receptions",
    "rec_yd": "receiving_yards",
    "rec_td": "receiving_tds",
    "fum_lost": "fumbles_lost",
}


def _sleeper_stats_to_statline(stats: dict[str, float]) -> dict[str, float] | None:
    """Map Sleeper's raw projected stat line to the canonical STAT_FIELDS, raw (no rounding) —
    mirroring _espn_stats_to_statline. Returns None when `stats` carries none of the mapped keys
    (an ADP-only Sleeper row with no real projection), so the caller stores NA rather than a
    fabricated all-zero line. Non-mapped keys (gp, cmp_pct, *_fd, bonus_*, *_2pt, adp_*) are ignored."""
    if not any(key in stats for key in SLEEPER_STAT_FIELDS):
        return None
    out: dict[str, float] = {field: 0.0 for field in STAT_FIELDS}
    for key, field in SLEEPER_STAT_FIELDS.items():
        if key in stats:
            out[field] = float(stats[key])
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_ingest/test_external_projections.py -k sleeper_stats_to_statline -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/projections/ingest/external_projections.py tests/test_ingest/test_external_projections.py
git commit -m "feat(ingest): _sleeper_stats_to_statline maps Sleeper raw stat line to STAT_FIELDS"
```

---

## Task 2: Populate Sleeper stat line in ingest + carry it through canonical + kill FutureWarning

**Files:**
- Modify: `src/projections/ingest/external_projections.py`
  - `parse_sleeper_projections` (~lines 162-188): add the stat line to each row; guarantee `STAT_FIELDS` columns exist.
  - `source_specs` Sleeper tuple (line 351): `has_stats` `False` → `True`.
  - `_to_canonical` (~lines 254-274): cast numeric columns to `Float64` for every frame so the `pd.concat` has no all-NA dtype-inference ambiguity (R6).
- Test: `tests/test_ingest/test_external_projections.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ingest/test_external_projections.py`:

```python
def test_parse_sleeper_projections_extracts_stat_line() -> None:
    payload: list[dict[str, Any]] = [
        {
            "player_id": "6794",
            "stats": {"adp_ppr": 14.5, "rec": 105.0, "rec_yd": 1501.0, "rec_td": 8.0},
            "player": {"first_name": "A", "last_name": "B", "position": "WR"},
        }
    ]
    df = ext.parse_sleeper_projections(payload)
    r = df.iloc[0]
    assert r["sleeper_id"] == "6794" and r["sleeper_adp"] == 14.5
    assert r["receptions"] == 105.0 and r["receiving_yards"] == 1501.0 and r["receiving_tds"] == 8.0
    assert r["passing_yards"] == 0.0  # absent mapped field defaults to 0.0


def test_parse_sleeper_adp_only_row_has_na_stats_but_columns_present() -> None:
    # All-ADP-only payload: stat columns must still exist (NA) so the has_stats path can read them.
    from projections.schemas import STAT_FIELDS

    payload: list[dict[str, Any]] = [
        {
            "player_id": "9",
            "stats": {"adp_ppr": 50.0},
            "player": {"first_name": "Deep", "last_name": "Guy", "position": "RB"},
        }
    ]
    df = ext.parse_sleeper_projections(payload)
    assert all(f in df.columns for f in STAT_FIELDS)
    assert all(pd.isna(df.iloc[0][f]) for f in STAT_FIELDS)


def test_sleeper_to_canonical_carries_stat_line() -> None:
    from datetime import date

    from projections.schemas import ExternalProjectionSchema, ProjectionSource

    sl = ext.parse_sleeper_projections(
        [
            {
                "player_id": "6794",
                "stats": {"adp_ppr": 14.5, "rec": 100.0, "rec_yd": 1400.0, "rec_td": 9.0},
                "player": {"first_name": "A", "last_name": "B", "position": "WR"},
            }
        ]
    )
    out = ext._to_canonical(
        sl,
        source=ProjectionSource.SLEEPER,
        id_col="sleeper_id",
        adp_col="sleeper_adp",
        rank_col=None,
        has_stats=True,
        season=2026,
        asof=date(2026, 7, 15),
        id_map=_tiny_id_map(),
    )
    ExternalProjectionSchema.validate(out)
    r = out.iloc[0]
    assert r["source"] == "SLEEPER" and r["adp"] == 14.5
    assert r["receptions"] == 100.0 and r["receiving_yards"] == 1400.0
    assert pd.isna(r["espn_draft_rank"])  # Sleeper has no draft rank


def test_refresh_emits_no_all_na_concat_futurewarning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import warnings
    from datetime import date

    espn_payload: dict[str, Any] = {
        "players": [
            {
                "player": {
                    "id": 4374302,
                    "fullName": "Ja'Marr Chase",
                    "defaultPositionId": 3,
                    "ownership": {"averageDraftPosition": 4.8},
                    "draftRanksByRankType": {"PPR": {"rank": 20}},
                    "stats": [
                        {
                            "seasonId": 2026,
                            "statSourceId": 1,
                            "statSplitTypeId": 0,
                            "stats": {"53": 105.0, "42": 1335.0, "43": 8.0},
                        }
                    ],
                }
            }
        ]
    }
    sleeper_payload: list[dict[str, Any]] = [
        {
            "player_id": "6794",
            "stats": {"adp_ppr": 14.5, "rec": 100.0, "rec_yd": 1400.0, "rec_td": 9.0},
            "player": {"first_name": "A", "last_name": "B", "position": "WR"},
        }
    ]
    monkeypatch.setattr(ext, "fetch_espn", lambda season: espn_payload)
    monkeypatch.setattr(ext, "fetch_sleeper_season", lambda season: sleeper_payload)
    (tmp_path / "raw").mkdir(parents=True)
    pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036900", "00-0011111"], dtype="string[pyarrow]"),
            "espn_id": pd.array(["4374302", "x"], dtype="string[pyarrow]"),
            "sleeper_id": pd.array(["y", "6794"], dtype="string[pyarrow]"),
        }
    ).to_parquet(tmp_path / "raw" / "id_map.parquet", index=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        ext.refresh_external_projections(tmp_path, season=2026, asof=date(2026, 7, 15))
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_ingest/test_external_projections.py -k "sleeper_projections_extracts or adp_only_row or sleeper_to_canonical_carries or no_all_na_concat" -v`
Expected: FAIL — stat columns absent on the parsed Sleeper frame (`KeyError`/missing columns); the FutureWarning test raises `FutureWarning` (or the carry test sees NA stats).

- [ ] **Step 3: Implement**

(3a) In `parse_sleeper_projections`, replace the row-append block (current lines ~179-187) with:

```python
        stats = item.get("stats") or {}
        row: dict[str, object] = {
            "sleeper_id": str(pid),
            "full_name": full_name,
            "position": position,
            "sleeper_adp": stats.get("adp_ppr"),
        }
        statline = _sleeper_stats_to_statline(stats)
        if statline is not None:
            row.update(statline)
        rows.append(row)
```

and replace the final `return pd.DataFrame(rows)` of that function with:

```python
    df = pd.DataFrame(rows)
    # Guarantee the canonical stat columns exist even if NO row carried a projection, so the
    # has_stats=True path in _to_canonical can read them (absent -> NA).
    for field in STAT_FIELDS:
        if field not in df.columns:
            df[field] = pd.NA
    return df
```

Also update the stale docstring on `parse_sleeper_projections` — change "Sleeper has no stat line at the season level" to: "Sleeper carries a raw season stat line (mapped via `_sleeper_stats_to_statline`); ADP-only rows leave the stat columns NA."

(3b) In `source_specs` (line 351), flip the Sleeper tuple's last element to `True`:

```python
        (sleeper, ProjectionSource.SLEEPER, "sleeper_id", "sleeper_adp", None, True),
```

(3c) In `_to_canonical`, after the stat-column loop (after current line 273 `out[f] = keyed[f] if has_stats else null_col`), cast every numeric column to `Float64` so concat across sources is dtype-unambiguous (kills the all-NA FutureWarning regardless of which source lacks a column). Replace the `for f in STAT_FIELDS:` block + `return` with:

```python
    for f in STAT_FIELDS:
        out[f] = keyed[f] if has_stats else null_col
    # Uniform nullable-float dtype across all source frames so pd.concat needs no dtype inference
    # over all-NA columns (e.g. Sleeper's espn_draft_rank) — avoids the all-NA-column FutureWarning.
    for col in ("adp", "espn_draft_rank", *STAT_FIELDS):
        out[col] = out[col].astype("Float64")
    return _finish_canonical(out, season=season, asof=asof)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_ingest/test_external_projections.py -v`
Expected: PASS — all tests, including the existing `test_sleeper_to_canonical_has_null_stat_line` (it builds a Sleeper row with no stat keys → still NA via `_sleeper_stats_to_statline` returning None) and `test_refresh_writes_validated_asof_snapshot`.

Note: the existing `test_sleeper_to_canonical_has_null_stat_line` passes `has_stats=False` explicitly, so it still asserts NA stats — unaffected. If it fails because it now expects `has_stats=True` semantics, do NOT change production to suit it; the test exercises the `has_stats=False` branch which remains valid. Leave it.

- [ ] **Step 5: Commit**

```bash
git add src/projections/ingest/external_projections.py tests/test_ingest/test_external_projections.py
git commit -m "feat(ingest): carry Sleeper stat line through canonical; uniform Float64 cols (no concat FutureWarning)"
```

---

## Task 3: Stat-bearing gate in `build_consensus`

**Files:**
- Modify: `src/projections/consensus/blend.py` (add `MIN_STAT_FIELDS` + `_is_stat_bearing`; restrict aggregation to stat-bearing rows — current lines 59-70)
- Test: `tests/test_consensus/test_blend.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_consensus/test_blend.py` (the `_row` helper already accepts `stats=`):

```python
def test_two_full_sources_blend_per_field_mean() -> None:
    espn = {c: 0.0 for c in _STAT_COLS} | {"receptions": 100.0, "receiving_yards": 1400.0, "receiving_tds": 8.0}
    sleeper = {c: 0.0 for c in _STAT_COLS} | {"receptions": 110.0, "receiving_yards": 1500.0, "receiving_tds": 10.0}
    ext = _external(
        [
            _row("ESPN", "00-0036900", adp=4.0, full_name="X", position="WR", placeholder=False, stats=espn),
            _row("SLEEPER", "00-0036900", adp=3.0, full_name="X", position="WR", placeholder=False, stats=sleeper),
        ]
    )
    r = build_consensus(ext, Ruleset()).iloc[0]
    assert r["receptions"] == 105.0  # mean(100,110)
    assert r["receiving_yards"] == 1450.0
    assert r["receiving_tds"] == 9.0
    assert bool(r["has_points"]) is True
    # half-PPR: 105 + 1450/10 + 9*6 = 105 + 145 + 54 = 304
    assert round(float(r["projected_points_ppr"]), 1) == 304.0


def test_stub_row_excluded_from_blend() -> None:
    # ESPN "stub": an all-zero stat line (the 2023 degenerate case). Sleeper has a real line.
    stub = {c: 0.0 for c in _STAT_COLS}
    sleeper = {c: 0.0 for c in _STAT_COLS} | {"receptions": 110.0, "receiving_yards": 1500.0, "receiving_tds": 10.0}
    ext = _external(
        [
            _row("ESPN", "00-0036900", adp=4.0, full_name="X", position="WR", placeholder=False, stats=stub),
            _row("SLEEPER", "00-0036900", adp=3.0, full_name="X", position="WR", placeholder=False, stats=sleeper),
        ]
    )
    r = build_consensus(ext, Ruleset()).iloc[0]
    # Sleeper-only, NOT mean(0,110)=55:
    assert r["receptions"] == 110.0 and r["receiving_yards"] == 1500.0 and r["receiving_tds"] == 10.0


def test_all_zero_rows_yield_no_points() -> None:
    stub = {c: 0.0 for c in _STAT_COLS}
    ext = _external(
        [
            _row("ESPN", "00-0036900", adp=4.0, full_name="X", position="WR", placeholder=False, stats=stub),
            _row("SLEEPER", "00-0036900", adp=3.0, full_name="X", position="WR", placeholder=False, stats=stub),
        ]
    )
    r = build_consensus(ext, Ruleset()).iloc[0]
    assert bool(r["has_points"]) is False
    assert pd.isna(r["projected_points_ppr"])


def test_one_nonzero_field_is_not_stat_bearing() -> None:
    # A single non-zero field (the rare 2023 scoring stub) is below MIN_STAT_FIELDS=2 -> excluded.
    one = {c: 0.0 for c in _STAT_COLS} | {"rushing_yards": 50.0}
    full = {c: 0.0 for c in _STAT_COLS} | {"receptions": 80.0, "receiving_yards": 900.0, "receiving_tds": 5.0}
    ext = _external(
        [
            _row("ESPN", "00-0036900", adp=4.0, full_name="X", position="WR", placeholder=False, stats=one),
            _row("SLEEPER", "00-0036900", adp=3.0, full_name="X", position="WR", placeholder=False, stats=full),
        ]
    )
    r = build_consensus(ext, Ruleset()).iloc[0]
    assert r["rushing_yards"] == 0.0  # the 1-field stub did NOT contribute
    assert r["receptions"] == 80.0


def test_adp_unaffected_by_stat_gating() -> None:
    # consensus_adp/rank come from ADP regardless of whether the stat line is gated out.
    stub = {c: 0.0 for c in _STAT_COLS}
    ext = _external(
        [
            _row("ESPN", "00-0036900", adp=4.0, full_name="X", position="WR", placeholder=False, stats=stub),
            _row("SLEEPER", "00-0036900", adp=2.0, full_name="X", position="WR", placeholder=False),
        ]
    )
    r = build_consensus(ext, Ruleset()).iloc[0]
    assert r["consensus_adp"] == 3.0  # mean(4,2), unaffected by stat gating
    assert r["n_adp_sources"] == 2
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_consensus/test_blend.py -k "two_full_sources or stub_row_excluded or all_zero_rows or one_nonzero_field" -v`
Expected: FAIL — current code averages all non-null rows, so `test_stub_row_excluded` sees `receptions == 55.0` and `test_all_zero_rows` sees `has_points == True`.

- [ ] **Step 3: Implement**

In `src/projections/consensus/blend.py`, add after the imports / before `_OUTPUT_COLUMNS` (after line 14):

```python
MIN_STAT_FIELDS = 2  # a source row must have >= this many non-null, non-zero STAT_FIELDS to count


def _is_stat_bearing(row: pd.Series) -> bool:
    """True iff `row` has >= MIN_STAT_FIELDS STAT_FIELDS values that are non-null AND non-zero.
    Both source statline constructors zero-fill absent fields, so a degenerate stub arrives as an
    all-0.0 (fully non-null) row; testing non-zero is what keeps it out of the blend."""
    vals = pd.to_numeric(row[list(STAT_FIELDS)], errors="coerce").fillna(0.0)
    return int((vals != 0).sum()) >= MIN_STAT_FIELDS
```

Then in `build_consensus`, replace the identity/statline block (current lines 59-70):

```python
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
```

with:

```python
        # Only "stat-bearing" source rows (>= MIN_STAT_FIELDS non-null, non-zero STAT_FIELDS)
        # contribute to the per-field mean — this keeps degenerate all-zero stubs (e.g. ESPN's
        # 2023 historical season projection) out of the blend.
        bearing = grp[grp.apply(_is_stat_bearing, axis=1)]

        # Prefer a stat-bearing row for identity (full_name/position); fall back to the first row.
        identity_row = bearing.iloc[0] if not bearing.empty else grp.iloc[0]

        statline: dict[str, float] = {}
        has_points = False
        for field in STAT_FIELDS:
            vals = bearing[field].dropna()
            if not vals.empty:
                statline[field] = float(vals.mean())
                has_points = True
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_consensus/test_blend.py -v`
Expected: PASS — new tests plus all existing tests. In particular `test_two_source_veteran_blends_adp_and_scores_points` still yields `322.1` (its Sleeper row has no stats → not stat-bearing → ESPN-only, unchanged = R4 regression), and `test_player_with_points_but_no_adp_gets_null_rank` still has `has_points=True` (its ESPN row is all-zero across `_STAT_COLS` → **now `has_points=False`**).

> **Watch-out (existing-test behavior change):** `test_player_with_points_but_no_adp_gets_null_rank` (line 148) and `test_only_nonpositive_adp_yields_null_adp_and_rank` (line 229) and `test_union_coverage_and_deterministic_rank` (line 109) build ESPN rows with `stats={c: 0.0 for c in _STAT_COLS}` (all-zero). Under the new gate those rows are NOT stat-bearing, so their `has_points` flips from `True` to `False`. Two of these assert `has_points is True`. This is the **intended** new semantics (all-zero ≠ a real projection), so update those assertions to `has_points is False` and `pd.isna(projected_points_ppr)`. The spec explicitly defines all-zero as not-stat-bearing (R3); these fixtures used all-zero as a stand-in for "has a stat line" which no longer holds. State this reason in the commit. The rank/adp/union assertions in those tests are unaffected (ADP gating is independent) and must continue to pass.

Specifically edit:
- `test_player_with_points_but_no_adp_gets_null_rank`: change `assert bool(r["has_points"]) is True` → `assert bool(r["has_points"]) is False`.
- `test_only_nonpositive_adp_yields_null_adp_and_rank`: change `assert bool(r["has_points"]) is True  # still appears (union coverage)` → `assert bool(r["has_points"]) is False  # all-zero stat line is not a projection; still appears via ADP union`. (Union coverage still holds — the row is still present; only `has_points` changes.)
- `test_union_coverage_and_deterministic_rank`: no `has_points` assertion — leave as is (verify it still passes).

- [ ] **Step 5: Commit**

```bash
git add src/projections/consensus/blend.py tests/test_consensus/test_blend.py
git commit -m "feat(consensus): blend only stat-bearing rows (>=2 non-zero fields); excludes degenerate stubs"
```

---

## Task 4: Verification — re-ingest 2021–2025 + R7 pool health + R8 cross-source sanity + gates

**Files:**
- Modify (docs only): `TODO.md` (note #38's "real multi-source points consensus" slice + the `pd.concat` FutureWarning item are addressed by this branch).
- No new production code. This task RE-INGESTS real data and runs verification. **Run by the controller** (network + the OpenMP-safe PowerShell crash-retry pattern), not a code subagent.

- [ ] **Step 1: Full project gate suite (on the code from Tasks 1–3)**

Run:
```
pytest -v -k "ingest or store or schemas or consensus or blend"
mypy src tests
ruff check src tests
ruff format --check src tests
```
Expected: all green. (CLAUDE.md requires the `ingest or store or schemas` subset for any ingest/schema-touching change; `consensus`/`blend` cover the gate change.)

- [ ] **Step 2: Re-ingest the blended snapshots for 2021–2025**

In PowerShell with the OpenMP-safe env (retry on `-1073741819`):
```
$env:KMP_DUPLICATE_LIB_OK="TRUE"; $env:OMP_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"
foreach ($yr in 2021,2022,2023,2024,2025) {
  python -m projections.ingest.external_projections --season $yr --data-root data
}
```
Expected: each prints `Wrote external-projection snapshot: data\raw\external_projections\season=<yr>\asof=<today>\part.parquet`. (`load_inputs`/`build_draft_basis` read `read_latest_partition`, so the new asof is picked up automatically.)

- [ ] **Step 3: Verify R7 (pool health ≥90% for all five seasons)**

Create `_verify_blend_pool.py` (throwaway, repo root):
```python
from pathlib import Path

from projections.draft.backtest.draft_basis import build_draft_basis
from projections.draft.league_config import LeagueConfig
from projections.schemas import ExternalProjectionSchema
from projections.store import read_latest_partition

cfg = LeagueConfig.model_validate_json(Path("_league_16_half.json").read_text())
ok = True
for yr in (2021, 2022, 2023, 2024, 2025):
    ext = ExternalProjectionSchema.validate(
        read_latest_partition(Path("data/raw"), "external_projections", season=yr)
    )
    pool = build_draft_basis(ext, league_config=cfg)
    frac = (pool["season_mean_fpts"].to_numpy() > 0).mean()
    status = "OK" if frac >= 0.90 else "FAIL"
    if frac < 0.90:
        ok = False
    print(f"{yr}: pool={len(pool)} season_mean_fpts>0 = {frac:.1%}  [{status}]")
print("R7", "PASS" if ok else "FAIL")
```
Run (PowerShell, OpenMP-safe, retry): `python -u _verify_blend_pool.py`
Expected: every season ≥90% (2023 jumps from 99/514≈19% to ~95%); final line `R7 PASS`.

- [ ] **Step 4: Verify R8 (cross-source value sanity on 2024/2025)**

Create `_verify_blend_xsource.py` (throwaway, repo root):
```python
from pathlib import Path

import numpy as np

from projections.consensus.blend import build_consensus
from projections.draft.league_config import LeagueConfig
from projections.schemas import ExternalProjectionSchema
from projections.store import read_latest_partition

cfg = LeagueConfig.model_validate_json(Path("_league_16_half.json").read_text())
ok = True
for yr in (2024, 2025):
    ext = ExternalProjectionSchema.validate(
        read_latest_partition(Path("data/raw"), "external_projections", season=yr)
    )
    espn_only = build_consensus(ext[ext["source"] == "ESPN"], cfg.ruleset).set_index("gsis_id")
    sleeper_only = build_consensus(ext[ext["source"] == "SLEEPER"], cfg.ruleset).set_index("gsis_id")
    j = espn_only[["projected_points_ppr"]].join(
        sleeper_only[["projected_points_ppr"]], how="inner", lsuffix="_e", rsuffix="_s"
    ).dropna()
    e = j["projected_points_ppr_e"].to_numpy(dtype=float)
    s = j["projected_points_ppr_s"].to_numpy(dtype=float)
    m = (e > 0) & (s > 0)
    r = float(np.corrcoef(e[m], s[m])[0, 1])
    ratio = float(np.median(s[m] / e[m]))
    good = r >= 0.85 and 0.85 <= ratio <= 1.15
    ok = ok and good
    print(f"{yr}: n={m.sum()} corr={r:.3f} median_ratio(Sleeper/ESPN)={ratio:.3f}  [{'OK' if good else 'FAIL'}]")
print("R8", "PASS" if ok else "FAIL")
```
Run (PowerShell, OpenMP-safe, retry): `python -u _verify_blend_xsource.py`
Expected: both seasons `corr ≥ 0.85`, `median_ratio ∈ [0.85,1.15]`; final line `R8 PASS`. (If R8 FAILs, the Sleeper mapping is wrong/incomplete — do NOT proceed; debug the mapping, not the threshold.)

- [ ] **Step 5: Update TODO.md and commit**

Edit `TODO.md` item #38 "Remaining scope" bullets: mark the **`pd.concat` FutureWarning** sub-item (currently the `external_projections.py` all-NA bullet) and the **"real multi-source points consensus"** scope as addressed by branch `feat/multi-source-projection-blend` (ESPN+Sleeper stat-line blend shipped; distribution-wrapping still deferred). Keep it factual; don't overclaim (single points-magnitude consensus, not distribution spread).

```bash
git add TODO.md
git commit -m "docs(todo): multi-source ESPN+Sleeper blend addresses #38 points-consensus + concat FutureWarning"
```

Note: the throwaway `_verify_blend_*.py` scripts and re-ingested snapshots are not committed (snapshots are data; verifiers are scratch).

---

## Self-Review

**1. Spec coverage:**
- R1 (Sleeper statline raw, None when adp-only) → Task 1 (`_sleeper_stats_to_statline` + 3 tests) + Task 2 (parse wiring).
- R2 (Sleeper rows carry real stats, schema-valid) → Task 2 (`test_sleeper_to_canonical_carries_stat_line`).
- R3 (gate: ≥2 non-null non-zero; the four blend cases) → Task 3 (`two_full_sources`, `stub_row_excluded`, `all_zero_rows`, `one_nonzero_field`, regression note).
- R4 (ESPN-only regression byte-identical) → Task 3 (existing `test_two_source_veteran_blends_adp_and_scores_points` stays 322.1).
- R5 (ADP unaffected) → Task 3 (`test_adp_unaffected_by_stat_gating`).
- R6 (no all-NA concat FutureWarning) → Task 2 (`test_refresh_emits_no_all_na_concat_futurewarning` + Float64 cast).
- R7 (pool ≥90% all five seasons) → Task 4 Step 3.
- R8 (cross-source value sanity) → Task 4 Step 4.
- Gates (pytest/mypy/ruff) → Task 4 Step 1.

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to". Every code step shows full code; every command states expected output.

**3. Type consistency:** `_sleeper_stats_to_statline(stats) -> dict[str, float] | None` used identically in Task 1 (def) and Task 2 (call). `_is_stat_bearing(row) -> bool` + `MIN_STAT_FIELDS` consistent in Task 3. `SLEEPER_STAT_FIELDS` values are `STAT_FIELDS` members. `has_stats=True` flip matches the `_to_canonical(has_stats=...)` signature. Verifier scripts use real APIs (`build_draft_basis`, `build_consensus`, `read_latest_partition`, `LeagueConfig.ruleset`) confirmed present in the codebase.
