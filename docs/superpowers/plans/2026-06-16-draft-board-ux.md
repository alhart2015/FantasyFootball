# Draft Board UX (name fix + click-to-pick + best-available filter) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-06-16-draft-board-ux-design.md`

**Goal:** Surface real names for every pool player (incl. placeholder-gsis rookies), replace the top search box with click-to-confirm picking from two panes, and give best-available a position dropdown + search.

**Architecture:** Pure UI + name-data plumbing over the existing `LiveDraftSession` controller. `full_name` is carried from `consensus_projections` into the VORP table; `LiveDraftSession.player_names` overlays pool names on id_map; a new `available_for_pick` powers the searchable/filterable picker pane; the Streamlit board records picks through a single `pending_pick` → `Confirm pick` flow. No draft/strategy/engine math changes.

**Tech Stack:** Python 3.12, pandas, pandera (`DataFrameModel`, `strict="filter"`, `coerce=True`), Streamlit 1.58 (`st.dataframe(selection_mode="single-row", on_select=<callable>)`), pytest, mypy (strict), ruff.

**Conventions for every task (from CLAUDE.md):**
- Reference enums (`Position.RB`), never bare strings, in new code.
- `df = SCHEMA.validate(df)` with reassignment at module boundaries.
- `pd.StringDtype("pyarrow")` (`_PYARROW_STR`) for nullable string columns.
- End each commit message with the trailer:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- Do **not** `git add .` — stage only the files each step names (an uncommitted `.gitignore` edit and untracked data artifacts must stay unstaged).
- Run commands from the repo root `C:\Users\HartAlden\FantasyFootball`.

**Final gates (run after the last task; CLAUDE.md directive #4):**
- `pytest -v`
- `mypy src tests`
- `ruff check src tests`
- `ruff format --check src tests`
- Schema-seam guard: `pytest -v -k "ingest or store or schemas"`

---

## Phase 1 — Name fix

### Task 1: `VorpTableSchema` gains Optional `full_name`

**Files:**
- Modify: `src/projections/schemas.py` (class `VorpTableSchema`, ~line 975-986)
- Test: `tests/test_schemas/test_dataframe_schemas.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_schemas/test_dataframe_schemas.py` (next to `test_vorp_table_schema_accepts_optional_consensus_adp`, ~line 1097):

```python
def test_vorp_table_schema_accepts_optional_full_name() -> None:
    """Consensus-fed VORP tables carry full_name; weekly-fed ones do not.
    The column is Optional + nullable so both validate."""
    base = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-1000001", "00-2000001"], dtype=_PYARROW_STR),
            "position": pd.array([Position.QB.value, Position.RB.value], dtype=_PYARROW_STR),
            "season_mean_fpts": pd.array([320.0, 260.0], dtype="float64"),
            "vorp": pd.array([80.0, 30.0], dtype="float64"),
            "replacement_fpts": pd.array([240.0, 230.0], dtype="float64"),
        }
    )
    # Without full_name (weekly path) -> still validates.
    VorpTableSchema.validate(base)

    # With full_name (consensus path), including a null -> validates and the column survives.
    with_names = base.copy()
    with_names["full_name"] = pd.array(["Patrick Mahomes", pd.NA], dtype=_PYARROW_STR)
    validated = VorpTableSchema.validate(with_names)
    assert "full_name" in validated.columns
    assert validated["full_name"].iloc[0] == "Patrick Mahomes"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas/test_dataframe_schemas.py::test_vorp_table_schema_accepts_optional_full_name -v`
Expected: FAIL — `strict="filter"` drops the unknown `full_name` column, so `assert "full_name" in validated.columns` fails.

- [ ] **Step 3: Add the field to `VorpTableSchema`**

In `src/projections/schemas.py`, in `class VorpTableSchema`, immediately after the `consensus_adp` field (line ~982), add:

```python
    # Optional (not-required): populated only on the consensus-fed path (the player's
    # display name, incl. placeholder-gsis rookies absent from id_map). Weekly-path VORP
    # tables omit it and still validate. Nullable: a player with no consensus name is NA.
    full_name: Series[str] | None = pa.Field(nullable=True)
```

(`Series[str]` matches the codebase's existing `full_name` declarations — e.g. `ConsensusProjectionSchema.full_name` — and accepts `_PYARROW_STR` input under `coerce=True`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas/test_dataframe_schemas.py -k "vorp_table" -v`
Expected: PASS (both the new test and `test_vorp_table_schema_round_trip` / `..._accepts_optional_consensus_adp`).

- [ ] **Step 5: Commit**

```bash
git add src/projections/schemas.py tests/test_schemas/test_dataframe_schemas.py
git commit -m "feat(schemas): VorpTableSchema gains Optional nullable full_name"
```

---

### Task 2: Generator carries `full_name` on the consensus path

**Files:**
- Modify: `scripts/generate_vorp_table.py` (the consensus merge in `main()`, lines 179-183)
- Test: `tests/test_scripts/test_generate_vorp_table_cli.py`

- [ ] **Step 1: Write the failing unit test for the merge helper**

Add to `tests/test_scripts/test_generate_vorp_table_cli.py` (it already imports `pd`, `pytest`, `_PYARROW_STR`, `Position`):

```python
def test_merge_consensus_columns_carries_full_name() -> None:
    """The consensus merge attaches consensus_adp AND full_name onto the VORP table
    (cheap unit check — no store round-trip)."""
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "scripts"))
    from generate_vorp_table import _merge_consensus_columns

    out_df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-1000001", "00-2000001"], dtype=_PYARROW_STR),
            "position": pd.array([Position.QB.value, Position.RB.value], dtype=_PYARROW_STR),
            "season_mean_fpts": [320.0, 260.0],
            "vorp": [80.0, 30.0],
            "replacement_fpts": [240.0, 230.0],
        }
    )
    consensus = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-1000001", "00-2000001"], dtype=_PYARROW_STR),
            "consensus_adp": pd.array([2.0, 14.0], dtype=pd.Float64Dtype()),
            "full_name": pd.array(["Patrick Mahomes", "Bijan Robinson"], dtype=_PYARROW_STR),
        }
    )
    merged = _merge_consensus_columns(out_df, consensus)
    assert "full_name" in merged.columns
    assert dict(zip(merged["gsis_id"], merged["full_name"], strict=False)) == {
        "00-1000001": "Patrick Mahomes",
        "00-2000001": "Bijan Robinson",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scripts/test_generate_vorp_table_cli.py::test_merge_consensus_columns_carries_full_name -v`
Expected: FAIL with `ImportError: cannot import name '_merge_consensus_columns'`.

- [ ] **Step 3: Extract the merge helper and include `full_name`**

In `scripts/generate_vorp_table.py`, add this helper just above `def main()` (line ~141):

```python
def _merge_consensus_columns(out_df: pd.DataFrame, consensus: pd.DataFrame) -> pd.DataFrame:
    """Attach the consensus market columns (consensus_adp, full_name) onto the VORP table.

    full_name is the player's display name (incl. placeholder-gsis rookies the live board
    must name). Returns a re-validated VorpTableSchema frame.
    """
    cols = consensus[["gsis_id", "consensus_adp", "full_name"]]
    merged = out_df.merge(cols, on="gsis_id", how="left")
    merged["gsis_id"] = merged["gsis_id"].astype(_PYARROW_STR)
    return VorpTableSchema.validate(merged)
```

Then replace the inline merge block in `main()` (lines 179-183):

```python
    if consensus is not None:
        adp = consensus[["gsis_id", "consensus_adp"]]
        out_df = out_df.merge(adp, on="gsis_id", how="left")
        out_df["gsis_id"] = out_df["gsis_id"].astype(_PYARROW_STR)
        out_df = VorpTableSchema.validate(out_df)
```

with:

```python
    if consensus is not None:
        out_df = _merge_consensus_columns(out_df, consensus)
```

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `pytest tests/test_scripts/test_generate_vorp_table_cli.py::test_merge_consensus_columns_carries_full_name -v`
Expected: PASS.

- [ ] **Step 5: Extend the E2E consensus round-trip to assert names**

In `tests/test_scripts/test_generate_vorp_table_cli.py::test_cli_consensus_mode_round_trip`, after the existing `assert out["consensus_adp"].notna().all()` (line ~342), add:

```python
    # full_name is carried from the consensus snapshot (in-pool players all have one).
    assert "full_name" in out.columns
    assert out["full_name"].notna().all()
    # A known QB maps to its consensus full_name ("QB Player 0").
    assert dict(zip(out["gsis_id"], out["full_name"], strict=False))["00-1000000"] == "QB Player 0"
```

- [ ] **Step 6: Run the consensus E2E test to verify it passes**

Run: `pytest tests/test_scripts/test_generate_vorp_table_cli.py -k "consensus" -v`
Expected: PASS (the round-trip now also asserts `full_name`). The weekly round-trip (`test_cli_parquet_round_trip`) stays green (weekly path never touches the helper, so no `full_name`).

- [ ] **Step 7: Commit**

```bash
git add scripts/generate_vorp_table.py tests/test_scripts/test_generate_vorp_table_cli.py
git commit -m "feat(draft): generator carries full_name on the consensus VORP path"
```

---

### Task 3: `player_names` overlays pool names; `record_pick` guard uses id_map directly

**Files:**
- Modify: `src/projections/draft/assistant/live.py` (`player_names` ~148-152; add `_id_map_ids`; `record_pick` guard line 207)
- Test: `tests/test_draft/test_assistant_live.py`

- [ ] **Step 1: Add a names-fixture helper + write the failing player_names tests**

In `tests/test_draft/test_assistant_live.py`, add a pool-with-names helper after `_pool()` (line ~64):

```python
def _pool_with_names() -> pd.DataFrame:
    """The shared pool plus a full_name column whose values differ from id_map's
    ('Pool i' vs the id_map 'Pi') so 'pool name wins' is observable."""
    pool = _pool()
    pool["full_name"] = pd.array([f"Pool {i}" for i in range(1, 40)], dtype=_PYARROW_STR)
    return pool


def _rookie_row() -> pd.DataFrame:
    """A placeholder-gsis rookie that lives in the VORP pool (with a name) but NOT in id_map."""
    return pd.DataFrame(
        {
            "gsis_id": pd.array(["99-8467088"], dtype=_PYARROW_STR),
            "position": pd.array(["RB"], dtype=_PYARROW_STR),
            "season_mean_fpts": [400.0],
            "vorp": [200.0],
            "replacement_fpts": [100.0],
            "consensus_adp": pd.array([0.5], dtype=pd.Float64Dtype()),  # lowest -> picked first
            "full_name": pd.array(["Rookie RB"], dtype=_PYARROW_STR),
        }
    )
```

Then add the resolution tests:

```python
def test_player_names_prefers_pool_full_name_for_off_id_map_rookie() -> None:
    pool = pd.concat([_pool_with_names(), _rookie_row()], ignore_index=True)
    s = _session(pool=pool)
    # Rookie absent from id_map resolves to its pool name (was "—" before the fix).
    assert s.name("99-8467088") == "Rookie RB"
    # Pool name wins over a differing id_map name ("Pool 1" beats id_map "P1").
    assert s.name("00-0000001") == "Pool 1"


def test_player_names_falls_back_to_id_map_when_pool_name_null() -> None:
    pool = _pool_with_names()
    pool.loc[0, "full_name"] = pd.NA  # gsis 00-0000001 has no pool name
    s = _session(pool=pool)
    assert s.name("00-0000001") == "P1"  # id_map fallback


def test_player_names_weekly_pool_without_full_name_uses_id_map() -> None:
    s = _session(pool=_pool())  # _pool() has no full_name column
    assert s.name("00-0000001") == "P1"
```

- [ ] **Step 2: Run the player_names tests to verify they fail**

Run: `pytest tests/test_draft/test_assistant_live.py -k "player_names" -v`
Expected: FAIL — `test_player_names_prefers_pool_full_name_for_off_id_map_rookie` fails (current `player_names` is id_map-only → `name("99-8467088") == "—"` and `name("00-0000001") == "P1"`).

- [ ] **Step 3: Broaden `player_names` and add `_id_map_ids`**

In `src/projections/draft/assistant/live.py`, replace the `player_names` cached_property (lines 148-152) with:

```python
    @cached_property
    def player_names(self) -> dict[str, str]:
        """gsis_id -> full_name: id_map names overlaid with the pool's own full_name.

        The consensus VORP pool carries full_name for players absent from id_map
        (placeholder-gsis rookies); those names win so every drafted/available player
        resolves. A pool player with a null name falls back to id_map, then '—'.
        """
        names: dict[str, str] = dict(
            zip(self.id_map["gsis_id"], self.id_map["full_name"], strict=False)
        )
        if "full_name" in self.pool.columns:
            for gid, nm in zip(self.pool["gsis_id"], self.pool["full_name"], strict=False):
                if pd.notna(nm):
                    names[str(gid)] = str(nm)
        return names

    @cached_property
    def _id_map_ids(self) -> frozenset[str]:
        """gsis_ids present in id_map — the players whose position build_draft_state can
        resolve. record_pick's my-pick guard checks this directly (player_names is no
        longer id_map-only, so it can't stand in for id_map membership)."""
        return frozenset(str(g) for g in self.id_map["gsis_id"])
```

- [ ] **Step 4: Write the failing guard-regression tests**

In `tests/test_draft/test_assistant_live.py`, add:

```python
def test_record_pick_off_id_map_rookie_rejected_for_my_pick_even_when_named() -> None:
    # The rookie now HAS a name in player_names (pool full_name), so the old guard
    # (gid not in player_names) would wrongly allow it. The id_map guard must still reject
    # it for my pick and NOT append (else the next state() would raise and poison the session).
    pool = pd.concat([_pool_with_names(), _rookie_row()], ignore_index=True)
    s = _session(picks=[f"00-000{i:04d}" for i in range(1, 7)], pool=pool)  # pick 7 is mine
    assert s.is_my_pick
    assert "99-8467088" in s.player_names  # broadened map contains the rookie
    with pytest.raises(ValueError, match="id_map"):
        s.record_pick("99-8467088")
    assert "99-8467088" not in s.picks  # not appended


def test_record_pick_off_id_map_rookie_allowed_for_opponent_when_named() -> None:
    pool = pd.concat([_pool_with_names(), _rookie_row()], ignore_index=True)
    s = _session(pool=pool)  # pick 1 is an opponent's
    assert not s.is_my_pick
    s.record_pick("99-8467088")
    assert "99-8467088" in s.picks
    s.state()  # must not raise (opponent picks need no id_map position)
```

- [ ] **Step 5: Run the guard tests to verify they fail**

Run: `pytest tests/test_draft/test_assistant_live.py -k "off_id_map_rookie" -v`
Expected: FAIL — `..._rejected_for_my_pick_even_when_named` fails: with the broadened `player_names` the current guard (`gid not in self.player_names`) no longer fires, so `record_pick` wrongly appends and raises no error.

- [ ] **Step 6: Switch the guard to `_id_map_ids`**

In `src/projections/draft/assistant/live.py`, in `record_pick` (line 207), change:

```python
        if self.on_clock_slot == self.my_slot and gid not in self.player_names:
```

to:

```python
        if self.on_clock_slot == self.my_slot and gid not in self._id_map_ids:
```

(Leave the surrounding comment block; it already explains the id_map dependency.)

- [ ] **Step 7: Run the live tests to verify they pass**

Run: `pytest tests/test_draft/test_assistant_live.py -v`
Expected: PASS — the new player_names + guard tests pass, and the existing
`test_record_pick_rejects_absent_from_id_map_for_my_pick`,
`test_record_pick_allows_off_id_map_opponent_pick`,
`test_attach_names_inserts_full_name` (still "P1"; `_pool()` has no full_name),
and `test_my_roster_view_assigns_slots_and_open_needs` ("P9") stay green.

- [ ] **Step 8: Commit**

```bash
git add src/projections/draft/assistant/live.py tests/test_draft/test_assistant_live.py
git commit -m "feat(draft): player_names overlays pool names; record_pick guard checks id_map directly"
```

---

### Task 4: CLI `format_table` prefers the pool's `full_name`

**Files:**
- Modify: `src/projections/draft/assistant/cli.py` (`generate_recommendation` ~79; `format_table` 82-99)
- Test: `tests/test_draft/test_assistant_cli.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_draft/test_assistant_cli.py`:

```python
def test_cli_prefers_pool_full_name_over_id_map(tmp_path: Path) -> None:
    """A placeholder-gsis rookie absent from id_map but named in the VORP pool prints its
    pool name; an in-id_map player still prints (id_map fallback when the pool has no name)."""
    state_path, _vorp_path, id_path = _setup(tmp_path)
    # Rewrite the vorp pool: keep the two id_map players, add a rookie absent from id_map,
    # and give the pool its own full_name column.
    vorp = pd.DataFrame(
        {
            "gsis_id": pd.array(
                ["00-0000010", "00-0000020", "99-8467088"], dtype=_PYARROW_STR
            ),
            "position": pd.array(["RB", "WR", "RB"], dtype=_PYARROW_STR),
            "season_mean_fpts": [250.0, 252.0, 260.0],
            "vorp": [50.0, 52.0, 60.0],
            "replacement_fpts": [200.0, 200.0, 200.0],
            "consensus_adp": pd.array([5.0, 7.0, 3.0], dtype=pd.Float64Dtype()),
            "full_name": pd.array(["RB One", "WR One", "Rookie RB"], dtype=_PYARROW_STR),
        }
    )
    vorp_path = tmp_path / "vorp_named.parquet"
    vorp.to_parquet(vorp_path, index=False)

    code = run(
        [
            "--state", str(state_path),
            "--vorp-table", str(vorp_path),
            "--id-map", str(id_path),
            "--strategy", "raw_vorp",
            "--top", "5",
        ]
    )
    assert code == 0
    out = capsys_out()
    assert "Rookie RB" in out  # named from the pool (absent from id_map)
```

Then add this tiny capture helper near the top of the file (after the imports) so the test can read stdout without threading `capsys` through `_setup`:

```python
def capsys_out() -> str:  # pragma: no cover - test scaffolding
    raise AssertionError("replaced per-test by capsys")
```

Replace the test body's `out = capsys_out()` line with the standard fixture instead — i.e. give the test the `capsys` parameter and use it directly:

```python
def test_cli_prefers_pool_full_name_over_id_map(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
```

and `out = capsys.readouterr().out`. (Do NOT add the `capsys_out` stub — it was only illustrative; use the `capsys` fixture exactly like `test_run_prints_table`.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_draft/test_assistant_cli.py::test_cli_prefers_pool_full_name_over_id_map -v`
Expected: FAIL — `format_table` resolves names only from id_map, so the rookie prints as `-`, not "Rookie RB".

- [ ] **Step 3: Enrich `rec` with the pool name and prefer it in `format_table`**

In `src/projections/draft/assistant/cli.py`, change the last line of `generate_recommendation` (line 79) from:

```python
    return strategy.recommend(state, vorp, league), id_map
```

to:

```python
    rec = strategy.recommend(state, vorp, league)
    if "full_name" in vorp.columns:
        # Carry the pool's display name (incl. placeholder-gsis rookies absent from id_map)
        # so format_table can prefer it over id_map — consistent with the live board.
        rec = rec.merge(vorp[["gsis_id", "full_name"]], on="gsis_id", how="left")
    return rec, id_map
```

Then update `format_table` (lines 82-99) to prefer `rec.full_name` when present:

```python
def format_table(rec: pd.DataFrame, id_map: pd.DataFrame, top: int) -> str:
    """Render the top-N recommendation as a fixed-width text table.

    Display name prefers the pool's full_name (carried on rec by generate_recommendation,
    incl. placeholder-gsis rookies) and falls back to id_map, then '-'.
    """
    names = dict(zip(id_map["gsis_id"], id_map["full_name"], strict=False))
    has_pool_name = "full_name" in rec.columns
    header = (
        f"{'#':>3}  {'PLAYER':<24} {'POS':<4} {'VORP':>7} {'ADP':>6} {'P(next)':>8} {'SCORE':>8}"
    )
    lines = [header]
    for row in rec.head(top).itertuples(index=False):
        pool_name = getattr(row, "full_name", None) if has_pool_name else None
        resolved = pool_name if pool_name is not None and pd.notna(pool_name) else names.get(
            row.gsis_id, "-"
        )
        name = str(resolved)[:24]
        adp = f"{float(row.consensus_adp):.1f}" if pd.notna(row.consensus_adp) else "-"
        p_next = f"{float(row.p_available_next):.2f}" if pd.notna(row.p_available_next) else "-"
        star = "*" if row.fills_starting_slot else " "
        lines.append(
            f"{int(row.rank):>3}  {name:<24} {row.position:<4} {row.vorp:>7.1f} "
            f"{adp:>6} {p_next:>8} {row.score:>7.2f}{star}"
        )
    lines.append("  (* = fills an open starting slot)")
    return "\n".join(lines)
```

- [ ] **Step 4: Run the CLI tests to verify they pass**

Run: `pytest tests/test_draft/test_assistant_cli.py -v`
Expected: PASS — the new test prints "Rookie RB"; `test_run_prints_table` still prints "RB One"/"WR One" (its pool has no `full_name` → id_map fallback); `test_generate_recommendation` still unpacks `(rec, id_map)` and `RecommendationSchema.validate(rec)` tolerates the extra `full_name` column under `strict="filter"`; the season_value CLI tests still print "Player ".

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/cli.py tests/test_draft/test_assistant_cli.py
git commit -m "feat(draft): CLI format_table prefers pool full_name (id_map fallback)"
```

---

## Phase 2 — Controller

### Task 5: `LiveDraftSession.available_for_pick`

**Files:**
- Modify: `src/projections/draft/assistant/live.py` (add method near `best_available_by_position`)
- Test: `tests/test_draft/test_assistant_live.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_draft/test_assistant_live.py` (reuses `_pool_with_names` / `_rookie_row` from Task 3, and `Position` is already imported):

```python
def test_available_for_pick_position_filter_and_all() -> None:
    s = _session(pool=_pool_with_names())
    rbs = s.available_for_pick(position=Position.RB, top=100)
    assert set(rbs["position"]) == {"RB"}
    allp = s.available_for_pick(position=None, top=100)
    assert {"RB", "WR", "QB", "TE"} <= set(allp["position"])


def test_available_for_pick_case_insensitive_name_query() -> None:
    pool = pd.concat([_pool_with_names(), _rookie_row()], ignore_index=True)
    s = _session(pool=pool)
    hits = s.available_for_pick(query="rookie", top=100)
    assert "99-8467088" in set(hits["gsis_id"])
    assert all("rookie" in str(n).lower() for n in hits["full_name"])


def test_available_for_pick_sorts_by_vorp_caps_and_attaches_names() -> None:
    s = _session(pool=_pool_with_names())
    out = s.available_for_pick(top=3)
    assert "full_name" in out.columns
    assert len(out) == 3
    assert list(out["vorp"]) == sorted(out["vorp"], reverse=True)


def test_available_for_pick_excludes_drafted() -> None:
    s = _session(picks=["00-0000001"], pool=_pool_with_names())
    out = s.available_for_pick(top=100)
    assert "00-0000001" not in set(out["gsis_id"])


def test_available_for_pick_cap_applied_after_filter_reaches_deep_player() -> None:
    s = _session(pool=_pool_with_names())
    cross_top3 = s.available_for_pick(position=None, top=3)  # highest VORP across positions
    all_wrs = s.available_for_pick(position=Position.WR, top=100)
    deep_wr = all_wrs.iloc[-1]["gsis_id"]  # lowest-VORP WR
    assert deep_wr not in set(cross_top3["gsis_id"])  # hidden by the cross-position cap
    assert deep_wr in set(all_wrs["gsis_id"])  # reachable once the position filter is applied
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_draft/test_assistant_live.py -k "available_for_pick" -v`
Expected: FAIL with `AttributeError: 'LiveDraftSession' object has no attribute 'available_for_pick'`.

- [ ] **Step 3: Implement `available_for_pick`**

In `src/projections/draft/assistant/live.py`, add this method to `LiveDraftSession` (immediately after `best_available_by_position`):

```python
    def available_for_pick(
        self, position: Position | None = None, query: str = "", top: int = 60
    ) -> pd.DataFrame:
        """Name-attached available players for the picker pane.

        Filters to `position` (None = all positions), then to rows whose resolved name
        contains `query` (case-insensitive substring; "" = no filter), sorts by vorp
        descending, and caps to `top`. The cap is applied LAST — after the position and
        query filters and the sort — so a position selection or a search can reach a deep
        player an unfiltered cross-position top-N would hide. Names use the same
        pool-over-id_map source as `player_names` (so rookies match what's displayed).
        """
        avail = self.available_pool()
        if position is not None:
            avail = avail[avail["position"] == position.value]
        named = attach_names(avail, self.player_names)
        if query:
            named = named[
                named["full_name"].str.contains(query, case=False, na=False, regex=False)
            ]
        return named.sort_values("vorp", ascending=False).head(top).reset_index(drop=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_draft/test_assistant_live.py -k "available_for_pick" -v`
Expected: PASS (all five).

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/live.py tests/test_draft/test_assistant_live.py
git commit -m "feat(draft): add LiveDraftSession.available_for_pick for the picker pane"
```

---

## Phase 3 — Board UI

### Task 6: Click-to-confirm board (remove search box, two selectable panes, best-available filter)

**Files:**
- Modify: `scripts/draft_board.py` (imports line 26; remove `_search_box` 211-224; add `_selectable` + `_confirm_bar`; rewrite `_recommend_col` 244-272; add `_best_available_col`; trim `_roster_col` 275-291; rewrite `main` 306-325)

- [ ] **Step 1: Add `Position` to the schemas import**

In `scripts/draft_board.py` line 26, change:

```python
from projections.schemas import _PYARROW_STR, IdMapSchema, VorpTableSchema
```

to:

```python
from projections.schemas import _PYARROW_STR, IdMapSchema, Position, VorpTableSchema
```

- [ ] **Step 2: Remove `_search_box` and add `_selectable` + `_confirm_bar`**

Delete the entire `_search_box` function (lines 211-224). In its place add:

```python
def _selectable(named: pd.DataFrame, cols: list[str], key: str) -> None:
    """Render a single-row-selectable table; selecting a row stages `pending_pick`.

    `named` must carry `gsis_id` with a clean 0..n-1 index (selection rows are positional).
    The on_select callback (a closure over this render's frame + key) fires only for the
    pane the user clicked, so across the two selectable panes the most-recent click wins.
    """
    show = [c for c in cols if c in named.columns]

    def _stage() -> None:
        state = st.session_state.get(key)
        rows = state["selection"]["rows"] if state else []
        if rows:
            st.session_state["pending_pick"] = str(named.iloc[rows[0]]["gsis_id"])

    st.dataframe(
        named[show],
        height=400,
        hide_index=True,
        selection_mode="single-row",
        on_select=_stage,
        key=key,
    )


def _confirm_bar(s: LiveDraftSession) -> None:
    """Shared 'Confirm pick' / 'clear' controls for the staged selection (`pending_pick`)."""
    pending = st.session_state.get("pending_pick")
    if not pending:
        return
    name = s.name(str(pending))
    confirm_col, clear_col = st.columns([4, 1])
    if confirm_col.button(f"✅ Confirm pick: {name}", key="confirm_pending", type="primary"):
        try:
            s.record_pick(str(pending))
        except ValueError as exc:  # already drafted, or my-pick rookie absent from id_map
            st.warning(str(exc))
            return
        st.session_state["pending_pick"] = None
        _autosave(s)
        st.rerun()
    if clear_col.button("✕ clear", key="clear_pending"):
        st.session_state["pending_pick"] = None
        st.rerun()
```

- [ ] **Step 3: Rewrite `_recommend_col` (selectable on your turn; ADP shortcut on opponents')**

Replace `_recommend_col` (lines 244-272) with:

```python
def _recommend_col(s: LiveDraftSession) -> None:
    st.markdown("**★ Recommendations**")
    if s.is_complete:
        st.success("Draft complete.")
        return
    if s.mode == "copilot" and not s.is_my_pick:
        sug = s.suggested_pick()
        if sug is not None:
            name = s.name(str(sug))
            st.info(f"Opponent on the clock. ADP suggests: **{name}**")
            if st.button(f"Confirm pick: {name}", key="confirm_adp", type="primary"):
                _record_and_rerun(s, str(sug))
        st.caption("…or click a player in **Best available** (right) to record a different pick.")
        return
    with st.spinner("Scoring candidates…"):
        token = st.session_state.get("session_token", "")
        rec = _cached_recommendation(token, tuple(s.picks), s.strategy_name, s.n_sims, s.sigma)
    named = attach_names(rec, s.player_names).reset_index(drop=True)
    cols = [
        "rank",
        "full_name",
        "position",
        "vorp",
        "consensus_adp",
        "p_available_next",
        "score",
        "fills_starting_slot",
    ]
    st.caption("Click a row to stage it, then **Confirm pick** above.")
    _selectable(named, cols, key=f"sel_rec_{s.current_pick}")
```

- [ ] **Step 4: Trim `_roster_col` and add `_best_available_col`**

Replace `_roster_col` (lines 275-291) with the roster-only version plus the new pane below it:

```python
def _roster_col(s: LiveDraftSession) -> None:
    st.markdown("**My Roster**")
    view = s.my_roster_view()
    st.dataframe(view.filled[["slot", "full_name", "position"]], hide_index=True)
    if view.open_slots:
        st.caption(
            "Open starting slots: "
            + ", ".join(f"{slot.value} x{n}" for slot, n in view.open_slots.items())
        )


def _best_available_col(s: LiveDraftSession) -> None:
    st.markdown("**🔎 Best available**")
    if s.is_complete:
        return
    pos_label = st.selectbox(
        "Position", ["All", "QB", "RB", "WR", "TE"], key=f"ba_pos_{s.current_pick}"
    )
    query = st.text_input("Search player", key=f"ba_query_{s.current_pick}", placeholder="name…")
    position = None if pos_label == "All" else Position(pos_label)
    avail = s.available_for_pick(position=position, query=query, top=60)
    if avail.empty:
        st.caption("No matching available players.")
        return
    st.caption("Click a row to stage it, then **Confirm pick** above.")
    _selectable(
        avail, ["full_name", "position", "vorp", "consensus_adp"], key=f"sel_ba_{s.current_pick}"
    )
```

(The previous `_roster_col` "Best available by position" block is intentionally dropped; `s.best_available_by_position` stays on the controller — it still has its own test — it is just no longer rendered.)

- [ ] **Step 5: Rewrite `main` (drop the search box; confirm bar; right column = roster + best-available)**

Replace `main` (lines 306-325) with:

```python
def main() -> None:
    st.set_page_config(page_title="Draft Board", layout="wide")
    st.title("🏈 Live Draft Board")
    _sidebar()

    s: LiveDraftSession | None = st.session_state.get("session")
    if s is None:
        st.info("Configure the draft in the sidebar and click **Start / restart draft**.")
        return
    _status_bar(s)
    _confirm_bar(s)
    _mock_controls(s)
    left, center, right = st.columns([1.1, 2.0, 1.3])
    with left:
        _board_log_col(s)
    with center:
        _recommend_col(s)
    with right:
        _roster_col(s)
        _best_available_col(s)
    st.caption(f"Mode: {s.mode} · strategy: {s.strategy_name} · {len(s.picks)} picks made")
```

- [ ] **Step 6: Verify the existing smoke test still passes (no-session path)**

Run: `pytest tests/test_scripts/test_draft_board_smoke.py -v`
Expected: PASS (`test_draft_board_loads_without_session` — the script still imports and renders the no-session prompt).

- [ ] **Step 7: Lint + type-check the board now (it has no per-line test)**

Run: `ruff check scripts/draft_board.py && mypy scripts/draft_board.py`
Expected: clean. (If mypy flags `on_select=_stage`, confirm `_stage` is typed `() -> None`; Streamlit's `WidgetCallback` accepts it.)

- [ ] **Step 8: Commit**

```bash
git add scripts/draft_board.py
git commit -m "feat(draft): click-to-confirm board with searchable best-available pane"
```

---

### Task 7: Board smoke tests (best-available widgets, confirm flow, ADP shortcut)

**Files:**
- Modify: `tests/test_scripts/test_draft_board_smoke.py`

- [ ] **Step 1: Write the failing smoke tests**

Replace the contents of `tests/test_scripts/test_draft_board_smoke.py` with:

```python
"""Headless smoke for the Streamlit draft board: it imports and runs without raising,
renders the best-available picker (not the old search box), and records picks via the
shared confirm flow + the opponent ADP shortcut."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _smoke_session(picks: list[str] | None = None, my_slot: int = 1):  # type: ignore[no-untyped-def]
    from projections.draft.assistant.live import LiveDraftSession
    from projections.draft.assistant.strategy import RawVorpStrategy
    from projections.draft.league_config import LeagueConfig
    from projections.schemas import _PYARROW_STR, RosterSlot, Ruleset, validate_gsis_id

    ids = [f"00-000{i:04d}" for i in range(1, 25)]
    positions = ["RB", "WR", "QB", "TE"] * 6
    names = [f"Player {i}" for i in range(1, 25)]
    id_map = pd.DataFrame(
        {
            "gsis_id": pd.array(ids, dtype=_PYARROW_STR),
            "position": pd.array(positions, dtype=_PYARROW_STR),
            "full_name": pd.array(names, dtype=_PYARROW_STR),
            "team": pd.array(["KC"] * 24, dtype=_PYARROW_STR),
        }
    )
    pool = pd.DataFrame(
        {
            "gsis_id": pd.array(ids, dtype=_PYARROW_STR),
            "position": pd.array(positions, dtype=_PYARROW_STR),
            "season_mean_fpts": [300.0 - i for i in range(24)],
            "vorp": [150.0 - i for i in range(24)],
            "replacement_fpts": [100.0] * 24,
            "consensus_adp": pd.array([float(i + 1) for i in range(24)], dtype=pd.Float64Dtype()),
            "full_name": pd.array(names, dtype=_PYARROW_STR),
        }
    )
    league = LeagueConfig(
        name="t",
        n_teams=12,
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 5,
        },
        ruleset=Ruleset.espn_ppr(),
    )
    return LiveDraftSession(
        league=league,
        my_slot=my_slot,
        id_map=id_map,
        pool=pool,
        strategy=RawVorpStrategy(),
        strategy_name="raw_vorp",
        mode="copilot",
        adp_jitter=0.0,
        picks=[validate_gsis_id(p) for p in (picks or [])],
    )


def test_draft_board_loads_without_session() -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("scripts/draft_board.py").run()
    assert not at.exception
    assert any("Start" in str(getattr(el, "value", "")) for el in at.info)


def test_board_shows_best_available_and_drops_search_box() -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("scripts/draft_board.py")
    at.session_state["session"] = _smoke_session(my_slot=1)
    at.session_state["session_token"] = "tok"
    at.session_state["autosave_path"] = None
    at.run()
    assert not at.exception
    # Best-available position dropdown is present with All + the skill positions.
    assert any(set(sb.options) >= {"All", "QB", "RB", "WR", "TE"} for sb in at.selectbox)
    # The old top "Record a pick — search a player" box is gone.
    labels = [str(getattr(ti, "label", "")) for ti in at.text_input]
    assert not any("Record a pick" in lbl for lbl in labels)


def test_board_confirm_records_staged_pick(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("scripts/draft_board.py")
    at.session_state["session"] = _smoke_session(my_slot=1)  # pick 1 is mine
    at.session_state["session_token"] = "tok"
    at.session_state["autosave_path"] = str(tmp_path / "auto.json")
    at.session_state["pending_pick"] = "00-0000003"  # a QB in the fixture, present in id_map
    at.run()
    assert not at.exception
    at.button(key="confirm_pending").click().run()
    assert not at.exception
    assert at.session_state["session"].picks == ["00-0000003"]


def test_board_opponent_adp_shortcut_records(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("scripts/draft_board.py")
    at.session_state["session"] = _smoke_session(my_slot=2)  # pick 1 is an opponent's
    at.session_state["session_token"] = "tok"
    at.session_state["autosave_path"] = str(tmp_path / "auto.json")
    at.run()
    assert not at.exception
    at.button(key="confirm_adp").click().run()
    assert not at.exception
    assert len(at.session_state["session"].picks) == 1
```

- [ ] **Step 2: Run the smoke tests**

Run: `pytest tests/test_scripts/test_draft_board_smoke.py -v`
Expected: PASS (all four). If `at.button(key=...)` / `sb.options` differ in this Streamlit build, adjust the accessor (e.g. iterate `at.button` and match `.key`) — but do not weaken the behavioral assertions (a pick is recorded; the search box is gone).

- [ ] **Step 3: Commit**

```bash
git add tests/test_scripts/test_draft_board_smoke.py
git commit -m "test(draft): board smoke for best-available picker + confirm flow"
```

---

## Phase 4 — Verify + sync

### Task 8: Regenerate the parquet, run all gates, manual board check, update PM/TODO

**Files:**
- Regenerate (untracked artifact, do NOT commit): `data/consensus_vorp_2026.parquet`
- Modify: `project_management.md`, `TODO.md`

- [ ] **Step 1: Regenerate the consensus VORP parquet so the live board picks up names**

Run:

```bash
python scripts/generate_vorp_table.py --source consensus --season 2026 \
  --league-config configs/league_espn_ppr_12team_skill.json \
  --data-root data --out data/consensus_vorp_2026.parquet
```

Expected: exit 0; the per-position summary prints. Then confirm `full_name` landed:

```bash
python -c "import pandas as pd; d=pd.read_parquet('data/consensus_vorp_2026.parquet'); print('full_name' in d.columns, d['full_name'].notna().sum(), '/', len(d)); print(d.loc[d['gsis_id']=='99-8467088', ['gsis_id','position','vorp','full_name']].to_string(index=False))"
```

Expected: `True`, a high non-null count, and `99-8467088` now shows a real name (no longer "—").

- [ ] **Step 2: Run the full gate suite (CLAUDE.md directive #4)**

Run each and fix any failure before proceeding:

```bash
pytest -v
mypy src tests
ruff check src tests
ruff format --check src tests
pytest -v -k "ingest or store or schemas"
```

Expected: all green / zero violations. (If `ruff format --check` reports drift, run `ruff format src tests` and re-commit the affected files under their owning task.)

- [ ] **Step 3: Manual board verification**

Run: `python -m streamlit run scripts/draft_board.py`
Confirm, then stop the server:
- Start a co-pilot draft (slot 1, strategy `now_or_never`, the regenerated parquet).
- On an opponent's turn the ADP suggestion shows a real name (incl. when it is rookie `99-8467088`); its one-click **Confirm pick** records and advances.
- **Best available** (right) renders the position dropdown (`All` shows all positions) + search; typing a name filters; clicking a row stages it and **Confirm pick** above records it.
- On your turn, clicking a recommendation row stages it; clicking a best-available row instead re-stages (most-recent wins); **Confirm pick** records.
- No "Record a pick — search a player" box remains at the top.

- [ ] **Step 4: Update `project_management.md` and `TODO.md`**

- In `project_management.md`: note the draft-board UX pass is done — pool `full_name` carried into `VorpTableSchema`/the consensus generator, `player_names` is pool-over-id_map, `record_pick`'s my-pick guard now checks id_map membership directly, `available_for_pick` added, and the board is click-to-confirm with a searchable/position-filterable best-available pane.
- In `TODO.md`: mark the three board-UX asks done; add the one deferred item from spec §9 — *resolve my-roster position from the pool's `position` column so placeholder-gsis rookies are draftable to my own roster (currently opponent-only)*.

- [ ] **Step 5: Commit**

```bash
git add project_management.md TODO.md
git commit -m "docs(draft): record draft-board UX pass + defer my-roster rookie pickability"
```

- [ ] **Step 6: Finish the branch**

Use **superpowers:finishing-a-development-branch** (verify tests → present options → execute). Default toward option 2 (push + open a PR) so the work reaches `main` via PR per the repo's workflow rule. Do NOT stage `.gitignore` or the untracked `data/`/checkpoint artifacts.

---

## Self-review

**Spec coverage:**
- §3 name fix → Task 1 (schema), Task 2 (generator merge + regen in Task 8), Task 3 (`player_names` overlay + guard decoupling via `_id_map_ids`), Task 4 (`cli.format_table`). ✔
- §4 pick flow (single `pending_pick`, two selectable panes, most-recent-click-wins via per-pane `on_select` closures, reset by keying widgets on `current_pick`, separate opponent ADP shortcut, search box removed) → Task 6. ✔
- §5 `available_for_pick` (position filter, case-insensitive query, vorp sort, top cap LAST, names attached) → Task 5. ✔
- §6 edge cases: weekly pool no full_name (Task 3 test + Task 5 `available_pool` path), null pool name → id_map (Task 3 test), stale/already-drafted pending (`record_pick` ValueError caught in `_confirm_bar`), search no matches (`avail.empty` guard), my-pick rookie rejected/no-poison (Task 3 guard test + `_confirm_bar` warn-and-return), selection reset per pick (widget keys on `current_pick`), `All` capped (`top=60`). ✔
- §7 testing: schema (T1), generator cheap unit + E2E (T2), player_names (T3), record_pick guard regression (T3), available_for_pick (T5), board smoke incl. removed search box + confirm + ADP shortcut (T7), regression gates (T8). ✔
- §8 phasing mirrored as Phases 1-4. ✔

**Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to Task N" — every code step shows full code and every test step shows the assertions.

**Type/name consistency:** `_id_map_ids` (cached frozenset), `available_for_pick(position, query, top)`, `_selectable(named, cols, key)`, `_confirm_bar(s)`, `_best_available_col(s)`, `pending_pick`/`confirm_pending`/`confirm_adp`/`sel_rec_{cp}`/`sel_ba_{cp}`/`ba_pos_{cp}`/`ba_query_{cp}` keys, and `_merge_consensus_columns(out_df, consensus)` are used identically wherever they appear. The CLI keeps its `(rec, id_map)` return so `test_generate_recommendation` is untouched.
