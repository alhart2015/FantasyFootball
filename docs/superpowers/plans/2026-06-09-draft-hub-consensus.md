# Draft Hub on Consensus — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-point the already-shipped Draft Hub (VORP → auction → snake cheat sheet) off the draft-invalid in-season model and onto PR #55's `ConsensusProjectionSchema`, and add the deferred ADP-delta column to the cheat sheet.

**Architecture:** A pure adapter converts `ConsensusProjectionSchema` → the `ProjectionSeasonSchema` frame `generate_vorp_table` already accepts. The VORP CLI gains a `--source consensus` mode that reads the published consensus partition, runs the adapter, and joins `consensus_adp` onto its output. `consensus_adp` rides the VORP parquet (a new *optional* `VorpTableSchema` column) to the cheat sheet, which computes an `adp_delta` value-vs-market column. The VORP math, auction generator, `_select_pool`, and the weekly path are untouched.

**Tech Stack:** Python 3.12, pandas + pandas pyarrow/extension dtypes, pandera (`DataFrameModel`, `strict="filter"`, `coerce=True`), pytest, mypy (strict), ruff. Source spec: `docs/superpowers/specs/2026-06-09-draft-hub-consensus-design.md`.

---

## File Structure

- **Create** `src/projections/draft/consensus_source.py` — the pure adapter (`consensus_to_season_projections`). One responsibility: bridge the published consensus contract to the season-projection contract VORP consumes.
- **Modify** `src/projections/schemas.py` — add an *optional* nullable `consensus_adp` to `VorpTableSchema`; add `consensus_adp` + `adp_delta` to `SnakeCheatSheetSchema`.
- **Modify** `src/projections/draft/snake_cheat_sheet.py` — carry `consensus_adp` through and compute `adp_delta`.
- **Modify** `scripts/generate_vorp_table.py` — `--source {weekly,consensus}` mode, consensus read, dropped-but-draftable warning, ADP left-join.
- **Modify** `scripts/generate_snake_cheat_sheet.py` — surface top-3 `adp_delta` in the stdout summary.
- **Create** `configs/league_espn_ppr_12team_skill.json` — skill-only example config (no K/DST).
- **Tests:** `tests/test_draft/test_consensus_source.py` (new), additions to `tests/test_schemas/test_dataframe_schemas.py`, `tests/test_draft/test_snake_cheat_sheet.py`, `tests/test_scripts/test_generate_vorp_table_cli.py`, `tests/test_scripts/test_generate_snake_cheat_sheet_cli.py`, `tests/test_scripts/test_generate_auction_values_cli.py`.

**Conventions (from CLAUDE.md):** reference `Position`/`RosterSlot`/`Ruleset` enums, never raw strings; `_PYARROW_STR` for nullable string columns, `pd.Int64Dtype()`/`pd.Float64Dtype()` for nullable int/float; `df = SCHEMA.validate(df)` with reassignment at every boundary; never `df.to_parquet` from library code (CLIs may, mirroring the existing draft CLIs).

---

## Task 1: Consensus → season-projection adapter

**Files:**
- Create: `src/projections/draft/consensus_source.py`
- Test: `tests/test_draft/test_consensus_source.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_draft/test_consensus_source.py`:

```python
"""Tests for `projections.draft.consensus_source.consensus_to_season_projections`."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.draft.consensus_source import consensus_to_season_projections
from projections.schemas import (
    _PYARROW_STR,
    ConsensusProjectionSchema,
    Position,
    ProjectionSeasonSchema,
)


def _consensus_row(
    *,
    gsis_id: str,
    position: Position,
    has_points: bool,
    projected_points_ppr: float | None,
    consensus_adp: float | None = 10.0,
    consensus_rank: int | None = 1,
    asof: str = "2026-06-09",
    season: int = 2026,
) -> dict[str, object]:
    """One ConsensusProjectionSchema-shaped row. Stat-line cols are irrelevant to
    the adapter (it reads projected_points_ppr), so they are left null."""
    return {
        "gsis_id": gsis_id,
        "season": season,
        "asof": asof,
        "full_name": "Test Player",
        "position": position.value,
        "consensus_adp": consensus_adp,
        "consensus_rank": consensus_rank,
        "n_adp_sources": 2,
        "has_points": has_points,
        "projected_points_ppr": projected_points_ppr,
        "passing_yards": None,
        "passing_tds": None,
        "interceptions": None,
        "rushing_yards": None,
        "rushing_tds": None,
        "receptions": None,
        "receiving_yards": None,
        "receiving_tds": None,
        "fumbles_lost": None,
        "is_placeholder_gsis": False,
        "ruleset": "ESPN_PPR",
    }


def _consensus_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["asof"] = df["asof"].astype(_PYARROW_STR)
    df["full_name"] = df["full_name"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df["ruleset"] = df["ruleset"].astype(_PYARROW_STR)
    return ConsensusProjectionSchema.validate(df)


def test_filters_to_has_points_and_maps_points_to_season_mean() -> None:
    frame = _consensus_frame(
        [
            _consensus_row(gsis_id="00-3000001", position=Position.WR, has_points=True,
                           projected_points_ppr=250.5),
            _consensus_row(gsis_id="00-3000002", position=Position.WR, has_points=False,
                           projected_points_ppr=None),
        ]
    )
    out = consensus_to_season_projections(frame)
    assert list(out["gsis_id"]) == ["00-3000001"]  # ADP-only row dropped
    assert out["season_mean"].iloc[0] == pytest.approx(250.5)


def test_degenerate_distribution_and_metadata() -> None:
    frame = _consensus_frame(
        [
            _consensus_row(gsis_id="00-1000001", position=Position.QB, has_points=True,
                           projected_points_ppr=300.0, asof="2026-06-09", season=2026),
        ]
    )
    out = consensus_to_season_projections(frame)
    row = out.iloc[0]
    assert row["season_p10"] == row["season_p50"] == row["season_p90"] == row["season_mean"]
    assert row["n_weeks"] == 17
    assert row["model_id"] == "consensus:2026-06-09"
    assert row["ruleset"] == "ESPN_PPR"
    assert int(row["season"]) == 2026
    assert out["generated_at"].dt.tz is not None  # tz-aware, ProjectionSeasonSchema requires it


def test_output_validates_against_projection_season_schema() -> None:
    frame = _consensus_frame(
        [
            _consensus_row(gsis_id="00-2000001", position=Position.RB, has_points=True,
                           projected_points_ppr=220.0),
        ]
    )
    out = consensus_to_season_projections(frame)
    # Idempotent re-validation = it is a conforming frame.
    pd.testing.assert_frame_equal(out, ProjectionSeasonSchema.validate(out))


def test_empty_and_all_adp_only_inputs_return_valid_empty_frame() -> None:
    # No has_points rows -> valid empty ProjectionSeasonSchema frame (not an error).
    frame = _consensus_frame(
        [
            _consensus_row(gsis_id="00-4000001", position=Position.TE, has_points=False,
                           projected_points_ppr=None),
        ]
    )
    out = consensus_to_season_projections(frame)
    assert out.empty
    ProjectionSeasonSchema.validate(out)


def test_raises_on_mixed_asof() -> None:
    frame = _consensus_frame(
        [
            _consensus_row(gsis_id="00-3000001", position=Position.WR, has_points=True,
                           projected_points_ppr=250.0, asof="2026-06-09"),
            _consensus_row(gsis_id="00-3000002", position=Position.WR, has_points=True,
                           projected_points_ppr=240.0, asof="2026-06-08"),
        ]
    )
    with pytest.raises(ValueError, match="asof"):
        consensus_to_season_projections(frame)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_draft/test_consensus_source.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'projections.draft.consensus_source'`.

- [ ] **Step 3: Write the adapter**

Create `src/projections/draft/consensus_source.py`:

```python
"""Adapter: published consensus projection -> the season-projection contract VORP consumes.

The Draft Hub's `generate_vorp_table` accepts a `ProjectionSeasonSchema` frame; the consensus
layer (PR #55) publishes `ConsensusProjectionSchema`. This pure adapter bridges them so the
draft tooling is fed the draft-valid external consensus instead of the in-season model.

Point estimates only: `season_p10 == season_p50 == season_p90 == season_mean` is the honest
representation of a single-source point projection (a real band waits for >=2 stat-line
sources, a later consensus slice). See
docs/superpowers/specs/2026-06-09-draft-hub-consensus-design.md.
"""

from __future__ import annotations

import pandas as pd

from projections.schemas import _PYARROW_STR, ConsensusProjectionSchema, ProjectionSeasonSchema

# Sentinel n_weeks for a consensus-derived season total: "full season", not a per-week
# aggregate. 17 reads as a complete season (vs the misleading 1 = "one week of data").
# VORP does not read n_weeks; no consumer should filter consensus rows on it.
_FULL_SEASON_WEEKS = 17


def consensus_to_season_projections(consensus: pd.DataFrame) -> pd.DataFrame:
    """Convert one consensus snapshot into a ProjectionSeasonSchema frame.

    Filters to `has_points` players (VORP needs a non-null season-points total),
    maps `projected_points_ppr` -> `season_mean` with a degenerate point-mass band,
    and stamps provenance (`model_id = "consensus:<asof>"`). Carries the snapshot's
    `ruleset` so `generate_vorp_table`'s ruleset-match guard fires on a mismatch.

    Raises ValueError if the frame mixes `asof` snapshots or seasons (a caller bug;
    the CLI reads exactly one partition).
    """
    df = ConsensusProjectionSchema.validate(consensus)

    if not df.empty:
        asofs = df["asof"].unique()
        if len(asofs) > 1:
            raise ValueError(f"consensus frame mixes asof snapshots: {sorted(asofs)}")
        seasons = df["season"].unique()
        if len(seasons) > 1:
            raise ValueError(f"consensus frame mixes seasons: {sorted(seasons.tolist())}")
        asof = str(asofs[0])
        season = int(seasons[0])
    else:
        # Fully-empty input: produce a 0-row conforming frame (placeholders never reach a row).
        asof, season = "", 0

    points = df[df["has_points"]]
    season_mean = points["projected_points_ppr"].astype("float64").to_numpy()

    out = pd.DataFrame(
        {
            "gsis_id": points["gsis_id"].to_numpy(),
            "season": season,
            "position": points["position"].to_numpy(),
            "ruleset": points["ruleset"].to_numpy(),
            "n_weeks": _FULL_SEASON_WEEKS,
            "season_mean": season_mean,
            "season_p10": season_mean,
            "season_p50": season_mean,
            "season_p90": season_mean,
            "model_id": f"consensus:{asof}",
            "generated_at": pd.Timestamp.now(tz="UTC").as_unit("us"),
        }
    )
    for col in ("gsis_id", "position", "ruleset", "model_id"):
        out[col] = out[col].astype(_PYARROW_STR)
    return ProjectionSeasonSchema.validate(out)


__all__ = ["consensus_to_season_projections"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_draft/test_consensus_source.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/consensus_source.py tests/test_draft/test_consensus_source.py
git commit -m "feat(draft): consensus -> season-projection adapter for VORP"
```

---

## Task 2: `VorpTableSchema` — optional nullable `consensus_adp`

**Files:**
- Modify: `src/projections/schemas.py` (the `VorpTableSchema` class, ~line 966)
- Test: `tests/test_schemas/test_dataframe_schemas.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_schemas/test_dataframe_schemas.py` (it already imports `VorpTableSchema`, `Position`, `_PYARROW_STR`, `pd`, `pytest`):

```python
def test_vorp_table_schema_accepts_optional_consensus_adp() -> None:
    """Consensus-fed VORP tables carry consensus_adp; weekly-fed ones do not.
    The column is Optional so both validate."""
    base = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-1000001", "00-2000001"], dtype=_PYARROW_STR),
            "position": pd.array([Position.QB.value, Position.RB.value], dtype=_PYARROW_STR),
            "season_mean_fpts": pd.array([320.0, 260.0], dtype="float64"),
            "vorp": pd.array([80.0, 30.0], dtype="float64"),
            "replacement_fpts": pd.array([240.0, 230.0], dtype="float64"),
        }
    )
    # Without consensus_adp (weekly path) -> still validates.
    VorpTableSchema.validate(base)

    # With consensus_adp (consensus path) -> validates and the column survives.
    with_adp = base.copy()
    with_adp["consensus_adp"] = pd.array([2.1, 15.4], dtype=pd.Float64Dtype())
    validated = VorpTableSchema.validate(with_adp)
    assert "consensus_adp" in validated.columns

    # Non-positive ADP is rejected (gt=0).
    bad = with_adp.copy()
    bad.loc[bad.index[0], "consensus_adp"] = 0.0
    with pytest.raises(SchemaError):
        VorpTableSchema.validate(bad)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_schemas/test_dataframe_schemas.py::test_vorp_table_schema_accepts_optional_consensus_adp -v`
Expected: FAIL — `with_adp` validation drops `consensus_adp` (strict="filter" removes the undeclared column), so `assert "consensus_adp" in validated.columns` fails.

- [ ] **Step 3: Add the column to `VorpTableSchema`**

In `src/projections/schemas.py`, inside `class VorpTableSchema`, add the field after `replacement_fpts`:

```python
    replacement_fpts: Series[float] = pa.Field(ge=0)
    # Optional (not-required): populated only on the consensus-fed path (the raw market ADP
    # the cheat sheet's adp_delta uses). Weekly-path VORP tables omit it and still validate.
    consensus_adp: Series[pd.Float64Dtype] | None = pa.Field(gt=0, nullable=True)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_schemas/test_dataframe_schemas.py::test_vorp_table_schema_accepts_optional_consensus_adp -v`
Expected: PASS.

- [ ] **Step 5: Confirm the existing VORP round-trip test still passes**

Run: `pytest tests/test_schemas/test_dataframe_schemas.py::test_vorp_table_schema_round_trip -v`
Expected: PASS (the Optional column is not required, so the 5-column frame still validates).

- [ ] **Step 6: Commit**

```bash
git add src/projections/schemas.py tests/test_schemas/test_dataframe_schemas.py
git commit -m "feat(schemas): optional consensus_adp on VorpTableSchema"
```

---

## Task 3: VORP CLI — `--source consensus` mode

**Files:**
- Modify: `scripts/generate_vorp_table.py`
- Test: `tests/test_scripts/test_generate_vorp_table_cli.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_scripts/test_generate_vorp_table_cli.py`. First, a helper that writes a consensus partition (place after `_make_weekly_partition`):

```python
def _make_consensus_partition(
    data_root: Path,
    season: int,
    asof: str,
    rows: list[dict[str, object]],
) -> Path:
    """Write one ConsensusProjectionSchema snapshot under
    data_root/processed/consensus_projections/season=YYYY/asof=YYYY-MM-DD/part.parquet."""
    from projections.schemas import ConsensusProjectionSchema

    snap_dir = (
        data_root / "processed" / "consensus_projections" / f"season={season}" / f"asof={asof}"
    )
    snap_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in ("gsis_id", "asof", "full_name", "position", "ruleset"):
        df[col] = df[col].astype(_PYARROW_STR)
    df = ConsensusProjectionSchema.validate(df)
    df.to_parquet(snap_dir / "part.parquet", index=False)
    return data_root


def _consensus_rows() -> list[dict[str, object]]:
    """A small skill-position consensus snapshot: every position has enough players to
    fill the tiny_test config's pool, plus one ADP-only (no-points) draftable player."""
    rows: list[dict[str, object]] = []
    counts = {Position.QB: 8, Position.RB: 12, Position.WR: 12, Position.TE: 8}
    rank = 1
    for pos, n in counts.items():
        for i in range(n):
            rows.append(
                {
                    "gsis_id": f"00-{_POSITION_ID_PREFIX[pos]}{i:06d}",
                    "season": 2026,
                    "asof": "2026-06-09",
                    "full_name": f"{pos.value} Player {i}",
                    "position": pos.value,
                    "consensus_adp": float(rank),
                    "consensus_rank": rank,
                    "n_adp_sources": 2,
                    "has_points": True,
                    "projected_points_ppr": 300.0 - rank,
                    "passing_yards": None, "passing_tds": None, "interceptions": None,
                    "rushing_yards": None, "rushing_tds": None, "receptions": None,
                    "receiving_yards": None, "receiving_tds": None, "fumbles_lost": None,
                    "is_placeholder_gsis": False,
                    "ruleset": "ESPN_PPR",
                }
            )
            rank += 1
    # One draftable (low ADP) player with NO points -> must be dropped AND warned about.
    rows.append(
        {
            "gsis_id": "00-3999999", "season": 2026, "asof": "2026-06-09",
            "full_name": "Hyped Rookie", "position": Position.WR.value,
            "consensus_adp": 5.0, "consensus_rank": 5, "n_adp_sources": 1,
            "has_points": False, "projected_points_ppr": None,
            "passing_yards": None, "passing_tds": None, "interceptions": None,
            "rushing_yards": None, "rushing_tds": None, "receptions": None,
            "receiving_yards": None, "receiving_tds": None, "fumbles_lost": None,
            "is_placeholder_gsis": True, "ruleset": "ESPN_PPR",
        }
    )
    return rows


def test_cli_consensus_mode_round_trip(cli_inputs: dict[str, Path], tmp_path: Path) -> None:
    from projections.schemas import VorpTableSchema

    data_root = tmp_path / "data"
    _make_consensus_partition(data_root, 2026, "2026-06-09", _consensus_rows())
    proc = _run_cli(
        "--source", "consensus",
        "--season", "2026",
        "--league-config", str(cli_inputs["config"]),
        "--data-root", str(data_root),
        "--out", str(cli_inputs["out_parquet"]),
    )
    assert proc.returncode == 0, proc.stderr
    out = pd.read_parquet(cli_inputs["out_parquet"])
    VorpTableSchema.validate(out)
    assert "consensus_adp" in out.columns
    assert out["consensus_adp"].notna().all()
    # The ADP-only "Hyped Rookie" is NOT in the VORP table (no points to rank on).
    assert "00-3999999" not in set(out["gsis_id"])
    # ...but the CLI WARNED about dropping a draftable player.
    assert "Hyped Rookie" in proc.stderr


def test_cli_consensus_mode_requires_data_root_not_weekly(cli_inputs: dict[str, Path]) -> None:
    """--source consensus must not require --weekly-projections."""
    # Missing --data-root falls back to default "data" (no partition there) -> a clear
    # FileNotFoundError, NOT an argparse 'weekly-projections required' error.
    proc = _run_cli(
        "--source", "consensus",
        "--season", "1999",
        "--league-config", str(cli_inputs["config"]),
        "--out", str(cli_inputs["out_parquet"]),
    )
    assert proc.returncode != 0
    assert "weekly-projections" not in (proc.stderr + proc.stdout).lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_scripts/test_generate_vorp_table_cli.py -k consensus -v`
Expected: FAIL — argparse rejects unknown `--source`/`--data-root` (or the required `--weekly-projections` errors).

- [ ] **Step 3: Rewrite the CLI to add the consensus mode**

In `scripts/generate_vorp_table.py`, update imports (top of file):

```python
import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from projections.aggregation.season import aggregate_to_season
from projections.draft.consensus_source import consensus_to_season_projections
from projections.draft.league_config import LeagueConfig
from projections.draft.vorp import generate_vorp_table
from projections.schemas import (
    ConsensusProjectionSchema,
    Position,
    ProjectionWeeklySchema,
    VorpTableSchema,
)
from projections.store import read_latest_partition, read_partition
```

Replace `_parse_args` with source-aware flags:

```python
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a per-player VORP table.")
    parser.add_argument("--season", type=int, required=True, help="Target season (e.g. 2026).")
    parser.add_argument(
        "--league-config", type=Path, required=True, help="Path to LeagueConfig JSON."
    )
    parser.add_argument(
        "--source",
        choices=["weekly", "consensus"],
        default="weekly",
        help="Projection source: 'weekly' (in-season model, default) or 'consensus' (preseason).",
    )
    parser.add_argument(
        "--weekly-projections",
        type=Path,
        default=None,
        help="[--source weekly] Weekly-projections partition root "
        "(e.g. data/projections/weekly/ruleset=espn_ppr).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="[--source consensus] Store root; reads <root>/processed/consensus_projections/.",
    )
    parser.add_argument(
        "--asof",
        type=date.fromisoformat,
        default=None,
        help="[--source consensus] Snapshot date YYYY-MM-DD; defaults to the latest snapshot.",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output path; .csv or .parquet (sniffed)."
    )
    return parser.parse_args()
```

Add a warning helper (after `_log_per_position_summary`):

```python
def _warn_dropped_draftable(consensus: pd.DataFrame, league_config: LeagueConfig) -> None:
    """Surface players the has_points filter drops whose ADP is inside the draftable range,
    so the coverage gap is an explicit eyeball check rather than silent (spec §3.3)."""
    dropped = consensus[~consensus["has_points"] & consensus["consensus_adp"].notna()]
    draftable = dropped[dropped["consensus_adp"] <= league_config.total_pool_size]
    if draftable.empty:
        print("0 draftable players dropped for missing points.", file=sys.stderr)
        return
    print(
        f"WARNING: {len(draftable)} draftable player(s) dropped — have an ADP inside the "
        f"top {league_config.total_pool_size} but no projected points:",
        file=sys.stderr,
    )
    for row in draftable.sort_values("consensus_adp").head(25).itertuples(index=False):
        print(
            f"  {row.full_name}  ADP={float(row.consensus_adp):.1f}  rank={row.consensus_rank}",
            file=sys.stderr,
        )
```

Replace the body of `main()` (from `args = _parse_args()` through `out_df = generate_vorp_table(...)`):

```python
def main() -> int:
    args = _parse_args()
    league_config = LeagueConfig.model_validate_json(args.league_config.read_text())

    consensus: pd.DataFrame | None = None
    if args.source == "weekly":
        if args.weekly_projections is None:
            print("ERROR: --weekly-projections is required for --source weekly", file=sys.stderr)
            return 1
        weekly_root: Path = args.weekly_projections
        weekly = read_partition(weekly_root.parent, table=weekly_root.name, season=args.season)
        weekly = ProjectionWeeklySchema.validate(weekly)
        season_proj = aggregate_to_season(weekly, ruleset=league_config.ruleset)
    else:
        processed = args.data_root / "processed"
        if args.asof is not None:
            consensus = read_partition(
                processed, "consensus_projections", season=args.season, asof=args.asof
            )
        else:
            consensus = read_latest_partition(
                processed, "consensus_projections", season=args.season
            )
        consensus = ConsensusProjectionSchema.validate(consensus)
        _warn_dropped_draftable(consensus, league_config)
        season_proj = consensus_to_season_projections(consensus)

    in_scope = {slot.value for slot, count in league_config.roster_slots.items() if count > 0}
    dropped_positions = (
        season_proj[~season_proj["position"].isin(in_scope)]["position"].value_counts().to_dict()
    )

    out_df = generate_vorp_table(season_proj, league_config)

    if consensus is not None:
        adp = consensus[["gsis_id", "consensus_adp"]]
        out_df = out_df.merge(adp, on="gsis_id", how="left")
        out_df = VorpTableSchema.validate(out_df)

    suffix = args.out.suffix.lower()
    if suffix == ".csv":
        out_df.sort_values("vorp", ascending=False).to_csv(args.out, index=False)
    elif suffix == ".parquet":
        out_df.to_parquet(args.out, index=False)
    else:
        print(
            f"ERROR: unsupported output extension {suffix!r}; use .csv or .parquet",
            file=sys.stderr,
        )
        return 1

    _log_per_position_summary(out_df, league_config, dropped_positions)
    return 0
```

Note: a `FileNotFoundError` from `read_latest_partition`/`read_partition` propagates as a non-zero exit with the store's message (satisfies `test_cli_consensus_mode_requires_data_root_not_weekly`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_scripts/test_generate_vorp_table_cli.py -v`
Expected: PASS — both new consensus tests AND the three existing weekly tests (`test_cli_parquet_round_trip`, `test_cli_errors_when_config_requires_missing_position`, `test_cli_errors_on_ruleset_mismatch`).

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_vorp_table.py tests/test_scripts/test_generate_vorp_table_cli.py
git commit -m "feat(draft): VORP CLI --source consensus mode + dropped-draftable warning"
```

---

## Task 4: Snake cheat sheet — ADP-delta columns

**Files:**
- Modify: `src/projections/schemas.py` (the `SnakeCheatSheetSchema` class, ~line 986)
- Modify: `src/projections/draft/snake_cheat_sheet.py`
- Test: `tests/test_draft/test_snake_cheat_sheet.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_draft/test_snake_cheat_sheet.py`. Add a helper that attaches ADP to a VORP table (place after `_make_vorp_table`):

```python
def _attach_consensus_adp(vorp: pd.DataFrame, adp_by_gsis: dict[str, float]) -> pd.DataFrame:
    """Return a copy of a VORP table with a consensus_adp column (re-validated)."""
    out = vorp.copy()
    out["consensus_adp"] = pd.array(
        [adp_by_gsis.get(g) for g in out["gsis_id"]], dtype=pd.Float64Dtype()
    )
    return VorpTableSchema.validate(out)
```

Then the tests:

```python
def test_cheat_sheet_without_adp_leaves_new_columns_na() -> None:
    """Weekly-path VORP table (no consensus_adp) -> consensus_adp/adp_delta all-NA,
    every other column unchanged (backward compatible)."""
    vorp = _make_vorp_table({Position.QB: 4, Position.RB: 4})
    sheet = generate_snake_cheat_sheet(vorp, _make_config())
    assert sheet["consensus_adp"].isna().all()
    assert sheet["adp_delta"].isna().all()


def test_cheat_sheet_adp_delta_value_and_reach() -> None:
    """A late-ADP, high-VORP player is a 'value' (+delta); an early-ADP, low-VORP
    player is a 'reach' (-delta). Within position."""
    # Two QBs: best VORP (00-1000000) but LATE ADP -> value; worst VORP but EARLY ADP -> reach.
    vorp = _make_vorp_table({Position.QB: 2})
    # gsis ids from _make_vorp_table: 00-1000000 (higher vorp), 00-1000001 (lower vorp)
    adp = {"00-1000000": 50.0, "00-1000001": 5.0}
    sheet = generate_snake_cheat_sheet(_attach_consensus_adp(vorp, adp), _make_config())
    by_gsis = sheet.set_index("gsis_id")
    # 00-1000000: vorp_rank 1, adp_rank 2 -> delta +1 (value)
    assert by_gsis.loc["00-1000000", "adp_delta"] == 1
    # 00-1000001: vorp_rank 2, adp_rank 1 -> delta -1 (reach)
    assert by_gsis.loc["00-1000001", "adp_delta"] == -1


def test_cheat_sheet_null_adp_row_gets_null_delta() -> None:
    """A player missing consensus_adp gets null adp_delta but keeps its (passed-through)
    null consensus_adp; other players' deltas are unaffected (population isolation)."""
    vorp = _make_vorp_table({Position.WR: 3})
    # Only two of three WRs have an ADP.
    adp = {"00-3000000": 10.0, "00-3000001": 20.0}  # 00-3000002 has none
    sheet = generate_snake_cheat_sheet(_attach_consensus_adp(vorp, adp), _make_config())
    by_gsis = sheet.set_index("gsis_id")
    assert pd.isna(by_gsis.loc["00-3000002", "adp_delta"])
    assert pd.isna(by_gsis.loc["00-3000002", "consensus_adp"])
    # The two ADP-bearing WRs both rank 1/2 by both keys -> delta 0 each.
    assert by_gsis.loc["00-3000000", "adp_delta"] == 0
    assert by_gsis.loc["00-3000001", "adp_delta"] == 0


def test_cheat_sheet_with_adp_validates_schema() -> None:
    vorp = _make_vorp_table({Position.QB: 4, Position.RB: 6})
    adp = {g: float(i + 1) for i, g in enumerate(vorp["gsis_id"])}
    sheet = generate_snake_cheat_sheet(_attach_consensus_adp(vorp, adp), _make_config())
    SnakeCheatSheetSchema.validate(sheet)
    assert "consensus_adp" in sheet.columns
    assert "adp_delta" in sheet.columns
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_draft/test_snake_cheat_sheet.py -k "adp or new_columns or null_adp" -v`
Expected: FAIL — `KeyError`/missing `consensus_adp`/`adp_delta` columns (schema + function don't emit them yet).

- [ ] **Step 3: Add the two columns to `SnakeCheatSheetSchema`**

In `src/projections/schemas.py`, inside `class SnakeCheatSheetSchema`, add after `tier`:

```python
    tier: Series[pd.Int64Dtype] = pa.Field(ge=1, nullable=True)
    # Raw consensus ADP (market view) carried from the VORP table; NA on the weekly path.
    consensus_adp: Series[pd.Float64Dtype] = pa.Field(gt=0, nullable=True)
    # Within-position (ADP-rank - VORP-rank): positive = value, negative = reach. NA when
    # consensus_adp is NA (weekly path, or a player no source gave an ADP).
    adp_delta: Series[pd.Int64Dtype] = pa.Field(nullable=True)
```

- [ ] **Step 4: Compute the columns in `generate_snake_cheat_sheet`**

In `src/projections/draft/snake_cheat_sheet.py`, immediately after `df["is_in_pool"] = df["gsis_id"].isin(in_pool_ids)` (so the column rides through the later sorts), add the passthrough:

```python
    df["is_in_pool"] = df["gsis_id"].isin(in_pool_ids)

    # Carry consensus_adp through (NA when the input has none — the weekly path).
    if "consensus_adp" in vorp.columns:
        df["consensus_adp"] = vorp["consensus_adp"]
    else:
        df["consensus_adp"] = pd.array([pd.NA] * len(df), dtype=pd.Float64Dtype())
```

Then, after the tier block (`df["tier"] = tier_col`) and before the `display_name` block, add the `adp_delta` computation:

```python
    df["tier"] = tier_col

    # ADP-delta: within position, over the non-null-consensus_adp subset, two deterministic
    # gap-free integer ranks (sort + cumcount, like positional_rank — never Series.rank(),
    # whose 'average' method yields fractional ranks). delta = adp_rank - vorp_rank;
    # positive = value (market drafts later than VORP rank), negative = reach.
    adp_delta = pd.Series(pd.array([pd.NA] * len(df), dtype=pd.Int64Dtype()), index=df.index)
    sub = df[df["consensus_adp"].notna()]
    if not sub.empty:
        adp_rank = (
            sub.sort_values(["position", "consensus_adp", "gsis_id"], ascending=[True, True, True])
            .groupby("position", sort=False)
            .cumcount()
            + 1
        )
        vorp_rank = (
            sub.sort_values(["position", "vorp", "gsis_id"], ascending=[True, False, True])
            .groupby("position", sort=False)
            .cumcount()
            + 1
        )
        delta = (adp_rank - vorp_rank).astype(pd.Int64Dtype())  # index-aligned subtraction
        adp_delta.loc[delta.index] = delta
    df["adp_delta"] = adp_delta
```

(`adp_rank` and `vorp_rank` keep `df`'s index through `sort_values`, so the subtraction aligns per row regardless of sort order; `adp_delta.loc[...]` writes back by that index.)

The final `df = df[list(SnakeCheatSheetSchema.to_schema().columns)]` line already picks up the two new schema columns — no change needed there.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_draft/test_snake_cheat_sheet.py -v`
Expected: PASS — the new ADP tests AND all pre-existing cheat-sheet tests (the new columns are additive; existing assertions don't touch them).

- [ ] **Step 6: Commit**

```bash
git add src/projections/schemas.py src/projections/draft/snake_cheat_sheet.py tests/test_draft/test_snake_cheat_sheet.py
git commit -m "feat(draft): ADP-delta (consensus_adp + adp_delta) on the snake cheat sheet"
```

---

## Task 5: Cheat-sheet CLI — surface ADP-delta + consensus-fed integration test

**Files:**
- Modify: `scripts/generate_snake_cheat_sheet.py` (stdout summary only)
- Test: `tests/test_scripts/test_generate_snake_cheat_sheet_cli.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_scripts/test_generate_snake_cheat_sheet_cli.py` a test that a VORP parquet carrying `consensus_adp` flows through to a cheat sheet with both new columns. (Mirror the file's existing VORP-parquet fixture; this snippet assumes its existing `_run_cli` and a `tmp_path`-based VORP parquet writer — reuse whatever helper the file already defines to write a `VorpTableSchema` parquet, and add `consensus_adp` to it.)

```python
def test_cheat_sheet_cli_carries_adp_delta(tmp_path: Path) -> None:
    """A consensus-fed VORP parquet (with consensus_adp) -> cheat sheet with
    consensus_adp + adp_delta columns populated."""
    from projections.schemas import SnakeCheatSheetSchema, _PYARROW_STR

    vorp = pd.DataFrame(
        {
            "gsis_id": pd.array(
                ["00-1000000", "00-1000001", "00-2000000", "00-2000001"], dtype=_PYARROW_STR
            ),
            "position": pd.array(["QB", "QB", "RB", "RB"], dtype=_PYARROW_STR),
            "season_mean_fpts": pd.array([320.0, 300.0, 260.0, 250.0], dtype="float64"),
            "vorp": pd.array([80.0, 60.0, 30.0, 20.0], dtype="float64"),
            "replacement_fpts": pd.array([240.0, 240.0, 230.0, 230.0], dtype="float64"),
            "consensus_adp": pd.array([3.0, 8.0, 12.0, 20.0], dtype=pd.Float64Dtype()),
        }
    )
    vorp_path = tmp_path / "vorp.parquet"
    vorp.to_parquet(vorp_path, index=False)

    cfg_path = tmp_path / "league.json"
    cfg_path.write_text(
        json.dumps(
            {
                "name": "tiny", "n_teams": 2, "budget": 100, "min_bid": 1,
                "roster_slots": {"QB": 1, "RB": 1, "BENCH": 1}, "ruleset": "espn_ppr",
            }
        )
    )
    out_path = tmp_path / "sheet.parquet"
    proc = _run_cli(
        "--season", "2026",
        "--league-config", str(cfg_path),
        "--vorp-input", str(vorp_path),
        "--out", str(out_path),
    )
    assert proc.returncode == 0, proc.stderr
    sheet = pd.read_parquet(out_path)
    SnakeCheatSheetSchema.validate(sheet)
    assert sheet["consensus_adp"].notna().any()
    assert sheet["adp_delta"].notna().any()
```

(If the test file lacks `json` / `_run_cli` imports, add them mirroring `test_generate_vorp_table_cli.py`'s `_run_cli` — same `subprocess` + `PYTHONPATH=src` pattern, pointed at `scripts/generate_snake_cheat_sheet.py`.)

- [ ] **Step 2: Run the test to verify it fails or passes**

Run: `pytest tests/test_scripts/test_generate_snake_cheat_sheet_cli.py::test_cheat_sheet_cli_carries_adp_delta -v`
Expected: PASS already if Task 4 is complete (the CLI passes the parquet straight to the function). If it FAILS, it is because the CLI drops `consensus_adp` before calling the function — inspect `scripts/generate_snake_cheat_sheet.py` and ensure it reads the full parquet via `pd.read_parquet` (not a hardcoded column subset) before calling `generate_snake_cheat_sheet`.

- [ ] **Step 3: Add the ADP-delta line to the stdout summary**

In `scripts/generate_snake_cheat_sheet.py`, locate the per-position stdout summary loop (the "top-3 with tier-1 cliff" block). For each position, after the existing top-3 line, add the biggest value/reach by `adp_delta` when the column has data:

```python
        if "adp_delta" in pos_rows.columns and pos_rows["adp_delta"].notna().any():
            best_value = pos_rows.loc[pos_rows["adp_delta"].idxmax()]
            biggest_reach = pos_rows.loc[pos_rows["adp_delta"].idxmin()]
            print(
                f"      ADP value: {best_value['display_name']} "
                f"(delta {int(best_value['adp_delta']):+d}); "
                f"reach: {biggest_reach['display_name']} "
                f"(delta {int(biggest_reach['adp_delta']):+d})"
            )
```

(Match the indentation/`print` style of the surrounding summary code; `pos_rows` is whatever the existing loop calls its per-position slice.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_scripts/test_generate_snake_cheat_sheet_cli.py -v`
Expected: PASS (new test + all existing cheat-sheet CLI tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_snake_cheat_sheet.py tests/test_scripts/test_generate_snake_cheat_sheet_cli.py
git commit -m "feat(draft): surface ADP value/reach in cheat-sheet CLI summary"
```

---

## Task 6: Skill-only example league config

**Files:**
- Create: `configs/league_espn_ppr_12team_skill.json`
- Test: `tests/test_draft/test_consensus_source.py` (append a config-load guard)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_draft/test_consensus_source.py`:

```python
def test_skill_only_config_loads_without_k_dst() -> None:
    """The shipped skill-only config must parse as a LeagueConfig and contain no K/DST
    (consensus has no kicker/defense rows; generate_vorp_table would raise on them)."""
    from pathlib import Path

    from projections.draft.league_config import LeagueConfig
    from projections.schemas import RosterSlot

    repo_root = Path(__file__).resolve().parents[2]
    cfg = LeagueConfig.model_validate_json(
        (repo_root / "configs" / "league_espn_ppr_12team_skill.json").read_text()
    )
    assert RosterSlot.K not in cfg.roster_slots
    assert RosterSlot.DST not in cfg.roster_slots
    assert cfg.ruleset.name == "ESPN_PPR"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_draft/test_consensus_source.py::test_skill_only_config_loads_without_k_dst -v`
Expected: FAIL — `FileNotFoundError` (config doesn't exist yet).

- [ ] **Step 3: Create the config**

Create `configs/league_espn_ppr_12team_skill.json` (the `league_espn_ppr_12team.json` shape, minus `K`/`DST`; the freed roster spots are folded into bench so total roster size is unchanged):

```json
{
  "name": "espn_ppr_12team_2026_skill",
  "n_teams": 12,
  "budget": 200,
  "min_bid": 1,
  "roster_slots": {
    "QB": 1,
    "RB": 2,
    "WR": 3,
    "TE": 1,
    "FLEX": 1,
    "BENCH": 9
  },
  "ruleset": "espn_ppr"
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_draft/test_consensus_source.py::test_skill_only_config_loads_without_k_dst -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/league_espn_ppr_12team_skill.json tests/test_draft/test_consensus_source.py
git commit -m "feat(draft): skill-only example league config for the consensus path"
```

---

## Task 7: Auction CLI — regression guard for the extra column

**Files:**
- Test: `tests/test_scripts/test_generate_auction_values_cli.py`

- [ ] **Step 1: Write the test**

Add to `tests/test_scripts/test_generate_auction_values_cli.py` a test that a consensus-fed VORP parquet (with `consensus_adp`) produces the same auction output as the same parquet without it — proving the auction generator is blind to the new column. (Reuse the file's existing VORP-parquet + `_run_cli` helpers; this snippet writes its own minimal parquet to be self-contained.)

```python
def test_auction_cli_ignores_consensus_adp_column(tmp_path: Path) -> None:
    from projections.schemas import _PYARROW_STR

    base = pd.DataFrame(
        {
            "gsis_id": pd.array(
                ["00-1000000", "00-2000000", "00-2000001", "00-3000000"], dtype=_PYARROW_STR
            ),
            "position": pd.array(["QB", "RB", "RB", "WR"], dtype=_PYARROW_STR),
            "season_mean_fpts": pd.array([320.0, 260.0, 250.0, 240.0], dtype="float64"),
            "vorp": pd.array([80.0, 30.0, 20.0, 10.0], dtype="float64"),
            "replacement_fpts": pd.array([240.0, 230.0, 230.0, 230.0], dtype="float64"),
        }
    )
    cfg_path = tmp_path / "league.json"
    cfg_path.write_text(
        json.dumps(
            {
                "name": "tiny", "n_teams": 2, "budget": 100, "min_bid": 1,
                "roster_slots": {"QB": 1, "RB": 1, "WR": 1, "BENCH": 1}, "ruleset": "espn_ppr",
            }
        )
    )

    def _run(vorp: pd.DataFrame, tag: str) -> pd.DataFrame:
        vorp_path = tmp_path / f"vorp_{tag}.parquet"
        out_path = tmp_path / f"auction_{tag}.parquet"
        vorp.to_parquet(vorp_path, index=False)
        proc = _run_cli(
            "--season", "2026",
            "--league-config", str(cfg_path),
            "--vorp-input", str(vorp_path),
            "--out", str(out_path),
        )
        assert proc.returncode == 0, proc.stderr
        return pd.read_parquet(out_path)

    without = _run(base, "without")
    with_adp = base.copy()
    with_adp["consensus_adp"] = pd.array([3.0, 8.0, 20.0, 12.0], dtype=pd.Float64Dtype())
    got = _run(with_adp, "with")

    pd.testing.assert_frame_equal(
        without.sort_values("gsis_id").reset_index(drop=True),
        got[without.columns].sort_values("gsis_id").reset_index(drop=True),
    )
```

(If the file lacks `json` / `_run_cli`, add them mirroring `test_generate_vorp_table_cli.py`, pointed at `scripts/generate_auction_values.py`.)

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_scripts/test_generate_auction_values_cli.py::test_auction_cli_ignores_consensus_adp_column -v`
Expected: PASS — `_read_vorp` checks only a required-column subset and coerces 4 columns, so the extra `consensus_adp` is ignored and outputs match. If it FAILS, the auction CLI is mutating behavior on the extra column — investigate `_read_vorp` before changing the test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_scripts/test_generate_auction_values_cli.py
git commit -m "test(draft): guard auction CLI ignores consensus_adp column"
```

---

## Task 8: Full verification gates + status docs

**Files:**
- Modify: `project_management.md`, `TODO.md`

- [ ] **Step 1: Run the full verification suite**

Run each and fix any failure before proceeding (CLAUDE.md end-of-effort checklist):

```bash
pytest -v -k "draft or scripts or schemas or consensus"
mypy src tests
ruff check src tests
ruff format --check src tests
pytest -v -k "ingest or store or schemas"   # schema/store seam touched
```

Expected: all green. (`ruff format` drift → run `ruff format src tests` and re-commit.)

- [ ] **Step 2: Run the broader suite to confirm no regressions**

Run: `pytest -q`
Expected: same pass set as `origin/main` plus the new tests; no new failures. Note any pre-existing skips (network / missing feature parquet) in the final summary.

- [ ] **Step 3: Update `project_management.md`**

Add a top entry (above the External Consensus Blend entry) summarizing: Draft Hub re-pointed onto `ConsensusProjectionSchema` via `consensus_to_season_projections`; VORP CLI `--source consensus`; `consensus_adp` rides `VorpTableSchema` (optional col) to the cheat sheet's new `adp_delta`; skill-positions-only v1 (K/DST → TODO #10); the dropped-but-draftable warning; non-`espn_ppr` blocked on a consensus-layer `--ruleset` (logged). Reference this plan + the spec.

- [ ] **Step 4: Update `TODO.md`**

Under TODO #38, mark the Draft-Hub-consumption item done and add follow-ups: K/DST in the Draft Hub (TODO #10 link); confidence band (unblocked by the scraping slice); non-`espn_ppr` consensus rulesets (needs `refresh_consensus` ruleset selection); overall cross-position draft board.

- [ ] **Step 5: Commit**

```bash
git add project_management.md TODO.md
git commit -m "docs(pm,todo): Draft Hub on consensus shipped"
```

---

## Self-Review

**1. Spec coverage**

| Spec section | Task |
|---|---|
| §3.1 adapter (`has_points` filter, points→mean, degenerate band, n_weeks=17, model_id, single-asof/season assert, empty) | Task 1 |
| §3.2 `VorpTableSchema` optional `consensus_adp` | Task 2 |
| §3.3 VORP CLI `--source consensus` (read partition/latest, asof, dropped-draftable warning, ADP left-join, weekly-flag-only-for-weekly) | Task 3 |
| §3.4 cheat-sheet `consensus_adp` + deterministic `adp_delta`; backward-compat NA; population isolation | Task 4 |
| §3.4 CLI summary top value/reach | Task 5 |
| §3.5 skill-only config | Task 6 |
| §6 adapter tests | Task 1 |
| §6 `VorpTableSchema` round-trip (+legacy still validates) | Task 2 |
| §6 `generate_vorp_table` unchanged (regression) | Task 3 Step 4 (existing tests green) |
| §6 cheat-sheet ADP-delta tests | Task 4 |
| §6 VORP CLI consensus mode + dropped-warning tests | Task 3 |
| §6 cheat-sheet CLI consensus-fed test | Task 5 |
| §6 auction regression guard | Task 7 |
| §6 verification gates | Task 8 |

No spec requirement is unmapped. The live-2026 smoke (§6) is manual/post-merge, not a plan task — noted in Task 8's PM entry as the operator step.

**2. Placeholder scan:** No `TBD`/`TODO`/"handle edge cases"/"similar to". Every code step shows complete code; Tasks 5 and 7 explicitly flag the one reused-helper assumption (the existing CLI tests' `_run_cli`/parquet writer) and how to add it if absent.

**3. Type consistency:** `consensus_to_season_projections(consensus: pd.DataFrame) -> pd.DataFrame` is used identically in Task 1 and Task 3. `consensus_adp` is `pd.Float64Dtype` (gt=0, nullable) everywhere (Task 2 VorpTableSchema, Task 4 SnakeCheatSheetSchema, all test fixtures). `adp_delta` is `pd.Int64Dtype` nullable in both schema (Task 4) and tests. CLI flag names (`--source`, `--data-root`, `--asof`, `--weekly-projections`) match between Task 3 implementation and its tests. `_FULL_SEASON_WEEKS = 17` matches the `n_weeks == 17` assertion in Task 1's tests and the existing `_bulk_rows` convention.
