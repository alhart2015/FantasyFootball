"""Tests for scripts/generate_preset_vorp_tables.py (re-score guard + table shape)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# script import (scripts/ on sys.path via conftest)
from generate_preset_vorp_tables import resolve_espn_auction_dollars

from projections.draft.assistant.presets import get_preset
from projections.schemas import STAT_FIELDS, Ruleset, VorpTableSchema

_STAT_COLS = list(STAT_FIELDS)


def _espn_row(
    gsis_id: str, position: str, full_name: str, stats: dict[str, float]
) -> dict[str, object]:
    r: dict[str, object] = {
        "source": "ESPN",
        "source_player_id": f"ESPN-{gsis_id}",
        "gsis_id": gsis_id,
        "is_placeholder_gsis": False,
        "full_name": full_name,
        "position": position,
        "season": 2026,
        "asof": "2026-06-09",
        "adp": 50.0,
        "espn_draft_rank": float("nan"),
    }
    for c in _STAT_COLS:
        # float("nan") (not pd.NA) so the DataFrame infers float64 columns that pass
        # ExternalProjectionSchema directly — no per-test pd.to_numeric coercion needed.
        r[c] = stats.get(c, float("nan"))
    return r


def _synthetic_external() -> pd.DataFrame:
    """Enough skill players (single ESPN source) to fill a 10-team skill pool, plus one
    high-reception WR whose points must drop under half-PPR."""
    rows: list[dict[str, object]] = []
    counts = {"QB": 16, "RB": 36, "WR": 52, "TE": 16}
    prefix = {"QB": 1, "RB": 2, "WR": 3, "TE": 4}
    for pos, n in counts.items():
        for i in range(n):
            gsis = f"00-{prefix[pos]}{i:06d}"
            if pos == "QB":
                stats = {"passing_yards": 4000.0 - i * 50, "passing_tds": 28.0 - i * 0.3}
            elif pos == "RB":
                stats = {
                    "rushing_yards": 1100.0 - i * 20,
                    "rushing_tds": 9.0 - i * 0.1,
                    "receptions": 30.0 - i * 0.3,
                }
            elif pos == "TE":
                stats = {
                    "receptions": 60.0 - i,
                    "receiving_yards": 700.0 - i * 10,
                    "receiving_tds": 5.0 - i * 0.1,
                }
            else:  # WR
                stats = {
                    "receptions": 70.0 - i,
                    "receiving_yards": 1000.0 - i * 12,
                    "receiving_tds": 7.0 - i * 0.1,
                }
            rows.append(_espn_row(gsis, pos, f"{pos} Player {i}", stats))
    return pd.DataFrame(rows)


def test_half_scores_lower_than_ppr_for_pass_catcher() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "scripts"))
    from generate_preset_vorp_tables import build_preset_table

    external = _synthetic_external()
    ppr = build_preset_table(external, get_preset("ppr", 10)).set_index("gsis_id")
    half = build_preset_table(external, get_preset("half", 10)).set_index("gsis_id")
    std = build_preset_table(external, get_preset("std", 10)).set_index("gsis_id")
    wr = "00-3000000"  # 70 receptions
    assert ppr.loc[wr, "season_mean_fpts"] > half.loc[wr, "season_mean_fpts"]
    assert half.loc[wr, "season_mean_fpts"] > std.loc[wr, "season_mean_fpts"]


def test_preset_table_validates_and_carries_names_and_adp() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "scripts"))
    from generate_preset_vorp_tables import build_preset_table

    table = build_preset_table(_synthetic_external(), get_preset("half", 10))
    VorpTableSchema.validate(table)
    assert "full_name" in table.columns and table["full_name"].notna().any()
    assert "consensus_adp" in table.columns


def test_preset_table_carries_espn_auction_dollars() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "scripts"))
    from generate_preset_vorp_tables import build_preset_table

    external = _synthetic_external()
    # One player gets a crowd auction value that rounds cleanly; the rest stay all-NA so we can
    # assert both the populated and NA cases survive the merge + VorpTableSchema.validate.
    with_auction = "00-3000000"  # WR Player 0
    external.loc[external["gsis_id"] == with_auction, "espn_auction_value_avg"] = 58.67

    table = build_preset_table(external, get_preset("half", 12)).set_index("gsis_id")
    auction = table["espn_auction_dollars"]
    assert str(auction.dtype) == "Int64"
    assert auction.loc[with_auction] == 59  # 58.67 -> crowd average rounded
    assert pd.isna(auction.loc["00-3000001"])  # WR Player 1 has no ESPN auction value


def _frame(**cols: object) -> pd.DataFrame:
    return pd.DataFrame({k: pd.array(v, dtype="Float64") for k, v in cols.items()})


def test_resolve_prefers_crowd_when_positive() -> None:
    frame = _frame(
        espn_auction_value_avg=[58.67], espn_auction_value_ppr=[40.0], espn_auction_value_std=[30.0]
    )
    out = resolve_espn_auction_dollars(frame, Ruleset.espn_half())
    assert out.iloc[0] == 59  # 58.67 rounded
    assert str(out.dtype) == "Int64"


def test_resolve_falls_back_to_ppr_expert_for_half_when_crowd_absent() -> None:
    frame = _frame(
        espn_auction_value_avg=[pd.NA], espn_auction_value_ppr=[40.0], espn_auction_value_std=[30.0]
    )
    out = resolve_espn_auction_dollars(frame, Ruleset.espn_half())
    assert out.iloc[0] == 40


def test_resolve_falls_back_to_expert_when_crowd_zero() -> None:
    frame = _frame(
        espn_auction_value_avg=[0.0], espn_auction_value_ppr=[40.0], espn_auction_value_std=[30.0]
    )
    out = resolve_espn_auction_dollars(frame, Ruleset.espn_half())
    assert out.iloc[0] == 40  # > 0 guard rejects the 0 sentinel -> PPR expert


def test_resolve_uses_ppr_expert_for_ppr() -> None:
    frame = _frame(
        espn_auction_value_avg=[pd.NA], espn_auction_value_ppr=[40.0], espn_auction_value_std=[30.0]
    )
    out = resolve_espn_auction_dollars(frame, Ruleset.espn_ppr())
    assert out.iloc[0] == 40


def test_resolve_uses_std_expert_for_standard() -> None:
    frame = _frame(
        espn_auction_value_avg=[pd.NA], espn_auction_value_ppr=[40.0], espn_auction_value_std=[30.0]
    )
    out = resolve_espn_auction_dollars(frame, Ruleset.standard())
    assert out.iloc[0] == 30


def test_resolve_na_when_no_value() -> None:
    frame = _frame(
        espn_auction_value_avg=[pd.NA],
        espn_auction_value_ppr=[pd.NA],
        espn_auction_value_std=[pd.NA],
    )
    out = resolve_espn_auction_dollars(frame, Ruleset.espn_half())
    assert pd.isna(out.iloc[0])


def test_resolve_all_na_when_columns_absent() -> None:
    frame = pd.DataFrame({"gsis_id": pd.array(["00-0011111"], dtype="string[pyarrow]")})
    out = resolve_espn_auction_dollars(frame, Ruleset.espn_half())
    assert pd.isna(out.iloc[0])
    assert str(out.dtype) == "Int64"


def test_vorp_schema_espn_auction_dollars_optional() -> None:
    base = {
        "gsis_id": pd.array(["00-0011111"], dtype="string[pyarrow]"),
        "position": pd.array(["RB"], dtype="string[pyarrow]"),
        "season_mean_fpts": [200.0],
        "vorp": [50.0],
        "replacement_fpts": [150.0],
    }
    VorpTableSchema.validate(pd.DataFrame(base))  # weekly-path frame, no column -> validates
    withcol = pd.DataFrame({**base, "espn_auction_dollars": pd.array([57], dtype="Int64")})
    out = VorpTableSchema.validate(withcol)
    assert str(out["espn_auction_dollars"].dtype) == "Int64"


def test_build_preset_table_for_season_preset_validates() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "scripts"))
    from generate_preset_vorp_tables import build_preset_table

    table = build_preset_table(_synthetic_external(), get_preset("half", 12, season=2023))
    VorpTableSchema.validate(table)
    assert "espn_auction_dollars" in table.columns


def test_main_writes_per_season_tables_and_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "scripts"))
    import generate_preset_vorp_tables as gp

    from projections.draft.assistant import presets

    # The generator reads the external snapshot via read_latest_partition then runs
    # ExternalProjectionSchema.validate; feed the synthetic frame directly (no on-disk partition)
    # and redirect the table dir to tmp_path. _synthetic_external() already builds float64 stat
    # columns (NaN sentinels), so the frame validates as-is.
    external = _synthetic_external()
    external["season"] = 2023
    external["asof"] = "2023-01-01"
    monkeypatch.setattr(gp, "read_latest_partition", lambda *a, **k: external)
    monkeypatch.setattr(presets, "_table_dir", lambda season: tmp_path / f"vorp_{season}")
    # The 120-player synthetic fixture fills 10/12-team pools but NOT 16-team (FLEX can't fill);
    # restrict main's grid to half/12-team so it builds a buildable preset — the per-season write
    # path (dir + .league.json) is what this test verifies, not the full 9-preset grid.
    monkeypatch.setattr(gp, "SCORING_KEYS", ("half",))
    monkeypatch.setattr(gp, "TEAM_SIZES", (12,))

    rc = gp.main(["--season", "2023", "--data-root", str(tmp_path)])
    assert rc == 0
    tbl = tmp_path / "vorp_2023" / "half_12team.parquet"
    cfg = tmp_path / "vorp_2023" / "half_12team.league.json"
    assert tbl.exists() and cfg.exists()
    assert "espn_auction_dollars" in pd.read_parquet(tbl).columns
    assert json.loads(cfg.read_text())["name"] == "half_12team_2023"  # config carries the season
