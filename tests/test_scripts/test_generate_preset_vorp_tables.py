"""Tests for scripts/generate_preset_vorp_tables.py (re-score guard + table shape)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from projections.schemas import STAT_FIELDS, VorpTableSchema

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
        "espn_draft_rank": pd.NA,
    }
    for c in _STAT_COLS:
        r[c] = stats.get(c, pd.NA)
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
    ppr = build_preset_table(external, "ppr", 10).set_index("gsis_id")
    half = build_preset_table(external, "half", 10).set_index("gsis_id")
    std = build_preset_table(external, "std", 10).set_index("gsis_id")
    wr = "00-3000000"  # 70 receptions
    assert ppr.loc[wr, "season_mean_fpts"] > half.loc[wr, "season_mean_fpts"]
    assert half.loc[wr, "season_mean_fpts"] > std.loc[wr, "season_mean_fpts"]


def test_preset_table_validates_and_carries_names_and_adp() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "scripts"))
    from generate_preset_vorp_tables import build_preset_table

    table = build_preset_table(_synthetic_external(), "half", 10)
    VorpTableSchema.validate(table)
    assert "full_name" in table.columns and table["full_name"].notna().any()
    assert "consensus_adp" in table.columns
