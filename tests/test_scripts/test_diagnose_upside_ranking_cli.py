"""End-to-end CLI smoke for diagnose_upside_ranking.py.

Builds synthetic weekly parquet + distributions CSV + raw weekly_stats for 2 seasons
x 4 positions, runs the CLI, asserts the markdown report exists and has the
Phase-2-decision line.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from projections.schemas import Ruleset
from tests.test_aggregation.test_season import _build_weekly_row, _to_weekly_frame

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RULESET = Ruleset.espn_ppr()
_POSITIONS = ("QB", "RB", "WR", "TE")


def _gsis_id(pos: str, player_idx: int) -> str:
    """Build a gsis_id matching the canonical pattern (\\d{2}-\\d{7}).

    Position is encoded in the leading digits of the 7-digit suffix so the IDs
    remain unique across positions in the synthetic fixture.
    """
    pos_digit = _POSITIONS.index(pos)
    suffix = pos_digit * 1_000_000 + player_idx
    return f"00-{suffix:07d}"


def _write_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Returns (raw_root, reports_dir, out_path)."""
    raw_root = tmp_path / "data" / "raw"
    reports_dir = tmp_path / "reports"
    out_path = reports_dir / "upside_ranking_diagnostic.md"

    # Threshold seasons: write minimal weekly_stats for 2019..2023.
    for season in range(2019, 2024):
        partition = raw_root / "weekly_stats" / f"season={season}"
        partition.mkdir(parents=True, exist_ok=True)
        rows = []
        for pos_idx, pos in enumerate(_POSITIONS):
            for player_idx in range(10):
                ppr = 100.0 + pos_idx * 50 + player_idx * 25
                for week in range(1, 11):
                    rows.append(
                        {
                            "gsis_id": _gsis_id(pos, player_idx),
                            "season": season,
                            "week": week,
                            "position": pos,
                            "passing_yards": (ppr * 25 / 10) if pos == "QB" else 0.0,
                            "passing_tds": 0,
                            "interceptions": 0,
                            "rushing_yards": (ppr * 10 / 10) if pos == "RB" else 0.0,
                            "rushing_tds": 0,
                            "receptions": int(ppr / 10) if pos in ("WR", "TE") else 0,
                            "receiving_yards": 0.0,
                            "receiving_tds": 0,
                            "fumbles_lost": 0,
                        }
                    )
        pd.DataFrame(rows).to_parquet(partition / "part.parquet", index=False)

    # Eval seasons: 2024 + 2025 — actuals + weekly parquets + distributions CSVs.
    reports_dir.mkdir(parents=True, exist_ok=True)
    for season in (2024, 2025):
        # Actuals.
        partition = raw_root / "weekly_stats" / f"season={season}"
        partition.mkdir(parents=True, exist_ok=True)
        rows = []
        for pos_idx, pos in enumerate(_POSITIONS):
            for player_idx in range(8):
                ppr = 100.0 + pos_idx * 50 + player_idx * 30
                for week in range(1, 18):
                    rows.append(
                        {
                            "gsis_id": _gsis_id(pos, player_idx),
                            "season": season,
                            "week": week,
                            "position": pos,
                            "passing_yards": (ppr * 25 / 17) if pos == "QB" else 0.0,
                            "passing_tds": 0,
                            "interceptions": 0,
                            "rushing_yards": (ppr * 10 / 17) if pos == "RB" else 0.0,
                            "rushing_tds": 0,
                            "receptions": int(ppr / 17) if pos in ("WR", "TE") else 0,
                            "receiving_yards": 0.0,
                            "receiving_tds": 0,
                            "fumbles_lost": 0,
                        }
                    )
        pd.DataFrame(rows).to_parquet(partition / "part.parquet", index=False)

        # Weekly parquet: 2 players per position x 17 weeks via _build_weekly_row.
        weekly_rows = []
        for pos in _POSITIONS:
            for player_idx in range(2):
                for week in range(1, 18):
                    weekly_rows.append(
                        _build_weekly_row(
                            gsis_id=_gsis_id(pos, player_idx),
                            season=season,
                            week=week,
                            position=pos,
                            rec_yards_mean=50.0 + player_idx * 30,
                        )
                    )
        weekly_df = _to_weekly_frame(weekly_rows)
        weekly_df.to_parquet(
            reports_dir / f"season_projection_weekly_{season}.parquet", index=False
        )

        # Distributions CSV (mirror of what project_season would write).
        from projections.aggregation import aggregate_to_season

        summary = aggregate_to_season(weekly_df, ruleset=_RULESET, n_samples=1000)
        summary["full_name"] = "Synthetic " + summary["gsis_id"]
        summary["team"] = "TST"
        summary.to_csv(reports_dir / f"season_projection_distributions_{season}.csv", index=False)

    return raw_root, reports_dir, out_path


def test_diagnose_cli_writes_report_with_decision_line(tmp_path: Path) -> None:
    raw_root, reports_dir, out_path = _write_synthetic_inputs(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "diagnose_upside_ranking.py"),
            "--seasons",
            "2024",
            "2025",
            "--raw-root",
            str(raw_root),
            "--weekly-parquet-template",
            str(reports_dir / "season_projection_weekly_{season}.parquet"),
            "--distributions-csv-template",
            str(reports_dir / "season_projection_distributions_{season}.csv"),
            "--out",
            str(out_path),
            "--n-samples",
            "1000",
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"CLI failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    assert out_path.exists()
    body = out_path.read_text()
    assert "Phase 2 decision" in body
    assert any(verdict in body for verdict in ("Greenlight", "Marginal", "No greenlight"))
    table_csv = out_path.parent / "upside_ranking_diagnostic_table.csv"
    assert table_csv.exists()
    table = pd.read_csv(table_csv)
    assert set(table.columns) >= {
        "season",
        "position",
        "gsis_id",
        "actual_total",
        "actual_rank",
        "mean",
        "p90",
        "blend_70_30",
        "p_elite",
    }
