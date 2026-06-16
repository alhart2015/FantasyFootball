"""End-to-end integration tests for `scripts/generate_vorp_table.py`."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from projections.distributions import (
    Distribution,
    ParametricGamma,
    ParametricNormal,
    pack_per_stat_params,
)
from projections.schemas import (
    _PYARROW_STR,
    DistributionFamily,
    Position,
    ProjectionWeeklySchema,
    RosterSlot,
    Ruleset,
    Stat,
)
from projections.scoring import derive_row_seed, score_distribution

_RULESET = Ruleset.espn_ppr()

_POSITION_ID_PREFIX = {Position.QB: 1, Position.RB: 2, Position.WR: 3, Position.TE: 4}


def _build_weekly_row(
    *,
    gsis_id: str,
    season: int,
    week: int,
    position: str,
    ruleset_name: str,
) -> dict[str, Any]:
    """Materialize one ProjectionWeeklySchema-valid row with SAMPLED_SUMMARY codec."""
    per_stat_dists: dict[Stat, Distribution] = {
        Stat.RECEIVING_YARDS: ParametricNormal(mean=50.0, std=18.0),
        Stat.RECEPTIONS: ParametricGamma(shape=4.0, scale=0.7),
    }
    blob = pack_per_stat_params(per_stat_dists)
    seed = derive_row_seed(gsis_id=gsis_id, season=season, week=week, ruleset_name=ruleset_name)
    points = score_distribution(per_stat_dists, _RULESET, n_samples=10_000, seed=seed)
    return {
        "gsis_id": gsis_id,
        "season": season,
        "week": week,
        "position": position,
        "team": "KC",
        "opponent": "BUF",
        "ruleset": ruleset_name,
        "family": DistributionFamily.SAMPLED_SUMMARY.value,
        "params": blob,
        "mean": points.mean(),
        "p10": points.quantile(0.1),
        "p50": points.quantile(0.5),
        "p90": points.quantile(0.9),
        "model_id": "test-model-v0",
        "generated_at": pd.Timestamp(datetime.now(UTC)).as_unit("us"),
    }


def _make_weekly_partition(
    partition_root: Path,
    season: int,
    ruleset_name: str,
    player_protos: list[dict[str, str]],
    weeks: tuple[int, ...] = (1, 2, 3),
) -> Path:
    """Write len(weeks) * len(players) rows under partition_root/season=YYYY/week=WW/."""
    for week in weeks:
        week_dir = partition_root / f"season={season}" / f"week={week}"
        week_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            _build_weekly_row(
                gsis_id=proto["gsis_id"],
                season=season,
                week=week,
                position=proto["position"],
                ruleset_name=ruleset_name,
            )
            for proto in player_protos
        ]
        df = pd.DataFrame(rows)
        for col in ("gsis_id", "team", "opponent", "ruleset", "family", "model_id", "position"):
            df[col] = df[col].astype(_PYARROW_STR)
        df = ProjectionWeeklySchema.validate(df)
        df.to_parquet(week_dir / "part.parquet", index=False)
    return partition_root


def _player_protos() -> list[dict[str, str]]:
    """One proto per draftable player (replicated across weeks by _make_weekly_partition)."""
    out: list[dict[str, str]] = []
    counts = {Position.QB: 20, Position.RB: 30, Position.WR: 30, Position.TE: 15}
    for pos, n in counts.items():
        for i in range(n):
            out.append(
                {
                    "gsis_id": f"00-{_POSITION_ID_PREFIX[pos]}{i:06d}",
                    "position": pos.value,
                }
            )
    return out


@pytest.fixture
def cli_inputs(tmp_path: Path) -> dict[str, Path]:
    cfg_path = tmp_path / "league.json"
    cfg_path.write_text(
        json.dumps(
            {
                "name": "tiny_test",
                "n_teams": 4,
                "budget": 100,
                "min_bid": 1,
                "roster_slots": {
                    RosterSlot.QB.value: 1,
                    RosterSlot.RB.value: 2,
                    RosterSlot.WR.value: 2,
                    RosterSlot.TE.value: 1,
                    RosterSlot.FLEX.value: 1,
                    RosterSlot.BENCH.value: 1,
                },
                "ruleset": "espn_ppr",
            }
        )
    )
    weekly_root = tmp_path / "weekly" / "ruleset=espn_ppr"
    # `ruleset_name` in the partition rows must equal `Ruleset.espn_ppr().name`
    # ("ESPN_PPR", uppercase) — aggregate_to_season's ruleset check is exact.
    _make_weekly_partition(
        partition_root=weekly_root,
        season=2026,
        ruleset_name=_RULESET.name,
        player_protos=_player_protos(),
    )
    return {
        "config": cfg_path,
        "weekly": weekly_root,
        "out_parquet": tmp_path / "vorp.parquet",
        "out_csv": tmp_path / "vorp.csv",
    }


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "generate_vorp_table.py"
    env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )


def test_cli_parquet_round_trip(cli_inputs: dict[str, Path]) -> None:
    from projections.schemas import VorpTableSchema

    proc = _run_cli(
        "--season",
        "2026",
        "--league-config",
        str(cli_inputs["config"]),
        "--weekly-projections",
        str(cli_inputs["weekly"]),
        "--out",
        str(cli_inputs["out_parquet"]),
    )
    assert proc.returncode == 0, proc.stderr
    out = pd.read_parquet(cli_inputs["out_parquet"])
    VorpTableSchema.validate(out)
    # row count == in-scope positions (QB+RB+WR+TE = 20+30+30+15 = 95)
    assert len(out) == 95


def test_cli_errors_when_config_requires_missing_position(
    cli_inputs: dict[str, Path], tmp_path: Path
) -> None:
    """LeagueConfig requires K but projection input has none → script exits non-zero
    with a clear error message from _select_pool's 'cannot fill' raise.

    Per spec §3.6 / §5.1 #17: VORP cannot produce a coherent output when a required
    position has zero input rows; the failure is explicit, not silent.
    """
    k_cfg_path = tmp_path / "league_with_k.json"
    k_cfg_path.write_text(
        json.dumps(
            {
                "name": "with_k",
                "n_teams": 4,
                "budget": 100,
                "min_bid": 1,
                "roster_slots": {
                    RosterSlot.QB.value: 1,
                    RosterSlot.RB.value: 2,
                    RosterSlot.WR.value: 2,
                    RosterSlot.TE.value: 1,
                    RosterSlot.FLEX.value: 1,
                    RosterSlot.K.value: 1,
                    RosterSlot.BENCH.value: 1,
                },
                "ruleset": "espn_ppr",
            }
        )
    )
    proc = _run_cli(
        "--season",
        "2026",
        "--league-config",
        str(k_cfg_path),
        "--weekly-projections",
        str(cli_inputs["weekly"]),
        "--out",
        str(cli_inputs["out_parquet"]),
    )
    assert proc.returncode != 0
    # The raise text from _take_top_n includes "cannot fill" and the slot label.
    combined = (proc.stderr + proc.stdout).lower()
    assert "cannot fill" in combined
    assert "k" in combined  # slot label "K"


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
            )
            rank += 1
    # One draftable (low ADP) player with NO points -> must be dropped AND warned about.
    rows.append(
        {
            "gsis_id": "00-3999999",
            "season": 2026,
            "asof": "2026-06-09",
            "full_name": "Hyped Rookie",
            "position": Position.WR.value,
            "consensus_adp": 5.0,
            "consensus_rank": 5,
            "n_adp_sources": 1,
            "has_points": False,
            "projected_points_ppr": None,
            "passing_yards": None,
            "passing_tds": None,
            "interceptions": None,
            "rushing_yards": None,
            "rushing_tds": None,
            "receptions": None,
            "receiving_yards": None,
            "receiving_tds": None,
            "fumbles_lost": None,
            "is_placeholder_gsis": True,
            "ruleset": "ESPN_PPR",
        }
    )
    return rows


def test_cli_consensus_mode_round_trip(cli_inputs: dict[str, Path], tmp_path: Path) -> None:
    from projections.schemas import VorpTableSchema

    data_root = tmp_path / "data"
    _make_consensus_partition(data_root, 2026, "2026-06-09", _consensus_rows())
    proc = _run_cli(
        "--source",
        "consensus",
        "--season",
        "2026",
        "--league-config",
        str(cli_inputs["config"]),
        "--data-root",
        str(data_root),
        "--out",
        str(cli_inputs["out_parquet"]),
    )
    assert proc.returncode == 0, proc.stderr
    out = pd.read_parquet(cli_inputs["out_parquet"])
    VorpTableSchema.validate(out)
    assert "consensus_adp" in out.columns
    assert out["consensus_adp"].notna().all()
    # full_name is carried from the consensus snapshot (in-pool players all have one).
    assert "full_name" in out.columns
    assert out["full_name"].notna().all()
    # A known QB maps to its consensus full_name ("QB Player 0").
    assert dict(zip(out["gsis_id"], out["full_name"], strict=False))["00-1000000"] == "QB Player 0"
    # The ADP-only "Hyped Rookie" is NOT in the VORP table (no points to rank on).
    assert "00-3999999" not in set(out["gsis_id"])
    # ...but the CLI WARNED about dropping a draftable player.
    assert "Hyped Rookie" in proc.stderr


def test_cli_consensus_mode_requires_data_root_not_weekly(cli_inputs: dict[str, Path]) -> None:
    """--source consensus must not require --weekly-projections."""
    # Missing --data-root falls back to default "data" (no partition there) -> a clear
    # FileNotFoundError, NOT an argparse 'weekly-projections required' error.
    proc = _run_cli(
        "--source",
        "consensus",
        "--season",
        "1999",
        "--league-config",
        str(cli_inputs["config"]),
        "--out",
        str(cli_inputs["out_parquet"]),
    )
    assert proc.returncode != 0
    combined = (proc.stderr + proc.stdout).lower()
    assert "weekly-projections" not in combined
    assert "no asof snapshots" in combined


def test_cli_errors_on_ruleset_mismatch(cli_inputs: dict[str, Path], tmp_path: Path) -> None:
    mismatch_cfg_path = tmp_path / "league_mismatch.json"
    mismatch_cfg_path.write_text(
        json.dumps(
            {
                "name": "mismatch",
                "n_teams": 4,
                "budget": 100,
                "min_bid": 1,
                "roster_slots": {
                    RosterSlot.QB.value: 1,
                    RosterSlot.RB.value: 2,
                    RosterSlot.WR.value: 2,
                    RosterSlot.TE.value: 1,
                    RosterSlot.FLEX.value: 1,
                    RosterSlot.BENCH.value: 1,
                },
                "ruleset": "standard",
            }
        )
    )
    proc = _run_cli(
        "--season",
        "2026",
        "--league-config",
        str(mismatch_cfg_path),
        "--weekly-projections",
        str(cli_inputs["weekly"]),
        "--out",
        str(cli_inputs["out_parquet"]),
    )
    assert proc.returncode != 0
    combined = (proc.stderr + proc.stdout).lower()
    assert "ruleset" in combined or "no rows" in combined


def test_cli_rejects_weekly_projections_in_consensus_mode(cli_inputs: dict[str, Path]) -> None:
    """A weekly-only flag passed in consensus mode is rejected, not silently ignored
    (otherwise the run would read the default data root instead of the intended path)."""
    proc = _run_cli(
        "--source",
        "consensus",
        "--season",
        "2026",
        "--league-config",
        str(cli_inputs["config"]),
        "--weekly-projections",
        str(cli_inputs["weekly"]),
        "--out",
        str(cli_inputs["out_parquet"]),
    )
    assert proc.returncode != 0
    assert "--weekly-projections is only valid with --source weekly" in proc.stderr


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


def test_cli_rejects_asof_in_weekly_mode(cli_inputs: dict[str, Path]) -> None:
    """A consensus-only flag passed in weekly mode is rejected, not silently ignored."""
    proc = _run_cli(
        "--source",
        "weekly",
        "--season",
        "2026",
        "--league-config",
        str(cli_inputs["config"]),
        "--weekly-projections",
        str(cli_inputs["weekly"]),
        "--asof",
        "2026-06-09",
        "--out",
        str(cli_inputs["out_parquet"]),
    )
    assert proc.returncode != 0
    assert "only valid with --source consensus" in proc.stderr
