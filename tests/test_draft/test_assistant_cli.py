"""End-to-end smoke test for the draft-assistant CLI core."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from projections.draft.assistant.cli import generate_recommendation, run
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, RecommendationSchema, RosterSlot, Ruleset
from projections.store import write_partition


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    cfg = LeagueConfig(
        name="t",
        n_teams=12,
        roster_slots={
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 5,
        },
        ruleset=Ruleset.espn_ppr(),
    )
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(cfg.model_dump_json())

    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"league_config": str(cfg_path), "my_slot": 7, "picks": []}))

    vorp = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000010", "00-0000020"], dtype=_PYARROW_STR),
            "position": pd.array(["RB", "WR"], dtype=_PYARROW_STR),
            "season_mean_fpts": [250.0, 252.0],
            "vorp": [50.0, 52.0],
            "replacement_fpts": [200.0, 200.0],
            "consensus_adp": pd.array([5.0, 7.0], dtype=pd.Float64Dtype()),
        }
    )
    vorp_path = tmp_path / "vorp.parquet"
    vorp.to_parquet(vorp_path, index=False)

    id_map = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000010", "00-0000020"], dtype=_PYARROW_STR),
            "espn_id": pd.array([pd.NA, pd.NA], dtype=_PYARROW_STR),
            "sleeper_id": pd.array([pd.NA, pd.NA], dtype=_PYARROW_STR),
            "pfr_id": pd.array([pd.NA, pd.NA], dtype=_PYARROW_STR),
            "full_name": pd.array(["RB One", "WR One"], dtype=_PYARROW_STR),
            "position": pd.array(["RB", "WR"], dtype=_PYARROW_STR),
            "team": pd.array([pd.NA, pd.NA], dtype=_PYARROW_STR),
        }
    )
    id_path = tmp_path / "id_map.parquet"
    id_map.to_parquet(id_path, index=False)
    return state_path, vorp_path, id_path


def test_generate_recommendation(tmp_path: Path) -> None:
    state_path, vorp_path, id_path = _setup(tmp_path)
    rec, id_map = generate_recommendation(
        state_path=state_path,
        vorp_path=vorp_path,
        id_map_path=id_path,
        strategy_name="now_or_never",
        sigma=None,
    )
    RecommendationSchema.validate(rec)
    assert set(rec["gsis_id"]) == {"00-0000010", "00-0000020"}
    assert set(id_map["gsis_id"]) == {"00-0000010", "00-0000020"}


def test_run_prints_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    state_path, vorp_path, id_path = _setup(tmp_path)
    code = run(
        [
            "--state",
            str(state_path),
            "--vorp-table",
            str(vorp_path),
            "--id-map",
            str(id_path),
            "--strategy",
            "raw_vorp",
            "--top",
            "5",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "WR One" in out and "RB One" in out


def _season_store(tmp_path: Path, gsis: list[str]) -> Path:
    """Write minimal weekly_stats(2022) + schedules(2026) + id_map; return data_root."""
    data_root = tmp_path / "data"
    raw = data_root / "raw"
    n = len(gsis)
    ws = pd.DataFrame(
        {
            "gsis_id": pd.array(gsis * 8, dtype=_PYARROW_STR),
            "season": [2022] * (n * 8),
            "week": [w for w in range(1, 9) for _ in range(n)],
            "position": pd.array(
                (["RB" if i % 2 else "WR" for i in range(n)]) * 8, dtype=_PYARROW_STR
            ),
        }
    )
    write_partition(raw, "weekly_stats", ws, season=2022)
    sched = pd.DataFrame(
        {
            "season": [2026, 2026],
            "week": [1, 2],
            "home_team": pd.array(["AA", "AA"], dtype=_PYARROW_STR),
            "away_team": pd.array(["BB", "BB"], dtype=_PYARROW_STR),
        }
    )
    write_partition(raw, "schedules", sched, season=2026)
    # Full IdMapSchema frame: the live CLI validates --id-map (we point it here), and
    # load_store_availability reads only gsis_id + team from the same file.
    na = pd.array([pd.NA] * n, dtype=_PYARROW_STR)
    pd.DataFrame(
        {
            "gsis_id": pd.array(gsis, dtype=_PYARROW_STR),
            "espn_id": na,
            "sleeper_id": na,
            "pfr_id": na,
            "full_name": pd.array([f"Player {i}" for i in range(n)], dtype=_PYARROW_STR),
            "position": pd.array(["RB" if i % 2 else "WR" for i in range(n)], dtype=_PYARROW_STR),
            "team": pd.array([pd.NA] * n, dtype=_PYARROW_STR),
        }
    ).to_parquet(raw / "id_map.parquet")
    return data_root


def _season_inputs(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    """Write a vorp pool + an empty-draft state file; return (state, vorp, gsis)."""
    n = 12
    gsis = [f"00-00000{i:02d}" for i in range(1, n + 1)]
    pool = pd.DataFrame(
        {
            "gsis_id": pd.array(gsis, dtype=_PYARROW_STR),
            "position": pd.array(["RB" if i % 2 else "WR" for i in range(n)], dtype=_PYARROW_STR),
            "season_mean_fpts": [float(200 - i) for i in range(n)],
            "vorp": [float(100 - i) for i in range(n)],
            "replacement_fpts": [100.0] * n,
            "consensus_adp": pd.array([float(i + 1) for i in range(n)], dtype=pd.Float64Dtype()),
        }
    )
    vorp_path = tmp_path / "vorp.parquet"
    pool.to_parquet(vorp_path)

    cfg = {
        "name": "t",
        "n_teams": 4,
        "roster_slots": {"RB": 1, "WR": 1, "FLEX": 1, "BENCH": 2},
        "ruleset": "espn_ppr",
    }
    cfg_path = tmp_path / "league.json"
    cfg_path.write_text(json.dumps(cfg))
    state = {"league_config": str(cfg_path), "my_slot": 1, "picks": []}
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))
    return state_path, vorp_path, gsis


def test_cli_season_value_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    state_path, vorp_path, gsis = _season_inputs(tmp_path)
    data_root = _season_store(tmp_path, gsis)
    id_map = data_root / "raw" / "id_map.parquet"
    code = run(
        [
            "--state",
            str(state_path),
            "--vorp-table",
            str(vorp_path),
            "--id-map",
            str(id_map),
            "--strategy",
            "season_value",
            "--season",
            "2026",
            "--n-sims",
            "15",
            "--data-root",
            str(data_root),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "PLAYER" in out  # the table header printed
    assert "Player " in out  # ...and at least one data row rendered (id_map full_name)


def test_cli_season_value_missing_weekly_stats_fails_loud(tmp_path: Path) -> None:
    # A valid --id-map (so _load_id_map passes) but an empty --data-root (no
    # weekly_stats) must hard-fail in availability loading, not silently fall back.
    state_path, vorp_path, gsis = _season_inputs(tmp_path)
    data_root = _season_store(tmp_path, gsis)  # writes a valid id_map under raw/
    id_map = data_root / "raw" / "id_map.parquet"
    empty_root = tmp_path / "empty"  # no raw/ partitions
    with pytest.raises(FileNotFoundError, match="weekly_stats"):
        run(
            [
                "--state",
                str(state_path),
                "--vorp-table",
                str(vorp_path),
                "--id-map",
                str(id_map),
                "--strategy",
                "season_value",
                "--season",
                "2026",
                "--n-sims",
                "15",
                "--data-root",
                str(empty_root),
            ]
        )


def test_cli_season_value_timing_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Mirror of test_cli_season_value_runs for the season_value_timing strategy.

    Exercises generate_recommendation's season_value_timing branch end-to-end:
    load_store_availability + SeasonValueTimingStrategy construction + recommend.
    A future constructor drift (missing arg, wrong type) will be caught here.
    """
    state_path, vorp_path, gsis = _season_inputs(tmp_path)
    data_root = _season_store(tmp_path, gsis)
    id_map = data_root / "raw" / "id_map.parquet"
    code = run(
        [
            "--state",
            str(state_path),
            "--vorp-table",
            str(vorp_path),
            "--id-map",
            str(id_map),
            "--strategy",
            "season_value_timing",
            "--season",
            "2026",
            "--n-sims",
            "15",
            "--data-root",
            str(data_root),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "PLAYER" in out  # the table header printed
    assert "Player " in out  # ...and at least one data row rendered (id_map full_name)


def test_cli_prefers_pool_full_name_over_id_map(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A placeholder-gsis rookie absent from id_map but named in the VORP pool prints its
    pool name; in-id_map players still print (id_map fallback when the pool has no name)."""
    state_path, _vorp_path, id_path = _setup(tmp_path)
    # A vorp pool with the two id_map players + a rookie absent from id_map, plus its own
    # full_name column (the consensus path now carries it — Task 2).
    vorp = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000010", "00-0000020", "99-8467088"], dtype=_PYARROW_STR),
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
            "--state",
            str(state_path),
            "--vorp-table",
            str(vorp_path),
            "--id-map",
            str(id_path),
            "--strategy",
            "raw_vorp",
            "--top",
            "5",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Rookie RB" in out  # named from the pool (absent from id_map)


def test_parse_args_accepts_season_value_timing() -> None:
    from projections.draft.assistant.cli import _parse_args

    args = _parse_args(
        ["--state", "s.json", "--vorp-table", "v.parquet", "--strategy", "season_value_timing"]
    )
    assert args.strategy == "season_value_timing"


def test_cli_parses_now_or_never_floored_flags() -> None:
    from projections.draft.assistant.cli import _parse_args

    args = _parse_args(
        [
            "--state",
            "s.json",
            "--vorp-table",
            "v.parquet",
            "--strategy",
            "now_or_never_floored",
            "--floor",
            "55",
            "--floor-weight",
            "2.5",
        ]
    )
    assert args.strategy == "now_or_never_floored"
    assert args.floor == 55.0
    assert args.floor_weight == 2.5
