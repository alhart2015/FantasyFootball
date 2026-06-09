from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from projections.consensus.refresh import ConsensusError, refresh_consensus
from projections.schemas import ConsensusProjectionSchema
from projections.store import read_partition, write_partition

_STAT_COLS = [
    "passing_yards",
    "passing_tds",
    "interceptions",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "fumbles_lost",
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
    espn_stats = {c: 0.0 for c in _STAT_COLS} | {
        "receptions": 119.0,
        "receiving_yards": 1506.0,
        "receiving_tds": 8.0,
    }
    espn = {**base, "source": "ESPN", "gsis_id": "00-0036900", "adp": 4.8, **espn_stats}
    sleeper = {
        **base,
        "source": "SLEEPER",
        "gsis_id": "00-0036900",
        "adp": 3.4,
        **{c: pd.NA for c in _STAT_COLS},
    }
    return pd.DataFrame([espn, sleeper])


def _seed_raw(data_root: Path) -> None:
    write_partition(
        data_root / "raw",
        "external_projections",
        _raw_external(),
        season=2026,
        asof=date(2026, 6, 9),
    )


def test_refresh_writes_validated_consensus_snapshot(tmp_path: Path) -> None:
    _seed_raw(tmp_path)
    out_path = refresh_consensus(tmp_path, season=2026)
    assert out_path.exists()
    assert "asof=2026-06-09" in str(out_path)
    df = read_partition(
        tmp_path / "processed", "consensus_projections", season=2026, asof=date(2026, 6, 9)
    )
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


def test_missing_explicit_asof_raises_consensus_error(tmp_path: Path) -> None:
    # explicit-asof read path (read_partition) -> FileNotFoundError -> ConsensusError
    with pytest.raises(ConsensusError):
        refresh_consensus(tmp_path, season=2026, asof=date(2026, 6, 9))


def test_empty_raw_snapshot_raises_consensus_error(tmp_path: Path) -> None:
    # A 0-row raw snapshot must be refused before writing an empty consensus snapshot.
    empty = _raw_external().iloc[0:0]
    write_partition(
        tmp_path / "raw", "external_projections", empty, season=2026, asof=date(2026, 6, 9)
    )
    with pytest.raises(ConsensusError):
        refresh_consensus(tmp_path, season=2026, asof=date(2026, 6, 9))
