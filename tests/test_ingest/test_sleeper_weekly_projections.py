from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from projections.ingest import sleeper_weekly_projections as swp
from projections.ingest.sleeper_weekly_projections import parse_sleeper_weekly
from projections.schemas import ExternalProjectionWeeklySchema
from projections.store import read_partition, write_partition

_PAYLOAD: list[dict[str, Any]] = [
    {  # WR with a stat line
        "player_id": "4046",
        "player": {"first_name": "Ja'Marr", "last_name": "Chase", "position": "WR"},
        "stats": {"rec": 6.0, "rec_yd": 78.0, "rec_td": 0.5, "fum_lost": 0.1},
    },
    {  # K — non-skill, must be dropped
        "player_id": "999",
        "player": {"first_name": "Foot", "last_name": "Ball", "position": "K"},
        "stats": {"pts_ppr": 8.0},
    },
    {  # empty stats — dropped
        "player_id": "111",
        "player": {"first_name": "No", "last_name": "Proj", "position": "RB"},
        "stats": {},
    },
]


def test_parse_keeps_skill_with_stats_maps_fields() -> None:
    df = parse_sleeper_weekly(_PAYLOAD, season=2023, week=5)
    assert df["sleeper_id"].tolist() == ["4046"]
    row = df.iloc[0]
    assert row["position"] == "WR"
    assert row["receptions"] == 6.0
    assert row["receiving_yards"] == 78.0
    assert row["season"] == 2023 and row["week"] == 5
    # unmapped stat keys ignored; absent mapped keys -> NA (not 0)
    assert "passing_yards" in df.columns


def test_refresh_attaches_gsis_and_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # id_map: a real gsis for sleeper_id 4046; nothing for 7777 (-> placeholder)
    id_map = pd.DataFrame(
        {"gsis_id": ["00-0036900"], "sleeper_id": ["4046.0"]}  # float-stringified on purpose
    )
    write_partition(tmp_path / "raw", "id_map", id_map, season=None)

    payload: list[dict[str, Any]] = [
        {
            "player_id": "4046",
            "player": {"first_name": "JaMarr", "last_name": "Chase", "position": "WR"},
            "stats": {"rec": 6.0, "rec_yd": 78.0},
        },
        {
            "player_id": "7777",
            "player": {"first_name": "Rook", "last_name": "Ie", "position": "RB"},
            "stats": {"rush_yd": 40.0, "rush_td": 0.3},
        },
    ]
    monkeypatch.setattr(swp, "_fetch_sleeper_weekly", lambda season, week: payload)

    out_path = swp.refresh_sleeper_weekly(tmp_path / "raw", season=2023, week=5)
    assert out_path.exists()

    stored = read_partition(tmp_path / "raw", "sleeper_weekly_projections", season=2023, week=5)
    stored = ExternalProjectionWeeklySchema.validate(stored)
    by_name = stored.set_index("full_name")
    # the float-stringified id_map join must still match -> real gsis
    assert by_name.loc["JaMarr Chase", "gsis_id"] == "00-0036900"
    assert bool(by_name.loc["JaMarr Chase", "is_placeholder_gsis"]) is False
    # rookie with no id_map entry -> placeholder gsis (99-...)
    assert by_name.loc["Rook Ie", "gsis_id"].startswith("99-")
    assert bool(by_name.loc["Rook Ie", "is_placeholder_gsis"]) is True
