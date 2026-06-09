from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from projections.ingest import external_projections as ext


def test_parse_espn_players_extracts_statline_adp_rank() -> None:
    payload: dict[str, Any] = {
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
            },
            {"player": {"id": 1, "defaultPositionId": 16, "stats": []}},  # DST -> dropped
        ]
    }
    df = ext.parse_espn_players(payload, season=2026)
    assert df["espn_id"].tolist() == ["4374302"]
    r = df.iloc[0]
    assert r["position"] == "WR" and r["full_name"] == "Ja'Marr Chase"
    assert r["espn_adp"] == 4.8 and r["espn_pos_rank"] == 20
    assert r["receptions"] == 105 and r["receiving_yards"] == 1335.0 and r["receiving_tds"] == 8


def test_parse_sleeper_projections_keeps_name_position_adp_filters_to_skill() -> None:
    payload: list[dict[str, Any]] = [
        {
            "player_id": "6794",
            "stats": {"adp_ppr": 14.5},
            "player": {"first_name": "A", "last_name": "B", "position": "WR"},
        },
        {
            "player_id": "99",
            "stats": {"adp_ppr": 1.0},
            "player": {"first_name": "K", "last_name": "K", "position": "K"},
        },  # kicker -> dropped
        {
            "player_id": None,
            "stats": {"adp_ppr": 9.0},
            "player": {"first_name": "x", "last_name": "y", "position": "RB"},
        },  # no id -> dropped
    ]
    df = ext.parse_sleeper_projections(payload)
    assert df["sleeper_id"].tolist() == ["6794"]
    r = df.iloc[0]
    assert r["full_name"] == "A B" and r["position"] == "WR" and r["sleeper_adp"] == 14.5


def test_make_placeholder_gsis_deterministic_pattern_valid_and_cross_source_stable() -> None:
    import re

    a = ext._make_placeholder_gsis("Jeremiyah Love", "RB")
    assert a == ext._make_placeholder_gsis("Jeremiyah Love", "RB")  # deterministic
    assert a.startswith("99-") and re.fullmatch(r"\d{2}-\d{7}", a)
    # suffix-insensitive: ESPN "Brian Thomas Jr." and Sleeper "Brian Thomas" reconcile
    assert ext._make_placeholder_gsis("Brian Thomas Jr.", "WR") == ext._make_placeholder_gsis(
        "Brian Thomas", "WR"
    )
    assert ext._make_placeholder_gsis("Some Other", "RB") != a  # distinct player -> distinct id


def test_placeholder_key_folds_accents_and_guards_degenerate_names() -> None:
    # Accents fold to ASCII, so the same player with/without accents reconciles across sources.
    assert ext._make_placeholder_gsis("José Álvarez", "WR") == ext._make_placeholder_gsis(
        "Jose Alvarez", "WR"
    )
    # Names that normalize to nothing (all-suffix / non-ASCII) must NOT collapse to one
    # position-only key — two distinct such players keep distinct placeholders.
    assert ext._make_placeholder_gsis("李明", "WR") != ext._make_placeholder_gsis("王伟", "WR")


def test_attach_gsis_id_real_for_matched_placeholder_for_rookie() -> None:
    df = pd.DataFrame(
        {
            "espn_id": ["4374302", "9999999"],
            "full_name": ["Ja'Marr Chase", "Jeremiyah Love"],
            "position": ["WR", "RB"],
        }
    )
    id_map = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036900"], dtype="string[pyarrow]"),
            "espn_id": pd.array(["4374302"], dtype="string[pyarrow]"),
        }
    )
    out = ext._attach_gsis_id(df, id_map, id_col="espn_id")
    veteran = out[out["espn_id"] == "4374302"].iloc[0]
    rookie = out[out["espn_id"] == "9999999"].iloc[0]
    assert veteran["gsis_id"] == "00-0036900"
    assert not bool(veteran["is_placeholder_gsis"])
    assert bool(rookie["is_placeholder_gsis"])
    assert rookie["gsis_id"] == ext._make_placeholder_gsis("Jeremiyah Love", "RB")
    assert len(out) == 2  # no row multiplication


def test_rookie_placeholder_reconciles_across_sources() -> None:
    # The SAME rookie, with different per-source ids, must get the SAME placeholder gsis so a
    # gsis_id join unifies ESPN's stat line with Sleeper's ADP instead of forking the player.
    empty_map = pd.DataFrame(
        {
            "gsis_id": pd.array([], dtype="string[pyarrow]"),
            "espn_id": pd.array([], dtype="string[pyarrow]"),
            "sleeper_id": pd.array([], dtype="string[pyarrow]"),
        }
    )
    espn = pd.DataFrame({"espn_id": ["111"], "full_name": ["Jeremiyah Love"], "position": ["RB"]})
    sleeper = pd.DataFrame(
        {"sleeper_id": ["222"], "full_name": ["Jeremiyah Love"], "position": ["RB"]}
    )
    e = ext._attach_gsis_id(espn, empty_map, id_col="espn_id")
    s = ext._attach_gsis_id(sleeper, empty_map, id_col="sleeper_id")
    assert bool(e.iloc[0]["is_placeholder_gsis"]) and bool(s.iloc[0]["is_placeholder_gsis"])
    assert e.iloc[0]["gsis_id"] == s.iloc[0]["gsis_id"]


def test_espn_statline_preserves_fractional_counts() -> None:
    # ESPN projects fractional counts (8.4 TDs); ingest stores them raw, never rounded.
    payload = {
        "players": [
            {
                "player": {
                    "id": 5,
                    "fullName": "Frac Guy",
                    "defaultPositionId": 4,
                    "stats": [
                        {
                            "seasonId": 2026,
                            "statSourceId": 1,
                            "statSplitTypeId": 0,
                            "stats": {"43": 8.4, "53": 6.6},
                        }
                    ],
                }
            }
        ]
    }
    r = ext.parse_espn_players(payload, season=2026).iloc[0]
    assert r["receiving_tds"] == 8.4 and r["receptions"] == 6.6


def _tiny_id_map() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036900"], dtype="string[pyarrow]"),
            "espn_id": pd.array(["4374302"], dtype="string[pyarrow]"),
            "sleeper_id": pd.array(["6794"], dtype="string[pyarrow]"),
        }
    )


def test_espn_to_canonical_is_schema_valid_with_stat_line() -> None:
    from datetime import date

    from projections.schemas import ExternalProjectionSchema, ProjectionSource

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
    espn = ext.parse_espn_players(espn_payload, season=2026)
    out = ext._to_canonical(
        espn,
        source=ProjectionSource.ESPN,
        id_col="espn_id",
        adp_col="espn_adp",
        rank_col="espn_pos_rank",
        has_stats=True,
        season=2026,
        asof=date(2026, 7, 15),
        id_map=_tiny_id_map(),
    )
    ExternalProjectionSchema.validate(out)
    r = out.iloc[0]
    assert r["source"] == "ESPN" and r["source_player_id"] == "4374302"
    assert r["gsis_id"] == "00-0036900" and r["asof"] == "2026-07-15"
    assert r["adp"] == 4.8 and r["receptions"] == 105.0


def test_sleeper_to_canonical_has_null_stat_line() -> None:
    from datetime import date

    from projections.schemas import ExternalProjectionSchema, ProjectionSource

    sl = ext.parse_sleeper_projections(
        [
            {
                "player_id": "6794",
                "stats": {"adp_ppr": 14.5},
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
        has_stats=False,
        season=2026,
        asof=date(2026, 7, 15),
        id_map=_tiny_id_map(),
    )
    ExternalProjectionSchema.validate(out)
    r = out.iloc[0]
    assert r["source"] == "SLEEPER" and r["adp"] == 14.5
    assert pd.isna(r["receptions"]) and pd.isna(r["espn_draft_rank"])


def test_refresh_writes_validated_asof_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import date

    from projections.schemas import ExternalProjectionSchema
    from projections.store import read_latest_partition

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
            "stats": {"adp_ppr": 14.5},
            "player": {"first_name": "A", "last_name": "B", "position": "WR"},
        }
    ]
    monkeypatch.setattr(ext, "fetch_espn", lambda season: espn_payload)
    monkeypatch.setattr(ext, "fetch_sleeper_season", lambda season: sleeper_payload)
    # id_map lives at <data_root>/raw/id_map.parquet
    (tmp_path / "raw").mkdir(parents=True)
    pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036900"], dtype="string[pyarrow]"),
            "espn_id": pd.array(["4374302"], dtype="string[pyarrow]"),
            "sleeper_id": pd.array(["6794"], dtype="string[pyarrow]"),
        }
    ).to_parquet(tmp_path / "raw" / "id_map.parquet", index=False)

    ext.refresh_external_projections(tmp_path, season=2026, asof=date(2026, 7, 15))
    latest = read_latest_partition(tmp_path / "raw", "external_projections", season=2026)
    ExternalProjectionSchema.validate(latest)
    assert set(latest["source"]) == {"ESPN", "SLEEPER"}
    assert (latest["gsis_id"] == "00-0036900").sum() == 2  # both sources crosswalked the veteran


def test_parse_espn_last_proj_entry_wins_when_duplicated() -> None:
    payload = {
        "players": [
            {
                "player": {
                    "id": 1,
                    "fullName": "Dup Guy",
                    "defaultPositionId": 3,
                    "ownership": {"averageDraftPosition": 5.0},
                    "draftRanksByRankType": {"PPR": {"rank": 3}},
                    "stats": [
                        {
                            "seasonId": 2026,
                            "statSourceId": 1,
                            "statSplitTypeId": 0,
                            "stats": {"42": 800.0},
                        },
                        {
                            "seasonId": 2026,
                            "statSourceId": 1,
                            "statSplitTypeId": 0,
                            "stats": {"42": 1200.0},
                        },  # last wins
                    ],
                }
            }
        ]
    }
    df = ext.parse_espn_players(payload, season=2026)
    assert df.iloc[0]["receiving_yards"] == 1200.0


def test_refresh_refuses_empty_pull(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ext, "fetch_espn", lambda season: {"players": []})
    monkeypatch.setattr(ext, "fetch_sleeper_season", lambda season: [])
    (tmp_path / "raw").mkdir(parents=True)
    pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036900"], dtype="string[pyarrow]"),
            "espn_id": pd.array(["x"], dtype="string[pyarrow]"),
            "sleeper_id": pd.array(["y"], dtype="string[pyarrow]"),
        }
    ).to_parquet(tmp_path / "raw" / "id_map.parquet", index=False)
    with pytest.raises(ext.ExternalProjectionError):
        ext.refresh_external_projections(tmp_path, season=2026)


def test_refresh_writes_single_source_when_other_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import date

    from projections.store import read_latest_partition

    # ESPN empty, Sleeper good -> write a Sleeper-only snapshot rather than discard the pull.
    monkeypatch.setattr(ext, "fetch_espn", lambda season: {"players": []})
    monkeypatch.setattr(
        ext,
        "fetch_sleeper_season",
        lambda season: [
            {
                "player_id": "6794",
                "stats": {"adp_ppr": 14.5},
                "player": {"first_name": "A", "last_name": "B", "position": "WR"},
            }
        ],
    )
    (tmp_path / "raw").mkdir(parents=True)
    pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036900"], dtype="string[pyarrow]"),
            "espn_id": pd.array(["4374302"], dtype="string[pyarrow]"),
            "sleeper_id": pd.array(["6794"], dtype="string[pyarrow]"),
        }
    ).to_parquet(tmp_path / "raw" / "id_map.parquet", index=False)

    ext.refresh_external_projections(tmp_path, season=2026, asof=date(2026, 7, 15))
    latest = read_latest_partition(tmp_path / "raw", "external_projections", season=2026)
    assert set(latest["source"]) == {"SLEEPER"}


def test_parse_espn_drops_player_with_null_full_name() -> None:
    payload = {
        "players": [
            {
                "player": {
                    "id": 7,
                    "defaultPositionId": 3,
                    "stats": [
                        {
                            "seasonId": 2026,
                            "statSourceId": 1,
                            "statSplitTypeId": 0,
                            "stats": {"42": 900.0},
                        }
                    ],
                }
            }
        ]
    }
    assert ext.parse_espn_players(payload, season=2026).empty  # no fullName -> dropped


def test_parse_sleeper_drops_empty_name() -> None:
    payload = [
        {
            "player_id": "1",
            "stats": {"adp_ppr": 5.0},
            "player": {"first_name": None, "last_name": None, "position": "WR"},
        }
    ]
    assert ext.parse_sleeper_projections(payload).empty
