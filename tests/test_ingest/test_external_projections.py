import pandas as pd

from projections.ingest import external_projections as ext


def test_parse_espn_players_extracts_statline_adp_rank():
    payload = {
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


def test_parse_sleeper_projections_keeps_name_position_adp_filters_to_skill():
    payload = [
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


def test_make_placeholder_gsis_is_deterministic_and_pattern_valid():
    import re

    a = ext._make_placeholder_gsis("ESPN", "5555")
    b = ext._make_placeholder_gsis("ESPN", "5555")
    assert a == b  # deterministic
    assert a.startswith("99-") and re.fullmatch(r"\d{2}-\d{7}", a)
    assert ext._make_placeholder_gsis("SLEEPER", "5555") != a  # source-scoped


def test_attach_gsis_id_real_for_matched_placeholder_for_rookie():
    df = pd.DataFrame({"espn_id": ["4374302", "9999999"], "x": [1, 2]})
    id_map = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036900"], dtype="string[pyarrow]"),
            "espn_id": pd.array(["4374302"], dtype="string[pyarrow]"),
        }
    )
    out = ext._attach_gsis_id(df, id_map, source="ESPN", id_col="espn_id")
    veteran = out[out["espn_id"] == "4374302"].iloc[0]
    rookie = out[out["espn_id"] == "9999999"].iloc[0]
    assert veteran["gsis_id"] == "00-0036900"
    assert not bool(veteran["is_placeholder_gsis"])
    assert bool(rookie["is_placeholder_gsis"])
    assert rookie["gsis_id"] == ext._make_placeholder_gsis("ESPN", "9999999")
    assert len(out) == 2  # no row multiplication


def _tiny_id_map():
    return pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036900"], dtype="string[pyarrow]"),
            "espn_id": pd.array(["4374302"], dtype="string[pyarrow]"),
            "sleeper_id": pd.array(["6794"], dtype="string[pyarrow]"),
        }
    )


def test_espn_to_canonical_is_schema_valid_with_stat_line():
    from datetime import date

    from projections.schemas import ExternalProjectionSchema

    espn = ext.parse_espn_players(
        {
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
        },
        season=2026,
    )
    out = ext._espn_to_canonical(espn, season=2026, asof=date(2026, 7, 15), id_map=_tiny_id_map())
    ExternalProjectionSchema.validate(out)
    r = out.iloc[0]
    assert r["source"] == "ESPN" and r["source_player_id"] == "4374302"
    assert r["gsis_id"] == "00-0036900" and r["asof"] == "2026-07-15"
    assert r["adp"] == 4.8 and r["receptions"] == 105.0


def test_sleeper_to_canonical_has_null_stat_line():
    from datetime import date

    from projections.schemas import ExternalProjectionSchema

    sl = ext.parse_sleeper_projections(
        [
            {
                "player_id": "6794",
                "stats": {"adp_ppr": 14.5},
                "player": {"first_name": "A", "last_name": "B", "position": "WR"},
            }
        ]
    )
    out = ext._sleeper_to_canonical(sl, season=2026, asof=date(2026, 7, 15), id_map=_tiny_id_map())
    ExternalProjectionSchema.validate(out)
    r = out.iloc[0]
    assert r["source"] == "SLEEPER" and r["adp"] == 14.5
    assert pd.isna(r["receptions"]) and pd.isna(r["espn_draft_rank"])
