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
        {"player_id": "6794", "stats": {"adp_ppr": 14.5},
         "player": {"first_name": "A", "last_name": "B", "position": "WR"}},
        {"player_id": "99", "stats": {"adp_ppr": 1.0},
         "player": {"first_name": "K", "last_name": "K", "position": "K"}},  # kicker -> dropped
        {"player_id": None, "stats": {"adp_ppr": 9.0},
         "player": {"first_name": "x", "last_name": "y", "position": "RB"}},  # no id -> dropped
    ]
    df = ext.parse_sleeper_projections(payload)
    assert df["sleeper_id"].tolist() == ["6794"]
    r = df.iloc[0]
    assert r["full_name"] == "A B" and r["position"] == "WR" and r["sleeper_adp"] == 14.5
