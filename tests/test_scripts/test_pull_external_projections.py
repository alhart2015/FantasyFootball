import pull_external_projections as pull


def test_espn_stats_to_statline_dict_maps_ids_and_rounds_counts() -> None:
    # Chase-like projected dict (subset): rec yds 1335.25, receptions 104.87,
    # rec tds 8.24, rush yds 18.16, fumbles lost 0.99.
    raw = {
        "42": 1335.25,
        "53": 104.87,
        "43": 8.24,
        "24": 18.16,
        "72": 0.99,
        "99": 123.0,
    }  # 99 is an unmapped id and must be ignored
    out = pull.espn_stats_to_statline_dict(raw)
    assert out["receiving_yards"] == 1335.25  # float kept
    assert out["receptions"] == 105  # count rounded to int
    assert out["receiving_tds"] == 8  # rounds 8.24 -> 8
    assert out["rushing_yards"] == 18.16
    assert out["fumbles_lost"] == 1
    assert out["passing_yards"] == 0.0  # missing id -> 0
    assert "99" not in out and 123.0 not in out.values()


def _fake_espn_payload() -> dict[str, object]:
    # One QB with a 2024 projected season entry, a 2024 actual entry, ADP + rank,
    # plus a defense (defaultPositionId 16) that must be filtered out.
    return {
        "players": [
            {
                "player": {
                    "id": 3918298,
                    "fullName": "Josh Allen",
                    "defaultPositionId": 1,
                    "ownership": {"averageDraftPosition": 25.3},
                    "draftRanksByRankType": {"PPR": {"rank": 3}},
                    "stats": [
                        {
                            "seasonId": 2024,
                            "statSourceId": 1,
                            "statSplitTypeId": 0,
                            "appliedTotal": 313.6,
                            "stats": {
                                "3": 3752.62,
                                "4": 23.02,
                                "20": 12.44,
                                "24": 495.76,
                                "25": 8.57,
                                "72": 4.08,
                            },
                        },
                        {
                            "seasonId": 2024,
                            "statSourceId": 0,
                            "statSplitTypeId": 0,
                            "appliedTotal": 379.04,
                            "stats": {},
                        },
                        {
                            "seasonId": 2023,
                            "statSourceId": 1,
                            "statSplitTypeId": 0,
                            "appliedTotal": 999.0,
                            "stats": {},
                        },  # wrong season, ignore
                    ],
                }
            },
            {"player": {"id": 9999, "fullName": "Some DST", "defaultPositionId": 16, "stats": []}},
        ]
    }


def test_parse_espn_players_extracts_proj_actual_adp_rank_and_filters_positions() -> None:
    df = pull.parse_espn_players(_fake_espn_payload(), season=2024)
    assert list(df["espn_id"]) == ["3918298"]  # DST filtered out
    row = df.iloc[0]
    assert row["position"] == "QB"
    assert row["full_name"] == "Josh Allen"
    assert row["espn_adp"] == 25.3
    assert row["espn_pos_rank"] == 3
    assert row["espn_actual_applied_total"] == 379.04
    assert row["passing_yards"] == 3752.62  # from the 2024 PROJ entry
    assert row["passing_tds"] == 23  # rounded
    assert row["interceptions"] == 12


def test_parse_sleeper_adp_keeps_id_and_ppr_adp() -> None:
    from typing import Any

    payload: list[dict[str, Any]] = [
        {"player_id": "4046", "stats": {"adp_ppr": 1.2, "gp": 17.0}},
        {"player_id": "6794", "stats": {"adp_ppr": 14.5}},
        {"player_id": None, "stats": {"adp_ppr": 9.0}},  # no id -> dropped
    ]
    df = pull.parse_sleeper_adp(payload)
    assert list(df["sleeper_id"]) == ["4046", "6794"]
    assert list(df["sleeper_adp"]) == [1.2, 14.5]
