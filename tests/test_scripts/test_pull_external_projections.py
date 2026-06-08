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
