# --- D/ST (issue #166) -----------------------------------------------------


def test_espn_season_points_includes_defenses_when_given_the_dst_table() -> None:
    """Defenses live in their own table, so they have to be added explicitly. Without this,
    a rostered D/ST is valued by our pool but not by the market, and build_values drops any
    player only one side has an opinion about — so no trade could ever include one."""
    import pandas as pd
    import pytest

    from projections.midseason.valuation import espn_season_points
    from projections.schemas import DST_GSIS_IDS, ProjectionSource, Ruleset, Team

    ruleset = Ruleset(name="ESPN_HALF", dst_stat_points=(("99", 1.0), ("95", 2.0)))
    external = pd.DataFrame(
        {
            "source": [ProjectionSource.ESPN.value],
            "gsis_id": ["00-0036322"],
            **{c: [0.0] for c in ("passing_yards", "passing_tds", "interceptions")},
            **{c: [0.0] for c in ("rushing_yards", "rushing_tds", "receptions")},
            **{c: [0.0] for c in ("receiving_yards", "receiving_tds", "fumbles_lost")},
        }
    )
    dst = pd.DataFrame(
        {
            "gsis_id": [DST_GSIS_IDS[Team.HOU], DST_GSIS_IDS[Team.HOU]],
            "stat_id": ["99", "95"],
            "value": [3.0, 1.0],
        }
    )
    without = espn_season_points(external, ruleset)
    with_dst = espn_season_points(external, ruleset, dst=dst)

    assert DST_GSIS_IDS[Team.HOU] not in without
    assert with_dst[DST_GSIS_IDS[Team.HOU]] == pytest.approx(5.0)  # 3(1) + 1(2)
