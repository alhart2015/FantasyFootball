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


def test_parse_espn_players_normalizes_undrafted_adp_zero_to_null() -> None:
    # ESPN encodes "undrafted / no draft data" as ADP 0; the parser must store null, not 0.0.
    payload: dict[str, Any] = {
        "players": [
            {
                "player": {
                    "id": 999,
                    "fullName": "Deep Roster Guy",
                    "defaultPositionId": 2,
                    "ownership": {"averageDraftPosition": 0.0},
                    "stats": [
                        {
                            "seasonId": 2026,
                            "statSourceId": 1,
                            "statSplitTypeId": 0,
                            "stats": {"24": 300.0},
                        }
                    ],
                }
            }
        ]
    }
    df = ext.parse_espn_players(payload, season=2026)
    assert len(df) == 1
    assert pd.isna(df.iloc[0]["espn_adp"])  # 0.0 -> None, not stored as a sentinel


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


def test_sleeper_stats_to_statline_maps_raw_no_rounding() -> None:
    # Real Sleeper WR line (fractional rec_td); maps to canonical fields, raw, unrounded.
    stats = {
        "rec": 105.0,
        "rec_yd": 1501.0,
        "rec_td": 8.4,
        "rush_yd": 44.0,
        "fum_lost": 1.0,
        "gp": 18.0,
        "cmp_pct": 0.0,
        "bonus_rec_wr": 105.0,
        "adp_ppr": 3.4,
        "rec_fd": 150.1,
        "rec_0_4": 21.0,
    }
    out = ext._sleeper_stats_to_statline(stats)
    assert out is not None
    assert out["receptions"] == 105.0
    assert out["receiving_yards"] == 1501.0
    assert out["receiving_tds"] == 8.4  # raw, not rounded to 8
    assert out["rushing_yards"] == 44.0
    assert out["fumbles_lost"] == 1.0
    # unmapped/absent canonical fields default to 0.0
    assert out["passing_yards"] == 0.0 and out["rushing_tds"] == 0.0
    # non-mapped Sleeper keys are ignored (no stray keys)
    from projections.schemas import STAT_FIELDS

    assert set(out) == set(STAT_FIELDS)


def test_sleeper_stats_to_statline_qb_fields() -> None:
    stats = {"pass_yd": 4193.0, "pass_td": 32.0, "pass_int": 14.0, "rush_yd": 599.0, "rush_td": 6.0}
    out = ext._sleeper_stats_to_statline(stats)
    assert out is not None
    assert out["passing_yards"] == 4193.0 and out["passing_tds"] == 32.0
    assert (
        out["interceptions"] == 14.0 and out["rushing_yards"] == 599.0 and out["rushing_tds"] == 6.0
    )


def test_sleeper_stats_to_statline_none_when_adp_only() -> None:
    # A Sleeper row with only ADP (no mapped stat keys) has no projection -> None.
    assert ext._sleeper_stats_to_statline({"adp_ppr": 14.5, "adp_std": 20.0, "gp": 0.0}) is None


def test_parse_sleeper_projections_extracts_stat_line() -> None:
    payload: list[dict[str, Any]] = [
        {
            "player_id": "6794",
            "stats": {"adp_ppr": 14.5, "rec": 105.0, "rec_yd": 1501.0, "rec_td": 8.0},
            "player": {"first_name": "A", "last_name": "B", "position": "WR"},
        }
    ]
    df = ext.parse_sleeper_projections(payload)
    r = df.iloc[0]
    assert r["sleeper_id"] == "6794" and r["sleeper_adp"] == 14.5
    assert r["receptions"] == 105.0 and r["receiving_yards"] == 1501.0 and r["receiving_tds"] == 8.0
    assert r["passing_yards"] == 0.0  # absent mapped field defaults to 0.0


def test_parse_sleeper_adp_only_row_has_na_stats_but_columns_present() -> None:
    # All-ADP-only payload: stat columns must still exist (NA) so the has_stats path can read them.
    from projections.schemas import STAT_FIELDS

    payload: list[dict[str, Any]] = [
        {
            "player_id": "9",
            "stats": {"adp_ppr": 50.0},
            "player": {"first_name": "Deep", "last_name": "Guy", "position": "RB"},
        }
    ]
    df = ext.parse_sleeper_projections(payload)
    assert all(f in df.columns for f in STAT_FIELDS)
    assert all(pd.isna(df.iloc[0][f]) for f in STAT_FIELDS)


def test_sleeper_to_canonical_carries_stat_line() -> None:
    from datetime import date

    from projections.schemas import ExternalProjectionSchema, ProjectionSource

    sl = ext.parse_sleeper_projections(
        [
            {
                "player_id": "6794",
                "stats": {"adp_ppr": 14.5, "rec": 100.0, "rec_yd": 1400.0, "rec_td": 9.0},
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
        has_stats=True,
        season=2026,
        asof=date(2026, 7, 15),
        id_map=_tiny_id_map(),
    )
    ExternalProjectionSchema.validate(out)
    r = out.iloc[0]
    assert r["source"] == "SLEEPER" and r["adp"] == 14.5
    assert r["receptions"] == 100.0 and r["receiving_yards"] == 1400.0
    assert pd.isna(r["espn_draft_rank"])  # Sleeper has no draft rank


def test_refresh_emits_no_all_na_concat_futurewarning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import warnings
    from datetime import date

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
            "stats": {"adp_ppr": 14.5, "rec": 100.0, "rec_yd": 1400.0, "rec_td": 9.0},
            "player": {"first_name": "A", "last_name": "B", "position": "WR"},
        }
    ]
    monkeypatch.setattr(ext, "fetch_espn", lambda season: espn_payload)
    monkeypatch.setattr(ext, "fetch_sleeper_season", lambda season: sleeper_payload)
    (tmp_path / "raw").mkdir(parents=True)
    pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036900", "00-0011111"], dtype="string[pyarrow]"),
            "espn_id": pd.array(["4374302", "x"], dtype="string[pyarrow]"),
            "sleeper_id": pd.array(["y", "6794"], dtype="string[pyarrow]"),
        }
    ).to_parquet(tmp_path / "raw" / "id_map.parquet", index=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        ext.refresh_external_projections(tmp_path, season=2026, asof=date(2026, 7, 15))


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


def _espn_player(
    pid: int,
    name: str,
    pos_id: int,
    *,
    adp: float = 5.0,
    auction_avg: float | None = None,
    ppr_av: float | None = None,
    std_av: float | None = None,
) -> dict[str, Any]:
    """Minimal kona_player_info entry for one player with a projection."""
    ownership: dict[str, Any] = {"averageDraftPosition": adp}
    if auction_avg is not None:
        ownership["auctionValueAverage"] = auction_avg
    ranks: dict[str, dict[str, Any]] = {"PPR": {"rank": 1}, "STANDARD": {"rank": 1}}
    if ppr_av is not None:
        ranks["PPR"]["auctionValue"] = ppr_av
    if std_av is not None:
        ranks["STANDARD"]["auctionValue"] = std_av
    return {
        "player": {
            "id": pid,
            "fullName": name,
            "defaultPositionId": pos_id,  # 2 == RB
            "stats": [
                {"seasonId": 2026, "statSplitTypeId": 0, "statSourceId": 1, "stats": {"24": 1000.0}}
            ],
            "ownership": ownership,
            "draftRanksByRankType": ranks,
        }
    }


def test_parse_espn_extracts_auction_values() -> None:
    payload = {
        "players": [_espn_player(1, "Crowd Guy", 2, auction_avg=58.67, ppr_av=57, std_av=55)]
    }
    df = ext.parse_espn_players(payload, 2026)
    row = df.iloc[0]
    assert row["espn_auction_value_avg"] == 58.67
    assert row["espn_auction_value_ppr"] == 57
    assert row["espn_auction_value_std"] == 55


def test_parse_espn_auction_values_non_positive_and_missing_become_none() -> None:
    payload = {
        "players": [
            _espn_player(1, "Zero Crowd", 2, auction_avg=0, ppr_av=0, std_av=0),
            _espn_player(2, "No Auction Keys", 2),  # no auction_avg / auctionValue at all
        ]
    }
    df = ext.parse_espn_players(payload, 2026).set_index("espn_id")
    for col in ("espn_auction_value_avg", "espn_auction_value_ppr", "espn_auction_value_std"):
        assert pd.isna(df.loc["1", col])  # <=0 normalized to None  (pd.isna: robust to dtype)
        assert pd.isna(df.loc["2", col])  # missing key -> None, no crash


def _external_row(**overrides: Any) -> pd.DataFrame:
    """One-row ExternalProjectionSchema-shaped frame; overrides add/replace columns."""
    cols: dict[str, Any] = {
        "source": pd.array(["ESPN"], dtype="string[pyarrow]"),  # isin(['ESPN','SLEEPER'])
        "source_player_id": pd.array(["1"], dtype="string[pyarrow]"),
        "gsis_id": pd.array(["00-0011111"], dtype="string[pyarrow]"),
        "is_placeholder_gsis": [False],
        "full_name": pd.array(["E"], dtype="string[pyarrow]"),
        "position": pd.array(["RB"], dtype="string[pyarrow]"),
        "season": [2026],
        "asof": pd.array(["2026-06-09"], dtype="string[pyarrow]"),
        "adp": pd.array([5.0], dtype="Float64"),
        "espn_draft_rank": pd.array([pd.NA], dtype="Float64"),
        **{
            f: pd.array([pd.NA], dtype="Float64")
            for f in (
                "passing_yards",
                "passing_tds",
                "interceptions",
                "rushing_yards",
                "rushing_tds",
                "receptions",
                "receiving_yards",
                "receiving_tds",
                "fumbles_lost",
            )
        },
    }
    cols.update(overrides)
    return pd.DataFrame(cols)


def test_external_schema_validates_without_auction_columns() -> None:
    # A stale-style frame lacking the new columns must still validate (Optional).
    from projections.schemas import ExternalProjectionSchema

    df = _external_row(source=pd.array(["SLEEPER"], dtype="string[pyarrow]"))
    out = ExternalProjectionSchema.validate(df)  # must not raise
    assert "espn_auction_value_avg" not in out.columns  # absent -> stays absent, no fabricate


def test_external_schema_auction_columns_are_float64() -> None:
    from projections.schemas import ExternalProjectionSchema

    df = _external_row(
        espn_auction_value_avg=pd.array([58.67], dtype="Float64"),
        espn_auction_value_ppr=pd.array([57.0], dtype="Float64"),
        espn_auction_value_std=pd.array([55.0], dtype="Float64"),
    )
    out = ExternalProjectionSchema.validate(df)
    assert str(out["espn_auction_value_avg"].dtype) == "Float64"


def test_to_canonical_sleeper_auction_columns_are_float64_na() -> None:
    # _to_canonical null-fills the ESPN-only auction columns for a Sleeper frame; they must land
    # as Float64/pd.NA (not float64/NaN — the CLAUDE.md dtype-regression trap, spec Testing).
    from datetime import date

    from projections.ingest.external_projections import _to_canonical
    from projections.schemas import STAT_FIELDS, ProjectionSource

    sleeper = pd.DataFrame(
        {
            "sleeper_id": ["s1"],
            "full_name": ["Sleeper Guy"],
            "position": ["RB"],
            "sleeper_adp": [12.0],
            **{f: [100.0] for f in STAT_FIELDS},
        }
    )
    id_map = pd.DataFrame({"gsis_id": ["00-0099999"], "sleeper_id": ["s1"]})
    out = _to_canonical(
        sleeper,
        source=ProjectionSource.SLEEPER,
        id_col="sleeper_id",
        adp_col="sleeper_adp",
        rank_col=None,
        has_stats=True,
        season=2026,
        asof=date(2026, 6, 9),
        id_map=id_map,
    )
    for col in ("espn_auction_value_avg", "espn_auction_value_ppr", "espn_auction_value_std"):
        assert str(out[col].dtype) == "Float64"
        assert pd.isna(out[col].iloc[0])
