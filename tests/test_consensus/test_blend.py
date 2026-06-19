from __future__ import annotations

import pandas as pd

from projections.consensus.blend import build_consensus
from projections.schemas import STAT_FIELDS, ConsensusProjectionSchema, Ruleset

_STAT_COLS = list(STAT_FIELDS)


def _row(
    source: str,
    gsis_id: str,
    *,
    adp: object,
    full_name: str,
    position: str,
    placeholder: bool,
    stats: dict[str, float] | None = None,
) -> dict[str, object]:
    r: dict[str, object] = {
        "source": source,
        "source_player_id": f"{source}-{gsis_id}",
        "gsis_id": gsis_id,
        "is_placeholder_gsis": placeholder,
        "full_name": full_name,
        "position": position,
        "season": 2026,
        "asof": "2026-06-09",
        "adp": adp,
        "espn_draft_rank": pd.NA,
    }
    for c in _STAT_COLS:
        r[c] = (stats or {}).get(c, pd.NA)
    return r


def _external(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_two_source_veteran_blends_adp_and_scores_points() -> None:
    chase_stats = {
        "receptions": 119.0,
        "receiving_yards": 1506.0,
        "receiving_tds": 8.4,
        "rushing_yards": 21.0,
        "passing_yards": 0.0,
        "passing_tds": 0.0,
        "interceptions": 0.0,
        "rushing_tds": 0.0,
        "fumbles_lost": 0.0,
    }
    ext = _external(
        [
            _row(
                "ESPN",
                "00-0036900",
                adp=4.8,
                full_name="Ja'Marr Chase",
                position="WR",
                placeholder=False,
                stats=chase_stats,
            ),
            _row(
                "SLEEPER",
                "00-0036900",
                adp=3.4,
                full_name="Ja'Marr Chase",
                position="WR",
                placeholder=False,
            ),
        ]
    )
    out = build_consensus(ext, Ruleset())
    out = ConsensusProjectionSchema.validate(out)
    assert len(out) == 1
    r = out.iloc[0]
    assert r["consensus_adp"] == 4.1  # mean(4.8, 3.4)
    assert r["n_adp_sources"] == 2
    assert bool(r["has_points"]) is True
    # 119 + 1506/10 + 8.4*6 + 21/10 = 119 + 150.6 + 50.4 + 2.1 = 322.1
    assert round(float(r["projected_points_ppr"]), 1) == 322.1
    assert r["receiving_tds"] == 8.4
    assert r["consensus_rank"] == 1


def test_two_full_sources_blend_per_field_mean() -> None:
    espn = {c: 0.0 for c in _STAT_COLS} | {
        "receptions": 100.0,
        "receiving_yards": 1400.0,
        "receiving_tds": 8.0,
    }
    sleeper = {c: 0.0 for c in _STAT_COLS} | {
        "receptions": 110.0,
        "receiving_yards": 1500.0,
        "receiving_tds": 10.0,
    }
    ext = _external(
        [
            _row(
                "ESPN",
                "00-0036900",
                adp=4.0,
                full_name="X",
                position="WR",
                placeholder=False,
                stats=espn,
            ),
            _row(
                "SLEEPER",
                "00-0036900",
                adp=3.0,
                full_name="X",
                position="WR",
                placeholder=False,
                stats=sleeper,
            ),
        ]
    )
    r = build_consensus(ext, Ruleset()).iloc[0]
    assert r["receptions"] == 105.0  # mean(100, 110)
    assert r["receiving_yards"] == 1450.0
    assert r["receiving_tds"] == 9.0
    assert bool(r["has_points"]) is True
    # Ruleset() default scores 1.0/reception: 105 + 1450/10 + 9*6 = 105 + 145 + 54 = 304
    assert round(float(r["projected_points_ppr"]), 1) == 304.0


def test_stub_row_excluded_from_blend() -> None:
    # ESPN "stub": an all-zero stat line (the 2023 degenerate case). Sleeper has a real line.
    stub = {c: 0.0 for c in _STAT_COLS}
    sleeper = {c: 0.0 for c in _STAT_COLS} | {
        "receptions": 110.0,
        "receiving_yards": 1500.0,
        "receiving_tds": 10.0,
    }
    ext = _external(
        [
            _row(
                "ESPN",
                "00-0036900",
                adp=4.0,
                full_name="X",
                position="WR",
                placeholder=False,
                stats=stub,
            ),
            _row(
                "SLEEPER",
                "00-0036900",
                adp=3.0,
                full_name="X",
                position="WR",
                placeholder=False,
                stats=sleeper,
            ),
        ]
    )
    r = build_consensus(ext, Ruleset()).iloc[0]
    # Sleeper-only, NOT mean(0, 110) = 55:
    assert (
        r["receptions"] == 110.0 and r["receiving_yards"] == 1500.0 and r["receiving_tds"] == 10.0
    )


def test_all_zero_rows_yield_no_points() -> None:
    stub = {c: 0.0 for c in _STAT_COLS}
    ext = _external(
        [
            _row(
                "ESPN",
                "00-0036900",
                adp=4.0,
                full_name="X",
                position="WR",
                placeholder=False,
                stats=stub,
            ),
            _row(
                "SLEEPER",
                "00-0036900",
                adp=3.0,
                full_name="X",
                position="WR",
                placeholder=False,
                stats=stub,
            ),
        ]
    )
    r = build_consensus(ext, Ruleset()).iloc[0]
    assert bool(r["has_points"]) is False
    assert pd.isna(r["projected_points_ppr"])


def test_one_nonzero_field_is_not_stat_bearing() -> None:
    # A single non-zero field (the rare 2023 scoring stub) is below MIN_STAT_FIELDS=2 -> excluded.
    one = {c: 0.0 for c in _STAT_COLS} | {"rushing_yards": 50.0}
    full = {c: 0.0 for c in _STAT_COLS} | {
        "receptions": 80.0,
        "receiving_yards": 900.0,
        "receiving_tds": 5.0,
    }
    ext = _external(
        [
            _row(
                "ESPN",
                "00-0036900",
                adp=4.0,
                full_name="X",
                position="WR",
                placeholder=False,
                stats=one,
            ),
            _row(
                "SLEEPER",
                "00-0036900",
                adp=3.0,
                full_name="X",
                position="WR",
                placeholder=False,
                stats=full,
            ),
        ]
    )
    r = build_consensus(ext, Ruleset()).iloc[0]
    assert r["rushing_yards"] == 0.0  # the 1-field stub did NOT contribute
    assert r["receptions"] == 80.0


def test_adp_unaffected_by_stat_gating() -> None:
    # consensus_adp/rank come from ADP regardless of whether the stat line is gated out.
    stub = {c: 0.0 for c in _STAT_COLS}
    ext = _external(
        [
            _row(
                "ESPN",
                "00-0036900",
                adp=4.0,
                full_name="X",
                position="WR",
                placeholder=False,
                stats=stub,
            ),
            _row("SLEEPER", "00-0036900", adp=2.0, full_name="X", position="WR", placeholder=False),
        ]
    )
    r = build_consensus(ext, Ruleset()).iloc[0]
    assert r["consensus_adp"] == 3.0  # mean(4, 2), unaffected by stat gating
    assert r["n_adp_sources"] == 2


def test_handles_non_unique_index_without_misblending() -> None:
    # build_consensus must tolerate a caller-supplied non-unique index (it resets internally):
    # the per-group stat-bearing mask must not pull a different player's stats into the blend.
    a_stats = {c: 0.0 for c in _STAT_COLS} | {
        "receptions": 80.0,
        "receiving_yards": 900.0,
        "receiving_tds": 5.0,
    }
    b_stats = {c: 0.0 for c in _STAT_COLS} | {"rushing_yards": 1000.0, "rushing_tds": 8.0}
    ext = _external(
        [
            _row(
                "ESPN",
                "00-0000001",
                adp=5.0,
                full_name="A",
                position="WR",
                placeholder=False,
                stats=a_stats,
            ),
            _row(
                "ESPN",
                "00-0000002",
                adp=6.0,
                full_name="B",
                position="RB",
                placeholder=False,
                stats=b_stats,
            ),
        ]
    )
    ext.index = [0, 0]  # duplicate labels — the scenario that broke the .loc[grp.index] lookup
    out = build_consensus(ext, Ruleset()).set_index("gsis_id")
    assert len(out) == 2
    assert (
        out.loc["00-0000001", "receptions"] == 80.0
        and out.loc["00-0000001", "rushing_yards"] == 0.0
    )
    assert (
        out.loc["00-0000002", "rushing_yards"] == 1000.0
        and out.loc["00-0000002", "receptions"] == 0.0
    )


def test_sleeper_only_player_has_adp_no_points() -> None:
    ext = _external(
        [
            _row(
                "SLEEPER",
                "00-0011111",
                adp=50.0,
                full_name="Deep Sleeper",
                position="RB",
                placeholder=False,
            ),
        ]
    )
    out = build_consensus(ext, Ruleset())
    r = out[out["gsis_id"] == "00-0011111"].iloc[0]
    assert r["n_adp_sources"] == 1
    assert bool(r["has_points"]) is False
    assert pd.isna(r["projected_points_ppr"])
    assert all(pd.isna(r[c]) for c in _STAT_COLS)


def test_union_coverage_and_deterministic_rank() -> None:
    ext = _external(
        [
            _row(
                "SLEEPER",
                "00-0000002",
                adp=10.0,
                full_name="B Player",
                position="WR",
                placeholder=False,
            ),
            _row(
                "ESPN",
                "00-0000001",
                adp=10.0,
                full_name="A Player",
                position="RB",
                placeholder=False,
                stats={c: 0.0 for c in _STAT_COLS},
            ),
            _row(
                "SLEEPER",
                "00-0000003",
                adp=5.0,
                full_name="C Player",
                position="TE",
                placeholder=False,
            ),
        ]
    )
    out = build_consensus(ext, Ruleset())
    assert len(out) == 3  # union
    by_id = out.set_index("gsis_id")
    assert by_id.loc["00-0000003", "consensus_rank"] == 1  # adp 5 first
    # tie at adp 10: ordered by gsis_id, so ...0001 (rank 2) before ...0002 (rank 3)
    assert by_id.loc["00-0000001", "consensus_rank"] == 2
    assert by_id.loc["00-0000002", "consensus_rank"] == 3


def test_player_with_points_but_no_adp_gets_null_rank() -> None:
    ext = _external(
        [
            _row(
                "ESPN",
                "00-0000009",
                adp=pd.NA,
                full_name="No ADP",
                position="WR",
                placeholder=False,
                stats={c: 0.0 for c in _STAT_COLS},
            ),
        ]
    )
    out = build_consensus(ext, Ruleset())
    r = out[out["gsis_id"] == "00-0000009"].iloc[0]
    assert pd.isna(r["consensus_adp"])
    assert pd.isna(r["consensus_rank"])
    assert r["n_adp_sources"] == 0
    # all-zero stat line is not a real projection -> not stat-bearing -> has_points False
    assert bool(r["has_points"]) is False


def test_placeholder_rookie_carried_through() -> None:
    ext = _external(
        [
            _row(
                "ESPN",
                "99-0001234",
                adp=17.6,
                full_name="Jeremiyah Love",
                position="RB",
                placeholder=True,
                stats={c: 0.0 for c in _STAT_COLS},
            ),
            _row(
                "SLEEPER",
                "99-0001234",
                adp=18.0,
                full_name="Jeremiyah Love",
                position="RB",
                placeholder=True,
            ),
        ]
    )
    out = build_consensus(ext, Ruleset())
    r = out[out["gsis_id"] == "99-0001234"].iloc[0]
    assert bool(r["is_placeholder_gsis"]) is True
    assert r["n_adp_sources"] == 2


def test_nonpositive_adp_treated_as_missing() -> None:
    # ESPN encodes "undrafted" as ADP 0; it must not corrupt the mean or violate the schema's
    # consensus_adp > 0. Blended with a real ADP, only the positive value counts.
    ext = _external(
        [
            _row(
                "ESPN",
                "00-0000010",
                adp=0.0,
                full_name="Zero ADP",
                position="WR",
                placeholder=False,
                stats={c: 0.0 for c in _STAT_COLS},
            ),
            _row(
                "SLEEPER",
                "00-0000010",
                adp=24.0,
                full_name="Zero ADP",
                position="WR",
                placeholder=False,
            ),
        ]
    )
    out = build_consensus(ext, Ruleset())
    out = ConsensusProjectionSchema.validate(out)  # must not raise on gt=0
    r = out[out["gsis_id"] == "00-0000010"].iloc[0]
    assert r["consensus_adp"] == 24.0  # the 0.0 is dropped, not averaged to 12.0
    assert r["n_adp_sources"] == 1


def test_only_nonpositive_adp_yields_null_adp_and_rank() -> None:
    ext = _external(
        [
            _row(
                "ESPN",
                "00-0000011",
                adp=0.0,
                full_name="Undrafted",
                position="RB",
                placeholder=False,
                stats={c: 0.0 for c in _STAT_COLS},
            ),
        ]
    )
    out = build_consensus(ext, Ruleset())
    r = out[out["gsis_id"] == "00-0000011"].iloc[0]
    assert pd.isna(r["consensus_adp"])
    assert pd.isna(r["consensus_rank"])
    assert r["n_adp_sources"] == 0
    # all-zero stat line is not a projection; row still appears via ADP union coverage
    assert bool(r["has_points"]) is False


def test_empty_input_returns_empty_conforming_frame() -> None:
    cols = [
        "source",
        "source_player_id",
        "gsis_id",
        "is_placeholder_gsis",
        "full_name",
        "position",
        "season",
        "asof",
        "adp",
        "espn_draft_rank",
        *_STAT_COLS,
    ]
    out = build_consensus(pd.DataFrame(columns=cols), Ruleset())
    assert out.empty
    ConsensusProjectionSchema.validate(out)  # empty frame still conforms


_AUCTION_STATS = (
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


def _ext_row(
    source: str,
    gsis_id: str,
    *,
    adp: float = 5.0,
    av_avg: object = pd.NA,
    stats: bool = True,
    **extra: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "source": source,
        "source_player_id": f"{source}-{gsis_id}",
        "gsis_id": gsis_id,
        "is_placeholder_gsis": False,
        "full_name": "Player X",
        "position": "RB",
        "season": 2026,
        "asof": "2026-06-09",
        "adp": adp,
        "espn_draft_rank": pd.NA,
        "espn_auction_value_avg": av_avg,
        "espn_auction_value_ppr": pd.NA,
        "espn_auction_value_std": pd.NA,
    }
    for s in _AUCTION_STATS:
        row[s] = 100.0 if stats else pd.NA
    row.update(extra)
    return row


def test_blend_carries_espn_auction_value_first_non_null() -> None:
    # ESPN row has the value, Sleeper row does not -> consensus keeps the value.
    external = pd.DataFrame(
        [
            _ext_row("espn", "00-0011111", av_avg=58.67),
            _ext_row("sleeper", "00-0011111", av_avg=pd.NA, stats=False),
        ]
    )
    out = build_consensus(external, Ruleset.espn_half()).set_index("gsis_id")
    assert out.loc["00-0011111", "espn_auction_value_avg"] == 58.67


def test_blend_sleeper_only_player_has_na_auction() -> None:
    external = pd.DataFrame([_ext_row("sleeper", "00-0022222", av_avg=pd.NA, stats=False)])
    out = build_consensus(external, Ruleset.espn_half()).set_index("gsis_id")
    assert pd.isna(out.loc["00-0022222", "espn_auction_value_avg"])


def test_blend_does_not_crash_when_auction_columns_absent() -> None:
    # A frame lacking the columns entirely (stale snapshot / existing tests) must not KeyError.
    external = pd.DataFrame([_ext_row("espn", "00-0033333", av_avg=58.0)]).drop(
        columns=["espn_auction_value_avg", "espn_auction_value_ppr", "espn_auction_value_std"]
    )
    out = build_consensus(external, Ruleset.espn_half()).set_index("gsis_id")
    assert pd.isna(out.loc["00-0033333", "espn_auction_value_avg"])  # seeded to NA


def test_blend_keeps_espn_value_when_sleeper_is_identity_row() -> None:
    # The distinguishing test (spec R3): ESPN row is NOT stat-bearing, so the Sleeper row becomes
    # the identity_row — but ESPN carries the auction value. First-non-null must still surface it;
    # an identity_row[col] pick would wrongly return the Sleeper row's NA.
    external = pd.DataFrame(
        [
            _ext_row("espn", "00-0044444", av_avg=58.0, stats=False),  # ESPN: value, no stat line
            _ext_row("sleeper", "00-0044444", av_avg=pd.NA, stats=True),  # Sleeper: stat-bearing
        ]
    )
    out = build_consensus(external, Ruleset.espn_half()).set_index("gsis_id")
    assert out.loc["00-0044444", "espn_auction_value_avg"] == 58.0


def test_blend_empty_input_carries_auction_columns() -> None:
    # _empty_output() builds from _OUTPUT_COLUMNS; the new names must be present on the empty path.
    empty = pd.DataFrame(
        columns=[
            "source",
            "source_player_id",
            "gsis_id",
            "is_placeholder_gsis",
            "full_name",
            "position",
            "season",
            "asof",
            "adp",
            "espn_draft_rank",
            "espn_auction_value_avg",
            "espn_auction_value_ppr",
            "espn_auction_value_std",
            *_AUCTION_STATS,
        ]
    )
    out = build_consensus(empty, Ruleset.espn_half())
    for col in (
        "espn_auction_value_avg",
        "espn_auction_value_ppr",
        "espn_auction_value_std",
    ):
        assert col in out.columns
