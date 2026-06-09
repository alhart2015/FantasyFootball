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
    assert bool(r["has_points"]) is True


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
    assert bool(r["has_points"]) is True  # still appears (union coverage)


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
