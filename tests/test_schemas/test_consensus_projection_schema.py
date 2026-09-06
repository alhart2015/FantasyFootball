from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from projections.schemas import STAT_FIELDS, ConsensusProjectionSchema


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": ["00-0036900", "99-0001234"],
            "season": pd.array([2026, 2026], dtype="Int64"),
            "asof": ["2026-06-09", "2026-06-09"],
            "full_name": ["Ja'Marr Chase", "Some Rookie"],
            "position": ["WR", "RB"],
            "consensus_adp": [4.1, pd.NA],
            "consensus_rank": pd.array([1, pd.NA], dtype="Int64"),
            "n_adp_sources": pd.array([2, 0], dtype="Int64"),
            "has_points": [True, False],
            "projected_points_ppr": [288.9, pd.NA],
            "passing_yards": [0.0, pd.NA],
            "passing_tds": [0.0, pd.NA],
            "interceptions": [0.0, pd.NA],
            "rushing_yards": [21.0, pd.NA],
            "rushing_tds": [0.0, pd.NA],
            "receptions": [119.0, pd.NA],
            "receiving_yards": [1506.0, pd.NA],
            "receiving_tds": [8.4, pd.NA],
            "fumbles_lost": [0.0, pd.NA],
            "is_placeholder_gsis": [False, True],
            "ruleset": ["ESPN_PPR", "ESPN_PPR"],
        }
    )


def test_accepts_wellformed_frame() -> None:
    out = ConsensusProjectionSchema.validate(_valid_frame())
    assert len(out) == 2
    assert pd.isna(out.loc[1, "consensus_adp"])
    assert pd.isna(out.loc[1, "consensus_rank"])


def test_rejects_bad_gsis_id() -> None:
    df = _valid_frame()
    df.loc[0, "gsis_id"] = "not-a-gsis"
    with pytest.raises((SchemaError, SchemaErrors)):
        ConsensusProjectionSchema.validate(df)


def test_rejects_unknown_position() -> None:
    """Uses K, not DST: issue #166 admitted defenses to this table, so DST is now valid here.
    K is still genuinely unmodelled -- kicker scoring has no Ruleset representation."""
    df = _valid_frame()
    df.loc[0, "position"] = "K"
    with pytest.raises((SchemaError, SchemaErrors)):
        ConsensusProjectionSchema.validate(df)


def test_accepts_dst() -> None:
    """The other half of #166: a defense must survive the consensus blend, not be filtered
    out of it."""
    df = _valid_frame()
    df.loc[0, "position"] = "DST"
    assert ConsensusProjectionSchema.validate(df) is not None


def test_gsis_id_must_be_unique() -> None:
    df = _valid_frame()
    df.loc[1, "gsis_id"] = "00-0036900"  # duplicate
    with pytest.raises((SchemaError, SchemaErrors)):
        ConsensusProjectionSchema.validate(df)


def test_consensus_schema_auction_columns_optional_and_float64() -> None:
    base = {
        "gsis_id": pd.array(["00-0011111"], dtype="string[pyarrow]"),
        "season": pd.array([2026], dtype="Int64"),
        "asof": pd.array(["2026-06-09"], dtype="string[pyarrow]"),
        "full_name": pd.array(["P"], dtype="string[pyarrow]"),
        "position": pd.array(["RB"], dtype="string[pyarrow]"),
        "consensus_adp": pd.array([5.0], dtype="Float64"),
        "consensus_rank": pd.array([1], dtype="Int64"),
        "n_adp_sources": pd.array([1], dtype="Int64"),
        "has_points": [True],
        "projected_points_ppr": pd.array([200.0], dtype="Float64"),
        **{f: pd.array([pd.NA], dtype="Float64") for f in STAT_FIELDS},
        "is_placeholder_gsis": [False],
        "ruleset": pd.array(["ESPN_HALF"], dtype="string[pyarrow]"),
    }
    # Without the auction columns: must validate (Optional).
    ConsensusProjectionSchema.validate(pd.DataFrame(base))
    # With them: validates and stays Float64.
    withcols = pd.DataFrame(
        {
            **base,
            "espn_auction_value_avg": pd.array([58.67], dtype="Float64"),
            "espn_auction_value_ppr": pd.array([57.0], dtype="Float64"),
            "espn_auction_value_std": pd.array([55.0], dtype="Float64"),
        }
    )
    out = ConsensusProjectionSchema.validate(withcols)
    assert str(out["espn_auction_value_avg"].dtype) == "Float64"
