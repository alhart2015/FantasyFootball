from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from projections.schemas import ConsensusProjectionSchema


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
    df = _valid_frame()
    df.loc[0, "position"] = "DST"
    with pytest.raises((SchemaError, SchemaErrors)):
        ConsensusProjectionSchema.validate(df)


def test_gsis_id_must_be_unique() -> None:
    df = _valid_frame()
    df.loc[1, "gsis_id"] = "00-0036900"  # duplicate
    with pytest.raises((SchemaError, SchemaErrors)):
        ConsensusProjectionSchema.validate(df)
