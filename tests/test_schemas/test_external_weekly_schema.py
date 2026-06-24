import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from projections.schemas import _PYARROW_STR, ExternalProjectionWeeklySchema


def _valid_row() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": pd.array(["SLEEPER"], dtype=_PYARROW_STR),
            "source_player_id": pd.array(["4046"], dtype=_PYARROW_STR),
            "gsis_id": pd.array(["00-0036900"], dtype=_PYARROW_STR),
            "is_placeholder_gsis": [False],
            "full_name": pd.array(["Ja'Marr Chase"], dtype=_PYARROW_STR),
            "position": pd.array(["WR"], dtype=_PYARROW_STR),
            "season": [2023],
            "week": pd.array([5], dtype="Int64"),
            "passing_yards": pd.array([0.0], dtype="Float64"),
            "passing_tds": pd.array([0.0], dtype="Float64"),
            "interceptions": pd.array([0.0], dtype="Float64"),
            "rushing_yards": pd.array([0.0], dtype="Float64"),
            "rushing_tds": pd.array([0.0], dtype="Float64"),
            "receptions": pd.array([6.0], dtype="Float64"),
            "receiving_yards": pd.array([78.0], dtype="Float64"),
            "receiving_tds": pd.array([0.5], dtype="Float64"),
            "fumbles_lost": pd.array([0.0], dtype="Float64"),
        }
    )


def test_valid_weekly_row_passes() -> None:
    out = ExternalProjectionWeeklySchema.validate(_valid_row())
    assert out["week"].iloc[0] == 5


def test_week_out_of_range_rejected() -> None:
    bad = _valid_row()
    bad["week"] = pd.array([23], dtype="Int64")
    with pytest.raises((SchemaError, SchemaErrors)):
        ExternalProjectionWeeklySchema.validate(bad)
