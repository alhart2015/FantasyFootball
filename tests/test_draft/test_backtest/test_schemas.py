import pandas as pd
import pandera.pandas as pa
import pytest

from projections.schemas import _PYARROW_STR, WeeklyActualSchema, WeeklyProjectionSchema


def _proj_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000001"], dtype=_PYARROW_STR),
            "season": pd.array([2025], dtype="Int64"),
            "week": pd.array([5], dtype="Int64"),
            "position": pd.array(["RB"], dtype=_PYARROW_STR),
            "projected_points": pd.array([14.3], dtype="Float64"),
        }
    )


def test_weekly_projection_schema_accepts_valid_frame() -> None:
    out = WeeklyProjectionSchema.validate(_proj_frame())
    assert len(out) == 1


def test_weekly_projection_rejects_week_18() -> None:
    bad = _proj_frame()
    bad["week"] = pd.array([18], dtype="Int64")
    with pytest.raises(pa.errors.SchemaError):
        WeeklyProjectionSchema.validate(bad)


def test_weekly_projection_allows_null_points() -> None:
    f = _proj_frame()
    f["projected_points"] = pd.array([None], dtype="Float64")
    assert len(WeeklyProjectionSchema.validate(f)) == 1


def test_weekly_actual_schema_accepts_valid_frame() -> None:
    f = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000001"], dtype=_PYARROW_STR),
            "season": pd.array([2025], dtype="Int64"),
            "week": pd.array([5], dtype="Int64"),
            "actual_points": pd.array([9.1], dtype="Float64"),
        }
    )
    assert len(WeeklyActualSchema.validate(f)) == 1
