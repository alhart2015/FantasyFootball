"""ProjectionWeeklySchema validation regression tests."""

from __future__ import annotations

import pandas as pd

from projections.schemas import ProjectionWeeklySchema


def test_projection_weekly_schema_validates_empty_dataframe() -> None:
    """An empty DataFrame with the right columns should validate cleanly.

    Without coerce=True, an empty pd.DataFrame(columns=...) produces object-dtype
    columns and pandera rejects them against the typed Series declarations.
    """
    cols = list(ProjectionWeeklySchema.to_schema().columns.keys())
    empty = pd.DataFrame(columns=cols)
    out = ProjectionWeeklySchema.validate(empty)
    assert out.empty
