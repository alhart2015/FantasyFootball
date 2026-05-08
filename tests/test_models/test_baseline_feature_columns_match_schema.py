"""Regression test pinning baseline.py's hardcoded per-position feature
tuples to each `*FeaturesSchema`'s declared columns.

Why: the lightgbm family derives feature lists from `<Schema>.to_schema().columns.keys()`
dynamically and auto-picks-up new columns when the schema changes. baseline.py
hardcodes per-position tuples (`_QB_FEATURE_COLUMNS` etc.) and must be updated
explicitly. Four PRs in a row (PR #21 RB PBP, PR #26 WR trajectory, PR #27 TE
trajectory, PR #29 weather) hit this same recurring spec-gap pattern, with the
miss caught only at code-review time. This test fires the moment a schema gains
or loses a column without the matching `_<POS>_FEATURE_COLUMNS` update.

The contract: per position,
    set(_<POS>_FEATURE_COLUMNS) == set(SCHEMA.columns) - identity
where `identity = {gsis_id, season, week, team, opponent}` are the five
non-feature columns common to every per-position schema.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
import pytest

from projections.models.baseline import (
    _QB_FEATURE_COLUMNS,
    _RB_FEATURE_COLUMNS,
    _TE_FEATURE_COLUMNS,
    _WR_FEATURE_COLUMNS,
)
from projections.schemas import (
    QbFeaturesSchema,
    RbFeaturesSchema,
    TeFeaturesSchema,
    WrFeaturesSchema,
)

_IDENTITY_COLS: frozenset[str] = frozenset({"gsis_id", "season", "week", "team", "opponent"})


@pytest.mark.parametrize(
    ("position", "feature_columns", "schema"),
    [
        ("QB", _QB_FEATURE_COLUMNS, QbFeaturesSchema),
        ("RB", _RB_FEATURE_COLUMNS, RbFeaturesSchema),
        ("TE", _TE_FEATURE_COLUMNS, TeFeaturesSchema),
        ("WR", _WR_FEATURE_COLUMNS, WrFeaturesSchema),
    ],
)
def test_baseline_feature_columns_match_schema_minus_identity(
    position: str,
    feature_columns: tuple[str, ...],
    schema: type[pa.DataFrameModel],
) -> None:
    """`_<POS>_FEATURE_COLUMNS` must equal the schema's columns minus identity."""
    schema_cols = set(schema.to_schema().columns.keys())
    expected_features = schema_cols - _IDENTITY_COLS
    actual_features = set(feature_columns)

    missing_in_baseline = expected_features - actual_features
    extra_in_baseline = actual_features - expected_features

    assert not missing_in_baseline, (
        f"{position}: schema declares columns that _{position}_FEATURE_COLUMNS does not "
        f"reference: {sorted(missing_in_baseline)}. The lightgbm family derives feature "
        f"lists from the schema dynamically and would pick these up; baseline.py is "
        f"hardcoded and must be updated to match."
    )
    assert not extra_in_baseline, (
        f"{position}: _{position}_FEATURE_COLUMNS references columns that the schema does "
        f"not declare: {sorted(extra_in_baseline)}. Either remove from baseline.py or add "
        f"to {schema.__name__}."
    )
    # No duplicates in the tuple — `set(tuple) == tuple` length is the standard check.
    assert len(feature_columns) == len(actual_features), (
        f"{position}: _{position}_FEATURE_COLUMNS has duplicate entries: "
        f"{[c for c in feature_columns if list(feature_columns).count(c) > 1]}"
    )


def test_identity_cols_present_in_every_schema() -> None:
    """Sanity check: the identity column set is the right one for every schema."""
    for schema in (QbFeaturesSchema, RbFeaturesSchema, TeFeaturesSchema, WrFeaturesSchema):
        cols = set(schema.to_schema().columns.keys())
        missing = _IDENTITY_COLS - cols
        assert not missing, (
            f"{schema.__name__} is missing identity column(s) {sorted(missing)}; "
            f"the parametrized test above will fail incorrectly until this is fixed "
            f"(or _IDENTITY_COLS narrowed)."
        )


# Suppress an unused-import warning (`pd` only referenced via type-checking).
_ = pd
