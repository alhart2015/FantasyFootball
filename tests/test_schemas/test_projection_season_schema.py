"""ProjectionSeasonSchema — validation behavior."""

from __future__ import annotations

import pandas as pd
import pandera.errors
import pytest

from projections.schemas import _PYARROW_STR, ProjectionSeasonSchema


def _canonical_season_row(**overrides: object) -> dict[str, object]:
    base = {
        "gsis_id": "00-0033873",
        "season": 2024,
        "position": "WR",
        "ruleset": "ESPN_PPR",
        "n_weeks": 17,
        "season_mean": 250.0,
        "season_p10": 180.0,
        "season_p50": 248.0,
        "season_p90": 320.0,
        "model_id": "baseline:wr:abcdef12:2018-2023",
        "generated_at": pd.Timestamp("2026-04-26", tz="UTC").as_unit("us"),
    }
    base.update(overrides)
    return base


def _to_validated_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ("gsis_id", "position", "ruleset", "model_id"):
        df[col] = df[col].astype(_PYARROW_STR)
    return ProjectionSeasonSchema.validate(df)


def test_canonical_row_validates() -> None:
    out = _to_validated_frame([_canonical_season_row()])
    assert len(out) == 1
    assert out["season_mean"].iloc[0] == 250.0


def test_invalid_gsis_id_rejected() -> None:
    with pytest.raises(pandera.errors.SchemaError):
        _to_validated_frame([_canonical_season_row(gsis_id="not-a-gsis-id")])


def test_n_weeks_zero_rejected() -> None:
    with pytest.raises(pandera.errors.SchemaError):
        _to_validated_frame([_canonical_season_row(n_weeks=0)])


def test_n_weeks_above_22_rejected() -> None:
    with pytest.raises(pandera.errors.SchemaError):
        _to_validated_frame([_canonical_season_row(n_weeks=23)])


def test_position_not_in_set_rejected() -> None:
    with pytest.raises(pandera.errors.SchemaError):
        _to_validated_frame([_canonical_season_row(position="ZZ")])


def test_season_below_1999_rejected() -> None:
    with pytest.raises(pandera.errors.SchemaError):
        _to_validated_frame([_canonical_season_row(season=1998)])


def test_naive_datetime_rejected() -> None:
    naive_ts = pd.Timestamp("2026-04-26").as_unit("us")
    with pytest.raises(pandera.errors.SchemaError):
        _to_validated_frame([_canonical_season_row(generated_at=naive_ts)])
