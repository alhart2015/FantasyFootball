"""id_map ingest tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.ingest import build_id_map
from projections.schemas import IdMapSchema
from projections.store import read_partition


def test_build_id_map_writes_validated_parquet(
    tmp_path: Path,
    fake_id_map_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.id_map._fetch_raw_id_map",
        lambda: fake_id_map_df,
    )
    out_path = build_id_map(tmp_path)
    assert out_path.exists()

    df = read_partition(tmp_path / "raw", "id_map", season=None, week=None)
    IdMapSchema.validate(df)  # raises if anything is off
    assert set(df["gsis_id"]) == {
        "00-0036322",
        "00-0034857",
        "00-0034796",
        "00-0030506",
    }


def test_build_id_map_renames_name_column(
    tmp_path: Path,
    fake_id_map_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.id_map._fetch_raw_id_map",
        lambda: fake_id_map_df,
    )
    build_id_map(tmp_path)
    df = read_partition(tmp_path / "raw", "id_map", season=None, week=None)
    assert "full_name" in df.columns
    assert "name" not in df.columns


def test_build_id_map_drops_rows_without_gsis_id(
    tmp_path: Path,
    fake_id_map_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    new_row = {
        "gsis_id": None,
        "espn_id": "1",
        "sleeper_id": "1",
        "pfr_id": "x",
        "name": "no gsis",
        "position": "WR",
        "team": "MIN",
    }
    bad = pd.concat([fake_id_map_df, pd.DataFrame([new_row])], ignore_index=True)
    monkeypatch.setattr("projections.ingest.id_map._fetch_raw_id_map", lambda: bad)
    build_id_map(tmp_path)
    df = read_partition(tmp_path / "raw", "id_map", season=None, week=None)
    assert df["gsis_id"].notna().all()
    assert len(df) == 4


def test_build_id_map_filters_unsupported_positions(
    tmp_path: Path,
    fake_id_map_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fb_row = {
        "gsis_id": "00-0099999",
        "espn_id": "9999999",
        "sleeper_id": "9999",
        "pfr_id": "FbPla00",
        "name": "FB Player",
        "position": "FB",
        "team": "MIN",
    }
    with_fb = pd.concat([fake_id_map_df, pd.DataFrame([fb_row])], ignore_index=True)
    monkeypatch.setattr("projections.ingest.id_map._fetch_raw_id_map", lambda: with_fb)
    build_id_map(tmp_path)
    df = read_partition(tmp_path / "raw", "id_map", season=None, week=None)
    assert "00-0099999" not in df["gsis_id"].tolist()
    assert len(df) == 4


def test_build_id_map_warns_on_placeholder_gsis_ids(
    tmp_path: Path,
    fake_id_map_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Mirrors pre-camp-rookie shape: nflverse holds PFR-style placeholder ids
    # for the current draft class until NFL assigns real gsis_ids ~July.
    rookie_rows = pd.DataFrame(
        [
            {
                "gsis_id": "MEN516487",
                "espn_id": "4837248",
                "sleeper_id": "13269",
                "pfr_id": "MendFe00",
                "name": "Fernando Mendoza",
                "position": "QB",
                "team": "LVR",
            },
            {
                "gsis_id": "TYS405541",
                "espn_id": "5000000",
                "sleeper_id": "14000",
                "pfr_id": "TysoJo00",
                "name": "Jordyn Tyson",
                "position": "WR",
                "team": "NOR",
            },
        ]
    )
    mixed = pd.concat([fake_id_map_df, rookie_rows], ignore_index=True)
    monkeypatch.setattr("projections.ingest.id_map._fetch_raw_id_map", lambda: mixed)

    import logging

    with caplog.at_level(logging.WARNING, logger="projections.ingest.id_map"):
        build_id_map(tmp_path)

    df = read_partition(tmp_path / "raw", "id_map", season=None, week=None)
    assert "MEN516487" not in df["gsis_id"].tolist()
    assert "TYS405541" not in df["gsis_id"].tolist()
    assert len(df) == 4
    assert any(
        "filtered 2 row(s) with non-GSIS placeholder ids" in r.getMessage() for r in caplog.records
    )


def test_coerce_external_id_leaves_string_pfr_ids_unchanged() -> None:
    import pandas as pd

    from projections.ingest.id_map import _coerce_external_id

    s = pd.Series(["ChASEJa00", None, "AlleJo02"])
    out = _coerce_external_id(s, numeric=False)
    assert out.tolist()[0] == "ChASEJa00"
    assert out.tolist()[2] == "AlleJo02"
    assert pd.isna(out.tolist()[1])


def test_coerce_external_id_object_dtype_floats_drop_dot_zero() -> None:
    import pandas as pd

    from projections.ingest.id_map import _coerce_external_id

    s = pd.Series([4374302.0, None, 6794.0], dtype=object)
    out = _coerce_external_id(s, numeric=True)
    assert out.tolist()[0] == "4374302" and out.tolist()[2] == "6794"


def test_float_valued_external_id_persists_without_trailing_dot_zero() -> None:
    # Upstream load_ff_playerids() returns espn_id/sleeper_id as float64 (NaNs force float),
    # so an integer id arrives as 4374302.0. It must persist as "4374302", not "4374302.0",
    # or the external-projection crosswalk join silently misses.
    import pandas as pd

    from projections.ingest.id_map import _coerce_external_id  # added in Step 3

    s = pd.Series([4374302.0, float("nan"), 6794.0])
    out = _coerce_external_id(s, numeric=True)
    assert out.tolist()[0] == "4374302"
    assert out.tolist()[2] == "6794"
    assert pd.isna(out.tolist()[1])
    assert str(out.dtype) == "string"


def test_coerce_external_id_preserves_leading_zero_string_when_not_numeric() -> None:
    import pandas as pd

    from projections.ingest.id_map import _coerce_external_id

    out = _coerce_external_id(pd.Series(["0012345", None]), numeric=False)
    assert out.tolist()[0] == "0012345"
