"""PFR advanced-stats ingest tests.

Three stat types through one module, all keyed on `pfr_player_id` rather than `gsis_id`, so the
id_map crosswalk is the seam that matters most here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from projections.ingest import build_id_map, refresh_pfr_advstats
from projections.ingest.pfr_advstats import (
    _KEEP_COMMON,
    _KEEP_FOR,
    STAT_TYPES,
    PfrStatType,
    _normalize_one_season,
)
from projections.schemas import PfrPassingSchema, PfrReceivingSchema, PfrRushingSchema
from projections.store import read_partition

_SCHEMAS = {
    "pass": PfrPassingSchema,
    "rush": PfrRushingSchema,
    "rec": PfrReceivingSchema,
}

#: Every column upstream ships in every table, including the ones that are entirely null for a
#: given stat type. The fixture carries all of them so the per-type selection is actually tested.
_ALL_STAT_COLS = sorted({c for cols in _KEEP_FOR.values() for c in cols if c not in _KEEP_COMMON})


def _setup_id_map(
    tmp_path: Path, fake_id_map_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("projections.ingest.id_map._fetch_raw_id_map", lambda: fake_id_map_df)
    build_id_map(tmp_path)


def _raw(**overrides: list[object]) -> pd.DataFrame:
    """A three-row `load_pfr_advstats` payload, in upstream's real dtypes (Int32 season/week).

    `pfr_player_id` values match `fake_id_map_df`; the `_pct` columns are fractions in [0, 1],
    which is what upstream actually returns despite the naming.
    """
    base: dict[str, list[object]] = {
        "game_id": ["2024_01_KC_BAL", "2024_01_MIN_NYG", "2024_01_PHI_GB"],
        "pfr_game_id": ["a", "b", "c"],
        "season": [2024, 2024, 2024],
        "week": [1, 1, 1],
        "game_type": ["REG", "REG", "REG"],
        "team": ["KC", "MIN", "PHI"],
        "opponent": ["BAL", "NYG", "GB"],
        "pfr_player_name": ["Patrick Mahomes", "Justin Jefferson", "Saquon Barkley"],
        "pfr_player_id": ["MahoPa00", "JeffJu00", "BarkSa00"],
    }
    for col in _ALL_STAT_COLS:
        base[col] = [0.5, 0.5, 0.5]
    base.update(overrides)
    frame = pd.DataFrame(base)
    return frame.astype({"season": "int32", "week": "int32"})


@pytest.mark.parametrize("stat_type", STAT_TYPES)
def test_each_stat_type_writes_a_validated_partition(
    stat_type: PfrStatType,
    tmp_path: Path,
    fake_id_map_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_id_map(tmp_path, fake_id_map_df, monkeypatch)
    monkeypatch.setattr(
        "projections.ingest.pfr_advstats._fetch_raw_pfr_advstats",
        lambda st, seasons: _raw(),
    )

    written = refresh_pfr_advstats(tmp_path, stat_type=stat_type, seasons=[2024])

    assert len(written) == 1
    df = read_partition(tmp_path / "raw", f"pfr_{stat_type}", season=2024)
    _SCHEMAS[stat_type].validate(df)
    assert len(df) == 3


@pytest.mark.parametrize("stat_type", STAT_TYPES)
def test_each_stat_type_keeps_only_its_own_columns(
    stat_type: PfrStatType,
    tmp_path: Path,
    fake_id_map_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upstream ships every type's columns in every table with the irrelevant ones all-null --
    `pass` carries an all-null `receiving_drop`. Selecting per type is what stops a meaningful
    column in one table becoming a column of nulls in another."""
    _setup_id_map(tmp_path, fake_id_map_df, monkeypatch)

    out = _normalize_one_season(stat_type, _raw(), tmp_path)

    assert set(out.columns) == set(_KEEP_FOR[stat_type])
    foreign = set(_ALL_STAT_COLS) - set(_KEEP_FOR[stat_type])
    assert not (foreign & set(out.columns))


def test_pfr_ids_are_resolved_to_gsis(
    tmp_path: Path, fake_id_map_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_id_map(tmp_path, fake_id_map_df, monkeypatch)

    out = _normalize_one_season("rush", _raw(), tmp_path)

    assert set(out["gsis_id"]) == {"00-0034857", "00-0036322", "00-0034796"}
    assert "pfr_player_id" not in out.columns


def test_rows_with_no_id_map_entry_are_dropped(
    tmp_path: Path, fake_id_map_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_id_map(tmp_path, fake_id_map_df, monkeypatch)
    raw = _raw(pfr_player_id=["MahoPa00", "NobdyWh00", "BarkSa00"])

    out = _normalize_one_season("rush", raw, tmp_path)

    assert set(out["gsis_id"]) == {"00-0034857", "00-0034796"}


def test_a_heavy_crosswalk_loss_warns(
    tmp_path: Path,
    fake_id_map_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A thin partition and a successful exit look identical without this. Losing two of three
    rows means a stale id_map, not bench players."""
    _setup_id_map(tmp_path, fake_id_map_df, monkeypatch)
    raw = _raw(pfr_player_id=["MahoPa00", "NobdyWh00", "AlsoNot00"])

    with caplog.at_level(logging.WARNING):
        _normalize_one_season("rush", raw, tmp_path)

    assert "stale id_map" in caplog.text


def test_without_an_id_map_it_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _normalize_one_season("rush", _raw(), tmp_path)


def test_negative_contact_yards_are_accepted(
    tmp_path: Path, fake_id_map_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run stuffed behind the line loses yards before contact; the count fields are ge=0 but
    the yardage fields deliberately are not."""
    _setup_id_map(tmp_path, fake_id_map_df, monkeypatch)
    raw = _raw(
        rushing_yards_before_contact=[-8.0, 0.0, 12.0],
        rushing_yards_before_contact_avg=[-4.0, 0.0, 6.0],
    )

    out = _normalize_one_season("rush", raw, tmp_path)

    assert out["rushing_yards_before_contact"].min() == -8.0


def test_a_null_per_carry_average_survives(
    tmp_path: Path, fake_id_map_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `_avg` columns divide by carries; a player charted with none yields a null."""
    _setup_id_map(tmp_path, fake_id_map_df, monkeypatch)
    raw = _raw(
        carries=[0.0, 5.0, 5.0],
        rushing_yards_before_contact_avg=[None, 2.0, 3.0],
        rushing_yards_after_contact_avg=[None, 2.0, 3.0],
    )

    out = _normalize_one_season("rush", raw, tmp_path)

    assert out["rushing_yards_before_contact_avg"].isna().sum() == 1


def test_postseason_weeks_are_kept(
    tmp_path: Path, fake_id_map_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unlike the injury report, PFR's postseason weeks continue rather than restart
    (WC=19 ... SB=22), so there is no collision to filter away."""
    _setup_id_map(tmp_path, fake_id_map_df, monkeypatch)
    raw = _raw(week=[19, 20, 22], game_type=["WC", "DIV", "SB"])

    out = _normalize_one_season("rush", raw, tmp_path)

    assert sorted(out["week"]) == [19, 20, 22]


def test_team_codes_are_normalized(
    tmp_path: Path, fake_id_map_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PFR still emits historical and variant codes -- LA, OAK, JAX -- across older seasons."""
    _setup_id_map(tmp_path, fake_id_map_df, monkeypatch)
    raw = _raw(team=["JAX", "LA", "OAK"], opponent=["LA", "OAK", "JAX"])

    out = _normalize_one_season("rush", raw, tmp_path)

    assert list(out["team"]) == ["JAC", "LAR", "LV"]
    assert list(out["opponent"]) == ["LAR", "LV", "JAC"]


def test_an_unknown_stat_type_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stat_type must be one of"):
        refresh_pfr_advstats(tmp_path, stat_type="def", seasons=[2024])  # type: ignore[arg-type]


def test_season_and_week_are_int64_not_int32(
    tmp_path: Path, fake_id_map_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_id_map(tmp_path, fake_id_map_df, monkeypatch)

    out = _normalize_one_season("rec", _raw(), tmp_path)

    assert out["season"].dtype == "int64"
    assert out["week"].dtype == "int64"
