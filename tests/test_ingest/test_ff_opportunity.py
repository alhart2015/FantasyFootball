"""Expected-fantasy-production ingest tests.

This source's upstream dtypes are looser than anywhere else in the repo (string season, float
week, null player ids on team-aggregate rows) and the normalization order is what makes them
survive a schema. One test per trap.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.ingest import refresh_ff_opportunity
from projections.ingest.ff_opportunity import _STAT_COLS, _normalize_one_season
from projections.schemas import FfOpportunitySchema
from projections.store import read_partition


def _raw(**overrides: list[object]) -> pd.DataFrame:
    """A three-row `load_ff_opportunity` payload in upstream's real dtypes.

    `season` really is a string and `week` really is a float here -- see the module docstring.
    """
    base: dict[str, list[object]] = {
        "season": ["2024", "2024", "2024"],
        "week": [1.0, 1.0, 1.0],
        "player_id": ["00-0034857", "00-0036322", "00-0034796"],
        "posteam": ["KC", "MIN", "PHI"],
        "position": ["QB", "WR", "RB"],
    }
    for col in _STAT_COLS:
        base[col] = [1.5, 2.5, 3.5]
    # Upstream also ships the `_diff` family and a `*_team` mirror of everything; the schema's
    # strict="filter" is what keeps them out, so the fixture carries a representative pair.
    base["total_fantasy_points_diff"] = [0.1, 0.2, 0.3]
    base["total_fantasy_points_team"] = [40.0, 41.0, 42.0]
    base.update(overrides)
    return pd.DataFrame(base)


def test_refresh_writes_a_validated_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "projections.ingest.ff_opportunity._fetch_raw_ff_opportunity", lambda seasons: _raw()
    )

    written = refresh_ff_opportunity(tmp_path, seasons=[2024])

    assert len(written) == 1
    df = read_partition(tmp_path / "raw", "ff_opportunity", season=2024)
    FfOpportunitySchema.validate(df)
    assert len(df) == 3


def test_string_season_becomes_int64() -> None:
    """Unique to this source: every other nflverse release here returns int32."""
    out = _normalize_one_season(_raw())

    assert out["season"].dtype == "int64"
    assert set(out["season"]) == {2024}


def test_float_week_becomes_int64() -> None:
    out = _normalize_one_season(_raw(week=[3.0, 3.0, 3.0]))

    assert out["week"].dtype == "int64"
    assert set(out["week"]) == {3}


def test_team_aggregate_rows_are_dropped() -> None:
    """Rows with no player carry a null id and a null position -- roughly 420 a season."""
    raw = _raw(
        player_id=["00-0034857", None, "00-0034796"],
        position=["QB", None, "RB"],
    )

    out = _normalize_one_season(raw)

    assert list(out["gsis_id"]) == ["00-0034857", "00-0034796"]


def test_unmodelled_positions_are_filtered() -> None:
    """The model scores everyone who touched the ball: a lineman on a fumble recovery, a punter
    on a fake. We model six positions."""
    raw = _raw(position=["QB", "OL", "DB"])

    out = _normalize_one_season(raw)

    assert list(out["position"]) == ["QB"]


def test_posteam_is_renamed_and_normalized() -> None:
    raw = _raw(posteam=["JAX", "LA", "PHI"])

    out = _normalize_one_season(raw)

    assert list(out["team"]) == ["JAC", "LAR", "PHI"]
    assert "posteam" not in out.columns


def test_postseason_weeks_are_kept() -> None:
    """Upstream runs to week 22. Stored as-is, matching SnapCountsSchema; the playoff filter is
    the consumer's call (issue #123)."""
    out = _normalize_one_season(_raw(week=[22.0, 22.0, 22.0]))

    assert set(out["week"]) == {22}


def test_diff_and_team_mirror_columns_are_not_stored() -> None:
    """`_diff` is a subtraction of two stored columns and `*_team` is a groupby of them; storing
    either invites it drifting from its inputs."""
    out = _normalize_one_season(_raw())

    assert "total_fantasy_points_diff" not in out.columns
    assert "total_fantasy_points_team" not in out.columns


def test_negative_yardage_is_accepted() -> None:
    """A sack loses passing yards and a stuffed carry loses rushing yards; the yardage fields
    are deliberately unbounded below where the count fields are not."""
    raw = _raw(
        pass_yards_gained=[-7.0, 0.0, 0.0],
        rush_yards_gained=[0.0, 0.0, -3.0],
    )

    out = _normalize_one_season(raw)

    assert out["pass_yards_gained"].min() == -7.0
    assert out["rush_yards_gained"].min() == -3.0


def test_placeholder_gsis_rows_are_dropped() -> None:
    raw = _raw(player_id=["00-0034857", "WAS569019", "00-0034796"])

    out = _normalize_one_season(raw)

    assert list(out["gsis_id"]) == ["00-0034857", "00-0034796"]


def test_every_expected_column_survives_validation() -> None:
    """The `_exp` columns are the entire point of this source; a rename upstream that silently
    dropped one would leave a schema-valid table with no signal in it."""
    out = _normalize_one_season(_raw())

    for col in _STAT_COLS:
        assert col in out.columns, col
    # Ten: completions, receptions, three yardage, three touchdown, interceptions, and the
    # diagnostic total. Pinned as a literal so an upstream rename shows up as a count change
    # rather than a quietly narrower table.
    assert sum(c.endswith("_exp") for c in out.columns) == 10
