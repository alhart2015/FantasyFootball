"""Depth chart ingest tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.ingest import refresh_depth_charts
from projections.ingest.depth_charts import (
    _derive_weekly_snapshots_from_new_format,
    _parse_depth_rank,
)
from projections.schemas import DepthChartsSchema, SchedulesSchema
from projections.store import read_partition


def test_refresh_depth_charts_writes_partition(
    tmp_path: Path,
    fake_depth_charts_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.depth_charts._fetch_raw_depth_charts",
        lambda seasons: fake_depth_charts_df,
    )
    written = refresh_depth_charts(tmp_path, seasons=[2024])
    assert len(written) == 1

    df = read_partition(tmp_path / "raw", "depth_charts", season=2024)
    DepthChartsSchema.validate(df)
    assert set(df["gsis_id"]) == {
        "00-0036322",
        "00-0034857",
        "00-0034796",
        "00-0030506",
    }


def test_refresh_depth_charts_renames_club_code_to_team(
    tmp_path: Path,
    fake_depth_charts_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.depth_charts._fetch_raw_depth_charts",
        lambda seasons: fake_depth_charts_df,
    )
    refresh_depth_charts(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "depth_charts", season=2024)
    assert "team" in df.columns
    assert "club_code" not in df.columns
    assert set(df["team"]) == {"MIN", "KC", "PHI"}


def test_refresh_depth_charts_filters_unsupported_positions(
    tmp_path: Path,
    fake_depth_charts_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OL, IDP positions are dropped — depth_charts also lists non-fantasy positions."""
    extra = pd.DataFrame(
        [
            {
                "season": 2024,
                "club_code": "MIN",
                "week": 3,
                "depth_team": "LT1",
                "last_name": "Doe",
                "first_name": "John",
                "formation": "Offense",
                "gsis_id": "00-0099998",
                "jersey_number": 71,
                "position": "OL",
                "elias_id": "DOE99998",
                "depth_position": 1,
                "football_name": "John Doe",
            }
        ]
    )
    with_ol = pd.concat([fake_depth_charts_df, extra], ignore_index=True)
    monkeypatch.setattr(
        "projections.ingest.depth_charts._fetch_raw_depth_charts",
        lambda seasons: with_ol,
    )
    refresh_depth_charts(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "depth_charts", season=2024)
    assert "00-0099998" not in df["gsis_id"].tolist()


def test_refresh_depth_charts_idempotent(
    tmp_path: Path,
    fake_depth_charts_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.depth_charts._fetch_raw_depth_charts",
        lambda seasons: fake_depth_charts_df,
    )
    refresh_depth_charts(tmp_path, seasons=[2024])
    refresh_depth_charts(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "depth_charts", season=2024)
    assert len(df) == 4


# --- _parse_depth_rank unit tests ---


def test_parse_depth_rank_prefers_numeric_depth_position() -> None:
    """If depth_position is a non-null int, that's the rank."""
    rank, warned = _parse_depth_rank(depth_team="LWR", depth_position=2)
    assert rank == 2
    assert warned is False


def test_parse_depth_rank_parses_trailing_digit_from_depth_team() -> None:
    """Falls back to the trailing digit in depth_team when depth_position is null."""
    rank, warned = _parse_depth_rank(depth_team="WR3", depth_position=None)
    assert rank == 3
    assert warned is False


def test_parse_depth_rank_falls_back_to_one_for_unrankable_label() -> None:
    """Unrankable label (no trailing digit, no depth_position) → 1, warned."""
    rank, warned = _parse_depth_rank(depth_team="LWR", depth_position=None)
    assert rank == 1
    assert warned is True


def test_parse_depth_rank_clamps_above_ten() -> None:
    """Parsed rank > 10 (impossible per schema) clamps to 10."""
    rank, warned = _parse_depth_rank(depth_team="WR99", depth_position=None)
    assert rank == 10
    assert warned is True


# --- _derive_weekly_snapshots_from_new_format (2025+ snapshot-by-timestamp feed) ---


def _make_schedules(
    games: list[tuple[int, str, str, str]],
) -> pd.DataFrame:
    """Build a SchedulesSchema-shaped frame from `(week, home, away, kickoff_iso)`.

    `kickoff_iso` is parsed as UTC.
    """
    df = pd.DataFrame(
        {
            "season": [2025] * len(games),
            "week": [g[0] for g in games],
            "game_id": [f"2025_{g[0]:02d}_{g[2]}_{g[1]}" for g in games],
            "home_team": [g[1] for g in games],
            "away_team": [g[2] for g in games],
            "kickoff": pd.to_datetime([g[3] for g in games], utc=True).as_unit("us"),
            "spread_line": [0.0] * len(games),
            "total_line": [45.0] * len(games),
            "home_moneyline": [0] * len(games),
            "away_moneyline": [0] * len(games),
            "surface": ["grass"] * len(games),
            "roof": ["outdoors"] * len(games),
            "temp": [70] * len(games),
            "wind": [0] * len(games),
        }
    )
    return SchedulesSchema.validate(df)


def _make_snapshots(
    rows: list[dict[str, object]],
) -> pd.DataFrame:
    """Build a raw new-format snapshot frame. `rows` is a list of column dicts."""
    df = pd.DataFrame(rows)
    # Coerce dtypes to match the real nflreadpy payload.
    if "pos_slot" in df.columns:
        df["pos_slot"] = df["pos_slot"].astype("int32")
    if "pos_rank" in df.columns:
        df["pos_rank"] = df["pos_rank"].astype("int32")
    return df


def test_derive_picks_closest_prior_snapshot() -> None:
    """Given two snapshots before kickoff, the larger-dt one is chosen.

    KC plays at 2025-09-04T13:00Z. Snapshots at 2025-09-02T10:00Z and
    2025-09-04T08:00Z both qualify (strict <); we must pick the latter.
    """
    schedules = _make_schedules([(1, "KC", "BAL", "2025-09-04T13:00:00Z")])
    snapshots = _make_snapshots(
        [
            {
                "dt": "2025-09-02T10:00:00Z",
                "team": "KC",
                "gsis_id": "00-0000001",
                "pos_abb": "WR",
                "pos_slot": 1,
                "pos_rank": 1,
            },
            {
                "dt": "2025-09-04T08:00:00Z",
                "team": "KC",
                "gsis_id": "00-0000002",
                "pos_abb": "WR",
                "pos_slot": 1,
                "pos_rank": 1,
            },
        ]
    )
    out = _derive_weekly_snapshots_from_new_format(snapshots, schedules)
    assert list(out["gsis_id"]) == ["00-0000002"], (
        "closest-prior rule should pick the snapshot with the largest dt strictly before kickoff"
    )


def test_derive_excludes_snapshot_at_exact_kickoff_instant() -> None:
    """A snapshot whose dt == kickoff is excluded (strict <)."""
    schedules = _make_schedules([(1, "KC", "BAL", "2025-09-04T13:00:00Z")])
    snapshots = _make_snapshots(
        [
            {
                "dt": "2025-09-04T13:00:00Z",  # exactly at kickoff
                "team": "KC",
                "gsis_id": "00-0000001",
                "pos_abb": "WR",
                "pos_slot": 1,
                "pos_rank": 1,
            },
            {
                "dt": "2025-09-04T12:59:00Z",  # one minute before — should be chosen
                "team": "KC",
                "gsis_id": "00-0000002",
                "pos_abb": "WR",
                "pos_slot": 1,
                "pos_rank": 1,
            },
        ]
    )
    out = _derive_weekly_snapshots_from_new_format(snapshots, schedules)
    assert list(out["gsis_id"]) == ["00-0000002"]


def test_derive_clamps_pos_rank_to_ten() -> None:
    """pos_rank above 10 (deep depth chart) clamps to depth_rank=10."""
    schedules = _make_schedules([(1, "KC", "BAL", "2025-09-04T13:00:00Z")])
    snapshots = _make_snapshots(
        [
            {
                "dt": "2025-09-02T10:00:00Z",
                "team": "KC",
                "gsis_id": "00-0000099",
                "pos_abb": "WR",
                "pos_slot": 1,
                "pos_rank": 15,
            }
        ]
    )
    out = _derive_weekly_snapshots_from_new_format(snapshots, schedules)
    assert list(out["depth_rank"]) == [10]
    assert list(out["depth_team"]) == ["10"]


def test_derive_synthesizes_depth_team_from_rank() -> None:
    """depth_team = str(depth_rank); legacy on-disk format uses {'1', '2', '3'}."""
    schedules = _make_schedules([(1, "KC", "BAL", "2025-09-04T13:00:00Z")])
    snapshots = _make_snapshots(
        [
            {
                "dt": "2025-09-02T10:00:00Z",
                "team": "KC",
                "gsis_id": f"00-000000{i}",
                "pos_abb": "WR",
                "pos_slot": 1,
                "pos_rank": i,
            }
            for i in range(1, 4)
        ]
    )
    out = _derive_weekly_snapshots_from_new_format(snapshots, schedules)
    assert sorted(out["depth_team"].tolist()) == ["1", "2", "3"]
    assert sorted(out["depth_rank"].tolist()) == [1, 2, 3]


def test_derive_filters_to_position_enum_values() -> None:
    """Defensive / special-teams rows (LDE, KR, etc.) are dropped."""
    schedules = _make_schedules([(1, "KC", "BAL", "2025-09-04T13:00:00Z")])
    snapshots = _make_snapshots(
        [
            {
                "dt": "2025-09-02T10:00:00Z",
                "team": "KC",
                "gsis_id": "00-0000001",
                "pos_abb": "WR",
                "pos_slot": 1,
                "pos_rank": 1,
            },
            {
                "dt": "2025-09-02T10:00:00Z",
                "team": "KC",
                "gsis_id": "00-0000002",
                "pos_abb": "LDE",  # defensive — drop
                "pos_slot": 1,
                "pos_rank": 1,
            },
            {
                "dt": "2025-09-02T10:00:00Z",
                "team": "KC",
                "gsis_id": "00-0000003",
                "pos_abb": "KR",  # special teams — drop
                "pos_slot": 1,
                "pos_rank": 1,
            },
        ]
    )
    out = _derive_weekly_snapshots_from_new_format(snapshots, schedules)
    assert list(out["gsis_id"]) == ["00-0000001"]
    assert list(out["position"]) == ["WR"]


def test_derive_skips_team_with_no_prior_snapshot() -> None:
    """Team-week with kickoff before any snapshot is skipped (warning logged)."""
    schedules = _make_schedules(
        [
            (1, "KC", "BAL", "2025-09-04T13:00:00Z"),
            (1, "PHI", "NYG", "2025-09-04T13:00:00Z"),
        ]
    )
    # Only KC has a prior snapshot. PHI has no snapshots at all.
    snapshots = _make_snapshots(
        [
            {
                "dt": "2025-09-02T10:00:00Z",
                "team": "KC",
                "gsis_id": "00-0000001",
                "pos_abb": "WR",
                "pos_slot": 1,
                "pos_rank": 1,
            }
        ]
    )
    out = _derive_weekly_snapshots_from_new_format(snapshots, schedules)
    assert list(out["team"]) == ["KC"]


def test_derive_skips_bye_week_teams() -> None:
    """A team not present in schedules for a given week emits no rows that week.

    The snapshot contains a player on TEN but TEN has no week-1 game in schedules,
    so no TEN rows should appear in the output.
    """
    schedules = _make_schedules([(1, "KC", "BAL", "2025-09-04T13:00:00Z")])
    snapshots = _make_snapshots(
        [
            {
                "dt": "2025-09-02T10:00:00Z",
                "team": "KC",
                "gsis_id": "00-0000001",
                "pos_abb": "WR",
                "pos_slot": 1,
                "pos_rank": 1,
            },
            {
                "dt": "2025-09-02T10:00:00Z",
                "team": "TEN",
                "gsis_id": "00-0000002",
                "pos_abb": "WR",
                "pos_slot": 1,
                "pos_rank": 1,
            },
        ]
    )
    out = _derive_weekly_snapshots_from_new_format(snapshots, schedules)
    assert set(out["team"]) == {"KC"}


def test_derive_dedupes_player_per_team_week() -> None:
    """A player listed twice in the same snapshot for the same team produces one row."""
    schedules = _make_schedules([(1, "KC", "BAL", "2025-09-04T13:00:00Z")])
    snapshots = _make_snapshots(
        [
            {
                "dt": "2025-09-02T10:00:00Z",
                "team": "KC",
                "gsis_id": "00-0000001",
                "pos_abb": "WR",
                "pos_slot": 1,
                "pos_rank": 1,
            },
            {
                "dt": "2025-09-02T10:00:00Z",
                "team": "KC",
                "gsis_id": "00-0000001",  # duplicate
                "pos_abb": "WR",
                "pos_slot": 1,
                "pos_rank": 2,
            },
        ]
    )
    out = _derive_weekly_snapshots_from_new_format(snapshots, schedules)
    assert len(out) == 1


def test_derive_round_trips_via_depth_charts_schema() -> None:
    """Output is `DepthChartsSchema`-validated and contains only legacy-schema columns."""
    schedules = _make_schedules([(1, "KC", "BAL", "2025-09-04T13:00:00Z")])
    snapshots = _make_snapshots(
        [
            {
                "dt": "2025-09-02T10:00:00Z",
                "team": "KC",
                "gsis_id": "00-0000001",
                "pos_abb": "QB",
                "pos_slot": 9,
                "pos_rank": 1,
            }
        ]
    )
    out = _derive_weekly_snapshots_from_new_format(snapshots, schedules)
    DepthChartsSchema.validate(out)
    assert set(out.columns) == {
        "gsis_id",
        "season",
        "week",
        "team",
        "position",
        "depth_team",
        "depth_rank",
    }


def test_derive_normalizes_raw_team_codes_jax_la() -> None:
    """Raw feed uses `JAX`/`LA`; schedules use `JAC`/`LAR` (post-normalize).

    Without normalizing the raw `team` column BEFORE the per-team groupby/merge,
    the team-match drops all JAC/LAR rows silently. Regression test for
    real-data bug observed during the 2025 ingest first pass.
    """
    schedules = _make_schedules(
        [
            (1, "JAC", "IND", "2025-09-04T13:00:00Z"),
            (1, "LAR", "SEA", "2025-09-04T16:00:00Z"),
        ]
    )
    snapshots = _make_snapshots(
        [
            {
                "dt": "2025-09-02T10:00:00Z",
                "team": "JAX",  # raw nflverse alias for JAC
                "gsis_id": "00-0000001",
                "pos_abb": "WR",
                "pos_slot": 1,
                "pos_rank": 1,
            },
            {
                "dt": "2025-09-02T10:00:00Z",
                "team": "LA",  # raw nflverse alias for LAR
                "gsis_id": "00-0000002",
                "pos_abb": "WR",
                "pos_slot": 1,
                "pos_rank": 1,
            },
        ]
    )
    out = _derive_weekly_snapshots_from_new_format(snapshots, schedules)
    assert set(out["team"]) == {"JAC", "LAR"}


def test_refresh_depth_charts_new_format_reads_schedules_from_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the raw payload is new-format and schedules is not passed, refresh
    reads schedules from `data/raw/schedules/season=<s>/` on disk."""
    schedules = _make_schedules([(1, "KC", "BAL", "2025-09-04T13:00:00Z")])
    # Persist schedules to the expected on-disk location.
    from projections.store import write_partition

    write_partition(tmp_path / "raw", "schedules", schedules, season=2025, week=None)

    snapshots = _make_snapshots(
        [
            {
                "dt": "2025-09-02T10:00:00Z",
                "team": "KC",
                "gsis_id": "00-0000001",
                "pos_abb": "WR",
                "pos_slot": 1,
                "pos_rank": 1,
            }
        ]
    )
    monkeypatch.setattr(
        "projections.ingest.depth_charts._fetch_raw_depth_charts",
        lambda seasons: snapshots,
    )

    refresh_depth_charts(tmp_path, seasons=[2025])
    df = read_partition(tmp_path / "raw", "depth_charts", season=2025)
    DepthChartsSchema.validate(df)
    assert list(df["gsis_id"]) == ["00-0000001"]


def test_refresh_depth_charts_new_format_raises_when_schedules_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refresh raises FileNotFoundError if new-format payload but no schedules on disk."""
    snapshots = _make_snapshots(
        [
            {
                "dt": "2025-09-02T10:00:00Z",
                "team": "KC",
                "gsis_id": "00-0000001",
                "pos_abb": "WR",
                "pos_slot": 1,
                "pos_rank": 1,
            }
        ]
    )
    monkeypatch.setattr(
        "projections.ingest.depth_charts._fetch_raw_depth_charts",
        lambda seasons: snapshots,
    )
    with pytest.raises(FileNotFoundError, match="schedules"):
        refresh_depth_charts(tmp_path, seasons=[2025])
