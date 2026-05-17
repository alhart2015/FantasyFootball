"""aggregate_to_season — pure-function tests over ProjectionWeeklySchema rows."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

from projections.aggregation import aggregate_to_season
from projections.distributions import (
    Distribution,
    ParametricGamma,
    ParametricNormal,
    pack_per_stat_params,
)
from projections.schemas import (
    _PYARROW_STR,
    DistributionFamily,
    ProjectionSeasonSchema,
    Ruleset,
    Stat,
)
from projections.scoring import derive_row_seed, score_distribution

_RULESET = Ruleset.espn_ppr()


def _build_weekly_row(
    *,
    gsis_id: str = "00-0033873",
    season: int = 2024,
    week: int = 1,
    position: str = "WR",
    team: str = "KC",
    opponent: str = "BAL",
    family: str = DistributionFamily.SAMPLED_SUMMARY.value,
    ruleset_name: str | None = None,
    rec_yards_mean: float = 50.0,
    rec_yards_std: float = 18.0,
    rec_shape: float = 4.0,
    rec_scale: float = 0.7,
    model_id: str = "baseline:wr:abcdef12:2018-2023",
) -> dict[str, Any]:
    rs_name = ruleset_name if ruleset_name is not None else _RULESET.name
    per_stat_dists: dict[Stat, Distribution] = {
        Stat.RECEIVING_YARDS: ParametricNormal(mean=rec_yards_mean, std=rec_yards_std),
        Stat.RECEPTIONS: ParametricGamma(shape=rec_shape, scale=rec_scale),
    }
    blob = pack_per_stat_params(per_stat_dists)
    seed = derive_row_seed(gsis_id=gsis_id, season=season, week=week, ruleset_name=rs_name)
    points = score_distribution(per_stat_dists, _RULESET, n_samples=10_000, seed=seed)
    return {
        "gsis_id": gsis_id,
        "season": season,
        "week": week,
        "position": position,
        "team": team,
        "opponent": opponent,
        "ruleset": rs_name,
        "family": family,
        "params": blob,
        "mean": points.mean(),
        "p10": points.quantile(0.1),
        "p50": points.quantile(0.5),
        "p90": points.quantile(0.9),
        "model_id": model_id,
        "generated_at": pd.Timestamp(datetime.now(UTC)).as_unit("us"),
    }


def _to_weekly_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ("gsis_id", "team", "opponent", "ruleset", "family", "model_id", "position"):
        df[col] = df[col].astype(_PYARROW_STR)
    return df


def test_empty_input_returns_empty_validated_frame() -> None:
    empty = pd.DataFrame(
        columns=[
            "gsis_id",
            "season",
            "week",
            "position",
            "team",
            "opponent",
            "ruleset",
            "family",
            "params",
            "mean",
            "p10",
            "p50",
            "p90",
            "model_id",
            "generated_at",
        ]
    )
    out = aggregate_to_season(empty, ruleset=_RULESET)
    assert out.empty
    ProjectionSeasonSchema.validate(out)


def test_single_player_single_week_n_weeks_is_one() -> None:
    weekly = _to_weekly_frame([_build_weekly_row(week=1)])
    out = aggregate_to_season(weekly, ruleset=_RULESET)
    assert len(out) == 1
    assert out["n_weeks"].iloc[0] == 1
    assert out["season_mean"].iloc[0] == pytest.approx(weekly["mean"].iloc[0], rel=0.02)


def test_single_player_multi_week_quantiles_widen() -> None:
    """Sum of independent random variables has wider quantile spread than any
    single one. season_p90 - season_p10 should exceed any single week's
    p90 - p10 because variances add."""
    weekly = _to_weekly_frame([_build_weekly_row(week=w) for w in range(1, 6)])
    out = aggregate_to_season(weekly, ruleset=_RULESET)
    assert len(out) == 1
    assert out["n_weeks"].iloc[0] == 5
    season_spread = out["season_p90"].iloc[0] - out["season_p10"].iloc[0]
    weekly_spread = weekly["p90"].iloc[0] - weekly["p10"].iloc[0]
    # 5 independent weeks: variance scales 5x => std scales sqrt(5)x => spread
    # also scales sqrt(5) ~= 2.24x. Be conservative; require >= 1.5x.
    assert season_spread >= 1.5 * weekly_spread


def test_multi_player_one_row_per_player() -> None:
    weekly = _to_weekly_frame(
        [_build_weekly_row(gsis_id="00-0033873", week=w) for w in range(1, 4)]
        + [_build_weekly_row(gsis_id="00-0035640", week=w) for w in range(1, 4)]
    )
    out = aggregate_to_season(weekly, ruleset=_RULESET)
    assert len(out) == 2
    assert set(out["gsis_id"]) == {"00-0033873", "00-0035640"}
    assert (out["n_weeks"] == 3).all()


def test_traded_player_modal_position() -> None:
    """Same gsis_id appears with two positions across weeks; modal value wins."""
    rows = [
        _build_weekly_row(week=1, position="WR"),
        _build_weekly_row(week=2, position="WR"),
        _build_weekly_row(week=3, position="WR"),
        _build_weekly_row(week=4, position="RB"),
    ]
    weekly = _to_weekly_frame(rows)
    out = aggregate_to_season(weekly, ruleset=_RULESET)
    assert len(out) == 1
    assert out["position"].iloc[0] == "WR"


def test_disallowed_family_raises() -> None:
    """family=SAMPLED is not in the allowed set; the codec doesn't decode raw
    sample arrays from the params blob, so the guard rejects it. Error message
    should list the new allowed set."""
    rows = [
        _build_weekly_row(week=1, family=DistributionFamily.SAMPLED.value),
    ]
    weekly = _to_weekly_frame(rows)
    # New error message lists the full allowed set (sorted); match a stable
    # substring rather than asserting the full ordering.
    with pytest.raises(ValueError, match="SAMPLED_SUMMARY"):
        aggregate_to_season(weekly, ruleset=_RULESET)


def test_schema_valid_but_disallowed_family_lists_allowed_set() -> None:
    """A family value that is in the ProjectionWeeklySchema enum but NOT in
    aggregate_to_season's allowed set (e.g., NORMAL — a per-stat-only family,
    never a row-level tag) gets past schema validation and is rejected by the
    function's own guard. The error message must list the currently allowed
    set so callers can see what's accepted."""
    rows = [_build_weekly_row(week=1, family=DistributionFamily.NORMAL.value)]
    weekly = _to_weekly_frame(rows)
    with pytest.raises(ValueError) as excinfo:
        aggregate_to_season(weekly, ruleset=_RULESET)
    msg = str(excinfo.value)
    assert "NORMAL" in msg  # surfaced in 'found' list
    assert "SAMPLED_SUMMARY" in msg
    assert "QUANTILE" in msg
    assert "MIXED" in msg


def test_mixed_ruleset_input_raises() -> None:
    rows = [
        _build_weekly_row(week=1, ruleset_name="ESPN_PPR"),
        _build_weekly_row(week=2, ruleset_name="ESPN_HALF"),
    ]
    weekly = _to_weekly_frame(rows)
    with pytest.raises(ValueError, match="ruleset"):
        aggregate_to_season(weekly, ruleset=_RULESET)


def test_aggregate_is_deterministic_across_calls() -> None:
    weekly = _to_weekly_frame([_build_weekly_row(week=w) for w in range(1, 6)])
    out_a = aggregate_to_season(weekly, ruleset=_RULESET)
    out_b = aggregate_to_season(weekly, ruleset=_RULESET)
    for col in ("season_mean", "season_p10", "season_p50", "season_p90"):
        assert (out_a[col].to_numpy() == out_b[col].to_numpy()).all()


def test_output_validates_against_projection_season_schema() -> None:
    weekly = _to_weekly_frame([_build_weekly_row(week=w) for w in range(1, 6)])
    out = aggregate_to_season(weekly, ruleset=_RULESET)
    ProjectionSeasonSchema.validate(out)
    assert "season_mean" in out.columns
    assert "n_weeks" in out.columns
    assert "position" in out.columns
