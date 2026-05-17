"""Tests for src/projections/preseason/model.py."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.preseason.model import NaivePreseasonModel, PreseasonModel
from projections.schemas import PreseasonFeaturesSchema, Ruleset


def test_naive_preseason_model_implements_protocol() -> None:
    """NaivePreseasonModel should satisfy the PreseasonModel Protocol at runtime."""
    m = NaivePreseasonModel()
    assert isinstance(m, PreseasonModel)
    assert m.model_id == "naive-preseason-v1"


def _make_features_row(**kwargs: object) -> pd.DataFrame:
    """Construct a one-row PreseasonFeaturesSchema-valid frame with overrides.

    Default row is a 7-year-vet QB in KC with no prior stats populated; tests
    override the keys they care about. All Optional prior_* columns are stubbed
    to NA so the schema accepts the frame.
    """
    base: dict[str, object] = {
        "gsis_id": "00-1000001",
        "season": pd.array([2026], dtype="int32"),
        "position": "QB",
        "team": "KC",
        "depth_chart_rank": pd.array([1], dtype="Int64"),
        "age": pd.array([29.0], dtype="float32"),
        "years_exp": pd.array([7], dtype="Int64"),
        "is_rookie": False,
        "draft_round": pd.array([1], dtype="Int64"),
        "draft_pick_overall": pd.array([10], dtype="Int64"),
    }
    for n in (1, 2, 3):
        base[f"prior_{n}_season_games_played"] = pd.array([pd.NA], dtype="Int64")
        for stat in (
            "passing_yards",
            "passing_tds",
            "passing_interceptions",
            "rushing_yards",
            "rushing_tds",
            "receptions",
            "receiving_yards",
            "receiving_tds",
        ):
            base[f"prior_{n}_season_per_game_{stat}"] = pd.array([pd.NA], dtype="Float32")
    base.update(kwargs)
    df_data: dict[str, object] = {}
    for k, v in base.items():
        if isinstance(v, pd.api.extensions.ExtensionArray):
            df_data[k] = v
        elif isinstance(v, list):
            df_data[k] = v
        else:
            df_data[k] = [v]
    df = pd.DataFrame(df_data)
    return PreseasonFeaturesSchema.validate(df)


def test_naive_predict_veteran_branch_prior_1() -> None:
    """Veteran with prior-1 stats: predicted = prior_1_per_game * 16."""
    features = _make_features_row(
        prior_1_season_games_played=pd.array([17], dtype="Int64"),
        prior_1_season_per_game_passing_yards=pd.array([275.0], dtype="float32"),
        prior_1_season_per_game_passing_tds=pd.array([2.0], dtype="float32"),
        prior_1_season_per_game_passing_interceptions=pd.array([0.5], dtype="float32"),
        prior_1_season_per_game_rushing_yards=pd.array([12.0], dtype="float32"),
        prior_1_season_per_game_rushing_tds=pd.array([0.1], dtype="float32"),
    )
    model = NaivePreseasonModel()
    out = model.predict_season_distribution(features, ruleset=Ruleset.espn_ppr())
    assert len(out) == 1
    # Veteran branch: 275 * 16 = 4400 passing yards.
    assert float(out["passing_yards_season_total_mean"].iloc[0]) == pytest.approx(4400.0)
    # Degenerate distribution: all quantiles equal.
    assert float(out["passing_yards_season_total_p10"].iloc[0]) == pytest.approx(4400.0)
    assert float(out["passing_yards_season_total_p90"].iloc[0]) == pytest.approx(4400.0)
