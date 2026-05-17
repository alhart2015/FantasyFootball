"""Tests for src/projections/preseason/model.py."""

from __future__ import annotations

import logging
from pathlib import Path

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


def test_naive_predict_fallback_to_prior_2() -> None:
    """Veteran missing prior_1 but has prior_2: falls through to prior_2."""
    features = _make_features_row(
        prior_1_season_games_played=pd.array([pd.NA], dtype="Int64"),
        prior_1_season_per_game_passing_yards=pd.array([pd.NA], dtype="Float32"),
        prior_2_season_games_played=pd.array([14], dtype="Int64"),
        prior_2_season_per_game_passing_yards=pd.array([300.0], dtype="Float32"),
    )
    model = NaivePreseasonModel()
    out = model.predict_season_distribution(features, ruleset=Ruleset.espn_ppr())
    # Falls back: 300 * 16 = 4800.
    assert float(out["passing_yards_season_total_mean"].iloc[0]) == pytest.approx(4800.0)


def test_naive_predict_fallback_to_prior_3() -> None:
    """Veteran missing prior_1 and prior_2: falls through to prior_3."""
    features = _make_features_row(
        prior_1_season_games_played=pd.array([pd.NA], dtype="Int64"),
        prior_2_season_games_played=pd.array([pd.NA], dtype="Int64"),
        prior_3_season_games_played=pd.array([16], dtype="Int64"),
        prior_3_season_per_game_passing_yards=pd.array([250.0], dtype="Float32"),
    )
    model = NaivePreseasonModel()
    out = model.predict_season_distribution(features, ruleset=Ruleset.espn_ppr())
    assert float(out["passing_yards_season_total_mean"].iloc[0]) == pytest.approx(4000.0)


def test_naive_predict_drops_player_with_all_priors_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Veteran with no prior 1/2/3 history: dropped with WARNING."""
    features = _make_features_row(
        is_rookie=False,
        prior_1_season_games_played=pd.array([pd.NA], dtype="Int64"),
        prior_2_season_games_played=pd.array([pd.NA], dtype="Int64"),
        prior_3_season_games_played=pd.array([pd.NA], dtype="Int64"),
    )
    model = NaivePreseasonModel()
    with caplog.at_level(logging.WARNING):
        out = model.predict_season_distribution(features, ruleset=Ruleset.espn_ppr())
    assert len(out) == 0
    assert "no_prior_3_seasons" in caplog.text


def test_naive_fit_populates_rookie_glms() -> None:
    """fit() should train one GLM per (position, stat) cell."""
    # Synthetic 2-year training data — 4 QB rookies + 4 WR rookies.
    weekly = pd.DataFrame(
        [
            # 2021 QBs (rookie year):
            {
                "gsis_id": "00-3000001",
                "season": 2021,
                "week": 1,
                "position": "QB",
                "team": "KC",
                "passing_yards": 250.0,
                "passing_tds": 1,
                "interceptions": 1,
                "rushing_yards": 5.0,
                "rushing_tds": 0,
                "receptions": 0,
                "receiving_yards": 0.0,
                "receiving_tds": 0,
            },
            {
                "gsis_id": "00-3000002",
                "season": 2021,
                "week": 1,
                "position": "QB",
                "team": "BUF",
                "passing_yards": 180.0,
                "passing_tds": 0,
                "interceptions": 2,
                "rushing_yards": 15.0,
                "rushing_tds": 0,
                "receptions": 0,
                "receiving_yards": 0.0,
                "receiving_tds": 0,
            },
            {
                "gsis_id": "00-3000003",
                "season": 2022,
                "week": 1,
                "position": "QB",
                "team": "DEN",
                "passing_yards": 300.0,
                "passing_tds": 2,
                "interceptions": 1,
                "rushing_yards": 0.0,
                "rushing_tds": 0,
                "receptions": 0,
                "receiving_yards": 0.0,
                "receiving_tds": 0,
            },
            {
                "gsis_id": "00-3000004",
                "season": 2022,
                "week": 1,
                "position": "QB",
                "team": "NYJ",
                "passing_yards": 200.0,
                "passing_tds": 1,
                "interceptions": 1,
                "rushing_yards": 20.0,
                "rushing_tds": 0,
                "receptions": 0,
                "receiving_yards": 0.0,
                "receiving_tds": 0,
            },
            # WRs:
            {
                "gsis_id": "00-3000005",
                "season": 2021,
                "week": 1,
                "position": "WR",
                "team": "DET",
                "passing_yards": 0.0,
                "passing_tds": 0,
                "interceptions": 0,
                "rushing_yards": 0.0,
                "rushing_tds": 0,
                "receptions": 5,
                "receiving_yards": 60.0,
                "receiving_tds": 1,
            },
            {
                "gsis_id": "00-3000006",
                "season": 2021,
                "week": 1,
                "position": "WR",
                "team": "PHI",
                "passing_yards": 0.0,
                "passing_tds": 0,
                "interceptions": 0,
                "rushing_yards": 0.0,
                "rushing_tds": 0,
                "receptions": 3,
                "receiving_yards": 40.0,
                "receiving_tds": 0,
            },
            {
                "gsis_id": "00-3000007",
                "season": 2022,
                "week": 1,
                "position": "WR",
                "team": "ATL",
                "passing_yards": 0.0,
                "passing_tds": 0,
                "interceptions": 0,
                "rushing_yards": 0.0,
                "rushing_tds": 0,
                "receptions": 6,
                "receiving_yards": 85.0,
                "receiving_tds": 1,
            },
            {
                "gsis_id": "00-3000008",
                "season": 2022,
                "week": 1,
                "position": "WR",
                "team": "JAC",
                "passing_yards": 0.0,
                "passing_tds": 0,
                "interceptions": 0,
                "rushing_yards": 0.0,
                "rushing_tds": 0,
                "receptions": 4,
                "receiving_yards": 55.0,
                "receiving_tds": 0,
            },
        ]
    )
    weekly["season"] = weekly["season"].astype("int32")
    weekly["week"] = weekly["week"].astype("int32")

    # Each rookie has a draft_picks row for their rookie year.
    draft = pd.DataFrame(
        [
            ("00-3000001", 2021, 1, 5),
            ("00-3000002", 2021, 2, 45),
            ("00-3000003", 2022, 1, 10),
            ("00-3000004", 2022, 3, 80),
            ("00-3000005", 2021, 1, 15),
            ("00-3000006", 2021, 4, 110),
            ("00-3000007", 2022, 1, 8),
            ("00-3000008", 2022, 2, 50),
        ],
        columns=["gsis_id", "season", "round", "pick"],
    ).astype({"season": "int32", "round": "Int64", "pick": "Int64"})
    id_map = pd.DataFrame({"gsis_id": draft["gsis_id"], "full_name": "Test", "birth_date": pd.NaT})

    model = NaivePreseasonModel()
    model.fit(weekly_stats=weekly, draft_picks=draft, id_map=id_map)

    # Expect entries per (position, stat) in _STATS_BY_POSITION:
    # QB stats: 5 (passing_yards, passing_tds, passing_interceptions, rushing_yards, rushing_tds)
    # WR stats: 5 (receptions, receiving_yards, receiving_tds, rushing_yards, rushing_tds)
    qb_keys = [k for k in model._rookie_glm if k[0] == "QB"]
    wr_keys = [k for k in model._rookie_glm if k[0] == "WR"]
    assert len(qb_keys) == 5
    assert len(wr_keys) == 5
    # Each GLM coefficient is an (intercept, slope) pair.
    intercept, slope = model._rookie_glm[("QB", "passing_yards")]
    assert isinstance(intercept, float)
    assert isinstance(slope, float)


def test_naive_predict_rookie_drafted_player() -> None:
    """Rookie with draft pick: predicted = exp(intercept + slope * log(pick + 1))."""
    weekly = pd.DataFrame(
        [
            {
                "gsis_id": "00-3000001",
                "season": 2021,
                "week": w,
                "position": "WR",
                "team": "KC",
                "passing_yards": 0.0,
                "passing_tds": 0,
                "interceptions": 0,
                "rushing_yards": 0.0,
                "rushing_tds": 0,
                "receptions": 5.0,
                "receiving_yards": 70.0,
                "receiving_tds": 0.4,
            }
            for w in range(1, 18)
        ]
    )
    weekly["season"] = weekly["season"].astype("int32")
    weekly["week"] = weekly["week"].astype("int32")

    draft = pd.DataFrame(
        [("00-3000001", 2021, 1, 10)],
        columns=["gsis_id", "season", "round", "pick"],
    ).astype({"season": "int32", "round": "Int64", "pick": "Int64"})
    id_map = pd.DataFrame({"gsis_id": ["00-3000001"], "full_name": ["x"], "birth_date": [pd.NaT]})
    model = NaivePreseasonModel()
    model.fit(weekly_stats=weekly, draft_picks=draft, id_map=id_map)

    features = _make_features_row(
        gsis_id="00-4000001",
        position="WR",
        is_rookie=True,
        years_exp=pd.array([0], dtype="Int64"),
        draft_round=pd.array([1], dtype="Int64"),
        draft_pick_overall=pd.array([10], dtype="Int64"),
    )
    out = model.predict_season_distribution(features, ruleset=Ruleset.espn_ppr())
    assert len(out) == 1
    # Should produce a non-zero receiving_yards prediction.
    assert float(out["receiving_yards_season_total_mean"].iloc[0]) > 0


def test_naive_predict_rookie_udfa_imputed_to_pick_300() -> None:
    """UDFA rookie (no draft_pick_overall): imputed to pick=300."""
    weekly = pd.DataFrame(
        [
            {
                "gsis_id": "00-3000001",
                "season": 2021,
                "week": 1,
                "position": "WR",
                "team": "KC",
                "passing_yards": 0.0,
                "passing_tds": 0,
                "interceptions": 0,
                "rushing_yards": 0.0,
                "rushing_tds": 0,
                "receptions": 4,
                "receiving_yards": 50.0,
                "receiving_tds": 0,
            },
        ]
    )
    weekly["season"] = weekly["season"].astype("int32")
    weekly["week"] = weekly["week"].astype("int32")
    draft = pd.DataFrame(
        [("00-3000001", 2021, 1, 10)],
        columns=["gsis_id", "season", "round", "pick"],
    ).astype({"season": "int32", "round": "Int64", "pick": "Int64"})
    id_map = pd.DataFrame({"gsis_id": ["00-3000001"], "full_name": ["x"], "birth_date": [pd.NaT]})

    model = NaivePreseasonModel()
    model.fit(weekly_stats=weekly, draft_picks=draft, id_map=id_map)

    features = _make_features_row(
        gsis_id="00-5000001",
        position="WR",
        is_rookie=True,
        draft_round=pd.array([pd.NA], dtype="Int64"),
        draft_pick_overall=pd.array([pd.NA], dtype="Int64"),
    )
    out = model.predict_season_distribution(features, ruleset=Ruleset.espn_ppr())
    assert len(out) == 1
    # Should still produce a value (imputed pick=300).
    assert float(out["receiving_yards_season_total_mean"].iloc[0]) >= 0


def test_naive_predict_fpts_uses_canonical_scoring() -> None:
    """fpts must be computed via projections.scoring (canonical path), not local math."""
    features = _make_features_row(
        prior_1_season_games_played=pd.array([17], dtype="Int64"),
        prior_1_season_per_game_passing_yards=pd.array([250.0], dtype="float32"),
        prior_1_season_per_game_passing_tds=pd.array([2.0], dtype="float32"),
        prior_1_season_per_game_passing_interceptions=pd.array([1.0], dtype="float32"),
        prior_1_season_per_game_rushing_yards=pd.array([10.0], dtype="float32"),
        prior_1_season_per_game_rushing_tds=pd.array([0.1], dtype="float32"),
    )
    model = NaivePreseasonModel()
    out = model.predict_season_distribution(features, ruleset=Ruleset.espn_ppr())
    # Computed via canonical scoring coefficients:
    # passing_yards / 25  = 250 * 16 / 25  = 160
    # passing_tds * 4     = 2.0 * 16 * 4   = 128
    # interceptions * -2  = 1.0 * 16 * -2  = -32
    # rushing_yards / 10  = 10 * 16 / 10   = 16
    # rushing_tds * 6     = 0.1 * 16 * 6   = 9.6
    # Total: 281.6
    expected = 250 * 16 / 25 + 2 * 16 * 4 + 1 * 16 * -2 + 10 * 16 / 10 + 0.1 * 16 * 6
    assert float(out["season_total_fpts_mean"].iloc[0]) == pytest.approx(expected, rel=0.02)


def test_naive_model_save_and_load_roundtrip(tmp_path: Path) -> None:
    weekly = pd.DataFrame(
        [
            {
                "gsis_id": "00-3000001",
                "season": 2021,
                "week": 1,
                "position": "WR",
                "team": "KC",
                "passing_yards": 0.0,
                "passing_tds": 0,
                "interceptions": 0,
                "rushing_yards": 0.0,
                "rushing_tds": 0,
                "receptions": 5,
                "receiving_yards": 70.0,
                "receiving_tds": 1,
            },
        ]
    )
    weekly["season"] = weekly["season"].astype("int32")
    weekly["week"] = weekly["week"].astype("int32")
    draft = pd.DataFrame(
        [("00-3000001", 2021, 1, 10)],
        columns=["gsis_id", "season", "round", "pick"],
    ).astype({"season": "int32", "round": "Int64", "pick": "Int64"})
    id_map = pd.DataFrame({"gsis_id": ["00-3000001"], "full_name": ["x"], "birth_date": [pd.NaT]})

    model = NaivePreseasonModel()
    model.fit(weekly_stats=weekly, draft_picks=draft, id_map=id_map)
    path = tmp_path / "naive-preseason-test.joblib"
    model.save(path)
    assert path.exists()

    reloaded = NaivePreseasonModel.load(path)
    assert reloaded.model_id == model.model_id
    assert reloaded._rookie_glm == model._rookie_glm
