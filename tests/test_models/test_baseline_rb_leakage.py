"""RB baseline leakage test. Mirrors test_baseline_leakage.py's WR test.

Strategy: fit on a feature build through week W. Mutate weekly_stats rows at
season=Y, week>=W+1. Re-build features through W and refit. Assert each
fitted regressor's coefficients are byte-identical pre and post.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.models import rb_baseline
from projections.schemas import Stat


def test_rb_baseline_fit_does_not_use_post_as_of_week_data(
    baseline_features_rb: pd.DataFrame, baseline_weekly_stats_rb: pd.DataFrame
) -> None:
    ws = baseline_weekly_stats_rb[baseline_weekly_stats_rb["season"] == 2024].copy()
    feats = baseline_features_rb[baseline_features_rb["season"] == 2024].copy()

    model_a = rb_baseline()
    model_a.fit(features=feats, weekly_stats=ws)

    # Mutate week-8 truth dramatically.
    ws_mut = ws.copy()
    mask = ws_mut["week"] >= 8
    ws_mut.loc[mask, "rushing_yards"] = 0.0
    ws_mut.loc[mask, "rushing_tds"] = 0
    ws_mut.loc[mask, "carries"] = 0
    ws_mut.loc[mask, "receptions"] = 0
    ws_mut.loc[mask, "receiving_yards"] = 999.0
    ws_mut.loc[mask, "receiving_tds"] = 9

    feats_through_7 = feats[feats["week"] <= 7].copy()
    ws_mut_through_7 = ws_mut[ws_mut["week"] <= 7].copy()
    ws_orig_through_7 = ws[ws["week"] <= 7].copy()

    model_b = rb_baseline()
    model_b.fit(features=feats_through_7, weekly_stats=ws_orig_through_7)
    model_c = rb_baseline()
    model_c.fit(features=feats_through_7, weekly_stats=ws_mut_through_7)

    for stat in model_b.target_stats:
        np.testing.assert_array_equal(
            model_b.ridges[stat].coef_,
            model_c.ridges[stat].coef_,
            err_msg=f"Leakage detected on stat {stat}",
        )
        assert model_b.ridges[stat].alpha_ == model_c.ridges[stat].alpha_

    # Control: full-fixture vs week<=7 SHOULD differ on the strongest signal.
    coef_a = model_a.ridges[Stat.RUSHING_YARDS].coef_
    coef_b = model_b.ridges[Stat.RUSHING_YARDS].coef_
    assert not np.array_equal(coef_a, coef_b), (
        "Sanity check: full-fixture fit and week<=7 fit should produce "
        "different coefficients on RUSHING_YARDS"
    )
