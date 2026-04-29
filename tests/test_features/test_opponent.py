"""Opponent-strength helper tests — Plan 9.

Schedule-of-strength residual EPA-allowed by play type. The new helper
replaces the v1 opp_allowed_fppg from Plan 2a.
"""

from __future__ import annotations

import math

import pandas as pd

from projections.features._opponent import opp_epa_allowed_residual


def _make_pbp_row(
    *,
    play_id: int,
    week: int,
    posteam: str,
    defteam: str,
    play_type: str,
    epa: float,
    qb_scramble: float = 0.0,
    sack: float = 0.0,
) -> dict[str, object]:
    """Helper to keep test data construction terse."""
    return {
        "play_id": play_id,
        "game_id": f"2024_{week:02d}_{posteam}_{defteam}",
        "season": 2024,
        "week": week,
        "posteam": posteam,
        "defteam": defteam,
        "play_type": play_type,
        "qb_dropback": 1.0 if play_type == "pass" else 0.0,
        "qb_scramble": qb_scramble,
        "sack": sack,
        "rush_attempt": 1.0 if play_type == "run" else 0.0,
        "pass_attempt": 1.0 if play_type == "pass" else 0.0,
        "epa": epa,
        "wpa": 0.0,
        "success": 1.0 if epa > 0 else 0.0,
    }


def test_opp_epa_allowed_residual_zero_when_offenses_are_league_mean() -> None:
    """If every offense has the same overall mean EPA and the defense allows
    that same mean, the residual is zero."""
    rows: list[dict[str, object]] = []
    play_id = 1
    for week in [1, 2, 3, 4]:
        # Two offenses, one play each, both at mean EPA = 0.10 vs DEF.
        for posteam in ("OFF1", "OFF2"):
            rows.append(
                _make_pbp_row(
                    play_id=play_id,
                    week=week,
                    posteam=posteam,
                    defteam="DEF",
                    play_type="pass",
                    epa=0.10,
                )
            )
            play_id += 1

    pbp = pd.DataFrame(rows)
    result = opp_epa_allowed_residual(pbp, play_type="pass", n_weeks=4)

    # DEF target_week=5 (after weeks 1-4 trailing window): residual ≈ 0.
    row = result[(result["week"] == 5) & (result["opp_team"] == "DEF")]
    assert len(row) == 1
    assert math.isclose(float(row.iloc[0]["opp_epa_allowed_residual"]), 0.0, abs_tol=1e-9)


def test_opp_epa_allowed_residual_positive_against_weak_offenses() -> None:
    """Defense faces only weak offenses but still allows mean EPA at the
    league level — implies they performed worse than expected (residual > 0)."""
    rows: list[dict[str, object]] = []
    play_id = 1
    # OFF_WEAK has overall mean EPA = -0.20 across all opponents.
    # DEF allows OFF_WEAK 0.0 EPA per play (above their average → DEF is weak).
    for week in [1, 2, 3, 4]:
        # OFF_WEAK vs DEF — DEF allows 0.0 EPA (vs OFF_WEAK overall mean -0.20).
        rows.append(
            _make_pbp_row(
                play_id=play_id,
                week=week,
                posteam="OFF_WEAK",
                defteam="DEF",
                play_type="pass",
                epa=0.0,
            )
        )
        play_id += 1
        # OFF_WEAK vs NO_DEF — sets OFF_WEAK overall mean.
        rows.append(
            _make_pbp_row(
                play_id=play_id,
                week=week,
                posteam="OFF_WEAK",
                defteam="NO_DEF",
                play_type="pass",
                epa=-0.40,  # so OFF_WEAK overall mean = (-0.40 + 0.0)/2 = -0.20
            )
        )
        play_id += 1

    pbp = pd.DataFrame(rows)
    result = opp_epa_allowed_residual(pbp, play_type="pass", n_weeks=4)

    row = result[(result["week"] == 5) & (result["opp_team"] == "DEF")]
    assert len(row) == 1
    # DEF allowed 0.0 vs OFF_WEAK whose overall mean is -0.20 → residual = +0.20.
    assert math.isclose(float(row.iloc[0]["opp_epa_allowed_residual"]), 0.20, abs_tol=1e-9)


def test_opp_epa_allowed_residual_negative_against_strong_offenses() -> None:
    """Defense holds strong offenses below their average → residual < 0."""
    rows: list[dict[str, object]] = []
    play_id = 1
    for week in [1, 2, 3, 4]:
        # OFF_STRONG vs DEF — DEF allows 0.0 EPA.
        rows.append(
            _make_pbp_row(
                play_id=play_id,
                week=week,
                posteam="OFF_STRONG",
                defteam="DEF",
                play_type="pass",
                epa=0.0,
            )
        )
        play_id += 1
        # OFF_STRONG vs NO_DEF — sets OFF_STRONG overall mean.
        rows.append(
            _make_pbp_row(
                play_id=play_id,
                week=week,
                posteam="OFF_STRONG",
                defteam="NO_DEF",
                play_type="pass",
                epa=0.40,  # OFF_STRONG overall = (0.40 + 0.0)/2 = +0.20
            )
        )
        play_id += 1

    pbp = pd.DataFrame(rows)
    result = opp_epa_allowed_residual(pbp, play_type="pass", n_weeks=4)

    row = result[(result["week"] == 5) & (result["opp_team"] == "DEF")]
    assert len(row) == 1
    # DEF allowed 0.0 vs OFF_STRONG whose mean is +0.20 → residual = -0.20.
    assert math.isclose(float(row.iloc[0]["opp_epa_allowed_residual"]), -0.20, abs_tol=1e-9)


def test_opp_epa_allowed_residual_pass_filter_includes_sacks_and_scrambles() -> None:
    """play_type='pass' must include sacks and scrambles. The setup below
    produces a known non-zero residual; a buggy classifier that excludes
    scrambles from the pass filter gives a DIFFERENT residual value (≈0
    instead of ≈+0.083), so this test catches misclassification.

    Setup:
      week 1: OFF pass vs DEF (epa=0.0), OFF pass vs OTHER (epa=0.0)
      week 2: OFF scramble vs DEF (epa=+0.5), OFF designed-run vs OTHER (epa=+1.0)

    Correct classifier (scramble counts as pass):
      OFF pass plays in window: 2 pure passes (epa=0.0) + 1 scramble (epa=0.5)
        → OFF pass mean = (0 + 0 + 0.5)/3 ≈ 0.167
      DEF residuals (per-play): w1 pass = 0 - 0.167 = -0.167
                                w2 scramble = 0.5 - 0.167 = +0.333
      Per-week means: w1 = -0.167, w2 = +0.333
      Two-stage trailing mean: mean(-0.167, +0.333) = +0.0833

    Buggy classifier (scramble excluded):
      OFF pass plays: only 2 pure passes (epa=0.0) → OFF pass mean = 0
      DEF residuals: w1 pass = 0 (only play in window vs DEF)
      Trailing mean = 0
    """
    rows = [
        # Week 1: pure passes
        _make_pbp_row(
            play_id=1,
            week=1,
            posteam="OFF",
            defteam="DEF",
            play_type="pass",
            epa=0.0,
        ),
        _make_pbp_row(
            play_id=2,
            week=1,
            posteam="OFF",
            defteam="OTHER",
            play_type="pass",
            epa=0.0,
        ),
        # Week 2: scramble vs DEF (must be classified as pass), designed run vs OTHER
        _make_pbp_row(
            play_id=3,
            week=2,
            posteam="OFF",
            defteam="DEF",
            play_type="run",
            epa=0.5,
            qb_scramble=1.0,
        ),
        _make_pbp_row(
            play_id=4,
            week=2,
            posteam="OFF",
            defteam="OTHER",
            play_type="run",
            epa=1.0,  # designed run, NOT pass
        ),
    ]
    pbp = pd.DataFrame(rows)
    result = opp_epa_allowed_residual(pbp, play_type="pass", n_weeks=4)

    row = result[(result["week"] == 3) & (result["opp_team"] == "DEF")]
    assert len(row) == 1
    # Asserting the EXACT correct value catches misclassification.
    expected = (0.0 - (0.5 / 3.0) + 0.5 - (0.5 / 3.0)) / 2.0  # = +0.0833...
    assert math.isclose(float(row.iloc[0]["opp_epa_allowed_residual"]), expected, abs_tol=1e-9), (
        f"Expected residual ≈ {expected:.4f}; got "
        f"{float(row.iloc[0]['opp_epa_allowed_residual']):.4f}. "
        "If this is ≈0.0, the classifier is wrongly excluding scrambles from pass."
    )


def test_opp_epa_allowed_residual_run_filter_excludes_scrambles() -> None:
    """play_type='run' must EXCLUDE scrambles. The setup below produces a
    known zero residual when scrambles are correctly excluded; a buggy
    classifier that wrongly includes scrambles gives a DIFFERENT residual
    (positive), so this test catches misclassification.

    Setup:
      week 1: OFF designed-run vs DEF (epa=0.5), OFF designed-run vs OTHER (epa=0.5)
      week 2: OFF scramble vs DEF (epa=+2.0), OFF designed-run vs OTHER (epa=0.5)

    Correct classifier (scramble excluded from run):
      OFF run plays in window: 3 designed runs (epa=0.5) → OFF run mean = 0.5
      DEF run plays vs OFF in window: w1 designed (residual = 0.5 - 0.5 = 0)
      Per-week means: w1 = 0
      Trailing mean = 0

    Buggy classifier (scramble wrongly included in run):
      OFF run plays: 3 designed (epa=0.5) + 1 scramble (epa=2.0)
        → OFF run mean = (0.5 + 0.5 + 0.5 + 2.0)/4 = 0.875
      DEF run plays vs OFF: w1 designed (residual = 0.5 - 0.875 = -0.375)
                            w2 scramble (residual = 2.0 - 0.875 = +1.125)
      Per-week means: w1 = -0.375, w2 = +1.125
      Trailing mean = mean(-0.375, +1.125) = +0.375
    """
    rows = [
        # Week 1: designed runs
        _make_pbp_row(
            play_id=1,
            week=1,
            posteam="OFF",
            defteam="DEF",
            play_type="run",
            epa=0.5,
        ),
        _make_pbp_row(
            play_id=2,
            week=1,
            posteam="OFF",
            defteam="OTHER",
            play_type="run",
            epa=0.5,
        ),
        # Week 2: scramble vs DEF (must be EXCLUDED from run filter), designed run vs OTHER
        _make_pbp_row(
            play_id=3,
            week=2,
            posteam="OFF",
            defteam="DEF",
            play_type="run",
            epa=2.0,
            qb_scramble=1.0,  # scramble — exclude!
        ),
        _make_pbp_row(
            play_id=4,
            week=2,
            posteam="OFF",
            defteam="OTHER",
            play_type="run",
            epa=0.5,
        ),
    ]
    pbp = pd.DataFrame(rows)
    result = opp_epa_allowed_residual(pbp, play_type="run", n_weeks=4)

    row = result[(result["week"] == 3) & (result["opp_team"] == "DEF")]
    assert len(row) == 1
    # Correct: only the designed-run vs DEF in w1 is counted, residual = 0.
    # Buggy (scramble included): residual ≈ +0.375 (week-2 scramble contributes).
    assert math.isclose(float(row.iloc[0]["opp_epa_allowed_residual"]), 0.0, abs_tol=1e-9), (
        f"Expected residual ≈ 0.0; got "
        f"{float(row.iloc[0]['opp_epa_allowed_residual']):.4f}. "
        "If this is ≈+0.375, the classifier is wrongly including scrambles in the run filter."
    )


def test_opp_epa_allowed_residual_target_week_shifted_plus_one() -> None:
    """Residual computed from weeks 1-4 joins onto opponent's week-5 row."""
    rows: list[dict[str, object]] = []
    play_id = 1
    for week in [1, 2, 3, 4]:
        rows.append(
            _make_pbp_row(
                play_id=play_id,
                week=week,
                posteam="OFF",
                defteam="DEF",
                play_type="pass",
                epa=0.0,
            )
        )
        play_id += 1
    pbp = pd.DataFrame(rows)
    result = opp_epa_allowed_residual(pbp, play_type="pass", n_weeks=4)

    # The week=5 row is what offense-side feature builders join onto.
    weeks_emitted = set(result["week"].unique())
    assert 5 in weeks_emitted


def test_opp_epa_allowed_residual_expanding_window_for_early_weeks() -> None:
    """Weeks 2-4 emit rows with underfilled trailing windows (expanding)."""
    rows: list[dict[str, object]] = []
    play_id = 1
    for week in [1, 2, 3, 4]:
        rows.append(
            _make_pbp_row(
                play_id=play_id,
                week=week,
                posteam="OFF",
                defteam="DEF",
                play_type="pass",
                epa=0.0,
            )
        )
        play_id += 1
    pbp = pd.DataFrame(rows)
    result = opp_epa_allowed_residual(pbp, play_type="pass", n_weeks=4)

    # Weeks 2-5 get rows. Week 1 cannot — no prior-week data exists.
    weeks_emitted = sorted(result["week"].unique())
    assert weeks_emitted == [2, 3, 4, 5]


def test_opp_epa_allowed_residual_skips_no_play_and_nan_epa() -> None:
    """Rows with epa=NaN are dropped (no_play / pre-snap penalties)."""
    rows: list[dict[str, object]] = [
        _make_pbp_row(
            play_id=1,
            week=1,
            posteam="OFF",
            defteam="DEF",
            play_type="pass",
            epa=0.1,
        ),
        # epa=NaN → drop.
        {
            "play_id": 2,
            "game_id": "x",
            "season": 2024,
            "week": 1,
            "posteam": "OFF",
            "defteam": "DEF",
            "play_type": "no_play",
            "qb_dropback": None,
            "qb_scramble": None,
            "sack": None,
            "rush_attempt": None,
            "pass_attempt": None,
            "epa": None,
            "wpa": None,
            "success": None,
        },
    ]
    pbp = pd.DataFrame(rows)
    # Should not raise; the NaN-epa row is filtered before averaging.
    result = opp_epa_allowed_residual(pbp, play_type="pass", n_weeks=4)
    assert "opp_epa_allowed_residual" in result.columns
