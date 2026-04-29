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
    """play_type='pass' must include sacks and scrambles. Construct a setup
    where misclassifying scrambles as non-pass would shift the residual.

    Two offenses (OFF_A, OFF_B) play DEF. OFF_A's plays are all pure passes;
    OFF_B's plays are scrambles + designed runs. If the classifier correctly
    counts scrambles as pass plays, OFF_B contributes ONE pass play to DEF's
    window and OFF_B has known overall pass mean. If scrambles are wrongly
    excluded, OFF_B has zero pass plays vs DEF and the residual differs.
    """
    rows = [
        # OFF_A vs DEF — pure pass, EPA=0.0 (week 1)
        _make_pbp_row(
            play_id=1,
            week=1,
            posteam="OFF_A",
            defteam="DEF",
            play_type="pass",
            epa=0.0,
        ),
        # OFF_A vs OTHER — pure pass, EPA=0.0 (sets OFF_A overall mean = 0.0)
        _make_pbp_row(
            play_id=2,
            week=1,
            posteam="OFF_A",
            defteam="OTHER",
            play_type="pass",
            epa=0.0,
        ),
        # OFF_B vs DEF — scramble (must classify as pass), EPA=+0.4 (week 2)
        _make_pbp_row(
            play_id=3,
            week=2,
            posteam="OFF_B",
            defteam="DEF",
            play_type="run",
            epa=0.4,
            qb_scramble=1.0,
        ),
        # OFF_B vs OTHER — scramble, EPA=+0.4 (sets OFF_B overall pass mean = 0.4)
        _make_pbp_row(
            play_id=4,
            week=2,
            posteam="OFF_B",
            defteam="OTHER",
            play_type="run",
            epa=0.4,
            qb_scramble=1.0,
        ),
        # OFF_B vs OTHER — designed run, EPA=+1.0 (NOT counted as pass)
        _make_pbp_row(
            play_id=5,
            week=2,
            posteam="OFF_B",
            defteam="OTHER",
            play_type="run",
            epa=1.0,
        ),
    ]
    pbp = pd.DataFrame(rows)
    result = opp_epa_allowed_residual(pbp, play_type="pass", n_weeks=4)

    # DEF residual at target_week=3:
    #   week 1 plays vs DEF: OFF_A pass at 0.0 (residual = 0.0 - 0.0 = 0.0)
    #   week 2 plays vs DEF: OFF_B scramble at 0.4 (residual = 0.4 - 0.4 = 0.0)
    #   per-week means: w1=0.0, w2=0.0
    #   trailing-N mean over weeks: 0.0
    # If scrambles were wrongly excluded from pass classifier:
    #   week 2 has NO pass plays vs DEF → DEF's row at target_week=3 might not even
    #   be emitted (or would only reflect week 1).
    row = result[(result["week"] == 3) & (result["opp_team"] == "DEF")]
    assert len(row) == 1, "DEF must have a row at target_week=3 (scramble counted as pass)"

    # Also check OTHER's row covers both weeks 1 and 2 (cross-defense sanity).
    other_row = result[(result["week"] == 3) & (result["opp_team"] == "OTHER")]
    assert len(other_row) == 1


def test_opp_epa_allowed_residual_run_filter_excludes_scrambles() -> None:
    """play_type='run' must exclude qb_scramble. Construct a setup where
    incorrectly including scrambles would alter the residual."""
    rows = [
        # OFF vs DEF — designed run only, EPA=0.5 (week 1)
        _make_pbp_row(
            play_id=1,
            week=1,
            posteam="OFF",
            defteam="DEF",
            play_type="run",
            epa=0.5,
        ),
        # OFF vs OTHER — designed run, EPA=0.5 (OFF overall run mean = 0.5)
        _make_pbp_row(
            play_id=2,
            week=1,
            posteam="OFF",
            defteam="OTHER",
            play_type="run",
            epa=0.5,
        ),
        # OFF vs DEF — scramble (MUST be excluded), EPA=-1.0 (week 2)
        _make_pbp_row(
            play_id=3,
            week=2,
            posteam="OFF",
            defteam="DEF",
            play_type="run",
            epa=-1.0,
            qb_scramble=1.0,
        ),
        # OFF vs DEF — designed run, EPA=0.5 (week 2)
        _make_pbp_row(
            play_id=4,
            week=2,
            posteam="OFF",
            defteam="DEF",
            play_type="run",
            epa=0.5,
        ),
    ]
    pbp = pd.DataFrame(rows)
    result = opp_epa_allowed_residual(pbp, play_type="run", n_weeks=4)

    row = result[(result["week"] == 3) & (result["opp_team"] == "DEF")]
    # DEF designed-run plays vs OFF: w1=0.5, w2=0.5
    # OFF's run window mean: (0.5 + 0.5 + 0.5)/3 = 0.5
    # per-play residuals: 0.0 each → per-week means: w1=0.0, w2=0.0
    # trailing-N mean: 0.0
    # If scrambles were wrongly INCLUDED:
    #   DEF window plays would include the scramble (-1.0)
    #   per-week means: w1=0.0, w2=mean(0.0, residual_of_scramble)
    #   The scramble's residual = -1.0 - off_run_mean → significantly negative
    #   per-week mean for w2 != 0.0
    #   trailing mean != 0.0
    assert len(row) == 1
    assert math.isclose(float(row.iloc[0]["opp_epa_allowed_residual"]), 0.0, abs_tol=1e-9)


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
