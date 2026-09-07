"""D/ST scoring — the dot product, and the Critts rules it has to reproduce.

Spec: docs/superpowers/specs/2026-09-06-dst-projections-design.md §1.3, §1.4.
"""

from __future__ import annotations

import pytest

from projections.ingest.espn_league import parse_ruleset
from projections.schemas import Ruleset
from projections.scoring.dst import DST_STAT_LABELS, DstScoringError, score_dst

#: The Critts league's 26 D/ST categories, confirmed 2026-09-06 against the league's own
#: League Info screens (26/26, no unmatched labels either way). Spec §1.4.
CRITTS_DST_POINTS: dict[str, float] = {
    "89": 5.0,
    "90": 4.0,
    "91": 3.0,
    "92": 1.0,
    "93": 6.0,
    "95": 2.0,
    "96": 2.0,
    "97": 2.0,
    "98": 2.0,
    "99": 1.0,
    "101": 6.0,
    "102": 6.0,
    "103": 6.0,
    "104": 6.0,
    "123": -1.0,
    "124": -3.0,
    "125": -5.0,
    "128": 5.0,
    "129": 3.0,
    "130": 2.0,
    "132": -1.0,
    "133": -3.0,
    "134": -5.0,
    "135": -6.0,
    "136": -7.0,
    "206": 2.0,
}


def _critts_ruleset() -> Ruleset:
    return Ruleset(
        name="ESPN_HALF",
        dst_stat_points=tuple(sorted(CRITTS_DST_POINTS.items(), key=lambda kv: int(kv[0]))),
    )


def _payload(items: list[dict[str, object]]) -> dict[str, object]:
    return {"settings": {"scoringSettings": {"scoringItems": items}}}


# --- the dot product -------------------------------------------------------


def test_scores_a_shutout_line() -> None:
    """3 sacks, 1 INT, 0 points allowed, 180 yards allowed:
    3(1) + 1(2) + 5 + 3 = 13."""
    ruleset = _critts_ruleset()
    line = {"99": 3.0, "95": 1.0, "89": 1.0, "129": 1.0}
    assert score_dst(line, ruleset) == pytest.approx(13.0)


def test_negative_categories_subtract() -> None:
    """2 sacks, 41 points allowed, 470 yards allowed: 2 - 3 - 5 = -6."""
    ruleset = _critts_ruleset()
    assert score_dst({"99": 2.0, "124": 1.0, "134": 1.0}, ruleset) == pytest.approx(-6.0)


def test_unscored_stat_ids_contribute_nothing() -> None:
    """ESPN's D/ST vector carries ~40 ids; a league scores a subset. The rest must be
    silently ignored, not raise and not leak in."""
    ruleset = _critts_ruleset()
    assert score_dst({"99": 1.0, "120": 22.4, "127": 349.7, "999": 5.0}, ruleset) == pytest.approx(
        1.0
    )


def test_fractional_projections_score_fractionally() -> None:
    """Projections are expected values, not integers — 2.46 sacks is a normal input."""
    ruleset = _critts_ruleset()
    assert score_dst({"99": 2.46, "95": 0.84}, ruleset) == pytest.approx(2.46 + 1.68)


def test_empty_stat_vector_is_zero_not_an_error() -> None:
    assert score_dst({}, _critts_ruleset()) == pytest.approx(0.0)


def test_a_ruleset_without_dst_scoring_raises() -> None:
    """A league with a D/ST slot and no D/ST scoring is a contradiction. Returning 0.0 would
    rank every defense as identically worthless — a wrong answer shaped like a real one."""
    with pytest.raises(DstScoringError, match="no D/ST scoring categories"):
        score_dst({"99": 3.0}, Ruleset.espn_half())


# --- parse_ruleset ---------------------------------------------------------


def test_parse_reads_dst_scoring_out_of_points_overrides() -> None:
    """The bug at the heart of #166: base points are 0 and the real value is in
    pointsOverrides["16"], so these categories were invisible to the parser."""
    ruleset, _ = parse_ruleset(
        _payload(
            [
                {"statId": 53, "points": 0.5},
                {"statId": 99, "points": 0, "pointsOverrides": {"16": 1.0}},
                {"statId": 95, "points": 0, "pointsOverrides": {"16": 2.0}},
            ]
        )
    )
    assert ruleset.dst_points_by_stat_id == {"99": 1.0, "95": 2.0}
    assert ruleset.scores_dst


def test_parse_ignores_base_points_without_a_dst_override() -> None:
    """Skill categories must not leak into the D/ST map. Measured 2026-09-06: a D/ST stat
    vector contains no skill stat ids, so a base-valued category cannot reach a defense.
    Including them would still score correctly (multiplied by zero every time) but would make
    every league look like it scores defenses."""
    ruleset, _ = parse_ruleset(
        _payload([{"statId": 53, "points": 0.5}, {"statId": 4, "points": 4.0}])
    )
    assert ruleset.dst_stat_points == ()


def test_parse_reports_that_dst_categories_were_found() -> None:
    _, notes = parse_ruleset(
        _payload(
            [
                {"statId": 53, "points": 0.5},
                {"statId": 99, "points": 0, "pointsOverrides": {"16": 1.0}},
            ]
        )
    )
    assert any("D/ST scoring categories parsed" in note for note in notes)


#: The "Ruleset applies one value to all positions" warning parse_ruleset emits for a
#: per-position override it cannot represent. Matched on this phrase rather than on the word
#: "pointsOverrides", which also appears in the informational D/ST note.
_OVERRIDE_WARNING = "applies one value to all positions"


def test_parse_does_not_warn_about_the_dst_override_itself() -> None:
    """Position 16 is modelled now. Warning about it would emit 26 misleading notes on a
    normal league and bury the ones that matter."""
    _, notes = parse_ruleset(
        _payload(
            [
                {"statId": 53, "points": 0.5},
                {"statId": 99, "points": 0, "pointsOverrides": {"16": 1.0}},
            ]
        )
    )
    assert not any(_OVERRIDE_WARNING in note for note in notes)


def test_parse_still_warns_about_non_dst_overrides() -> None:
    _, notes = parse_ruleset(
        _payload([{"statId": 53, "points": 0.5, "pointsOverrides": {"4": 1.5, "16": 2.0}}])
    )
    warnings = [note for note in notes if _OVERRIDE_WARNING in note]
    assert len(warnings) == 1
    assert "'4'" in warnings[0]
    assert "'16'" not in warnings[0]


def test_a_league_without_dst_scoring_parses_to_no_dst_points() -> None:
    ruleset, _ = parse_ruleset(
        _payload([{"statId": 53, "points": 1.0}, {"statId": 4, "points": 4}])
    )
    assert ruleset.dst_stat_points == ()
    assert not ruleset.scores_dst


# --- invariants ------------------------------------------------------------


def test_ruleset_stays_hashable_with_dst_points() -> None:
    """Ruleset's docstring promises hashability so it can be cached. A dict field would
    break that; this is why dst_stat_points is a tuple of pairs."""
    assert hash(_critts_ruleset()) == hash(_critts_ruleset())


def test_ruleset_stays_frozen() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _critts_ruleset().dst_stat_points = ()


def test_labels_cover_every_critts_category() -> None:
    """DST_STAT_LABELS is display-only, but a label missing for a scored category means a
    report shows a bare number id to the user."""
    assert set(CRITTS_DST_POINTS) <= set(DST_STAT_LABELS)


def test_labels_are_not_used_for_scoring() -> None:
    """A ruleset scoring an id with no label must still score. The label map must never
    become a gate on the scoring path."""
    ruleset = Ruleset(name="ESPN_HALF", dst_stat_points=(("31337", 3.0),))
    assert "31337" not in DST_STAT_LABELS
    assert score_dst({"31337": 2.0}, ruleset) == pytest.approx(6.0)
