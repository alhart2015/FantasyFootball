"""Tests for `scripts/auction_field_bakeoff.build_field`.

The script had no tests; these cover the opponent-mix knob added for the field-composition sweep,
because a silently-wrong mix would produce a plausible-looking table for a room nobody ran.
"""

from __future__ import annotations

from itertools import pairwise

import pytest
from scripts.auction_field_bakeoff import build_field

from projections.draft.assistant.auction.market import (
    AggressiveBot,
    BalancedBot,
    PatientValueBot,
)


def test_build_field_n_patient_none_reproduces_the_historical_mix() -> None:
    """The default path must re-simulate identically: every prior run used the every-5th rule."""
    field = build_field("overbidder", 0.2, n_bots=11)
    hoarders = [i for i, b in enumerate(field) if isinstance(b, PatientValueBot)]
    assert hoarders == [4, 9]  # 9 aggressive / 2 conservative


@pytest.mark.parametrize("n_bots", range(1, 16))
def test_build_field_n_patient_places_exactly_that_many(n_bots: int) -> None:
    """Exact count for every k, including k == n_bots.

    A plain `round((i + 0.5) * n / k)` spread silently collides once k approaches n — it returned
    6 hoarders for n_patient=11 — so a sweep would have quietly run a different mix than the one
    it reported. Hence the exhaustive check rather than a couple of spot values.
    """
    for k in range(n_bots + 1):
        field = build_field("overbidder", 0.2, n_bots=n_bots, n_patient=k)
        assert len(field) == n_bots
        assert sum(isinstance(b, PatientValueBot) for b in field) == k


def test_build_field_n_patient_spreads_rather_than_clusters() -> None:
    """Hoarders must not bunch at one end — that would confound the mix with seat adjacency."""
    field = build_field("overbidder", 0.2, n_bots=11, n_patient=3)
    hoarders = [i for i, b in enumerate(field) if isinstance(b, PatientValueBot)]
    assert hoarders == [1, 5, 9]
    gaps = [b - a for a, b in pairwise(hoarders)]
    assert max(gaps) - min(gaps) <= 1  # evenly spaced


@pytest.mark.parametrize("n_patient", [12, -1])
def test_build_field_n_patient_out_of_range_raises(n_patient: int) -> None:
    with pytest.raises(ValueError, match="n_patient"):
        build_field("overbidder", 0.2, n_bots=11, n_patient=n_patient)


# --- the knob must be honored or refused, never silently dropped ---------------------------------
# `_run_chunk` writes n_patient into the chunk payload unconditionally and `_guard_homogeneous`
# treats it as a config key, so a path that accepts the knob without acting on it produces an
# artifact labelled with a room that was never simulated -- the exact "plausible table for a room
# nobody ran" failure this file exists to prevent. Every such path must raise instead.


@pytest.mark.parametrize("name", ["realistic", "overbidder_unpaced", "balanced_field"])
def test_build_field_rejects_n_patient_for_fields_that_cannot_honor_it(name: str) -> None:
    with pytest.raises(ValueError, match="n_patient"):
        build_field(name, 0.2, n_bots=11, n_patient=5)


def test_build_field_rejects_n_patient_when_the_uniform_cap_path_would_discard_it() -> None:
    """`pace_jitter <= 0` (and `n_bots is None`) short-circuit to the fixed 5-entry cycle."""
    with pytest.raises(ValueError, match="n_patient"):
        build_field("overbidder", 0.2, n_bots=11, pace_jitter=0.0, n_patient=8)
    with pytest.raises(ValueError, match="n_patient"):
        build_field("overbidder", 0.2, n_bots=None, n_patient=8)


def test_build_field_rejects_a_nonzero_n_patient_for_overbidder_only() -> None:
    """`overbidder_only` has no conservative seats by definition, so asking for some is an error.

    Replaces a test that asserted the request was silently ignored: that pinned the mislabel rather
    than the intent. Zero is still accepted, since it agrees with what the field does.
    """
    with pytest.raises(ValueError, match="n_patient"):
        build_field("overbidder_only", 0.2, n_bots=11, n_patient=5)
    field = build_field("overbidder_only", 0.2, n_bots=11, n_patient=0)
    assert not any(isinstance(b, PatientValueBot) for b in field)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # `realistic` pins the concrete archetypes rather than `== list(_REALISTIC_FIELD)`, which
        # would hold for any contents of that constant and so would follow it silently if the
        # shared baseline itself drifted.
        ("realistic", [AggressiveBot, PatientValueBot, BalancedBot]),
        ("overbidder_unpaced", [AggressiveBot] * 4 + [PatientValueBot]),
        ("balanced_field", [BalancedBot]),
    ],
)
def test_build_field_fixed_mix_composition(name: str, expected: list[type]) -> None:
    """The refusal must not disturb the fixed-mix paths -- only the mislabelling request."""
    assert [type(b) for b in build_field(name, 0.2, n_bots=11)] == expected


def test_build_field_overbidder_only_has_no_hoarders() -> None:
    field = build_field("overbidder_only", 0.2, n_bots=11)
    assert len(field) == 11
    assert not any(isinstance(b, PatientValueBot) for b in field)


def test_build_field_unknown_name_says_so_even_with_n_patient() -> None:
    """The honorability guard must not shadow the unknown-field diagnosis.

    `choices=FIELDS` shields every CLI, so this is reachable only from a REPL or a test -- but a
    typo there was being reported as "has a fixed archetype mix", pointing at the wrong knob.
    """
    with pytest.raises(ValueError, match="unknown field"):
        build_field("overbiddr", 0.2, n_bots=11, n_patient=3)
    with pytest.raises(ValueError, match="unknown field"):
        build_field("overbiddr", 0.2, n_bots=11)
