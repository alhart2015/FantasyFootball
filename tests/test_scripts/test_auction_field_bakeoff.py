"""Tests for `scripts/auction_field_bakeoff.build_field`.

The script had no tests; these cover the opponent-mix knob added for the field-composition sweep,
because a silently-wrong mix would produce a plausible-looking table for a room nobody ran.
"""

from __future__ import annotations

from itertools import pairwise

import pytest
from scripts.auction_field_bakeoff import build_field

from projections.draft.assistant.auction.market import PatientValueBot


def test_build_field_n_patient_none_reproduces_the_historical_mix() -> None:
    """The default path must stay byte-identical: every prior run used the every-5th rule."""
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


def test_build_field_n_patient_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="n_patient"):
        build_field("overbidder", 0.2, n_bots=11, n_patient=12)


def test_build_field_n_patient_ignored_for_fields_without_hoarders() -> None:
    """`overbidder_only` has no conservative seats by definition; the knob must not sneak any in."""
    field = build_field("overbidder_only", 0.2, n_bots=11, n_patient=5)
    assert not any(isinstance(b, PatientValueBot) for b in field)
