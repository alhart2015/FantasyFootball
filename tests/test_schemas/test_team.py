"""Team enum tests — including alias normalization."""

from __future__ import annotations

import pytest

from projections.schemas import Team, normalize_team_code


def test_thirty_two_teams() -> None:
    assert len(list(Team)) == 32


def test_canonical_codes_are_uppercase_short() -> None:
    for t in Team:
        assert t.value.isupper()
        assert 2 <= len(t.value) <= 3


@pytest.mark.parametrize(
    "alias, expected",
    [
        ("JAX", Team.JAC),
        ("JAC", Team.JAC),
        ("LA", Team.LAR),
        ("LAR", Team.LAR),
        ("STL", Team.LAR),  # Rams pre-2016
        ("SD", Team.LAC),   # Chargers pre-2017
        ("OAK", Team.LV),   # Raiders pre-2020
        ("WAS", Team.WAS),
        ("WSH", Team.WAS),
    ],
)
def test_normalize_known_aliases(alias: str, expected: Team) -> None:
    assert normalize_team_code(alias) is expected


def test_normalize_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown team code"):
        normalize_team_code("XXX")
