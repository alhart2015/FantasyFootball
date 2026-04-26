"""Codec tests — pack_per_stat_params / unpack_per_stat_params round-trip."""

from __future__ import annotations

import pytest

from projections.distributions import (
    Distribution,
    ParametricGamma,
    ParametricNormal,
    pack_per_stat_params,
)
from projections.schemas import Stat


def test_pack_normal_dist_returns_bytes() -> None:
    dists = {Stat.RECEIVING_YARDS: ParametricNormal(mean=36.3, std=18.1)}
    blob = pack_per_stat_params(dists)
    assert isinstance(blob, bytes)
    assert len(blob) > 0


def test_pack_gamma_dist_returns_bytes() -> None:
    dists = {Stat.RECEPTIONS: ParametricGamma(shape=4.2, scale=0.7)}
    blob = pack_per_stat_params(dists)
    assert isinstance(blob, bytes)
    assert len(blob) > 0


def test_pack_mixed_families_returns_bytes() -> None:
    dists: dict[Stat, Distribution] = {
        Stat.RECEIVING_YARDS: ParametricNormal(mean=36.3, std=18.1),
        Stat.RECEPTIONS: ParametricGamma(shape=4.2, scale=0.7),
    }
    blob = pack_per_stat_params(dists)
    assert isinstance(blob, bytes)
    assert len(blob) > 0


def test_pack_unknown_distribution_type_raises() -> None:
    class _NotADistribution:
        def mean(self) -> float:
            return 0.0

        def std(self) -> float:
            return 1.0

    with pytest.raises(ValueError, match="codec"):
        pack_per_stat_params({Stat.RECEIVING_YARDS: _NotADistribution()})  # type: ignore[dict-item]
