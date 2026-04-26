"""Codec tests — pack_per_stat_params / unpack_per_stat_params round-trip."""

from __future__ import annotations

import msgpack
import pytest

from projections.distributions import (
    Distribution,
    ParametricGamma,
    ParametricNormal,
    pack_per_stat_params,
    unpack_per_stat_params,
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


def test_round_trip_normal_preserves_params() -> None:
    original: dict[Stat, Distribution] = {
        Stat.RECEIVING_YARDS: ParametricNormal(mean=36.3, std=18.1)
    }
    blob = pack_per_stat_params(original)
    decoded = unpack_per_stat_params(blob)
    assert set(decoded.keys()) == {Stat.RECEIVING_YARDS}
    d = decoded[Stat.RECEIVING_YARDS]
    assert isinstance(d, ParametricNormal)
    assert d.mean() == pytest.approx(36.3)
    assert d.std() == pytest.approx(18.1)


def test_round_trip_gamma_preserves_params() -> None:
    original: dict[Stat, Distribution] = {Stat.RECEPTIONS: ParametricGamma(shape=4.2, scale=0.7)}
    blob = pack_per_stat_params(original)
    decoded = unpack_per_stat_params(blob)
    d = decoded[Stat.RECEPTIONS]
    assert isinstance(d, ParametricGamma)
    assert d.shape == pytest.approx(4.2)
    assert d.scale == pytest.approx(0.7)


def test_round_trip_six_stats_mixed_families_preserves_all() -> None:
    original: dict[Stat, Distribution] = {
        Stat.PASSING_YARDS: ParametricNormal(mean=199.5, std=84.5),
        Stat.PASSING_TDS: ParametricGamma(shape=4.2, scale=0.29),
        Stat.INTERCEPTIONS: ParametricGamma(shape=1.6, scale=0.43),
        Stat.RUSHING_YARDS: ParametricNormal(mean=18.2, std=17.9),
        Stat.RUSHING_TDS: ParametricGamma(shape=0.8, scale=0.24),
        Stat.FUMBLES_LOST: ParametricGamma(shape=0.5, scale=0.41),
    }
    blob = pack_per_stat_params(original)
    decoded = unpack_per_stat_params(blob)
    assert set(decoded.keys()) == set(original.keys())
    for stat, original_dist in original.items():
        round_tripped = decoded[stat]
        assert type(round_tripped) is type(original_dist)
        assert round_tripped.mean() == pytest.approx(original_dist.mean())
        assert round_tripped.std() == pytest.approx(original_dist.std())


def test_unknown_schema_version_raises() -> None:
    bad = msgpack.packb({"schema_version": 999, "stats": {}}, use_bin_type=True)
    with pytest.raises(ValueError, match="schema_version"):
        unpack_per_stat_params(bytes(bad))


def test_unknown_family_raises() -> None:
    bad = msgpack.packb(
        {"schema_version": 1, "stats": {"receiving_yards": {"family": "WEIBULL", "k": 1.0}}},
        use_bin_type=True,
    )
    with pytest.raises(ValueError, match="WEIBULL"):
        unpack_per_stat_params(bytes(bad))


def test_unknown_stat_name_raises() -> None:
    bad = msgpack.packb(
        {
            "schema_version": 1,
            "stats": {"this_is_not_a_stat": {"family": "NORMAL", "mean": 0.0, "std": 1.0}},
        },
        use_bin_type=True,
    )
    with pytest.raises(ValueError, match="this_is_not_a_stat"):
        unpack_per_stat_params(bytes(bad))
