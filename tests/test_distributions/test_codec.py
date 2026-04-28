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


def test_codec_round_trip_neg_binomial() -> None:
    """NB packed via pack_per_stat_params and round-tripped."""
    from projections.distributions import (
        ParametricNegativeBinomial,
        pack_per_stat_params,
        unpack_per_stat_params,
    )
    from projections.schemas import Stat

    dist = ParametricNegativeBinomial(mean=0.3, dispersion=2.0)
    blob = pack_per_stat_params({Stat.RECEIVING_TDS: dist})
    decoded = unpack_per_stat_params(blob)
    assert Stat.RECEIVING_TDS in decoded
    decoded_dist = decoded[Stat.RECEIVING_TDS]
    assert isinstance(decoded_dist, ParametricNegativeBinomial)
    assert decoded_dist.mean() == pytest.approx(0.3)
    # Round-trip preserves (mean, dispersion) directly via persisted entries.


def test_codec_round_trip_student_t() -> None:
    from projections.distributions import (
        ParametricStudentT,
        pack_per_stat_params,
        unpack_per_stat_params,
    )
    from projections.schemas import Stat

    dist = ParametricStudentT(loc=250.0, scale=70.0, df=8.0)
    blob = pack_per_stat_params({Stat.PASSING_YARDS: dist})
    decoded = unpack_per_stat_params(blob)
    decoded_dist = decoded[Stat.PASSING_YARDS]
    assert isinstance(decoded_dist, ParametricStudentT)
    assert decoded_dist.mean() == pytest.approx(250.0)
    assert decoded_dist.std() == pytest.approx(dist.std())


def test_codec_round_trip_after_bucketed_fit() -> None:
    """End-to-end: emit per-row distributions for the 4 family types currently
    in use after Plan 3e (NORMAL, GAMMA, NB; STUDENT_T preserved as
    infrastructure). Confirm pack + unpack round-trips correctly for each."""
    from projections.distributions import (
        ParametricGamma,
        ParametricNegativeBinomial,
        ParametricNormal,
        ParametricStudentT,
        pack_per_stat_params,
        unpack_per_stat_params,
    )
    from projections.schemas import Stat

    blob = pack_per_stat_params(
        {
            Stat.PASSING_YARDS: ParametricNormal(mean=250.0, std=70.0),
            Stat.RECEPTIONS: ParametricGamma(shape=2.5, scale=1.0),
            Stat.PASSING_TDS: ParametricNegativeBinomial(mean=1.5, dispersion=4.0),
            Stat.RUSHING_YARDS: ParametricStudentT(loc=20.0, scale=15.0, df=4.0),
        }
    )
    decoded = unpack_per_stat_params(blob)
    assert isinstance(decoded[Stat.PASSING_YARDS], ParametricNormal)
    assert isinstance(decoded[Stat.RECEPTIONS], ParametricGamma)
    assert isinstance(decoded[Stat.PASSING_TDS], ParametricNegativeBinomial)
    assert isinstance(decoded[Stat.RUSHING_YARDS], ParametricStudentT)


def test_codec_round_trip_quantile() -> None:
    """Plan 5 — QUANTILE family round-trips through msgpack codec."""
    import numpy as np

    from projections.distributions import (
        QuantileDistribution,
        pack_per_stat_params,
        unpack_per_stat_params,
    )
    from projections.schemas import Stat

    qs = np.array([0.05, 0.10, 0.50, 0.90, 0.95])
    vs = np.array([1.0, 2.5, 5.0, 8.5, 12.0])
    original: dict[Stat, Distribution] = {
        Stat.RECEIVING_YARDS: QuantileDistribution(quantiles=qs, values=vs)
    }

    blob = pack_per_stat_params(original)
    decoded = unpack_per_stat_params(blob)

    assert set(decoded.keys()) == {Stat.RECEIVING_YARDS}
    decoded_dist = decoded[Stat.RECEIVING_YARDS]
    assert isinstance(decoded_dist, QuantileDistribution)
    np.testing.assert_array_equal(decoded_dist.quantiles_, qs)
    np.testing.assert_array_equal(decoded_dist.values_, vs)


def test_codec_round_trip_mixed_with_quantile() -> None:
    """Plan 5 — QUANTILE coexists with NORMAL / GAMMA / NB in a single per-row blob."""
    import numpy as np

    from projections.distributions import (
        ParametricGamma,
        ParametricNegativeBinomial,
        ParametricNormal,
        QuantileDistribution,
        pack_per_stat_params,
        unpack_per_stat_params,
    )
    from projections.schemas import Stat

    original: dict[Stat, Distribution] = {
        Stat.RECEIVING_YARDS: QuantileDistribution(
            quantiles=np.array([0.05, 0.10, 0.50, 0.90, 0.95]),
            values=np.array([1.0, 2.5, 5.0, 8.5, 12.0]),
        ),
        Stat.RECEPTIONS: ParametricNormal(mean=3.0, std=1.5),
        Stat.RUSHING_YARDS: ParametricGamma(shape=2.0, scale=4.0),
        Stat.RECEIVING_TDS: ParametricNegativeBinomial(mean=0.3, dispersion=2.0),
    }

    blob = pack_per_stat_params(original)
    decoded = unpack_per_stat_params(blob)

    assert set(decoded.keys()) == set(original.keys())
    assert isinstance(decoded[Stat.RECEIVING_YARDS], QuantileDistribution)
    assert isinstance(decoded[Stat.RECEPTIONS], ParametricNormal)
    assert isinstance(decoded[Stat.RUSHING_YARDS], ParametricGamma)
    assert isinstance(decoded[Stat.RECEIVING_TDS], ParametricNegativeBinomial)


# Plan 6 Phase 1 — MIXTURE codec branch tests
def test_codec_mixture_round_trip_normal_normal() -> None:
    import numpy as np

    from projections.distributions import (
        MixtureDistribution,
        ParametricNormal,
        pack_per_stat_params,
        unpack_per_stat_params,
    )
    from projections.schemas import Stat

    _ = np  # keep numpy import grouped with the other mixture tests' imports
    a = ParametricNormal(mean=5.0, std=2.0)
    b = ParametricNormal(mean=15.0, std=3.0)
    mix = MixtureDistribution(component_a=a, component_b=b, weight=0.4)
    blob = pack_per_stat_params({Stat.PASSING_YARDS: mix})
    decoded = unpack_per_stat_params(blob)
    out = decoded[Stat.PASSING_YARDS]
    assert isinstance(out, MixtureDistribution)
    assert out.weight == pytest.approx(0.4, abs=1e-12)
    assert isinstance(out.component_a, ParametricNormal)
    assert out.component_a.mean() == pytest.approx(5.0)
    assert out.component_a.std() == pytest.approx(2.0)
    assert isinstance(out.component_b, ParametricNormal)
    assert out.component_b.mean() == pytest.approx(15.0)
    assert out.component_b.std() == pytest.approx(3.0)


def test_codec_mixture_round_trip_gamma_quantile() -> None:
    import numpy as np

    from projections.distributions import (
        MixtureDistribution,
        ParametricGamma,
        QuantileDistribution,
        pack_per_stat_params,
        unpack_per_stat_params,
    )
    from projections.schemas import Stat

    a = ParametricGamma(shape=4.0, scale=2.5)
    b = QuantileDistribution(
        quantiles=np.array([0.05, 0.25, 0.5, 0.75, 0.95], dtype=np.float64),
        values=np.array([1.0, 5.0, 10.0, 18.0, 30.0], dtype=np.float64),
    )
    mix = MixtureDistribution(component_a=a, component_b=b, weight=0.65)
    blob = pack_per_stat_params({Stat.RUSHING_YARDS: mix})
    decoded = unpack_per_stat_params(blob)
    out = decoded[Stat.RUSHING_YARDS]
    assert isinstance(out, MixtureDistribution)
    assert out.weight == pytest.approx(0.65, abs=1e-12)
    assert isinstance(out.component_a, ParametricGamma)
    assert isinstance(out.component_b, QuantileDistribution)
    np.testing.assert_array_almost_equal(
        out.component_b.quantiles_, np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    )
    np.testing.assert_array_almost_equal(
        out.component_b.values_, np.array([1.0, 5.0, 10.0, 18.0, 30.0])
    )


def test_codec_mixture_round_trip_nb_nb() -> None:
    from projections.distributions import (
        MixtureDistribution,
        ParametricNegativeBinomial,
        pack_per_stat_params,
        unpack_per_stat_params,
    )
    from projections.schemas import Stat

    a = ParametricNegativeBinomial(mean=1.5, dispersion=4.0)
    b = ParametricNegativeBinomial(mean=2.5, dispersion=8.0)
    mix = MixtureDistribution(component_a=a, component_b=b, weight=0.3)
    blob = pack_per_stat_params({Stat.RECEIVING_TDS: mix})
    decoded = unpack_per_stat_params(blob)
    out = decoded[Stat.RECEIVING_TDS]
    assert isinstance(out, MixtureDistribution)
    assert out.weight == pytest.approx(0.3, abs=1e-12)
    assert isinstance(out.component_a, ParametricNegativeBinomial)
    assert out.component_a.mean() == pytest.approx(1.5)
    assert out.component_a.dispersion_ == pytest.approx(4.0)
    assert isinstance(out.component_b, ParametricNegativeBinomial)
    assert out.component_b.mean() == pytest.approx(2.5)
    assert out.component_b.dispersion_ == pytest.approx(8.0)


def test_codec_mixture_family_name() -> None:
    """The encoded blob carries family='MIXTURE' for mixture entries."""
    import msgpack

    from projections.distributions import (
        MixtureDistribution,
        ParametricNormal,
        pack_per_stat_params,
    )
    from projections.schemas import DistributionFamily, Stat

    a = ParametricNormal(mean=5.0, std=2.0)
    b = ParametricNormal(mean=15.0, std=3.0)
    mix = MixtureDistribution(component_a=a, component_b=b, weight=0.5)
    blob = pack_per_stat_params({Stat.PASSING_YARDS: mix})
    payload = msgpack.unpackb(blob, raw=False)
    assert payload["schema_version"] == 2
    entry = payload["stats"]["passing_yards"]
    assert entry["family"] == DistributionFamily.MIXTURE.value
    assert entry["weight"] == pytest.approx(0.5)
    assert entry["component_a"]["family"] == DistributionFamily.NORMAL.value
    assert entry["component_b"]["family"] == DistributionFamily.NORMAL.value


def test_codec_v1_blobs_no_longer_decodable() -> None:
    """Plan 5/5b/5c blobs (v1) without MIXTURE entries are no longer
    forward-compat after the schema_version bump to 2.

    No v1 readers exist outside this codec, so we test by directly
    synthesizing a v1-shaped blob and asserting unpack rejects it.
    """
    import msgpack

    from projections.distributions import unpack_per_stat_params
    from projections.schemas import DistributionFamily

    payload = {
        "schema_version": 1,  # legacy
        "stats": {
            "passing_yards": {
                "family": DistributionFamily.NORMAL.value,
                "mean": 250.0,
                "std": 50.0,
            }
        },
    }
    blob = msgpack.packb(payload, use_bin_type=True)
    with pytest.raises(ValueError, match="schema_version"):
        unpack_per_stat_params(bytes(blob))
