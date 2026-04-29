"""Symmetric codec for per-stat distribution params persisted in
ProjectionWeeklySchema.params.

The encoded blob is msgpack-packed with shape:

    {
        "schema_version": 2,
        "stats": {
            "<stat_value>": {
                "family": "NORMAL"|"GAMMA"|"NEGATIVE_BINOMIAL"|"STUDENT_T"|
                          "QUANTILE"|"MIXTURE",
                ... family-specific params ...
            },
            ...
        }
    }

Currently registered families:
    NORMAL:            {"family": "NORMAL",            "mean": float, "std": float}
    GAMMA:             {"family": "GAMMA",             "shape": float, "scale": float}
    NEGATIVE_BINOMIAL: {"family": "NEGATIVE_BINOMIAL", "mean": float, "dispersion": float}
    STUDENT_T:         {"family": "STUDENT_T",         "loc": float, "scale": float, "df": float}
    QUANTILE:          {"family": "QUANTILE",          "quantiles": list[float],
                                                       "values":    list[float]}
    MIXTURE:           {"family": "MIXTURE",           "weight": float,
                                                       "component_a": {<single>},
                                                       "component_b": {<single>}}

Schema version 2 (Plan 6): MIXTURE branch added; v1 blobs are no longer
forward-compatible (no v1 readers in the codebase). Adding a new family means
adding one branch each to _pack_single and _unpack_single.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import msgpack
import numpy as np

from projections.distributions.base import Distribution
from projections.distributions.mixture import MixtureDistribution
from projections.distributions.parametric import (
    ParametricGamma,
    ParametricNegativeBinomial,
    ParametricNormal,
    ParametricStudentT,
)
from projections.distributions.quantile import QuantileDistribution
from projections.schemas import DistributionFamily, Stat

_SCHEMA_VERSION: Final[int] = 2


def _pack_single(dist: Distribution) -> dict[str, Any]:
    """Encode a single Distribution as a family-tagged dict.

    Used by both top-level pack_per_stat_params (one dict per stat) and the
    MIXTURE recursion (one dict per child component).

    Raises ValueError on Distribution types without a registered codec entry.
    """
    if isinstance(dist, MixtureDistribution):
        return {
            "family": DistributionFamily.MIXTURE.value,
            "weight": float(dist.weight),
            "component_a": _pack_single(dist.component_a),
            "component_b": _pack_single(dist.component_b),
        }
    if isinstance(dist, ParametricNormal):
        return {
            "family": DistributionFamily.NORMAL.value,
            "mean": dist.mean(),
            "std": dist.std(),
        }
    if isinstance(dist, ParametricGamma):
        return {
            "family": DistributionFamily.GAMMA.value,
            "shape": dist.shape,
            "scale": dist.scale,
        }
    if isinstance(dist, ParametricNegativeBinomial):
        return {
            "family": DistributionFamily.NEGATIVE_BINOMIAL.value,
            "mean": dist.mean(),
            "dispersion": dist.dispersion_,
        }
    if isinstance(dist, ParametricStudentT):
        return {
            "family": DistributionFamily.STUDENT_T.value,
            "loc": dist.mean(),
            "scale": dist.scale_,
            "df": dist.df_,
        }
    if isinstance(dist, QuantileDistribution):
        return {
            "family": DistributionFamily.QUANTILE.value,
            "quantiles": dist.quantiles_.tolist(),
            "values": dist.values_.tolist(),
        }
    raise ValueError(
        f"No codec entry for Distribution type {type(dist).__name__}; "
        f"add a branch to _pack_single in distributions/codec.py."
    )


def _unpack_single(entry: Mapping[str, Any]) -> Distribution:
    """Decode a single family-tagged dict back into a Distribution.

    Raises ValueError on unknown family.
    """
    family_value = entry["family"]
    if family_value == DistributionFamily.MIXTURE.value:
        return MixtureDistribution(
            component_a=_unpack_single(entry["component_a"]),
            component_b=_unpack_single(entry["component_b"]),
            weight=float(entry["weight"]),
        )
    if family_value == DistributionFamily.NORMAL.value:
        return ParametricNormal(mean=float(entry["mean"]), std=float(entry["std"]))
    if family_value == DistributionFamily.GAMMA.value:
        return ParametricGamma(shape=float(entry["shape"]), scale=float(entry["scale"]))
    if family_value == DistributionFamily.NEGATIVE_BINOMIAL.value:
        return ParametricNegativeBinomial(
            mean=float(entry["mean"]),
            dispersion=float(entry["dispersion"]),
        )
    if family_value == DistributionFamily.STUDENT_T.value:
        return ParametricStudentT(
            loc=float(entry["loc"]),
            scale=float(entry["scale"]),
            df=float(entry["df"]),
        )
    if family_value == DistributionFamily.QUANTILE.value:
        return QuantileDistribution(
            quantiles=np.asarray(entry["quantiles"], dtype=np.float64),
            values=np.asarray(entry["values"], dtype=np.float64),
        )
    raise ValueError(
        f"Unknown family in params blob: {family_value!r}; "
        f"add a branch to _unpack_single in distributions/codec.py."
    )


def pack_per_stat_params(per_stat_dists: Mapping[Stat, Distribution]) -> bytes:
    """Encode a per-row per-stat distribution dict for ProjectionWeeklySchema.params.

    Raises:
        ValueError: a Distribution type without a registered codec entry.
    """
    stats_blob: dict[str, dict[str, Any]] = {
        stat.value: _pack_single(dist) for stat, dist in per_stat_dists.items()
    }
    payload = {"schema_version": _SCHEMA_VERSION, "stats": stats_blob}
    return bytes(msgpack.packb(payload, use_bin_type=True))


def unpack_per_stat_params(blob: bytes) -> dict[Stat, Distribution]:
    """Decode the params blob into a {Stat -> Distribution} dict.

    Raises:
        ValueError: unknown schema_version, unknown family, or unknown stat name.
    """
    payload = msgpack.unpackb(blob, raw=False)
    version = payload.get("schema_version")
    if version != _SCHEMA_VERSION:
        raise ValueError(
            f"Unknown per-stat params schema_version: {version!r} (supported: {_SCHEMA_VERSION})"
        )
    stats_blob = payload["stats"]
    out: dict[Stat, Distribution] = {}
    for stat_name, entry in stats_blob.items():
        try:
            stat = Stat(stat_name)
        except ValueError as exc:
            raise ValueError(f"Unknown stat name in params blob: {stat_name!r}") from exc
        out[stat] = _unpack_single(entry)
    return out
