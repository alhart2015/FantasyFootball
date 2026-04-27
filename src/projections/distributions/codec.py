"""Symmetric codec for per-stat distribution params persisted in
ProjectionWeeklySchema.params.

The encoded blob is msgpack-packed with shape:

    {
        "schema_version": 1,
        "stats": {
            "<stat_value>": {
                "family": "NORMAL"|"GAMMA"|"NEGATIVE_BINOMIAL"|"STUDENT_T"|"QUANTILE",
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

Adding a new family means adding one branch each to pack_per_stat_params and
unpack_per_stat_params. schema_version=1 is the only supported version today.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import msgpack
import numpy as np

from projections.distributions.base import Distribution
from projections.distributions.parametric import (
    ParametricGamma,
    ParametricNegativeBinomial,
    ParametricNormal,
    ParametricStudentT,
)
from projections.distributions.quantile import QuantileDistribution
from projections.schemas import DistributionFamily, Stat

_SCHEMA_VERSION: Final[int] = 1


def pack_per_stat_params(per_stat_dists: Mapping[Stat, Distribution]) -> bytes:
    """Encode a per-row per-stat distribution dict for ProjectionWeeklySchema.params.

    Raises:
        ValueError: a Distribution type without a registered codec entry.
    """
    stats_blob: dict[str, dict[str, object]] = {}
    for stat, dist in per_stat_dists.items():
        if isinstance(dist, ParametricNormal):
            stats_blob[stat.value] = {
                "family": DistributionFamily.NORMAL.value,
                "mean": dist.mean(),
                "std": dist.std(),
            }
        elif isinstance(dist, ParametricGamma):
            stats_blob[stat.value] = {
                "family": DistributionFamily.GAMMA.value,
                "shape": dist.shape,
                "scale": dist.scale,
            }
        elif isinstance(dist, ParametricNegativeBinomial):
            stats_blob[stat.value] = {
                "family": DistributionFamily.NEGATIVE_BINOMIAL.value,
                "mean": dist.mean(),
                "dispersion": dist.dispersion_,
            }
        elif isinstance(dist, ParametricStudentT):
            stats_blob[stat.value] = {
                "family": DistributionFamily.STUDENT_T.value,
                "loc": dist.mean(),
                "scale": dist.scale_,
                "df": dist.df_,
            }
        elif isinstance(dist, QuantileDistribution):
            stats_blob[stat.value] = {
                "family": DistributionFamily.QUANTILE.value,
                "quantiles": dist.quantiles_.tolist(),
                "values": dist.values_.tolist(),
            }
        else:
            raise ValueError(
                f"No codec entry for Distribution type {type(dist).__name__}; "
                f"add a branch to pack_per_stat_params in distributions/codec.py."
            )
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
        family_value = entry["family"]
        if family_value == DistributionFamily.NORMAL.value:
            out[stat] = ParametricNormal(mean=float(entry["mean"]), std=float(entry["std"]))
        elif family_value == DistributionFamily.GAMMA.value:
            out[stat] = ParametricGamma(shape=float(entry["shape"]), scale=float(entry["scale"]))
        elif family_value == DistributionFamily.NEGATIVE_BINOMIAL.value:
            out[stat] = ParametricNegativeBinomial(
                mean=float(entry["mean"]),
                dispersion=float(entry["dispersion"]),
            )
        elif family_value == DistributionFamily.STUDENT_T.value:
            out[stat] = ParametricStudentT(
                loc=float(entry["loc"]),
                scale=float(entry["scale"]),
                df=float(entry["df"]),
            )
        elif family_value == DistributionFamily.QUANTILE.value:
            out[stat] = QuantileDistribution(
                quantiles=np.asarray(entry["quantiles"], dtype=np.float64),
                values=np.asarray(entry["values"], dtype=np.float64),
            )
        else:
            raise ValueError(
                f"Unknown family in params blob: {family_value!r}; "
                f"add a branch to unpack_per_stat_params in distributions/codec.py."
            )
    return out
