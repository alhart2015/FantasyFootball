"""Reconcile placeholder gsis_ids in a draft pool to real ones via id_map.

The per-season preset VORP pools were generated from raw external-projection
snapshots that assigned deterministic ``99-`` placeholder gsis_ids to (nearly)
every player. Such a pool joins to neither ``weekly_stats`` (the injury history
behind ``availability.p``) nor ``id_map`` (team -> bye week), so the availability
model silently degrades to a position-average ``p`` with no byes -- and any eval
that scores rosters under availability becomes blind to player-specific injury
and bye risk.

``reconcile_pool_gsis`` relabels each placeholder row to its real gsis using the
canonical name+position key (``ingest.identity.placeholder_name_key``), so the
match rule agrees with ingest by construction. It is a pure relabel: same rows,
same per-row values, only the ``gsis_id`` label changes -- so every downstream
join (availability, rosters, scoring) is consistent afterward.
"""

from __future__ import annotations

import pandas as pd

from projections.ingest.identity import placeholder_name_key
from projections.schemas import _PYARROW_STR

_PLACEHOLDER_PREFIX = "99-"
_REAL_PREFIX = "00-"


def real_gsis_by_key(id_map: pd.DataFrame) -> dict[str, str]:
    """name+position key -> real gsis, dropping keys that map to >1 distinct gsis.

    Exposed so a caller reconciling many pools against one id_map can build this once and
    pass it as ``reconcile_pool_gsis(..., key_map=...)`` rather than rebuilding it per pool.
    """
    by_key: dict[str, set[str]] = {}
    real = id_map[id_map["gsis_id"].astype(str).str.startswith(_REAL_PREFIX)]
    for name, pos, gid in zip(
        real["full_name"], real["position"], real["gsis_id"].astype(str), strict=True
    ):
        if pd.isna(name) or pd.isna(pos):
            continue
        by_key.setdefault(placeholder_name_key(str(name), str(pos)), set()).add(gid)
    return {k: next(iter(v)) for k, v in by_key.items() if len(v) == 1}


def reconcile_pool_gsis(
    pool: pd.DataFrame,
    id_map: pd.DataFrame,
    *,
    key_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return ``pool`` with placeholder gsis_ids relabeled to real ones where possible.

    A placeholder row is relabeled only when its (full_name, position) key maps to a
    single real gsis in ``id_map`` AND that real gsis does not collide with another
    kept pool id (gsis uniqueness is a VorpTableSchema invariant and ``project_draft``
    indexes the pool by gsis). Rows already carrying a real gsis, or with no/ambiguous/
    colliding match, pass through unchanged. ``pool`` is not mutated.

    ``key_map`` is an optional prebuilt ``real_gsis_by_key(id_map)``; pass it when reconciling
    many pools against the same id_map to skip the per-call rebuild (``id_map`` is then unused).
    Requires ``pool`` to carry ``gsis_id``, ``full_name`` and ``position`` columns.
    """
    if "full_name" not in pool.columns:
        raise ValueError("reconcile_pool_gsis requires a full_name column on the pool")
    uniq = real_gsis_by_key(id_map) if key_map is None else key_map

    out = pool.copy()
    ids = out["gsis_id"].astype(str).tolist()
    names = out["full_name"].tolist()
    positions = out["position"].astype(str).tolist()

    # `seen` = every gsis already committed to the output, so a reconciled id never collides
    # with a real id the pool already had nor with an earlier reconcile. Placeholders are never
    # reconcile targets (uniq's values are all real), so they need not be tracked.
    seen: set[str] = {g for g in ids if not g.startswith(_PLACEHOLDER_PREFIX)}
    for i, gid in enumerate(ids):
        if not gid.startswith(_PLACEHOLDER_PREFIX) or pd.isna(names[i]):
            continue
        real = uniq.get(placeholder_name_key(str(names[i]), positions[i]))
        if real is not None and real not in seen:
            ids[i] = real
            seen.add(real)

    out["gsis_id"] = pd.array(ids, dtype=_PYARROW_STR)
    return out
