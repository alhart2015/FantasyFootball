"""Orchestrator + CLI: read one external_projections snapshot, blend, validate, write the
derived consensus_projections snapshot.

Usage:
    python -m projections.consensus.refresh --season 2026 [--asof YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from projections.consensus.blend import build_consensus
from projections.ingest.manifest import record as record_manifest
from projections.schemas import ConsensusProjectionSchema, Ruleset
from projections.store import read_latest_partition, read_partition, write_partition

_log = logging.getLogger(__name__)


class ConsensusError(RuntimeError):
    """Raised when the raw external_projections snapshot needed to build a consensus is missing.
    The CLI converts it to SystemExit; programmatic callers can catch it."""


def refresh_consensus(data_root: Path, *, season: int, asof: date | None = None) -> Path:
    """Build and write one consensus_projections snapshot from an external_projections snapshot.

    `asof=None` uses the latest raw snapshot for the season; otherwise the named one. The written
    consensus snapshot's asof MIRRORS the raw snapshot it was derived from (reproducible from
    input).
    """
    raw_root = data_root / "raw"
    try:
        if asof is not None:
            external = read_partition(raw_root, "external_projections", season=season, asof=asof)
        else:
            external = read_latest_partition(raw_root, "external_projections", season=season)
    except FileNotFoundError as exc:
        raise ConsensusError(
            f"No external_projections snapshot for season={season}"
            f"{f' asof={asof.isoformat()}' if asof else ''}; run the ingest first."
        ) from exc

    # Guard emptiness before reading asof off the frame — an empty latest snapshot would
    # otherwise IndexError on .iloc[0] and bypass this curated error.
    if external.empty:
        which = f"asof={asof.isoformat()}" if asof is not None else "the latest snapshot"
        raise ConsensusError(
            f"external_projections snapshot for season={season} ({which}) is empty; "
            f"refusing to write an empty consensus snapshot."
        )

    # The consensus snapshot's asof mirrors the raw snapshot it was derived from.
    snapshot_asof = asof if asof is not None else date.fromisoformat(str(external["asof"].iloc[0]))

    frame = build_consensus(external, ruleset=Ruleset())
    frame = ConsensusProjectionSchema.validate(frame)
    _log.info(
        "consensus_projections season=%s asof=%s: wrote %d players "
        "(with_points=%d, placeholders=%d).",
        season,
        snapshot_asof.isoformat(),
        len(frame),
        int(frame["has_points"].sum()),
        int(frame["is_placeholder_gsis"].sum()),
    )
    out = write_partition(
        data_root / "processed", "consensus_projections", frame, season=season, asof=snapshot_asof
    )
    record_manifest(data_root, table="consensus_projections", season=season, df=frame)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build the consensus preseason projection from an external_projections snapshot."
        )
    )
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument(
        "--asof",
        type=date.fromisoformat,
        default=None,
        help="Raw snapshot date YYYY-MM-DD; defaults to the latest snapshot for the season.",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        out = refresh_consensus(args.data_root, season=args.season, asof=args.asof)
    except ConsensusError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Wrote consensus snapshot: {out}", flush=True)


if __name__ == "__main__":
    main()
