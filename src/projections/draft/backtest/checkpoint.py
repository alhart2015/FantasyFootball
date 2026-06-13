"""Pure helpers for the resumable, checkpointed H2H backtest runner.

The runner splits the seed range into chunks, runs each as an isolated subprocess, and
serializes its raw per-seed results to a JSON checkpoint. These helpers are the testable
core (chunk planning + result (de)serialization); the subprocess/retry orchestration lives
in scripts/h2h_backtest_chunked.py.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from projections.draft.backtest.league import LeagueResult


def plan_chunks(*, n_seeds: int, chunk_size: int) -> list[tuple[int, int]]:
    """Split ``[0, n_seeds)`` into contiguous ``(lo, hi)`` chunks of at most ``chunk_size``.

    Chunk boundaries need not respect the mirrored-seed pairing: every chunk is pooled into
    one final aggregation, so the full range — and thus every (base, mirror) pair — is always
    present regardless of where the cuts fall.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    return [(lo, min(lo + chunk_size, n_seeds)) for lo in range(0, n_seeds, chunk_size)]


def dump_results(
    actual: list[LeagueResult], projected: list[LeagueResult]
) -> dict[str, list[dict[str, Any]]]:
    """Serialize a chunk's raw (actual, projected) results to a JSON-friendly dict."""
    return {
        "actual": [dataclasses.asdict(r) for r in actual],
        "projected": [dataclasses.asdict(r) for r in projected],
    }


def load_results(
    payload: dict[str, list[dict[str, Any]]],
) -> tuple[list[LeagueResult], list[LeagueResult]]:
    """Reconstruct (actual, projected) LeagueResult lists from a dump_results payload."""
    actual = [LeagueResult(**r) for r in payload["actual"]]
    projected = [LeagueResult(**r) for r in payload["projected"]]
    return actual, projected


def verify_or_write_manifest(checkpoint_dir: Path, run_key: dict[str, object]) -> None:
    """Pin a checkpoint dir to the run identity that produced its chunks.

    Writes a manifest.json on first use; on resume, raises if `run_key` differs, so a
    reused dir built with a different strategy pair / season / MC params can't silently
    pool mismatched chunks (the per-chunk row-count check alone can't catch that).
    """
    manifest = checkpoint_dir / "manifest.json"
    if manifest.exists():
        prior = json.loads(manifest.read_text())
        if prior != run_key:
            raise ValueError(
                f"checkpoint dir {checkpoint_dir} was built with params {prior}, but this "
                f"run is {run_key}. Use a fresh --checkpoint-dir."
            )
    else:
        manifest.write_text(json.dumps(run_key, sort_keys=True))
