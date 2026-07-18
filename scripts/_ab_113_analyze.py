"""One-off analysis for issue #113: now_or_never_floored vs season_value H2H A/B.

Reads the chunked-runner checkpoints for a season, prints the harness marginal-CI
table (matching Test 10's convention), and additionally computes the CRN-paired
per-seed difference (mean of the 4 floored seats - mean of the 4 season_value
seats, per seed) with a percentile bootstrap CI. The paired diff is the tighter,
direct answer to "does the floor close now_or_never's gap to season_value?" because
both strategies share the same board/bot field within each seed (mirrored seats).

Not a shipped tool -- a throwaway analysis script kept alongside the run for
reproducibility. Season + strategy pair are read from the checkpoint dir's
manifest.json. Usage:
    python scripts/_ab_113_analyze.py --checkpoint-dir _h2h_ckpt_113_2025
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from projections.draft.assistant._compare import Interval, bootstrap_mean
from projections.draft.backtest.checkpoint import load_results
from projections.draft.backtest.cli import format_result
from projections.draft.backtest.harness import aggregate
from projections.draft.backtest.league import LeagueResult

FLOORED = "now_or_never_floored"
SV = "season_value"
N_TEAMS = 16

# Single source of truth for the four metrics: key -> (per-row extractor, display
# label, is-percentage). Shared by the paired-diff computation and its printer, so a
# metric is added/removed in exactly one place.
_METRICS: dict[str, tuple[Callable[[LeagueResult], float], str, bool]] = {
    "win_pct": (lambda r: r.wins / (r.wins + r.losses), "win%   ", True),
    "playoff": (lambda r: float(r.made_playoffs), "playoff", True),
    "champ": (lambda r: float(r.is_champion), "champ% ", True),
    "points_for": (lambda r: r.points_for, "pts_for", False),
}


def _chunk_files(checkpoint_dir: Path) -> list[tuple[int, int, Path]]:
    """(lo, hi, path) per chunk_LO_HI.json, sorted by lo."""
    out: list[tuple[int, int, Path]] = []
    for f in checkpoint_dir.glob("chunk_*.json"):
        lo, hi = (int(x) for x in f.stem.split("_")[1:3])
        out.append((lo, hi, f))
    return sorted(out)


def _pool(checkpoint_dir: Path) -> tuple[list[LeagueResult], list[LeagueResult]]:
    """Pool all chunks in seed order, asserting the pool is complete and untruncated.

    Both guards are load-bearing. The chunks must tile [0, N) contiguously (a gapped or
    short pool would otherwise be analyzed silently -- N*16 is still a multiple of
    N_TEAMS), and each chunk must hold its full (hi-lo)*N_TEAMS rows: a published
    checkpoint can still be truncated or overwritten after the fact (concurrent run, AV,
    disk hiccup), which the runner itself guards with the same pool-time row-count
    re-check. (The even-N parity precondition is enforced in main, since it gates both
    the marginal table and the paired diff -- see the comment there.)
    """
    chunks = _chunk_files(checkpoint_dir)
    if not chunks:
        raise FileNotFoundError(f"no chunk_*.json files in {checkpoint_dir}")
    expected_lo = 0
    for lo, hi, _ in chunks:
        if lo != expected_lo:
            raise ValueError(f"chunks not contiguous from 0: expected seed {expected_lo}, got {lo}")
        expected_lo = hi

    actual: list[LeagueResult] = []
    projected: list[LeagueResult] = []
    for lo, hi, f in chunks:
        a, p = load_results(json.loads(f.read_text()))
        want = (hi - lo) * N_TEAMS
        if len(a) != want or len(p) != want:
            raise ValueError(
                f"{f.name}: want {want} rows, got actual={len(a)} projected={len(p)} "
                f"(truncated/corrupt checkpoint)"
            )
        actual += a
        projected += p
    return actual, projected


def _seat_mean(rs: list[LeagueResult], metric: str) -> float:
    """Mean of `metric` over a set of same-strategy seat rows within one seed."""
    extract = _METRICS[metric][0]
    return float(np.mean([extract(r) for r in rs]))


def _paired_diffs(results: list[LeagueResult]) -> dict[str, np.ndarray]:
    """Per-seed paired diff arrays: floored - season_value, one value per seed.

    Consecutive N_TEAMS-row blocks are one seed (collect_results appends exactly
    n_teams rows per seed, chunks pool in seed order). Within a block, filter by
    strategy label -> 4 floored + 4 season_value + 8 bot. Assumes an even seed count
    (enforced by main, which gates both outputs on parity) so the seat-position term
    cancels; see the parity comment in main.
    """
    if len(results) % N_TEAMS != 0:
        raise ValueError(f"pooled rows {len(results)} not a multiple of {N_TEAMS}")
    n_seeds = len(results) // N_TEAMS
    out: dict[str, list[float]] = {m: [] for m in _METRICS}
    for s in range(n_seeds):
        block = results[s * N_TEAMS : (s + 1) * N_TEAMS]
        floored = [r for r in block if r.strategy == FLOORED]
        sv = [r for r in block if r.strategy == SV]
        if len(floored) != 4 or len(sv) != 4:
            raise ValueError(f"seed {s}: expected 4 floored + 4 sv, got {len(floored)} + {len(sv)}")
        for m in _METRICS:
            out[m].append(_seat_mean(floored, m) - _seat_mean(sv, m))
    return {m: np.array(v) for m, v in out.items()}


def _fmt(iv: Interval, *, pct: bool) -> str:
    if pct:
        return f"{iv.point * 100:+6.2f}pp  [{iv.lo_95 * 100:+.2f}, {iv.hi_95 * 100:+.2f}]"
    return f"{iv.point:+7.2f}    [{iv.lo_95:+.1f}, {iv.hi_95:+.1f}]"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    args = p.parse_args()

    # Season, strategy pair, and floor come from the dir's manifest (written by the
    # runner), not flags -- so the header can't be mislabeled, a wrong-pair dir fails loud
    # here, and a non-default floor shows in the header instead of passing unnoticed.
    manifest = json.loads((args.checkpoint_dir / "manifest.json").read_text())
    pair = {manifest["strategy_a"], manifest["strategy_b"]}
    if pair != {FLOORED, SV}:
        raise ValueError(f"{args.checkpoint_dir} is a {sorted(pair)} run, not {FLOORED}-vs-{SV}")
    season, floor, floor_weight = manifest["season"], manifest["floor"], manifest["floor_weight"]

    actual, projected = _pool(args.checkpoint_dir)
    n_seeds = len(actual) // N_TEAMS
    # Parity gate for BOTH outputs: seat_layout swaps floored/sv by seed parity, so the
    # floored-vs-sv comparison -- in the marginal table AND the paired diff -- only sheds
    # the seat-position term when even/odd seeds balance, i.e. N is even. Odd N biases both.
    if n_seeds % 2 != 0:
        raise ValueError(f"seed count {n_seeds} is odd -> parity-imbalanced (biases floored-vs-sv)")
    result = aggregate(actual, projected, n_seeds=n_seeds, base_seed=0)

    print(
        f"\n########## SEASON {season}  (F={floor}, lambda={floor_weight})  "
        f"({args.checkpoint_dir}) ##########\n"
    )
    print(format_result(result))

    for axis_name, res in (("ACTUAL", actual), ("PROJECTED", projected)):
        diffs = _paired_diffs(res)
        print(f"\n[PAIRED per-seed diff  (floored - season_value)  -- {axis_name}]")
        print(f"  (n_seeds={n_seeds}; >0 favors floored, CI excludes 0 = significant)")
        for m, (_extract, label, pct) in _METRICS.items():
            iv = bootstrap_mean(diffs[m], seed=0)
            print(f"    {label}: {_fmt(iv, pct=pct)}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
