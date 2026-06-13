"""Checkpointed, resumable H2H backtest runner (survives process segfaults and full BSODs).

Long single-process runs of the season-value draft MC are unstable on this Windows box
(native BLAS/OpenMP access violations; the machine has also hard-crashed). This runner splits
the seed range into chunks, runs each as an isolated subprocess that writes its raw per-seed
results to a JSON checkpoint, and retries a crashed chunk. On a fresh invocation, completed
checkpoints are skipped — so after a reboot you just re-run the same command and it resumes.
Once every chunk is present, results are pooled in seed order and aggregated once, which is
byte-identical to a monolithic run_backtest (pinned by test_chunked_collection_matches_monolithic).

Driver (what you run):
    python scripts/h2h_backtest_chunked.py --season 2025 \
        --league-config configs/league_espn_half_16team.json \
        --n-seeds 200 --strategy-n-sims 200 --jitter 8 \
        --chunk-size 20 --checkpoint-dir _h2h_ckpt --data-root data

Worker (spawned by the driver per chunk; not run by hand):
    ... --worker --chunk-lo L --chunk-hi H --out <file>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from projections.draft.backtest.checkpoint import dump_results, load_results, plan_chunks
from projections.draft.backtest.cli import format_result
from projections.draft.backtest.harness import aggregate, collect_results
from projections.draft.backtest.inputs import load_inputs
from projections.draft.backtest.league import LeagueResult
from projections.draft.league_config import LeagueConfig

# Force single-threaded native libs in the worker — the multi-threaded BLAS/OpenMP path is
# what destabilizes long runs. KMP_DUPLICATE_LIB_OK works around the duplicate-OpenMP abort.
_WORKER_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resumable chunked H2H backtest.")
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--league-config", type=Path, required=True)
    p.add_argument("--n-seeds", type=int, default=200)
    p.add_argument("--strategy-n-sims", type=int, default=200)
    p.add_argument("--jitter", type=float, default=8.0)
    p.add_argument("--chunk-size", type=int, default=20)
    p.add_argument("--checkpoint-dir", type=Path, default=Path("_h2h_ckpt"))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--max-retries", type=int, default=5)
    # Worker-only flags (the driver sets these when spawning a chunk subprocess):
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--chunk-lo", type=int)
    p.add_argument("--chunk-hi", type=int)
    p.add_argument("--out", type=Path)
    return p.parse_args(argv)


def _chunk_file(checkpoint_dir: Path, lo: int, hi: int) -> Path:
    return checkpoint_dir / f"chunk_{lo:05d}_{hi:05d}.json"


def _valid_chunk_file(path: Path, expected_rows: int) -> bool:
    """A checkpoint counts as complete only if it parses and has the expected seat-row count."""
    if not path.exists():
        return False
    try:
        actual, projected = load_results(json.loads(path.read_text()))
    except (json.JSONDecodeError, KeyError, TypeError, OSError):
        return False
    return len(actual) == expected_rows and len(projected) == expected_rows


def _run_worker(args: argparse.Namespace) -> int:
    config = LeagueConfig.model_validate_json(args.league_config.read_text())
    inputs = load_inputs(season=args.season, config=config, data_root=args.data_root)
    actual, projected = collect_results(
        seed_lo=args.chunk_lo,
        seed_hi=args.chunk_hi,
        pool=inputs.pool,
        config=config,
        availability=inputs.availability,
        proj_lookup=inputs.proj_lookup,
        actual_lookup=inputs.actual_lookup,
        calendar=inputs.calendar,
        jitter=args.jitter,
        strategy_n_sims=args.strategy_n_sims,
        base_seed=0,
    )
    out: Path = args.out
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(dump_results(actual, projected)))
    tmp.replace(out)  # atomic publish — a half-written file is never seen as complete
    return 0


def _run_driver(args: argparse.Namespace) -> int:
    config = LeagueConfig.model_validate_json(args.league_config.read_text())
    n_teams = config.n_teams
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    chunks = plan_chunks(n_seeds=args.n_seeds, chunk_size=args.chunk_size)
    env = {**os.environ, **_WORKER_ENV}

    for lo, hi in chunks:
        out = _chunk_file(args.checkpoint_dir, lo, hi)
        expected = (hi - lo) * n_teams
        if _valid_chunk_file(out, expected):
            print(f"[skip] chunk {lo:>4}-{hi:<4} already complete", flush=True)
            continue
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--worker", "--chunk-lo", str(lo), "--chunk-hi", str(hi), "--out", str(out),
            "--season", str(args.season), "--league-config", str(args.league_config),
            "--strategy-n-sims", str(args.strategy_n_sims), "--jitter", str(args.jitter),
            "--data-root", str(args.data_root),
        ]  # fmt: skip
        for attempt in range(1, args.max_retries + 1):
            print(f"[run ] chunk {lo:>4}-{hi:<4} attempt {attempt}/{args.max_retries}", flush=True)
            rc = subprocess.run(cmd, env=env).returncode
            if rc == 0 and _valid_chunk_file(out, expected):
                print(f"[ok  ] chunk {lo:>4}-{hi:<4}", flush=True)
                break
            print(f"[fail] chunk {lo:>4}-{hi:<4} rc={rc}; retrying", flush=True)
        else:
            print(f"[ERROR] chunk {lo}-{hi} failed after {args.max_retries} attempts", flush=True)
            return 1

    all_actual: list[LeagueResult] = []
    all_projected: list[LeagueResult] = []
    for lo, hi in chunks:
        out = _chunk_file(args.checkpoint_dir, lo, hi)
        # Re-validate at pool time: a checkpoint can vanish/truncate between the run loop and
        # here (concurrent run, AV, disk hiccup). Fail loud with a re-run hint, not a raw crash.
        if not _valid_chunk_file(out, (hi - lo) * n_teams):
            print(
                f"[ERROR] chunk {lo}-{hi} checkpoint missing/corrupt at pool time; "
                f"re-run to resume.",
                flush=True,
            )
            return 1
        actual, projected = load_results(json.loads(out.read_text()))
        all_actual += actual
        all_projected += projected
    result = aggregate(all_actual, all_projected, n_seeds=args.n_seeds, base_seed=0)
    print(format_result(result))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return _run_worker(args) if args.worker else _run_driver(args)


if __name__ == "__main__":
    sys.exit(main())
