"""Resumable hero-vs-bots strategy evaluation (run + report). See hero_cli for the core.

Always run in PowerShell with KMP_DUPLICATE_LIB_OK=TRUE + single-thread BLAS; the box
BSODs on long MC runs (memory h2h-backtest-native-crash) -- the sweep is resumable, so
re-run the same command to continue after a crash/reboot.
"""

from __future__ import annotations

from projections.draft.backtest.hero_cli import (  # noqa: F401
    _parse_args,
    _report,
    _run,
    _run_key,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return _run(args) if args.cmd == "run" else _report(args)


if __name__ == "__main__":
    raise SystemExit(main())
