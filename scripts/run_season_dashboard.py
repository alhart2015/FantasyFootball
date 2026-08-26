"""Run the season dashboard on localhost.

    python scripts/run_season_dashboard.py --league-id 856974 --season 2026 --team-id 17 \
        --pool data/vorp_2026/critts_half16_snake.parquet

Read-only: both pages are readers over data the repo already computes. Nothing here writes to
the store, and there is no auth because it binds to localhost for one person.

Prints the resolved data root before starting, which the model repo does too and is worth
copying -- "which data am I actually looking at" is the first question when a number looks
wrong, and the answer should not require reading the source.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from projections.web import DashboardConfig, create_app

_DEFAULT_LEAGUE_DIR = Path("data/leagues/critts_2025_2026")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--league-id", type=int, required=True)
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--team-id", type=int, default=None, help="Highlight this team as mine.")
    p.add_argument("--pool", type=Path, required=True, help="VORP parquet for this league.")
    p.add_argument("--league-dir", type=Path, default=_DEFAULT_LEAGUE_DIR)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--n-sims", type=int, default=2000)
    p.add_argument("--port", type=int, default=5002)
    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = DashboardConfig(
        data_root=args.data_root,
        league_dir=args.league_dir,
        pool_path=args.pool,
        season=args.season,
        league_id=args.league_id,
        my_team_id=args.team_id,
        n_sims=args.n_sims,
    )

    missing = [
        str(path)
        for path in (config.data_root, config.league_dir, config.pool_path)
        if not path.exists()
    ]
    print(f"data root : {config.data_root.resolve()}")
    print(f"league dir: {config.league_dir.resolve()}")
    print(f"pool      : {config.pool_path.resolve()}")
    if missing:
        # Not fatal: the pages have empty states and saying which input is absent is more
        # useful than refusing to start, especially before the season has begun.
        print(f"WARNING: these do not exist yet: {', '.join(missing)}", file=sys.stderr)

    print(f"\nhttp://127.0.0.1:{args.port}\n")
    create_app(config).run(host="127.0.0.1", port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
