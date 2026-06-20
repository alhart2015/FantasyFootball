"""One-time maintenance: reconcile placeholder gsis in existing preset VORP tables.

The per-season preset pools (`data/vorp_{season}/*.parquet`) were generated from raw
external-projection snapshots that assigned 100% placeholder (`99-`) gsis_ids; those
snapshots have since been deleted, so the tables cannot be regenerated through the normal
pipeline. This rewrites each table IN PLACE with real gsis recovered via id_map
(name+position), so the availability model (injury p + byes) joins. Idempotent: a table
already carrying real gsis is rewritten unchanged.

Run:  python scripts/reconcile_vorp_gsis.py [--data-root data] [--seasons 2021 ... 2026]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.draft.assistant.pool_identity import reconcile_pool_gsis

_DEFAULT_SEASONS = (2021, 2022, 2023, 2024, 2025, 2026)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--seasons", type=int, nargs="+", default=list(_DEFAULT_SEASONS))
    args = p.parse_args(argv)

    id_map = pd.read_parquet(args.data_root / "raw" / "id_map.parquet")
    total = 0
    for season in args.seasons:
        season_dir = args.data_root / f"vorp_{season}"
        if not season_dir.is_dir():
            print(f"  skip {season}: {season_dir} not found")
            continue
        for table_path in sorted(season_dir.glob("*.parquet")):
            pool = pd.read_parquet(table_path)
            before = int(pool["gsis_id"].astype(str).str.startswith("99-").sum())
            fixed = reconcile_pool_gsis(pool, id_map)
            after = int(fixed["gsis_id"].astype(str).str.startswith("99-").sum())
            fixed.to_parquet(table_path, index=False)
            total += 1
            print(f"  {table_path.name} ({season}): placeholders {before} -> {after}")
    print(f"reconciled {total} table(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
