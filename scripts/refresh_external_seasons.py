"""Loop external_projections ingest + preset-table generation across seasons (TODO #49a inputs).

For each season: pull the ESPN+Sleeper snapshot, then regenerate that season's preset VORP tables
+ league configs under data/vorp_{season}/. 2026 is intentionally NOT in the default list — its
baseline snapshot (asof of the published Runs A-H) must be preserved; regenerate 2026 separately
via `generate_preset_vorp_tables.py --season 2026` (no re-ingest).

Usage (from the repo root; tables write cwd-relative — see the generator's Global Constraints):
    python scripts/refresh_external_seasons.py [--seasons 2021..2025] [--data-root data]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import generate_preset_vorp_tables  # sibling script (scripts/ on sys.path)
from pandera.errors import SchemaError

from projections.ingest.external_projections import (
    ExternalProjectionError,
    refresh_external_projections,
)

_log = logging.getLogger(__name__)
_DEFAULT_SEASONS: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)


def run(seasons: list[int], data_root: Path) -> dict[int, str]:
    """Ingest + regenerate per season; isolate per-season failures (one bad season — flaky API,
    missing file, or data that fails validation — must not discard the rest). Returns
    {season: "ok" | "failed: <reason>"}."""
    status: dict[int, str] = {}
    for year in seasons:
        try:
            refresh_external_projections(data_root, season=year)
            generate_preset_vorp_tables.main(["--season", str(year), "--data-root", str(data_root)])
            status[year] = "ok"
        # ExternalProjectionError: flaky/empty ingest pull; OSError: missing snapshot/file;
        # SchemaError: a season whose external data fails pandera validation in the generator.
        except (ExternalProjectionError, OSError, SchemaError) as exc:
            _log.warning("season %s failed: %s", year, exc)
            status[year] = f"failed: {exc}"
    return status


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Refresh external_projections + preset tables across seasons (TODO #49a)."
    )
    p.add_argument("--seasons", type=int, nargs="+", default=list(_DEFAULT_SEASONS))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    args = p.parse_args(argv)
    status = run(args.seasons, args.data_root)
    for year, st in status.items():
        print(f"  {year}: {st}")
    return 0 if all(v == "ok" for v in status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
