"""Generate a VORP table for an ARBITRARY LeagueConfig JSON (not one of the 3x3 presets).

`generate_preset_vorp_tables.py` covers the canonical scoring x size grid. Real leagues drift from
it — a custom pass-TD value, an extra FLEX, a shallower bench — and every one of those changes the
re-scored consensus and/or the replacement level, so a preset table is the wrong input. This script
runs the identical pipeline (`build_preset_table` + `reconcile_pool_gsis`) for a config read off
disk, so a bespoke league gets a correctly re-scored pool.

The config's `ruleset.name` must stay one of the known scoring FAMILIES (`ESPN_PPR`, `ESPN_HALF`,
`STANDARD`, `DRAFTKINGS`) — `ConsensusProjectionSchema` whitelists it, and
`resolve_espn_auction_dollars` keys its ESPN expert-column fallback on it. Custom per-stat values
(pass TD 5, etc.) ride on the same ruleset object and are what actually score the projections; the
name is only the family tag, and it is not persisted on the VORP table.

Run:  python scripts/generate_league_vorp_table.py \
          --league-config configs/my.league.json --out data/vorp_2026/my.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from projections.draft.assistant.pool_identity import real_gsis_by_key, reconcile_pool_gsis
from projections.draft.assistant.presets import DraftPreset
from projections.draft.league_config import LeagueConfig
from projections.schemas import ExternalProjectionSchema, VorpTableSchema
from projections.store import read_latest_partition

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling-script import (see below)

from generate_preset_vorp_tables import build_preset_table


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VORP table for a custom LeagueConfig JSON.")
    p.add_argument("--league-config", type=Path, required=True)
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--out", type=Path, required=True, help="Output parquet path.")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = LeagueConfig.model_validate_json(args.league_config.read_text())
    external = ExternalProjectionSchema.validate(
        read_latest_partition(args.data_root / "raw", "external_projections", season=args.season)
    )
    # DraftPreset is just the (config, table_path) carrier build_preset_table reads; the scoring-key
    # / label fields are display-only here.
    preset = DraftPreset(
        scoring_key=config.name,
        n_teams=config.n_teams,
        label=config.name,
        league_config=config,
        table_path=args.out,
    )
    id_map = pd.read_parquet(args.data_root / "raw" / "id_map.parquet")
    # Reconcile placeholder gsis to real ones so the table joins to weekly_stats (injury p) and
    # id_map (byes); re-validate after, since reconcile rewrites the unique join key.
    table = VorpTableSchema.validate(
        reconcile_pool_gsis(
            build_preset_table(external, preset), id_map, key_map=real_gsis_by_key(id_map)
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    priced = int(table["espn_auction_dollars"].notna().sum())
    print(
        f"{config.name} ({args.season}): {len(table)} players ({priced} ESPN-priced) -> {args.out}"
        f"\n  roster_size={config.roster_size} pool_needed={config.total_pool_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
