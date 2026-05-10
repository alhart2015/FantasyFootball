"""Internal: cross-check refined-weather coverage against PR #30 audit.

Used by Plan Task 8 (real-data execution). Prints a per-(position, season)
coverage table for the 8 refined weather cols. Compare against
`reports/feature_probe_weather_refined_override_audit.md` to catch
builder-wiring drift before the dual-run gate.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_COLS = (
    "is_cold_weather",
    "is_a_turf",
    "is_astroturf",
    "is_fieldturf",
    "is_grass",
    "is_matrixturf",
    "is_sportturf",
    "is_primetime",
)


def main() -> None:
    for pos in ("rb", "wr"):
        print(f"=== {pos.upper()} ===")
        for season in range(2021, 2025):
            season_frames: list[pd.DataFrame] = []
            for week in range(1, 19):
                p = Path(f"data/features/{pos}/season={season}/week={week:02d}/part.parquet")
                if p.exists():
                    season_frames.append(pd.read_parquet(p))
            if not season_frames:
                print(f"  season {season}: NO DATA")
                continue
            df = pd.concat(season_frames, ignore_index=True)
            print(f"  season {season} ({len(df)} rows):")
            for c in _COLS:
                if c in df.columns:
                    rate = df[c].notna().mean()
                    print(f"    {c}: non-NaN rate = {rate:.4f}")
                else:
                    print(f"    {c}: MISSING from cache")


if __name__ == "__main__":
    main()
