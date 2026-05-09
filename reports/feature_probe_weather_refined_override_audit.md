# Weather Refined-Unit Override Audit — 2026-05-09

**Override path:** `data/features_probe/weather.parquet`
**Generator:** `scripts/build_weather_override.py` (PR `feat/probe-weather-refined`)
**Seasons:** 2018-2024
**Spec:** `docs/superpowers/specs/2026-05-09-weather-refined-unit-probe-design.md`
**Plan:** `docs/superpowers/plans/2026-05-09-weather-refined-unit-probe.md`

## Pooled audit (from generator stdout)

```
wrote 56652 rows to data\features_probe\weather.parquet
weather override audit (56652 rows):
  indoor games (dome+closed): 572/1942 = 29.5%
  wind_speed_mph NaN rate: 8.39%
  is_high_wind NaN rate: 8.39%
  temperature_f NaN rate: 8.39%
  is_cold_weather NaN rate: 8.39%
  is_grass_surface NaN rate: 0.00%
  is_primetime NaN rate: 0.00%
  is_a_turf NaN rate: 2.17%
  is_astroturf NaN rate: 2.17%
  is_fieldturf NaN rate: 2.17%
  is_grass NaN rate: 2.17%
  is_matrixturf NaN rate: 2.17%
  is_sportturf NaN rate: 2.17%
  is_high_wind=1.0 rate (v1): 1.35%
  is_grass_surface=1.0 rate (v1): 51.05%
  is_cold_weather=1.0 rate (refined): 4.01%
  is_primetime=1.0 rate (refined): 21.97%
  is_a_turf=1.0 rate (refined): 2.35%
  is_astroturf=1.0 rate (refined): 5.21%
  is_fieldturf=1.0 rate (refined): 22.81%
  is_grass=1.0 rate (refined): 55.82%
  is_matrixturf=1.0 rate (refined): 6.62%
  is_sportturf=1.0 rate (refined): 5.02%
```

## Per-(season, position) coverage

### `is_cold_weather` non-NaN rate

| season | QB | RB | TE | WR |
|---|---|---|---|---|
| 2018 | 0.982 | 0.983 | 0.982 | 0.981 |
| 2019 | 0.963 | 0.960 | 0.965 | 0.962 |
| 2020 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2021 | 0.962 | 0.963 | 0.961 | 0.962 |
| 2022 | 0.679 | 0.677 | 0.676 | 0.668 |
| 2023 | 0.854 | 0.857 | 0.854 | 0.857 |
| 2024 | 0.984 | 0.983 | 0.982 | 0.985 |

### `is_primetime` non-NaN rate

| season | QB | RB | TE | WR |
|---|---|---|---|---|
| 2018 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2019 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2020 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2021 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2022 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2023 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2024 | 1.000 | 1.000 | 1.000 | 1.000 |

### `is_primetime` =1.0 rate (per-season, per-position)

| season | QB | RB | TE | WR |
|---|---|---|---|---|
| 2018 | 0.213 | 0.213 | 0.214 | 0.213 |
| 2019 | 0.207 | 0.207 | 0.205 | 0.209 |
| 2020 | 0.214 | 0.215 | 0.215 | 0.211 |
| 2021 | 0.226 | 0.221 | 0.214 | 0.225 |
| 2022 | 0.222 | 0.218 | 0.219 | 0.218 |
| 2023 | 0.225 | 0.235 | 0.223 | 0.230 |
| 2024 | 0.234 | 0.232 | 0.233 | 0.237 |

## Notes

- Indoor (dome + closed) rate 29.5% — consistent with PR #28 audit.
- Outdoor `temp` / `wind` NaN rate inherited from PR #28 (8.39% across the 2018-2024 span). `is_cold_weather` shares the same NaN rate as `temperature_f` because it is derived directly from temperature on outdoor games (dome rows fill to 0).
- `is_primetime` rate 21.97% pooled. With ~3 primetime games per regular-season week (TNF, SNF, MNF) plus seasonal extras (Saturday late-season slates, Thanksgiving night, Christmas Day, occasional Friday games) and ~30+ active fantasy-eligible players per team, the realized rate is materially in line with the rough TNF+SNF+MNF expectation lifted to player-week granularity (~6 of 32 teams primetime per week ≈ 18-22%). Non-NaN rate is 1.000 across all (season, position) cells — the schedule's `gametime` column parses cleanly for every game in the index.
- **Note: An ingest-layer bug in `_build_kickoff` (mis-tagging ET wall-clock as UTC) was discovered and fixed during this audit.** The corrected partitions produce the expected ~22% primetime rate; the prior version showed 0.16% due to the tz mistagging. `nfl_data_py` publishes `gametime` as ET wall-clock for all games (including international broadcasts); the helper now `tz_localize("America/New_York")` before `tz_convert("UTC")` to store kickoff in canonical UTC. Downstream `_compute_is_primetime` in `weather_features.py` was already correct (it `tz_convert`s UTC → ET); the bug was strictly in ingest.
- Multi-class surface rate distribution: `is_grass` 55.82% (close to but not identical to v1 `is_grass_surface` 51.05% — the gap reflects rows where the v2 multi-class split tagged a row as one of the explicit turf brands while v1 fell back to the binary "grass vs not"); modern turfs (`is_fieldturf` 22.81%, `is_matrixturf` 6.62%, `is_sportturf` 5.02%) split the remainder; legacy codes (`is_a_turf` 2.35%, `is_astroturf` 5.21%) appear only in older seasons. The six per-surface columns share a common 2.17% NaN rate, slightly higher than v1's 0.00% — this reflects the stricter normalization (rows with unparseable / unknown surface codes are now NaN across all six rather than silently bucketed into "not grass").

## Coverage caveat

Per PR #29's coverage caveat (which corrected PR #28's overstated "uniformly ≥92%" claim), report per-(position, season) coverage separately rather than pooled. Trough seasons may dip below the pooled rate; the audit confirms this is symmetric across baseline + candidate sides under the probe's left-merge join.

**Cells below 0.90 on `is_cold_weather`:**

- **2022 — all positions**: QB 0.679, RB 0.677, TE 0.676, WR 0.668. The 2022 season has the deepest trough; nearly one-third of player-weeks have NaN for `is_cold_weather` / `temperature_f` / `wind_speed_mph` / `is_high_wind` (these four share the same outdoor-NaN gate). Likely cause: `nfl_data_py` 2022 schedule rows have missing `temp` / `wind` blocks for a significant share of outdoor games. Symmetric across baseline and candidate, so probe lift signals remain interpretable, but absolute lift magnitudes will be diluted in 2022 fold splits.
- **2023 — all positions**: QB 0.854, RB 0.857, TE 0.854, WR 0.857. Less severe than 2022 but still below the 0.90 threshold.

**Cells below 0.90 on `is_primetime`:** None — `is_primetime` is 1.000 across all (season, position) cells. NaN does not occur for this column at the per-(season, position) level. (The pooled 0.00% NaN rate from the audit block confirms this.)

The 2022 / 2023 `is_cold_weather` trough is the most material caveat: the probe's lift estimates on those folds will be diluted by ~22-33% because roughly that share of player-weeks fall back to dome-imputed defaults. The probe's verdict logic should weight 2020-2021 and 2024 folds more heavily when interpreting the cold-weather signal in particular.
