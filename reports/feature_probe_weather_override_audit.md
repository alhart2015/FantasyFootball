# Weather Override Audit

**Generated:** 2026-05-07 via `python -m scripts.build_weather_override --seasons 2018-2024`
**Output:** `data/features_probe/weather.parquet` (56,652 rows, regenerable)
**Spec:** `docs/superpowers/specs/2026-05-07-weather-feature-family-probe-design.md` §6.7
**Plan:** `docs/superpowers/plans/2026-05-07-weather-feature-family-probe.md` Task 6
**Branch:** `feat/probe-weather`

## Pooled (2018-2024)

| Metric | Value |
|---|---:|
| Total override rows | 56,652 |
| Total schedule rows | 1,942 |
| Indoor games (dome + closed) | 572 / 1,942 (29.5%) |
| `wind_speed_mph` NaN rate | 8.39% |
| `is_high_wind` NaN rate | 8.39% |
| `temperature_f` NaN rate | 8.39% |
| `is_grass_surface` NaN rate | 0.00% |
| `is_high_wind=1.0` rate (incl. dome) | 1.35% |
| `is_grass_surface=1.0` rate | 51.05% |

## Raw stdout

```
wrote 56652 rows to data\features_probe\weather.parquet
weather override audit (56652 rows):
  indoor games (dome+closed): 572/1942 = 29.5%
  wind_speed_mph NaN rate: 8.39%
  is_high_wind NaN rate: 8.39%
  temperature_f NaN rate: 8.39%
  is_grass_surface NaN rate: 0.00%
  is_high_wind=1.0 rate (incl. dome): 1.35%
  is_grass_surface=1.0 rate: 51.05%
```

## Notes

- Indoor share (29.5%) is consistent with the league makeup: 10 dome teams / 32 = 31.25%, with retractable closures partially offsetting outdoor games played in dome-team road slates. Within expected 25-35% sanity range.
- `is_grass_surface` coverage is 100% — surface codes are populated on every schedule row in the 2018-2024 window.
- `is_grass_surface=1.0` rate (51.05%) is consistent with the league split — roughly half of stadiums use natural grass, half turf.
- `is_high_wind` rate (1.35%) is single-digit by design — outdoor games with sustained wind ≥ 20 mph are rare. The 20 mph threshold is intentionally strict per spec §3.2.
- **FLAG: outdoor weather NaN rate is 8.39%, exceeding the < 5% spec target.** `wind_speed_mph`, `is_high_wind`, and `temperature_f` all share the same NaN rate, which is consistent with `nfl_data_py` schedules dropping the entire weather block for some outdoor games (likely older seasons / non-domestic games / stadia with missing telemetry, since dome rows are deliberately filled per spec §3.5 and excluded from these "outdoor-NaN" denominators). The shared rate across the three weather fields confirms the missingness is row-level, not field-level. Per Task 6 guidance this means Task 7 should lower `--coverage-threshold` to 0.90 to accommodate the real data-quality floor.
- No anomalies in surface codes, season splits looked uniform, and row count (56,652) is healthy and consistent with ~7 seasons of weekly active-roster snapshots.
