# Weekly report — design

**Branch** `feat/weekly-report` · **Date** 2026-09-08 · **Sub-project** Mid-season Manager

## 1. The question

*It is Sunday morning. What do I do?*

Four tools answer four parts of that, and answering it today means running four commands and
reading four headers. One command, one document.

## 2. What this is actually fixing

The naive version — a script that runs the four in sequence — was measured before being
rejected. All four together take **~31 seconds** (standings 2s, start/sit 3s, waivers 3s,
trades 23s), so **speed is not the problem** and any argument built on "four redundant ESPN
calls" is weak.

The problem is that **the four tools do not currently agree with each other**, and stacking
them exposes it. From a full audit of the four scripts:

- `fetch_league_payload(league_id, season, creds)` — the **same call, four times**.
- `build_my_team(payload, pool, id_map, weekly_stats, config, ...)` — **byte-identical argument
  lists** in the waiver and start/sit paths.
- `rostered_limit = config.n_teams * (sum(config.roster_slots.values()) + 2)` — the same line,
  copy-pasted into two scripts, and it sizes a network request.
- `fetch_free_agents(..., scoring_period=week, limit=rostered_limit, statuses=("ONTEAM",))` —
  the **same request** in both; the payload is fully reusable.
- The week is derived **three separate ways** (inside `project_league_standings`, inside
  `build_my_team`, and inline in `trade_analyzer`).
- `LeagueConfig` comes from **two different sources**: `league_config.json` on disk for waivers
  and start/sit, `build_league_config(payload)` for trades. `build_league_config` drops the K
  slot, and `n_teams`/`roster_slots` feed `rostered_limit` — so even the *size of the network
  request* differs by route.
- `trade_analyzer` is the only one that skips `_PYARROW_STR` + `VorpTableSchema.validate` on
  the pool, so it can carry a different `gsis_id` dtype into its joins.
- `attach_is_rookie` and `load_store_availability` each rescan `weekly_stats` for
  2018..season-1. Running all four = **up to eight full historical scans**, none cached.

They agree today. **Nothing makes them agree.** That is the actual defect, and #175 already
showed what it looks like when one of them drifts: the standings simulator was injury-blind
while the other two were not, and a CLI and a web page printed different playoff odds for the
same team.

## 3. Share the inputs. Do NOT unify the outputs.

The distinction is the whole design, and getting it backwards would be worse than doing
nothing.

**Shared — because divergence here is accidental:** the payload, the credentials, the
`LeagueConfig`, the week, the validated pool, the `id_map`, `weekly_stats`, the `MyTeamRun`,
the ONTEAM projections payload, `VarianceParams`, and availability.

**NOT shared — because divergence here is deliberate and load-bearing:**

- **This week's points.** The waiver tool scores ESPN's weekly line once under the ruleset.
  Start/sit blends ESPN with Sleeper per-stat and applies its own injury multiplier. These are
  *supposed* to differ — a second opinion is the entire reason start/sit exists. Forcing one
  number would delete the feature.
- **Rest-of-season value.** `build_my_team` subtracts real points-to-date;
  `project_league_standings` currently passes `points_to_date={}` (a standing TODO);
  `trade_analyzer` blends the pool with an external ESPN snapshot. Three horizons for three
  questions.
- **Where injuries are applied.** Season-multiplier on the pool for the simulators, weekly
  multiplier for start/sit. #175 made the *simulator* side consistent; the weekly side is a
  different horizon and stays separate.

**So the report labels rather than reconciles.** Where two sections show a number for the same
player, each says which basis it used. A section that quietly averaged them would be inventing
a quantity no tool computes.

## 4. `InSeasonContext`

New module `src/projections/midseason/context.py`. One dataclass, built once, handed to each
section.

```python
@dataclass(frozen=True)
class InSeasonContext:
    target: LeagueTarget
    creds: EspnCredentials
    payload: Mapping[str, Any]
    config: LeagueConfig
    pool: pd.DataFrame           # validated, _PYARROW_STR, is_rookie attached — ONCE
    id_map: pd.DataFrame
    weekly_stats: pd.DataFrame
    my_team: MyTeamRun
    week: int
    data_root: Path
```

with **memoised** accessors for the expensive optional pieces, because three of the four
sections need some and none needs all:

```python
    def availability(self) -> PlayerAvailability   # rescans 2018..season-1; do it once
    def variance_params(self) -> VarianceParams
    def onteam_payload(self) -> Mapping[str, Any]  # the shared ONTEAM fetch
    def roster(self) -> pd.DataFrame               # my team's rows
```

**`week` is a field, not a method, and it is the single horizon.** Everything downstream —
the injury discount, `scoring_period` on the free-agent fetch, `games_remaining` — reads this
one integer. Two copies of a horizon that must stay identical is how an IR player gets
discounted over seventeen games while the simulation prorates him over nine, and #175's review
caught exactly that shape.

**`onteam_payload` is memoised for a reason beyond speed.** It carries `scoring_period=week`,
so it is only reusable if both consumers agree on the week — which, with `week` a context
field, they now do by construction rather than by luck.

## 5. Each script grows a `report(ctx, args)`

Not four new result dataclasses. The existing printing is good and specific; inventing a
common result type would flatten four deliberately different reports into one shape.

```python
def report(ctx: InSeasonContext, args: argparse.Namespace) -> int:
    """Analysis + printing, reading everything from `ctx`."""

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ctx = build_context(args)          # resolve, fetch, validate, derive the week
    return report(ctx, args)
```

Every script stays runnable exactly as it is today, with the same flags and the same output.
`weekly_report` builds one context and calls the four `report`s.

## 6. Section order: actions first, context after

```
START/SIT   — deadline is kickoff
WAIVERS     — deadline is Wednesday
STANDINGS   — where this is heading
TRADES      — no deadline, slowest to act on
```

Not the order the tools were built in, and not "context first". A Sunday-morning reader has a
lineup lock coming; the thing with the nearest deadline goes at the top, and the strategic
sections go where they can be skipped.

## 7. Flags

`--only start-sit,waivers` and `--skip trades` to cut the run down (trades is 23 of the 31
seconds). `--fast` propagates to the sections that have one. The five shared league flags
behave exactly as they do everywhere else.

**A failing section must not kill the report.** Each is wrapped: the failure prints under that
section's header and the rest continue. A weekly report that dies at section three because
`external_projections` is stale has told you nothing, and the sections are genuinely
independent.

## 8. Scope cuts

- **No new analysis.** Every number already exists; this is composition.
- **No writing anything back to ESPN.** Still all GETs.
- **No dashboard page.** CLI first, as with start/sit.
- **No caching to disk.** The context lives for one run.
- **The `points_to_date={}` TODO in `project_league_standings` is NOT fixed here.** It is a real
  inconsistency (§3) but it is a modelling change with its own blast radius, and folding it into
  a composition PR would hide it. Issue instead.

## 9. Risks

| Risk | Why it bites | Mitigation |
|---|---|---|
| Refactoring four working tools | These produce numbers the user acts on; a silent regression is worse than no wrapper | Each script migrated in its own commit, with a before/after output diff on the live league |
| `build_league_config` vs `league_config.json` | Picking one could change `trade_analyzer`'s roster slots and its ruleset | **Measured 2026-09-08: they are identical on the live league** — same `roster_slots`, no ruleset difference, same derived `rostered_limit` of 272. So the switch is a no-op today. The context reads the FILE (three of four already do) and **warns when the derived config differs**, turning a latent divergence into a visible one instead of trusting today's agreement to hold |
| Memoised accessors hiding cost | A section that quietly triggers an eight-second scan | Each accessor logs on first computation |
| Section isolation | A shared context means one section's mutation could corrupt another's | `InSeasonContext` is frozen and every accessor returns a copy of any frame it hands out |

## 10. Reuse — what this does NOT rewrite

`build_my_team`, `project_league_standings`, `rank_free_agents`, `recommend_start_sit`,
`generate_all`/`simulate_trades`, `injury_adjusted_pool_at_current_week`,
`add_league_arguments`/`resolve_league_target`, and every existing print helper. New code is
the context, the four `report(ctx, args)` extractions, and the composing script.
