"""One league, fetched and assembled once, for every in-season tool to read.

**Shares the INPUTS, not the outputs.** The four tools each fetched the same payload, derived
the current week three different ways, read `LeagueConfig` from two different sources, and
rescanned a decade of `weekly_stats` up to eight times per combined run. All of that is
accidental divergence, and #175 showed what it costs: the standings simulator was injury-blind
while the other two tools were not, so a CLI and a web page printed different playoff odds for
the same team.

What it deliberately does NOT unify is the analysis. The waiver tool scores ESPN's weekly line
once under the ruleset; start/sit blends ESPN with Sleeper per-stat and applies its own weekly
injury multiplier. Those numbers are *supposed* to differ — a second opinion is the entire
reason start/sit exists — so a combined report labels them rather than reconciling them.
Averaging the two would invent a quantity no tool computes.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.assistant.league_profile import LeagueTarget, resolve_league_target
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.assistant.rookies import attach_is_rookie
from projections.draft.league_calendar import LeagueCalendar
from projections.draft.league_config import LeagueConfig
from projections.ingest.espn_league import (
    EspnCredentials,
    build_league_config,
    fetch_free_agents,
    fetch_league_payload,
    parse_rosters,
    parse_schedule,
    parse_teams,
)
from projections.midseason.my_team import MyTeamRun, build_my_team
from projections.midseason.standings import first_unplayed_week
from projections.schemas import _PYARROW_STR, VorpTableSchema
from projections.store import read_partition

logger = logging.getLogger(__name__)


@dataclass
class InSeasonContext:
    """Everything the in-season tools share, resolved once.

    Not frozen, because the memo slots below are assigned on first use.

    **`roster()` returns a copy; `pool`, `id_map`, `weekly_stats` and `payload` do not.** They
    are handed to all four sections by reference, and today no section mutates them — but that
    is a property of the current callers, not something this object enforces. An in-place
    `df["col"] = ...` in any section would reach the other three. `roster()` is a copy because
    it is derived per call anyway; making the rest copies would mean four copies of the pool
    per report for a hazard that has not occurred. Stated rather than claimed away.
    """

    target: LeagueTarget
    creds: EspnCredentials
    payload: dict[str, Any]
    #: None when the caller did not need one. `projected_standings` never read a
    #: `league_config.json` — `project_league_standings` derives its own from the payload —
    #: so demanding one here would have made a working command start failing. Sections that
    #: need it call `require_config()`.
    config: LeagueConfig | None
    #: Schema-validated with `is_rookie` attached, normalised ONCE — the trade analyzer
    #: skipped both and could carry a different `gsis_id` dtype into its joins.
    pool: pd.DataFrame
    id_map: pd.DataFrame
    weekly_stats: pd.DataFrame
    #: **The week being ASKED ABOUT.** `--week` moves it. This is what prices a weekly
    #: projection: `scoring_period` on the ESPN request, the blend start/sit solves, the
    #: lineup gain the waiver tool filters on.
    week: int
    #: **The week the league is actually IN**, from the schedule. `--week` does NOT move it.
    #:
    #: These are two different things and collapsing them was a real bug on this branch.
    #: `project_league_standings` takes no week and re-derives this one internally, so a
    #: season simulation always replays from here — and the injury discount handed to it must
    #: use the same number, because `season_multiplier` divides games missed by games
    #: REMAINING. With `--week 12` during real week 3, a shared horizon haircut IR players
    #: over 6 games while the simulator replayed 15, and the playoff odds still looked fine.
    #: That is the exact failure #175's `injury_adjusted_pool_at_current_week` was written to
    #: prevent, reintroduced here by making one week serve both jobs.
    schedule_week: int
    data_root: Path
    my_team_id: int | None
    #: Anything the assembly wants the reader to distrust. Printed by the tools, not here.
    notes: tuple[str, ...] = ()

    _my_team: MyTeamRun | None = field(default=None, repr=False)
    _availability: PlayerAvailability | None = field(default=None, repr=False)
    _params: VarianceParams | None = field(default=None, repr=False)
    _onteam: dict[str, Any] | None = field(default=None, repr=False)

    def availability(self) -> PlayerAvailability:
        """Fitted from 2018..season-1. Memoised because it rescans that whole history."""
        if self._availability is None:
            logger.info("loading availability history (2018-%d)", self.target.season - 1)
            self._availability = load_store_availability(
                self.pool, season=self.target.season, data_root=self.data_root
            )
        return self._availability

    def variance_params(self) -> VarianceParams:
        if self._params is None:
            self._params = VarianceParams.load()
        return self._params

    def rostered_limit(self) -> int:
        """How many players to ask ESPN for when pricing rostered players.

        Sized to the WHOLE LEAGUE on purpose, and it lives here because it was the same line
        copy-pasted into two scripts while also sizing a network request. Sharing a
        free-agent-sized limit once silently dropped the waiver tool's own starters from the
        projections, left holes in the baseline lineup, and inflated every candidate's gain.
        """
        config = self.require_config()
        return config.n_teams * (sum(config.roster_slots.values()) + 2)

    def onteam_payload(self) -> dict[str, Any]:
        """ESPN's weekly projections for every ROSTERED player, at `self.week`.

        Memoised for correctness as much as for the round trip: the request carries
        `scoring_period=self.week`, so it is only reusable between the waiver tool and
        start/sit if both agree on the week. With `week` a field of this object they agree by
        construction rather than by luck.
        """
        if self._onteam is None:
            logger.info("fetching ONTEAM projections for week %d", self.week)
            self._onteam = fetch_free_agents(
                self.target.league_id,
                self.target.season,
                self.creds,
                scoring_period=self.week,
                limit=self.rostered_limit(),
                statuses=("ONTEAM",),
            )
        return self._onteam

    def my_team(self) -> MyTeamRun:
        """The `MyTeamRun`, built on first use.

        **Lazy on purpose.** `projected_standings` needs no team at all, and building this
        eagerly leaked its rest-of-season diagnostics into that report — which already prints
        the same warning from its own `run.diagnostics`. A shared context must not put notes
        on a section's page that the section never asked for.
        """
        if self._my_team is None:
            self._my_team = build_my_team(
                self.payload,
                self.pool,
                self.id_map,
                self.weekly_stats,
                self.require_config(),
                my_team_id=self.require_team_id(),
                season=self.target.season,
            )
        return self._my_team

    def require_config(self) -> LeagueConfig:
        """The league config, or a clear failure naming the flag that supplies it."""
        if self.config is None:
            raise ValueError(
                "this section needs a league_config.json; pass --league-dir <dir>, or drop the "
                "explicit arguments and let the league profile supply them"
            )
        return self.config

    def require_team_id(self) -> int:
        if self.my_team_id is None:
            raise ValueError("--team-id is required for this section.")
        return self.my_team_id

    def roster(self) -> pd.DataFrame:
        """My team's `parse_rosters` rows. A fresh copy each call — see the class docstring."""
        rosters = parse_rosters(dict(self.payload))
        return rosters[rosters["team_id"] == self.require_team_id()].copy()

    def teams(self) -> pd.DataFrame:
        return parse_teams(dict(self.payload))


def load_pool(pool_path: Path, *, season: int, data_root: Path) -> pd.DataFrame:
    """The pool, normalised the one way every tool should have been normalising it.

    **The `astype` is kept byte-for-byte from the three scripts that had it, including the fact
    that it does not survive.** `VorpTableSchema` declares `gsis_id: Series[str]`, which is
    object dtype in pandera, so validation coerces the pyarrow string straight back — the line
    has been a no-op in every caller. Removing it here would be a behaviour change smuggled
    into a composition PR, and *fixing* it means changing the schema, which every VORP consumer
    in the repo reads. Left alone deliberately, with an issue against the schema.

    What this function does actually guarantee: one schema validation and one `is_rookie` pass,
    for all four tools, from one code path.
    """
    pool = pd.read_parquet(pool_path)
    pool["gsis_id"] = pool["gsis_id"].astype(_PYARROW_STR)
    return attach_is_rookie(VorpTableSchema.validate(pool), season=season, data_root=data_root)


def _config_notes(from_file: LeagueConfig, payload: dict[str, Any]) -> tuple[str, ...]:
    """Warn when the on-disk config and ESPN's live settings disagree.

    Three of the four tools read `league_config.json` and the trade analyzer derived its own
    from the payload. Measured identical on the live league on 2026-09-08 — same
    `roster_slots`, no ruleset difference, same derived request size — so unifying on the file
    is a no-op today. **This check exists for the day that stops being true**, because the
    failure would otherwise be silent: `roster_slots` feeds `rostered_limit`, so a drift would
    change the size of a network request and quietly thin the projections behind every number.
    """
    espn_logger = logging.getLogger("projections.ingest.espn_league")
    previously = espn_logger.disabled
    try:
        # Silenced: `build_league_config` narrates ESPN's scoring categories, which is useful
        # when a tool is genuinely deriving its config and pure noise when we are only
        # cross-checking one we already have. It printed three lines on every run.
        espn_logger.disabled = True
        derived = build_league_config(dict(payload), name=from_file.name)
    except Exception as exc:
        return (f"could not derive a league config from ESPN to cross-check the file: {exc}",)
    finally:
        espn_logger.disabled = previously

    notes: list[str] = []
    if derived.roster_slots != from_file.roster_slots:
        notes.append(
            "league_config.json and ESPN disagree on roster slots — "
            f"file {dict(sorted((s.value, n) for s, n in from_file.roster_slots.items()))}, "
            f"ESPN {dict(sorted((s.value, n) for s, n in derived.roster_slots.items()))}. "
            "The file wins here; it also sizes the rostered-player request."
        )
    if derived.n_teams != from_file.n_teams:
        notes.append(
            f"league_config.json says {from_file.n_teams} teams, ESPN says {derived.n_teams}."
        )
    if derived.ruleset.model_dump(exclude={"name"}) != from_file.ruleset.model_dump(
        exclude={"name"}
    ):
        notes.append(
            "league_config.json and ESPN disagree on scoring. Every points number in every "
            "section is computed under the file's ruleset."
        )
    return tuple(notes)


def build_context(
    args: argparse.Namespace,
    *,
    require_team_id: bool = True,
    require_config: bool = True,
) -> InSeasonContext:
    """Resolve, fetch and assemble. The one place any in-season tool starts.

    Raises `ValueError` for an unusable target (the callers turn that into an exit code and a
    message, as they always have) and lets `EspnLeagueError` / `OSError` out for the same
    reason.

    **`args.week` moves everything or nothing.** Today `--week` shifts the waiver and start/sit
    horizons and leaves the standings and trade horizons on the real week, which means the same
    flag means two different things depending on which tool reads it. Here it sets `week` for
    every consumer.
    """
    target = resolve_league_target(args, require_team_id=require_team_id)
    return assemble_context(target, args, require_config=require_config)


def assemble_context(
    target: LeagueTarget, args: argparse.Namespace, *, require_config: bool = True
) -> InSeasonContext:
    """The fetch-and-load half, for a target that is already resolved.

    Split out because **`resolve_league_target` deletes the five league flags off the
    Namespace**, so it cannot be called twice on one `args`. A caller that needs the target
    before committing to the full assembly — the waiver tool lists the league's teams and
    exits when no team was named, and loading a pool and an id_map to print that list would be
    absurd — resolves once and hands the result here.
    """
    my_team_id = target.team_id
    league_config_path = (
        target.require_league_config() if require_config else target.league_config_path
    )

    creds = EspnCredentials.resolve(args.credentials)
    payload = fetch_league_payload(target.league_id, target.season, creds)

    # A path that was demanded and is not there fails HERE, naming the file. Letting it fall
    # through as `config=None` meant `require_config()` raised later from inside `report()`,
    # which several callers invoke outside their try block -- so the user got a traceback, and
    # a message telling them to pass `--league-dir` when they just had.
    if require_config and (league_config_path is None or not league_config_path.exists()):
        raise ValueError(
            f"no league_config.json at {league_config_path}. Pass --league-dir <dir> "
            "containing one, or drop the explicit arguments and let the league profile supply "
            "them."
        )
    config = (
        LeagueConfig.model_validate_json(league_config_path.read_text(encoding="utf-8"))
        if league_config_path is not None and league_config_path.exists()
        else None
    )
    pool = load_pool(target.pool, season=target.season, data_root=args.data_root)
    id_map = pd.read_parquet(args.data_root / "raw" / "id_map.parquet")

    try:
        weekly_stats = read_partition(args.data_root / "raw", "weekly_stats", season=target.season)
    except FileNotFoundError:
        # Absent before the first refresh of a new season. An empty frame means "no points
        # scored yet", which is the truth then, rather than an abort.
        weekly_stats = pd.DataFrame()

    # **Derived from the payload, not from `my_team`.** The week is the league's, not one
    # franchise's, and tying it to `MyTeamRun` would mean a tool that needs no team (projected
    # standings) could not have a week. It is the same derivation `build_my_team` AND
    # `project_league_standings` perform internally, so all three agree; tests pin that.
    calendar = LeagueCalendar.from_espn_settings(
        (payload.get("settings", {}) or {}).get("scheduleSettings", {}) or {}
    )
    schedule = parse_schedule(dict(payload), parse_teams(dict(payload)))
    schedule_week = first_unplayed_week(schedule, calendar) if not schedule.empty else 1

    return InSeasonContext(
        target=target,
        creds=creds,
        payload=dict(payload),
        config=config,
        pool=pool,
        id_map=id_map,
        weekly_stats=weekly_stats,
        week=getattr(args, "week", None) or schedule_week,
        schedule_week=schedule_week,
        data_root=args.data_root,
        my_team_id=my_team_id,
        notes=_config_notes(config, dict(payload)) if config is not None else (),
    )
