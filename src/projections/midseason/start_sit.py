"""Should I change my lineup this week?

The blend is the reason this tool exists. A start/sit report built on ESPN's weekly
projection alone tells a manager what the ESPN app already shows him, which is no reason to
run anything. Two independent sources, scored under *this* league's ruleset, disagreeing
where the call is close — that is a number he cannot get anywhere else.

Keyed by **ESPN player id, not gsis_id**, for the reason `waivers` records: going through the
crosswalk drops exactly the just-signed players an in-season tool is about. Sleeper arrives
keyed by `sleeper_id` and is crosswalked *in*; a Sleeper row the `id_map` cannot place falls
back to ESPN alone rather than vanishing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from projections.draft.assistant.performance_variance import (
    SEASON_GAMES,
    VarianceParams,
    sample_weekly_points,
)
from projections.draft.roster_eligibility import (
    NON_STARTING_SLOTS,
    choose_starters_with_slots,
)
from projections.ingest.external_projections import WEEKLY_BLEND_FIELDS
from projections.ingest.identity import normalize_join_id
from projections.midseason.injuries import weekly_multiplier
from projections.midseason.waivers import is_on_ir, player_id
from projections.schemas import InjuryStatus, RosterSlot, Ruleset, parse_injury_status
from projections.scoring.score import expected_points

_ESPN_SUFFIX = "_espn"
_SLEEPER_SUFFIX = "_slp"


@dataclass(frozen=True)
class BlendedPoints:
    """One player's weekly points, and the evidence behind them."""

    #: Blended points under the league ruleset. The number the lineup is solved on.
    points: float
    #: ESPN's line scored alone, or None when ESPN did not price him.
    espn: float | None
    #: Sleeper's line scored alone, or None when Sleeper did not price him.
    sleeper: float | None
    #: "both" | "espn" | "sleeper" — printed, so a missing spread is legible rather than
    #: mysterious. A K or D/ST is always "espn": `parse_sleeper_weekly` filters to QB/RB/WR/TE.
    sources: str

    @property
    def spread(self) -> float | None:
        """How far apart the sources are. None unless both priced him."""
        if self.espn is None or self.sleeper is None:
            return None
        return abs(self.espn - self.sleeper)


def _score(line: Mapping[str, float], ruleset: Ruleset) -> float:
    return float(expected_points(dict(line), ruleset))


def _line(row: Mapping[str, Any], suffix: str) -> dict[str, float]:
    """The present, non-null stat fields of one side of the merge."""
    out: dict[str, float] = {}
    for field in WEEKLY_BLEND_FIELDS:
        value = row.get(f"{field}{suffix}")
        # `is None` first: the merge leaves a missing side as NaN/NA, but a column absent
        # entirely returns None from `.get`, and `pd.notna(None)` is False either way while
        # only the explicit check narrows the type for mypy.
        if value is None or pd.isna(value):
            continue
        out[field] = float(value)
    return out


def _sleeper_keyed_by_espn_id(sleeper: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    """Crosswalk `sleeper_id` -> `espn_id`, dropping rows the map cannot place.

    Dropping is right here and would be wrong in the other direction: an unplaceable Sleeper
    row means we lose one source for a player ESPN still prices, which the `sources` column
    reports. Losing the ESPN row would lose the player.
    """
    crosswalk = id_map[["espn_id", "sleeper_id"]].dropna(subset=["espn_id", "sleeper_id"]).copy()
    # BOTH keys through `normalize_join_id`: its docstring names `espn_id` as a column the
    # id_map has stored float-stringified ("4374302.0"). Today's file is clean, and a bare
    # `.astype(str)` would survive that regression by yielding an empty inner merge -- the
    # whole roster silently reading "espn only", which is the exact degradation the notes
    # exist to warn about, reported as a coverage problem rather than a join bug.
    crosswalk["sleeper_id"] = normalize_join_id(crosswalk["sleeper_id"])
    crosswalk["espn_id"] = normalize_join_id(crosswalk["espn_id"])
    # Deduped on both sides. `espn_gsis_crosswalk` records that the live id_map holds ESPN ids
    # mapping to two different players; two sleeper_ids sharing one espn_id would otherwise fan
    # out through the outer merge and the last row would win, attaching an arbitrary Sleeper
    # line to the wrong player.
    crosswalk = crosswalk.drop_duplicates("sleeper_id").drop_duplicates("espn_id")

    keyed = sleeper.copy()
    keyed["sleeper_id"] = normalize_join_id(keyed["sleeper_id"])
    keyed = keyed.merge(crosswalk, on="sleeper_id", how="inner")
    return keyed.reindex(columns=["espn_id", *WEEKLY_BLEND_FIELDS])


def blend_weekly_points(
    espn: pd.DataFrame,
    sleeper: pd.DataFrame,
    id_map: pd.DataFrame,
    *,
    weight_espn: float,
    ruleset: Ruleset,
) -> dict[str, BlendedPoints]:
    """Blend two weekly stat lines per-stat, score once, key by ESPN id.

    `espn` is `espn_weekly.espn_weekly_statlines` shape; `sleeper` is
    `ingest.sleeper_weekly_projections.parse_sleeper_weekly` shape.

    **Per-stat, not per-player.** `Ruleset` is linear, so the two agree whenever both sources
    report the same fields — the difference is the field only one source carries. Sleeper omits
    receptions for a player and ESPN does not: that reception count belongs in the blend at
    ESPN's full weight, and averaging the two scored totals instead would read low by half of
    ESPN's reception points with nothing on screen to say so. Same rule as
    `dfs.blend.blend_statlines`.

    **A player neither source prices is absent from the result, not present at 0.0.**
    `choose_starters` reads a missing value as unstartable, which is how bye weeks work here
    without a rule about bye weeks; a 0.0 would instead let him fill a slot nobody else is
    eligible for.
    """
    if not 0.0 <= weight_espn <= 1.0:
        raise ValueError(f"weight_espn must be in [0, 1], got {weight_espn}")

    left = espn.reindex(columns=["espn_id", *WEEKLY_BLEND_FIELDS]).copy()
    left["espn_id"] = left["espn_id"].astype(str)
    right = _sleeper_keyed_by_espn_id(sleeper, id_map)

    # Reindexing both sides to the FULL field set above is what guarantees every field gets a
    # suffix on the merge. Without it a field present in only one frame stays unsuffixed and
    # `row.get(f"{field}_slp")` silently reads None -- that source's contribution vanishes.
    # `dfs.blend` documents the same trap.
    merged = left.merge(
        right,
        on="espn_id",
        how="outer",
        suffixes=(_ESPN_SUFFIX, _SLEEPER_SUFFIX),
        indicator=True,
    )

    out: dict[str, BlendedPoints] = {}
    for _, row in merged.iterrows():
        espn_line = _line(row, _ESPN_SUFFIX)
        sleeper_line = _line(row, _SLEEPER_SUFFIX)
        side = str(row["_merge"])
        has_espn = side in {"left_only", "both"}
        has_sleeper = side in {"right_only", "both"}

        blended: dict[str, float] = {}
        for field in WEEKLY_BLEND_FIELDS:
            weighted = [
                (weight_espn, espn_line.get(field)),
                (1.0 - weight_espn, sleeper_line.get(field)),
            ]
            total = sum(w * v for w, v in weighted if v is not None)
            weights = sum(w for w, v in weighted if v is not None)
            if weights > 0:
                blended[field] = total / weights

        out[str(row["espn_id"])] = BlendedPoints(
            points=_score(blended, ruleset),
            espn=_score(espn_line, ruleset) if has_espn else None,
            sleeper=_score(sleeper_line, ruleset) if has_sleeper else None,
            sources="both" if has_espn and has_sleeper else ("espn" if has_espn else "sleeper"),
        )
    return out


def startable_points(projected: float | None, status: InjuryStatus) -> float | None:
    """Weekly points after the injury rule, or None when the player cannot be started at all.

    **Not `waivers.adjusted_weekly_points`.** That takes one `source_is_injury_aware` boolean,
    and a blend of an injury-aware source and an unmeasured one has no honest value for it.
    ESPN zeroes the players it lists as `Out`; Sleeper's behaviour there has never been
    measured, so a blended `Out` player carries roughly *half* of Sleeper's projection.
    Declaring the source injury-aware would leave that half standing — a perfectly plausible
    number for a player who will not take a snap. We apply the multiplier ourselves instead.

    **`INJURY_RESERVE` is the one None, and the distinction is not pedantry.** `choose_starters`
    reads `None` as "cannot fill this slot at all" and `0.0` as "a real projection of nothing,
    which can still fill a slot no one else is eligible for". IR earns `None` because it is a
    *roster* slot ESPN will not let you start out of — structural, not statistical.
    `OUT` and `SUSPENSION` earn 0.0: they sort below every healthy player and get slotted only
    when nobody else is eligible, which is exactly what a manager forced to field a body does.
    `DOUBTFUL` keeps its measured 0.04 for the same reason — against a bye-week alternative,
    who really is `None`, starting him is the right call and the tool must be able to say so.
    """
    if projected is None or status is InjuryStatus.INJURY_RESERVE:
        return None
    return float(projected) * weekly_multiplier(status, source_is_injury_aware=False)


@dataclass(frozen=True)
class LineupRow:
    """One roster player as the lineup solver sees him."""

    #: ESPN player id. The key everything here joins on; see the module docstring.
    player_id: int
    name: str
    position: str
    status: InjuryStatus
    #: The slot ESPN has him in right now, or None when he is benched, on IR, or in a slot
    #: id we do not recognise.
    current_slot: RosterSlot | None
    #: Weekly points after `startable_points`. None means unstartable, not zero.
    points: float | None
    #: The evidence behind `points`, for display. None when neither source priced him.
    blend: BlendedPoints | None
    #: Feeds `VarianceParams.log_sd`, which has a genuinely wider rookie tier. Defaulting
    #: every player to veteran made `p_right` read MORE confident than the fit supports on
    #: exactly the comparisons where it should read less.
    is_rookie: bool = False


@dataclass(frozen=True)
class Swap:
    """One change worth making."""

    #: None when a starting slot was simply EMPTY, so nobody comes out.
    start: LineupRow | None
    #: None when a current starter leaves with no replacement -- a slot the solver cannot
    #: fill, which is what a D/ST neither source prices looks like.
    sit: LineupRow | None
    #: Points the LINEUP gains, which is not `start.points - sit.points` when the cascade
    #: moves a third player. See `build_swaps`.
    gain: float
    #: P(`start` outscores `sit` this week). 0.5 means a coin flip, and saying so is the point.
    p_right: float | None


def starting_slot(raw: object) -> RosterSlot | None:
    """The starting slot this `parse_rosters` `lineup_slot` names, or None if it is not one.

    None covers three cases that all mean the same thing downstream — bench, IR, and a value
    that is not a `RosterSlot` at all. `parse_rosters` maps an ESPN slot id it does not
    recognise to an empty string, so the third case should not arise from real data; it is
    handled rather than raised because **the alternative failure is silent and worse**.
    Reading an unplaceable player as a STARTER credits him to the current lineup, which
    understates the gain from fixing it — the same trap the My Team page documents, where an
    unknown slot inflated the starters-only total.

    `NON_STARTING_SLOTS` is the shared definition; the My Team page reads the same set.
    """
    value = str(raw or "")
    if not value:
        return None
    try:
        slot = RosterSlot(value)
    except ValueError:
        return None
    return None if slot in NON_STARTING_SLOTS else slot


def current_starter_ids(roster: pd.DataFrame) -> set[int]:
    """ESPN player ids currently occupying a starting slot."""
    return {
        player_id(player)
        for _, player in roster.iterrows()
        if starting_slot(player.get("lineup_slot")) is not None
    }


def optimal_starters(
    rows: Sequence[LineupRow], roster_slots: Mapping[RosterSlot, int]
) -> tuple[list[int], list[RosterSlot]]:
    """The best startable lineup: indices into `rows`, and the slot each one fills.

    Calls the greedy directly rather than `waivers.lineup_points`, which does the same thing
    over a `_LineupRow` — that type is private and shaped for the waiver path. Fourth caller,
    not a fourth copy.

    **The slots come from the solver, not from re-walking the taxonomy.** An earlier cut
    reconstructed them by zipping the slot order against the returned indices, which is wrong
    whenever a slot cannot be filled: the greedy skips such a slot, so the indices are dense
    and the slot order is not. On the first live run that printed a FLEX running back as the
    D/ST, because the defense had no weekly stat line and its slot went unfilled.
    """
    filled = choose_starters_with_slots(
        list(rows),
        roster_slots,
        value=lambda row: row.points,
        position=lambda row: row.position,
    )
    return [index for index, _ in filled], [slot for _, slot in filled]


def lineup_total(rows: Sequence[LineupRow], chosen: Iterable[int]) -> float:
    """Points scored by the players at `chosen`. Unstartable counts as nothing."""
    return sum(float(rows[i].points or 0.0) for i in chosen)


def p_right(
    start: LineupRow,
    sit: LineupRow,
    params: VarianceParams,
    *,
    n_sims: int,
    rng: np.random.Generator,
) -> float:
    """P(`start` outscores `sit` this week).

    **The resolvable uncertainty question.** Change in expected season wins is not: paired
    noise there is 0.062 wins against roughly 140 season points to a win, so a 3-point weekly
    swap is ~0.021 wins — a signal three times smaller than the error on it. This costs no
    Monte-Carlo season and answers what a manager actually wants to know: real edge, or coin
    flip?

    Draws from `sample_weekly_points` with `n_weeks=1` rather than hand-rolling a Gamma, so it
    keeps BOTH components — the lognormal season-mean multiplier (how sure are we of his
    talent) and the per-week Gamma (how much does he bounce). Dropping the first would make
    this read more confident than the evidence supports, which is the wrong direction for a
    number whose whole job is to say "that is a coin flip".

    `sample_weekly_points` takes SEASON means and divides by `SEASON_GAMES` internally; we hold
    weekly means, so they are multiplied back up on the way in.
    """
    if sit.points is None:
        return 1.0 if start.points is not None else 0.5
    if start.points is None:
        return 0.0
    draws = sample_weekly_points(
        params,
        np.array([start.position, sit.position], dtype=object),
        np.array([start.points * SEASON_GAMES, sit.points * SEASON_GAMES], dtype=np.float64),
        np.array([start.is_rookie, sit.is_rookie]),
        n_sims=n_sims,
        n_weeks=1,
        rng=rng,
    )
    return float(np.mean(draws[:, 0, 0] > draws[:, 0, 1]))


def build_swaps(
    rows: Sequence[LineupRow],
    roster_slots: Mapping[RosterSlot, int],
    *,
    params: VarianceParams | None,
    n_sims: int = 20_000,
    rng: np.random.Generator | None = None,
) -> list[Swap]:
    """Every change between the lineup currently set and the optimal one.

    **`gain` is the LINEUP gain, not the pairwise difference**, and the two are not the same
    number. A receiver beating the started WR by 0.2 pushes that WR into the FLEX and the FLEX
    RB out of the lineup entirely, for a gain of 1.2. Comparing a player against the man he
    appears to replace understates every move — the waiver work paid for that lesson once.

    It falls out of taking the set difference between two *solved* lineups rather than pairing
    players up first: the optimal total minus the current total is exactly the sum of the
    entering players' points minus the leaving players' points, so each pair's share is honest
    and the shares add up to the total. Which entering player is displayed against which
    leaving one is presentation only, and prefers a same-position match because that is how it
    reads.

    `params=None` skips `p_right` and leaves it None. The arithmetic above is exact and does
    not need the simulation.
    """
    chosen, _labels = optimal_starters(rows, roster_slots)
    optimal_ids = {rows[i].player_id for i in chosen}
    current = [row for row in rows if row.current_slot is not None]
    current_ids = {row.player_id for row in current}

    entering = sorted(
        (rows[i] for i in chosen if rows[i].player_id not in current_ids),
        key=lambda r: (-(r.points or 0.0), r.player_id),
    )
    leaving = sorted(
        (row for row in current if row.player_id not in optimal_ids),
        key=lambda r: (-(r.points or 0.0), r.player_id),
    )

    generator = rng if rng is not None else np.random.default_rng(0)
    swaps: list[Swap] = []
    for start in entering:
        # **The two lists need not be the same length, and an earlier cut assumed they were.**
        # `break`ing when `leaving` ran dry silently discarded every remaining recommendation,
        # so a roster with an EMPTY starting slot -- a waiver claim that landed on the bench,
        # a starter moved to IR leaving his slot open -- printed "your lineup is already
        # optimal" directly under an optimal total 12 points above the current one. The
        # unmatched cases are rows in their own right, not rows to drop.
        match = None
        if leaving:
            match = next((row for row in leaving if row.position == start.position), leaving[0])
            leaving.remove(match)
        swaps.append(
            Swap(
                start=start,
                sit=match,
                gain=float(start.points or 0.0) - float(match.points or 0.0 if match else 0.0),
                p_right=(
                    None
                    if params is None or match is None
                    else p_right(start, match, params, n_sims=n_sims, rng=generator)
                ),
            )
        )
    # Whoever is left leaves with nobody replacing him. Reported only when he was STARTABLE:
    # then a slot really is being vacated and that is a decision. An unstartable leftover is
    # not a change at all -- he is a slot the solver could not fill, he contributes 0.0 to
    # both totals, and `_notes` already names him and says why. Listing him as a "change worth
    # making" worth -0.0 points is noise dressed as a recommendation, which is the failure
    # this whole tool exists to avoid.
    #
    # This is NOT the dropping finding 1 was about. That discarded rows carrying real positive
    # gain; this drops rows carrying none, and the totals are unchanged either way.
    swaps.extend(
        Swap(start=None, sit=row, gain=-float(row.points), p_right=None)
        for row in leaving
        if row.points is not None
    )
    return swaps


@dataclass(frozen=True)
class StartSitRun:
    """Everything the report needs, with no rendering decisions taken yet."""

    team_name: str
    week: int
    #: Every roster player, priced. Bench and IR included — the answer is about both sides.
    rows: list[LineupRow]
    #: Indices into `rows` for the optimal lineup, in `choose_starters` fill order.
    starters: list[int]
    #: The slot each entry of `starters` fills. Positional, same length.
    slots: list[RosterSlot]
    #: The league's slot counts, carried so a presenter can number repeated slots (RB1/RB2)
    #: from what the league HAS rather than from what the greedy managed to fill.
    roster_slots: dict[RosterSlot, int]
    current_total: float
    optimal_total: float
    swaps: list[Swap]
    weight_espn: float
    notes: tuple[str, ...] = ()

    @property
    def gain(self) -> float:
        """Points the optimal lineup adds over the one currently set."""
        return self.optimal_total - self.current_total


def _notes(rows: Sequence[LineupRow], week: int) -> tuple[str, ...]:
    """What the reader needs to distrust, said out loud.

    The Sleeper coverage line is the important one. This tool's whole claim over the ESPN app
    is a second opinion; if the crosswalk cannot place most of a roster, the report has quietly
    degraded to ESPN alone and the reader deserves to know before he benches somebody on it.
    """
    out: list[str] = []
    priced = [row for row in rows if row.blend is not None]
    single = [row for row in priced if row.blend is not None and row.blend.sources != "both"]
    if single:
        # Naming them, and naming WHICH source, because "one source" is not one situation:
        # a Sleeper-only player is one ESPN has no line for this week, an ESPN-only one is
        # usually a crosswalk miss. Both mean no second opinion, which is the whole pitch.
        listed = ", ".join(
            f"{row.name} ({row.blend.sources} only)" for row in single if row.blend is not None
        )
        out.append(
            f"one source only, so no second opinion on: {listed}. "
            f"{len(priced) - len(single)}/{len(priced)} of the roster is priced by both."
        )
    unpriced = [row for row in rows if row.blend is None]
    if unpriced:
        names = ", ".join(sorted(row.name for row in unpriced))
        out.append(
            f"no week {week} projection from either source: {names}. Left out of the lineup "
            "solve entirely rather than priced at zero, so those slots are yours to set. "
            "K and D/ST land here by construction: neither source ships a stat line this "
            "layer can score for them."
        )
    return tuple(out)


def recommend_start_sit(
    roster: pd.DataFrame,
    espn_statlines: pd.DataFrame,
    sleeper: pd.DataFrame,
    id_map: pd.DataFrame,
    roster_slots: Mapping[RosterSlot, int],
    ruleset: Ruleset,
    *,
    team_name: str,
    week: int,
    weight_espn: float = 0.5,
    params: VarianceParams | None = None,
    rookies: frozenset[str] | None = None,
    n_sims: int = 20_000,
    rng: np.random.Generator | None = None,
) -> StartSitRun:
    """Already-fetched data in, a `StartSitRun` out. No I/O, no formatting.

    Same contract as `build_my_team`, and for the same reason: the seam between ingest and
    logic is where the web UI's defects lived, so the end-to-end tests drive this from real
    payload shapes through the real parsers rather than around them.
    """
    blended = blend_weekly_points(
        espn_statlines, sleeper, id_map, weight_espn=weight_espn, ruleset=ruleset
    )
    rows: list[LineupRow] = []
    for _, player in roster.iterrows():
        pid = player_id(player)
        status, _raw = parse_injury_status(player.get("injury_status"))
        blend = blended.get(str(pid))
        rows.append(
            LineupRow(
                player_id=pid,
                name=str(player.get("player") or pid),
                position=str(player.get("pos") or ""),
                status=status,
                current_slot=starting_slot(player.get("lineup_slot")),
                # `is_on_ir` reads the LINEUP SLOT; `startable_points` reads the injury
                # STATUS. Both are needed and they are different fields: ESPN routinely
                # reports an IR-slotted player as OUT (PUP/NFI) or, once designated to
                # return, as QUESTIONABLE -- and a QUESTIONABLE status alone would have him
                # priced at 0.86 and recommended into a starting slot ESPN will not accept.
                points=(
                    None
                    if is_on_ir(player)
                    else startable_points(blend.points if blend else None, status)
                ),
                blend=blend,
                is_rookie=bool(rookies and str(pid) in rookies),
            )
        )

    starters, slots = optimal_starters(rows, roster_slots)
    current = [row for row in rows if row.current_slot is not None]
    return StartSitRun(
        team_name=team_name,
        week=week,
        rows=rows,
        starters=starters,
        slots=slots,
        roster_slots=dict(roster_slots),
        current_total=lineup_total(current, range(len(current))),
        optimal_total=lineup_total(rows, starters),
        swaps=build_swaps(rows, roster_slots, params=params, n_sims=n_sims, rng=rng),
        weight_espn=weight_espn,
        notes=_notes(rows, week),
    )
