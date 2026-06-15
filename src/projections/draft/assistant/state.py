"""DraftState — the live draft as the engine sees it, loaded from a JSON file.

The state file is an ordered list of drafted gsis_ids plus my slot and a path to
the LeagueConfig; a pick's slot is derived from its (1-based) position via snake
order. My roster's positions are resolved through id_map (the universal position
source — the consensus VORP table can't supply a position for off-board picks).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from projections.draft.assistant.pick_timing import slot_for
from projections.draft.league_config import LeagueConfig
from projections.schemas import GsisId, Position, validate_gsis_id


@dataclass(frozen=True)
class DraftState:
    """An immutable snapshot of an in-progress draft."""

    my_slot: int
    n_teams: int
    rounds: int  # picks per team (== LeagueConfig.roster_size)
    picks: tuple[GsisId, ...]  # drafted gsis_ids, in pick order
    my_roster: tuple[Position, ...]  # positions of the picks I made

    @property
    def drafted_ids(self) -> frozenset[GsisId]:
        return frozenset(self.picks)

    @property
    def current_pick(self) -> int:
        return len(self.picks) + 1

    @property
    def my_pick_ids(self) -> tuple[GsisId, ...]:
        """The gsis_ids of *my* picks (snake-slot-derived), in pick order.

        Mirrors load_draft_state's roster derivation: pick #k is mine iff its
        snake slot equals my_slot. Parallel to my_roster (positions) but ids.
        """
        return tuple(
            gid
            for index, gid in enumerate(self.picks)
            if slot_for(index + 1, self.n_teams) == self.my_slot
        )


def build_draft_state(
    picks: Sequence[str],
    *,
    my_slot: int,
    league: LeagueConfig,
    id_map: pd.DataFrame,
) -> DraftState:
    """Build a `DraftState` from in-memory picks (the file-free half of load_draft_state).

    Raises ValueError on: my_slot out of range, a malformed/duplicate gsis_id, or
    one of *my* picks being absent from id_map (unknown position).
    """
    if not 1 <= my_slot <= league.n_teams:
        raise ValueError(f"my_slot must be in 1..{league.n_teams}; got {my_slot}")

    parsed = tuple(validate_gsis_id(str(p)) for p in picks)
    if len(set(parsed)) != len(parsed):
        raise ValueError("draft state has a duplicate pick (a player drafted twice)")

    pos_by_id = dict(zip(id_map["gsis_id"], id_map["position"], strict=False))
    my_roster: list[Position] = []
    for index, gid in enumerate(parsed):
        pick_number = index + 1
        if slot_for(pick_number, league.n_teams) != my_slot:
            continue
        if gid not in pos_by_id:
            raise ValueError(
                f"my pick {gid} (pick #{pick_number}) is absent from id_map; "
                "cannot resolve its position for roster accounting"
            )
        my_roster.append(Position(pos_by_id[gid]))

    return DraftState(
        my_slot=my_slot,
        n_teams=league.n_teams,
        rounds=league.roster_size,
        picks=parsed,
        my_roster=tuple(my_roster),
    )


def load_draft_state(state_path: Path, id_map: pd.DataFrame) -> tuple[DraftState, LeagueConfig]:
    """Parse a draft-state JSON file into a `DraftState` + its `LeagueConfig`.

    Raises ValueError on: my_slot out of range, a malformed/duplicate gsis_id, or
    one of *my* picks being absent from id_map (unknown position).
    """
    data = json.loads(state_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("draft-state JSON must be an object")
    missing = [k for k in ("league_config", "my_slot", "picks") if k not in data]
    if missing:
        raise ValueError(f"draft-state JSON missing required key(s): {', '.join(missing)}")
    if not isinstance(data["picks"], list):
        raise ValueError("draft-state JSON 'picks' must be a list")
    league = LeagueConfig.model_validate_json(Path(data["league_config"]).read_text())
    state = build_draft_state(
        data["picks"], my_slot=int(data["my_slot"]), league=league, id_map=id_map
    )
    return state, league
