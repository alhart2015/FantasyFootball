"""Unit tests for `projections.draft.league_config.LeagueConfig`."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from projections.draft.league_config import LeagueConfig
from projections.schemas import RosterSlot, Ruleset


def _base_kwargs() -> dict[str, object]:
    return {
        "name": "test_league",
        "n_teams": 12,
        "budget": 200,
        "min_bid": 1,
        "roster_slots": {
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 3,
            RosterSlot.TE: 1,
            RosterSlot.FLEX: 1,
            RosterSlot.K: 1,
            RosterSlot.DST: 1,
            RosterSlot.BENCH: 7,
        },
        "ruleset": Ruleset.espn_ppr(),
    }


def test_roster_size_excludes_ir() -> None:
    kwargs = _base_kwargs()
    kwargs["roster_slots"] = {**kwargs["roster_slots"], RosterSlot.IR: 2}  # type: ignore[dict-item]
    cfg = LeagueConfig(**kwargs)  # type: ignore[arg-type]
    # IR (2) excluded; remaining = 1+2+3+1+1+1+1+7 = 17
    assert cfg.roster_size == 17


def test_roster_size_and_pool_and_budget_properties() -> None:
    cfg = LeagueConfig(**_base_kwargs())  # type: ignore[arg-type]
    # _base_kwargs roster slots: 1+2+3+1+1+1+1+7 = 17 drafted slots/team.
    assert cfg.roster_size == 17
    assert cfg.total_pool_size == 12 * 17
    assert cfg.total_budget == 12 * 200


def test_json_round_trip() -> None:
    cfg = LeagueConfig(**_base_kwargs())  # type: ignore[arg-type]
    blob = cfg.model_dump_json()
    restored = LeagueConfig.model_validate_json(blob)
    assert restored == cfg


def test_ruleset_string_preset_deserialization() -> None:
    kwargs = _base_kwargs()
    kwargs["ruleset"] = "espn_ppr"
    cfg = LeagueConfig(**kwargs)  # type: ignore[arg-type]
    assert cfg.ruleset == Ruleset.espn_ppr()


def test_ruleset_full_object_deserialization() -> None:
    kwargs = _base_kwargs()
    custom = Ruleset(name="CUSTOM", reception_pts=0.25)
    kwargs["ruleset"] = json.loads(custom.model_dump_json())
    cfg = LeagueConfig(**kwargs)  # type: ignore[arg-type]
    assert cfg.ruleset == custom


def test_rejects_n_teams_le_1() -> None:
    kwargs = _base_kwargs()
    kwargs["n_teams"] = 1
    with pytest.raises(ValidationError):
        LeagueConfig(**kwargs)  # type: ignore[arg-type]


def test_rejects_budget_le_0() -> None:
    kwargs = _base_kwargs()
    kwargs["budget"] = 0
    with pytest.raises(ValidationError):
        LeagueConfig(**kwargs)  # type: ignore[arg-type]


def test_rejects_empty_roster_slots() -> None:
    kwargs = _base_kwargs()
    kwargs["roster_slots"] = {}
    with pytest.raises(ValidationError):
        LeagueConfig(**kwargs)  # type: ignore[arg-type]


def test_rejects_min_bid_lt_1() -> None:
    kwargs = _base_kwargs()
    kwargs["min_bid"] = 0
    with pytest.raises(ValidationError):
        LeagueConfig(**kwargs)  # type: ignore[arg-type]


def test_frozen() -> None:
    cfg = LeagueConfig(**_base_kwargs())  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        cfg.n_teams = 14


def test_rejects_ir_only_roster_slots() -> None:
    """A config with only IR slots is rejected — roster_size would be 0."""
    kwargs = _base_kwargs()
    kwargs["roster_slots"] = {RosterSlot.IR: 1}
    with pytest.raises(ValidationError, match="at least one non-IR slot"):
        LeagueConfig(**kwargs)  # type: ignore[arg-type]


def test_rejects_unknown_ruleset_preset_string() -> None:
    """`field_validator` raises a clear error on an unrecognized preset name."""
    kwargs = _base_kwargs()
    kwargs["ruleset"] = "made_up_preset"
    with pytest.raises(ValidationError, match=r"Unknown ruleset preset 'made_up_preset'"):
        LeagueConfig(**kwargs)  # type: ignore[arg-type]
