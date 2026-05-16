"""LeagueConfig — pydantic model shared by VORP, auction-values, and snake-draft tooling.

Captures the user's league rules in one immutable, hashable object: team count,
auction budget, roster slot composition, and scoring ruleset. Constructed in code
or deserialized from JSON via `model_validate_json`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from projections.schemas import RosterSlot, Ruleset

_RULESET_PRESETS: dict[str, Ruleset] = {
    "espn_ppr": Ruleset.espn_ppr(),
    "espn_half": Ruleset.espn_half(),
    "standard": Ruleset.standard(),
}


class LeagueConfig(BaseModel):
    """Frozen league configuration. Shared input for all Draft Hub tooling."""

    model_config = ConfigDict(frozen=True)

    name: str
    n_teams: int = Field(gt=1)
    budget: int = Field(gt=0, default=200)
    min_bid: int = Field(ge=1, default=1)
    roster_slots: dict[RosterSlot, int] = Field(min_length=1)
    ruleset: Ruleset

    @field_validator("ruleset", mode="before")
    @classmethod
    def _resolve_ruleset(cls, v: Any) -> Any:
        """Allow string preset names (`espn_ppr`, `espn_half`, `standard`) for ergonomics
        in config JSON. Pass through pydantic-shaped dict or `Ruleset` instances unchanged.
        """
        if isinstance(v, str):
            try:
                return _RULESET_PRESETS[v]
            except KeyError as exc:
                allowed = ", ".join(sorted(_RULESET_PRESETS))
                raise ValueError(
                    f"Unknown ruleset preset {v!r}; expected one of: {allowed}"
                ) from exc
        return v

    @model_validator(mode="after")
    def _require_at_least_one_drafted_slot(self) -> LeagueConfig:
        """`min_length=1` on roster_slots only checks the dict is non-empty.
        The downstream auction algorithm divides by `total_pool_size`, so we
        also need at least one slot that actually consumes a draft pick (i.e.,
        not all IR). Without this, a config of `{RosterSlot.IR: 1}` would pass
        field validation and cause a ZeroDivisionError in generate_auction_values.
        """
        if self.roster_size < 1:
            raise ValueError(
                "roster_slots must include at least one non-IR slot "
                "(IR slots don't count toward drafted positions)."
            )
        return self

    @property
    def roster_size(self) -> int:
        """Drafted slots per team (excludes IR, which is post-draft)."""
        return sum(count for slot, count in self.roster_slots.items() if slot != RosterSlot.IR)

    @property
    def total_pool_size(self) -> int:
        return self.n_teams * self.roster_size

    @property
    def total_budget(self) -> int:
        return self.n_teams * self.budget
