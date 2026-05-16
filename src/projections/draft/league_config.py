"""LeagueConfig — pydantic model shared by VORP, auction-values, and snake-draft tooling.

Captures the user's league rules in one immutable, hashable object: team count,
auction budget, roster slot composition, and scoring ruleset. Constructed in code
or deserialized from JSON via `model_validate_json`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
