"""Trajectory feature family — career-arc / role / volume-trend signals.

Probe-only at this stage: the override produced by build_trajectory_overrides
is consumed by scripts/probe_feature_signal.py via the standard --override
mechanism. Schema integration into per-position FeaturesSchemas is deferred
to a SIGNAL-greenlit follow-up.

Each compute_* function returns every (gsis_id, season[, week]) combo with
the feature value. The assembler merges all per-week feature frames onto
the player-team-week index in one pass.

Spec: docs/superpowers/specs/2026-05-03-trajectory-feature-family-probe-design.md.
"""

# pd and Position are referenced by compute_* fns added in subsequent tasks
# (Tasks 5-12); imported here so each task is a pure addition.
from __future__ import annotations

import re
from typing import Final

import pandas as pd  # noqa: F401  # used in subsequent tasks (compute_* fns)

from projections.schemas import (  # noqa: F401  # Position used in subsequent tasks
    GSIS_ID_PATTERN,
    Position,
)

# DraftLookup maps gsis_id -> (draft_year, draft_age). draft_age may be NaN
# (drafted-but-missing-age, rare). Missing key: UDFA / pre-coverage; falls
# back to inferred draft year from earliest weekly_stats appearance.
DraftLookup = dict[str, tuple[int, float]]

_GSIS_RE: Final[re.Pattern[str]] = re.compile(rf"^{GSIS_ID_PATTERN}$")
_AGE_OFFSET_FALLBACK: Final[float] = 22.0  # mean entry age for inferred path
