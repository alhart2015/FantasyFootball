"""Model interface for position-specific projection models.

`Model` is a structural Protocol: any class implementing the listed methods
satisfies it without explicit inheritance. mypy enforces signatures at use
sites; we deliberately do NOT mark it @runtime_checkable because nothing in
the codebase needs isinstance() against Model (cf. Distribution which does).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import pandas as pd

from projections.schemas import Position, Ruleset


class Model(Protocol):
    """Position-specific projection model. Plugs in at the fit/predict seam.

    Implementations:
        - BaselineModel (Plan 3a, this plan): per-stat Ridge regressions with
          parametric residual variance.
        - (future) GBMModel (Plan 5): LightGBM with quantile regression.
        - (future) EnsembleModel: stack of A and C.
    """

    @property
    def position(self) -> Position: ...

    @property
    def model_id(self) -> str:
        """Stable identifier of the form
        ``"<class>:<position>:<8-char-code-hash>:<train-start>-<train-end>"``.

        Persisted into every projection row produced by predict_distribution
        so we can always trace which model produced which projection.

        Implementations may raise ``RuntimeError`` if accessed on an unfitted
        instance -- the model_id depends on training-time state.
        """
        ...

    def fit(self, features: pd.DataFrame, weekly_stats: pd.DataFrame) -> None:
        """Train the model. Inner-joins (gsis_id, season, week) to align
        feature inputs with truth. ``features`` must validate against the
        position's *FeaturesSchema; ``weekly_stats`` against WeeklyStatsSchema.
        """
        ...

    def predict_distribution(self, features: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame:
        """Predict per-player-week fantasy-points distributions under
        ``ruleset``. Returns a DataFrame validated against
        ``ProjectionWeeklySchema``. Re-scoring under a different ruleset is a
        second call with the same features — no retraining."""
        ...

    def save(self, path: Path) -> None:
        """Serialize to disk via joblib."""
        ...

    @classmethod
    def load(cls, path: Path) -> Model:
        """Deserialize from disk. Class methods on Protocols are unusual;
        BaselineModel implements this as a regular @classmethod and structural
        matching covers the contract."""
        ...


def compute_code_hash(paths: Iterable[Path]) -> str:
    """SHA-256 (first 8 hex chars) of the concatenated content of ``paths``.

    Used as the ``code_hash`` component of every model_id so we can detect when
    a model artifact is stale relative to the current source.

    Order-independent: paths are sorted by their string representation before
    hashing so callers don't have to maintain a canonical order.
    """
    sorted_paths = sorted(paths, key=str)
    hasher = hashlib.sha256()
    for path in sorted_paths:
        hasher.update(path.read_bytes())
    return hasher.hexdigest()[:8]
