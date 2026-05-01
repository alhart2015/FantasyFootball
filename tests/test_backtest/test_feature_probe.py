# tests/test_backtest/test_feature_probe.py
"""Feature signal probe — tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from projections.backtest.adoption_gate import BootstrapDelta, PositionVerdict
from projections.backtest.feature_probe import (
    PerStatVerdict,
    ProbeReport,
    _build_factory_with_columns,
    _coerce_bools,
    _loosened_features_schema,
    _verdict_for_per_stat,
    family_verdict_from_reports,
    phase1_should_fire_phase2,
    probe_composite,
    probe_per_stat,
)
from projections.models import POSITION_DISPATCH, BaselineModel
from projections.schemas import Position, Ruleset, Stat


def test_per_stat_verdict_is_frozen_dataclass() -> None:
    psv = PerStatVerdict(
        position=Position.QB,
        stat=Stat.PASSING_YARDS,
        year_or_pooled=2024,
        n_paired=670,
        rmse_delta=BootstrapDelta(
            point=-0.42, lo_95=-0.71, hi_95=-0.13, n_paired_rows=670, n_bootstrap=1000
        ),
        r_squared_delta=0.0023,
        verdict="SIGNAL",
    )
    assert psv.position is Position.QB
    assert psv.year_or_pooled == 2024
    assert psv.verdict == "SIGNAL"
    # FrozenInstanceError subclasses AttributeError; accept either to keep the
    # test robust across Python's dataclass internals.
    with pytest.raises(Exception):  # noqa: B017
        psv.verdict = "NULL"  # type: ignore[misc]


def test_per_stat_verdict_year_or_pooled_accepts_pooled_literal() -> None:
    psv = PerStatVerdict(
        position=Position.QB,
        stat=Stat.PASSING_YARDS,
        year_or_pooled="pooled",
        n_paired=2676,
        rmse_delta=BootstrapDelta(
            point=0.0, lo_95=-0.1, hi_95=0.1, n_paired_rows=2676, n_bootstrap=1000
        ),
        r_squared_delta=0.0,
        verdict="NULL",
    )
    assert psv.year_or_pooled == "pooled"


def test_probe_report_phase2_optional() -> None:
    psv = PerStatVerdict(
        position=Position.QB,
        stat=Stat.PASSING_YARDS,
        year_or_pooled="pooled",
        n_paired=2676,
        rmse_delta=BootstrapDelta(
            point=0.0, lo_95=-0.1, hi_95=0.1, n_paired_rows=2676, n_bootstrap=1000
        ),
        r_squared_delta=0.0,
        verdict="NULL",
    )
    report = ProbeReport(
        candidate_name="test",
        model_class="baseline",
        baseline_features_path="data/features",
        override_paths=("data/features_probe/x.parquet",),
        drop_columns=(),
        phase1=[psv],
        phase2=None,
    )
    assert report.phase2 is None
    assert report.candidate_name == "test"
    assert len(report.phase1) == 1


def test_phase1_should_fire_phase2_truth_table() -> None:
    bd_null = BootstrapDelta(point=0.0, lo_95=-0.1, hi_95=0.1, n_paired_rows=2676, n_bootstrap=1000)
    bd_signal = BootstrapDelta(
        point=-0.5, lo_95=-0.9, hi_95=-0.1, n_paired_rows=2676, n_bootstrap=1000
    )
    bd_regression = BootstrapDelta(
        point=0.5, lo_95=0.1, hi_95=0.9, n_paired_rows=2676, n_bootstrap=1000
    )

    def _verdict(label: str, bd: BootstrapDelta) -> PerStatVerdict:
        return PerStatVerdict(
            position=Position.QB,
            stat=Stat.PASSING_YARDS,
            year_or_pooled="pooled",
            n_paired=bd.n_paired_rows,
            rmse_delta=bd,
            r_squared_delta=0.0,
            verdict=label,  # type: ignore[arg-type]
        )

    assert phase1_should_fire_phase2([]) is False
    assert phase1_should_fire_phase2([_verdict("NULL", bd_null)]) is False
    assert phase1_should_fire_phase2([_verdict("REGRESSION", bd_regression)]) is False
    assert (
        phase1_should_fire_phase2(
            [_verdict("NULL", bd_null), _verdict("REGRESSION", bd_regression)]
        )
        is False
    )
    assert phase1_should_fire_phase2([_verdict("SIGNAL", bd_signal)]) is True
    assert (
        phase1_should_fire_phase2([_verdict("NULL", bd_null), _verdict("SIGNAL", bd_signal)])
        is True
    )

    # NEW behavior: per-year SIGNAL cells do NOT fire Phase 2 — only pooled cells do.
    per_year_signal = PerStatVerdict(
        position=Position.QB,
        stat=Stat.PASSING_YARDS,
        year_or_pooled=2024,  # NOT "pooled"
        n_paired=670,
        rmse_delta=bd_signal,
        r_squared_delta=0.0,
        verdict="SIGNAL",
    )
    pooled_signal = PerStatVerdict(
        position=Position.QB,
        stat=Stat.PASSING_YARDS,
        year_or_pooled="pooled",
        n_paired=2676,
        rmse_delta=bd_signal,
        r_squared_delta=0.0,
        verdict="SIGNAL",
    )
    # Per-year SIGNAL alone should NOT fire phase 2.
    assert phase1_should_fire_phase2([per_year_signal]) is False
    # Pooled SIGNAL DOES fire phase 2.
    assert phase1_should_fire_phase2([pooled_signal]) is True
    # Mixed: per-year SIGNAL + pooled NULL → no fire.
    pooled_null = PerStatVerdict(
        position=Position.QB,
        stat=Stat.PASSING_YARDS,
        year_or_pooled="pooled",
        n_paired=2676,
        rmse_delta=bd_null,
        r_squared_delta=0.0,
        verdict="NULL",
    )
    assert phase1_should_fire_phase2([per_year_signal, pooled_null]) is False


def test_probe_per_stat_signal_on_orthogonal_signal_column(
    probe_synthetic_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """A candidate column orthogonal to baseline features but driving the
    target should produce SIGNAL on the relevant stat."""
    features, weekly_stats = probe_synthetic_dataset
    verdicts = probe_per_stat(
        position=Position.QB,
        features_baseline_cols=("base_x1", "base_x2", "base_x3"),
        features_candidate_cols=("base_x1", "base_x2", "base_x3", "cand_signal"),
        features=features,
        weekly_stats=weekly_stats,
        target_stats=(Stat.PASSING_YARDS,),
        holdout_years=(2022,),
    )
    # 1 stat x (1 year + 1 pooled) = 2 verdicts.
    assert len(verdicts) == 2
    pooled = next(v for v in verdicts if v.year_or_pooled == "pooled")
    assert pooled.stat is Stat.PASSING_YARDS
    assert pooled.verdict == "SIGNAL", (
        f"expected SIGNAL on pooled passing_yards but got {pooled.verdict}; "
        f"rmse_delta={pooled.rmse_delta}"
    )
    # The signal column carries the bulk of the target; delta-R^2 should be substantially positive.
    assert pooled.r_squared_delta > 0.05, (
        f"expected r_squared_delta > 0.05 on a strong signal column, got {pooled.r_squared_delta}"
    )


def test_probe_per_stat_null_on_pure_noise_column(
    probe_synthetic_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """A candidate column with no relationship to the target should produce NULL."""
    features, weekly_stats = probe_synthetic_dataset
    verdicts = probe_per_stat(
        position=Position.QB,
        features_baseline_cols=("base_x1", "base_x2", "base_x3"),
        features_candidate_cols=("base_x1", "base_x2", "base_x3", "cand_null"),
        features=features,
        weekly_stats=weekly_stats,
        target_stats=(Stat.PASSING_YARDS,),
        holdout_years=(2022,),
    )
    pooled = next(v for v in verdicts if v.year_or_pooled == "pooled")
    assert pooled.verdict == "NULL", (
        f"expected NULL on pure-noise column, got {pooled.verdict}; rmse_delta={pooled.rmse_delta}"
    )


def test_probe_per_stat_null_on_redundant_column(
    probe_synthetic_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """A candidate column collinear with an existing baseline column should
    produce NULL -- Ridge shrinks one of the correlated pair, so the candidate
    adds no marginal CV-RMSE benefit."""
    features, weekly_stats = probe_synthetic_dataset
    verdicts = probe_per_stat(
        position=Position.QB,
        features_baseline_cols=("base_x1", "base_x2", "base_x3"),
        features_candidate_cols=("base_x1", "base_x2", "base_x3", "cand_redundant"),
        features=features,
        weekly_stats=weekly_stats,
        target_stats=(Stat.PASSING_YARDS,),
        holdout_years=(2022,),
    )
    pooled = next(v for v in verdicts if v.year_or_pooled == "pooled")
    assert pooled.verdict == "NULL", (
        f"expected NULL on collinear-with-baseline column, got {pooled.verdict}"
    )


def test_probe_per_stat_emits_per_year_and_pooled_rows(
    probe_synthetic_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    features, weekly_stats = probe_synthetic_dataset
    verdicts = probe_per_stat(
        position=Position.QB,
        features_baseline_cols=("base_x1", "base_x2", "base_x3"),
        features_candidate_cols=("base_x1", "base_x2", "base_x3", "cand_signal"),
        features=features,
        weekly_stats=weekly_stats,
        target_stats=(Stat.PASSING_YARDS, Stat.PASSING_TDS),
        holdout_years=(2021, 2022),
    )
    # 2 stats x (2 years + 1 pooled) = 6 rows.
    assert len(verdicts) == 6
    years = sorted({v.year_or_pooled for v in verdicts if v.year_or_pooled != "pooled"})
    assert years == [2021, 2022]
    pooled_rows = [v for v in verdicts if v.year_or_pooled == "pooled"]
    assert len(pooled_rows) == 2
    assert {v.stat for v in pooled_rows} == {Stat.PASSING_YARDS, Stat.PASSING_TDS}


def test_probe_per_stat_is_deterministic(
    probe_synthetic_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    features, weekly_stats = probe_synthetic_dataset
    kwargs = dict(
        position=Position.QB,
        features_baseline_cols=("base_x1", "base_x2", "base_x3"),
        features_candidate_cols=("base_x1", "base_x2", "base_x3", "cand_signal"),
        features=features,
        weekly_stats=weekly_stats,
        target_stats=(Stat.PASSING_YARDS,),
        holdout_years=(2022,),
    )
    a = probe_per_stat(**kwargs)
    b = probe_per_stat(**kwargs)
    assert len(a) == len(b)
    for va, vb in zip(a, b, strict=True):
        assert va.rmse_delta.point == vb.rmse_delta.point
        assert va.rmse_delta.lo_95 == vb.rmse_delta.lo_95
        assert va.rmse_delta.hi_95 == vb.rmse_delta.hi_95
        assert va.verdict == vb.verdict


def test_verdict_for_per_stat_nan_bootstrap_degrades_to_null() -> None:
    """A degenerate fit (NaN bootstrap statistics) must NOT bubble through as
    SIGNAL or REGRESSION — the conservative fallback is NULL, matching
    adoption_gate.verdict_for_position's NaN handling."""
    nan_bd = BootstrapDelta(
        point=float("nan"),
        lo_95=float("nan"),
        hi_95=float("nan"),
        n_paired_rows=500,
        n_bootstrap=1000,
    )
    assert _verdict_for_per_stat(nan_bd) == "NULL"

    partial_nan = BootstrapDelta(
        point=-0.5,
        lo_95=-0.9,
        hi_95=float("nan"),  # NaN on hi_95 alone
        n_paired_rows=500,
        n_bootstrap=1000,
    )
    assert _verdict_for_per_stat(partial_nan) == "NULL"


def test_verdict_for_per_stat_signal_regression_null_boundaries() -> None:
    """Boundary cases for the SIGNAL/REGRESSION/NULL rule."""
    # SIGNAL: hi_95 strictly below 0
    signal_bd = BootstrapDelta(
        point=-0.5, lo_95=-0.9, hi_95=-0.001, n_paired_rows=500, n_bootstrap=1000
    )
    assert _verdict_for_per_stat(signal_bd) == "SIGNAL"

    # REGRESSION: lo_95 strictly above 0
    regression_bd = BootstrapDelta(
        point=0.5, lo_95=0.001, hi_95=0.9, n_paired_rows=500, n_bootstrap=1000
    )
    assert _verdict_for_per_stat(regression_bd) == "REGRESSION"

    # NULL at exact-zero boundary (hi_95 == 0 doesn't satisfy strict-below)
    boundary_bd = BootstrapDelta(
        point=0.0, lo_95=-0.5, hi_95=0.0, n_paired_rows=500, n_bootstrap=1000
    )
    assert _verdict_for_per_stat(boundary_bd) == "NULL"

    # NEW behavior: effect-size floor (default 0.05). Below-floor effects → NULL
    # even when statistically significant.
    noise_floor_signal_bd = BootstrapDelta(
        point=-0.001,  # tiny effect, well below 0.05 floor
        lo_95=-0.0018,
        hi_95=-0.0001,  # CI strictly below 0 — statistically significant
        n_paired_rows=3723,  # large sample
        n_bootstrap=1000,
    )
    # Default floor (0.05): NULL despite CI<0
    assert _verdict_for_per_stat(noise_floor_signal_bd) == "NULL"
    # Explicit lower floor: SIGNAL recovers
    assert _verdict_for_per_stat(noise_floor_signal_bd, effect_size_floor=0.0) == "SIGNAL"
    # Effect just at the floor: SIGNAL (>= condition)
    just_at_floor_bd = BootstrapDelta(
        point=-0.05, lo_95=-0.10, hi_95=-0.001, n_paired_rows=2000, n_bootstrap=1000
    )
    assert _verdict_for_per_stat(just_at_floor_bd) == "SIGNAL"
    # REGRESSION at noise floor: also NULL
    noise_floor_regression_bd = BootstrapDelta(
        point=0.001, lo_95=0.0001, hi_95=0.0018, n_paired_rows=3723, n_bootstrap=1000
    )
    assert _verdict_for_per_stat(noise_floor_regression_bd) == "NULL"


def test_coerce_bools_converts_to_int8() -> None:
    """_coerce_bools must convert bool columns to int8 and leave other dtypes alone."""
    df = pd.DataFrame(
        {
            "bool_col": pd.array([True, False, True], dtype=bool),
            "float_col": [1.0, 2.0, 3.0],
            "int_col": [10, 20, 30],
        }
    )
    out = _coerce_bools(df)
    assert out["bool_col"].dtype == np.int8
    assert out["float_col"].dtype == df["float_col"].dtype  # unchanged
    assert out["int_col"].dtype == df["int_col"].dtype  # unchanged
    # Original frame unmodified (defensive copy).
    assert df["bool_col"].dtype == bool


def test_loosened_features_schema_flips_strict_mode() -> None:
    """The loosened schema is a subclass of the base with Config.strict=False.
    Production schemas use strict='filter'; the loosened version disables
    column-filtering so undeclared candidate columns survive validation."""
    base_schema = POSITION_DISPATCH[Position.QB].feature_schema
    loose = _loosened_features_schema(base_schema)
    assert loose.Config.strict is False
    assert base_schema.Config.strict == "filter"  # production schema unchanged


def test_loosened_features_schema_keeps_candidate_column_through_validate(
    baseline_features_qb: pd.DataFrame,
) -> None:
    """End-to-end: add an undeclared column to a QB-features frame; validate
    against the loosened schema; assert the column survives. Validate the
    same frame against the production schema; assert the column is dropped."""
    base_schema = POSITION_DISPATCH[Position.QB].feature_schema
    loose = _loosened_features_schema(base_schema)
    augmented = baseline_features_qb.copy()
    augmented["candidate_extra"] = 0.5
    loose_validated = loose.validate(augmented)
    assert "candidate_extra" in loose_validated.columns
    prod_validated = base_schema.validate(augmented)
    assert "candidate_extra" not in prod_validated.columns


def test_loosened_features_schema_accepts_frame_missing_declared_column(
    baseline_features_qb: pd.DataFrame,
) -> None:
    """Swap-mode flow: --drop removes a column from baseline, then the model
    is asked to validate the post-drop frame. The loosened schema must accept
    a frame missing one or more declared columns (so the probe's Phase 2 can
    actually run in swap mode). The production schema rejects the same frame."""
    base_schema = POSITION_DISPATCH[Position.QB].feature_schema
    loose = _loosened_features_schema(base_schema)
    dropped = baseline_features_qb.drop(columns=["opp_allowed_qb_fppg_l4"])
    # Loose accepts the missing-column frame.
    loose_validated = loose.validate(dropped)
    assert "opp_allowed_qb_fppg_l4" not in loose_validated.columns
    # Production rejects it.
    import pandera.errors

    with pytest.raises(pandera.errors.SchemaError):
        base_schema.validate(dropped)


def test_build_factory_with_columns_overrides_feature_columns_and_schema() -> None:
    """The factory returned by _build_factory_with_columns produces an
    unfitted model whose feature_columns and feature_schema are the candidate
    ones; the production factory is unmodified across calls."""
    base_schema = POSITION_DISPATCH[Position.QB].feature_schema
    loose = _loosened_features_schema(base_schema)
    candidate_cols = ("base_x1", "base_x2", "extra_signal_col")
    factory = _build_factory_with_columns(
        position=Position.QB,
        model_class="baseline",
        columns=candidate_cols,
        feature_schema=loose,
    )
    model = factory()
    # The factory's return type is the Model Protocol; narrow to the concrete
    # BaselineModel before reading dataclass fields the Protocol doesn't expose.
    assert isinstance(model, BaselineModel)
    assert model.feature_columns == candidate_cols
    assert model.feature_schema is loose
    assert model.position is Position.QB

    # Sanity: the production factory is NOT mutated by the override; a fresh
    # build via the production dispatch returns the production columns.
    # Factories are typed Callable[[], Model] (Protocol), but every concrete
    # model exposes feature_columns/feature_schema — cast for the assertion.
    fresh = cast(BaselineModel, POSITION_DISPATCH[Position.QB].factories["baseline"]())
    assert fresh.feature_columns != candidate_cols
    assert fresh.feature_schema is base_schema


def test_build_factory_with_columns_returns_fresh_instances() -> None:
    """Each call to the returned factory must produce a fresh model (no shared
    mutable state between calls)."""
    base_schema = POSITION_DISPATCH[Position.QB].feature_schema
    loose = _loosened_features_schema(base_schema)
    factory = _build_factory_with_columns(
        position=Position.QB,
        model_class="baseline",
        columns=("base_x1",),
        feature_schema=loose,
    )
    a = factory()
    b = factory()
    assert a is not b
    # Narrow to the concrete BaselineModel for attribute access — the factory
    # return type is the Model Protocol.
    assert isinstance(a, BaselineModel)
    assert isinstance(b, BaselineModel)
    # Mutating one's feature_columns should not affect the other.
    a.feature_columns = ("mutated",)
    assert b.feature_columns == ("base_x1",)

    # Symmetric: feature_schema mutation on `a` must not bleed into `b`.
    class _OtherSchema:
        pass

    a.feature_schema = _OtherSchema  # type: ignore[assignment]
    assert b.feature_schema is loose


def test_build_factory_with_columns_lightgbm_nb_overrides_config() -> None:
    """For LightGBMNbModel (and its parents), the override must update the
    inner _LightGBMConfig — NOT just set stray attributes on the instance,
    which the model methods would never read."""
    from projections.models.lightgbm_nb import LightGBMNbModel

    base_schema = POSITION_DISPATCH[Position.QB].feature_schema
    loose = _loosened_features_schema(base_schema)
    candidate_cols = ("base_x1", "base_x2", "extra_signal_col")
    factory = _build_factory_with_columns(
        position=Position.QB,
        model_class="lightgbm-nb",
        columns=candidate_cols,
        feature_schema=loose,
    )
    model = factory()
    assert isinstance(model, LightGBMNbModel)
    # The override must reach the inner config — that's where the model reads from.
    assert model._config.feature_columns == candidate_cols
    assert model._config.feature_schema is loose
    # Production factory must be unmodified.
    fresh = POSITION_DISPATCH[Position.QB].factories["lightgbm-nb"]()
    assert isinstance(fresh, LightGBMNbModel)
    assert fresh._config.feature_columns != candidate_cols
    assert fresh._config.feature_schema is base_schema


def test_build_factory_with_columns_unknown_model_class_raises() -> None:
    """Hypothetical future model class that the helper doesn't know how to
    override must raise NotImplementedError, not silently produce a model
    with the production feature set."""
    # Construct a fake factory directly and run the same closure path.
    # We can't easily inject an unknown model_class string into POSITION_DISPATCH,
    # so we exercise this by invoking the helper with a custom monkey-patched
    # factory that returns an unfitted instance of an unrelated class.

    class _UnknownModel:
        position = Position.QB

    base_schema = POSITION_DISPATCH[Position.QB].feature_schema
    loose = _loosened_features_schema(base_schema)
    # POSITION_DISPATCH[Position.QB].factories is typed as Mapping but is
    # backed by a mutable dict in the production registry. Cast to the
    # concrete type so we can add and remove a fake entry for this test only.
    qb_factories = cast(
        "dict[str, Callable[[], object]]",
        POSITION_DISPATCH[Position.QB].factories,
    )
    qb_factories["_test_unknown"] = _UnknownModel
    try:
        factory = _build_factory_with_columns(
            position=Position.QB,
            model_class="_test_unknown",
            columns=("x",),
            feature_schema=loose,
        )
        with pytest.raises(NotImplementedError, match="_UnknownModel"):
            factory()
    finally:
        del qb_factories["_test_unknown"]


@dataclass
class _MockModel:
    """Test double — fit() records the seasons it saw; predict_distribution()
    returns a deterministic per-row predicted mean derived from the input
    features.

    Implements all 6 members of the Model Protocol (position, model_id, fit,
    predict_distribution, save, load) so it can be passed via
    Callable[[], Model] without casts. Stubs raise NotImplementedError for the
    members the probe doesn't exercise (save, load, model_id property)."""

    feature_columns: tuple[str, ...]
    train_seasons_seen: tuple[int, ...] = ()
    multiplier: float = 1.0  # baseline=1.0; candidate=0.5 -> smaller residuals -> SIGNAL

    @property
    def position(self) -> Position:
        return Position.QB

    @property
    def model_id(self) -> str:
        raise NotImplementedError("test double — model_id not exercised by probe_composite")

    def fit(self, features: pd.DataFrame, weekly_stats: pd.DataFrame) -> None:
        self.train_seasons_seen = tuple(sorted(features["season"].unique().tolist()))

    def predict_distribution(self, features: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame:
        # Predict mean = multiplier * sum(feature_columns); other ProjectionWeeklySchema
        # fields are zero/dummy. Probe code reads only `gsis_id`, `season`, `week`, `mean`.
        out = features[["gsis_id", "season", "week"]].copy()
        out["mean"] = self.multiplier * features[list(self.feature_columns)].sum(axis=1).to_numpy()
        out["p10"] = out["mean"] - 1.0
        out["p50"] = out["mean"]
        out["p90"] = out["mean"] + 1.0
        out["position"] = "QB"
        out["team"] = "KC"
        out["opponent"] = "BUF"
        return out

    def save(self, path: Path) -> None:
        raise NotImplementedError("test double — save not exercised by probe_composite")

    @classmethod
    def load(cls, path: Path) -> _MockModel:
        raise NotImplementedError("test double — load not exercised by probe_composite")


def test_probe_composite_walk_forward_order(
    probe_synthetic_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Each year's training set is exactly [min_season, year-1] and prediction
    is on the year itself."""
    features, weekly_stats = probe_synthetic_dataset
    # Build fpts truth: composite is just the target stat for the mock test.
    weekly_stats = weekly_stats.copy()
    weekly_stats["fpts"] = weekly_stats["passing_yards"]

    seen_per_year_baseline: list[tuple[int, ...]] = []
    seen_per_year_candidate: list[tuple[int, ...]] = []

    def factory_baseline() -> _MockModel:
        m = _MockModel(feature_columns=("base_x1", "base_x2", "base_x3"))
        # Wrap fit to capture the seen seasons after each call.
        original_fit = m.fit

        def wrapped(f: pd.DataFrame, w: pd.DataFrame) -> None:
            original_fit(f, w)
            seen_per_year_baseline.append(m.train_seasons_seen)

        # Deliberate test-double pattern: replace the bound `fit` method on
        # this dataclass instance with a wrapper that records call args.
        # mypy's [method-assign] doesn't cover dataclass instance attribute
        # rebinding here — narrow to the actual [assignment] code.
        m.fit = wrapped  # type: ignore[assignment]
        return m

    def factory_candidate() -> _MockModel:
        m = _MockModel(
            feature_columns=("base_x1", "base_x2", "base_x3", "cand_signal"),
            multiplier=0.5,  # smaller residuals than baseline since cand_signal is in y
        )
        original_fit = m.fit

        def wrapped(f: pd.DataFrame, w: pd.DataFrame) -> None:
            original_fit(f, w)
            seen_per_year_candidate.append(m.train_seasons_seen)

        # Same test-double pattern as factory_baseline above.
        m.fit = wrapped  # type: ignore[assignment]
        return m

    verdict = probe_composite(
        position=Position.QB,
        factory_baseline=factory_baseline,
        factory_candidate=factory_candidate,
        features_baseline=features,
        features_candidate=features,
        weekly_stats=weekly_stats,
        composite_truth_column="fpts",
        holdout_years=(2021, 2022),
        ruleset=Ruleset(),  # ignored by mock
    )
    # Walk-forward: 2021 trains on 2018-2020; 2022 trains on 2018-2021.
    assert seen_per_year_baseline == [(2018, 2019, 2020), (2018, 2019, 2020, 2021)]
    assert seen_per_year_candidate == [(2018, 2019, 2020), (2018, 2019, 2020, 2021)]
    # Sanity check that the verdict object is well-formed (the next test
    # asserts the full structure).
    assert verdict.position is Position.QB


def test_probe_composite_returns_position_verdict_with_per_year_breakdown(
    probe_synthetic_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    features, weekly_stats = probe_synthetic_dataset
    weekly_stats = weekly_stats.copy()
    weekly_stats["fpts"] = weekly_stats["passing_yards"]

    def factory_baseline() -> _MockModel:
        return _MockModel(feature_columns=("base_x1", "base_x2", "base_x3"))

    def factory_candidate() -> _MockModel:
        return _MockModel(
            feature_columns=("base_x1", "base_x2", "base_x3", "cand_signal"),
            multiplier=0.5,
        )

    verdict = probe_composite(
        position=Position.QB,
        factory_baseline=factory_baseline,
        factory_candidate=factory_candidate,
        features_baseline=features,
        features_candidate=features,
        weekly_stats=weekly_stats,
        composite_truth_column="fpts",
        holdout_years=(2021, 2022),
        ruleset=Ruleset(),
    )
    assert verdict.position is Position.QB
    assert verdict.incumbent_class == "_baseline_features"
    assert verdict.candidate_class == "_candidate_features"
    assert verdict.verdict in {"ADOPT", "MARGINAL", "DO_NOT_ADOPT"}
    assert isinstance(verdict.per_year_breakdown, pd.DataFrame)
    assert len(verdict.per_year_breakdown) == 2
    assert set(verdict.per_year_breakdown["year"]) == {2021, 2022}


def _stub_pooled_psv(verdict: str) -> PerStatVerdict:
    return PerStatVerdict(
        position=Position.QB,
        stat=Stat.PASSING_YARDS,
        year_or_pooled="pooled",
        n_paired=2676,
        rmse_delta=BootstrapDelta(
            point=-0.5 if verdict == "SIGNAL" else 0.0,
            lo_95=-0.9 if verdict == "SIGNAL" else -0.1,
            hi_95=-0.1 if verdict == "SIGNAL" else 0.1,
            n_paired_rows=2676,
            n_bootstrap=1000,
        ),
        r_squared_delta=0.0,
        verdict=verdict,  # type: ignore[arg-type]
    )


def _stub_phase2_verdict(verdict: str) -> PositionVerdict:
    """Minimal PositionVerdict with the right `verdict` field; numbers
    don't matter for the family-verdict helper."""
    bd_zero = BootstrapDelta(point=0.0, lo_95=-0.1, hi_95=0.1, n_paired_rows=100, n_bootstrap=1000)
    return PositionVerdict(
        position=Position.QB,
        incumbent_class="_baseline_features",
        candidate_class="_candidate_features",
        rmse_delta=bd_zero,
        spearman_delta=bd_zero,
        verdict=verdict,  # type: ignore[arg-type]
        reason="stub",
        per_year_breakdown=pd.DataFrame(),
    )


def _stub_report(*, phase1_verdict: str, phase2_verdict: str | None) -> ProbeReport:
    """Build a one-row Phase 1 report; Phase 2 either present or skipped."""
    phase2 = [_stub_phase2_verdict(phase2_verdict)] if phase2_verdict is not None else None
    return ProbeReport(
        candidate_name="stub",
        model_class="baseline",
        baseline_features_path="data/features",
        override_paths=("data/features_probe/x.parquet",),
        drop_columns=(),
        phase1=[_stub_pooled_psv(phase1_verdict)],
        phase2=phase2,
        phase2_skip_reason=None if phase2 is not None else "no_signal",
    )


def test_family_verdict_signal_via_phase1() -> None:
    """Pooled Phase 1 SIGNAL on any report flips family to SIGNAL."""
    reports = [
        _stub_report(phase1_verdict="NULL", phase2_verdict=None),
        _stub_report(phase1_verdict="SIGNAL", phase2_verdict="DO_NOT_ADOPT"),
    ]
    assert family_verdict_from_reports(reports) == "SIGNAL"


def test_family_verdict_signal_via_phase2() -> None:
    """Phase 2 ADOPT or MARGINAL anywhere flips family to SIGNAL even if all
    Phase 1 cells are NULL."""
    reports = [
        _stub_report(phase1_verdict="NULL", phase2_verdict=None),
        _stub_report(phase1_verdict="NULL", phase2_verdict="MARGINAL"),
    ]
    assert family_verdict_from_reports(reports) == "SIGNAL"


def test_family_verdict_null_when_all_null() -> None:
    """Family is NULL only when every pooled Phase 1 cell is NULL/REGRESSION
    AND every Phase 2 cell is DO_NOT_ADOPT (or absent)."""
    reports = [
        _stub_report(phase1_verdict="NULL", phase2_verdict=None),
        _stub_report(phase1_verdict="REGRESSION", phase2_verdict="DO_NOT_ADOPT"),
    ]
    assert family_verdict_from_reports(reports) == "NULL"
