# tests/test_scripts/test_probe_feature_signal_cli.py
"""Feature signal probe CLI tests."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scripts.probe_feature_signal import (
    OverrideCollisionError,
    OverrideCoverageError,
    load_features_with_overrides,
    parse_args,
    render_csv,
    render_markdown,
    validate_override_coverage,
)

from projections.backtest.adoption_gate import BootstrapDelta, PositionVerdict
from projections.backtest.feature_probe import PerStatVerdict, ProbeReport
from projections.schemas import _PYARROW_STR, Position, Stat


def test_parse_args_minimum_required_args() -> None:
    args = parse_args(
        [
            "--candidate-name",
            "test_candidate",
            "--override",
            "/tmp/overrides.parquet",
        ]
    )
    assert args.candidate_name == "test_candidate"
    assert args.override == [Path("/tmp/overrides.parquet")]
    assert args.drop == []
    assert args.model == "baseline"
    assert args.position == ["QB", "RB", "WR", "TE"]
    assert args.seasons == (2018, 2024)
    assert args.holdout_years == (2021, 2024)
    assert args.n_bootstrap == 1000
    assert args.seed == 42
    assert args.csv_out is None
    assert args.composite is True


def test_parse_args_drop_only_is_valid() -> None:
    """Ablation mode — drop a column from baseline, no override."""
    args = parse_args(
        [
            "--candidate-name",
            "ablation_drop_one",
            "--drop",
            "opp_allowed_qb_fppg_l4",
        ]
    )
    assert args.override == []
    assert args.drop == ["opp_allowed_qb_fppg_l4"]


def test_parse_args_rejects_no_override_and_no_drop() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--candidate-name", "test"])


def test_parse_args_repeat_override() -> None:
    args = parse_args(
        [
            "--candidate-name",
            "multi",
            "--override",
            "/tmp/a.parquet",
            "--override",
            "/tmp/b.parquet",
        ]
    )
    assert args.override == [Path("/tmp/a.parquet"), Path("/tmp/b.parquet")]


def test_parse_args_drop_comma_separated() -> None:
    args = parse_args(
        [
            "--candidate-name",
            "x",
            "--drop",
            "col_a,col_b,col_c",
        ]
    )
    assert args.drop == ["col_a", "col_b", "col_c"]


def test_parse_args_repeat_position() -> None:
    args = parse_args(
        [
            "--candidate-name",
            "x",
            "--drop",
            "y",
            "--position",
            "QB",
            "--position",
            "WR",
        ]
    )
    assert args.position == ["QB", "WR"]


def test_parse_args_seasons_range() -> None:
    args = parse_args(
        [
            "--candidate-name",
            "x",
            "--drop",
            "y",
            "--seasons",
            "2019-2023",
            "--holdout-years",
            "2022-2023",
        ]
    )
    assert args.seasons == (2019, 2023)
    assert args.holdout_years == (2022, 2023)


def test_parse_args_no_composite_flag() -> None:
    args = parse_args(
        [
            "--candidate-name",
            "x",
            "--drop",
            "y",
            "--no-composite",
        ]
    )
    assert args.composite is False


def test_parse_args_model_choice() -> None:
    args = parse_args(
        [
            "--candidate-name",
            "x",
            "--drop",
            "y",
            "--model",
            "lightgbm-nb",
        ]
    )
    assert args.model == "lightgbm-nb"


def test_parse_args_rejects_unknown_model() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--candidate-name",
                "x",
                "--drop",
                "y",
                "--model",
                "bogus",
            ]
        )


@pytest.fixture
def synthetic_baseline_features(tmp_path: Path) -> Path:
    """Tiny baseline features cache: 1 position (QB) x 100 rows."""
    rng = np.random.default_rng(0)
    n = 100
    rows = []
    for i in range(n):
        rows.append(
            {
                "gsis_id": f"00-003{i:04d}",
                "season": 2022,
                "week": (i % 17) + 1,
                "team": "KC",
                "opponent": "BUF",
                "base_x1": float(rng.normal()),
            }
        )
    df = pd.DataFrame(rows)
    for col in ("gsis_id", "team", "opponent"):
        df[col] = df[col].astype(_PYARROW_STR)
    out_dir = tmp_path / "qb" / "season=2022" / "week=1"
    out_dir.mkdir(parents=True)
    df.to_parquet(out_dir / "part.parquet")
    return tmp_path


@pytest.fixture
def synthetic_override(tmp_path: Path) -> Path:
    """Override parquet covering 100% of synthetic_baseline_features rows."""
    rows = [
        {
            "gsis_id": f"00-003{i:04d}",
            "season": 2022,
            "week": (i % 17) + 1,
            "cand_signal": float(i) / 100.0,
        }
        for i in range(100)
    ]
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    out_path = tmp_path / "override.parquet"
    df.to_parquet(out_path)
    return out_path


def test_validate_override_coverage_passes_at_full_coverage(
    synthetic_baseline_features: Path, synthetic_override: Path
) -> None:
    baseline = pd.DataFrame(
        {
            "gsis_id": [f"00-003{i:04d}" for i in range(100)],
            "season": [2022] * 100,
            "week": [1] * 100,
        }
    )
    override = pd.DataFrame(
        {
            "gsis_id": [f"00-003{i:04d}" for i in range(100)],
            "season": [2022] * 100,
            "week": [1] * 100,
            "cand_signal": [float(i) for i in range(100)],
        }
    )
    # Should not raise.
    validate_override_coverage(
        baseline=baseline,
        joined=baseline.merge(override, on=["gsis_id", "season", "week"], how="left"),
        candidate_columns=("cand_signal",),
        threshold=0.95,
    )


def test_validate_override_coverage_raises_below_threshold() -> None:
    baseline = pd.DataFrame(
        {
            "gsis_id": [f"00-003{i:04d}" for i in range(100)],
            "season": [2022] * 100,
            "week": [1] * 100,
        }
    )
    # Only first 80 rows have the candidate value.
    override = pd.DataFrame(
        {
            "gsis_id": [f"00-003{i:04d}" for i in range(80)],
            "season": [2022] * 80,
            "week": [1] * 80,
            "cand_signal": [float(i) for i in range(80)],
        }
    )
    joined = baseline.merge(override, on=["gsis_id", "season", "week"], how="left")
    with pytest.raises(OverrideCoverageError, match="80%"):
        validate_override_coverage(
            baseline=baseline,
            joined=joined,
            candidate_columns=("cand_signal",),
            threshold=0.95,
        )


def test_load_features_with_overrides_raises_on_collision_without_drop(
    synthetic_baseline_features: Path, tmp_path: Path
) -> None:
    """An override that introduces a column name already in the baseline,
    with that column not listed in --drop, must raise."""
    colliding = pd.DataFrame(
        {
            "gsis_id": [f"00-003{i:04d}" for i in range(100)],
            "season": [2022] * 100,
            "week": [(i % 17) + 1 for i in range(100)],
            "base_x1": [0.0] * 100,  # collides with baseline column!
        }
    )
    colliding["gsis_id"] = colliding["gsis_id"].astype(_PYARROW_STR)
    override_path = tmp_path / "colliding.parquet"
    colliding.to_parquet(override_path)
    with pytest.raises(OverrideCollisionError, match="base_x1"):
        load_features_with_overrides(
            position="QB",
            features_root=synthetic_baseline_features,
            override_paths=[override_path],
            drop_columns=[],  # base_x1 NOT in drop list — this is the error condition
            seasons=range(2022, 2023),
            baseline_columns=("base_x1",),
            coverage_threshold=0.95,
        )


def test_load_features_with_overrides_collision_allowed_when_dropped(
    synthetic_baseline_features: Path, tmp_path: Path
) -> None:
    """Same colliding override as above, but base_x1 IS in --drop — allowed."""
    override = pd.DataFrame(
        {
            "gsis_id": [f"00-003{i:04d}" for i in range(100)],
            "season": [2022] * 100,
            "week": [(i % 17) + 1 for i in range(100)],
            "base_x1": [99.0] * 100,
        }
    )
    override["gsis_id"] = override["gsis_id"].astype(_PYARROW_STR)
    override_path = tmp_path / "override.parquet"
    override.to_parquet(override_path)
    # Should not raise.
    df = load_features_with_overrides(
        position="QB",
        features_root=synthetic_baseline_features,
        override_paths=[override_path],
        drop_columns=["base_x1"],
        seasons=range(2022, 2023),
        baseline_columns=("base_x1",),
        coverage_threshold=0.95,
    )
    # The override value survives.
    assert (df["base_x1"] == 99.0).all()


def _build_sample_report(*, with_phase2: bool) -> ProbeReport:
    psv_signal = PerStatVerdict(
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
    psv_null = PerStatVerdict(
        position=Position.QB,
        stat=Stat.PASSING_YARDS,
        year_or_pooled="pooled",
        n_paired=2676,
        rmse_delta=BootstrapDelta(
            point=0.005, lo_95=-0.10, hi_95=0.11, n_paired_rows=2676, n_bootstrap=1000
        ),
        r_squared_delta=0.0001,
        verdict="NULL",
    )
    pv = PositionVerdict(
        position=Position.QB,
        incumbent_class="_baseline_features",
        candidate_class="_candidate_features",
        rmse_delta=BootstrapDelta(
            point=-0.04, lo_95=-0.18, hi_95=0.10, n_paired_rows=2676, n_bootstrap=1000
        ),
        spearman_delta=BootstrapDelta(
            point=0.001, lo_95=-0.003, hi_95=0.005, n_paired_rows=2676, n_bootstrap=1000
        ),
        verdict="DO_NOT_ADOPT",
        reason="RMSE inconclusive",
        per_year_breakdown=pd.DataFrame(
            {
                "year": [2024],
                "n_paired": [670],
                "rmse_delta_point": [-0.04],
                "rmse_delta_lo": [-0.18],
                "rmse_delta_hi": [0.10],
                "spearman_delta_point": [0.001],
                "spearman_delta_lo": [-0.003],
                "spearman_delta_hi": [0.005],
            }
        ),
    )
    return ProbeReport(
        candidate_name="example_candidate",
        model_class="baseline",
        baseline_features_path="data/features",
        override_paths=("data/features_probe/x.parquet",),
        drop_columns=("opp_allowed_qb_fppg_l4",),
        phase1=[psv_signal, psv_null],
        phase2=[pv] if with_phase2 else None,
    )


def test_render_markdown_phase1_only() -> None:
    md = render_markdown(_build_sample_report(with_phase2=False))
    assert "# Feature signal probe — example_candidate" in md
    assert "Baseline features: data/features" in md
    assert "data/features_probe/x.parquet" in md
    assert "opp_allowed_qb_fppg_l4" in md
    assert "Model class:      baseline" in md
    assert "## Phase 1 — per-stat screening" in md
    assert "### QB" in md
    assert "passing_yards" in md
    assert "SIGNAL" in md
    assert "NULL" in md
    assert "## Phase 1 verdict" in md
    # Phase 2 not run, so no Phase 2 section.
    assert "## Phase 2" not in md
    assert "## Probe verdict" in md
    assert "Phase 2 skipped" in md or "Phase 2: not run" in md or "no Phase 2" in md.lower()


def test_render_markdown_phase1_and_phase2() -> None:
    md = render_markdown(_build_sample_report(with_phase2=True))
    assert "## Phase 2 — composite ΔRMSE" in md
    assert "DO_NOT_ADOPT" in md
    assert "## Probe verdict" in md


def test_render_csv_long_format() -> None:
    csv_text = render_csv(_build_sample_report(with_phase2=True))
    df = pd.read_csv(io.StringIO(csv_text))
    assert set(df.columns) >= {
        "phase",
        "position",
        "stat_or_composite",
        "year_or_pooled",
        "metric_name",
        "point",
        "lo_95",
        "hi_95",
        "n_paired",
        "verdict",
    }
    # Phase 1 rows for SIGNAL + NULL + Phase 2 row(s).
    assert "phase1" in df["phase"].unique()
    assert "phase2" in df["phase"].unique()
    qb_rows = df[df["position"] == "QB"]
    assert "passing_yards" in qb_rows["stat_or_composite"].unique()
    assert "composite" in qb_rows["stat_or_composite"].unique()


def test_render_csv_phase1_only_omits_phase2_rows() -> None:
    csv_text = render_csv(_build_sample_report(with_phase2=False))
    df = pd.read_csv(io.StringIO(csv_text))
    assert "phase2" not in df["phase"].unique()
