# tests/test_scripts/test_probe_feature_signal_cli.py
"""Feature signal probe CLI tests."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scripts.probe_feature_signal import (
    OverrideCollisionError,
    OverrideCoverageError,
    load_features_with_overrides,
    main,
    parse_args,
    render_csv,
    render_markdown,
    validate_override_coverage,
)

from projections.backtest.adoption_gate import BootstrapDelta, PositionVerdict
from projections.backtest.feature_probe import PerStatVerdict, ProbeReport
from projections.models import POSITION_DISPATCH
from projections.models.baseline import BaselineModel
from projections.schemas import _PYARROW_STR, DistributionFamily, Position, Stat


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
    assert args.force_composite is False


def test_parse_args_force_composite_flag() -> None:
    args = parse_args(
        [
            "--candidate-name",
            "x",
            "--drop",
            "y",
            "--force-composite",
        ]
    )
    assert args.composite is True
    assert args.force_composite is True


def test_parse_args_force_composite_default_false() -> None:
    args = parse_args(
        [
            "--candidate-name",
            "x",
            "--drop",
            "y",
        ]
    )
    assert args.force_composite is False


def test_parse_args_force_composite_mutex_with_no_composite() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--candidate-name",
                "x",
                "--drop",
                "y",
                "--no-composite",
                "--force-composite",
            ]
        )


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


def test_parse_args_coverage_threshold_default_and_override() -> None:
    """The --coverage-threshold flag defaults to 0.95 and accepts a custom value."""
    args = parse_args(
        [
            "--candidate-name",
            "x",
            "--drop",
            "y",
        ]
    )
    assert args.coverage_threshold == 0.95

    args = parse_args(
        [
            "--candidate-name",
            "x",
            "--drop",
            "y",
            "--coverage-threshold",
            "0.80",
        ]
    )
    assert args.coverage_threshold == 0.80


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


def _per_year_signal_pooled_null_phase1() -> list[PerStatVerdict]:
    """Phase 1 result set with a per-year SIGNAL on QB passing_yards 2023 but
    a pooled NULL — exactly the QB swap retro shape that motivated
    --force-composite. Used by the no_pooled_signal render tests."""
    return [
        PerStatVerdict(
            position=Position.QB,
            stat=Stat.PASSING_YARDS,
            year_or_pooled=2023,
            n_paired=552,
            rmse_delta=BootstrapDelta(
                point=-0.29, lo_95=-0.53, hi_95=-0.05, n_paired_rows=552, n_bootstrap=1000
            ),
            r_squared_delta=0.001,
            verdict="SIGNAL",
        ),
        PerStatVerdict(
            position=Position.QB,
            stat=Stat.PASSING_YARDS,
            year_or_pooled="pooled",
            n_paired=2223,
            rmse_delta=BootstrapDelta(
                point=-0.07, lo_95=-0.14, hi_95=0.01, n_paired_rows=2223, n_bootstrap=1000
            ),
            r_squared_delta=0.001,
            verdict="NULL",
        ),
    ]


def test_render_markdown_no_pooled_signal_skip_reason_suggests_force_composite() -> None:
    """Per-year SIGNAL only (no pooled) under default gating renders a hint
    about --force-composite, not the misleading 'disabled by --no-composite'."""
    report = ProbeReport(
        candidate_name="x",
        model_class="baseline",
        baseline_features_path="data/features",
        override_paths=("/tmp/o.parquet",),
        drop_columns=(),
        phase1=_per_year_signal_pooled_null_phase1(),
        phase2=None,
        phase2_skip_reason="no_pooled_signal",
    )
    md = render_markdown(report)
    assert "--force-composite" in md
    assert "disabled by --no-composite" not in md


def test_render_markdown_no_signal_skip_reason() -> None:
    """No SIGNAL cells → render says 'No SIGNAL cells', predicts DO_NOT_ADOPT."""
    psv_null = PerStatVerdict(
        position=Position.QB,
        stat=Stat.PASSING_YARDS,
        year_or_pooled="pooled",
        n_paired=2223,
        rmse_delta=BootstrapDelta(
            point=0.001, lo_95=-0.01, hi_95=0.01, n_paired_rows=2223, n_bootstrap=1000
        ),
        r_squared_delta=0.0,
        verdict="NULL",
    )
    report = ProbeReport(
        candidate_name="x",
        model_class="baseline",
        baseline_features_path="data/features",
        override_paths=("/tmp/o.parquet",),
        drop_columns=(),
        phase1=[psv_null],
        phase2=None,
        phase2_skip_reason="no_signal",
    )
    md = render_markdown(report)
    assert "No SIGNAL cells" in md


def test_render_markdown_user_disabled_skip_reason() -> None:
    """--no-composite with a SIGNAL hit renders the user-disabled message."""
    psv_signal = PerStatVerdict(
        position=Position.QB,
        stat=Stat.PASSING_YARDS,
        year_or_pooled="pooled",
        n_paired=2223,
        rmse_delta=BootstrapDelta(
            point=-0.30, lo_95=-0.50, hi_95=-0.10, n_paired_rows=2223, n_bootstrap=1000
        ),
        r_squared_delta=0.001,
        verdict="SIGNAL",
    )
    report = ProbeReport(
        candidate_name="x",
        model_class="baseline",
        baseline_features_path="data/features",
        override_paths=("/tmp/o.parquet",),
        drop_columns=(),
        phase1=[psv_signal],
        phase2=None,
        phase2_skip_reason="user_disabled",
    )
    md = render_markdown(report)
    assert "disabled by --no-composite" in md


# ----------------------------------------------------------------------
# Integration tests for main() — Task 3.2
# ----------------------------------------------------------------------


class _MockBaseline(BaselineModel):
    """Subclass of BaselineModel that overrides fit/predict_distribution to
    skip schema validation and produce deterministic mock predictions, but
    still satisfies isinstance(BaselineModel) so _get_production_columns_and_stats
    and _build_factory_with_columns dispatch correctly."""

    def fit(self, features: pd.DataFrame, weekly_stats: pd.DataFrame) -> None:
        # No-op fit. Skips WeeklyStatsSchema and feature_schema validation
        # that real BaselineModel.fit performs — the synthetic universe
        # doesn't satisfy those schemas, and the integration test only
        # cares about main()'s control-flow / IO wiring, not Ridge math.
        return None

    def predict_distribution(self, features: pd.DataFrame, ruleset: object) -> pd.DataFrame:
        # Skip schema validation; produce a synthetic prediction frame
        # matching the columns probe_composite reads (gsis_id, season, week, mean).
        out = features[["gsis_id", "season", "week"]].copy()
        out["mean"] = features[list(self.feature_columns)].sum(axis=1).to_numpy() / 25.0
        out["p10"] = out["mean"] - 1.0
        out["p50"] = out["mean"]
        out["p90"] = out["mean"] + 1.0
        out["position"] = self.position.value
        out["team"] = "KC"
        out["opponent"] = "BUF"
        return out


def _make_mock_baseline() -> _MockBaseline:
    """Construct a _MockBaseline analogue of qb_baseline() for the synthetic
    universe — BaselineModel subclass so the production isinstance() branches
    in _get_production_columns_and_stats and _build_factory_with_columns
    dispatch correctly."""
    return _MockBaseline(
        position=Position.QB,
        target_stats=(Stat.PASSING_YARDS,),
        feature_columns=("base_x1", "base_x2", "base_x3"),
        dist_families={Stat.PASSING_YARDS: DistributionFamily.NORMAL},
        feature_schema=POSITION_DISPATCH[Position.QB].feature_schema,
        code_hash_files=(),
    )


@pytest.fixture
def monkeypatch_probe_universe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Path]:
    """Set up a synthetic features cache + weekly_stats + override on disk.

    Returns a dict of paths so the test can pass them to the CLI.
    """
    rng = np.random.default_rng(0)
    n_players, seasons, weeks_per = 25, range(2018, 2023), range(1, 5)

    rows: list[dict[str, object]] = []
    for player_idx in range(n_players):
        gsis_id = f"00-003{player_idx:04d}"
        for season in seasons:
            for week in weeks_per:
                rows.append(
                    {
                        "gsis_id": gsis_id,
                        "season": season,
                        "week": week,
                        "team": "KC",
                        "opponent": "BUF",
                        "position": "QB",
                    }
                )
    base = pd.DataFrame(rows)
    n = len(base)
    base["base_x1"] = rng.normal(size=n)
    base["base_x2"] = rng.normal(size=n)
    base["base_x3"] = rng.normal(size=n)
    cand_signal = rng.normal(size=n)

    for col in ("gsis_id", "team", "opponent", "position"):
        base[col] = base[col].astype(_PYARROW_STR)

    # Write per-(season, week) parquet partitions to mimic the feature cache layout.
    features_root = tmp_path / "features"
    for season in seasons:
        for week in weeks_per:
            mask = (base["season"] == season) & (base["week"] == week)
            part_dir = features_root / "qb" / f"season={season}" / f"week={week}"
            part_dir.mkdir(parents=True, exist_ok=True)
            base[mask].to_parquet(part_dir / "part.parquet")

    # Override parquet (cand_signal column).
    override_path = tmp_path / "override.parquet"
    override_df = base[["gsis_id", "season", "week"]].copy()
    override_df["cand_signal"] = cand_signal
    override_df.to_parquet(override_path)

    # weekly_stats partition with target stats. Synthetic passing_yards is a
    # linear combination of the baseline features + a strong cand_signal term
    # so the candidate model strictly improves CV-RMSE — i.e. Phase 1 SIGNAL
    # should fire on the augment-mode test.
    weekly_stats = base[["gsis_id", "season", "week", "position"]].copy()
    weekly_stats["passing_yards"] = (
        base["base_x1"] + 0.5 * base["base_x2"] + 1.0 * cand_signal + rng.normal(scale=0.5, size=n)
    )
    for stat_col in (
        "passing_tds",
        "interceptions",
        "rushing_yards",
        "rushing_tds",
        "fumbles_lost",
    ):
        weekly_stats[stat_col] = 0.0
    raw_root = tmp_path / "raw"
    for season in seasons:
        mask = weekly_stats["season"] == season
        part_dir = raw_root / "weekly_stats" / f"season={season}"
        part_dir.mkdir(parents=True, exist_ok=True)
        weekly_stats[mask].to_parquet(part_dir / "part.parquet")

    # Monkeypatch read_partition to redirect "data/raw" to our tmp_path/raw.
    # main() hardcodes Path("data/raw") for weekly_stats; the redirector also
    # passes through the (real tmp_path) features_root call unchanged.
    import projections.store as store_mod

    real_read = store_mod.read_partition
    raw_data_root = Path("data/raw")

    def _redirected_read(root: Path, table: str, **kwargs: Any) -> pd.DataFrame:
        if Path(root) == raw_data_root:
            return real_read(raw_root, table, **kwargs)
        return real_read(root, table, **kwargs)

    monkeypatch.setattr(store_mod, "read_partition", _redirected_read)
    monkeypatch.setattr("scripts.probe_feature_signal.read_partition", _redirected_read)

    # Monkeypatch POSITION_DISPATCH[QB].factories["baseline"] to our mock.
    # The factories Mapping is in fact a plain dict (see models/__init__.py),
    # so setitem works.
    monkeypatch.setitem(POSITION_DISPATCH[Position.QB].factories, "baseline", _make_mock_baseline)

    return {
        "features_root": features_root,
        "override": override_path,
    }


def test_main_augment_mode_runs_end_to_end(
    monkeypatch_probe_universe: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    csv_out = tmp_path / "report.csv"
    main(
        [
            "--candidate-name",
            "augment_test",
            "--baseline-features",
            str(monkeypatch_probe_universe["features_root"]),
            "--override",
            str(monkeypatch_probe_universe["override"]),
            "--position",
            "QB",
            "--seasons",
            "2018-2022",
            "--holdout-years",
            "2021-2022",
            "--n-bootstrap",
            "200",
            "--csv-out",
            str(csv_out),
        ]
    )
    captured = capsys.readouterr()
    assert "# Feature signal probe — augment_test" in captured.out
    assert "### QB" in captured.out
    assert "passing_yards" in captured.out
    # The candidate column carries strong synthetic signal — Phase 2 typically
    # fires, but on this small fixture the bootstrap CI may bracket zero.
    # Accept either outcome — the probe ran end-to-end is the contract here.
    assert (
        "## Phase 2" in captured.out
        or "Phase 2 skipped" in captured.out
        or "Phase 2: not run" in captured.out
    )
    assert csv_out.exists()
    df = pd.read_csv(csv_out)
    assert "phase1" in df["phase"].unique()


def test_main_no_composite_skips_phase2(
    monkeypatch_probe_universe: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(
        [
            "--candidate-name",
            "no_composite_test",
            "--baseline-features",
            str(monkeypatch_probe_universe["features_root"]),
            "--override",
            str(monkeypatch_probe_universe["override"]),
            "--position",
            "QB",
            "--seasons",
            "2018-2022",
            "--holdout-years",
            "2021-2022",
            "--n-bootstrap",
            "200",
            "--no-composite",
        ]
    )
    captured = capsys.readouterr()
    # Phase 2 disabled; either we never SIGNAL'd (Phase 2 skipped) or we did
    # but --no-composite suppresses execution. No "## Phase 2" composite table.
    assert "## Phase 2 — composite ΔRMSE" not in captured.out
    assert "Phase 2 disabled by --no-composite" in captured.out or "Phase 2 skipped" in captured.out


def test_main_force_composite_runs_phase2_without_pooled_signal(
    monkeypatch_probe_universe: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--force-composite runs Phase 2 even when Phase 1 finds no pooled SIGNAL.

    Drops a column that doesn't exist in baseline so the transform is a no-op:
    baseline_cols == candidate_cols, Phase 1 returns all-NULL, default gating
    would skip Phase 2. --force-composite overrides the gate.
    """
    main(
        [
            "--candidate-name",
            "force_composite_test",
            "--baseline-features",
            str(monkeypatch_probe_universe["features_root"]),
            "--drop",
            "nonexistent_column",
            "--position",
            "QB",
            "--seasons",
            "2018-2022",
            "--holdout-years",
            "2021-2022",
            "--n-bootstrap",
            "200",
            "--force-composite",
        ]
    )
    captured = capsys.readouterr()
    assert "## Phase 2 — composite ΔRMSE" in captured.out


def test_main_swap_mode_drop_baseline_column(
    monkeypatch_probe_universe: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Same override but with --drop base_x1 — i.e., swap the new column for
    an existing one."""
    main(
        [
            "--candidate-name",
            "swap_test",
            "--baseline-features",
            str(monkeypatch_probe_universe["features_root"]),
            "--override",
            str(monkeypatch_probe_universe["override"]),
            "--drop",
            "base_x1",
            "--position",
            "QB",
            "--seasons",
            "2018-2022",
            "--holdout-years",
            "2021-2022",
            "--n-bootstrap",
            "200",
            "--no-composite",
        ]
    )
    captured = capsys.readouterr()
    assert "swap_test" in captured.out
    # The drop column appears in the header.
    assert "base_x1" in captured.out


def test_main_ablation_mode_drop_only(
    monkeypatch_probe_universe: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(
        [
            "--candidate-name",
            "ablation_test",
            "--baseline-features",
            str(monkeypatch_probe_universe["features_root"]),
            "--drop",
            "base_x1",
            "--position",
            "QB",
            "--seasons",
            "2018-2022",
            "--holdout-years",
            "2021-2022",
            "--n-bootstrap",
            "200",
            "--no-composite",
        ]
    )
    captured = capsys.readouterr()
    assert "ablation_test" in captured.out
