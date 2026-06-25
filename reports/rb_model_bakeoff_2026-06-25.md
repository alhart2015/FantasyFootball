# RB model bake-off — old feature set (baseline-to-beat)

RB-only walk-forward 2021–2024, ESPN-PPR composite. Driver: `scripts/_rb_model_bakeoff.py`
(run this session, 2026-06-25). Lower MAE/RMSE = better; higher spearman = better.
`baseline` (RidgeCV) is the incumbent `POSITION_DISPATCH[Position.RB].default_model_class`.

This is the **old-feature-set** reference the Phase 2(b) re-bake-off (after adding the
trajectory/Vegas features) is compared against. It also grounds the spec §1 claim that
RB is **signal-limited on the current features**: a flexible learner (lightgbm) cannot
beat a linear one (ridge) here, so the only lever is new feature signal.

```
=== RB walk-forward backtest (ESPN-PPR composite) ===
model_class          baseline  ensemble  lightgbm-nb
metric         year
composite_mae  2021    5.2321    5.3774       5.4636
               2022    5.0363    5.1179       5.1640
               2023    4.7250    4.8667       4.8842
               2024    4.9547    5.0579       5.0733
composite_rmse 2021    6.8355    6.8576       6.9067
               2022    6.6120    6.6008       6.6217
               2023    6.3068    6.3684       6.3820
               2024    6.5101    6.5318       6.5387
spearman_topN  2021    0.9714    0.9658       0.9624
               2022    0.9670    0.9688       0.9689
               2023    0.9629    0.9637       0.9622
               2024    0.9763    0.9755       0.9758

=== Pooled (mean over 2021-2024) ===
model_class     baseline  ensemble  lightgbm-nb
metric
composite_mae     4.9870    5.1050       5.1463
composite_rmse    6.5661    6.5896       6.6123
spearman_topN     0.9694    0.9685       0.9673
```

**Verdict:** `baseline` wins all three pooled metrics and nearly every year-cell
(gradient `baseline > ensemble > lightgbm-nb`). Confirms both cheap lift paths are dead:
decomposition NULL'd (`reports/feature_probe_rb_decomposition_summary.md`) and a model-class
swap loses to baseline here. New feature signal is the only remaining lever — the premise
of the trajectory/Vegas probe in `docs/superpowers/plans/2026-06-25-rb-trajectory-vegas-features.md`.
