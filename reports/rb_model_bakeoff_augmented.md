# RB model bake-off — AUGMENTED feature set (trajectory-trend added)

RB-only walk-forward 2021-2024, ESPN-PPR composite, features_root=data/features_rb_aug.
Compare vs reports/rb_model_bakeoff_2026-06-25.md (old baseline pooled composite_rmse 6.5661, mae 4.9870, spearman 0.9694).

```

=== RB walk-forward backtest (ESPN-PPR composite) ===
model_class          baseline  ensemble  lightgbm-nb
metric         year
composite_mae  2021    5.2344    5.4062       5.4981
               2022    5.0252    5.1128       5.1646
               2023    4.6773    4.8182       4.8751
               2024    4.9006    5.0267       5.0688
composite_rmse 2021    6.8733    6.8743       6.9225
               2022    6.6158    6.5852       6.6066
               2023    6.2907    6.3295       6.3600
               2024    6.4722    6.5026       6.5282
spearman_topN  2021    0.9705    0.9666       0.9624
               2022    0.9703    0.9706       0.9703
               2023    0.9668    0.9656       0.9633
               2024    0.9761    0.9749       0.9759

=== Pooled (mean over 2021-2024) ===
model_class     baseline  ensemble  lightgbm-nb
metric
composite_mae     4.9594    5.0910       5.1516
composite_rmse    6.5630    6.5729       6.6043
spearman_topN     0.9709    0.9694       0.9680
```
