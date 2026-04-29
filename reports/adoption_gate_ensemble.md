# Adoption gate report — ensemble vs baseline

Run: `data\backtest\run_20260429T003552Z`
n_bootstrap: 1000, seed: 42

### QB — ensemble vs baseline: **ADOPT**

_RMSE delta -0.176 (95% CI [-0.227, -0.124]); Spearman lo_95 +0.0098 > floor -0.020_

- n_paired: 2676; n_bootstrap: 1000
- RMSE delta: -0.1760 (95% CI [-0.2274, -0.1242])
- Spearman delta: +0.0184 (95% CI [+0.0098, +0.0262])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021       665         -0.194596      -0.290334      -0.102828              0.023834           0.010285           0.037695
 2022       657         -0.182966      -0.292255      -0.080135              0.020844           0.004219           0.037440
 2023       670         -0.131127      -0.258441       0.008467              0.005896          -0.015841           0.025447
 2024       684         -0.193423      -0.278764      -0.113485              0.022859           0.010178           0.036001

### RB — ensemble vs baseline: **DO_NOT_ADOPT**

_RMSE inconclusive: 95% CI [-0.002, +0.046] brackets / exceeds zero_

- n_paired: 5273; n_bootstrap: 1000
- RMSE delta: +0.0212 (95% CI [-0.0021, +0.0455])
- Spearman delta: +0.0003 (95% CI [-0.0037, +0.0043])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      1315          0.046147      -0.014013       0.101883             -0.010878          -0.020918          -0.000859
 2022      1331         -0.016225      -0.063251       0.031569              0.007197          -0.001154           0.015379
 2023      1311          0.034232      -0.008124       0.076136              0.003744          -0.003560           0.011173
 2024      1316          0.020953      -0.015434       0.060062              0.001179          -0.005914           0.008046

### TE — ensemble vs baseline: **DO_NOT_ADOPT**

_RMSE inconclusive: 95% CI [-0.045, +0.010] brackets / exceeds zero_

- n_paired: 4257; n_bootstrap: 1000
- RMSE delta: -0.0208 (95% CI [-0.0454, +0.0097])
- Spearman delta: +0.0076 (95% CI [+0.0016, +0.0137])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      1030         -0.061477      -0.109881      -0.015441              0.021597           0.008130           0.034386
 2022      1088         -0.047474      -0.097489       0.001064              0.002590          -0.010116           0.016990
 2023      1058         -0.038145      -0.082399       0.005819              0.012362           0.002225           0.022757
 2024      1081          0.063151       0.004998       0.139135             -0.006308          -0.017422           0.003771

### WR — ensemble vs baseline: **ADOPT**

_RMSE delta -0.032 (95% CI [-0.053, -0.009]); Spearman lo_95 +0.0028 > floor -0.020_

- n_paired: 8460; n_bootstrap: 1000
- RMSE delta: -0.0320 (95% CI [-0.0531, -0.0092])
- Spearman delta: +0.0069 (95% CI [+0.0028, +0.0109])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      2109         -0.036698      -0.079619       0.006475              0.011408           0.003545           0.019274
 2022      2102         -0.034566      -0.075400       0.006580              0.007309          -0.000197           0.014934
 2023      2201         -0.023893      -0.064906       0.018754              0.004511          -0.003354           0.011996
 2024      2048         -0.032910      -0.074794       0.011067              0.004459          -0.003832           0.012802

Wrote CSV: reports\adoption_gate_ensemble.csv
