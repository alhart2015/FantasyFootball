# Adoption gate report — lightgbm vs baseline

Run: `data\backtest\run_20260429T003552Z`
n_bootstrap: 1000, seed: 42

### QB — lightgbm vs baseline: **DO_NOT_ADOPT**

_RMSE inconclusive: 95% CI [-0.124, +0.076] brackets / exceeds zero_

- n_paired: 2676; n_bootstrap: 1000
- RMSE delta: -0.0233 (95% CI [-0.1239, +0.0758])
- Spearman delta: +0.0155 (95% CI [+0.0010, +0.0296])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021       665          0.004487      -0.193920       0.228875              0.020961          -0.008628           0.051931
 2022       657          0.045638      -0.176469       0.257159              0.015142          -0.012396           0.044381
 2023       670          0.012110      -0.192993       0.212298             -0.000582          -0.028174           0.025062
 2024       684         -0.147273      -0.318077       0.023743              0.026380           0.002998           0.050381

### RB — lightgbm vs baseline: **DO_NOT_ADOPT**

_RMSE inconclusive: 95% CI [+0.144, +0.242] brackets / exceeds zero_

- n_paired: 5273; n_bootstrap: 1000
- RMSE delta: +0.1916 (95% CI [+0.1438, +0.2421])
- Spearman delta: -0.0023 (95% CI [-0.0082, +0.0028])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      1315          0.220218       0.111837       0.331397             -0.017526          -0.030492          -0.004334
 2022      1331          0.201037       0.103093       0.301518              0.011539          -0.000301           0.023239
 2023      1311          0.192651       0.115292       0.269792             -0.002703          -0.013861           0.008117
 2024      1316          0.150998       0.081904       0.218343             -0.000528          -0.010849           0.008380

### TE — lightgbm vs baseline: **DO_NOT_ADOPT**

_RMSE inconclusive: 95% CI [+0.110, +0.206] brackets / exceeds zero_

- n_paired: 4257; n_bootstrap: 1000
- RMSE delta: +0.1553 (95% CI [+0.1096, +0.2060])
- Spearman delta: +0.0043 (95% CI [-0.0052, +0.0132])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      1030          0.127120       0.032120       0.226965              0.014328          -0.008421           0.034904
 2022      1088          0.121154       0.024771       0.225505             -0.002525          -0.022798           0.016806
 2023      1058          0.190175       0.101004       0.275991              0.011008          -0.004974           0.028226
 2024      1081          0.185776       0.105352       0.273056             -0.005722          -0.022200           0.009631

### WR — lightgbm vs baseline: **DO_NOT_ADOPT**

_RMSE inconclusive: 95% CI [+0.096, +0.172] brackets / exceeds zero_

- n_paired: 8460; n_bootstrap: 1000
- RMSE delta: +0.1338 (95% CI [+0.0963, +0.1721])
- Spearman delta: +0.0045 (95% CI [-0.0012, +0.0101])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      2109          0.150418       0.072024       0.224069              0.009741          -0.001150           0.021335
 2022      2102          0.122328       0.047484       0.191625              0.005235          -0.005672           0.016047
 2023      2201          0.196985       0.118829       0.273069             -0.001402          -0.013361           0.010148
 2024      2048          0.061186       0.001639       0.123813              0.004231          -0.006518           0.014814

Wrote CSV: reports\adoption_gate_lightgbm.csv
