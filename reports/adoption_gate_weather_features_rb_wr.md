=== (baseline, RB) ===
# Adoption gate report — _candidate_run vs _baseline_run

Run: baseline=`data\backtest\run_baseline_baseline` candidate=`data\backtest\run_candidate_baseline`
n_bootstrap: 1000, seed: 42

### RB — _candidate_run vs _baseline_run: **DO_NOT_ADOPT**

_RMSE inconclusive: 95% CI [-0.008, +0.008] brackets / exceeds zero_

- n_paired: 5273; n_bootstrap: 1000
- RMSE delta: -0.0001 (95% CI [-0.0081, +0.0078])
- Spearman delta: +0.0002 (95% CI [-0.0012, +0.0017])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      1315         -0.005201      -0.022766       0.013234              0.001519          -0.001597           0.004615
 2022      1331         -0.007953      -0.022228       0.006898              0.001360          -0.001162           0.003722
 2023      1311          0.003313      -0.011486       0.017558             -0.000757          -0.003093           0.002075
 2024      1316          0.010099      -0.003576       0.024187             -0.001165          -0.003839           0.001495

Wrote CSV: reports\adoption_gate_weather_baseline_RB.csv

=== (lightgbm-nb, RB) ===
# Adoption gate report — _candidate_run vs _baseline_run

Run: baseline=`data\backtest\run_baseline_lightgbm-nb` candidate=`data\backtest\run_candidate_lightgbm-nb`
n_bootstrap: 1000, seed: 42

### RB — _candidate_run vs _baseline_run: **ADOPT**

_RMSE delta -0.008 (95% CI [-0.016, -0.000]); Spearman lo_95 -0.0008 > floor -0.020_

- n_paired: 5273; n_bootstrap: 1000
- RMSE delta: -0.0077 (95% CI [-0.0157, -0.0001])
- Spearman delta: +0.0004 (95% CI [-0.0008, +0.0016])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      1315         -0.023261      -0.043632      -0.003424              0.001692          -0.001240           0.004569
 2022      1331         -0.008384      -0.022102       0.005205             -0.000021          -0.002448           0.002505
 2023      1311          0.005253      -0.007990       0.020255              0.000131          -0.002163           0.002282
 2024      1316         -0.003284      -0.013119       0.006123             -0.000275          -0.001916           0.001426

Wrote CSV: reports\adoption_gate_weather_lightgbm-nb_RB.csv

=== (baseline, WR) ===
# Adoption gate report — _candidate_run vs _baseline_run

Run: baseline=`data\backtest\run_baseline_baseline` candidate=`data\backtest\run_candidate_baseline`
n_bootstrap: 1000, seed: 42

### WR — _candidate_run vs _baseline_run: **DO_NOT_ADOPT**

_RMSE inconclusive: 95% CI [-0.011, +0.006] brackets / exceeds zero_

- n_paired: 8460; n_bootstrap: 1000
- RMSE delta: -0.0026 (95% CI [-0.0106, +0.0061])
- Spearman delta: -0.0005 (95% CI [-0.0020, +0.0010])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      2109         -0.012897      -0.031820       0.006509              0.000569          -0.003358           0.004410
 2022      2102          0.009760      -0.006908       0.026085             -0.002278          -0.005598           0.001026
 2023      2201         -0.000140      -0.015885       0.019505             -0.000386          -0.003287           0.002344
 2024      2048         -0.007212      -0.018039       0.006070              0.000268          -0.002449           0.002908

Wrote CSV: reports\adoption_gate_weather_baseline_WR.csv

=== (lightgbm-nb, WR) ===
# Adoption gate report — _candidate_run vs _baseline_run

Run: baseline=`data\backtest\run_baseline_lightgbm-nb` candidate=`data\backtest\run_candidate_lightgbm-nb`
n_bootstrap: 1000, seed: 42

### WR — _candidate_run vs _baseline_run: **ADOPT**

_RMSE delta -0.010 (95% CI [-0.017, -0.004]); Spearman lo_95 +0.0008 > floor -0.020_

- n_paired: 8460; n_bootstrap: 1000
- RMSE delta: -0.0104 (95% CI [-0.0165, -0.0042])
- Spearman delta: +0.0021 (95% CI [+0.0008, +0.0034])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      2109         -0.014728      -0.027114      -0.002211              0.002940           0.000087           0.005659
 2022      2102         -0.000893      -0.015981       0.013444             -0.000244          -0.003292           0.002838
 2023      2201         -0.009464      -0.020354       0.000754              0.002477           0.000153           0.004786
 2024      2048         -0.016428      -0.028537      -0.004745              0.003277           0.000665           0.005924

Wrote CSV: reports\adoption_gate_weather_lightgbm-nb_WR.csv
