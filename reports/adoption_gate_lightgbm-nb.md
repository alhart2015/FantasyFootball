# Adoption gate report — lightgbm-nb vs baseline

Run: `data\backtest\run_20260429T003552Z`
n_bootstrap: 1000, seed: 42

### QB — lightgbm-nb vs baseline: **ADOPT**

_RMSE delta -0.193 (95% CI [-0.272, -0.110]); Spearman lo_95 +0.0045 > floor -0.020_

- n_paired: 2676; n_bootstrap: 1000
- RMSE delta: -0.1933 (95% CI [-0.2719, -0.1102])
- Spearman delta: +0.0183 (95% CI [+0.0045, +0.0313])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021       665         -0.269079      -0.447331      -0.092868              0.036484           0.007064           0.066690
 2022       657         -0.162376      -0.331445       0.006448              0.014574          -0.013246           0.041390
 2023       670         -0.093997      -0.256938       0.074353             -0.000931          -0.026835           0.022847
 2024       684         -0.239849      -0.374333      -0.109081              0.023237           0.001435           0.045511

### RB — lightgbm-nb vs baseline: **DO_NOT_ADOPT**

_RMSE inconclusive: 95% CI [+0.013, +0.074] brackets / exceeds zero_

- n_paired: 5273; n_bootstrap: 1000
- RMSE delta: +0.0420 (95% CI [+0.0133, +0.0740])
- Spearman delta: -0.0012 (95% CI [-0.0068, +0.0039])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      1315          0.080586       0.007029       0.147821             -0.017557          -0.030035          -0.004765
 2022      1331         -0.002804      -0.068155       0.066128              0.010451          -0.001244           0.022300
 2023      1311          0.060088       0.005082       0.113711              0.001046          -0.008451           0.010672
 2024      1316          0.029707      -0.018593       0.076839              0.001149          -0.007673           0.010598

### TE — lightgbm-nb vs baseline: **DO_NOT_ADOPT**

_RMSE inconclusive: 95% CI [-0.029, +0.042] brackets / exceeds zero_

- n_paired: 4257; n_bootstrap: 1000
- RMSE delta: +0.0028 (95% CI [-0.0289, +0.0422])
- Spearman delta: +0.0071 (95% CI [-0.0014, +0.0160])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      1030         -0.022779      -0.099729       0.051112              0.021525          -0.001096           0.041925
 2022      1088         -0.040690      -0.102641       0.019983              0.000260          -0.017721           0.019843
 2023      1058         -0.010856      -0.067723       0.048080              0.010920          -0.004147           0.027467
 2024      1081          0.085848       0.016062       0.169487             -0.004415          -0.019683           0.010081

### WR — lightgbm-nb vs baseline: **DO_NOT_ADOPT**

_RMSE inconclusive: 95% CI [-0.032, +0.029] brackets / exceeds zero_

- n_paired: 8460; n_bootstrap: 1000
- RMSE delta: -0.0016 (95% CI [-0.0316, +0.0291])
- Spearman delta: +0.0027 (95% CI [-0.0032, +0.0080])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      2109         -0.008906      -0.066207       0.050199              0.008682          -0.002147           0.019424
 2022      2102         -0.006631      -0.062995       0.050846              0.004292          -0.006323           0.015059
 2023      2201          0.022152      -0.040928       0.086283             -0.002381          -0.014473           0.009177
 2024      2048         -0.013901      -0.069005       0.041349              0.000072          -0.010979           0.010330

Wrote CSV: reports\adoption_gate_lightgbm-nb.csv
