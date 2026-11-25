# GridGuard UK — Error Analysis & Conformal Diagnostics Report

Generated: `2026-11-25 09:59:01 UTC`

## 1. Executive Summary (Out-of-Fold Spatial CV)

- **Evaluated Properties:** `500`
- **Spatial CV RMSE:** `£168.78/yr`
- **Spatial CV MAE:** `£116.56/yr`
- **Mean Residual Bias (`y_true - y_pred`):** `£+9.36/yr`
- **Empirical 90% Conformal Coverage:** `95.20%` (calibration gap vs 90.0% target: `+5.20%`)
- **Mean 90% Prediction Interval Width:** `£620.45/yr`

## 2. Global SHAP Feature Attributions

| feature                   |   mean_abs_shap_gbp |   mean_signed_shap_gbp |
|:--------------------------|--------------------:|-----------------------:|
| total_floor_area          |             324.063 |                 -2.583 |
| energy_consumption_kwh_m2 |             257.399 |                  6.714 |
| main_heat_fuel            |             126.409 |                 -7.903 |
| current_energy_efficiency |              19.101 |                 -1.018 |
| imd_score                 |               8.305 |                  0.387 |
| roof_insulation_mm        |               5.788 |                 -0.218 |
| property_type             |               3.112 |                  0.483 |
| wall_type                 |               3.021 |                  0.145 |
| tenure                    |               2.728 |                 -0.079 |
| built_form                |               2.57  |                  0.115 |

## 3. Slice-Based Error, Residual Bias & Conformal Coverage

### EPC Energy Rating Band (`current_energy_rating`)

| current_energy_rating   |   n_samples |   rmse_gbp |   mae_gbp |   residual_bias_gbp |   conformal_coverage_90 |   mean_interval_width_gbp |
|:------------------------|------------:|-----------:|----------:|--------------------:|------------------------:|--------------------------:|
| A                       |           1 |     857.49 |    857.49 |             -857.49 |                  0      |                   1121.19 |
| F                       |          17 |     243.85 |    185.61 |              107.92 |                  0.7647 |                    660.25 |
| B                       |          20 |     209.66 |    157.57 |              -70.41 |                  0.9    |                    626.82 |
| E                       |         156 |     209.63 |    138.54 |               34.39 |                  0.9295 |                    638.27 |
| C                       |          87 |     125.03 |     97.06 |               -6.12 |                  0.977  |                    623.01 |
| D                       |         219 |     124.66 |     96.16 |                1.27 |                  0.9817 |                    600.79 |

### Wall Construction Type (`wall_type`)

| wall_type              |   n_samples |   rmse_gbp |   mae_gbp |   residual_bias_gbp |   conformal_coverage_90 |   mean_interval_width_gbp |
|:-----------------------|------------:|-----------:|----------:|--------------------:|------------------------:|--------------------------:|
| Timber frame           |          14 |     210.69 |    139.67 |               18.54 |                  0.9286 |                    607.38 |
| Cavity wall (unfilled) |         162 |     183.08 |    120.3  |               30.19 |                  0.9506 |                    624.76 |
| Cavity wall (filled)   |         147 |     175.09 |    117.79 |               -5    |                  0.9524 |                    622.97 |
| Solid brick            |         140 |     148.01 |    111.56 |                4.48 |                  0.95   |                    612.02 |
| System built           |          37 |     129.33 |    105.48 |               -9.81 |                  0.973  |                    628.5  |

### Main Heating Fuel (`main_heat_fuel`)

| main_heat_fuel   |   n_samples |   rmse_gbp |   mae_gbp |   residual_bias_gbp |   conformal_coverage_90 |   mean_interval_width_gbp |
|:-----------------|------------:|-----------:|----------:|--------------------:|------------------------:|--------------------------:|
| Electricity      |          93 |     259.02 |    172.02 |              107.71 |                  0.8602 |                    646.83 |
| Oil              |          24 |     170.85 |    152.36 |              128.13 |                  1      |                    597.46 |
| Heat network     |          25 |     154.09 |    121.38 |              -77.38 |                  0.96   |                    611.9  |
| Mains gas        |         358 |     136.91 |     99.41 |              -18.1  |                  0.9721 |                    615.74 |

### Local Authority (Spatial Fold) (`local_authority`)

| local_authority   |   n_samples |   rmse_gbp |   mae_gbp |   residual_bias_gbp |   conformal_coverage_90 |   mean_interval_width_gbp |
|:------------------|------------:|-----------:|----------:|--------------------:|------------------------:|--------------------------:|
| Sefton            |         252 |     183.1  |    122.5  |               16.85 |                  0.9444 |                    652.01 |
| Manchester        |         100 |     167.92 |    116.32 |              -19.04 |                  0.94   |                    609.43 |
| Leeds             |         100 |     157.5  |    117.36 |               22.41 |                  0.96   |                    609.57 |
| Knowsley          |          19 |     104.01 |     88.15 |               -2.35 |                  1      |                    494.99 |
| Liverpool         |          29 |      99.98 |     81.61 |                4.87 |                  1      |                    503.96 |

### ONS IMD Deprivation Band (`imd_deprivation_band`)

| imd_deprivation_band          |   n_samples |   rmse_gbp |   mae_gbp |   residual_bias_gbp |   conformal_coverage_90 |   mean_interval_width_gbp |
|:------------------------------|------------:|-----------:|----------:|--------------------:|------------------------:|--------------------------:|
| Deciles 1-3 (Most Deprived)   |         149 |     175.23 |    114.06 |               22.74 |                  0.9463 |                    625.19 |
| Deciles 4-7 (Mid Deprivation) |         292 |     173.31 |    120.85 |                5.39 |                  0.9452 |                    619.94 |
| Deciles 8-10 (Least Deprived) |          59 |     123.4  |    101.63 |               -4.81 |                  1      |                    611.05 |

## 4. Diagnostic Findings & Mitigation Recommendations

### Under-Covered Slices (< 85% Empirical Coverage)
- `current_energy_rating=A` coverage `0.00%` (<85% alert, n=1)
- `current_energy_rating=F` coverage `76.47%` (<85% alert, n=17)

### Elevated Residual Bias Slices (|Bias| > £45/yr)
- `current_energy_rating=A` residual bias `£-857.49/yr` (RMSE £857.49)
- `current_energy_rating=F` residual bias `£+107.92/yr` (RMSE £243.85)
- `current_energy_rating=B` residual bias `£-70.41/yr` (RMSE £209.66)
- `main_heat_fuel=Electricity` residual bias `£+107.71/yr` (RMSE £259.02)
- `main_heat_fuel=Oil` residual bias `£+128.13/yr` (RMSE £170.85)
- `main_heat_fuel=Heat network` residual bias `£-77.38/yr` (RMSE £154.09)

### Recommended Next Iterations
- Apply Mondrian (group-conditional) conformal calibration across `main_heat_fuel` or `current_energy_rating` if heteroscedastic residual widths diverge across off-gas-grid homes.
- Enrich solid-wall and pre-1919 stock with local climate degree-days and PAS 2035 ventilation-risk flags to reduce tail RMSE.
