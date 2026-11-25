# GridGuard UK — Error Analysis & Conformal Diagnostics Report

Generated: `2026-11-25 01:25:15 UTC`

## 1. Executive Summary (Out-of-Fold Spatial CV)

- **Evaluated Properties:** `970`
- **Spatial CV RMSE:** `£130.89/yr`
- **Spatial CV MAE:** `£87.18/yr`
- **Mean Residual Bias (`y_true - y_pred`):** `£+8.31/yr`
- **Empirical 90% Conformal Coverage:** `93.71%` (calibration gap vs 90.0% target: `+3.71%`)
- **Mean 90% Prediction Interval Width:** `£423.52/yr`

## 2. Global SHAP Feature Attributions

| feature                   |   mean_abs_shap_gbp |   mean_signed_shap_gbp |
|:--------------------------|--------------------:|-----------------------:|
| total_floor_area          |             304.779 |                  9.268 |
| energy_consumption_kwh_m2 |             289.341 |                  5.822 |
| main_heat_fuel            |             127.161 |                  5.676 |
| imd_score                 |               5.544 |                 -0.61  |
| current_energy_efficiency |               5.213 |                 -0.67  |
| roof_insulation_mm        |               5.192 |                  0.072 |
| tenure                    |               3.732 |                 -0.291 |
| construction_age_band     |               3.647 |                  0.099 |
| built_form                |               2.453 |                  0.103 |
| wall_type                 |               2.104 |                  0.582 |

## 3. Slice-Based Error, Residual Bias & Conformal Coverage

### EPC Energy Rating Band (`current_energy_rating`)

| current_energy_rating   |   n_samples |   rmse_gbp |   mae_gbp |   residual_bias_gbp |   conformal_coverage_90 |   mean_interval_width_gbp |
|:------------------------|------------:|-----------:|----------:|--------------------:|------------------------:|--------------------------:|
| G                       |           1 |     881.41 |    881.41 |              881.41 |                  0      |                    527.89 |
| F                       |          16 |     267.55 |    240.71 |              -56.93 |                  0.625  |                    543.93 |
| E                       |         231 |     177.31 |    102.45 |               23.97 |                  0.9004 |                    435.72 |
| B                       |          75 |     107.54 |     74.97 |              -10.29 |                  0.9467 |                    417.29 |
| C                       |         228 |     103.82 |     80.68 |                8.63 |                  0.9518 |                    415.87 |
| D                       |         406 |      99.47 |     77.26 |                4    |                  0.9631 |                    417.05 |
| A                       |          13 |      84.55 |     59.82 |              -20.68 |                  0.9231 |                    423.01 |

### Wall Construction Type (`wall_type`)

| wall_type              |   n_samples |   rmse_gbp |   mae_gbp |   residual_bias_gbp |   conformal_coverage_90 |   mean_interval_width_gbp |
|:-----------------------|------------:|-----------:|----------:|--------------------:|------------------------:|--------------------------:|
| Solid brick            |         211 |     190.79 |    107.77 |               21.37 |                  0.9147 |                    429.35 |
| System built           |          68 |     114.16 |     88.2  |                2.82 |                  0.9412 |                    426.46 |
| Cavity wall (unfilled) |         305 |     112.88 |     83.06 |                8.35 |                  0.941  |                    425.83 |
| Cavity wall (filled)   |         353 |     105.86 |     80.1  |                3.11 |                  0.9405 |                    417.34 |
| Timber frame           |          33 |      79.76 |     67.18 |               -8.64 |                  1      |                    424.99 |

### Main Heating Fuel (`main_heat_fuel`)

| main_heat_fuel   |   n_samples |   rmse_gbp |   mae_gbp |   residual_bias_gbp |   conformal_coverage_90 |   mean_interval_width_gbp |
|:-----------------|------------:|-----------:|----------:|--------------------:|------------------------:|--------------------------:|
| Electricity      |         143 |     228.75 |    130.5  |               77.45 |                  0.8322 |                    450.68 |
| LPG              |          28 |     153.58 |    126.22 |               90.9  |                  0.8929 |                    436.54 |
| Heat network     |          25 |     152.96 |    125.06 |              -85.32 |                  0.84   |                    453.72 |
| Oil              |          34 |     141.67 |    109.22 |               59.5  |                  0.8529 |                    436    |
| Mains gas        |         740 |      98.7  |     75.04 |               -7.37 |                  0.9662 |                    416.19 |

### Local Authority (Spatial Fold) (`local_authority`)

| local_authority     |   n_samples |   rmse_gbp |   mae_gbp |   residual_bias_gbp |   conformal_coverage_90 |   mean_interval_width_gbp |
|:--------------------|------------:|-----------:|----------:|--------------------:|------------------------:|--------------------------:|
| Liverpool           |         122 |     187.17 |     86.01 |               27.82 |                  0.9672 |                    437.06 |
| Bristol             |         127 |     136.71 |     98.54 |              -11.97 |                  0.8976 |                    424.13 |
| Leeds               |         123 |     130.29 |     89.6  |               16.53 |                  0.9431 |                    409.76 |
| Birmingham          |         125 |     123.19 |     90.15 |                6.59 |                  0.912  |                    424.28 |
| Newcastle upon Tyne |         117 |     122.57 |     96.08 |                2.54 |                  0.9316 |                    413.48 |
| Manchester          |         120 |     119.71 |     83    |               14.44 |                  0.95   |                    435.66 |
| Tower Hamlets       |         107 |     110.86 |     80.7  |                7.04 |                  0.9346 |                    417.31 |
| Sheffield           |         129 |      96.24 |     73.11 |                4.23 |                  0.9612 |                    425.47 |

### ONS IMD Deprivation Band (`imd_deprivation_band`)

| imd_deprivation_band          |   n_samples |   rmse_gbp |   mae_gbp |   residual_bias_gbp |   conformal_coverage_90 |   mean_interval_width_gbp |
|:------------------------------|------------:|-----------:|----------:|--------------------:|------------------------:|--------------------------:|
| Deciles 4-7 (Mid Deprivation) |         518 |     138.82 |     87.74 |               10.6  |                  0.9363 |                    421.9  |
| Deciles 8-10 (Least Deprived) |         109 |     129.7  |     99.37 |               10.84 |                  0.9083 |                    427.07 |
| Deciles 1-3 (Most Deprived)   |         343 |     118.33 |     82.47 |                4.05 |                  0.9475 |                    424.84 |

## 4. Diagnostic Findings & Mitigation Recommendations

### Under-Covered Slices (< 85% Empirical Coverage)
- `current_energy_rating=G` coverage `0.00%` (<85% alert, n=1)
- `current_energy_rating=F` coverage `62.50%` (<85% alert, n=16)
- `main_heat_fuel=Electricity` coverage `83.22%` (<85% alert, n=143)
- `main_heat_fuel=Heat network` coverage `84.00%` (<85% alert, n=25)

### Elevated Residual Bias Slices (|Bias| > £45/yr)
- `current_energy_rating=G` residual bias `£+881.41/yr` (RMSE £881.41)
- `current_energy_rating=F` residual bias `£-56.93/yr` (RMSE £267.55)
- `main_heat_fuel=Electricity` residual bias `£+77.45/yr` (RMSE £228.75)
- `main_heat_fuel=LPG` residual bias `£+90.90/yr` (RMSE £153.58)
- `main_heat_fuel=Heat network` residual bias `£-85.32/yr` (RMSE £152.96)
- `main_heat_fuel=Oil` residual bias `£+59.50/yr` (RMSE £141.67)

### Recommended Next Iterations
- Apply Mondrian (group-conditional) conformal calibration across `main_heat_fuel` or `current_energy_rating` if heteroscedastic residual widths diverge across off-gas-grid homes.
- Enrich solid-wall and pre-1919 stock with local climate degree-days and PAS 2035 ventilation-risk flags to reduce tail RMSE.
