# GridGuard UK — 300-Skill Engineering & Applied Science Matrix

This matrix maps **300 concrete capabilities** across six core disciplines directly to the implemented modules, classes, and functions in `gridguard-uk`.

---

## Pillar 1: Data Engineering & UK Open Data Ingestion (Skills 001–050)

| # | Skill / Capability | Implementation File | Symbol / Function |
|---|---|---|---|
| 001 | Declarative Pandera dataframe contract definition | `src/ingestion/epc_pipeline.py` | `EPCRecordSchema` |
| 002 | Strict UK UPRN regex format enforcement (`[0-9]{6,12}`) | `src/ingestion/epc_pipeline.py` | `EPCRecordSchema` |
| 003 | ONS LSOA code regex validation (`^[EWS]01[0-9]{6}$`) | `src/ingestion/epc_pipeline.py` | `EPCRecordSchema` |
| 004 | Categorical domain allow-list checks for EPC ratings A–G | `src/ingestion/epc_pipeline.py` | `VALID_EPC_RATINGS` |
| 005 | Categorical domain checks for UK property types | `src/ingestion/epc_pipeline.py` | `VALID_PROPERTY_TYPES` |
| 006 | Categorical domain checks for built forms | `src/ingestion/epc_pipeline.py` | `VALID_BUILT_FORMS` |
| 007 | Categorical domain checks for residential tenure types | `src/ingestion/epc_pipeline.py` | `VALID_TENURES` |
| 008 | Categorical domain checks for main heating fuels | `src/ingestion/epc_pipeline.py` | `VALID_HEAT_FUELS` |
| 009 | Categorical domain checks for wall construction types | `src/ingestion/epc_pipeline.py` | `VALID_WALL_TYPES` |
| 010 | Physical range bounds on floor area (`12–1000 m²`) | `src/ingestion/epc_pipeline.py` | `EPCRecordSchema` |
| 011 | SAP efficiency score bounds validation (`1–100`) | `src/ingestion/epc_pipeline.py` | `EPCRecordSchema` |
| 012 | Cross-column invariant (`potential >= current` SAP) | `src/ingestion/epc_pipeline.py` | `EPCRecordSchema` |
| 013 | Automatic type coercion during contract validation | `src/ingestion/epc_pipeline.py` | `EPCRecordSchema` |
| 014 | Non-strict schema passthrough for spatial enrichment metadata | `src/ingestion/epc_pipeline.py` | `EPCRecordSchema` |
| 015 | Lazy multi-error Pandera failure collection | `src/ingestion/epc_pipeline.py` | `validate_and_quarantine` |
| 016 | Row-level failure reason extraction from `failure_cases` | `src/ingestion/epc_pipeline.py` | `_extract_row_reasons` |
| 017 | Row-by-row coercion fallback isolating mixed corrupt types | `src/ingestion/epc_pipeline.py` | `_row_by_row_fallback` |
| 018 | Iterative secondary-error quarantine loop | `src/ingestion/epc_pipeline.py` | `validate_and_quarantine` |
| 019 | Parquet Dead-Letter Quarantine serialization | `src/ingestion/epc_pipeline.py` | `_write_quarantine_parquet` |
| 020 | Mixed-type object stringification for safe Parquet writes | `src/ingestion/epc_pipeline.py` | `_write_quarantine_parquet` |
| 021 | UTC ISO-8601 quarantine audit timestamping | `src/ingestion/epc_pipeline.py` | `validate_and_quarantine` |
| 022 | Missing required column guard before validation | `src/ingestion/epc_pipeline.py` | `validate_and_quarantine` |
| 023 | UK GDPR salt resolution via `GRIDGUARD_SALT` env var | `src/ingestion/epc_pipeline.py` | `get_gdpr_salt` |
| 024 | Irreversible HMAC-SHA256 UPRN pseudonymisation | `src/ingestion/epc_pipeline.py` | `hash_uprn` |
| 025 | Raw PII column stripping (`uprn` -> `uprn_hash`) | `src/ingestion/epc_pipeline.py` | `anonymise_epc_dataframe` |
| 026 | Canonical column ordering with `uprn_hash` first | `src/ingestion/epc_pipeline.py` | `anonymise_epc_dataframe` |
| 027 | Embedded DuckDB analytical table persistence | `src/ingestion/epc_pipeline.py` | `write_epc_to_duckdb` |
| 028 | Read-only DuckDB analytical dataframe extraction | `src/ingestion/epc_pipeline.py` | `load_epc_from_duckdb` |
| 029 | Typed ingestion summary dataclass (`IngestionResult`) | `src/ingestion/epc_pipeline.py` | `IngestionResult` |
| 030 | Atomic end-to-end ingestion orchestration | `src/ingestion/epc_pipeline.py` | `ingest_epc_data` |
| 031 | Live UK Postcodes.io REST client integration | `src/ingestion/live_uk_connectors.py` | `_query_postcodes_io` |
| 032 | Live DESNZ EPC Open Data API integration | `src/ingestion/live_uk_connectors.py` | `_query_desnz_epc_api` |
| 033 | HTTP Basic Auth token normalization (`email:key` or base64) | `src/ingestion/live_uk_connectors.py` | `_query_desnz_epc_api` |
| 034 | Exponential backoff retry policy on HTTP errors (`tenacity`) | `src/ingestion/live_uk_connectors.py` | `_query_postcodes_io` |
| 035 | Exponential backoff retry on DESNZ EPC API timeouts | `src/ingestion/live_uk_connectors.py` | `_query_desnz_epc_api` |
| 036 | Postcodes.io JSON payload normalization (LSOA, GSS, lat/lon) | `src/ingestion/live_uk_connectors.py` | `_normalise_postcodes_results` |
| 037 | Multi-prefix spatial diversity widening for Postcodes.io | `src/ingestion/live_uk_connectors.py` | `_collect_live_locations` |
| 038 | `LIVE_FEED_DEGRADED` structured warning emission | `src/ingestion/live_uk_connectors.py` | `_collect_live_locations` |
| 039 | Offline curated Liverpool/England LSOA+GSS fallback | `src/ingestion/live_uk_connectors.py` | `FALLBACK_UK_LOCATIONS` |
| 040 | DESNZ wall description taxonomy mapping | `src/ingestion/live_uk_connectors.py` | `_map_wall_type` |
| 041 | DESNZ main heating fuel taxonomy mapping | `src/ingestion/live_uk_connectors.py` | `_map_heat_fuel` |
| 042 | DESNZ property type taxonomy normalization | `src/ingestion/live_uk_connectors.py` | `_map_property_type` |
| 043 | DESNZ built-form taxonomy normalization | `src/ingestion/live_uk_connectors.py` | `_map_built_form` |
| 044 | DESNZ tenure classification mapping | `src/ingestion/live_uk_connectors.py` | `_map_tenure` |
| 045 | Calibrated real-LSOA housing profile synthesis | `src/ingestion/live_uk_connectors.py` | `_build_calibrated_epc_rows` |
| 046 | Live ingestion contract enforcement via quarantine | `src/ingestion/live_uk_connectors.py` | `fetch_live_ons_and_epc_data` |
| 047 | SAP-to-EPC band A–G deterministic conversion | `src/ingestion/live_uk_connectors.py` | `_sap_to_rating` |
| 048 | Synthetic UK DESNZ EPC + ONS IMD dataset generator | `src/run_pipeline.py` | `generate_synthetic_uk_epc_dataset` |
| 049 | Controlled 3% multi-mode corruption injection for QA | `src/run_pipeline.py` | `generate_synthetic_uk_epc_dataset` |
| 050 | CLI flag orchestration (`--live` / `--no-live`, `--limit`) | `src/run_pipeline.py` | `_parse_args` |

---

## Pillar 2: Geospatial Econometrics & Fuel Poverty Analytics (Skills 051–100)

| # | Skill / Capability | Implementation File | Symbol / Function |
|---|---|---|---|
| 051 | ONS LSOA-level spatial aggregation | `src/analytics/spatial_uplift.py` | `compute_lsoa_fuel_poverty_priority` |
| 052 | Local Authority district grouping alongside LSOA | `src/analytics/spatial_uplift.py` | `compute_lsoa_fuel_poverty_priority` |
| 053 | EPC Band D–G low-efficiency stock share calculation | `src/analytics/spatial_uplift.py` | `compute_lsoa_fuel_poverty_priority` |
| 054 | DESNZ £31,000 low-income household threshold indicator | `src/analytics/spatial_uplift.py` | `compute_lsoa_fuel_poverty_priority` |
| 055 | DESNZ LILEE (Low Income Low Energy Efficiency) proxy share | `src/analytics/spatial_uplift.py` | `compute_lsoa_fuel_poverty_priority` |
| 056 | Mean LSOA SAP energy efficiency computation | `src/analytics/spatial_uplift.py` | `compute_lsoa_fuel_poverty_priority` |
| 057 | Mean LSOA annual heating cost burden calculation | `src/analytics/spatial_uplift.py` | `compute_lsoa_fuel_poverty_priority` |
| 058 | ONS IMD decile inversion to deprivation intensity | `src/analytics/spatial_uplift.py` | `compute_lsoa_fuel_poverty_priority` |
| 059 | Mean LSOA IMD continuous score aggregation | `src/analytics/spatial_uplift.py` | `compute_lsoa_fuel_poverty_priority` |
| 060 | Capped heating-cost burden index normalization (`£3,500/yr`) | `src/analytics/spatial_uplift.py` | `compute_lsoa_fuel_poverty_priority` |
| 061 | Composite 0–100 Fuel Poverty Priority weighted index | `src/analytics/spatial_uplift.py` | `compute_lsoa_fuel_poverty_priority` |
| 062 | Four-tier spatial targeting classification (`Critical`–`Low`) | `src/analytics/spatial_uplift.py` | `compute_lsoa_fuel_poverty_priority` |
| 063 | Priority-ranked LSOA league table sorting | `src/analytics/spatial_uplift.py` | `compute_lsoa_fuel_poverty_priority` |
| 064 | Empty-dataframe schema-preserving spatial guard | `src/analytics/spatial_uplift.py` | `compute_lsoa_fuel_poverty_priority` |
| 065 | Property count weighting per LSOA catchment | `src/analytics/spatial_uplift.py` | `compute_lsoa_fuel_poverty_priority` |
| 066 | LSOA mean annual bill savings aggregation | `src/analytics/spatial_uplift.py` | `estimate_spatial_uplift` |
| 067 | LSOA median annual bill savings aggregation | `src/analytics/spatial_uplift.py` | `estimate_spatial_uplift` |
| 068 | Total aggregate LSOA decarbonisation bill savings (£/yr) | `src/analytics/spatial_uplift.py` | `estimate_spatial_uplift` |
| 069 | Priority-to-uplift spatial join (`lsoa_code`) | `src/analytics/spatial_uplift.py` | `estimate_spatial_uplift` |
| 070 | DuckDB persistence of `lsoa_priority_uplift` mart | `src/analytics/spatial_uplift.py` | `estimate_spatial_uplift` |
| 071 | DuckDB persistence of `property_uplift` mart | `src/analytics/spatial_uplift.py` | `estimate_spatial_uplift` |
| 072 | Postcode-to-LSOA11/LSOA21 ONS code resolution | `src/ingestion/live_uk_connectors.py` | `_normalise_postcodes_results` |
| 073 | ONS Local Authority GSS code resolution (`E08000012`, etc.) | `src/ingestion/live_uk_connectors.py` | `_normalise_postcodes_results` |
| 074 | WGS84 latitude/longitude coordinate extraction | `src/ingestion/live_uk_connectors.py` | `_normalise_postcodes_results` |
| 075 | Country prefix filtering (`E01`, `W01`, `S01` LSOA/DZ) | `src/ingestion/live_uk_connectors.py` | `_normalise_postcodes_results` |
| 076 | Spatial fallback guaranteeing `>= 2` Local Authorities | `src/ingestion/live_uk_connectors.py` | `_collect_live_locations` |
| 077 | LSOA-anchored deterministic deprivation calibration | `src/ingestion/live_uk_connectors.py` | `_build_calibrated_epc_rows` |
| 078 | Postcode-keyed join between EPC rows and Postcodes.io | `src/ingestion/live_uk_connectors.py` | `_build_calibrated_epc_rows` |
| 079 | Regional Local Authority spatial priors (8 English cities) | `src/run_pipeline.py` | `LOCAL_AUTHORITIES` |
| 080 | Deprivation-correlated household income simulation | `src/run_pipeline.py` | `generate_synthetic_uk_epc_dataset` |
| 081 | Building age-band vs wall-construction spatial correlation | `src/run_pipeline.py` | `generate_synthetic_uk_epc_dataset` |
| 082 | Building age-band vs loft-insulation depth modelling | `src/run_pipeline.py` | `generate_synthetic_uk_epc_dataset` |
| 083 | Fuel-type price multiplier calibration (Gas vs Electricity/Oil/LPG) | `src/run_pipeline.py` | `generate_synthetic_uk_epc_dataset` |
| 084 | Built-form vs floor-area spatial stock parameterization | `src/run_pipeline.py` | `generate_synthetic_uk_epc_dataset` |
| 085 | Spatial GroupKFold stratification by Local Authority | `src/models/train_conformal.py` | `train_conformal_model` |
| 086 | Unseen council out-of-fold spatial generalization testing | `src/models/train_conformal.py` | `train_conformal_model` |
| 087 | Spatial fold assignment tracking (`cv_fold`) | `src/models/train_conformal.py` | `train_conformal_model` |
| 088 | Local Authority slice error auditing | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 089 | IMD deprivation tertile equity slicing (Deciles 1–3, 4–7, 8–10) | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 090 | Off-gas-grid fuel slice residual diagnostics | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 091 | Solid-wall vs cavity-wall housing slice evaluation | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 092 | EPC Band A–G performance stratification | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 093 | Multi-authority dashboard spatial filtering | `src/app/dashboard.py` | `load_tables` |
| 094 | Priority-tier interactive LSOA drilldown | `src/app/dashboard.py` | `load_tables` |
| 095 | Cached DuckDB spatial mart loading in Streamlit | `src/app/dashboard.py` | `load_tables` |
| 096 | Top-15 LSOA priority vs savings visual comparison | `src/app/dashboard.py` | `load_tables` |
| 097 | Distinct LA and LSOA telemetry logging in pipeline runner | `src/run_pipeline.py` | `run_end_to_end` |
| 098 | Regional postcode prefix parameterization (`--postcode-prefix`) | `src/run_pipeline.py` | `_parse_args` |
| 099 | Live vs offline spatial source provenance tagging (`data_source`) | `src/ingestion/live_uk_connectors.py` | `fetch_live_ons_and_epc_data` |
| 100 | Spatial `GroupShuffleSplit` fallback for small authority counts | `src/models/train_conformal.py` | `_fit_mapie_and_predict` |

---

## Pillar 3: Conformal ML, Spatial CV & Error Diagnostics (Skills 101–150)

| # | Skill / Capability | Implementation File | Symbol / Function |
|---|---|---|---|
| 101 | Mixed numeric + categorical scikit-learn `ColumnTransformer` | `src/models/train_conformal.py` | `_build_preprocessor` |
| 102 | Unseen-category safe `OneHotEncoder(handle_unknown="ignore")` | `src/models/train_conformal.py` | `_build_preprocessor` |
| 103 | Leakage-free feature selection excluding bill/potential SAP | `src/models/train_conformal.py` | `MODEL_FEATURES` |
| 104 | `LGBMRegressor` gradient-boosted tabular modelling | `src/models/train_conformal.py` | `_fit_mapie_and_predict` |
| 105 | Local Authority `GroupKFold` outer spatial CV loop | `src/models/train_conformal.py` | `train_conformal_model` |
| 106 | Inner `GroupKFold` spatial conformal calibration | `src/models/train_conformal.py` | `_fit_mapie_and_predict` |
| 107 | MAPIE 1.x `CrossConformalRegressor` (`method="plus"`) | `src/models/train_conformal.py` | `_fit_mapie_and_predict` |
| 108 | MAPIE 1.x `SplitConformalRegressor` fallback for `<3` groups | `src/models/train_conformal.py` | `_fit_mapie_and_predict` |
| 109 | Backward-compatible MAPIE 0.9.x `MapieRegressor` adapter | `src/models/train_conformal.py` | `_fit_mapie_and_predict` |
| 110 | 90% nominal conformal prediction interval (`alpha=0.10`) | `src/models/train_conformal.py` | `train_conformal_model` |
| 111 | Non-negative heating cost lower-bound clamping | `src/models/train_conformal.py` | `_fit_mapie_and_predict` |
| 112 | Monotonic interval bound enforcement (`upper >= lower`) | `src/models/train_conformal.py` | `_fit_mapie_and_predict` |
| 113 | Out-of-fold point and interval prediction assembly | `src/models/train_conformal.py` | `train_conformal_model` |
| 114 | Per-row conformal interval width calculation | `src/models/train_conformal.py` | `train_conformal_model` |
| 115 | Per-row empirical 90% coverage indicator (`covered_90`) | `src/models/train_conformal.py` | `train_conformal_model` |
| 116 | Spatial CV RMSE (`root_mean_squared_error`) evaluation | `src/models/train_conformal.py` | `train_conformal_model` |
| 117 | Spatial CV MAE (`mean_absolute_error`) evaluation | `src/models/train_conformal.py` | `train_conformal_model` |
| 118 | Spatial CV coefficient of determination (`r2_score`) | `src/models/train_conformal.py` | `train_conformal_model` |
| 119 | Global empirical 90% conformal coverage evaluation | `src/models/train_conformal.py` | `train_conformal_model` |
| 120 | Mean conformal interval width diagnostic (£/yr) | `src/models/train_conformal.py` | `train_conformal_model` |
| 121 | Mean out-of-fold residual bias diagnostic (`y_true - y_pred`) | `src/models/train_conformal.py` | `train_conformal_model` |
| 122 | TreeSHAP exact attribution via `shap.TreeExplainer` | `src/models/train_conformal.py` | `compute_shap_importance` |
| 123 | One-hot SHAP attribution rollup to raw domain features | `src/models/train_conformal.py` | `compute_shap_importance` |
| 124 | Mean absolute SHAP global importance ranking | `src/models/train_conformal.py` | `compute_shap_importance` |
| 125 | Mean signed SHAP directional effect tracking | `src/models/train_conformal.py` | `compute_shap_importance` |
| 126 | Deterministic SHAP sample subsampling (`max_samples=400`) | `src/models/train_conformal.py` | `compute_shap_importance` |
| 127 | Full-dataset production conformal bundle refit | `src/models/train_conformal.py` | `train_conformal_model` |
| 128 | Unified bundle inference supporting MAPIE 0.9 & 1.x | `src/models/train_conformal.py` | `_predict_with_bundle` |
| 129 | Single-property point + 90% interval + SHAP serving | `src/models/train_conformal.py` | `predict_property_heating_cost` |
| 130 | Structured training result dataclass (`ConformalTrainingResult`) | `src/models/train_conformal.py` | `ConformalTrainingResult` |
| 131 | Minimum-groups spatial CV validation guard (`>= 2` LAs) | `src/models/train_conformal.py` | `train_conformal_model` |
| 132 | Dynamic `n_splits` capping to available Local Authorities | `src/models/train_conformal.py` | `train_conformal_model` |
| 133 | Slice-level RMSE calculation across housing strata | `src/evaluation/error_analysis.py` | `_compute_slice_metrics` |
| 134 | Slice-level MAE calculation across housing strata | `src/evaluation/error_analysis.py` | `_compute_slice_metrics` |
| 135 | Slice-level residual bias (`y_true - y_pred`) audit | `src/evaluation/error_analysis.py` | `_compute_slice_metrics` |
| 136 | Slice-level 90% conformal coverage audit | `src/evaluation/error_analysis.py` | `_compute_slice_metrics` |
| 137 | Slice-level mean interval width audit | `src/evaluation/error_analysis.py` | `_compute_slice_metrics` |
| 138 | Conformal calibration gap computation (`coverage - 0.90`) | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 139 | Automated under-coverage alert threshold (`< 85%`) | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 140 | Automated systematic bias alert threshold (`|bias| > £45/yr`) | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 141 | Markdown table rendering via `DataFrame.to_markdown` | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 142 | Automated `docs/ERROR_ANALYSIS_REPORT.md` artifact generation | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 143 | Top-10 SHAP importance embedding in error report | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 144 | Worst-slice descending RMSE prioritization | `src/evaluation/error_analysis.py` | `_compute_slice_metrics` |
| 145 | Missing OOF column defensive contract validation | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 146 | Typed error analysis output dataclass (`ErrorAnalysisResult`) | `src/evaluation/error_analysis.py` | `ErrorAnalysisResult` |
| 147 | Actionable Mondrian conformal & feature mitigation guidance | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 148 | In-UI error diagnostics report rendering in Streamlit | `src/app/dashboard.py` | `REPORT_PATH` |
| 149 | FastAPI live conformal inference integration | `src/app/main.py` | `advise_retrofit` |
| 150 | SAP-based analytical heating cost fallback interval | `src/app/main.py` | `_heuristic_heating_fallback` |

---

## Pillar 4: Causal Inference & Policy Uplift Modeling (Skills 151–200)

| # | Skill / Capability | Implementation File | Symbol / Function |
|---|---|---|---|
| 151 | Two-Model (T-Learner) meta-learner architecture | `src/analytics/spatial_uplift.py` | `TLearnerUpliftEstimator` |
| 152 | Separate control-arm regressor $\hat{\mu}_0(x)$ (`T=0`) | `src/analytics/spatial_uplift.py` | `TLearnerUpliftEstimator.fit` |
| 153 | Separate treated-arm regressor $\hat{\mu}_1(x)$ (`T=1`) | `src/analytics/spatial_uplift.py` | `TLearnerUpliftEstimator.fit` |
| 154 | CATE bill savings uplift estimation ($\hat{\mu}_0 - \hat{\mu}_1$) | `src/analytics/spatial_uplift.py` | `TLearnerUpliftEstimator.predict_outcomes` |
| 155 | Independent seed offset for control vs treated LightGBM learners | `src/analytics/spatial_uplift.py` | `TLearnerUpliftEstimator._make_pipeline` |
| 156 | Mixed-type preprocessing inside T-Learner pipelines | `src/analytics/spatial_uplift.py` | `_build_preprocessor` |
| 157 | Control vs treated positivity support guard (`empty` check) | `src/analytics/spatial_uplift.py` | `TLearnerUpliftEstimator.fit` |
| 158 | Fitted-state lifecycle guard (`is_fitted_`) | `src/analytics/spatial_uplift.py` | `TLearnerUpliftEstimator.predict_outcomes` |
| 159 | Counterfactual untreated bill prediction (`mu0_bill_gbp`) | `src/analytics/spatial_uplift.py` | `TLearnerUpliftEstimator.predict_outcomes` |
| 160 | Counterfactual retrofitted bill prediction (`mu1_bill_gbp`) | `src/analytics/spatial_uplift.py` | `TLearnerUpliftEstimator.predict_outcomes` |
| 161 | Vectorized NumPy uplift accessor (`predict_uplift`) | `src/analytics/spatial_uplift.py` | `TLearnerUpliftEstimator.predict_uplift` |
| 162 | Average Treatment Effect (ATE) savings summarization | `src/analytics/spatial_uplift.py` | `estimate_spatial_uplift` |
| 163 | Inclusion of `household_income_gbp` socio-economic confounder | `src/analytics/spatial_uplift.py` | `NUMERIC_FEATURES` |
| 164 | Inclusion of `imd_decile` & `imd_score` spatial confounders | `src/analytics/spatial_uplift.py` | `NUMERIC_FEATURES` |
| 165 | Inclusion of physical fabric confounders (`wall_type`, `roof_insulation_mm`) | `src/analytics/spatial_uplift.py` | `FEATURE_COLUMNS` |
| 166 | Inclusion of heating fuel & tenure confounders | `src/analytics/spatial_uplift.py` | `CATEGORICAL_FEATURES` |
| 167 | Confounded treatment propensity simulation (`sap < 60`, `imd <= 3`) | `src/run_pipeline.py` | `generate_synthetic_uk_epc_dataset` |
| 168 | Heterogeneous ground-truth savings CATE simulation | `src/run_pipeline.py` | `generate_synthetic_uk_epc_dataset` |
| 169 | Solid-wall vs cavity-wall uplift heterogeneity modelling | `src/run_pipeline.py` | `generate_synthetic_uk_epc_dataset` |
| 170 | Uninsulated-loft uplift heterogeneity modelling | `src/run_pipeline.py` | `generate_synthetic_uk_epc_dataset` |
| 171 | Floor-area proportional savings scaling in DGP | `src/run_pipeline.py` | `generate_synthetic_uk_epc_dataset` |
| 172 | Calibrated real-LSOA treatment & outcome generation | `src/ingestion/live_uk_connectors.py` | `_build_calibrated_epc_rows` |
| 173 | SAP-threshold treatment assignment in live DESNZ EPC mode | `src/ingestion/live_uk_connectors.py` | `_build_calibrated_epc_rows` |
| 174 | Structured causal output dataclass (`SpatialUpliftResult`) | `src/analytics/spatial_uplift.py` | `SpatialUpliftResult` |
| 175 | Property-level causal enrichment dataframe output | `src/analytics/spatial_uplift.py` | `estimate_spatial_uplift` |
| 176 | LSOA-level mean CATE savings aggregation | `src/analytics/spatial_uplift.py` | `estimate_spatial_uplift` |
| 177 | LSOA-level median CATE savings robustness statistic | `src/analytics/spatial_uplift.py` | `estimate_spatial_uplift` |
| 178 | LSOA-level total budget-impact savings sum | `src/analytics/spatial_uplift.py` | `estimate_spatial_uplift` |
| 179 | Combining fuel-poverty priority with causal savings ROI | `src/analytics/spatial_uplift.py` | `estimate_spatial_uplift` |
| 180 | Currency rounding hygiene across counterfactual outputs | `src/analytics/spatial_uplift.py` | `TLearnerUpliftEstimator.predict_outcomes` |
| 181 | Schema validation of treatment indicator (`isin([0, 1])`) | `src/ingestion/epc_pipeline.py` | `EPCRecordSchema` |
| 182 | Schema validation of observed outcome bill (`£50–£20,000`) | `src/ingestion/epc_pipeline.py` | `EPCRecordSchema` |
| 183 | Configurable treatment and outcome column names | `src/analytics/spatial_uplift.py` | `TLearnerUpliftEstimator.fit` |
| 184 | Configurable `n_estimators` regularization in T-Learner | `src/analytics/spatial_uplift.py` | `TLearnerUpliftEstimator.__init__` |
| 185 | `min_child_samples=10` leaf regularization for subgroup stability | `src/analytics/spatial_uplift.py` | `TLearnerUpliftEstimator._make_pipeline` |
| 186 | Index-aligned counterfactual output DataFrame construction | `src/analytics/spatial_uplift.py` | `TLearnerUpliftEstimator.predict_outcomes` |
| 187 | Dashboard KPI surface for mean T-Learner ATE savings | `src/app/dashboard.py` | `load_tables` |
| 188 | Visual juxtaposition of LSOA priority vs causal uplift | `src/app/dashboard.py` | `load_tables` |
| 189 | DuckDB causal auditability via persisted `property_uplift` table | `src/analytics/spatial_uplift.py` | `estimate_spatial_uplift` |
| 190 | DuckDB policy-targeting mart via `lsoa_priority_uplift` table | `src/analytics/spatial_uplift.py` | `estimate_spatial_uplift` |
| 191 | End-to-end causal stage invocation in runner | `src/run_pipeline.py` | `run_end_to_end` |
| 192 | Automated positivity assertion (`ate_savings_gbp > 0`) in tests | `tests/test_contracts_and_errors.py` | `test_spatial_uplift_and_conformal_training_and_error_analysis` |
| 193 | End-to-end live-mode T-Learner verification | `src/run_pipeline.py` | `run_end_to_end` |
| 194 | Unseen-category safe OneHotEncoder in counterfactual inference | `src/analytics/spatial_uplift.py` | `_build_preprocessor` |
| 195 | Deterministic random-state reproducibility in causal splits | `src/analytics/spatial_uplift.py` | `TLearnerUpliftEstimator.__init__` |
| 196 | Socio-spatial LILEE eligibility alignment with ECO4 Flex | `src/analytics/spatial_uplift.py` | `compute_lsoa_fuel_poverty_priority` |
| 197 | Clean separation between causal uplift (`spatial_uplift.py`) & cost forecasting (`train_conformal.py`) | `src/models/train_conformal.py` | `MODEL_FEATURES` |
| 198 | Leakage prevention (excluding `observed_annual_bill_gbp` from heating cost model) | `src/models/train_conformal.py` | `MODEL_FEATURES` |
| 199 | Leakage prevention (excluding `retrofit_treated` from baseline heating cost model) | `src/models/train_conformal.py` | `MODEL_FEATURES` |
| 200 | Quantitative bill-saving telemetry reporting in CLI summary | `src/run_pipeline.py` | `run_end_to_end` |

---

## Pillar 5: Guardrailed RAG, Statutory Parsing & LangGraph Agents (Skills 201–250)

| # | Skill / Capability | Implementation File | Symbol / Function |
|---|---|---|---|
| 201 | Structured statutory policy dataclass (`PolicyRule`) | `src/agents/retrofit_copilot.py` | `PolicyRule` |
| 202 | Built-in Ofgem ECO4 / GBIS / PAS 2035 fallback corpus | `src/agents/retrofit_copilot.py` | `POLICY_CORPUS` |
| 203 | Automated multi-format statutory directory loader | `src/agents/retrofit_copilot.py` | `load_regulatory_corpus` |
| 204 | PDF text extraction via `pypdf.PdfReader` | `src/agents/retrofit_copilot.py` | `_extract_file_text` |
| 205 | Markdown & plain-text statutory file ingestion | `src/agents/retrofit_copilot.py` | `_extract_file_text` |
| 206 | Heading-aware semantic clause chunking (`[RULE-ID]`) | `src/agents/retrofit_copilot.py` | `_chunk_statutory_text` |
| 207 | Paragraph-level semantic chunking fallback for unheaded PDFs | `src/agents/retrofit_copilot.py` | `_chunk_statutory_text` |
| 208 | Automatic framework classification (`ECO4` / `GBIS` / `PAS2035`) | `src/agents/retrofit_copilot.py` | `_infer_framework` |
| 209 | Automatic domain keyword extraction per statutory clause | `src/agents/retrofit_copilot.py` | `_extract_keywords` |
| 210 | Automatic EPC band eligibility inference from clause text | `src/agents/retrofit_copilot.py` | `_infer_epc_bands` |
| 211 | Collision-safe unique rule ID deduplication across docs | `src/agents/retrofit_copilot.py` | `load_regulatory_corpus` |
| 212 | Graceful fallback to `POLICY_CORPUS` when `docs_dir` empty/corrupt | `src/agents/retrofit_copilot.py` | `load_regulatory_corpus` |
| 213 | Curated Ofgem ECO4 statutory markdown guidance | `data/regulatory_docs/ofgem_eco4_guidance.md` | `[ECO4-ELIG-01]`..`[ECO4-TECH-04]` |
| 214 | Curated Ofgem GBIS statutory markdown guidance | `data/regulatory_docs/ofgem_gbis_guidance.md` | `[GBIS-ELIG-01]`..`[GBIS-TECH-02]` |
| 215 | Curated BSI PAS 2035:2023 statutory markdown guidance | `data/regulatory_docs/bsi_pas2035_guidance.md` | `[PAS2035-VENT-01]`..`[PAS2035-RISK-02]` |
| 216 | Custom regex alphanumeric tokenizer for BM25 | `src/agents/retrofit_copilot.py` | `_tokenize` |
| 217 | `BM25Okapi` lexical sparse retrieval indexing | `src/agents/retrofit_copilot.py` | `HybridPolicyRetriever.__init__` |
| 218 | Unigram+bigram `TfidfVectorizer` dense-sparse matrix | `src/agents/retrofit_copilot.py` | `HybridPolicyRetriever.__init__` |
| 219 | Cosine similarity scoring over TF-IDF statutory vectors | `src/agents/retrofit_copilot.py` | `HybridPolicyRetriever.retrieve` |
| 220 | Hybrid score fusion (45% BM25 + 30% TF-IDF + 25% metadata) | `src/agents/retrofit_copilot.py` | `HybridPolicyRetriever.retrieve` |
| 221 | Property-context query enrichment (EPC, wall, roof, IMD) | `src/agents/retrofit_copilot.py` | `HybridPolicyRetriever.retrieve` |
| 222 | Mandatory PAS 2035 safety-rule retrieval boosting | `src/agents/retrofit_copilot.py` | `HybridPolicyRetriever.retrieve` |
| 223 | Direct rule-ID lookup accessor (`get_rule`) | `src/agents/retrofit_copilot.py` | `HybridPolicyRetriever.get_rule` |
| 224 | Typed LangGraph state dictionary (`CopilotState`) | `src/agents/retrofit_copilot.py` | `CopilotState` |
| 225 | LangGraph `StateGraph` compilation with conditional routing | `src/agents/retrofit_copilot.py` | `build_retrofit_copilot_graph` |
| 226 | Prompt-injection & EPC-forgery regex guardrail patterns | `src/agents/retrofit_copilot.py` | `INJECTION_PATTERNS` |
| 227 | Input compliance guardrail node (`input_guardrail`) | `src/agents/retrofit_copilot.py` | `build_retrofit_copilot_graph` |
| 228 | EPC band A–G input sanity validation inside guardrail | `src/agents/retrofit_copilot.py` | `build_retrofit_copilot_graph` |
| 229 | Safe compliance refusal branch (`refusal_node`) | `src/agents/retrofit_copilot.py` | `build_retrofit_copilot_graph` |
| 230 | Hybrid retrieval LangGraph node (`hybrid_retrieve`) | `src/agents/retrofit_copilot.py` | `build_retrofit_copilot_graph` |
| 231 | Grounded ECO4 / ECO4 Flex / GBIS synthesis node | `src/agents/retrofit_copilot.py` | `build_retrofit_copilot_graph` |
| 232 | Fabric-first solid/cavity wall recommendation logic | `src/agents/retrofit_copilot.py` | `build_retrofit_copilot_graph` |
| 233 | Sub-150mm loft top-up recommendation logic | `src/agents/retrofit_copilot.py` | `build_retrofit_copilot_graph` |
| 234 | Off-gas clean heating heat-pump transition check | `src/agents/retrofit_copilot.py` | `build_retrofit_copilot_graph` |
| 235 | Mandatory PAS 2035 ventilation & Risk Path injection | `src/agents/retrofit_copilot.py` | `build_retrofit_copilot_graph` |
| 236 | Regex inline citation extractor (`CITATION_PATTERN`) | `src/agents/retrofit_copilot.py` | `CITATION_PATTERN` |
| 237 | Independent citation-verifier judge node | `src/agents/retrofit_copilot.py` | `build_retrofit_copilot_graph` |
| 238 | Hallucinated/unknown `[RULE-ID]` detection | `src/agents/retrofit_copilot.py` | `verify_citations_against_rules` |
| 239 | Missing-citation rejection in judge verifier | `src/agents/retrofit_copilot.py` | `verify_citations_against_rules` |
| 240 | Mandatory PAS 2035 citation enforcement for insulation | `src/agents/retrofit_copilot.py` | `verify_citations_against_rules` |
| 241 | Conditional retry-repair loop (`route_after_judge`) | `src/agents/retrofit_copilot.py` | `build_retrofit_copilot_graph` |
| 242 | Testable hallucination injection hook (`force_unverified_first_draft`) | `src/agents/retrofit_copilot.py` | `build_retrofit_copilot_graph` |
| 243 | Safe unverified-draft fallback response (`finalize_response`) | `src/agents/retrofit_copilot.py` | `build_retrofit_copilot_graph` |
| 244 | Structured citation metadata resolution (`rule_id`, `framework`, `title`) | `src/agents/retrofit_copilot.py` | `build_retrofit_copilot_graph` |
| 245 | Single-call copilot entrypoint (`run_retrofit_copilot`) | `src/agents/retrofit_copilot.py` | `run_retrofit_copilot` |
| 246 | FastAPI integration of guardrailed LangGraph copilot | `src/app/main.py` | `advise_retrofit` |
| 247 | Streamlit interactive form integration for copilot | `src/app/dashboard.py` | `submitted` |
| 248 | End-to-end runner verification of copilot output | `src/run_pipeline.py` | `run_end_to_end` |
| 249 | Unit verification of judge repair & guardrail blocking | `tests/test_contracts_and_errors.py` | `test_langgraph_citation_verification_and_guardrails` |
| 250 | Automated PDF/MD parsing verification in unit tests | `tests/test_contracts_and_errors.py` | `test_regulatory_corpus_loader_parses_md_and_pdf` |

---

## Pillar 6: MLOps, Serving, Testing & Production Infrastructure (Skills 251–300)

| # | Skill / Capability | Implementation File | Symbol / Function |
|---|---|---|---|
| 251 | PEP 621 modern Python project packaging | `pyproject.toml` | `[project]` |
| 252 | Optional `dev` and `ui` dependency extras separation | `pyproject.toml` | `[project.optional-dependencies]` |
| 253 | Ruff linter configuration (`E`, `F`, `I`, `UP`, `B`) | `pyproject.toml` | `[tool.ruff.lint]` |
| 254 | Pytest test-discovery & `pythonpath` configuration | `pyproject.toml` | `[tool.pytest.ini_options]` |
| 255 | Slim Python 3.12 Docker image with `libgomp1` for LightGBM | `Dockerfile` | `FROM python:3.12-slim` |
| 256 | Containerised Uvicorn ASGI API entrypoint | `Dockerfile` | `CMD ["uvicorn", ...]` |
| 257 | Multi-service `docker-compose.yml` orchestration | `docker-compose.yml` | `services` |
| 258 | `pgvector/pgvector:pg16` vector database service + healthcheck | `docker-compose.yml` | `services.pgvector` |
| 259 | Dedicated MLflow tracking server container + SQLite store | `docker-compose.yml` | `services.mlflow` |
| 260 | Health-gated `api` container dependency wiring | `docker-compose.yml` | `services.api.depends_on` |
| 261 | GitHub Actions CI pipeline (`ruff`, `run_pipeline`, `pytest`) | `.github/workflows/ci.yml` | `jobs.lint-and-test` |
| 262 | Reproducible `uv` virtual environment provisioning in CI | `.github/workflows/ci.yml` | `Install dependencies` |
| 263 | SQLite-backed MLflow tracking URI (`sqlite:///mlruns/mlflow.db`) | `src/models/train_conformal.py` | `train_conformal_model` |
| 264 | Environment-overridable `MLFLOW_TRACKING_URI` support | `src/models/train_conformal.py` | `train_conformal_model` |
| 265 | MLflow parameter logging (`model_family`, `cv_strategy`, `alpha`) | `src/models/train_conformal.py` | `train_conformal_model` |
| 266 | MLflow metric logging (`spatial_cv_rmse_gbp`, `conformal_coverage_90`) | `src/models/train_conformal.py` | `train_conformal_model` |
| 267 | MLflow CSV artifact logging (`shap_feature_importance.csv`) | `src/models/train_conformal.py` | `train_conformal_model` |
| 268 | Pickle model bundle serialization for inference serving | `src/models/train_conformal.py` | `train_conformal_model` |
| 269 | FastAPI `/health` readiness probe reporting DuckDB & model status | `src/app/main.py` | `health_check` |
| 270 | Pydantic v2 request schema validation with field bounds | `src/app/main.py` | `RetrofitAdviceRequest` |
| 271 | Pydantic v2 typed response contract (`RetrofitAdviceResponse`) | `src/app/main.py` | `RetrofitAdviceResponse` |
| 272 | Environment-configurable model & DuckDB artifact paths | `src/app/main.py` | `DEFAULT_MODEL_PATH` |
| 273 | Graceful degradation when conformal model pickle is missing | `src/app/main.py` | `advise_retrofit` |
| 274 | Graceful exception-safe fallback if runtime inference fails | `src/app/main.py` | `advise_retrofit` |
| 275 | Graceful copilot runtime fallback advising TrustMark referral | `src/app/main.py` | `advise_retrofit` |
| 276 | Warning & `fallback_used` observability flags in API payload | `src/app/main.py` | `RetrofitAdviceResponse` |
| 277 | Streamlit wide-layout interactive dashboard | `src/app/dashboard.py` | `st.set_page_config` |
| 278 | Missing-database friendly onboarding prompt in UI | `src/app/dashboard.py` | `DB_PATH.exists()` |
| 279 | Interactive three-tab UI (Spatial, Error Analysis, Copilot) | `src/app/dashboard.py` | `st.tabs` |
| 280 | Live single-property conformal forecast display in Streamlit | `src/app/dashboard.py` | `predict_property_heating_cost` |
| 281 | Automated raw CSV snapshot archival in `data/raw/` | `src/run_pipeline.py` | `run_end_to_end` |
| 282 | Standardized logging configuration for live connector telemetry | `src/run_pipeline.py` | `__main__` |
| 283 | Isolated `tmp_path` DuckDB/Parquet contract testing | `tests/test_contracts_and_errors.py` | `test_quarantine_isolation_and_gdpr_hashing` |
| 284 | Deterministic salt-sensitivity assertion for GDPR hashing | `tests/test_contracts_and_errors.py` | `test_quarantine_isolation_and_gdpr_hashing` |
| 285 | End-to-end conformal coverage & report structure test | `tests/test_contracts_and_errors.py` | `test_spatial_uplift_and_conformal_training_and_error_analysis` |
| 286 | FastAPI `TestClient` monkeypatched fallback verification | `tests/test_contracts_and_errors.py` | `test_fastapi_health_and_fallback_advice` |
| 287 | Mocked Postcodes.io + DESNZ EPC API integration test | `tests/test_contracts_and_errors.py` | `test_live_connectors_mocked_apis_and_degraded_fallback` |
| 288 | Offline `LIVE_FEED_DEGRADED` logging & fallback test | `tests/test_contracts_and_errors.py` | `test_live_connectors_mocked_apis_and_degraded_fallback` |
| 289 | `--live` and `--no-live` pipeline runner integration test | `tests/test_contracts_and_errors.py` | `test_run_pipeline_live_mode_fallback` |
| 290 | Clean `.gitignore` exclusion of `.venv/`, `mlruns/`, `*.egg-info/`, and generated data | `.gitignore` | `.gitignore` |
| 291 | Comprehensive operator documentation (`README.md`) | `README.md` | `Quickstart` |
| 292 | Cross-package modular namespace layout (`src/*/__init__.py`) | `src/__init__.py` | `src` |
| 293 | `tenacity` transient-network resilience across external UK APIs | `src/ingestion/live_uk_connectors.py` | `_query_postcodes_io` |
| 294 | HTTP timeout enforcement (`10s` / `15s`) preventing hung workers | `src/ingestion/live_uk_connectors.py` | `_query_desnz_epc_api` |
| 295 | Zero-key offline fallback guarantee across ML & RAG modules | `src/agents/retrofit_copilot.py` | `run_retrofit_copilot` |
| 296 | Reproducible synthetic benchmark seed (`random_state=42`) | `src/run_pipeline.py` | `generate_synthetic_uk_epc_dataset` |
| 297 | Strict Parquet schema compatibility on mixed corrupt payloads | `src/ingestion/epc_pipeline.py` | `_write_quarantine_parquet` |
| 298 | Context-managed DuckDB connection lifecycle hygiene | `src/ingestion/epc_pipeline.py` | `write_epc_to_duckdb` |
| 299 | Context-managed MLflow run lifecycle hygiene | `src/models/train_conformal.py` | `train_conformal_model` |
| 300 | End-to-end automated regression gate (`pytest -v` + `ruff`) | `.github/workflows/ci.yml` | `Run pytest suite` |
