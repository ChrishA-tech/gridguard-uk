# GridGuard UK

**GridGuard UK** is an end-to-end UK residential retrofit intelligence, spatial fuel poverty analytics, conformal ML forecasting, and Ofgem-compliant LangGraph advisory platform.

## Architecture Overview

- **Live UK Open Data & Pandera Contracts** ([src/ingestion/live_uk_connectors.py](src/ingestion/live_uk_connectors.py), [src/ingestion/epc_pipeline.py](src/ingestion/epc_pipeline.py)):
  - Queries live **Postcodes.io** (`https://api.postcodes.io/postcodes`) for real ONS LSOA codes, Local Authority GSS codes (`E08000012`, etc.), and geospatial coordinates, and pairs them with live **DESNZ EPC Open Data API** records (when `EPC_API_TOKEN` is configured) or calibrated real-LSOA housing profiles.
  - Enforces `EPCRecordSchema` via Pandera, isolating invalid rows into Parquet Dead-Letter Quarantine (`data/quarantine/`), hashing UPRNs with HMAC-SHA256 for UK GDPR compliance, and persisting clean tables to DuckDB (`data/processed/gridguard.duckdb`).
- **Spatial Fuel Poverty & Causal Uplift** ([src/analytics/spatial_uplift.py](src/analytics/spatial_uplift.py)):
  - Computes LSOA Fuel Poverty Priority scores (DESNZ LILEE proxy + ONS IMD) and Two-Model (`TLearnerUpliftEstimator`) causal bill savings uplift (£/yr).
- **Spatial Conformal Prediction & Error Analysis** ([src/models/train_conformal.py](src/models/train_conformal.py), [src/evaluation/error_analysis.py](src/evaluation/error_analysis.py)):
  - Trains LightGBM under Local Authority `GroupKFold` spatial CV with `MAPIE` 90% conformal prediction intervals, TreeSHAP feature attributions, MLflow tracking, and slice-level diagnostics in [docs/ERROR_ANALYSIS_REPORT.md](docs/ERROR_ANALYSIS_REPORT.md).
- **Guardrailed LangGraph Retrofit Copilot** ([src/agents/retrofit_copilot.py](src/agents/retrofit_copilot.py)):
  - Parses `.pdf`, `.md`, and `.txt` Ofgem ECO4 / GBIS / PAS 2035 statutory guidance from `data/regulatory_docs/` into semantic clause chunks indexed with hybrid BM25 + TF-IDF retrieval, protected by an input guardrail and citation-verifier judge node.
- **Serving & Interactive UI** ([src/app/main.py](src/app/main.py), [src/app/dashboard.py](src/app/dashboard.py)):
  - FastAPI `/health` and `/advise-retrofit` endpoints with SAP heuristic fallback, plus a Streamlit Spatial Explorer & Retrofit Copilot dashboard.

## Quickstart

### 1. Set Up Environment

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]" streamlit tabulate
```

### 2. Run the End-to-End Pipeline (`--live`)

By default, `--live` queries Postcodes.io (and DESNZ EPC Open Data API if `EPC_API_TOKEN` is exported) with automatic `LIVE_FEED_DEGRADED` fallback when offline:

```bash
# Optional: enable authenticated DESNZ EPC Open Data API records
export EPC_API_TOKEN="your-email@example.com:your-epc-api-key"

# Run with live UK Postcodes.io / EPC enrichment (default)
python -m src.run_pipeline --live

# Or customise postcode prefix / record limit, or force offline synthetic mode
python -m src.run_pipeline --live --postcode-prefix L1 --limit 500
python -m src.run_pipeline --no-live
```

### 3. Launch the Streamlit Dashboard & FastAPI Service

```bash
# Launch the interactive Streamlit Spatial Explorer & Retrofit Copilot
streamlit run src/app/dashboard.py

# Launch the FastAPI prediction & advisory service
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

### 4. Run Tests & Linting

```bash
pytest -v
ruff check src tests
```

## Documentation

- [docs/ERROR_ANALYSIS_REPORT.md](docs/ERROR_ANALYSIS_REPORT.md) — Spatial CV metrics, global SHAP drivers, slice RMSE/MAE, residual bias, and 90% conformal coverage.
- [docs/SKILLS_MATRIX.md](docs/SKILLS_MATRIX.md) — 300-skill engineering, econometrics, conformal ML, causal inference, RAG, and MLOps capability matrix mapped to `gridguard-uk`.
