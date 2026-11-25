"""End-to-end GridGuard UK pipeline runner supporting live Postcodes.io + DESNZ EPC
open data enrichment (``--live``, default) with automatic fallback, plus synthetic
offline DESNZ EPC + ONS IMD generation (``--no-live``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.retrofit_copilot import run_retrofit_copilot
from src.analytics.spatial_uplift import estimate_spatial_uplift
from src.evaluation.error_analysis import generate_error_analysis_report
from src.ingestion.epc_pipeline import ingest_epc_data
from src.ingestion.live_uk_connectors import fetch_live_ons_and_epc_data
from src.models.train_conformal import train_conformal_model

LOCAL_AUTHORITIES: dict[str, tuple[str, float]] = {
    "Manchester": ("E010051", 1.8),
    "Birmingham": ("E010089", 1.9),
    "Tower Hamlets": ("E010042", 1.5),
    "Leeds": ("E010113", 3.2),
    "Liverpool": ("E010065", 1.6),
    "Sheffield": ("E010078", 3.5),
    "Bristol": ("E010145", 4.8),
    "Newcastle upon Tyne": ("E010083", 2.7),
}


def _sap_to_rating(sap: float) -> str:
    if sap >= 92:
        return "A"
    if sap >= 81:
        return "B"
    if sap >= 69:
        return "C"
    if sap >= 55:
        return "D"
    if sap >= 39:
        return "E"
    if sap >= 21:
        return "F"
    return "G"


def generate_synthetic_uk_epc_dataset(
    n_rows: int = 1000,
    corrupt_fraction: float = 0.03,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate realistic UK DESNZ EPC + ONS IMD records with controlled corrupt rows."""
    rng = np.random.default_rng(random_state)

    authorities = list(LOCAL_AUTHORITIES.keys())
    la_choices = rng.choice(authorities, size=n_rows, replace=True)

    lsoa_codes = []
    imd_deciles = []
    for la in la_choices:
        prefix, mean_decile = LOCAL_AUTHORITIES[la]
        lsoa_suffix = int(rng.integers(10, 15))
        lsoa_codes.append(f"{prefix}{lsoa_suffix:02d}")
        decile = int(np.clip(np.round(rng.normal(loc=mean_decile + 2.0, scale=2.2)), 1, 10))
        imd_deciles.append(decile)

    imd_deciles_arr = np.asarray(imd_deciles, dtype=int)
    imd_scores = np.clip((11 - imd_deciles_arr) * 6.8 + rng.normal(0.0, 4.5, size=n_rows), 2.0, 95.0)
    household_income = np.clip(16000.0 + imd_deciles_arr * 4200.0 + rng.normal(0.0, 6500.0, size=n_rows), 11000.0, 180000.0)

    property_types = rng.choice(
        ["House", "Flat", "Bungalow", "Maisonette"],
        size=n_rows,
        p=[0.58, 0.24, 0.12, 0.06],
    )
    built_forms = rng.choice(
        ["Detached", "Semi-Detached", "Mid-Terrace", "End-Terrace"],
        size=n_rows,
        p=[0.18, 0.34, 0.31, 0.17],
    )
    age_bands = rng.choice(
        ["before 1900", "1900-1929", "1930-1949", "1950-1966", "1967-1975", "1983-1990", "2003-2006", "2012 onwards"],
        size=n_rows,
        p=[0.12, 0.16, 0.14, 0.16, 0.12, 0.12, 0.10, 0.08],
    )
    tenures = rng.choice(
        ["Owner-occupied", "Rented (social)", "Rented (private)"],
        size=n_rows,
        p=[0.54, 0.24, 0.22],
    )
    wall_types = np.where(
        np.isin(age_bands, ["before 1900", "1900-1929"]),
        rng.choice(["Solid brick", "Cavity wall (unfilled)"], size=n_rows, p=[0.78, 0.22]),
        np.where(
            np.isin(age_bands, ["2003-2006", "2012 onwards"]),
            rng.choice(["Cavity wall (filled)", "Timber frame"], size=n_rows, p=[0.75, 0.25]),
            rng.choice(["Cavity wall (unfilled)", "Cavity wall (filled)", "System built"], size=n_rows, p=[0.48, 0.40, 0.12]),
        ),
    )
    heat_fuels = rng.choice(
        ["Mains gas", "Electricity", "Oil", "LPG", "Heat network"],
        size=n_rows,
        p=[0.74, 0.15, 0.05, 0.03, 0.03],
    )

    base_area = np.where(property_types == "Flat", 62.0, np.where(built_forms == "Detached", 125.0, 88.0))
    total_floor_area = np.round(np.clip(base_area + rng.normal(0.0, 16.0, size=n_rows), 30.0, 320.0), 1)
    roof_insulation_mm = np.round(
        np.clip(
            np.where(np.isin(age_bands, ["2003-2006", "2012 onwards"]), 270.0, 120.0)
            + rng.normal(0.0, 55.0, size=n_rows),
            0.0,
            400.0,
        ),
        0,
    )

    wall_bonus = np.where(wall_types == "Cavity wall (filled)", 9.0, np.where(wall_types == "Solid brick", -10.0, -3.0))
    sap_raw = (
        61.0
        + wall_bonus
        + (roof_insulation_mm - 120.0) * 0.04
        + np.where(heat_fuels == "Electricity", -6.0, 2.0)
        + rng.normal(0.0, 8.5, size=n_rows)
    )
    current_sap = np.round(np.clip(sap_raw, 12.0, 95.0), 0)
    potential_sap = np.round(np.clip(current_sap + rng.integers(8, 28, size=n_rows), current_sap, 99.0), 0)
    current_rating = [_sap_to_rating(s) for s in current_sap]

    energy_kwh_m2 = np.round(np.clip(460.0 - current_sap * 3.8 + rng.normal(0.0, 25.0, size=n_rows), 45.0, 750.0), 1)
    fuel_factor = np.where(
        heat_fuels == "Electricity",
        1.32,
        np.where(heat_fuels == "Oil", 1.18, np.where(heat_fuels == "LPG", 1.24, 1.0)),
    )
    annual_heating_cost = np.round(
        np.clip(
            (total_floor_area * energy_kwh_m2 * 0.072 * fuel_factor) + rng.normal(0.0, 75.0, size=n_rows),
            240.0,
            8500.0,
        ),
        2,
    )

    treat_prob = np.clip(0.25 + 0.18 * (current_sap < 60) + 0.12 * (imd_deciles_arr <= 3), 0.1, 0.85)
    retrofit_treated = rng.binomial(1, treat_prob)
    true_savings = (
        180.0
        + 2.4 * total_floor_area
        + np.where(wall_types == "Solid brick", 190.0, np.where(wall_types == "Cavity wall (unfilled)", 140.0, 40.0))
        + np.where(roof_insulation_mm < 100.0, 110.0, 25.0)
    )
    observed_bill = np.round(
        np.clip(
            annual_heating_cost + 380.0 - retrofit_treated * true_savings + rng.normal(0.0, 65.0, size=n_rows),
            300.0,
            9500.0,
        ),
        2,
    )

    uprns = [f"{100000000000 + i}" for i in range(n_rows)]

    df = pd.DataFrame(
        {
            "uprn": uprns,
            "lsoa_code": lsoa_codes,
            "local_authority": la_choices,
            "property_type": property_types,
            "built_form": built_forms,
            "construction_age_band": age_bands,
            "tenure": tenures,
            "total_floor_area": total_floor_area,
            "current_energy_efficiency": current_sap,
            "current_energy_rating": current_rating,
            "potential_energy_efficiency": potential_sap,
            "main_heat_fuel": heat_fuels,
            "wall_type": wall_types,
            "roof_insulation_mm": roof_insulation_mm,
            "annual_heating_cost_gbp": annual_heating_cost,
            "energy_consumption_kwh_m2": energy_kwh_m2,
            "imd_decile": imd_deciles_arr,
            "imd_score": np.round(imd_scores, 2),
            "household_income_gbp": np.round(household_income, 2),
            "retrofit_treated": retrofit_treated,
            "observed_annual_bill_gbp": observed_bill,
        }
    )

    n_corrupt = int(round(n_rows * corrupt_fraction))
    if n_corrupt > 0:
        corrupt_indices = rng.choice(n_rows, size=n_corrupt, replace=False)
        for idx_pos, row_idx in enumerate(corrupt_indices):
            mode = idx_pos % 6
            if mode == 0:
                df.loc[row_idx, "uprn"] = "BAD_UPRN"
            elif mode == 1:
                df.loc[row_idx, "lsoa_code"] = "INVALID_LSOA"
            elif mode == 2:
                df.loc[row_idx, "total_floor_area"] = -45.0
            elif mode == 3:
                df.loc[row_idx, "current_energy_efficiency"] = 145.0
            elif mode == 4:
                df.loc[row_idx, "current_energy_rating"] = "Z"
            else:
                df.loc[row_idx, "imd_decile"] = 99

    return df


def run_end_to_end(
    live: bool = True,
    postcode_prefix: str = "L1",
    limit: int = 500,
    n_rows: int = 1000,
    corrupt_fraction: float = 0.03,
    data_dir: Path = Path("data"),
    docs_dir: Path = Path("docs"),
) -> dict[str, object]:
    """Execute full GridGuard UK ingestion, spatial uplift, conformal ML, error analysis, and copilot check."""
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    quarantine_dir = data_dir / "quarantine"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if live:
        raw_df = fetch_live_ons_and_epc_data(
            postcode_prefix=postcode_prefix,
            limit=limit,
            quarantine_dir=quarantine_dir,
        )
        raw_csv_path = raw_dir / f"live_uk_epc_{postcode_prefix.lower()}.csv"
    else:
        raw_df = generate_synthetic_uk_epc_dataset(n_rows=n_rows, corrupt_fraction=corrupt_fraction)
        raw_csv_path = raw_dir / "synthetic_desnz_epc_imd.csv"

    raw_df.to_csv(raw_csv_path, index=False)

    db_path = processed_dir / "gridguard.duckdb"
    ingestion_res = ingest_epc_data(
        raw_df,
        db_path=db_path,
        quarantine_dir=quarantine_dir,
        quarantine_filename="epc_quarantine_latest.parquet",
    )

    uplift_res = estimate_spatial_uplift(ingestion_res.clean_df, db_path=db_path)

    conformal_res = train_conformal_model(
        ingestion_res.clean_df,
        model_path=processed_dir / "conformal_heating_model.pkl",
    )

    error_res = generate_error_analysis_report(
        conformal_res.oof_predictions_df,
        shap_importance_df=conformal_res.shap_importance_df,
        output_path=docs_dir / "ERROR_ANALYSIS_REPORT.md",
    )

    sample_copilot = run_retrofit_copilot(
        query="Assess ECO4 / GBIS eligibility and PAS 2035 ventilation rules for this home.",
        property_profile={
            "current_energy_rating": "E",
            "wall_type": "Solid brick",
            "roof_insulation_mm": 75.0,
            "tenure": "Owner-occupied",
            "main_heat_fuel": "Mains gas",
            "imd_decile": 2,
            "household_income_gbp": 22500.0,
        },
        docs_dir=data_dir / "regulatory_docs",
    )

    mode_label = f"LIVE ({postcode_prefix})" if live else "SYNTHETIC OFFLINE"
    sources = (
        sorted(ingestion_res.clean_df["data_source"].unique().tolist())
        if "data_source" in ingestion_res.clean_df.columns
        else ["synthetic_desnz_epc_imd"]
    )

    print(f"=== GridGuard UK End-to-End Pipeline Complete [{mode_label}] ===")
    print(f"Data Sources:         {sources}")
    print(f"Distinct LAs / LSOAs: {ingestion_res.clean_df['local_authority'].nunique()} / {ingestion_res.clean_df['lsoa_code'].nunique()}")
    print(f"Total raw rows:       {ingestion_res.total_rows}")
    print(f"Clean rows in DuckDB: {ingestion_res.clean_rows}")
    print(f"Quarantined rows:     {ingestion_res.quarantined_rows} ({ingestion_res.quarantine_path})")
    print(f"T-Learner ATE savings: £{uplift_res.ate_savings_gbp:.2f}/yr")
    print(f"Spatial CV Metrics:   {conformal_res.metrics}")
    print(f"Error Report:         {error_res.report_path}")
    print(f"Copilot Verified:     {sample_copilot['citation_verified']} | Schemes: {sample_copilot['eligible_schemes']}")

    return {
        "ingestion": ingestion_res,
        "uplift": uplift_res,
        "conformal": conformal_res,
        "error_analysis": error_res,
        "copilot": sample_copilot,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GridGuard UK end-to-end intelligence pipeline.")
    parser.add_argument(
        "--live",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enrich with live Postcodes.io + DESNZ EPC API feeds (default: True; use --no-live for synthetic mode).",
    )
    parser.add_argument(
        "--postcode-prefix",
        default="L1",
        help="UK postcode prefix for live Postcodes.io / DESNZ EPC retrieval (default: L1).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum number of live records to ingest (default: 500).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    run_end_to_end(live=args.live, postcode_prefix=args.postcode_prefix, limit=args.limit)
