"""Streamlit Explorer & Retrofit Advisory Copilot Dashboard for GridGuard UK."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

from src.agents.retrofit_copilot import run_retrofit_copilot
from src.models.train_conformal import predict_property_heating_cost

DB_PATH = Path("data/processed/gridguard.duckdb")
MODEL_PATH = Path("data/processed/conformal_heating_model.pkl")
REPORT_PATH = Path("docs/ERROR_ANALYSIS_REPORT.md")

st.set_page_config(page_title="GridGuard UK — Fuel Poverty & Retrofit Copilot", layout="wide")
st.title("GridGuard UK — Spatial Fuel Poverty & Retrofit Intelligence")


@st.cache_data(show_spinner=False)
def load_tables(db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with duckdb.connect(str(db_path), read_only=True) as con:
        lsoa_df = con.execute("SELECT * FROM lsoa_priority_uplift").df()
        epc_df = con.execute("SELECT * FROM epc_records").df()
    return lsoa_df, epc_df


if not DB_PATH.exists():
    st.warning("DuckDB store not found. Run `python -m src.run_pipeline` first to populate data.")
    st.stop()

lsoa_df, epc_df = load_tables(DB_PATH)

with st.sidebar:
    st.header("Spatial Filters")
    authorities = sorted(lsoa_df["local_authority"].unique().tolist())
    selected_authorities = st.multiselect("Local Authority", authorities, default=authorities)
    tiers = ["Critical", "High", "Moderate", "Low"]
    selected_tiers = st.multiselect("Priority Tier", tiers, default=tiers)

filtered_lsoa = lsoa_df[
    lsoa_df["local_authority"].isin(selected_authorities) & lsoa_df["priority_tier"].isin(selected_tiers)
]
filtered_epc = epc_df[epc_df["local_authority"].isin(selected_authorities)]

col1, col2, col3, col4 = st.columns(4)
col1.metric("GDPR-Hashed Properties", f"{len(filtered_epc):,}")
col2.metric("Filtered LSOAs", f"{len(filtered_lsoa):,}")
col3.metric(
    "Critical / High LSOAs",
    f"{int(filtered_lsoa['priority_tier'].isin(['Critical', 'High']).sum()):,}",
)
col4.metric(
    "Mean T-Learner Savings",
    f"£{filtered_lsoa['mean_bill_savings_gbp'].mean():.0f}/yr" if not filtered_lsoa.empty else "N/A",
)

tab_spatial, tab_error, tab_copilot = st.tabs(
    ["Spatial Priority & Causal Uplift", "Error Analysis Report", "Retrofit Advisory Copilot"]
)

with tab_spatial:
    st.subheader("Top LSOAs by Fuel Poverty Priority Score & Causal Bill Savings")
    st.dataframe(filtered_lsoa.head(25), use_container_width=True)
    if not filtered_lsoa.empty:
        chart_df = filtered_lsoa.head(15).set_index("lsoa_code")[
            ["fuel_poverty_priority_score", "mean_bill_savings_gbp"]
        ]
        st.bar_chart(chart_df)

with tab_error:
    st.subheader("Conformal Calibration & Slice Diagnostics")
    if REPORT_PATH.exists():
        st.markdown(REPORT_PATH.read_text(encoding="utf-8"))
    else:
        st.info("Error analysis report not yet generated.")

with tab_copilot:
    st.subheader("Guardrailed Ofgem ECO4 / GBIS / PAS 2035 Copilot")
    with st.form("copilot_form"):
        c1, c2, c3 = st.columns(3)
        epc_band = c1.selectbox("EPC Band", ["A", "B", "C", "D", "E", "F", "G"], index=4)
        wall_type = c2.selectbox(
            "Wall Type",
            ["Solid brick", "Cavity wall (unfilled)", "Cavity wall (filled)", "Timber frame", "System built"],
        )
        tenure = c3.selectbox("Tenure", ["Owner-occupied", "Rented (social)", "Rented (private)"])
        roof_mm = c1.slider("Roof Insulation (mm)", 0, 400, 75, step=25)
        imd_decile = c2.slider("IMD Decile (1=Most Deprived)", 1, 10, 2)
        income = c3.number_input("Household Income (£/yr)", min_value=5000, max_value=250000, value=24000, step=1000)
        query = st.text_input(
            "Advisory Query",
            value="Assess ECO4/GBIS eligibility and PAS 2035 fabric-first retrofit measures.",
        )
        submitted = st.form_submit_button("Run Copilot & Conformal Forecast")

    if submitted:
        profile = {
            "total_floor_area": 85.0,
            "current_energy_efficiency": 52.0,
            "roof_insulation_mm": float(roof_mm),
            "energy_consumption_kwh_m2": 290.0,
            "imd_decile": int(imd_decile),
            "imd_score": float((11 - imd_decile) * 5.0),
            "household_income_gbp": float(income),
            "property_type": "House",
            "built_form": "Mid-Terrace",
            "construction_age_band": "1900-1929",
            "tenure": tenure,
            "main_heat_fuel": "Mains gas",
            "wall_type": wall_type,
            "current_energy_rating": epc_band,
        }
        if MODEL_PATH.exists():
            pred = predict_property_heating_cost(profile, model_path=MODEL_PATH)
            st.info(
                f"Predicted Annual Heating Cost: **£{pred['predicted_heating_cost_gbp']:,.2f}/yr** "
                f"(90% Conformal Interval: £{pred['conformal_lower_90_gbp']:,.2f} – £{pred['conformal_upper_90_gbp']:,.2f})"
            )
        advice = run_retrofit_copilot(query=query, property_profile=profile)
        st.success(f"Citation Verified: `{advice['citation_verified']}` | Guardrail Passed: `{advice['guardrail_passed']}`")
        st.markdown(advice["advisory_markdown"])
