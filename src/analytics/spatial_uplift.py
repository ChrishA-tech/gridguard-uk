"""Spatial fuel poverty priority scoring (DESNZ LILEE proxy + ONS IMD)
and Two-Model (T-Learner) causal annual bill savings uplift estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = [
    "total_floor_area",
    "current_energy_efficiency",
    "roof_insulation_mm",
    "energy_consumption_kwh_m2",
    "imd_decile",
    "imd_score",
    "household_income_gbp",
]

CATEGORICAL_FEATURES = [
    "property_type",
    "built_form",
    "construction_age_band",
    "tenure",
    "main_heat_fuel",
    "wall_type",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def compute_lsoa_fuel_poverty_priority(df: pd.DataFrame) -> pd.DataFrame:
    """Compute LSOA-level Fuel Poverty Priority scores using a DESNZ LILEE proxy + ONS IMD."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "lsoa_code",
                "local_authority",
                "property_count",
                "pct_epc_d_to_g",
                "mean_sap_score",
                "mean_annual_heating_cost_gbp",
                "mean_imd_decile",
                "mean_imd_score",
                "low_income_share",
                "lilee_proxy_share",
                "fuel_poverty_priority_score",
                "priority_tier",
            ]
        )

    work = df.copy()
    work["is_epc_d_to_g"] = work["current_energy_rating"].isin(["D", "E", "F", "G"]).astype(float)
    work["is_low_income"] = (work["household_income_gbp"] < 31000.0).astype(float)
    work["is_lilee_proxy"] = ((work["is_epc_d_to_g"] == 1.0) & (work["is_low_income"] == 1.0)).astype(float)

    grouped = (
        work.groupby(["lsoa_code", "local_authority"], as_index=False)
        .agg(
            property_count=("lsoa_code", "size"),
            pct_epc_d_to_g=("is_epc_d_to_g", "mean"),
            mean_sap_score=("current_energy_efficiency", "mean"),
            mean_annual_heating_cost_gbp=("annual_heating_cost_gbp", "mean"),
            mean_imd_decile=("imd_decile", "mean"),
            mean_imd_score=("imd_score", "mean"),
            low_income_share=("is_low_income", "mean"),
            lilee_proxy_share=("is_lilee_proxy", "mean"),
        )
    )

    lilee_component = grouped["lilee_proxy_share"] * 100.0
    imd_component = ((11.0 - grouped["mean_imd_decile"].clip(1.0, 10.0)) / 10.0) * 100.0
    epc_component = grouped["pct_epc_d_to_g"] * 100.0
    cost_component = (grouped["mean_annual_heating_cost_gbp"] / 3500.0).clip(0.0, 1.0) * 100.0

    grouped["fuel_poverty_priority_score"] = (
        0.35 * lilee_component
        + 0.25 * imd_component
        + 0.25 * epc_component
        + 0.15 * cost_component
    ).round(2)

    grouped["priority_tier"] = np.select(
        [
            grouped["fuel_poverty_priority_score"] >= 65.0,
            grouped["fuel_poverty_priority_score"] >= 50.0,
            grouped["fuel_poverty_priority_score"] >= 35.0,
        ],
        ["Critical", "High", "Moderate"],
        default="Low",
    )

    return grouped.sort_values("fuel_poverty_priority_score", ascending=False).reset_index(drop=True)


class TLearnerUpliftEstimator:
    """Two-Model (T-Learner) causal estimator for annual energy bill savings (£/yr).

    Fits separate outcome regressors for control (``T=0``) and retrofitted (``T=1``)
    homes, then estimates CATE savings as ``mu_0(x) - mu_1(x)``.
    """

    def __init__(self, random_state: int = 42, n_estimators: int = 120) -> None:
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.model_control: Pipeline | None = None
        self.model_treated: Pipeline | None = None
        self.is_fitted_: bool = False

    def _make_pipeline(self, seed_offset: int) -> Pipeline:
        return Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor()),
                (
                    "regressor",
                    LGBMRegressor(
                        n_estimators=self.n_estimators,
                        learning_rate=0.06,
                        num_leaves=31,
                        min_child_samples=10,
                        random_state=self.random_state + seed_offset,
                        verbose=-1,
                    ),
                ),
            ]
        )

    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str = "retrofit_treated",
        outcome_col: str = "observed_annual_bill_gbp",
    ) -> TLearnerUpliftEstimator:
        """Fit control and treated outcome models."""
        control_df = df[df[treatment_col] == 0]
        treated_df = df[df[treatment_col] == 1]

        if control_df.empty or treated_df.empty:
            raise ValueError("Both control (T=0) and treated (T=1) observations are required for T-Learner.")

        self.model_control = self._make_pipeline(seed_offset=0)
        self.model_treated = self._make_pipeline(seed_offset=1)

        self.model_control.fit(control_df[FEATURE_COLUMNS], control_df[outcome_col])
        self.model_treated.fit(treated_df[FEATURE_COLUMNS], treated_df[outcome_col])
        self.is_fitted_ = True
        return self

    def predict_outcomes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict counterfactual bills ``mu0_bill_gbp``, ``mu1_bill_gbp``, and savings uplift."""
        if not self.is_fitted_ or self.model_control is None or self.model_treated is None:
            raise RuntimeError("TLearnerUpliftEstimator must be fitted before calling predict_outcomes.")

        features = df[FEATURE_COLUMNS]
        mu0 = self.model_control.predict(features)
        mu1 = self.model_treated.predict(features)
        savings = mu0 - mu1

        return pd.DataFrame(
            {
                "mu0_bill_gbp": np.round(mu0, 2),
                "mu1_bill_gbp": np.round(mu1, 2),
                "estimated_bill_savings_gbp": np.round(savings, 2),
            },
            index=df.index,
        )

    def predict_uplift(self, df: pd.DataFrame) -> np.ndarray:
        """Return array of estimated annual bill savings (£/yr) per property."""
        return self.predict_outcomes(df)["estimated_bill_savings_gbp"].to_numpy()


@dataclass(frozen=True)
class SpatialUpliftResult:
    """Outputs produced by :func:`estimate_spatial_uplift`."""

    property_uplift_df: pd.DataFrame
    lsoa_summary_df: pd.DataFrame
    estimator: TLearnerUpliftEstimator
    ate_savings_gbp: float


def estimate_spatial_uplift(
    df: pd.DataFrame,
    random_state: int = 42,
    db_path: Path | str | None = None,
) -> SpatialUpliftResult:
    """Compute LSOA Fuel Poverty Priority scores and fit T-Learner causal bill savings uplift."""
    lsoa_priority = compute_lsoa_fuel_poverty_priority(df)

    estimator = TLearnerUpliftEstimator(random_state=random_state)
    estimator.fit(df)
    outcomes = estimator.predict_outcomes(df)

    property_df = df.copy()
    property_df["mu0_bill_gbp"] = outcomes["mu0_bill_gbp"]
    property_df["mu1_bill_gbp"] = outcomes["mu1_bill_gbp"]
    property_df["estimated_bill_savings_gbp"] = outcomes["estimated_bill_savings_gbp"]

    lsoa_uplift = (
        property_df.groupby("lsoa_code", as_index=False)
        .agg(
            mean_bill_savings_gbp=("estimated_bill_savings_gbp", "mean"),
            median_bill_savings_gbp=("estimated_bill_savings_gbp", "median"),
            total_lsoa_savings_gbp=("estimated_bill_savings_gbp", "sum"),
        )
    )
    lsoa_uplift["mean_bill_savings_gbp"] = lsoa_uplift["mean_bill_savings_gbp"].round(2)
    lsoa_uplift["median_bill_savings_gbp"] = lsoa_uplift["median_bill_savings_gbp"].round(2)
    lsoa_uplift["total_lsoa_savings_gbp"] = lsoa_uplift["total_lsoa_savings_gbp"].round(2)

    lsoa_summary = (
        lsoa_priority.merge(lsoa_uplift, on="lsoa_code", how="left")
        .sort_values("fuel_poverty_priority_score", ascending=False)
        .reset_index(drop=True)
    )

    if db_path is not None:
        target = Path(db_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(target)) as con:
            con.register("lsoa_summary_view", lsoa_summary)
            con.execute("CREATE OR REPLACE TABLE lsoa_priority_uplift AS SELECT * FROM lsoa_summary_view")
            con.register("property_uplift_view", property_df)
            con.execute("CREATE OR REPLACE TABLE property_uplift AS SELECT * FROM property_uplift_view")

    ate_savings = float(np.round(property_df["estimated_bill_savings_gbp"].mean(), 2))
    return SpatialUpliftResult(
        property_uplift_df=property_df,
        lsoa_summary_df=lsoa_summary,
        estimator=estimator,
        ate_savings_gbp=ate_savings,
    )
