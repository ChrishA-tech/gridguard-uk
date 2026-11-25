"""FastAPI service exposing /health and /advise-retrofit with graceful fallback handling."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.agents.retrofit_copilot import run_retrofit_copilot
from src.models.train_conformal import predict_property_heating_cost

DEFAULT_MODEL_PATH = Path(os.environ.get("GRIDGUARD_MODEL_PATH", "data/processed/conformal_heating_model.pkl"))
DEFAULT_DB_PATH = Path(os.environ.get("GRIDGUARD_DB_PATH", "data/processed/gridguard.duckdb"))

app = FastAPI(
    title="GridGuard UK Retrofit Intelligence API",
    version="0.1.0",
    description="Conformal EPC heating cost estimation and Ofgem-compliant LangGraph retrofit advisory.",
)


class RetrofitAdviceRequest(BaseModel):
    query: str = Field(
        default="Recommend eligible Ofgem ECO4/GBIS retrofit measures and PAS 2035 checks.",
        min_length=3,
        max_length=1000,
    )
    local_authority: str = "Manchester"
    property_type: str = "House"
    built_form: str = "Mid-Terrace"
    construction_age_band: str = "1900-1929"
    tenure: str = "Owner-occupied"
    total_floor_area: float = Field(default=85.0, ge=12.0, le=1000.0)
    current_energy_efficiency: float = Field(default=52.0, ge=1.0, le=100.0)
    current_energy_rating: str = "E"
    main_heat_fuel: str = "Mains gas"
    wall_type: str = "Solid brick"
    roof_insulation_mm: float = Field(default=75.0, ge=0.0, le=500.0)
    energy_consumption_kwh_m2: float = Field(default=290.0, ge=10.0, le=1500.0)
    imd_decile: int = Field(default=2, ge=1, le=10)
    imd_score: float = Field(default=42.0, ge=0.0, le=100.0)
    household_income_gbp: float = Field(default=24000.0, ge=5000.0, le=500000.0)


class RetrofitAdviceResponse(BaseModel):
    predicted_heating_cost_gbp: float | None
    conformal_lower_90_gbp: float | None
    conformal_upper_90_gbp: float | None
    top_shap_drivers: list[dict[str, Any]]
    guardrail_passed: bool
    citation_verified: bool
    verification_notes: str
    eligible_schemes: list[str]
    recommended_measures: list[str]
    citations: list[dict[str, str]]
    advisory_markdown: str
    fallback_used: bool = False
    warning: str | None = None


def _heuristic_heating_fallback(req: RetrofitAdviceRequest) -> dict[str, Any]:
    """Deterministic SAP-based heating cost fallback when conformal pickle is unavailable."""
    sap_penalty = max(0.2, (100.0 - req.current_energy_efficiency) / 45.0)
    estimate = round(req.total_floor_area * 12.5 * sap_penalty, 2)
    return {
        "predicted_heating_cost_gbp": estimate,
        "conformal_lower_90_gbp": round(max(50.0, estimate * 0.80), 2),
        "conformal_upper_90_gbp": round(estimate * 1.20, 2),
        "top_shap_drivers": [],
    }


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Service health probe reporting model and DuckDB readiness."""
    return {
        "status": "ok",
        "service": "gridguard-uk",
        "model_available": DEFAULT_MODEL_PATH.exists(),
        "duckdb_available": DEFAULT_DB_PATH.exists(),
    }


@app.post("/advise-retrofit", response_model=RetrofitAdviceResponse)
def advise_retrofit(payload: RetrofitAdviceRequest) -> RetrofitAdviceResponse:
    """Return 90% conformal heating cost forecast and verified ECO4/GBIS/PAS 2035 advice."""
    profile = payload.model_dump(exclude={"query"})
    fallback_used = False
    warnings: list[str] = []

    try:
        if DEFAULT_MODEL_PATH.exists():
            forecast = predict_property_heating_cost(profile, model_path=DEFAULT_MODEL_PATH)
        else:
            fallback_used = True
            warnings.append("Conformal model artifact not found; served SAP heuristic estimate.")
            forecast = _heuristic_heating_fallback(payload)
    except Exception as exc:  # pragma: no cover - defensive runtime fallback
        fallback_used = True
        warnings.append(f"Conformal inference failed ({type(exc).__name__}); served SAP heuristic estimate.")
        forecast = _heuristic_heating_fallback(payload)

    try:
        copilot_out = run_retrofit_copilot(query=payload.query, property_profile=profile)
    except Exception as exc:  # pragma: no cover - defensive runtime fallback
        fallback_used = True
        warnings.append(f"Copilot execution fallback triggered ({type(exc).__name__}).")
        copilot_out = {
            "guardrail_passed": True,
            "citation_verified": False,
            "verification_notes": "Fallback advisory triggered due to copilot error.",
            "eligible_schemes": [],
            "recommended_measures": ["Consult TrustMark Retrofit Coordinator (PAS 2035)"],
            "citations": [],
            "advisory_markdown": (
                "Automated policy synthesis is temporarily unavailable. "
                "Please refer this property to a TrustMark-registered PAS 2035 Retrofit Coordinator."
            ),
        }

    return RetrofitAdviceResponse(
        predicted_heating_cost_gbp=forecast["predicted_heating_cost_gbp"],
        conformal_lower_90_gbp=forecast["conformal_lower_90_gbp"],
        conformal_upper_90_gbp=forecast["conformal_upper_90_gbp"],
        top_shap_drivers=forecast["top_shap_drivers"],
        guardrail_passed=bool(copilot_out["guardrail_passed"]),
        citation_verified=bool(copilot_out["citation_verified"]),
        verification_notes=str(copilot_out["verification_notes"]),
        eligible_schemes=list(copilot_out["eligible_schemes"]),
        recommended_measures=list(copilot_out["recommended_measures"]),
        citations=list(copilot_out["citations"]),
        advisory_markdown=str(copilot_out["advisory_markdown"]),
        fallback_used=fallback_used,
        warning="; ".join(warnings) if warnings else None,
    )
