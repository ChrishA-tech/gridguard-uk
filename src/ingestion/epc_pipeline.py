"""EPC ingestion pipeline with Pandera schema validation, UK GDPR UPRN hashing,
DuckDB analytical storage, and Parquet Dead-Letter Quarantine.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

try:
    import pandera.pandas as pa
    from pandera.errors import SchemaError, SchemaErrors
except ImportError:  # pragma: no cover - fallback for older pandera releases
    import pandera as pa
    from pandera.errors import SchemaError, SchemaErrors

VALID_EPC_RATINGS = ["A", "B", "C", "D", "E", "F", "G"]
VALID_PROPERTY_TYPES = ["House", "Flat", "Bungalow", "Maisonette", "Park home"]
VALID_BUILT_FORMS = [
    "Detached",
    "Semi-Detached",
    "Mid-Terrace",
    "End-Terrace",
    "Enclosed Mid-Terrace",
    "Enclosed End-Terrace",
]
VALID_TENURES = ["Owner-occupied", "Rented (social)", "Rented (private)"]
VALID_HEAT_FUELS = ["Mains gas", "Electricity", "Oil", "LPG", "Solid fuel", "Heat network"]
VALID_WALL_TYPES = [
    "Solid brick",
    "Cavity wall (unfilled)",
    "Cavity wall (filled)",
    "Timber frame",
    "System built",
]

EPCRecordSchema = pa.DataFrameSchema(
    columns={
        "uprn": pa.Column(
            str,
            nullable=False,
            coerce=True,
            checks=[pa.Check.str_matches(r"^[0-9]{6,12}$", error="Invalid UK UPRN format")],
        ),
        "lsoa_code": pa.Column(
            str,
            nullable=False,
            coerce=True,
            checks=[pa.Check.str_matches(r"^[EWS]01[0-9]{6}$", error="Invalid ONS LSOA code")],
        ),
        "local_authority": pa.Column(
            str,
            nullable=False,
            coerce=True,
            checks=[pa.Check.str_length(min_value=2, max_value=100)],
        ),
        "property_type": pa.Column(
            str,
            nullable=False,
            coerce=True,
            checks=[pa.Check.isin(VALID_PROPERTY_TYPES)],
        ),
        "built_form": pa.Column(
            str,
            nullable=False,
            coerce=True,
            checks=[pa.Check.isin(VALID_BUILT_FORMS)],
        ),
        "construction_age_band": pa.Column(
            str,
            nullable=False,
            coerce=True,
            checks=[pa.Check.str_length(min_value=3, max_value=40)],
        ),
        "tenure": pa.Column(
            str,
            nullable=False,
            coerce=True,
            checks=[pa.Check.isin(VALID_TENURES)],
        ),
        "total_floor_area": pa.Column(
            float,
            nullable=False,
            coerce=True,
            checks=[pa.Check.in_range(min_value=12.0, max_value=1000.0)],
        ),
        "current_energy_efficiency": pa.Column(
            float,
            nullable=False,
            coerce=True,
            checks=[pa.Check.in_range(min_value=1.0, max_value=100.0)],
        ),
        "current_energy_rating": pa.Column(
            str,
            nullable=False,
            coerce=True,
            checks=[pa.Check.isin(VALID_EPC_RATINGS)],
        ),
        "potential_energy_efficiency": pa.Column(
            float,
            nullable=False,
            coerce=True,
            checks=[pa.Check.in_range(min_value=1.0, max_value=100.0)],
        ),
        "main_heat_fuel": pa.Column(
            str,
            nullable=False,
            coerce=True,
            checks=[pa.Check.isin(VALID_HEAT_FUELS)],
        ),
        "wall_type": pa.Column(
            str,
            nullable=False,
            coerce=True,
            checks=[pa.Check.isin(VALID_WALL_TYPES)],
        ),
        "roof_insulation_mm": pa.Column(
            float,
            nullable=False,
            coerce=True,
            checks=[pa.Check.in_range(min_value=0.0, max_value=500.0)],
        ),
        "annual_heating_cost_gbp": pa.Column(
            float,
            nullable=False,
            coerce=True,
            checks=[pa.Check.in_range(min_value=50.0, max_value=20000.0)],
        ),
        "energy_consumption_kwh_m2": pa.Column(
            float,
            nullable=False,
            coerce=True,
            checks=[pa.Check.in_range(min_value=10.0, max_value=1500.0)],
        ),
        "imd_decile": pa.Column(
            int,
            nullable=False,
            coerce=True,
            checks=[pa.Check.in_range(min_value=1, max_value=10)],
        ),
        "imd_score": pa.Column(
            float,
            nullable=False,
            coerce=True,
            checks=[pa.Check.in_range(min_value=0.0, max_value=100.0)],
        ),
        "household_income_gbp": pa.Column(
            float,
            nullable=False,
            coerce=True,
            checks=[pa.Check.in_range(min_value=5000.0, max_value=500000.0)],
        ),
        "retrofit_treated": pa.Column(
            int,
            nullable=False,
            coerce=True,
            checks=[pa.Check.isin([0, 1])],
        ),
        "observed_annual_bill_gbp": pa.Column(
            float,
            nullable=False,
            coerce=True,
            checks=[pa.Check.in_range(min_value=50.0, max_value=20000.0)],
        ),
    },
    checks=[
        pa.Check(
            lambda df: df["potential_energy_efficiency"] >= df["current_energy_efficiency"],
            name="potential_gte_current_sap",
            error="potential_energy_efficiency must be >= current_energy_efficiency",
        ),
    ],
    strict=False,
    coerce=True,
)


@dataclass(frozen=True)
class IngestionResult:
    """Summary returned by :func:`ingest_epc_data`."""

    clean_df: pd.DataFrame
    quarantined_df: pd.DataFrame
    quarantine_path: Path | None
    db_path: Path
    total_rows: int
    clean_rows: int
    quarantined_rows: int


def get_gdpr_salt(salt: str | None = None) -> str:
    """Resolve the UK GDPR pseudonymisation salt."""
    return salt or os.environ.get("GRIDGUARD_SALT", "gridguard-uk-gdpr-salt-2026")


def hash_uprn(uprn: Any, salt: str | None = None) -> str:
    """Pseudonymise a UK UPRN using HMAC-SHA256 for UK GDPR compliance."""
    key = get_gdpr_salt(salt).encode("utf-8")
    msg = str(uprn).strip().encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def anonymise_epc_dataframe(df: pd.DataFrame, salt: str | None = None) -> pd.DataFrame:
    """Replace raw ``uprn`` identifiers with irreversible ``uprn_hash`` digests."""
    if "uprn" not in df.columns:
        raise ValueError("Expected 'uprn' column in EPC dataframe prior to anonymisation")
    resolved_salt = get_gdpr_salt(salt)
    out = df.copy()
    out["uprn_hash"] = out["uprn"].map(lambda val: hash_uprn(val, salt=resolved_salt))
    out = out.drop(columns=["uprn"])
    cols = ["uprn_hash"] + [c for c in out.columns if c != "uprn_hash"]
    return out[cols]


def _extract_row_reasons(failure_cases: pd.DataFrame, valid_indices: pd.Index) -> dict[Any, list[str]]:
    """Map row index -> list of human-readable Pandera validation errors."""
    reasons: dict[Any, list[str]] = {}
    if failure_cases.empty or "index" not in failure_cases.columns:
        return reasons

    for _, row in failure_cases.iterrows():
        idx = row.get("index")
        if idx is None or pd.isna(idx) or idx not in valid_indices:
            continue
        col = row.get("column") or "dataframe"
        check = row.get("check") or "schema_check"
        val = row.get("failure_case")
        msg = f"{col}:{check} (value={val})"
        reasons.setdefault(idx, []).append(msg)
    return reasons


def _row_by_row_fallback(df: pd.DataFrame) -> tuple[list[Any], dict[Any, list[str]]]:
    """Fallback row-wise validation when column-level coercion hides row indices."""
    bad_indices: list[Any] = []
    reasons: dict[Any, list[str]] = {}
    for idx in df.index:
        single = df.loc[[idx]]
        try:
            EPCRecordSchema.validate(single, lazy=True)
        except (SchemaErrors, SchemaError) as exc:
            bad_indices.append(idx)
            if isinstance(exc, SchemaErrors) and not exc.failure_cases.empty:
                msgs = [
                    f"{r.get('column')}:{r.get('check')}"
                    for _, r in exc.failure_cases.iterrows()
                ]
            else:
                msgs = [str(exc).splitlines()[0]]
            reasons[idx] = msgs
    return bad_indices, reasons


def _write_quarantine_parquet(
    quarantined_df: pd.DataFrame,
    quarantine_dir: Path,
    quarantine_filename: str | None = None,
) -> Path:
    """Persist quarantined rows safely to Parquet, stringifying mixed-type payload columns."""
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    if quarantine_filename is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        quarantine_filename = f"epc_quarantine_{stamp}.parquet"
    out_path = quarantine_dir / quarantine_filename

    serialisable = quarantined_df.copy()
    for col in serialisable.columns:
        if serialisable[col].dtype == object:
            serialisable[col] = serialisable[col].map(lambda v: None if pd.isna(v) else str(v))
    serialisable.to_parquet(out_path, index=False)
    return out_path


def validate_and_quarantine(
    df: pd.DataFrame,
    quarantine_dir: Path | str = Path("data/quarantine"),
    quarantine_filename: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path | None]:
    """Validate EPC records against :data:`EPCRecordSchema`.

    Invalid rows are isolated into a Parquet Dead-Letter Quarantine file with
    failure diagnostics while valid rows are returned schema-coerced and clean.
    """
    if df.empty:
        return df.copy(), pd.DataFrame(columns=list(df.columns) + ["quarantine_reason", "quarantined_at"]), None

    work_df = df.reset_index(drop=True).copy()
    required_cols = set(EPCRecordSchema.columns.keys())
    missing_cols = sorted(required_cols - set(work_df.columns))
    if missing_cols:
        raise ValueError(f"Input dataframe is missing required EPC columns: {missing_cols}")

    try:
        validated = EPCRecordSchema.validate(work_df, lazy=True)
        empty_q = pd.DataFrame(columns=list(work_df.columns) + ["quarantine_reason", "quarantined_at"])
        return validated, empty_q, None
    except SchemaErrors as exc:
        reasons = _extract_row_reasons(exc.failure_cases, work_df.index)
        bad_idx = sorted(reasons.keys(), key=lambda x: int(x))
        # If column-wide coercion failed without populating row indices, fall back row-by-row.
        if not bad_idx:
            bad_idx, reasons = _row_by_row_fallback(work_df)
    except SchemaError:
        bad_idx, reasons = _row_by_row_fallback(work_df)

    # Re-check remaining rows to catch any secondary errors masked by earlier coercion failure.
    remaining = work_df.drop(index=bad_idx)
    while not remaining.empty:
        try:
            validated_clean = EPCRecordSchema.validate(remaining, lazy=True)
            break
        except SchemaErrors as exc2:
            extra_reasons = _extract_row_reasons(exc2.failure_cases, remaining.index)
            extra_idx = list(extra_reasons.keys())
            if not extra_idx:
                extra_idx, extra_reasons = _row_by_row_fallback(remaining)
            if not extra_idx:  # pragma: no cover
                raise
            reasons.update(extra_reasons)
            bad_idx.extend(extra_idx)
            remaining = work_df.drop(index=bad_idx)
        except SchemaError:
            extra_idx, extra_reasons = _row_by_row_fallback(remaining)
            if not extra_idx:  # pragma: no cover
                raise
            reasons.update(extra_reasons)
            bad_idx.extend(extra_idx)
            remaining = work_df.drop(index=bad_idx)
    else:
        validated_clean = work_df.iloc[0:0].copy()

    bad_idx_sorted = sorted(set(bad_idx), key=lambda x: int(x))
    quarantined = work_df.loc[bad_idx_sorted].copy()
    quarantined["quarantine_reason"] = [
        "; ".join(dict.fromkeys(reasons.get(idx, ["schema_validation_failed"])))
        for idx in bad_idx_sorted
    ]
    quarantined["quarantined_at"] = datetime.now(UTC).isoformat()

    q_path = _write_quarantine_parquet(
        quarantined,
        Path(quarantine_dir),
        quarantine_filename=quarantine_filename,
    )
    return validated_clean.reset_index(drop=True), quarantined.reset_index(drop=True), q_path


def write_epc_to_duckdb(
    clean_df: pd.DataFrame,
    db_path: Path | str = Path("data/processed/gridguard.duckdb"),
    table_name: str = "epc_records",
) -> Path:
    """Persist validated, GDPR-anonymised EPC records into DuckDB."""
    target = Path(db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(target)) as con:
        con.register("clean_epc_view", clean_df)
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM clean_epc_view")
    return target


def load_epc_from_duckdb(
    db_path: Path | str = Path("data/processed/gridguard.duckdb"),
    table_name: str = "epc_records",
) -> pd.DataFrame:
    """Read persisted EPC records from DuckDB."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        return con.execute(f"SELECT * FROM {table_name}").df()


def ingest_epc_data(
    df: pd.DataFrame,
    db_path: Path | str = Path("data/processed/gridguard.duckdb"),
    quarantine_dir: Path | str = Path("data/quarantine"),
    table_name: str = "epc_records",
    quarantine_filename: str | None = None,
    salt: str | None = None,
) -> IngestionResult:
    """End-to-end EPC ingestion: validate -> quarantine -> GDPR hash -> DuckDB write."""
    validated_df, quarantined_df, q_path = validate_and_quarantine(
        df,
        quarantine_dir=quarantine_dir,
        quarantine_filename=quarantine_filename,
    )
    anonymised_df = anonymise_epc_dataframe(validated_df, salt=salt)
    resolved_db = write_epc_to_duckdb(anonymised_df, db_path=db_path, table_name=table_name)

    return IngestionResult(
        clean_df=anonymised_df,
        quarantined_df=quarantined_df,
        quarantine_path=q_path,
        db_path=resolved_db,
        total_rows=len(df),
        clean_rows=len(anonymised_df),
        quarantined_rows=len(quarantined_df),
    )
