"""Slice-based error analysis, residual bias audit, and MAPIE 90% conformal
coverage diagnostics producing ``docs/ERROR_ANALYSIS_REPORT.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


@dataclass(frozen=True)
class ErrorAnalysisResult:
    """Structured error analysis summary and generated Markdown report path."""

    overall_metrics: dict[str, float]
    slice_tables: dict[str, pd.DataFrame]
    report_path: Path
    markdown_text: str


def _compute_slice_metrics(
    df: pd.DataFrame,
    slice_col: str,
    target_col: str = "annual_heating_cost_gbp",
    pred_col: str = "predicted_heating_cost_gbp",
    lower_col: str = "conformal_lower_90_gbp",
    upper_col: str = "conformal_upper_90_gbp",
) -> pd.DataFrame:
    """Compute RMSE, MAE, residual bias, and 90% conformal coverage per slice."""
    rows: list[dict[str, object]] = []
    for slice_val, grp in df.groupby(slice_col, observed=True):
        y_true = grp[target_col].to_numpy(dtype=float)
        y_pred = grp[pred_col].to_numpy(dtype=float)
        lower = grp[lower_col].to_numpy(dtype=float)
        upper = grp[upper_col].to_numpy(dtype=float)

        residuals = y_true - y_pred
        covered = (y_true >= lower) & (y_true <= upper)
        widths = upper - lower

        rows.append(
            {
                slice_col: str(slice_val),
                "n_samples": int(len(grp)),
                "rmse_gbp": round(float(root_mean_squared_error(y_true, y_pred)), 2),
                "mae_gbp": round(float(mean_absolute_error(y_true, y_pred)), 2),
                "residual_bias_gbp": round(float(np.mean(residuals)), 2),
                "conformal_coverage_90": round(float(np.mean(covered)), 4),
                "mean_interval_width_gbp": round(float(np.mean(widths)), 2),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("rmse_gbp", ascending=False).reset_index(drop=True)
    return out


def generate_error_analysis_report(
    oof_df: pd.DataFrame,
    shap_importance_df: pd.DataFrame | None = None,
    output_path: Path | str = Path("docs/ERROR_ANALYSIS_REPORT.md"),
    target_col: str = "annual_heating_cost_gbp",
    pred_col: str = "predicted_heating_cost_gbp",
    lower_col: str = "conformal_lower_90_gbp",
    upper_col: str = "conformal_upper_90_gbp",
) -> ErrorAnalysisResult:
    """Compute slice diagnostics and write ``docs/ERROR_ANALYSIS_REPORT.md``."""
    required = {target_col, pred_col, lower_col, upper_col}
    missing = sorted(required - set(oof_df.columns))
    if missing:
        raise ValueError(f"OOF predictions dataframe is missing required columns: {missing}")

    work = oof_df.copy()
    if "imd_decile" in work.columns:
        work["imd_deprivation_band"] = np.select(
            [work["imd_decile"] <= 3, work["imd_decile"] <= 7],
            ["Deciles 1-3 (Most Deprived)", "Deciles 4-7 (Mid Deprivation)"],
            default="Deciles 8-10 (Least Deprived)",
        )

    y_true = work[target_col].to_numpy(dtype=float)
    y_pred = work[pred_col].to_numpy(dtype=float)
    lower = work[lower_col].to_numpy(dtype=float)
    upper = work[upper_col].to_numpy(dtype=float)

    residuals = y_true - y_pred
    covered = (y_true >= lower) & (y_true <= upper)

    overall = {
        "n_samples": int(len(work)),
        "rmse_gbp": round(float(root_mean_squared_error(y_true, y_pred)), 2),
        "mae_gbp": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "residual_bias_gbp": round(float(np.mean(residuals)), 2),
        "conformal_coverage_90": round(float(np.mean(covered)), 4),
        "mean_interval_width_90_gbp": round(float(np.mean(upper - lower)), 2),
    }

    candidate_slices = [
        ("EPC Energy Rating Band", "current_energy_rating"),
        ("Wall Construction Type", "wall_type"),
        ("Main Heating Fuel", "main_heat_fuel"),
        ("Local Authority (Spatial Fold)", "local_authority"),
        ("ONS IMD Deprivation Band", "imd_deprivation_band"),
    ]

    slice_tables: dict[str, pd.DataFrame] = {}
    for _, col in candidate_slices:
        if col in work.columns:
            slice_tables[col] = _compute_slice_metrics(
                work,
                slice_col=col,
                target_col=target_col,
                pred_col=pred_col,
                lower_col=lower_col,
                upper_col=upper_col,
            )

    coverage_gap = overall["conformal_coverage_90"] - 0.90
    lines: list[str] = [
        "# GridGuard UK — Error Analysis & Conformal Diagnostics Report",
        "",
        f"Generated: `{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}`",
        "",
        "## 1. Executive Summary (Out-of-Fold Spatial CV)",
        "",
        f"- **Evaluated Properties:** `{overall['n_samples']}`",
        f"- **Spatial CV RMSE:** `£{overall['rmse_gbp']:.2f}/yr`",
        f"- **Spatial CV MAE:** `£{overall['mae_gbp']:.2f}/yr`",
        f"- **Mean Residual Bias (`y_true - y_pred`):** `£{overall['residual_bias_gbp']:+.2f}/yr`",
        f"- **Empirical 90% Conformal Coverage:** `{overall['conformal_coverage_90']:.2%}` "
        f"(calibration gap vs 90.0% target: `{coverage_gap:+.2%}`)",
        f"- **Mean 90% Prediction Interval Width:** `£{overall['mean_interval_width_90_gbp']:.2f}/yr`",
        "",
    ]

    if shap_importance_df is not None and not shap_importance_df.empty:
        lines.extend(
            [
                "## 2. Global SHAP Feature Attributions",
                "",
                shap_importance_df.head(10).to_markdown(index=False),
                "",
            ]
        )

    lines.append("## 3. Slice-Based Error, Residual Bias & Conformal Coverage")
    lines.append("")

    undercovered_alerts: list[str] = []
    bias_alerts: list[str] = []

    for title, col in candidate_slices:
        tbl = slice_tables.get(col)
        if tbl is None or tbl.empty:
            continue
        lines.append(f"### {title} (`{col}`)")
        lines.append("")
        lines.append(tbl.to_markdown(index=False))
        lines.append("")

        low_cov = tbl[tbl["conformal_coverage_90"] < 0.85]
        for _, r in low_cov.iterrows():
            undercovered_alerts.append(
                f"- `{col}={r[col]}` coverage `{r['conformal_coverage_90']:.2%}` (<85% alert, n={r['n_samples']})"
            )

        high_bias = tbl[tbl["residual_bias_gbp"].abs() > 45.0]
        for _, r in high_bias.iterrows():
            bias_alerts.append(
                f"- `{col}={r[col]}` residual bias `£{r['residual_bias_gbp']:+.2f}/yr` (RMSE £{r['rmse_gbp']:.2f})"
            )

    lines.append("## 4. Diagnostic Findings & Mitigation Recommendations")
    lines.append("")
    if undercovered_alerts:
        lines.append("### Under-Covered Slices (< 85% Empirical Coverage)")
        lines.extend(undercovered_alerts)
        lines.append("")
    else:
        lines.append("- All evaluated housing and spatial slices achieved >= 85% empirical conformal coverage.")
        lines.append("")

    if bias_alerts:
        lines.append("### Elevated Residual Bias Slices (|Bias| > £45/yr)")
        lines.extend(bias_alerts)
        lines.append("")
    else:
        lines.append("- No slice exhibited systematic residual bias exceeding ±£45/yr.")
        lines.append("")

    lines.extend(
        [
            "### Recommended Next Iterations",
            "- Apply Mondrian (group-conditional) conformal calibration across `main_heat_fuel` or `current_energy_rating` if heteroscedastic residual widths diverge across off-gas-grid homes.",
            "- Enrich solid-wall and pre-1919 stock with local climate degree-days and PAS 2035 ventilation-risk flags to reduce tail RMSE.",
        ]
    )

    markdown_text = "\n".join(lines) + "\n"
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown_text, encoding="utf-8")

    return ErrorAnalysisResult(
        overall_metrics=overall,
        slice_tables=slice_tables,
        report_path=out,
        markdown_text=markdown_text,
    )
