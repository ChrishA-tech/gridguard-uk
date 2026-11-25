"""Spatial GroupKFold LightGBM training, MAPIE 90% conformal prediction intervals,
SHAP feature attributions, and MLflow experiment logging.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = [
    "total_floor_area",
    "current_energy_efficiency",
    "roof_insulation_mm",
    "energy_consumption_kwh_m2",
    "imd_decile",
    "imd_score",
]

CATEGORICAL_FEATURES = [
    "property_type",
    "built_form",
    "construction_age_band",
    "tenure",
    "main_heat_fuel",
    "wall_type",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "annual_heating_cost_gbp"


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


def _fit_mapie_and_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    X_eval: pd.DataFrame,
    alpha: float = 0.10,
    random_state: int = 42,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a MAPIE 90% conformal regressor over a LightGBM pipeline.

    Supports both MAPIE 0.9.x (``MapieRegressor``) and MAPIE 1.x
    (``CrossConformalRegressor`` / ``SplitConformalRegressor``).
    """
    preprocessor = _build_preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train)
    X_eval_enc = preprocessor.transform(X_eval)

    base_lgb = LGBMRegressor(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,
        random_state=random_state,
        verbose=-1,
    )

    unique_groups = int(groups_train.nunique())
    n_inner_splits = min(4, unique_groups) if unique_groups >= 2 else 2

    try:
        from mapie.regression import MapieRegressor  # MAPIE 0.9.x

        if unique_groups >= 2:
            inner_cv = GroupKFold(n_splits=n_inner_splits)
            conformal = MapieRegressor(estimator=base_lgb, method="plus", cv=inner_cv)
            conformal.fit(X_train_enc, y_train.to_numpy(), groups=groups_train.to_numpy())
        else:
            conformal = MapieRegressor(estimator=base_lgb, method="plus", cv=3)
            conformal.fit(X_train_enc, y_train.to_numpy())
        y_pred, y_pis = conformal.predict(X_eval_enc, alpha=alpha)
        lower = y_pis[:, 0, 0]
        upper = y_pis[:, 1, 0]
    except ImportError:  # MAPIE >= 1.0
        confidence_level = 1.0 - alpha
        from mapie.regression import CrossConformalRegressor, SplitConformalRegressor

        if unique_groups >= 3:
            inner_cv = GroupKFold(n_splits=n_inner_splits)
            conformal = CrossConformalRegressor(
                estimator=base_lgb,
                confidence_level=confidence_level,
                method="plus",
                cv=inner_cv,
            )
            conformal.fit_conformalize(
                X_train_enc,
                y_train.to_numpy(),
                groups=groups_train.to_numpy(),
            )
        else:
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
            fit_idx, cal_idx = next(splitter.split(X_train_enc, y_train, groups=groups_train))
            conformal = SplitConformalRegressor(
                estimator=base_lgb,
                confidence_level=confidence_level,
                prefit=False,
            )
            conformal.fit(X_train_enc[fit_idx], y_train.to_numpy()[fit_idx])
            conformal.conformalize(X_train_enc[cal_idx], y_train.to_numpy()[cal_idx])
        y_pred, y_pis = conformal.predict_interval(X_eval_enc)
        lower = y_pis[:, 0, 0]
        upper = y_pis[:, 1, 0]

    lower = np.maximum(0.0, lower)
    upper = np.maximum(lower, upper)
    bundle = {"preprocessor": preprocessor, "conformal_model": conformal}
    return bundle, np.asarray(y_pred, dtype=float), np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def _predict_with_bundle(bundle: dict[str, Any], X: pd.DataFrame, alpha: float = 0.10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict point estimate and 90% conformal interval from a saved bundle."""
    X_enc = bundle["preprocessor"].transform(X[MODEL_FEATURES])
    conformal = bundle["conformal_model"]
    if hasattr(conformal, "predict_interval"):
        y_pred, y_pis = conformal.predict_interval(X_enc)
    else:
        y_pred, y_pis = conformal.predict(X_enc, alpha=alpha)
    lower = np.maximum(0.0, y_pis[:, 0, 0])
    upper = np.maximum(lower, y_pis[:, 1, 0])
    return np.asarray(y_pred, dtype=float), np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def compute_shap_importance(
    df: pd.DataFrame,
    random_state: int = 42,
    max_samples: int = 400,
) -> tuple[Pipeline, pd.DataFrame]:
    """Fit a LightGBM pipeline and aggregate TreeSHAP attributions to original features."""
    preprocessor = _build_preprocessor()
    lgbm = LGBMRegressor(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,
        random_state=random_state,
        verbose=-1,
    )
    pipe = Pipeline([("preprocessor", preprocessor), ("regressor", lgbm)])
    X = df[MODEL_FEATURES]
    y = df[TARGET_COLUMN]
    pipe.fit(X, y)

    sample_X = X.sample(n=min(max_samples, len(X)), random_state=random_state)
    transformed = pipe.named_steps["preprocessor"].transform(sample_X)
    encoded_names = list(pipe.named_steps["preprocessor"].get_feature_names_out())

    explainer = shap.TreeExplainer(pipe.named_steps["regressor"])
    shap_values = np.asarray(explainer.shap_values(transformed))

    # Map transformed column names back to original raw feature names
    raw_shap: dict[str, np.ndarray] = {feat: np.zeros(len(sample_X), dtype=float) for feat in MODEL_FEATURES}
    for col_idx, enc_name in enumerate(encoded_names):
        if enc_name.startswith("num__"):
            base_feat = enc_name.removeprefix("num__")
            if base_feat in raw_shap:
                raw_shap[base_feat] += shap_values[:, col_idx]
        elif enc_name.startswith("cat__"):
            suffix = enc_name.removeprefix("cat__")
            for cat_feat in CATEGORICAL_FEATURES:
                if suffix.startswith(f"{cat_feat}_"):
                    raw_shap[cat_feat] += shap_values[:, col_idx]
                    break

    importance_rows = [
        {
            "feature": feat,
            "mean_abs_shap_gbp": round(float(np.mean(np.abs(vals))), 3),
            "mean_signed_shap_gbp": round(float(np.mean(vals)), 3),
        }
        for feat, vals in raw_shap.items()
    ]
    importance_df = (
        pd.DataFrame(importance_rows)
        .sort_values("mean_abs_shap_gbp", ascending=False)
        .reset_index(drop=True)
    )
    return pipe, importance_df


@dataclass(frozen=True)
class ConformalTrainingResult:
    """Outputs and diagnostics from :func:`train_conformal_model`."""

    oof_predictions_df: pd.DataFrame
    shap_importance_df: pd.DataFrame
    model_bundle: dict[str, Any]
    shap_pipeline: Pipeline
    metrics: dict[str, float]
    model_path: Path
    mlflow_run_id: str | None


def train_conformal_model(
    df: pd.DataFrame,
    alpha: float = 0.10,
    n_splits: int = 5,
    random_state: int = 42,
    model_path: Path | str = Path("data/processed/conformal_heating_model.pkl"),
    mlflow_tracking_uri: str | None = None,
    experiment_name: str = "gridguard-uk-conformal",
) -> ConformalTrainingResult:
    """Train LightGBM with Local Authority GroupKFold spatial CV, MAPIE 90% intervals,
    SHAP attributions, and MLflow logging.
    """
    unique_authorities = int(df["local_authority"].nunique())
    effective_splits = min(n_splits, unique_authorities)
    if effective_splits < 2:
        raise ValueError("At least 2 distinct local_authority groups are required for GroupKFold spatial CV.")

    gkf = GroupKFold(n_splits=effective_splits)
    X = df[MODEL_FEATURES]
    y = df[TARGET_COLUMN]
    groups = df["local_authority"]

    oof_pred = np.zeros(len(df), dtype=float)
    oof_lower = np.zeros(len(df), dtype=float)
    oof_upper = np.zeros(len(df), dtype=float)
    oof_fold = np.zeros(len(df), dtype=int)

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        _, fold_pred, fold_low, fold_high = _fit_mapie_and_predict(
            X.iloc[train_idx],
            y.iloc[train_idx],
            groups.iloc[train_idx],
            X.iloc[test_idx],
            alpha=alpha,
            random_state=random_state + fold_idx,
        )
        oof_pred[test_idx] = fold_pred
        oof_lower[test_idx] = fold_low
        oof_upper[test_idx] = fold_high
        oof_fold[test_idx] = fold_idx

    # Fit final production conformal bundle on full dataset
    final_bundle, _, _, _ = _fit_mapie_and_predict(
        X, y, groups, X.iloc[:5], alpha=alpha, random_state=random_state
    )
    shap_pipe, shap_importance_df = compute_shap_importance(df, random_state=random_state)

    oof_df = df.copy()
    oof_df["cv_fold"] = oof_fold
    oof_df["predicted_heating_cost_gbp"] = np.round(oof_pred, 2)
    oof_df["conformal_lower_90_gbp"] = np.round(oof_lower, 2)
    oof_df["conformal_upper_90_gbp"] = np.round(oof_upper, 2)
    oof_df["conformal_width_90_gbp"] = np.round(oof_upper - oof_lower, 2)
    oof_df["residual_gbp"] = np.round(oof_df[TARGET_COLUMN] - oof_df["predicted_heating_cost_gbp"], 2)
    oof_df["covered_90"] = (
        (oof_df[TARGET_COLUMN] >= oof_df["conformal_lower_90_gbp"])
        & (oof_df[TARGET_COLUMN] <= oof_df["conformal_upper_90_gbp"])
    )

    rmse = float(root_mean_squared_error(oof_df[TARGET_COLUMN], oof_df["predicted_heating_cost_gbp"]))
    mae = float(mean_absolute_error(oof_df[TARGET_COLUMN], oof_df["predicted_heating_cost_gbp"]))
    r2 = float(r2_score(oof_df[TARGET_COLUMN], oof_df["predicted_heating_cost_gbp"]))
    coverage_90 = float(oof_df["covered_90"].mean())
    mean_width = float(oof_df["conformal_width_90_gbp"].mean())
    mean_bias = float(oof_df["residual_gbp"].mean())

    metrics = {
        "spatial_cv_rmse_gbp": round(rmse, 3),
        "spatial_cv_mae_gbp": round(mae, 3),
        "spatial_cv_r2": round(r2, 4),
        "conformal_coverage_90": round(coverage_90, 4),
        "mean_interval_width_90_gbp": round(mean_width, 2),
        "mean_residual_bias_gbp": round(mean_bias, 3),
    }

    target_model_path = Path(model_path)
    target_model_path.parent.mkdir(parents=True, exist_ok=True)
    with target_model_path.open("wb") as fh:
        pickle.dump(
            {
                "bundle": final_bundle,
                "shap_pipeline": shap_pipe,
                "shap_importance": shap_importance_df,
                "alpha": alpha,
                "metrics": metrics,
            },
            fh,
        )

    tracking_uri = mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")
    if tracking_uri.startswith("sqlite:///"):
        db_file = Path(tracking_uri.removeprefix("sqlite:///"))
        db_file.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    run_id: str | None = None
    with mlflow.start_run(run_name="lightgbm-mapie-spatial-cv") as run:
        run_id = run.info.run_id
        mlflow.log_params(
            {
                "model_family": "LightGBMRegressor",
                "conformal_wrapper": "MAPIE",
                "cv_strategy": "GroupKFold(local_authority)",
                "n_splits": effective_splits,
                "conformal_alpha": alpha,
                "n_samples": len(df),
                "n_local_authorities": unique_authorities,
            }
        )
        mlflow.log_metrics(metrics)
        shap_artifact = target_model_path.parent / "shap_feature_importance.csv"
        shap_importance_df.to_csv(shap_artifact, index=False)
        mlflow.log_artifact(str(shap_artifact))

    return ConformalTrainingResult(
        oof_predictions_df=oof_df,
        shap_importance_df=shap_importance_df,
        model_bundle=final_bundle,
        shap_pipeline=shap_pipe,
        metrics=metrics,
        model_path=target_model_path,
        mlflow_run_id=run_id,
    )


def predict_property_heating_cost(
    property_features: dict[str, Any],
    model_path: Path | str = Path("data/processed/conformal_heating_model.pkl"),
) -> dict[str, Any]:
    """Run point + 90% conformal interval prediction for a single property."""
    with Path(model_path).open("rb") as fh:
        payload = pickle.load(fh)
    df_one = pd.DataFrame([property_features])[MODEL_FEATURES]
    pred, low, high = _predict_with_bundle(payload["bundle"], df_one, alpha=payload.get("alpha", 0.10))
    return {
        "predicted_heating_cost_gbp": round(float(pred[0]), 2),
        "conformal_lower_90_gbp": round(float(low[0]), 2),
        "conformal_upper_90_gbp": round(float(high[0]), 2),
        "top_shap_drivers": payload["shap_importance"].head(5).to_dict(orient="records"),
    }
