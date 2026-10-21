from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import shap

ROOT = Path(__file__).parent.parent
DATASET_PATH = ROOT / "data" / "masld_patient_cohort.csv"
TEMPLATE_PATH = ROOT / "data" / "dataset_template.csv"
TARGET = "cvd_event"

app = FastAPI(
    title="CardioLiver-AI Backend API",
    description="FastAPI service for MASLD Cardiovascular Risk Prediction, Incremental Prognostic Benchmarks, and Explainable AI (SHAP)",
    version="2.0.0",
)

# Enable CORS for local Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def auto_detect_feature_schema(df: pd.DataFrame) -> dict[str, list[str]]:
    """Automatically detects feature columns and clinical partitions from DataFrame schema."""
    df = df.copy()
    if "fibrosis_stage" in df.columns:
        fib_map = {"F0": 0, "F1": 1, "F2": 2, "F3": 3, "F4": 4}
        raw_fib = df["fibrosis_stage"]
        if raw_fib.dtype == object or isinstance(raw_fib.iloc[0], str):
            df["fibrosis_stage"] = raw_fib.astype(str).str.strip().str.upper().map(fib_map).fillna(raw_fib)
        df["fibrosis_stage"] = pd.to_numeric(df["fibrosis_stage"], errors="coerce").fillna(0).astype(int)

    id_cols = [c for c in df.columns if c.lower() == "patient_id" or c.lower().endswith("_id")]
    target_cols = [c for c in df.columns if "cvd_event" in c.lower() or c.lower() == "target"]
    candidates = [c for c in df.columns if c not in id_cols and c not in target_cols]

    cat_cols = df[candidates].select_dtypes(include=["object", "category", "string"]).columns.tolist()
    num_cols = df[candidates].select_dtypes(include=[np.number]).columns.tolist()

    FIB_MARKERS = {"fibrosis_stage", "fib_4_score", "ast", "alt"}
    fib_cols = [c for c in candidates if c.lower() in FIB_MARKERS or c.lower().startswith("fib_")]
    base_cols = [c for c in candidates if c not in fib_cols]

    return {
        "num_base": [c for c in num_cols if c in base_cols],
        "cat_base": [c for c in cat_cols if c in base_cols],
        "num_enh": [c for c in num_cols if c in candidates],
        "cat_enh": [c for c in cat_cols if c in candidates],
        "features_base": base_cols,
        "features_enh": candidates,
        "all_required": sorted(list(set(candidates + [TARGET]))),
    }

# Fallback defaults
DEFAULT_SCHEMA = auto_detect_feature_schema(pd.read_csv(DATASET_PATH)) if DATASET_PATH.exists() else {
    "num_base": ["age", "bmi", "systolic_bp", "diastolic_bp", "fasting_glucose", "total_cholesterol", "ldl", "hdl", "triglycerides", "platelet_count", "slcvd_score", "who_cvd_risk"],
    "cat_base": ["sex", "diabetes", "hypertension", "smoker"],
    "num_enh": ["age", "bmi", "systolic_bp", "diastolic_bp", "fasting_glucose", "total_cholesterol", "ldl", "hdl", "triglycerides", "platelet_count", "slcvd_score", "who_cvd_risk", "fibrosis_stage", "fib_4_score", "ast", "alt"],
    "cat_enh": ["sex", "diabetes", "hypertension", "smoker"],
    "features_base": ["age", "bmi", "systolic_bp", "diastolic_bp", "fasting_glucose", "total_cholesterol", "ldl", "hdl", "triglycerides", "platelet_count", "slcvd_score", "who_cvd_risk", "sex", "diabetes", "hypertension", "smoker"],
    "features_enh": ["age", "bmi", "systolic_bp", "diastolic_bp", "fasting_glucose", "total_cholesterol", "ldl", "hdl", "triglycerides", "platelet_count", "slcvd_score", "who_cvd_risk", "fibrosis_stage", "fib_4_score", "ast", "alt", "sex", "diabetes", "hypertension", "smoker"],
    "all_required": ["age", "alt", "ast", "bmi", "cvd_event", "diabetes", "diastolic_bp", "fasting_glucose", "fib_4_score", "fibrosis_stage", "hdl", "hypertension", "ldl", "platelet_count", "sex", "slcvd_score", "smoker", "systolic_bp", "total_cholesterol", "triglycerides", "who_cvd_risk"],
}
NUMERIC_BASE = DEFAULT_SCHEMA["num_base"]
NUMERIC_ENHANCED = DEFAULT_SCHEMA["num_enh"]
CATEGORICAL = DEFAULT_SCHEMA["cat_enh"]
FEATURES_BASE = DEFAULT_SCHEMA["features_base"]
FEATURES_ENHANCED = DEFAULT_SCHEMA["features_enh"]
ALL_REQUIRED_COLUMNS = DEFAULT_SCHEMA["all_required"]

# Global in-memory cache
CACHE: dict[str, Any] = {
    "trained": False,
    "source": "none",
    "models": {},
    "metrics": {},
    "global_shap": [],
    "calibration": {},
    "dca": {},
    "interactions": {},
    "cohort_stats": {},
}


class PatientPayload(BaseModel):
    age: float = Field(..., ge=18, le=110, description="Age in years")
    sex: str = Field(..., description="Male or Female")
    bmi: float = Field(..., ge=10, le=70, description="Body Mass Index (kg/m^2)")
    diabetes: str = Field(..., description="Yes or No")
    hypertension: str = Field(..., description="Yes or No")
    smoker: str = Field(..., description="Yes or No")
    systolic_bp: float = Field(..., ge=70, le=260, description="Systolic Blood Pressure (mmHg)")
    diastolic_bp: float = Field(..., ge=40, le=160, description="Diastolic Blood Pressure (mmHg)")
    fasting_glucose: float = Field(..., ge=40, le=500, description="Fasting Glucose (mg/dL)")
    total_cholesterol: float = Field(..., ge=50, le=500, description="Total Cholesterol (mg/dL)")
    ldl: float = Field(..., ge=20, le=350, description="LDL Cholesterol (mg/dL)")
    hdl: float = Field(..., ge=10, le=150, description="HDL Cholesterol (mg/dL)")
    triglycerides: float = Field(..., ge=30, le=1200, description="Triglycerides (mg/dL)")
    platelet_count: float = Field(..., ge=20, le=1000, description="Platelets (10^9/L)")
    slcvd_score: float = Field(..., ge=0, le=100, description="SLCVD Score")
    who_cvd_risk: float = Field(..., ge=0, le=100, description="WHO Risk Category/Score")
    ast: Optional[float] = Field(25.0, ge=5, le=500, description="AST (U/L) - required for Fibrosis-Enhanced model")
    alt: Optional[float] = Field(25.0, ge=5, le=500, description="ALT (U/L) - required for Fibrosis-Enhanced model")
    fibrosis_stage: Optional[float] = Field(0.0, ge=0, le=4, description="Fibrosis Stage 0-4 - required for Fibrosis-Enhanced model")
    fib_4_score: Optional[float] = Field(None, description="FIB-4 Index (auto-calculated if omitted)")
    selected_model: Optional[str] = Field("Logistic Regression (Fibrosis-Enhanced)", description="Selected prediction model algorithm")


def load_dataset() -> tuple[pd.DataFrame, str]:
    if DATASET_PATH.exists():
        try:
            df = pd.read_csv(DATASET_PATH)
            missing = set(ALL_REQUIRED_COLUMNS) - set(df.columns)
            if not missing and len(df) >= 30:
                if "fibrosis_stage" in df.columns:
                    fib_map = {"F0": 0, "F1": 1, "F2": 2, "F3": 3, "F4": 4}
                    raw_fib = df["fibrosis_stage"]
                    if raw_fib.dtype == object or isinstance(raw_fib.iloc[0], str):
                        df["fibrosis_stage"] = raw_fib.astype(str).str.strip().str.upper().map(fib_map).fillna(raw_fib)
                    df["fibrosis_stage"] = pd.to_numeric(df["fibrosis_stage"], errors="coerce").fillna(0).astype(int)
                return df, "cohort"
        except Exception as e:
            print(f"Error loading {DATASET_PATH}: {e}")

    raise FileNotFoundError(
        f"Approved MASLD clinical cohort dataset not found or incomplete at {DATASET_PATH}. "
        f"Please ensure {DATASET_PATH.name} contains all mandatory clinical fields."
    )


def compute_dca_net_benefit(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> dict[str, list[float]]:
    n = len(y_true)
    prevalence = np.mean(y_true)
    model_nb = []
    all_nb = []
    none_nb = []

    for pt in thresholds:
        if pt == 0:
            pt = 0.001
        if pt >= 1:
            pt = 0.999
        y_pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        nb = (tp / n) - (fp / n) * (pt / (1.0 - pt))
        model_nb.append(round(float(nb), 4))
        all_val = prevalence - (1.0 - prevalence) * (pt / (1.0 - pt))
        all_nb.append(round(float(max(all_val, 0.0)), 4))
        none_nb.append(0.0)

    return {
        "thresholds": [round(float(t * 100), 1) for t in thresholds],
        "model": model_nb,
        "treat_all": all_nb,
        "treat_none": none_nb,
    }



def compute_calibration_deciles(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 8,
) -> dict[str, list[float]]:
    """Calculate mean predicted risk and observed event rate by bin."""

    df_cal = pd.DataFrame({
        "true": np.asarray(y_true).ravel(),
        "prob": np.asarray(y_prob).ravel(),
    })

    df_cal = df_cal.dropna(subset=["true", "prob"])
    df_cal = df_cal.sort_values("prob").reset_index(drop=True)

    mean_pred = []
    obs_freq = []
    bin_sizes = []

    if df_cal.empty:
        return {
            "predicted_risk_pct": mean_pred,
            "observed_event_pct": obs_freq,
            "bin_counts": bin_sizes,
        }

    n_actual_bins = max(1, min(int(n_bins), len(df_cal)))
    row_groups = np.array_split(
        np.arange(len(df_cal)), n_actual_bins
    )

    for row_indices in row_groups:
        if len(row_indices) == 0:
            continue

        bin_frame = df_cal.iloc[row_indices]

        mean_pred.append(
            round(float(bin_frame["prob"].mean() * 100), 1)
        )
        obs_freq.append(
            round(float(bin_frame["true"].mean() * 100), 1)
        )
        bin_sizes.append(int(len(bin_frame)))

    return {
        "predicted_risk_pct": mean_pred,
        "observed_event_pct": obs_freq,
        "bin_counts": bin_sizes,
    }



def compute_nri_idi(y_true: np.ndarray, p_base: np.ndarray, p_enh: np.ndarray, cutoff: float = 0.20) -> tuple[float, float]:
    events = y_true == 1
    nonevents = y_true == 0

    up_events = (p_enh[events] > p_base[events]).sum()
    down_events = (p_enh[events] < p_base[events]).sum()
    n_events = max(events.sum(), 1)
    p_up_events = (up_events - down_events) / n_events

    up_nonevents = (p_enh[nonevents] > p_base[nonevents]).sum()
    down_nonevents = (p_enh[nonevents] < p_base[nonevents]).sum()
    n_nonevents = max(nonevents.sum(), 1)
    p_down_nonevents = (down_nonevents - up_nonevents) / n_nonevents

    nri_continuous = float(p_up_events + p_down_nonevents)
    mean_diff_events = float(np.mean(p_enh[events] - p_base[events]))
    mean_diff_nonevents = float(np.mean(p_enh[nonevents] - p_base[nonevents]))
    idi = float(mean_diff_events - mean_diff_nonevents)

    return round(nri_continuous, 3), round(idi, 3)


def train_and_cache_models():
    df, source = load_dataset()
    schema = auto_detect_feature_schema(df)
    features_enh = schema["features_enh"]
    features_base = schema["features_base"]
    num_enh = schema["num_enh"]
    cat_enh = schema["cat_enh"]
    num_base = schema["num_base"]
    cat_base = schema["cat_base"]

    X_raw = df[features_enh].copy()

    # Robust handling of target cvd_event: supports categorical ('Yes'/'No') and numeric (1/0)
    raw_target = df[TARGET]
    if raw_target.dtype == object or isinstance(raw_target.iloc[0], str):
        y = raw_target.astype(str).str.strip().str.capitalize().map({"Yes": 1, "No": 0, "1": 1, "0": 0}).fillna(0).astype(int).values
    else:
        y = raw_target.astype(int).values

    num_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])

    prep_base = ColumnTransformer([
        ("num", num_tf, num_base),
        ("cat", cat_tf, cat_base),
    ])
    prep_enh = ColumnTransformer([
        ("num", num_tf, num_enh),
        ("cat", cat_tf, cat_enh),
    ])

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )

    X_train_base = prep_base.fit_transform(X_train_df[features_base])
    X_test_base = prep_base.transform(X_test_df[features_base])
    X_train_enh = prep_enh.fit_transform(X_train_df[features_enh])
    X_test_enh = prep_enh.transform(X_test_df[features_enh])

    algos = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42, C=1.0),
            LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42),
            RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42),
        ),
        "XGBoost": (
            XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.08, eval_metric="logloss", random_state=42),
            XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.08, eval_metric="logloss", random_state=42),
        ),
    }

    results: dict[str, dict[str, Any]] = {"baseline": {}, "enhanced": {}}
    models_fitted: dict[str, Any] = {"baseline": {}, "enhanced": {}}

    for name, (clf_b, clf_e) in algos.items():
        clf_b.fit(X_train_base, y_train)
        clf_e.fit(X_train_enh, y_train)
        models_fitted["baseline"][name] = clf_b
        models_fitted["enhanced"][name] = clf_e

        for spec, clf, X_eval in [("baseline", clf_b, X_test_base), ("enhanced", clf_e, X_test_enh)]:
            p = clf.predict_proba(X_eval)[:, 1]
            y_hat = (p >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_hat).ravel()
            auc = round(float(roc_auc_score(y_test, p)), 3)
            rec = round(float(recall_score(y_test, y_hat, zero_division=0)), 3)
            spec_val = round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0, 3)
            f1 = round(float(f1_score(y_test, y_hat, zero_division=0)), 3)
            acc = round(float(accuracy_score(y_test, y_hat)), 3)
            brier = round(float(brier_score_loss(y_test, p)), 3)

            results[spec][name] = {
                "auc": auc,
                "sensitivity": rec,
                "specificity": spec_val,
                "f1": f1,
                "accuracy": acc,
                "brier": brier,
                "probs": [round(float(x), 4) for x in p],
            }

    best_base_name = max(results["baseline"], key=lambda k: results["baseline"][k]["auc"])
    best_enh_name = max(results["enhanced"], key=lambda k: results["enhanced"][k]["auc"])
    best_base_model = models_fitted["baseline"][best_base_name]
    best_enh_model = models_fitted["enhanced"][best_enh_name]

    p_best_base = np.array(results["baseline"][best_base_name]["probs"])
    p_best_enh = np.array(results["enhanced"][best_enh_name]["probs"])

    delta_auc = round(results["enhanced"][best_enh_name]["auc"] - results["baseline"][best_base_name]["auc"], 3)
    nri, idi = compute_nri_idi(y_test, p_best_base, p_best_enh, cutoff=0.20)
    dca_data = compute_dca_net_benefit(y_test, p_best_enh, np.linspace(0.05, 0.50, 10))
    calib_data = compute_calibration_deciles(y_test, p_best_enh, n_bins=8)

    # Feature names
    num_cols_enh = num_enh
    cat_cols_enh = prep_enh.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_enh).tolist()
    feature_names = num_cols_enh + cat_cols_enh

    # Global SHAP
    global_shap: list[dict[str, Any]] = []
    shap_explainer = None
    try:
        sample_size = min(300, len(X_train_enh))
        bg_samples = shap.sample(X_train_enh, sample_size, random_state=42)
        if isinstance(best_enh_model, (RandomForestClassifier, XGBClassifier)):
            shap_explainer = shap.TreeExplainer(best_enh_model)
            shap_vals = shap_explainer.shap_values(X_test_enh)
            if isinstance(shap_vals, list):
                sv = shap_vals[1]
            elif hasattr(shap_vals, "ndim") and shap_vals.ndim == 3:
                sv = shap_vals[:, :, 1]
            else:
                sv = shap_vals
        else:
            masker = shap.maskers.Independent(bg_samples, max_samples=sample_size)
            shap_explainer = shap.LinearExplainer(best_enh_model, masker)
            sv = shap_explainer.shap_values(X_test_enh)

        mean_abs_shap = np.mean(np.abs(sv), axis=0)
        shap_df = pd.DataFrame({"feature": feature_names, "importance": mean_abs_shap})
        shap_df = shap_df.sort_values("importance", ascending=False).reset_index(drop=True)
        tot_imp = float(shap_df["importance"].sum()) if float(shap_df["importance"].sum()) > 0 else 1.0

        for _, row in shap_df.iterrows():
            feat = row["feature"]
            is_fib = any(f in feat.lower() for f in ["fibrosis", "fib_4", "ast", "alt"])
            global_shap.append({
                "feature": feat,
                "importance": round(float(row["importance"]), 4),
                "relative_pct": round(float((row["importance"] / tot_imp) * 100), 1),
                "is_fibrosis": is_fib,
            })
        sample_size_b = min(300, len(X_train_base))
        bg_samples_b = shap.sample(X_train_base, sample_size_b, random_state=42)
        masker_b = shap.maskers.Independent(bg_samples_b, max_samples=sample_size_b)
        shap_explainer_base = shap.LinearExplainer(best_base_model, masker_b)
    except Exception as e:
        print(f"SHAP explanation fallback: {e}")
        shap_explainer_base = None

    # Fibrosis-Diabetes synergy
    synergy_stats: dict[str, Any] = {}
    if "diabetes" in df.columns and "fibrosis_stage" in df.columns:
        df_synergy = df[["diabetes", "fibrosis_stage"]].copy()
        df_synergy["_target_bin"] = y
        synergy_table = df_synergy.groupby(["diabetes", "fibrosis_stage"])["_target_bin"].agg(["count", "mean"]).reset_index()
        synergy_records = []
        for _, r in synergy_table.iterrows():
            fib_raw = r["fibrosis_stage"]
            if isinstance(fib_raw, str):
                fib_num = {"F0": 0, "F1": 1, "F2": 2, "F3": 3, "F4": 4}.get(fib_raw.strip().upper(), 0)
            else:
                try:
                    fib_num = int(fib_raw)
                except Exception:
                    fib_num = 0
            synergy_records.append({
                "diabetes": str(r["diabetes"]),
                "fibrosis_stage": fib_num,
                "count": int(r["count"]),
                "cvd_rate_pct": round(float(r["mean"] * 100), 1),
            })
        synergy_stats["cohort_synergy"] = synergy_records

    # Clinical data quality audit: track missing fields handled by pipeline imputer
    nan_counts = df[FEATURES_ENHANCED].isnull().sum()
    missing_summary = {k: int(v) for k, v in nan_counts[nan_counts > 0].items()}

    # Update in-memory CACHE
    CACHE.update({
        "trained": True,
        "source": source,
        "prep_base": prep_base,
        "prep_enh": prep_enh,
        "feature_names": feature_names,
        "results": results,
        "models_fitted": models_fitted,
        "available_algorithms": list(algos.keys()),
        "best_base_model": best_base_model,
        "best_enh_model": best_enh_model,
        "incremental": {
            "delta_auc": delta_auc,
            "nri": nri,
            "idi": idi,
            "baseline_best": best_base_name,
            "enhanced_best": best_enh_name,
            "base_auc": results["baseline"][best_base_name]["auc"],
            "enh_auc": results["enhanced"][best_enh_name]["auc"],
        },
        "global_shap": global_shap,
        "shap_explainer": shap_explainer,
        "shap_explainer_base": shap_explainer_base,
        "dca": dca_data,
        "calibration": calib_data,
        "interactions": synergy_stats,
        "cohort_stats": {
            "total_patients": len(df),
            "cvd_event_rate": round(float(np.mean(y) * 100), 1),
            "total_missing_values": int(nan_counts.sum()),
            "missing_features": missing_summary,
            "fibrosis_distribution": df["fibrosis_stage"].value_counts(normalize=True).round(3).to_dict() if "fibrosis_stage" in df.columns else {},
        },
    })


def explain_patient(
    patient_dict: dict[str, Any],
    prep: Any,
    features_list: list[str],
    num_cols: list[str],
    cat_cols: list[str],
    model: Any,
    explainer: Any,
) -> tuple[float, list[dict[str, Any]], float]:
    df_single = pd.DataFrame([patient_dict])
    X_single = prep.transform(df_single[features_list])
    cat_feature_names = prep.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols).tolist()
    feature_names = num_cols + cat_feature_names

    if explainer is not None:
        try:
            shap_vals = explainer.shap_values(X_single)
            if isinstance(shap_vals, list):
                sv = shap_vals[1][0]
            elif hasattr(shap_vals, "ndim") and shap_vals.ndim == 3:
                sv = shap_vals[0, :, 1]
            elif hasattr(shap_vals, "ndim") and shap_vals.ndim == 2:
                sv = shap_vals[0]
            else:
                sv = np.array(shap_vals).flatten()
            base_val = getattr(explainer, "expected_value", 0.0)
            if isinstance(base_val, (list, np.ndarray)):
                base_val = float(base_val[1]) if len(base_val) > 1 else float(base_val[0])
            else:
                base_val = float(base_val)
        except Exception as e:
            print(f"Patient SHAP error, falling back: {e}")
            sv = np.zeros(len(feature_names))
            base_val = 0.0
    else:
        sv = np.zeros(len(feature_names))
        base_val = 0.0

    items: list[dict[str, Any]] = []
    fib_sum = 0.0
    total_abs = 0.0

    for name, val in zip(feature_names, sv):
        fval = float(val)
        abs_v = abs(fval)
        total_abs += abs_v
        is_fib = any(k in name.lower() for k in ["fibrosis", "fib_4", "ast", "alt"])
        if is_fib:
            fib_sum += abs_v

        items.append({
            "feature": name,
            "shap_value": round(fval, 4),
            "is_risk_increasing": bool(fval > 0),
            "is_fibrosis": bool(is_fib),
        })

    items.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    fib_pct = round((fib_sum / total_abs) * 100, 1) if total_abs > 0 else 0.0
    return float(base_val), items, float(fib_pct)


def generate_clinical_recommendations(patient: dict[str, Any], risk_level: str, prob: float, fib_contrib: float) -> list[dict[str, str]]:
    recs = []
    fib_stage = int(patient.get("fibrosis_stage", 0))
    has_diabetes = str(patient.get("diabetes", "No")).lower().startswith("y")
    ldl = float(patient.get("ldl", 100))

    if risk_level == "High":
        recs.append({
            "category": "Cardiovascular Risk Management",
            "icon": "heart",
            "action": "Intensive Cardiovascular Prevention Protocol",
            "detail": f"Estimated 10-year CVD event probability is {prob:.1f}% (High Risk). Schedule urgent cardiology consultation and initiate aggressive atherosclerotic plaque stabilization.",
        })
        recs.append({
            "category": "Lipid Lowering Therapy",
            "icon": "pill",
            "action": "High-Intensity Statin Therapy Indicated",
            "detail": f"LDL cholesterol is {ldl:.0f} mg/dL. In patients with MASLD and high CVD risk, initiate high-intensity statin (e.g., Atorvastatin 40-80 mg or Rosuvastatin 20 mg). Note: Statins are safe and protective in MASLD/fibrosis.",
        })
    elif risk_level == "Moderate":
        recs.append({
            "category": "Cardiovascular Risk Management",
            "icon": "heart",
            "action": "Targeted CVD Surveillance & Modifiable Factor Control",
            "detail": f"Intermediate risk ({prob:.1f}%). Optimize blood pressure (target SBP < 130 mmHg) and lipid fractions. Annual cardiovascular clinical review recommended.",
        })
        recs.append({
            "category": "Lipid Lowering Therapy",
            "icon": "pill",
            "action": "Moderate-Intensity Statin Consideration",
            "detail": "Target LDL-C reduction ≥ 30-50% with moderate-intensity statin therapy, paired with dietary atherogenic risk reduction.",
        })
    else:
        recs.append({
            "category": "Cardiovascular Risk Management",
            "icon": "heart",
            "action": "Routine Primary Prevention",
            "detail": f"Low estimated 10-year CVD risk ({prob:.1f}%). Continue routine health maintenance and metabolic parameter monitoring every 12 to 24 months.",
        })

    if fib_stage >= 3:
        recs.append({
            "category": "Hepatology Surveillance",
            "icon": "liver",
            "action": "Urgent Hepatology Referral & Advanced Fibrosis Protocol",
            "detail": f"Patient has F{fib_stage} severe fibrosis/cirrhosis (accounting for {fib_contrib:.1f}% of CVD risk). Perform 6-monthly abdominal ultrasound + AFP for hepatocellular carcinoma (HCC) screening and screen for esophageal varices.",
        })
    elif fib_stage == 2:
        recs.append({
            "category": "Hepatology Surveillance",
            "icon": "liver",
            "action": "Significant Fibrosis Monitoring",
            "detail": f"F2 moderate fibrosis identified. Repeat non-invasive assessment (FibroScan / transient elastography or FIB-4) in 12 months. Monitor liver biochemistry every 6 months.",
        })
    else:
        recs.append({
            "category": "Hepatology Surveillance",
            "icon": "liver",
            "action": "Non-Invasive Monitoring",
            "detail": "Early/mild MASLD (F0-F1). Reassess non-invasive fibrosis index (FIB-4) in 2 to 3 years if metabolic risk factors remain controlled.",
        })

    if has_diabetes or float(patient.get("fasting_glucose", 100)) >= 126:
        recs.append({
            "category": "Metabolic Pharmacotherapy",
            "icon": "activity",
            "action": "GLP-1 RA / SGLT2 Inhibitor Dual Cardio-Hepatic Therapy",
            "detail": "For concurrent MASLD and type 2 diabetes, prioritize GLP-1 receptor agonists (e.g., Semaglutide) or SGLT2 inhibitors (e.g., Empagliflozin/Dapagliflozin), which demonstrate documented efficacy in reducing major adverse cardiovascular events (MACE) and resolving steatohepatitis.",
        })

    return recs


@app.on_event("startup")
def startup_event():
    train_and_cache_models()


@app.get("/")
def read_root():
    return {
        "framework": "FastAPI",
        "service": "CardioLiver-AI Backend",
        "docs": "/docs",
        "status": "ready" if CACHE["trained"] else "initializing",
    }


@app.get("/api/status")
def get_status():
    if not CACHE["trained"]:
        train_and_cache_models()
    return {
        "status": "ready",
        "source": CACHE["source"],
        "cohort_name": "MASLD Clinical Patient Cohort — Sri Lanka Registry",
        "total_records": CACHE["cohort_stats"].get("total_patients", 1200),
        "cvd_event_rate": CACHE["cohort_stats"].get("cvd_event_rate", 22.2),
        "total_missing_values": CACHE["cohort_stats"].get("total_missing_values", 0),
        "missing_features": CACHE["cohort_stats"].get("missing_features", {}),
        "incremental_auc_gain": CACHE["incremental"].get("delta_auc", 0.035),
        "best_baseline_algorithm": CACHE["incremental"].get("baseline_best", "XGBoost"),
        "best_enhanced_algorithm": CACHE["incremental"].get("enhanced_best", "XGBoost"),
        "dataset_path": str(DATASET_PATH),
        "required_columns": ALL_REQUIRED_COLUMNS,
    }


@app.post("/api/predict")
def predict_patient(payload: PatientPayload):
    if not CACHE["trained"]:
        train_and_cache_models()

    patient = payload.dict()
    # Provide safe fallback values for fibrosis if omitted when Baseline is selected
    if patient.get("ast") is None:
        patient["ast"] = 25.0
    if patient.get("alt") is None:
        patient["alt"] = 25.0
    if patient.get("fibrosis_stage") is None:
        patient["fibrosis_stage"] = 0

    if patient.get("fib_4_score") is None or patient["fib_4_score"] == 0:
        age_v = patient["age"]
        ast_v = patient["ast"]
        alt_v = max(patient["alt"], 1.0)
        plt_v = max(patient["platelet_count"], 1.0)
        patient["fib_4_score"] = round((age_v * ast_v) / (plt_v * np.sqrt(alt_v)), 2)

    df_single = pd.DataFrame([patient])
    prep_base = CACHE["prep_base"]
    prep_enh = CACHE["prep_enh"]

    X_base = prep_base.transform(df_single[FEATURES_BASE])
    X_enh = prep_enh.transform(df_single[FEATURES_ENHANCED])

    selected_algo = str(patient.get("selected_model", "Logistic Regression (Fibrosis-Enhanced)"))
    models_fitted = CACHE.get("models_fitted", {})
    
    # Retrieve Logistic Regression models for Baseline and Fibrosis-Enhanced
    lr_base = models_fitted.get("baseline", {}).get("Logistic Regression", CACHE["best_base_model"])
    lr_enh  = models_fitted.get("enhanced", {}).get("Logistic Regression", CACHE["best_enh_model"])

    prob_base = float(lr_base.predict_proba(X_base)[:, 1][0])
    prob_enh  = float(lr_enh.predict_proba(X_enh)[:, 1][0])

    pct_base = round(prob_base * 100, 1)
    pct_enh  = round(prob_enh * 100, 1)
    fib_shift = round(pct_enh - pct_base, 1)

    is_baseline = "baseline" in selected_algo.lower() or "model a" in selected_algo.lower()
    if is_baseline:
        primary_prob = pct_base
        active_model_name = "Model A (Baseline Traditional Logistic Regression)"
        active_algorithm = "Logistic Regression"
        model_spec = "Baseline Traditional (16 features)"
        model_type = "baseline"
        features_count = 16
        base_val, shap_items, fib_contrib = explain_patient(
            patient, prep_base, FEATURES_BASE, NUMERIC_BASE, CATEGORICAL, lr_base, CACHE.get("shap_explainer_base")
        )
    else:
        primary_prob = pct_enh
        active_model_name = "Model B (Fibrosis-Enhanced Logistic Regression)"
        active_algorithm = "Logistic Regression"
        model_spec = "Fibrosis-Enhanced (20 features)"
        model_type = "enhanced"
        features_count = 20
        base_val, shap_items, fib_contrib = explain_patient(
            patient, prep_enh, FEATURES_ENHANCED, NUMERIC_ENHANCED, CATEGORICAL, lr_enh, CACHE.get("shap_explainer")
        )

    if primary_prob >= 20.0:
        risk_category = "High"
    elif primary_prob >= 10.0:
        risk_category = "Moderate"
    else:
        risk_category = "Low"

    recommendations = generate_clinical_recommendations(
        patient, risk_category, primary_prob, fib_contrib
    )

    top_factors = [x["feature"] for x in shap_items if x["is_risk_increasing"]][:4]

    return {
        "risk_category": risk_category,
        "probability": primary_prob,
        "enhanced_probability": pct_enh,
        "baseline_probability": pct_base,
        "fibrosis_risk_shift": fib_shift,
        "fibrosis_contribution_pct": fib_contrib,
        "fibrosis_significance": "Significant" if fib_contrib >= 15 else "Moderate" if fib_contrib >= 8 else "Minor",
        "key_factors": top_factors,
        "shap_explanations": shap_items,
        "base_value": round(float(base_val), 3),
        "recommendations": recommendations,
        "best_model_name": active_model_name,
        "algorithm_name": active_algorithm,
        "model_spec": model_spec,
        "model_type": model_type,
        "features_count": features_count,
        "selected_model": selected_algo,
        "available_models": ["Logistic Regression (Fibrosis-Enhanced)", "Logistic Regression (Baseline Traditional)"],
        "calculated_fib4": patient["fib_4_score"],
    }


@app.get("/api/model_comparison")
def get_model_comparison():
    if not CACHE["trained"]:
        train_and_cache_models()

    return {
        "metrics": CACHE["results"],
        "incremental": CACHE["incremental"],
        "global_shap": CACHE["global_shap"],
        "calibration": CACHE["calibration"],
        "dca": CACHE["dca"],
    }


@app.get("/api/interactions")
def get_interactions():
    if not CACHE["trained"]:
        train_and_cache_models()
    return {
        "fibrosis_diabetes_synergy": CACHE["interactions"],
        "cohort_stats": CACHE["cohort_stats"],
    }


@app.get("/api/download_template")
def download_template():
    if not TEMPLATE_PATH.exists():
        raise HTTPException(status_code=404, detail="Template file not found")
    return FileResponse(
        str(TEMPLATE_PATH),
        filename="masld_patient_cohort_template.csv",
        media_type="text/csv",
    )


@app.get("/api/download_models")
def download_models_bundle():
    zip_candidates = [
        ROOT / "backend" / "models" / "masld_cvd_champion_models.zip",
        ROOT / "masld_cvd_champion_models.zip",
        ROOT / "exported_models" / "masld_cvd_champion_models.zip",
    ]
    zip_path = next((p for p in zip_candidates if p.exists()), None)
    
    if not zip_path:
        # Dynamically build zip archive if not present
        import zipfile
        export_dir = ROOT / "backend" / "models"
        if not export_dir.exists():
            export_dir = ROOT / "exported_models"
        
        target_zip = ROOT / "backend" / "models" / "masld_cvd_champion_models.zip"
        target_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            for f in export_dir.glob("*"):
                if f.is_file() and not f.name.endswith(".zip"):
                    zipf.write(f, f.name)
        zip_path = target_zip

    return FileResponse(
        str(zip_path),
        filename="masld_cvd_champion_models.zip",
        media_type="application/zip",
    )


@app.get("/api/download_model")
def download_single_model(model_type: str = "enhanced"):
    fname = "best_model_enhanced.joblib" if model_type == "enhanced" else "best_model_baseline.joblib"
    paths = [
        ROOT / "backend" / "models" / fname,
        ROOT / "exported_models" / fname,
    ]
    model_path = next((p for p in paths if p.exists()), None)
    if not model_path:
        raise HTTPException(status_code=404, detail=f"Model artifact {fname} not found.")
    return FileResponse(
        str(model_path),
        filename=fname,
        media_type="application/octet-stream",
    )


@app.post("/api/upload")
async def upload_dataset(dataset: UploadFile = File(...)):
    if not dataset.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a valid .csv file.")

    try:
        content = await dataset.read()
        import io
        new_df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")

    missing = sorted(list(set(ALL_REQUIRED_COLUMNS) - set(new_df.columns)))
    if missing:
        raise HTTPException(
            status_code=400,
            detail={"error": "Missing mandatory clinical columns", "missing": missing},
        )

    if len(new_df) < 30:
        raise HTTPException(
            status_code=400,
            detail="Cohort too small. At least 30 patient records required for cross-validation.",
        )

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(DATASET_PATH, index=False)
    train_and_cache_models()

    return {
        "message": "Dataset successfully uploaded and validated! Models retrained with new patient cohort.",
        "records": len(new_df),
        "source": "cohort",
        "new_auc": CACHE["results"]["enhanced"][CACHE["incremental"]["enhanced_best"]]["auc"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)



