"""
screening.py  --  the decision logic the app uses, kept separate from the screens.

Same rules as run_all.py:
  * risk      = calibrated probability of significant fibrosis
  * decision  = conformal decision layer -> SCAN / NO SCAN / UNSURE
  * FIB-4     = (age x AST) / (platelets x sqrt(ALT)), guideline cut-offs 1.30 and 2.67
  * reasons   = SHAP values (positive = raises risk)
"""
import os
import numpy as np
import pandas as pd
import joblib
from explain import shap_values_for, shap_base_value

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(HERE, "..", "outputs")

NAMES = {"age": "Age", "sex": "Sex", "bmi": "BMI", "waist": "Waist", "diabetes": "Diabetes",
         "ast": "AST", "alt": "ALT", "ggt": "GGT", "platelets": "Platelets",
         "albumin": "Albumin", "hba1c": "HbA1c", "triglycerides": "Triglycerides"}
UNITS = {"age": "years", "bmi": "kg/m²", "waist": "cm", "ast": "U/L", "alt": "U/L",
         "ggt": "U/L", "platelets": "×10⁹/L", "albumin": "g/dL", "hba1c": "%",
         "triglycerides": "mg/dL"}
FIB4_NEEDS = ["age", "ast", "alt", "platelets"]


try:
    # The group's shared FIB-4 formula (shared/fib4.py), so every component agrees.
    from shared.fib4 import fib4 as fib4_one
except ImportError:          # running this component on its own, outside the shared repo layout
    fib4_one = None


def fib4(df):
    """FIB-4 for every row: (age x AST) / (platelets x sqrt(ALT))."""
    if fib4_one is not None:
        return pd.Series([fib4_one(a, s, l, p) for a, s, l, p in
                          zip(df["age"], df["ast"], df["alt"], df["platelets"])], index=df.index)
    return (df["age"] * df["ast"]) / (df["platelets"] * np.sqrt(df["alt"]))


def fib4_zone(score):
    return np.where(score < 1.30, "NO SCAN", np.where(score > 2.67, "SCAN", "UNSURE"))


def shown_value(feature, value):
    """A patient's value written for people (Male/Female, Yes/No, units)."""
    if feature == "sex":
        return "Male" if value == 1 else "Female"
    if feature == "diabetes":
        return "Yes" if value == 1 else "No"
    return f"{value:g} {UNITS.get(feature, '')}".strip()


class Screener:
    def __init__(self):
        saved = joblib.load(os.path.join(OUTPUTS, "model.joblib"))
        self.model, self.raw = saved["model"], saved["raw"]
        self.q_pos, self.q_neg = saved["q_pos"], saved["q_neg"]
        self.features = saved["features"]
        self.background = saved["background"]
        self.model_name = saved["chosen_model"]
        self.data_source = saved["data_source"]
        # risk boundaries, for display: below LOW -> NO SCAN, above HIGH -> SCAN
        self.low, self.high = 1 - self.q_pos, self.q_neg
        # SHAP: every patient's explanation starts from this average model output
        self.shap_base = shap_base_value(self.raw, self.background)

    def risk(self, X):
        return self.model.predict_proba(X[self.features].astype(float))[:, 1]

    def decide(self, p):
        """Conformal decision layer, identical to model_zone() in run_all.py."""
        p = np.asarray(p)
        could_be_sick = (1 - p) <= self.q_pos
        could_be_healthy = p <= self.q_neg
        out = np.full(len(p), "UNSURE", dtype=object)
        out[could_be_healthy & ~could_be_sick] = "NO SCAN"
        out[could_be_sick & ~could_be_healthy] = "SCAN"
        return out

    def reasons(self, X):
        return shap_values_for(self.raw, self.background, X[self.features].astype(float))

    def screen_table(self, X):
        """Screen many patients at once; returns one row per patient."""
        p = self.risk(X)
        f4 = fib4(X).values
        sv = self.reasons(X)
        top = [", ".join(f"{NAMES[self.features[j]]} {'↑' if row[j] > 0 else '↓'}"
                         for j in np.argsort(-np.abs(row))[:3]) for row in sv]
        return pd.DataFrame({
            "Decision": self.decide(p),
            "Risk %": np.round(100 * p, 1),
            "FIB-4": np.round(f4, 2),
            "FIB-4 would say": fib4_zone(f4),
            "Top SHAP reasons (↑ raises, ↓ lowers risk)": top,
        }, index=X.index)


def parse_upload(df, features):
    """Accept a spreadsheet with the 12 columns; sex/diabetes as words or 0/1."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    sex = df["sex"].astype(str).str.strip().str.lower()
    df["sex"] = sex.map({"1": 1, "1.0": 1, "m": 1, "male": 1, "0": 0, "0.0": 0, "f": 0, "female": 0})
    dia = df["diabetes"].astype(str).str.strip().str.lower()
    df["diabetes"] = dia.map({"1": 1, "1.0": 1, "yes": 1, "y": 1, "0": 0, "0.0": 0, "no": 0, "n": 0})
    for c in features:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    incomplete = df[FIB4_NEEDS + ["sex", "diabetes"]].isna().any(axis=1)
    return df, incomplete
