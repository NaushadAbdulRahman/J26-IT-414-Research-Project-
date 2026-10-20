"""
load_data.py  --  THE ONLY FILE YOU CHANGE WHEN NEW DATA ARRIVES.

Every other file just calls load_data() and expects a table with these columns:

    age, sex, bmi, waist, diabetes, ast, alt, ggt, platelets,
    albumin, hba1c, triglycerides, label

    sex       : 1 = male, 0 = female
    diabetes  : 1 = yes,  0 = no
    platelets : in 10^9 per litre (e.g. 250)
    label     : 1 = significant fibrosis on FibroScan (>= F2), 0 = not

Three sources, chosen with DATA_SOURCE below:
    "synthetic" : made-up practice data so the code runs today. NOT evidence.
    "nhanes"    : public US survey data (you download the files yourself).
    "local"     : the local cohort, saved as data/local.csv with the columns above.
"""
import os
import numpy as np
import pandas as pd

DATA_SOURCE = "nhanes"             # "synthetic", "nhanes" or "local"
COHORT = "masld"                   # "masld" (fatty liver, as in the proposal) or "all" adults
STIFFNESS_CUTOFF_KPA = 8.0         # FibroScan stiffness treated as >= F2
CAP_CUTOFF = 275                   # CAP (dB/m) treated as fatty liver

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")

FEATURES = ["age", "sex", "bmi", "waist", "diabetes", "ast", "alt", "ggt",
            "platelets", "albumin", "hba1c", "triglycerides"]


def _synthetic(n=3000, seed=42):
    """Practice data only. Shapes are plausible, numbers are invented."""
    rng = np.random.default_rng(seed)
    age = rng.normal(50, 14, n).clip(18, 85)
    sex = rng.integers(0, 2, n)
    bmi = rng.normal(27, 5, n).clip(16, 50)
    diabetes = (rng.random(n) < 0.12 + 0.25 * (bmi > 30) + 0.1 * (age > 55)).astype(int)

    # hidden "true risk" -> who really has fibrosis
    z = (-3.4 + 0.035 * (age - 50) + 0.09 * (bmi - 27) + 1.0 * diabetes
         + 0.2 * sex + rng.normal(0, 0.6, n))
    label = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    # blood tests shift when fibrosis is present, with noise
    ast = np.exp(rng.normal(3.15 + 0.45 * label, 0.38, n))
    alt = np.exp(rng.normal(3.25 + 0.25 * label, 0.45, n))
    ggt = np.exp(rng.normal(3.3 + 0.5 * label + 0.02 * (bmi - 27), 0.6, n))
    platelets = rng.normal(255 - 45 * label - 0.5 * (age - 50), 55, n).clip(60, 500)
    albumin = rng.normal(4.3 - 0.15 * label, 0.3, n).clip(2.5, 5.5)
    hba1c = rng.normal(5.5 + 1.3 * diabetes + 0.2 * label, 0.6, n).clip(4, 13)
    trig = np.exp(rng.normal(4.8 + 0.03 * (bmi - 27) + 0.15 * diabetes, 0.45, n))
    waist = (bmi * 3.3 + 8 * sex + rng.normal(0, 6, n)).clip(55, 160)

    return pd.DataFrame(dict(age=age, sex=sex, bmi=bmi, waist=waist, diabetes=diabetes,
                             ast=ast, alt=alt, ggt=ggt, platelets=platelets,
                             albumin=albumin, hba1c=hba1c, triglycerides=trig,
                             label=label)).round(2)


def _nhanes(folder="nhanes", suffix_or_prefix=("P_", "")):
    """
    NHANES liver elastography survey (public US data, CDC).

    Main data, 2017-March 2020: put these files in data/nhanes/
        P_DEMO  P_BMX  P_BIOPRO  P_CBC  P_GHB  P_DIQ  P_LUX
    Second, separate data for external validation, August 2021-August 2023:
    put these in data/nhanes_2021/
        DEMO_L  BMX_L  BIOPRO_L  CBC_L  GHB_L  DIQ_L  LUX_L

    Who is kept (COHORT = "masld", the population in the proposal):
      * adults (18+) with a complete FibroScan exam
      * reliable stiffness reading: IQR/median <= 30% (standard quality rule)
      * fatty liver: CAP >= 275 dB/m
      * at least one cardiometabolic risk factor (BMI >= 25, large waist,
        HbA1c >= 5.7% or diabetes, triglycerides >= 150 mg/dL)
    MASLD is approximated: NHANES alcohol and viral hepatitis data are not used here.
    COHORT = "all" keeps every adult with a reliable scan (sensitivity analysis).
    """
    d = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Folder not found: {d}")
    files = {f.upper(): f for f in os.listdir(d)}           # accepts .xpt or .XPT
    pre, suf = suffix_or_prefix

    def read(name):
        key = f"{pre}{name}{suf}.XPT"
        if key not in files:
            raise FileNotFoundError(f"{pre}{name}{suf}.xpt not found in {d}")
        return pd.read_sas(os.path.join(d, files[key]), format="xport")
    df = read("DEMO")[["SEQN", "RIDAGEYR", "RIAGENDR"]]
    for name, cols in [("BMX", ["BMXBMI", "BMXWAIST"]),
                       ("BIOPRO", ["LBXSASSI", "LBXSATSI", "LBXSGTSI", "LBXSAL", "LBXSTR"]),
                       ("CBC", ["LBXPLTSI"]),
                       ("GHB", ["LBXGH"]),
                       ("DIQ", ["DIQ010"]),
                       ("LUX", ["LUAXSTAT", "LUXSMED", "LUXSIQRM", "LUXCAPM"])]:
        how = "inner" if name in ("BIOPRO", "CBC", "LUX") else "left"   # optional results may be missing
        df = df.merge(read(name)[["SEQN"] + cols], on="SEQN", how=how)

    keep = (df.RIDAGEYR >= 18) & (df.LUAXSTAT == 1) & (df.LUXSIQRM <= 30)
    if COHORT == "masld":
        male = df.RIAGENDR == 1
        risk = ((df.BMXBMI >= 25) | ((df.BMXWAIST >= 94) & male) | ((df.BMXWAIST >= 80) & ~male)
                | (df.LBXGH >= 5.7) | (df.DIQ010 == 1) | (df.LBXSTR >= 150))
        keep &= (df.LUXCAPM >= CAP_CUTOFF) & risk
    df = df[keep]
    out = pd.DataFrame({
        "age": df.RIDAGEYR, "sex": (df.RIAGENDR == 1).astype(int),
        "bmi": df.BMXBMI, "waist": df.BMXWAIST,
        "diabetes": (df.DIQ010 == 1).astype(int),
        "ast": df.LBXSASSI, "alt": df.LBXSATSI, "ggt": df.LBXSGTSI,
        "platelets": df.LBXPLTSI, "albumin": df.LBXSAL,
        "hba1c": df.LBXGH, "triglycerides": df.LBXSTR,
        "label": (df.LUXSMED >= STIFFNESS_CUTOFF_KPA).astype(int),
        "SEQN": df.SEQN.astype(int),            # NHANES participant ID (for sharing results with the group)
    })
    # FIB-4 needs these four, so rows without them are dropped
    return out.dropna(subset=["age", "ast", "alt", "platelets"]).reset_index(drop=True)


def load_nhanes_with_ids():
    """The NHANES cohort with each participant's SEQN, for files shared with other components."""
    return _nhanes("nhanes", ("P_", ""))


def load_external():
    """The second NHANES dataset (2021-2023), or None if its files are not there yet."""
    try:
        return _nhanes("nhanes_2021", ("", "_L"))[FEATURES + ["label"]]
    except FileNotFoundError:
        return None


def load_data():
    if DATA_SOURCE == "synthetic":
        df = _synthetic()
    elif DATA_SOURCE == "nhanes":
        df = _nhanes("nhanes", ("P_", ""))
    elif DATA_SOURCE == "local":
        df = pd.read_csv(os.path.join(DATA_DIR, "local.csv"))
    else:
        raise ValueError("DATA_SOURCE must be synthetic, nhanes or local")

    missing = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(f"Data is missing columns: {missing}")
    return df[FEATURES + ["label"]]
