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

DATA_SOURCE = "synthetic"          # <-- change to "nhanes" or "local"
STIFFNESS_CUTOFF_KPA = 8.0         # FibroScan stiffness treated as >= F2

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


def _nhanes():
    """
    NHANES 2017-March 2020. Download these .XPT files from the CDC NHANES
    website and put them in data/nhanes/ :
        P_DEMO  P_BMX  P_BIOPRO  P_CBC  P_GHB  P_DIQ  P_LUX
    NOTE: written from the NHANES documentation but NOT yet run on the real
    files. If a column name is reported missing, check that file's codebook.
    """
    d = os.path.join(DATA_DIR, "nhanes")
    read = lambda name: pd.read_sas(os.path.join(d, name + ".XPT"))
    df = read("P_DEMO")[["SEQN", "RIDAGEYR", "RIAGENDR"]]
    for name, cols in [("P_BMX", ["BMXBMI", "BMXWAIST"]),
                       ("P_BIOPRO", ["LBXSASSI", "LBXSATSI", "LBXSGTSI", "LBXSAL", "LBXSTR"]),
                       ("P_CBC", ["LBXPLTSI"]),
                       ("P_GHB", ["LBXGH"]),
                       ("P_DIQ", ["DIQ010"]),
                       ("P_LUX", ["LUAXSTAT", "LUXSMED"])]:
        df = df.merge(read(name)[["SEQN"] + cols], on="SEQN", how="inner")

    df = df[(df.RIDAGEYR >= 18) & (df.LUAXSTAT == 1)]        # adults, complete scan
    out = pd.DataFrame({
        "age": df.RIDAGEYR, "sex": (df.RIAGENDR == 1).astype(int),
        "bmi": df.BMXBMI, "waist": df.BMXWAIST,
        "diabetes": (df.DIQ010 == 1).astype(int),
        "ast": df.LBXSASSI, "alt": df.LBXSATSI, "ggt": df.LBXSGTSI,
        "platelets": df.LBXPLTSI, "albumin": df.LBXSAL,
        "hba1c": df.LBXGH, "triglycerides": df.LBXSTR,
        "label": (df.LUXSMED >= STIFFNESS_CUTOFF_KPA).astype(int),
    })
    # FIB-4 needs these four, so rows without them are dropped
    return out.dropna(subset=["age", "ast", "alt", "platelets"]).reset_index(drop=True)


def load_data():
    if DATA_SOURCE == "synthetic":
        df = _synthetic()
    elif DATA_SOURCE == "nhanes":
        df = _nhanes()
    elif DATA_SOURCE == "local":
        df = pd.read_csv(os.path.join(DATA_DIR, "local.csv"))
    else:
        raise ValueError("DATA_SOURCE must be synthetic, nhanes or local")

    missing = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(f"Data is missing columns: {missing}")
    return df[FEATURES + ["label"]]
