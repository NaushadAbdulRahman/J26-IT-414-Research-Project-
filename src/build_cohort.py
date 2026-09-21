"""Build the NHANES adult diabetic cohort and report outcome prevalence."""

import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")

def load(name):
    return pd.read_sas(RAW / f"{name}.xpt", format="xport")

lux = load("P_LUX")[["SEQN", "LUAXSTAT", "LUXSMED"]]
demo = load("P_DEMO")[["SEQN", "RIDAGEYR", "RIAGENDR", "WTMECPRP"]]
bio = load("P_BIOPRO")[["SEQN", "LBXSASSI", "LBXSATSI"]]
cbc = load("P_CBC")[["SEQN", "LBXPLTSI"]]
bmx = load("P_BMX")[["SEQN", "BMXBMI", "BMXWAIST"]]
diq = load("P_DIQ")[["SEQN", "DIQ010"]]
ghb = load("P_GHB")[["SEQN", "LBXGH"]]

df = lux
for other in [demo, bio, cbc, bmx, diq, ghb]:
    df = df.merge(other, on="SEQN", how="left")
print(f"Start (everyone with an elastography record): {len(df)}")

df = df[df["LUAXSTAT"] == 1]
print(f"After complete exams only:                    {len(df)}")

df = df[df["RIDAGEYR"] >= 18]
print(f"After adults (18+):                           {len(df)}")

df["diabetic"] = (df["DIQ010"] == 1) | (df["LBXGH"] >= 6.5)
df = df[df["diabetic"]]
print(f"After diabetic (diagnosed or HbA1c >= 6.5):   {len(df)}")

needed = ["LUXSMED", "RIDAGEYR", "LBXSASSI", "LBXSATSI", "LBXPLTSI"]
df = df.dropna(subset=needed)
print(f"After complete FIB-4 inputs:                  {len(df)}")

df["label"] = (df["LUXSMED"] >= 8).astype(int)
df["FIB4"] = (df["RIDAGEYR"] * df["LBXSASSI"]) / (df["LBXPLTSI"] * np.sqrt(df["LBXSATSI"]))

n_pos = df["label"].sum()
print(f"\nAt or above 8 kPa (unweighted): {n_pos} of {len(df)} = {100 * n_pos / len(df):.1f}%")

w = df["WTMECPRP"]
weighted = (w * df["label"]).sum() / w.sum()
print(f"At or above 8 kPa (weighted):   {100 * weighted:.1f}%")

sick = df[df["label"] == 1]
caught = (sick["FIB4"] >= 1.3).sum()
print(f"\nFIB-4 >= 1.3 catches {caught} of {len(sick)} fibrosis patients "
      f"= sensitivity {100 * caught / len(sick):.1f}%")

OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "nhanes_diabetic_cohort.csv", index=False)
print(f"\nSaved to {OUT / 'nhanes_diabetic_cohort.csv'}")