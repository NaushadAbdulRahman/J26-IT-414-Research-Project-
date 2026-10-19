"""Build the Component 2 patient table from the NHANES diabetic cohort.
One row per patient. Decisions of both FIB-4 rules come from the shared helper.
Run from the repo root:  python component-2-failure-finder\\src\\build_table.py
"""
import sys
from pathlib import Path

import pandas as pd

# let this script import the shared helper from the repo root
sys.path.append(str(Path(__file__).resolve().parents[2]))
from shared.fib4 import fib4, is_cleared

IN = Path("data/processed/nhanes_diabetic_cohort.csv")
OUT = Path("data/processed/c2_patient_table.csv")

raw = pd.read_csv(IN)

t = pd.DataFrame({
    "id": raw["SEQN"],
    "age": raw["RIDAGEYR"],
    "sex": raw["RIAGENDR"].map({1: "male", 2: "female"}),
    "bmi": raw["BMXBMI"],
    "ast": raw["LBXSASSI"],
    "alt": raw["LBXSATSI"],
    "platelets": raw["LBXPLTSI"],
    "kpa": raw["LUXSMED"],
    "fibrosis": raw["label"],        # 1 = FibroScan >= 8 kPa (the truth)
})

rows = list(zip(t["age"], t["ast"], t["alt"], t["platelets"]))
t["fib4"] = [fib4(a, s, l, p) for a, s, l, p in rows]
t["cleared_A"] = [is_cleared(a, s, l, p, "flat") for a, s, l, p in rows]
t["cleared_B"] = [is_cleared(a, s, l, p, "age_adjusted") for a, s, l, p in rows]

print(f"Patients: {len(t)}   with fibrosis: {t['fibrosis'].sum()}")
print("\nMissing values per column:")
print(t[["age", "sex", "bmi", "ast", "alt", "platelets"]].isna().sum())

for name, col in [("Rule A (flat 1.3)", "cleared_A"), ("Rule B (age-adjusted)", "cleared_B")]:
    cleared = t[col]
    wrongly = (cleared & (t["fibrosis"] == 1)).sum()
    referred = (~cleared).sum()
    found = ((~cleared) & (t["fibrosis"] == 1)).sum()
    print(f"\n{name}")
    print(f"  cleared {cleared.sum()}, wrongly cleared {wrongly}")
    print(f"  referred {referred}, fibrosis among referred {found} = {100 * found / referred:.1f}%")

t.to_csv(OUT, index=False)
print(f"\nSaved to {OUT}")