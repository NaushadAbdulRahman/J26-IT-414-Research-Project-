"""First look at the liver stiffness outcome."""

import pandas as pd
from pathlib import Path

lux = pd.read_sas(Path("data/raw/P_LUX.xpt"), format="xport")

print("Exam status counts (LUAXSTAT):")
print(lux["LUAXSTAT"].value_counts(dropna=False).sort_index())

print("\nStiffness (LUXSMED) summary, all records:")
print(lux["LUXSMED"].describe())

complete = lux[lux["LUAXSTAT"] == 1]
print(f"\nComplete exams only: {len(complete)} rows")
print(complete["LUXSMED"].describe())

print("\nAt or above 8 kPa, complete exams:")
above = (complete["LUXSMED"] >= 8).sum()
valid = complete["LUXSMED"].notna().sum()
print(f"  {above} of {valid} = {100 * above / valid:.1f}%")