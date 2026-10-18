"""Compare two FIB-4 clearing rules on the NHANES diabetic cohort."""
import pandas as pd

df = pd.read_csv("data/processed/nhanes_diabetic_cohort.csv")

rules = {
    "flat 1.3 for everyone": pd.Series(1.3, index=df.index),
    "age-adjusted (2.0 if 65+)": df["RIDAGEYR"].apply(lambda a: 2.0 if a >= 65 else 1.3),
}

for name, bar in rules.items():
    cleared = df["FIB4"] < bar
    missed = (cleared & (df["label"] == 1)).sum()
    print(f"{name}: cleared {cleared.sum()}, wrongly cleared (missed) {missed}")