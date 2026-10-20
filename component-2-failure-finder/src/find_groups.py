"""F1 - blind-spot audit: score every group of 1-3 traits for one FIB-4 rule.
Run from the repo root, once per rule:
    python component-2-failure-finder\\src\\find_groups.py flat
    python component-2-failure-finder\\src\\find_groups.py age_adjusted
"""
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

IN = Path("data/processed/c2_patient_table.csv")
OUT_DIR = Path("data/processed")

MIN_PATIENTS = 30   # smallest group we will score
MIN_CASES = 10      # smallest number of fibrosis patients in a group
MAX_TRAITS = 3

RULE = sys.argv[1] if len(sys.argv) > 1 else "flat"
CLEARED_COL = {"flat": "cleared_A", "age_adjusted": "cleared_B"}[RULE]

# Fixed bins, written down BEFORE looking at any result: (edges, labels)
BIN_RULES = {
    "age":       ([0, 50, 65, 1000],     ["<50", "50-64", ">=65"]),
    "bmi":       ([0, 25, 30, 35, 1000], ["<25", "25-29.9", "30-34.9", ">=35"]),
    "ast":       ([0, 25, 40, 10000],    ["<25", "25-39", ">=40"]),
    "alt":       ([0, 25, 40, 10000],    ["<25", "25-39", ">=40"]),
    "platelets": ([0, 150, 300, 10000],  ["<150", "150-299", ">=300"]),
}
TRAITS = ["age", "sex", "bmi", "ast", "alt", "platelets"]

t = pd.read_csv(IN)
for trait, (edges, labels) in BIN_RULES.items():
    t[trait + "_bin"] = pd.cut(t[trait], bins=edges, labels=labels, right=False)
t["sex_bin"] = t["sex"]

t["cleared"] = t[CLEARED_COL].astype(int)
t["wrong"] = ((t[CLEARED_COL]) & (t["fibrosis"] == 1)).astype(int)  # cleared but has fibrosis

all_cases = t["fibrosis"].sum()
print(f"Rule: {RULE}")
print(f"Whole cohort: miss rate {100 * t['wrong'].sum() / all_cases:.1f}%, "
      f"fibrosis share among cleared {100 * t['wrong'].sum() / t['cleared'].sum():.1f}%")

parts = []
for k in range(1, MAX_TRAITS + 1):
    for traits in combinations(TRAITS, k):
        cols = [x + "_bin" for x in traits]
        g = t.groupby(cols, observed=True).agg(
            n=("id", "size"),
            cases=("fibrosis", "sum"),
            cleared=("cleared", "sum"),
            wrongly_cleared=("wrong", "sum"),
        ).reset_index()
        g["group"] = g.apply(
            lambda r: " AND ".join(f"{tr} {r[c]}" for tr, c in zip(traits, cols)), axis=1)
        g["n_traits"] = k
        parts.append(g[["group", "n_traits", "n", "cases", "cleared", "wrongly_cleared"]])

res = pd.concat(parts, ignore_index=True)
print(f"Groups before size filter: {len(res)}")
res = res[(res["n"] >= MIN_PATIENTS) & (res["cases"] >= MIN_CASES) & (res["cleared"] > 0)].copy()
print(f"Groups scored (n >= {MIN_PATIENTS}, cases >= {MIN_CASES}): {len(res)}")

res["miss_rate"] = res["wrongly_cleared"] / res["cases"]
res["share_among_cleared"] = res["wrongly_cleared"] / res["cleared"]
res = res.sort_values("miss_rate", ascending=False)

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 60)
print("\nTop 10 blind spots (highest miss rate):")
print(res.head(10).round(3).to_string(index=False))

out = OUT_DIR / f"c2_groups_{RULE}.csv"
res.to_csv(out, index=False)
print(f"\nSaved to {out}")