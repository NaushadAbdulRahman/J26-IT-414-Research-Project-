"""Step 2: worth-a-scan test plus guard 1 (BH) for every group.
Run from the repo root:  python component-2-failure-finder/src/test_groups.py flat
"""
import sys
import pandas as pd
from scipy.stats import binomtest, false_discovery_control

RULE = sys.argv[1] if len(sys.argv) > 1 else "flat"
CLEARED_COL = {"flat": "cleared_A", "age_adjusted": "cleared_B"}[RULE]

patients = pd.read_csv("data/processed/c2_patient_table.csv")
groups = pd.read_csv(f"data/processed/c2_groups_{RULE}.csv")

# The bar = fibrosis share among patients FIB-4 already refers
referred = patients[~patients[CLEARED_COL].astype(bool)]
bar = referred["fibrosis"].mean()
print(f"Rule {RULE}: bar = {bar:.3f} ({referred['fibrosis'].sum()} of {len(referred)} referred)")

def luck_chance(row):
    # If the true share were only the bar, how likely is a result this high?
    test = binomtest(int(row["wrongly_cleared"]), int(row["cleared"]), bar, alternative="greater")
    return test.pvalue

groups["p_luck"] = groups.apply(luck_chance, axis=1)

# Guard 1: BH. Adjusted number = the chance of luck after the stricter limits are applied
groups["p_adjusted"] = false_discovery_control(groups["p_luck"], method="bh")
groups["survives_bh"] = groups["p_adjusted"] <= 0.05

groups = groups.sort_values("p_luck")
print(f"{groups['survives_bh'].sum()} of {len(groups)} groups survive BH")
survivors = groups[groups["survives_bh"]]
print(survivors[["group", "n", "cleared", "wrongly_cleared", "share_among_cleared", "p_luck", "p_adjusted"]].to_string(index=False))
groups.to_csv(f"data/processed/c2_groups_{RULE}_tested.csv", index=False)