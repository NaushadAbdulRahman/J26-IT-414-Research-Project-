"""Guard 2: find groups on half 1, retest only those on half 2 (never seen before).
Run from the repo root:  python component-2-failure-finder/src/retest_groups.py flat 42
(the last number is the random seed for the split; default 42)
"""
import sys

import pandas as pd

import group_tools as gt

RULE = sys.argv[1] if len(sys.argv) > 1 else "flat"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
ALPHA = 0.05   # "holds up" = adjusted chance of luck <= 0.05 on half 2 (fixed before the run)

patients = gt.prepare(pd.read_csv("data/processed/c2_patient_table.csv"), RULE)

# 1. Split: half 1 = past paper (search here), half 2 = real exam (look only at the end)
half1 = patients.sample(frac=0.5, random_state=SEED)
half2 = patients.drop(half1.index)
print(f"Rule {RULE}, seed {SEED}: half 1 = {len(half1)} patients, half 2 = {len(half2)} patients")

# 2. Discovery on half 1 only
bar1 = gt.referral_bar(half1)
found = gt.worth_a_scan(gt.score_groups(half1), bar1)
candidates = found[found["p_adjusted"] <= ALPHA]
print(f"Half 1: bar {bar1:.3f}; {len(candidates)} of {len(found)} groups survive BH")
if candidates.empty:
    print("No candidates to retest. Closest groups in half 1:")
    close = ["group", "n", "cleared", "wrongly_cleared", "share_among_cleared", "p_luck", "p_adjusted"]
    print(found.sort_values("p_luck")[close].head(5).round(4).to_string(index=False))
    sys.exit()

# 3. Retest on half 2: only the candidates, using half 2's own bar
bar2 = gt.referral_bar(half2)
scored2 = gt.score_groups(half2)
tested = scored2[scored2["group"].isin(candidates["group"])]
if tested.empty:
    tested = tested.assign(p_luck=[], p_adjusted=[])
else:
    tested = gt.worth_a_scan(tested, bar2)

keep = ["group", "n", "cleared", "wrongly_cleared", "share_among_cleared"]
report = candidates[keep].merge(
    tested[keep + ["p_luck", "p_adjusted"]], on="group", how="left", suffixes=("_h1", "_h2"))
report["holds_up"] = report["p_adjusted"] <= ALPHA
report["note"] = ""
report.loc[report["n_h2"].isna(), "note"] = "too small in half 2"

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
cols = ["group", "n_h1", "share_among_cleared_h1", "n_h2", "cleared_h2",
        "wrongly_cleared_h2", "share_among_cleared_h2", "p_adjusted", "holds_up", "note"]
print(f"\nHalf 2 (never seen): bar {bar2:.3f}")
print(report.sort_values("p_adjusted")[cols].round(3).to_string(index=False))
print(f"\n{report['holds_up'].sum()} of {len(report)} candidates hold up on half 2 "
      f"(adjusted p <= {ALPHA})")
report.to_csv(f"data/processed/c2_retest_{RULE}_seed{SEED}.csv", index=False)