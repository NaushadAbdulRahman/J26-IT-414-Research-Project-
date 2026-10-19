"""Guard 2, repeated: run the split-and-retest for many fixed seeds and report ALL of them.
Run from the repo root:  python component-2-failure-finder/src/repeat_splits.py flat
"""
import sys

import pandas as pd

import group_tools as gt

RULE = sys.argv[1] if len(sys.argv) > 1 else "flat"
SEEDS = list(range(20))   # fixed before running; every seed is reported, none dropped
ALPHA = 0.05

patients = gt.prepare(pd.read_csv("data/processed/c2_patient_table.csv"), RULE)

rows = []   # one row per seed
hits = []   # one row per (seed, group that held up on half 2)
for seed in SEEDS:
    half1 = patients.sample(frac=0.5, random_state=seed)
    half2 = patients.drop(half1.index)

    found = gt.worth_a_scan(gt.score_groups(half1), gt.referral_bar(half1))
    candidates = found[found["p_adjusted"] <= ALPHA]

    held = []
    if not candidates.empty:
        scored2 = gt.score_groups(half2)
        tested = scored2[scored2["group"].isin(candidates["group"])]
        if not tested.empty:
            tested = gt.worth_a_scan(tested, gt.referral_bar(half2))
            held = tested[tested["p_adjusted"] <= ALPHA]["group"].tolist()

    rows.append({"seed": seed, "candidates_half1": len(candidates), "held_up_half2": len(held)})
    for g in held:
        hits.append({"seed": seed, "group": g})

summary = pd.DataFrame(rows)
print(f"Rule {RULE}: {len(SEEDS)} random 50/50 splits (seeds {SEEDS[0]}-{SEEDS[-1]})")
print(summary.to_string(index=False))
print(f"\nSplits with at least one candidate in half 1: {(summary['candidates_half1'] > 0).sum()} of {len(SEEDS)}")
print(f"Splits with at least one group holding up in half 2: {(summary['held_up_half2'] > 0).sum()} of {len(SEEDS)}")

if hits:
    freq = pd.DataFrame(hits).groupby("group").size().sort_values(ascending=False)
    print("\nGroups that held up, and in how many splits:")
    print(freq.to_string())

summary.to_csv(f"data/processed/c2_repeat_summary_{RULE}.csv", index=False)
pd.DataFrame(hits).to_csv(f"data/processed/c2_repeat_hits_{RULE}.csv", index=False)