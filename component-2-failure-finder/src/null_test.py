"""Null test: does the method raise false alarms when there is nothing to find?

Idea: keep every patient, but give each CLEARED patient a coin-flip fibrosis
result that comes up 'fibrosis' exactly at the bar. Now NO group is truly worse
than the bar. Run the whole audit on that fake data, many times.
If the method is honest, it should flag a group in very few runs.

Run from the repo root:  python component-2-failure-finder/src/null_test.py
"""
import sys
import numpy as np
import pandas as pd
from group_tools import prepare, score_groups, referral_bar, worth_a_scan

RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 200

t = pd.read_csv("data/processed/c2_patient_table.csv")
cleared = t["cleared_A"].astype(bool)
bar = t.loc[~cleared, "fibrosis"].mean()
print(f"Bar = {bar:.3f}. Cleared patients = {cleared.sum()}. Runs = {RUNS}")

flagged_runs = 0
for seed in range(RUNS):
    rng = np.random.default_rng(seed)
    fake = t.copy()
    # coin flip at exactly the bar for every cleared patient
    fake.loc[cleared, "fibrosis"] = (rng.random(cleared.sum()) < bar).astype(int)
    fake = prepare(fake, "flat")
    res = worth_a_scan(score_groups(fake), referral_bar(fake))
    if (res["p_adjusted"] <= 0.05).any():
        flagged_runs += 1

rate = flagged_runs / RUNS
print(f"Runs that flagged at least one group: {flagged_runs} of {RUNS} ({rate:.1%})")
print("PASS: false-alarm rate is at or below about 5%" if rate <= 0.05 else
      "CHECK: false-alarm rate is above 5% - the method needs a closer look")