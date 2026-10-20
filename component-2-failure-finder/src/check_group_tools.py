"""Check: group_tools reproduces the 275 groups and the 9 BH survivors on the full data.
Run from the repo root:  python component-2-failure-finder/src/check_group_tools.py
"""
import pandas as pd
import group_tools as gt

t = pd.read_csv("data/processed/c2_patient_table.csv")
t = gt.prepare(t, "flat")
res = gt.score_groups(t)
bar = gt.referral_bar(t)
res = gt.worth_a_scan(res, bar)

print(f"groups scored: {len(res)}   (expect 275)")
print(f"bar: {bar:.3f}   (expect 0.275)")
print(f"survive BH: {(res['p_adjusted'] <= 0.05).sum()}   (expect 9)")