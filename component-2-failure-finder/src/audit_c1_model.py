"""C1 -> C2 audit: where does Sharuka's model wrongly clear patients?
[Standard] binomial test + Benjamini-Hochberg (same as the FIB-4 audit).
[Mine] the equal-scans setup and the bar for her model."""
import pandas as pd
import sys

from group_tools import prepare, score_groups, referral_bar, worth_a_scan

t = pd.read_csv("data/processed/c2_patient_table.csv")
path = sys.argv[1] if len(sys.argv) > 1 else "data/interim/c1_predictions_diabetic.csv"
c = pd.read_csv(path)
m = c.merge(t, left_on="SEQN", right_on="id")
print(f"C1 patients {len(c)}, matched in my table {len(m)}")

# Give the model the same number of scans as FIB-4 flat 1.3 on these patients
k = int((m["cleared_A"] == 0).sum())
rank = m["probability"].rank(method="first", ascending=False)
m["cleared_model"] = rank > k          # True = the model clears this patient

m = prepare(m, "model")
bar = referral_bar(m)
res = worth_a_scan(score_groups(m), bar)

print(f"patients {len(m)}, scans {k}, cleared {int(m['cleared'].sum())}, "
      f"wrongly cleared {int(m['wrong'].sum())}, bar {bar:.1%}")
print(f"groups scored {len(res)}, survive BH: {int((res['p_adjusted'] < 0.05).sum())}")
show = res.sort_values("miss_rate", ascending=False).head(8)
print("Highest miss rates (blind-spot map, not red flags):")
print(show[["group", "cases", "wrongly_cleared", "miss_rate"]].to_string(index=False))