"""C1 -> C2 connection: put Sharuka's model next to FIB-4 on the same patients.
[Standard] reading the CSV and joining on id.
[Mine] the equal-scans comparison (same number of scans for both rules)."""
import pandas as pd
import sys

t = pd.read_csv("data/processed/c2_patient_table.csv")
path = sys.argv[1] if len(sys.argv) > 1 else "data/interim/c1_predictions_diabetic.csv"
c = pd.read_csv(path)
m = c.merge(t, left_on="SEQN", right_on="id")
print(f"C1 patients {len(c)}, matched in my table {len(m)}")


def summary(name, referred):
    scans = int(referred.sum())
    found = int(((referred == 1) & (m["fibrosis"] == 1)).sum())
    missed = int(m["fibrosis"].sum()) - found
    print(f"{name}: scans {scans}, fibrosis found {found}, wrongly cleared {missed}")


summary("C1 model (as sent)", m["referred"])

fib4_ref = (m["cleared_A"] == 0).astype(int)
summary("FIB-4 flat 1.3", fib4_ref)

k = int(fib4_ref.sum())
top_k = (m["probability"].rank(method="first", ascending=False) <= k).astype(int)
summary(f"C1 model, same {k} scans as FIB-4", top_k)