"""
premise_test.py
Question: if a clinic only has scan results for patients it REFERRED
(FIB-4 >= 1.3), can a model trained on those patients still find the
fibrosis cases that FIB-4 missed (FIB-4 < 1.3)?

Method: 5-fold stratified CV on the NHANES diabetic cohort (everyone has a scan).
  Model ALL      : trained on every training-fold patient.
  Model REFERRED : trained only on training-fold patients with FIB-4 >= 1.3.
Both are scored on the SAME held-out patients (the full test fold).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

FEATURES = ["RIDAGEYR", "RIAGENDR", "LBXSASSI", "LBXSATSI",
            "LBXPLTSI", "BMXBMI", "BMXWAIST", "LBXGH"]
THRESHOLD = 1.3

df = pd.read_csv("data/processed/nhanes_diabetic_cohort.csv")
X = df[FEATURES].to_numpy(dtype=float)
y = df["label"].to_numpy(dtype=int)
fib4 = df["FIB4"].to_numpy(dtype=float)
referred = fib4 >= THRESHOLD          # who a FIB-4 pathway would send for a scan
missed = (y == 1) & ~referred          # sick, but FIB-4 says "fine"

def new_model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=1000))
score_all = np.zeros(len(df))
score_ref = np.zeros(len(df))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for tr, te in skf.split(X, y):
    m_all = new_model().fit(X[tr], y[tr])
    tr_ref = tr[referred[tr]]
    m_ref = new_model().fit(X[tr_ref], y[tr_ref])
    score_all[te] = m_all.predict_proba(X[te])[:, 1]
    score_ref[te] = m_ref.predict_proba(X[te])[:, 1]
    print(f"fold: trained ALL on {len(tr)} patients ({y[tr].sum()} sick); "
          f"REFERRED on {len(tr_ref)} patients ({y[tr_ref].sum()} sick)")

budget = int(referred.sum())           # same number of scans FIB-4 uses
def found(score):
    top = np.argsort(-score)[:budget]
    flag = np.zeros(len(df), dtype=bool)
    flag[top] = True
    return int((flag & (y == 1)).sum()), int((flag & missed).sum())

lines = []
lines.append(f"Patients: {len(df)} | sick: {y.sum()} | FIB-4>=1.3 (referred): {referred.sum()}")
lines.append(f"Sick patients FIB-4 would send home (MISSED): {missed.sum()}")
lines.append(f"Scan budget used for every method: {budget}")
lines.append("")
lines.append(f"{'Method':<28}{'AUC all':>9}{'AUC low-FIB4':>14}{'Found/sick':>12}{'Found/missed':>14}")
low = ~referred
rows = [("FIB-4 >= 1.3", fib4, None),
        ("Model trained on ALL", score_all, None),
        ("Model trained REFERRED only", score_ref, None)]
for name, s, _ in rows:
    f_all, f_miss = found(s)
    auc_all = roc_auc_score(y, s)
    auc_low = roc_auc_score(y[low], s[low])
    lines.append(f"{name:<28}{auc_all:>9.3f}{auc_low:>14.3f}{f_all:>8}/{y.sum():<3}{f_miss:>10}/{missed.sum():<3}")
lines.append("")
lines.append("AUC low-FIB4 = ranking quality inside the group FIB-4 sends home.")
lines.append("Found/missed = of the sick patients FIB-4 misses, how many the method puts in its top-budget list.")
out = "\n".join(lines)
print("\n" + out)
with open("outputs/premise_test_results.txt", "w") as f:
    f.write(out + "\n")