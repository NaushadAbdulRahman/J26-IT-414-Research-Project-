"""
premise_test_repeat.py
Same question as premise_test.py, repeated over 20 different random splits
to see whether the gap between the two models is stable or just noise.

Decision rule fixed BEFORE running:
  gap = (missed patients found by ALL model) - (found by REFERRED-only model)
  consistently < ~5 patients  -> selective-labels direction is weak
  consistently >= ~10 patients -> something to build on
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

FEATURES = ["RIDAGEYR", "RIAGENDR", "LBXSASSI", "LBXSATSI",
            "LBXPLTSI", "BMXBMI", "BMXWAIST", "LBXGH"]
THRESHOLD = 1.3
N_REPEATS = 20

df = pd.read_csv("data/processed/nhanes_diabetic_cohort.csv")
X = df[FEATURES].to_numpy(dtype=float)
y = df["label"].to_numpy(dtype=int)
fib4 = df["FIB4"].to_numpy(dtype=float)
referred = fib4 >= THRESHOLD
missed = (y == 1) & ~referred
low = ~referred
budget = int(referred.sum())

def new_model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=1000))

def found_missed(score):
    top = np.argsort(-score)[:budget]
    flag = np.zeros(len(df), dtype=bool)
    flag[top] = True
    return int((flag & missed).sum())

res = {"all": [], "ref": [], "auc_all": [], "auc_ref": []}
for seed in range(N_REPEATS):
    s_all = np.zeros(len(df))
    s_ref = np.zeros(len(df))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        s_all[te] = new_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        tr_ref = tr[referred[tr]]
        s_ref[te] = new_model().fit(X[tr_ref], y[tr_ref]).predict_proba(X[te])[:, 1]
    res["all"].append(found_missed(s_all))
    res["ref"].append(found_missed(s_ref))
    res["auc_all"].append(roc_auc_score(y[low], s_all[low]))
    res["auc_ref"].append(roc_auc_score(y[low], s_ref[low]))
    print(f"seed {seed:2d}: ALL found {res['all'][-1]}/{missed.sum()}  "
          f"REFERRED found {res['ref'][-1]}/{missed.sum()}  "
          f"gap {res['all'][-1]-res['ref'][-1]}")

a, r = np.array(res["all"]), np.array(res["ref"])
gap = a - r
lines = [
    f"Missed sick patients (FIB-4 < 1.3 but fibrosis): {missed.sum()}  | repeats: {N_REPEATS}",
    f"ALL model      found of missed: mean {a.mean():.1f}  SD {a.std(ddof=1):.1f}  range {a.min()}-{a.max()}",
    f"REFERRED model found of missed: mean {r.mean():.1f}  SD {r.std(ddof=1):.1f}  range {r.min()}-{r.max()}",
    f"GAP (ALL - REFERRED): mean {gap.mean():.1f}  SD {gap.std(ddof=1):.1f}  "
    f"2.5-97.5 percentile {np.percentile(gap,2.5):.1f} to {np.percentile(gap,97.5):.1f}  "
    f"min {gap.min()}  max {gap.max()}",
    f"Splits where the gap was >= 10: {(gap>=10).sum()} of {N_REPEATS};  < 5: {(gap<5).sum()} of {N_REPEATS}",
    f"AUC inside low-FIB4 group: ALL {np.mean(res['auc_all']):.3f}  REFERRED {np.mean(res['auc_ref']):.3f}",
    "",
    "Note: this varies the random split only. It does not capture the uncertainty from having",
    "only 1,249 patients; a patient-level bootstrap would be the next refinement.",
]
out = "\n".join(lines)
print("\n" + out)
with open("outputs/premise_test_repeat_results.txt", "w") as f:
    f.write(out + "\n")