"""Train the source model on NHANES adult diabetics and compare with FIB-4."""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_csv(Path("data/processed/nhanes_diabetic_cohort.csv"))

features = ["RIDAGEYR", "RIAGENDR", "LBXSASSI", "LBXSATSI",
            "LBXPLTSI", "BMXBMI", "BMXWAIST", "LBXGH"]
X = df[features]
y = df["label"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic regression": make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=1000)),
    "Gradient boosting": HistGradientBoostingClassifier(random_state=42),
}

fib4_flagged = (df["FIB4"] >= 1.3)
n_flag = fib4_flagged.sum()
sick = y == 1

print(f"Cohort: {len(df)} adult diabetics, {sick.sum()} with fibrosis\n")
print(f"FIB-4: AUC {roc_auc_score(y, df['FIB4']):.3f} | "
      f"flags {n_flag} people, catches {(fib4_flagged & sick).sum()} of {sick.sum()}")

for name, model in models.items():
    scores = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    top = pd.Series(scores).rank(ascending=False, method="first") <= n_flag
    caught = (top.values & sick.values).sum()
    print(f"{name}: AUC {roc_auc_score(y, scores):.3f} | "
          f"flags {n_flag} people, catches {caught} of {sick.sum()}")