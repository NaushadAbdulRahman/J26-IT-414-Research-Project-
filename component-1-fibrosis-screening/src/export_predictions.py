"""
export_predictions.py  --  Component 1's predictions for Component 2 (Failure-Finder).

    python src/export_predictions.py

Writes outputs/c1_predictions_diabetic.csv with three columns:
    SEQN         NHANES participant ID
    probability  calibrated risk of significant fibrosis (FibroScan >= 8 kPa), 0 to 1
    referred     1 = sent for a FibroScan (decision Uncertain or Refer), 0 = cleared (No scan)

Out-of-fold: the cohort is split into 5 parts. Each part is scored by a model built
only on the other 4 parts (model, calibration AND the decision cut-offs), so no patient
is scored by a model that saw them. Same method as run_all.py.
Rows: patients in the cohort who report diagnosed diabetes (DIQ010 = 1).
"""
import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import load_data as ld
import analyses as an

ALPHA, SEED = 0.10, 42
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")


def main():
    df = ld.load_nhanes_with_ids().reset_index(drop=True)
    X, y = df[ld.FEATURES], df["label"].values
    base = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))

    prob = np.full(len(df), np.nan)
    referred = np.full(len(df), -1)
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y), 1):
        # inside the 4 training parts: fit on 75%, set the conformal cut-offs on the other 25%
        i_fit, i_cal = train_test_split(tr, test_size=0.25, stratify=y[tr], random_state=SEED)
        model = CalibratedClassifierCV(clone(base), method="sigmoid", cv=5).fit(X.iloc[i_fit], y[i_fit])
        q_pos, q_neg = an.conformal_thresholds(model.predict_proba(X.iloc[i_cal])[:, 1], y[i_cal], ALPHA)
        p = model.predict_proba(X.iloc[te])[:, 1]
        prob[te] = p
        referred[te] = (~an.cleared_by_model(p, q_pos, q_neg)).astype(int)
        print(f"fold {k}: scored {len(te)} patients, cut-offs {1 - q_pos:.1%} / {q_neg:.1%}")

    out = pd.DataFrame({"SEQN": df["SEQN"], "probability": prob.round(4), "referred": referred})
    diab = df["diabetes"] == 1
    out = out[diab].sort_values("SEQN").reset_index(drop=True)

    # checks before writing, so the file is never half-right
    assert out["SEQN"].is_unique and out["probability"].between(0, 1).all()
    assert set(out["referred"].unique()) <= {0, 1}

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "c1_predictions_diabetic.csv")
    out.to_csv(path, index=False)

    yd = df.loc[diab, "label"].values
    r = out["referred"].values
    missed = int(((r == 0) & (yd == 1)).sum())
    print(f"\nWrote {path}")
    print(f"Diabetic patients: {len(out)} | with fibrosis: {int(yd.sum())} | referred: {int(r.sum())} "
          f"| cleared but had fibrosis: {missed} ({missed / max(1, yd.sum()):.1%})")


if __name__ == "__main__":
    main()
