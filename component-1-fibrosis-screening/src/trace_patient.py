"""
trace_patient.py  --  shows, step by step, how one patient's output is calculated.

    python src/trace_patient.py            # Example C from the app
    python src/trace_patient.py A          # or A / B / C

Every number printed here is recomputed by hand from the saved model and then
checked against what the app shows, so you can follow the output back to the input.
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screening import Screener, fib4          # noqa: E402

EXAMPLES = {
    "A": dict(age=23, sex=1, bmi=27.6, waist=89.8, diabetes=0, ast=29, alt=85, ggt=95,
              platelets=299, albumin=4.7, hba1c=5.4, triglycerides=94),
    "B": dict(age=59, sex=0, bmi=41.8, waist=116.8, diabetes=0, ast=17, alt=22, ggt=28,
              platelets=284, albumin=4.4, hba1c=5.6, triglycerides=115),
    "C": dict(age=70, sex=1, bmi=34.6, waist=121.3, diabetes=1, ast=26, alt=25, ggt=79,
              platelets=224, albumin=3.7, hba1c=7.0, triglycerides=63),
}


def main(which="C"):
    S = Screener()
    x = pd.DataFrame([EXAMPLES[which]])[S.features].astype(float)
    imp, scale, lr = S.raw[0], S.raw[1], S.raw[-1]
    if not hasattr(lr, "coef_"):
        print("This walkthrough is written for logistic regression; the chosen model is", S.model_name)
        return

    print(f"\n=== Example {which} ===\n")
    print("STEP 1  Input: the 12 results")
    print("        " + ", ".join(f"{f}={v:g}" for f, v in x.iloc[0].items()))

    print("\nSTEP 2  Standardise: z = (value - training mean) / training SD")
    print("STEP 3  Logistic regression: score = intercept + sum(weight x z)\n")
    z = scale.transform(imp.transform(x))[0]
    total = lr.intercept_[0]
    print(f"        {'result':14}{'value':>8}{'mean':>9}{'SD':>8}{'z':>7}{'weight':>8}{'w x z':>8}")
    for f, v, m, s, zi, w in zip(S.features, x.iloc[0], scale.mean_, scale.scale_, z, lr.coef_[0]):
        print(f"        {f:14}{v:8.1f}{m:9.2f}{s:8.2f}{zi:7.2f}{w:8.3f}{w * zi:8.3f}")
        total += w * zi
    print(f"        intercept {lr.intercept_[0]:.3f}  ->  score (log-odds) = {total:.3f}")
    print(f"        raw risk = 1 / (1 + e^-score) = {1 / (1 + np.exp(-total)):.3f}")

    print(f"\nSTEP 4  Calibration ({getattr(S, 'model_name', '')}, method saved with the model)")
    p = float(S.risk(x)[0])
    if hasattr(S.model, "calibrated_classifiers_"):
        parts = []
        for k, cc in enumerate(S.model.calibrated_classifiers_, 1):
            d = cc.estimator.decision_function(x)[0]
            cal = cc.calibrators[0]
            pk = float(cal.predict(np.array([d]))[0])
            parts.append(pk)
            extra = f" a={cal.a_:.3f}, b={cal.b_:.3f}: 1/(1+e^(a*score+b))" if hasattr(cal, "a_") else ""
            print(f"        copy {k}: score {d:6.3f} ->{extra} = {pk:.3f}")
        print(f"        average of the {len(parts)} copies = {np.mean(parts):.3f}")
    print(f"        RISK SHOWN IN THE APP = {p:.1%}")

    print("\nSTEP 5  Decision layer (conformal cut-offs set on the calibration patients)")
    print(f"        NO SCAN below {S.low:.1%}   (90% of sick calibration patients scored above this)")
    print(f"        SCAN    above {S.high:.1%}  (90% of healthy calibration patients scored below this)")
    label = S.decide([p])[0]
    print(f"        {p:.1%} -> {label}")

    print("\nSTEP 6  SHAP reasons: SHAP = weight x (z - average z of reference patients)")
    bg = scale.transform(imp.transform(S.background)).mean(axis=0)
    sv = S.reasons(x)[0]
    for j in np.argsort(-np.abs(sv)):
        print(f"        {S.features[j]:14} {sv[j]:+.3f} = {lr.coef_[0][j]:.3f} x ({z[j]:.2f} - {bg[j]:.2f})"
              f"   {'raises' if sv[j] > 0 else 'lowers'} risk")
    print(f"        average patient {S.shap_base:.3f} + sum of SHAP {sv.sum():.3f} = {S.shap_base + sv.sum():.3f}"
          f"  (= the score in step 3)")

    f4 = float(fib4(x).iloc[0])
    print("\nFIB-4 for comparison = age x AST / (platelets x sqrt(ALT))")
    print(f"        = {x.age[0]:g} x {x.ast[0]:g} / ({x.platelets[0]:g} x sqrt({x.alt[0]:g})) = {f4:.2f}"
          f"  -> {'No scan' if f4 < 1.30 else 'Scan' if f4 > 2.67 else 'Uncertain'} (cut-offs 1.30 / 2.67)\n")


if __name__ == "__main__":
    main((sys.argv[1] if len(sys.argv) > 1 else "C").upper())
