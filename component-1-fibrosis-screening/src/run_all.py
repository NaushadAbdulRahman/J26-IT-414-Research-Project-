"""
run_all.py  --  runs the whole component, start to finish.

    python src/run_all.py

Steps (same numbers as the build diagram):
    1 Data   2 Preprocessing   3 ML model   4 Calibration
    5 Decision layer   6 Explanation   7 Output  + FIB-4 benchmark
Everything it produces is saved in the outputs/ folder.
"""
import os, json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss

import load_data as ld
from load_data import load_data, FEATURES
from explain import shap_values_for
import analyses as an

ALPHA = 0.10      # allowed miss rate: at most 10% of sick patients may be cleared
SEED = 42
TIE_MARGIN = 0.01  # if models are this close in AUROC, the simpler one is chosen
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
os.makedirs(OUT, exist_ok=True)


# ---------- FIB-4 benchmark ----------
def fib4(df):
    """FIB-4 = (age x AST) / (platelets x sqrt(ALT))"""
    return (df.age * df.ast) / (df.platelets * np.sqrt(df.alt))


def fib4_zone(score):
    return np.where(score < 1.30, "clear", np.where(score > 2.67, "refer", "uncertain"))


# ---------- Step 5: conformal decision layer ----------
def conformal_thresholds(p_cal, y_cal, alpha):
    """
    Class-conditional split conformal prediction.
    We look at patients the model has NOT trained on (the conformal set):
      * among truly SICK patients, find the probability below which only
        alpha of them fall  -> clearing below this misses at most ~alpha.
      * among truly HEALTHY patients, find the probability above which only
        alpha of them fall  -> referring above this is rarely a false alarm.
    """
    def q(scores):
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return np.quantile(scores, level, method="higher")
    q_pos = q(1 - p_cal[y_cal == 1])     # scores for sick patients
    q_neg = q(p_cal[y_cal == 0])         # scores for healthy patients
    return q_pos, q_neg


def model_zone(p, q_pos, q_neg):
    could_be_sick = (1 - p) <= q_pos
    could_be_healthy = p <= q_neg
    zone = np.full(len(p), "uncertain", dtype=object)
    zone[could_be_healthy & ~could_be_sick] = "clear"
    zone[could_be_sick & ~could_be_healthy] = "refer"
    return zone


def zone_summary(zone, y):
    n, n_sick = len(y), int(y.sum())
    missed = int(((zone == "clear") & (y == 1)).sum())
    return {
        "cleared_no_scan": int((zone == "clear").sum()),
        "uncertain": int((zone == "uncertain").sum()),
        "refer": int((zone == "refer").sum()),
        "scans_needed_pct": round(100 * float((zone != "clear").mean()), 1),
        "sick_patients_missed": missed,
        "miss_rate_pct": round(100 * missed / n_sick, 1),
    }


def net_benefit(y, treat, pt):
    tp = ((treat == 1) & (y == 1)).sum()
    fp = ((treat == 1) & (y == 0)).sum()
    return tp / len(y) - fp / len(y) * pt / (1 - pt)


# ---------- fair comparison helpers ----------
# A "rule-out" rule: patients whose score is below t are NOT scanned.
def rule_out(score, t, y):
    """% of patients sent for a scan, and % of sick patients missed."""
    clear = score < t
    return (round(100 * float((~clear).mean()), 1),
            round(100 * float((clear & (y == 1)).sum() / y.sum()), 1))


def threshold_for_miss_rate(score, y, target):
    """Highest cut-off that misses at most `target` of the sick patients."""
    sick = np.sort(score[y == 1])
    return sick[int(np.floor(target * len(sick)))]


def threshold_for_scan_budget(score, budget):
    """Cut-off that sends about `budget` of patients for a scan."""
    return np.quantile(score, 1 - budget)


def fib4_age_adjusted_clear(f4, age):
    """Guideline rule: low cut-off 1.30, or 2.0 for patients aged 65+."""
    return np.where(age >= 65, f4 < 2.0, f4 < 1.30)


def all_adults_check(pipeline, cal_method="isotonic"):
    """Sensitivity analysis: the same method on ALL adults with a reliable scan,
    not only those with fatty liver."""
    if ld.DATA_SOURCE != "nhanes":
        return None
    saved, ld.COHORT = ld.COHORT, "all"
    try:
        d = load_data()
    finally:
        ld.COHORT = saved
    X, y = d[FEATURES], d["label"].values
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)
    m = (clone(pipeline) if cal_method == "none" else
         CalibratedClassifierCV(clone(pipeline), method=cal_method, cv=5)).fit(X_tr, y_tr)
    pc, pt = m.predict_proba(X_cal)[:, 1], m.predict_proba(X_te)[:, 1]
    qp, qn = conformal_thresholds(pc, y_cal, ALPHA)
    f4 = fib4(X_te).values
    s_m, m_m = an.scans_and_misses(an.cleared_by_model(pt, qp, qn), y_te)
    s_f, m_f = an.scans_and_misses(f4 < 1.30, y_te)
    return {"patients": int(len(y)), "with_fibrosis": int(y.sum()),
            "auroc": {"model": round(roc_auc_score(y_te, pt), 3), "fib4": round(roc_auc_score(y_te, f4), 3)},
            "model_three_zone": {"scans_pct": s_m, "miss_rate_pct": m_m},
            "fib4_guideline_1_30": {"scans_pct": s_f, "miss_rate_pct": m_f}}


def main():
    # ---------- Step 1: data ----------
    df = load_data()
    X, y = df[FEATURES], df["label"].values
    print(f"Data source: {ld.DATA_SOURCE} ({ld.COHORT}) | patients: {len(df)} | "
          f"with fibrosis: {y.mean():.1%}")

    # ---------- Step 2: split (test set is locked away until the end) ----------
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4,
                                                stratify=y, random_state=SEED)
    X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5,
                                                stratify=y_tmp, random_state=SEED)

    # ---------- Step 3: try several models, keep the best ----------
    # Listed simplest first. Each is scored by 5-fold cross-validation on the
    # TRAINING data only, so the test set plays no part in choosing the model.
    fill = lambda: SimpleImputer(strategy="median")        # fills missing lab values
    candidates = {
        "Logistic regression": make_pipeline(fill(), StandardScaler(),
                                             LogisticRegression(max_iter=2000)),
        "Random forest": make_pipeline(fill(), RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, random_state=SEED)),
        "Gradient boosting (XGBoost)": make_pipeline(fill(), XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=SEED)),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv_auroc = {name: float(cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc").mean())
                for name, m in candidates.items()}
    best_score = max(cv_auroc.values())
    # simplest model that is within TIE_MARGIN of the best score
    chosen = next(n for n in candidates if cv_auroc[n] >= best_score - TIE_MARGIN)
    print("Cross-validated AUROC:", {k: round(v, 3) for k, v in cv_auroc.items()})
    print("Chosen model:", chosen)

    # ---------- Step 4: calibration ----------
    # Calibration makes "30% risk" mean about 30 in 100. Which method to use is
    # decided like the model: by 5-fold cross-validated Brier score on the TRAINING
    # data only (lower is better). "none" is allowed to win if calibrating does not help.
    raw = clone(candidates[chosen]).fit(X_tr, y_tr)        # uncalibrated, for comparison + SHAP
    makers = {"none": lambda: clone(candidates[chosen]),
              "sigmoid": lambda: CalibratedClassifierCV(clone(candidates[chosen]), method="sigmoid", cv=5),
              "isotonic": lambda: CalibratedClassifierCV(clone(candidates[chosen]), method="isotonic", cv=5)}
    cal_cv = {k: float(brier_score_loss(y_tr, cross_val_predict(f(), X_tr, y_tr, cv=cv,
                                                                method="predict_proba")[:, 1]))
              for k, f in makers.items()}
    cal_method = min(cal_cv, key=cal_cv.get)
    print("Cross-validated Brier by calibration method:", {k: round(v, 4) for k, v in cal_cv.items()},
          "-> chosen:", cal_method)
    model = raw if cal_method == "none" else makers[cal_method]().fit(X_tr, y_tr)

    p_cal = model.predict_proba(X_cal)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]
    p_raw = raw.predict_proba(X_te)[:, 1]

    # ---------- Step 5: decision layer ----------
    q_pos, q_neg = conformal_thresholds(p_cal, y_cal, ALPHA)
    zone_m = model_zone(p_te, q_pos, q_neg)

    # ---------- FIB-4 on the SAME test patients ----------
    f4 = fib4(X_te).values
    zone_f = fib4_zone(f4)

    results = {
        "data_source": ld.DATA_SOURCE,
        "cohort": ld.COHORT,
        "n_patients": int(len(y)),
        "n_with_fibrosis": int(y.sum()),
        "split": {"train": int(len(y_tr)), "calibration": int(len(y_cal)), "test": int(len(y_te))},
        "n_test": int(len(y_te)),
        "n_test_with_fibrosis": int(y_te.sum()),
        "allowed_miss_rate_pct": 100 * ALPHA,
        "chosen_model": chosen,
        "calibration_method": cal_method,
        "cross_validated_brier_by_calibration_method": {k: round(v, 4) for k, v in cal_cv.items()},
        "cross_validated_auroc_on_training_data": {k: round(v, 3) for k, v in cv_auroc.items()},
        "auroc": {"fib4": round(roc_auc_score(y_te, f4), 3),
                  "model": round(roc_auc_score(y_te, p_te), 3)},
        "brier": {"model_uncalibrated": round(brier_score_loss(y_te, p_raw), 4),
                  "model_calibrated": round(brier_score_loss(y_te, p_te), 4)},
        "fib4": zone_summary(zone_f, y_te),
        "model": zone_summary(zone_m, y_te),
    }
    # ---------- Fair comparison with FIB-4 ----------
    # Every cut-off is chosen on the CALIBRATION patients. The test patients
    # are only used to report what happens, so neither side is tuned on them.
    f4_cal = fib4(X_cal).values
    fair = {}
    for target in (0.05, 0.10):
        t_f = threshold_for_miss_rate(f4_cal, y_cal, target)
        t_m = threshold_for_miss_rate(p_cal, y_cal, target)
        f_scan, f_miss = rule_out(f4, t_f, y_te)
        m_scan, m_miss = rule_out(p_te, t_m, y_te)
        fair[f"same_miss_target_{int(target * 100)}pct"] = {
            "fib4": {"cutoff": round(float(t_f), 2), "scans_pct": f_scan, "miss_rate_pct": f_miss},
            "model": {"cutoff": round(float(t_m), 3), "scans_pct": m_scan, "miss_rate_pct": m_miss},
        }

    # same number of scans as guideline FIB-4 (budget measured on calibration set)
    budget = float((f4_cal >= 1.30).mean())
    t_m = threshold_for_scan_budget(p_cal, budget)
    f_scan, f_miss = rule_out(f4, 1.30, y_te)
    m_scan, m_miss = rule_out(p_te, t_m, y_te)
    fair["same_scans_as_guideline_fib4"] = {
        "budget_pct_from_calibration_set": round(100 * budget, 1),
        "fib4": {"cutoff": 1.30, "scans_pct": f_scan, "miss_rate_pct": f_miss},
        "model": {"cutoff": round(float(t_m), 3), "scans_pct": m_scan, "miss_rate_pct": m_miss},
    }

    # guideline FIB-4 with age-adjusted low cut-off (2.0 for age 65+)
    clear_age = fib4_age_adjusted_clear(f4, X_te.age.values)
    fair["fib4_age_adjusted"] = {
        "scans_pct": round(100 * float((~clear_age).mean()), 1),
        "miss_rate_pct": round(100 * float((clear_age & (y_te == 1)).sum() / y_te.sum()), 1),
    }

    # what the model does with the patients FIB-4 calls "unsure"
    grey = zone_f == "uncertain"
    fair["fib4_grey_zone"] = {
        "patients": int(grey.sum()),
        "sick_patients": int(y_te[grey].sum()),
        "model_cleared": int((grey & (zone_m == "clear")).sum()),
        "sick_among_model_cleared": int((grey & (zone_m == "clear") & (y_te == 1)).sum()),
        "model_unsure": int((grey & (zone_m == "uncertain")).sum()),
        "model_referred": int((grey & (zone_m == "refer")).sum()),
        "sick_among_model_referred": int((grey & (zone_m == "refer") & (y_te == 1)).sum()),
    }
    results["fair_comparison"] = fair

    # ---------- Proposal objectives and research questions ----------
    print("Running the extra analyses (bootstrap, stress test, budgets, RQ3)...")
    cleared_m, cleared_f = zone_m == "clear", f4 < 1.30
    results["uncertainty_95ci"] = an.uncertainty(y_te, p_te, f4, cleared_m, cleared_f)
    results["calibration"] = an.calibration_report(y_te, p_raw, p_te)
    results["uncalibrated_benchmark"] = an.uncalibrated_benchmark(raw, X_cal, y_cal, X_te, y_te, ALPHA)
    X_pool, y_pool = pd.concat([X_cal, X_te]), np.concatenate([y_cal, y_te])
    stress = an.guarantee_stress_test(model, X_pool, y_pool, ALPHA)
    results["guarantee_stress_test"] = {k: v for k, v in stress.items() if not k.startswith("_")}
    results["scan_budget"] = an.scan_budget_table(p_cal, f4_cal, p_te, f4, y_te)
    results["composite_vs_decomposed"] = an.composite_vs_decomposed(
        candidates[chosen], X_tr, y_tr, X_te, y_te, FEATURES, cal_method)
    results["sensitivity_all_adults"] = all_adults_check(candidates[chosen], cal_method)

    print(json.dumps(results, indent=2))
    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2)

    # ---------- Trade-off chart: scans needed vs sick patients missed ----------
    def curve(score):
        pts = [rule_out(score, t, y_te) for t in np.unique(np.quantile(score, np.linspace(0, 1, 201)))]
        return [a for a, b in pts], [b for a, b in pts]
    plt.figure(figsize=(6.5, 5))
    xs_f, ys_f = curve(f4)
    xs_m, ys_m = curve(p_te)
    plt.plot(xs_f, ys_f, "--", color="black", label="FIB-4 (any cut-off)")
    plt.plot(xs_m, ys_m, "-", color="black", label=f"Model: {chosen} (any cut-off)")
    s_g, m_g = rule_out(f4, 1.30, y_te)
    plt.plot(s_g, m_g, "o", color="black", mfc="white", ms=9, label="FIB-4 guideline (1.30)")
    plt.plot(fair["fib4_age_adjusted"]["scans_pct"], fair["fib4_age_adjusted"]["miss_rate_pct"],
             "^", color="black", mfc="white", ms=9, label="FIB-4 age-adjusted")
    zs = results["model"]
    plt.plot(zs["scans_needed_pct"], zs["miss_rate_pct"], "s", color="black", ms=9,
             label="Model: three-zone decision")
    plt.xlabel("% of patients sent for a FibroScan")
    plt.ylabel("% of sick patients missed")
    plt.title("Scans needed vs sick patients missed (lower-left is better)")
    plt.xlim(0, 100); plt.ylim(0, 100); plt.grid(color="0.85", lw=0.6)
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "tradeoff_curve.png"), dpi=200); plt.close()

    # ---------- charts (black and white) ----------
    plt.figure(figsize=(5, 5))
    for p, name, ls in [(p_raw, "Before calibration", "--"), (p_te, "After calibration", "-")]:
        frac, mean = calibration_curve(y_te, p, n_bins=8, strategy="quantile")
        plt.plot(mean, frac, ls, color="black", marker="o", label=name)
    plt.plot([0, 1], [0, 1], ":", color="grey", label="Perfect")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed proportion")
    plt.title("Calibration curve"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "calibration_curve.png"), dpi=200); plt.close()

    pts = np.linspace(0.05, 0.5, 30)
    plt.figure(figsize=(6, 4.5))
    plt.plot(pts, [net_benefit(y_te, (p_te >= t).astype(int), t) for t in pts],
             "-", color="black", label="Model")
    plt.plot(pts, [net_benefit(y_te, (f4 >= 1.30).astype(int), t) for t in pts],
             "--", color="black", label="FIB-4 (>= 1.30)")
    plt.plot(pts, [net_benefit(y_te, np.ones_like(y_te), t) for t in pts],
             ":", color="grey", label="Scan everyone")
    plt.axhline(0, color="grey", lw=0.8)
    plt.xlabel("Threshold probability"); plt.ylabel("Net benefit")
    plt.title("Decision curve analysis"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "decision_curve.png"), dpi=200); plt.close()

    labels = ["clear", "uncertain", "refer"]
    w = 0.38; xs = np.arange(3)
    plt.figure(figsize=(6, 4.5))
    plt.bar(xs - w / 2, [(zone_f == l).mean() * 100 for l in labels], w,
            color="white", edgecolor="black", hatch="//", label="FIB-4")
    plt.bar(xs + w / 2, [(zone_m == l).mean() * 100 for l in labels], w,
            color="grey", edgecolor="black", label="Model")
    plt.xticks(xs, ["No scan", "Unsure", "Scan"]); plt.ylabel("% of patients")
    plt.title("Where patients are sent"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "zones_vs_fib4.png"), dpi=200); plt.close()

    # ---------- Step 6: SHAP explanation ----------
    sv = shap_values_for(raw, X_tr, X_te)
    shap.summary_plot(sv, X_te, show=False, color_bar=True)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "shap_summary.png"), dpi=200); plt.close()
    results["shap_alignment"] = an.shap_alignment(sv, FEATURES)

    # ---------- External validation on a different NHANES cycle (if downloaded) ----------
    ext = ld.load_external() if ld.DATA_SOURCE == "nhanes" else None
    results["external_validation"] = (an.external_validation(model, q_pos, q_neg, ext, FEATURES)
                                      if ext is not None else None)
    print("External validation:", results["external_validation"] or "files not in data/nhanes_2021/ yet")
    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2)

    # ---------- RQ1 chart: miss rate over many new splits ----------
    plt.figure(figsize=(6.5, 4.2))
    plt.hist(stress["_misses"], bins=25, color="0.75", edgecolor="black")
    plt.axvline(100 * ALPHA, color="black", ls="--", label=f"Allowed: {100 * ALPHA:.0f}%")
    plt.axvline(stress["mean_miss_rate_pct"], color="black", ls="-",
                label=f"Average: {stress['mean_miss_rate_pct']:.1f}%")
    plt.xlabel("% of sick patients cleared (missed) on a new split")
    plt.ylabel(f"Number of splits (of {stress['repeats']})")
    plt.title("Safety bound on 500 new calibration/test splits"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "guarantee_stress_test.png"), dpi=200); plt.close()

    # ---------- Objective 4 chart: limited scans ----------
    rows = results["scan_budget"]
    xs = np.arange(len(rows)); w = 0.38
    plt.figure(figsize=(6.5, 4.2))
    plt.bar(xs - w / 2, [x["fib4"]["miss_rate_pct"] for x in rows], w, color="white",
            edgecolor="black", hatch="//", label="FIB-4")
    plt.bar(xs + w / 2, [x["model"]["miss_rate_pct"] for x in rows], w, color="0.55",
            edgecolor="black", label="Model")
    plt.xticks(xs, [f"{x['budget_pct']}%" for x in rows])
    plt.xlabel("Share of patients that can be scanned"); plt.ylabel("% of sick patients missed")
    plt.title("When FibroScans are limited"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "scan_budget.png"), dpi=200); plt.close()

    # ---------- Step 7: one example patient card ----------
    # Illustrative example: a patient FIB-4 calls "unsure" (1.30-2.67) whom the
    # model refers; of those, the one with the median model risk.
    # Chosen without looking at the true label.
    cand = np.where(grey & (zone_m == "refer"))[0]
    if len(cand) == 0:
        cand = np.where(grey)[0]
    cand = cand[np.argsort(p_te[cand])]
    i = int(cand[len(cand) // 2])
    top = np.argsort(-np.abs(sv[i]))[:3]
    words = {"clear": "NO SCAN", "uncertain": "UNSURE - scan to be safe", "refer": "SCAN"}
    card = ["Illustrative example (not evidence)",
            f"Decision : {words[zone_m[i]]}",
            f"Risk     : {p_te[i]:.0%}",
            f"FIB-4    : {f4[i]:.2f} ({zone_f[i]})",
            "Main reasons:"]
    for j in top:
        direction = "raises" if sv[i][j] > 0 else "lowers"
        card.append(f"  - {FEATURES[j]} = {X_te.iloc[i, j]:.1f} ({direction} risk)")
    card = "\n".join(card)
    print("\nExample patient card\n" + card)
    open(os.path.join(OUT, "example_patient_card.txt"), "w").write(card)

    # save everything the demo app needs
    joblib.dump({"model": model, "raw": raw, "q_pos": q_pos, "q_neg": q_neg,
                 "features": FEATURES, "chosen_model": chosen, "calibration_method": cal_method,
                 "cohort": ld.COHORT,
                 "background": X_tr.sample(min(300, len(X_tr)), random_state=SEED), "data_source": ld.DATA_SOURCE},
                os.path.join(OUT, "model.joblib"))
    print("\nSaved results, charts and model in outputs/")


if __name__ == "__main__":
    main()
