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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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


def main():
    # ---------- Step 1: data ----------
    df = load_data()
    X, y = df[FEATURES], df["label"].values
    print(f"Data source: {ld.DATA_SOURCE} | patients: {len(df)} | "
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
    raw = clone(candidates[chosen]).fit(X_tr, y_tr)        # uncalibrated, for comparison + SHAP
    model = CalibratedClassifierCV(clone(candidates[chosen]), method="isotonic",
                                   cv=5).fit(X_tr, y_tr)

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
        "n_test": int(len(y_te)),
        "allowed_miss_rate_pct": 100 * ALPHA,
        "chosen_model": chosen,
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
                 "features": FEATURES, "chosen_model": chosen,
                 "background": X_tr.sample(min(300, len(X_tr)), random_state=SEED), "data_source": ld.DATA_SOURCE},
                os.path.join(OUT, "model.joblib"))
    print("\nSaved results, charts and model in outputs/")


if __name__ == "__main__":
    main()
