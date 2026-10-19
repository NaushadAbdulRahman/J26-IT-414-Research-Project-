"""
analyses.py  --  the extra analyses the proposal asks for, called from run_all.py.

    uncertainty()          bootstrap 95% confidence intervals (objective 5)
    calibration_report()   Brier score, calibration slope/intercept, ECE (objective 2, 5)
    uncalibrated_benchmark()  the same model without calibration, at the usual 0.5
                           cut-off and with the decision layer (objective 5)
    guarantee_stress_test()   is the 10% miss bound kept on many new splits? (RQ1)
    scan_budget_table()    misses when only a limited share can be scanned (objective 4)
    composite_vs_decomposed()  FIB-4 as one score vs its parts as inputs (RQ3)
    shap_alignment()       do the top SHAP drivers match known risk factors? (RQ4)
    external_validation()  frozen model on a different NHANES cycle, if present

Every function only reads the patients it is given; none of them refit anything
on the test patients.
"""
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


# ---------- shared pieces (the same rules as run_all.py) ----------
def fib4(df):
    return (df.age * df.ast) / (df.platelets * np.sqrt(df.alt))


def conformal_thresholds(p_cal, y_cal, alpha):
    def q(scores):
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return np.quantile(scores, level, method="higher")
    return q(1 - p_cal[y_cal == 1]), q(p_cal[y_cal == 0])


def cleared_by_model(p, q_pos, q_neg):
    """True where the three-zone decision is NO SCAN."""
    return (p <= q_neg) & ~((1 - p) <= q_pos)


def scans_and_misses(cleared, y):
    """% of patients sent for a scan, % of sick patients missed."""
    return (round(100 * float((~cleared).mean()), 1),
            round(100 * float((cleared & (y == 1)).sum() / max(1, y.sum())), 1))


def r(x, d=3):
    return round(float(x), d)


# ---------- objective 5: uncertainty ----------
def uncertainty(y, p_model, f4, cleared_model, cleared_fib4, n_boot=1000, seed=42):
    """Bootstrap the test patients: 95% intervals for AUROC (both, and the difference)
    and for the miss rate and scan rate of both decision rules."""
    rng = np.random.default_rng(seed)
    n = len(y)
    keep = {k: [] for k in ["auc_m", "auc_f", "auc_diff", "miss_m", "miss_f", "scan_m", "scan_f"]}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        am, af = roc_auc_score(y[i], p_model[i]), roc_auc_score(y[i], f4[i])
        keep["auc_m"].append(am); keep["auc_f"].append(af); keep["auc_diff"].append(am - af)
        sm, mm = scans_and_misses(cleared_model[i], y[i])
        sf, mf = scans_and_misses(cleared_fib4[i], y[i])
        keep["miss_m"].append(mm); keep["miss_f"].append(mf)
        keep["scan_m"].append(sm); keep["scan_f"].append(sf)
    ci = lambda v, d: [r(np.percentile(v, 2.5), d), r(np.percentile(v, 97.5), d)]
    return {"bootstrap_samples": len(keep["auc_m"]),
            "auroc_model_95ci": ci(keep["auc_m"], 3), "auroc_fib4_95ci": ci(keep["auc_f"], 3),
            "auroc_difference_95ci": ci(keep["auc_diff"], 3),
            "auroc_difference_excludes_zero": bool(np.percentile(keep["auc_diff"], 2.5) > 0),
            "model_miss_rate_pct_95ci": ci(keep["miss_m"], 1), "fib4_miss_rate_pct_95ci": ci(keep["miss_f"], 1),
            "model_scans_pct_95ci": ci(keep["scan_m"], 1), "fib4_scans_pct_95ci": ci(keep["scan_f"], 1)}


# ---------- objectives 2 and 5: calibration ----------
def calibration_numbers(y, p):
    """Brier score (lower is better), calibration slope (1 is ideal), intercept (0 is ideal)
    and expected calibration error (average gap between predicted and observed, in %)."""
    p = np.clip(p, 1e-4, 1 - 1e-4)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)
    bins = np.quantile(p, np.linspace(0, 1, 11))
    idx = np.clip(np.searchsorted(bins, p, side="right") - 1, 0, 9)
    ece = sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
              for b in range(10) if (idx == b).any())
    return {"brier": r(brier_score_loss(y, p), 4), "calibration_slope": r(lr.coef_[0][0], 2),
            "calibration_intercept": r(lr.intercept_[0], 2), "ece_pct": r(100 * ece, 1)}


def calibration_report(y, p_raw, p_cal):
    return {"uncalibrated": calibration_numbers(y, p_raw), "calibrated": calibration_numbers(y, p_cal)}


# ---------- objective 5: uncalibrated benchmark ----------
def uncalibrated_benchmark(raw, X_cal, y_cal, X_te, y_te, alpha):
    """The chosen model without calibration:
       (a) the usual 0.5 cut-off (scan if risk >= 50%)
       (b) the same decision layer, set on uncalibrated scores"""
    p_cal, p_te = raw.predict_proba(X_cal)[:, 1], raw.predict_proba(X_te)[:, 1]
    s_a, m_a = scans_and_misses(p_te < 0.5, y_te)
    q_pos, q_neg = conformal_thresholds(p_cal, y_cal, alpha)
    s_b, m_b = scans_and_misses(cleared_by_model(p_te, q_pos, q_neg), y_te)
    return {"default_cutoff_0_5": {"scans_pct": s_a, "miss_rate_pct": m_a},
            "with_decision_layer": {"scans_pct": s_b, "miss_rate_pct": m_b},
            "auroc": r(roc_auc_score(y_te, p_te))}


# ---------- RQ1: does the bound hold on new patients? ----------
def guarantee_stress_test(model, X_pool, y_pool, alpha, repeats=500, seed=7):
    """The fitted model stays fixed. The patients it never trained on are re-split
    into a new calibration half and a new test half, many times. Each time the
    thresholds are set on one half and the miss rate measured on the other.
    Conformal theory promises the miss rate is at most alpha ON AVERAGE."""
    p = model.predict_proba(X_pool)[:, 1]
    rng = np.random.RandomState(seed)
    misses, scans = [], []
    for k in range(repeats):
        i_cal, i_te = train_test_split(np.arange(len(y_pool)), test_size=0.5, stratify=y_pool,
                                       random_state=rng.randint(1_000_000))
        q_pos, q_neg = conformal_thresholds(p[i_cal], y_pool[i_cal], alpha)
        s, m = scans_and_misses(cleared_by_model(p[i_te], q_pos, q_neg), y_pool[i_te])
        misses.append(m); scans.append(s)
    misses = np.array(misses)
    return {"repeats": repeats, "allowed_miss_rate_pct": 100 * alpha,
            "mean_miss_rate_pct": r(misses.mean(), 1),
            "median_miss_rate_pct": r(np.median(misses), 1),
            "miss_rate_pct_range_5_to_95": [r(np.percentile(misses, 5), 1), r(np.percentile(misses, 95), 1)],
            "share_of_splits_within_bound_pct": r(100 * (misses <= 100 * alpha).mean(), 1),
            "mean_scans_pct": r(np.mean(scans), 1),
            "bound_kept_on_average": bool(misses.mean() <= 100 * alpha + 0.5),
            "_misses": misses.tolist()}


# ---------- objective 4: limited number of FibroScans ----------
def scan_budget_table(p_cal, f4_cal, p_te, f4_te, y_te, budgets=(0.2, 0.3, 0.4, 0.5, 0.6)):
    """If only a fixed share of patients can be scanned, who misses fewer sick patients?
    Each cut-off is set on the calibration patients, then applied to the test patients."""
    rows = []
    for b in budgets:
        tm, tf = np.quantile(p_cal, 1 - b), np.quantile(f4_cal, 1 - b)
        sm, mm = scans_and_misses(p_te < tm, y_te)
        sf, mf = scans_and_misses(f4_te < tf, y_te)
        rows.append({"budget_pct": int(100 * b), "model": {"scans_pct": sm, "miss_rate_pct": mm},
                     "fib4": {"scans_pct": sf, "miss_rate_pct": mf}})
    return rows


# ---------- RQ3: composite score vs its parts ----------
def composite_vs_decomposed(pipeline, X_tr, y_tr, X_te, y_te, features, cal_method="isotonic", seed=42):
    """Same model type, three input sets:
       A  FIB-4 score alone
       B  FIB-4 score + the other 8 results (age, AST, ALT, platelets only via FIB-4)
       C  all 12 results separately (age, AST, ALT and platelets as their own inputs)"""
    parts = ["age", "ast", "alt", "platelets"]
    others = [f for f in features if f not in parts]

    def with_fib4(X, cols):
        Z = X[cols].copy()
        Z.insert(0, "fib4", fib4(X).values)
        return Z
    sets = {"A: FIB-4 score only": (lambda X: with_fib4(X, [])),
            "B: FIB-4 score + 8 other results": (lambda X: with_fib4(X, others)),
            "C: 12 results, FIB-4 parts separate": (lambda X: X[features])}
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    out = {}
    for name, make in sets.items():
        Ztr, Zte = make(X_tr), make(X_te)
        cv_auc = cross_val_score(clone(pipeline), Ztr, y_tr, cv=cv, scoring="roc_auc").mean()
        m = (clone(pipeline) if cal_method == "none" else
             CalibratedClassifierCV(clone(pipeline), method=cal_method, cv=5)).fit(Ztr, y_tr)
        p = m.predict_proba(Zte)[:, 1]
        out[name] = {"cv_auroc_training": r(cv_auc), "test_auroc": r(roc_auc_score(y_te, p)),
                     "test_brier": r(brier_score_loss(y_te, p), 4)}
    return out


# ---------- RQ4: do the explanations make clinical sense? ----------
KNOWN_RISK_FACTORS = {"ast", "alt", "platelets", "age", "diabetes", "bmi", "waist", "ggt", "hba1c", "albumin"}


def shap_alignment(sv, features, top=5):
    """Rank results by average |SHAP|; mark those that are recognised fibrosis risk factors."""
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(-imp)
    ranking = [{"feature": features[j], "mean_abs_shap": r(imp[j], 4),
                "recognised_risk_factor": features[j] in KNOWN_RISK_FACTORS} for j in order]
    return {"ranking": ranking,
            "top_k": top,
            "recognised_in_top_k": sum(x["recognised_risk_factor"] for x in ranking[:top])}


# ---------- external validation ----------
def external_validation(model, q_pos, q_neg, df_ext, features):
    """The frozen model and thresholds, applied unchanged to a different survey cycle."""
    X, y = df_ext[features], df_ext["label"].values
    p, f4 = model.predict_proba(X)[:, 1], fib4(X).values
    sm, mm = scans_and_misses(cleared_by_model(p, q_pos, q_neg), y)
    sf, mf = scans_and_misses(f4 < 1.30, y)
    return {"patients": int(len(y)), "with_fibrosis": int(y.sum()),
            "auroc": {"model": r(roc_auc_score(y, p)), "fib4": r(roc_auc_score(y, f4))},
            "model_three_zone": {"scans_pct": sm, "miss_rate_pct": mm},
            "fib4_guideline_1_30": {"scans_pct": sf, "miss_rate_pct": mf},
            "calibration": calibration_numbers(y, p)}
