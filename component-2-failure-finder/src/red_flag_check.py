"""Function 2, piece 2: check one patient against the candidate red-flag groups."""
import pandas as pd

import group_tools as gt                      # same fixed bins as the group search
from check_patient import fib4_decision      # piece 1

# What the guard-2 runs (20 splits, fixed seeds) showed for each rule.
VALIDATION = {
    "flat": "Passed BH on all patients. Held up on unseen patients in 5-6 of 20 splits. Candidate, not proven.",
    "age_adjusted": "Passed BH on all patients. Held up on unseen patients in 0 of 20 splits. Candidate, not proven.",
}


def patient_labels(age, sex, bmi, ast, alt, platelets):
    """Put one patient into the fixed bins, written like the group names (e.g. 'bmi >=35')."""
    values = {"age": age, "bmi": bmi, "ast": ast, "alt": alt, "platelets": platelets}
    labels = {}
    for trait, v in values.items():
        edges, names = gt.BIN_RULES[trait]
        labels[trait] = f"{trait} {pd.cut([v], bins=edges, labels=names, right=False)[0]}"
    labels["sex"] = f"sex {sex}"
    return set(labels.values())


def load_candidate_flags(rule):
    """The groups that survived the BH luck guard (saved by test_groups.py)."""
    d = pd.read_csv(f"data/processed/c2_groups_{rule}_tested.csv")
    return d[d["survives_bh"]]


def check_patient(age, sex, bmi, ast, alt, platelets, rule="flat"):
    result = fib4_decision(age, ast, alt, platelets, rule)        # piece 1
    if result["status"] != "ok":
        return result
    if not isinstance(bmi, (int, float)) or isinstance(bmi, bool) or not 10 <= bmi <= 90:
        return {"status": "invalid", "reason": "bmi is missing or outside 10-90"}
    if sex not in ("male", "female"):
        return {"status": "invalid", "reason": "sex must be 'male' or 'female'"}
    result["flags"] = []
    if not result["cleared"]:
        result["message"] = "FIB-4 refers this patient for a scan."
        return result
    mine = patient_labels(age, sex, bmi, ast, alt, platelets)
    for _, row in load_candidate_flags(rule).iterrows():           # [Mine] matching step
        if set(row["group"].split(" AND ")) <= mine:
            result["flags"].append({
                "group": row["group"],
                "share_among_cleared": round(float(row["share_among_cleared"]), 3),
                "p_adjusted": float(row["p_adjusted"]),
            })
    result["flags"].sort(key=lambda f: f["p_adjusted"])
    if result["flags"]:
        result["message"] = (f"FIB-4 cleared this patient, but they match {len(result['flags'])} "
                             f"candidate red-flag group(s). {VALIDATION[rule]}")
    else:
        result["message"] = "FIB-4 cleared this patient. No candidate red flag matches."
    return result


if __name__ == "__main__":
    tests = {
        "A cleared, matches flags": (55, "male", 38, 28, 45, 250),
        "B cleared, no flag":       (40, "female", 22, 20, 20, 300),
        "C referred":               (70, "male", 40, 60, 30, 120),
        "D bad sex value":          (55, "x", 38, 28, 45, 250),
        "E BMI exactly 35":         (55, "male", 35, 28, 30, 250),
        "F BMI 34.9":               (55, "male", 34.9, 28, 30, 250),
    }
    for name, args in tests.items():
        r = check_patient(*args, rule="flat")
        print(name, "->", r.get("status"), "| fib4", r.get("fib4"), "| flags:", len(r.get("flags", [])))
        print("   ", r.get("message", r.get("reason")))