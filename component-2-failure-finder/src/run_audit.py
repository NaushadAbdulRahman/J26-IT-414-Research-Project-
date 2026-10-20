"""One audit, one function. Used by the Streamlit page and by the command line.

Input : the patient table + a screening decision (FIB-4 Rule A, FIB-4 Rule B,
        or Component 1's model with a CSV of SEQN, probability, referred).
Output: headline counts, the referral bar, the groups that pass the luck check,
        and a blind-spot map (groups with the highest miss rate).

Run from the repo root:  python component-2-failure-finder/src/run_audit.py flat
"""
import sys
import pandas as pd
from group_tools import prepare, score_groups, referral_bar, worth_a_scan


def run_audit(table, source="flat", c1=None):
    t = table.copy()
    if source == "model":
        if c1 is None:
            return {"status": "invalid", "reason": "no Component 1 predictions given"}
        needed = {"SEQN", "probability"}
        if not needed.issubset(c1.columns):
            return {"status": "invalid", "reason": f"file needs columns {sorted(needed)}"}
        t = c1.merge(t, left_on="SEQN", right_on="id")
        if len(t) == 0:
            return {"status": "invalid", "reason": "no patients matched (check the SEQN numbers)"}
        k = int((t["cleared_A"] == 0).sum())          # same number of scans as FIB-4 Rule A
        rank = t["probability"].rank(method="first", ascending=False)
        t["cleared_model"] = rank > k
    t = prepare(t, source)
    bar = referral_bar(t)
    res = worth_a_scan(score_groups(t), bar)
    res["survives"] = res["p_adjusted"] <= 0.05
    referred = t[t["cleared"] == 0]
    return {
        "status": "ok",
        "patients": len(t),
        "fibrosis": int(t["fibrosis"].sum()),
        "scans": len(referred),
        "found": int(referred["fibrosis"].sum()),
        "wrongly_cleared": int(t["wrong"].sum()),
        "bar": float(bar),
        "groups_scored": len(res),
        "survivors": res[res["survives"]].sort_values("p_adjusted"),
        "blind_spots": res.sort_values("miss_rate", ascending=False).head(8),
    }


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else "flat"
    c1 = pd.read_csv(sys.argv[2]) if len(sys.argv) > 2 else None
    r = run_audit(pd.read_csv("data/processed/c2_patient_table.csv"), source, c1)
    if r["status"] != "ok":
        sys.exit(f"Cannot audit: {r['reason']}")
    print(f"Source {source}: {r['patients']} patients, {r['fibrosis']} with fibrosis")
    print(f"Scans {r['scans']}, found {r['found']}, wrongly cleared {r['wrongly_cleared']}")
    print(f"Bar {r['bar']:.1%}, groups scored {r['groups_scored']}, passing the luck check {len(r['survivors'])}")
    cols = ["group", "n", "cleared", "wrongly_cleared", "miss_rate", "p_adjusted"]
    print(r["blind_spots"][cols].to_string(index=False))