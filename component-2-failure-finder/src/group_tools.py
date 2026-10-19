"""Shared helpers for Component 2: fixed bins, group scoring, worth-a-scan test."""
from itertools import combinations

import pandas as pd
from scipy.stats import binomtest, false_discovery_control

MIN_PATIENTS = 30   # smallest group we will score
MIN_CASES = 10      # smallest number of fibrosis patients in a group
MAX_TRAITS = 3

# Fixed bins, written down BEFORE looking at any result: (edges, labels)
BIN_RULES = {
    "age":       ([0, 50, 65, 1000],     ["<50", "50-64", ">=65"]),
    "bmi":       ([0, 25, 30, 35, 1000], ["<25", "25-29.9", "30-34.9", ">=35"]),
    "ast":       ([0, 25, 40, 10000],    ["<25", "25-39", ">=40"]),
    "alt":       ([0, 25, 40, 10000],    ["<25", "25-39", ">=40"]),
    "platelets": ([0, 150, 300, 10000],  ["<150", "150-299", ">=300"]),
}
TRAITS = ["age", "sex", "bmi", "ast", "alt", "platelets"]
CLEARED_COLS = {"flat": "cleared_A", "age_adjusted": "cleared_B"}


def prepare(t, rule):
    """Add the bin columns and the 'cleared' / 'wrong' flags for one FIB-4 rule."""
    t = t.copy()
    for trait, (edges, labels) in BIN_RULES.items():
        t[trait + "_bin"] = pd.cut(t[trait], bins=edges, labels=labels, right=False)
    t["sex_bin"] = t["sex"]
    cleared_col = CLEARED_COLS[rule]
    t["cleared"] = t[cleared_col].astype(int)
    t["wrong"] = ((t[cleared_col]) & (t["fibrosis"] == 1)).astype(int)
    return t


def score_groups(t):
    """Count patients, cases, cleared and wrongly cleared for every group of 1-3 traits."""
    parts = []
    for k in range(1, MAX_TRAITS + 1):
        for traits in combinations(TRAITS, k):
            cols = [x + "_bin" for x in traits]
            g = t.groupby(cols, observed=True).agg(
                n=("id", "size"),
                cases=("fibrosis", "sum"),
                cleared=("cleared", "sum"),
                wrongly_cleared=("wrong", "sum"),
            ).reset_index()
            g["group"] = g.apply(
                lambda r: " AND ".join(f"{tr} {r[c]}" for tr, c in zip(traits, cols)), axis=1)
            g["n_traits"] = k
            parts.append(g[["group", "n_traits", "n", "cases", "cleared", "wrongly_cleared"]])
    res = pd.concat(parts, ignore_index=True)
    res = res[(res["n"] >= MIN_PATIENTS) & (res["cases"] >= MIN_CASES) & (res["cleared"] > 0)].copy()
    res["miss_rate"] = res["wrongly_cleared"] / res["cases"]
    res["share_among_cleared"] = res["wrongly_cleared"] / res["cleared"]
    return res


def referral_bar(t):
    """The bar: fibrosis share among patients FIB-4 already refers for a scan."""
    return t[t["cleared"] == 0]["fibrosis"].mean()


def worth_a_scan(res, bar):
    """Exact one-sided test per group (is its share among cleared above the bar?), then BH."""
    res = res.copy()
    res["p_luck"] = res.apply(
        lambda r: binomtest(int(r["wrongly_cleared"]), int(r["cleared"]), bar,
                            alternative="greater").pvalue, axis=1)
    res["p_adjusted"] = false_discovery_control(res["p_luck"], method="bh")
    return res