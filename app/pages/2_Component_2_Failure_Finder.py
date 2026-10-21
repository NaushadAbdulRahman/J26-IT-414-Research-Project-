"""Component 2 - Failure-Finder (IT23365896).

Which kinds of diabetic patients does a screening tool wrongly send home?

The logic lives in component-2-failure-finder/src:
    run_audit.py       Function 1: audit a screening tool (FIB-4 Rule A, Rule B, or Component 1's model)
    red_flag_check.py  Function 2: check one patient against the red-flag groups
The look lives in c2_ui.py, built on the group's shared style (Component 1's ui.py).
This page only wires the inputs to those functions and shows the outputs.
Run from the repo root:  streamlit run app/Home.py
"""
import sys
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))                                               # shared/
sys.path.append(str(ROOT / "component-2-failure-finder" / "src"))        # Component 2 code
sys.path.append(str(ROOT / "component-1-fibrosis-screening" / "src"))    # shared look (ui.py) only

st.set_page_config(page_title="Component 2 - Failure-Finder", layout="wide")

import c2_ui as look                                                     # noqa: E402
from red_flag_check import VALIDATION, check_patient, patient_labels     # noqa: E402
from run_audit import run_audit                                          # noqa: E402

TABLE = ROOT / "data" / "processed" / "c2_patient_table.csv"
C1_FILES = [ROOT / "component-1-fibrosis-screening" / "outputs" / "c1_predictions_diabetic_all.csv",
            ROOT / "data" / "interim" / "c1_predictions_diabetic_all.csv"]
TOOLS = {"FIB-4 Rule A": ("flat", "cut-off 1.3"),
         "FIB-4 Rule B": ("age_adjusted", "cut-off 2.0 if age 65+"),
         "Component 1 model": ("model", "its exported predictions")}
EXAMPLES = {
    "Enter values myself": None,
    "Example A: FIB-4 sends home, red flag": dict(age=55, sex="male", bmi=38.0, ast=28.0, alt=45.0, platelets=250.0),
    "Example B: FIB-4 sends home, no red flag": dict(age=40, sex="female", bmi=22.0, ast=20.0, alt=20.0, platelets=300.0),
    "Example C: FIB-4 sends for a scan": dict(age=70, sex="male", bmi=40.0, ast=60.0, alt=30.0, platelets=120.0),
}


@st.cache_data
def load_table(path, changed):          # 'changed' = file time, so a rebuilt table is re-read
    return pd.read_csv(path)


look.style()
look.title("Which patients does the screening tool wrongly send home?",
           "The Failure-Finder audits a screening tool against the FibroScan result. It finds the kinds of "
           "patients the tool sends home who are worth a scan, and checks that the pattern is not luck.")

tab_audit, tab_check, tab_how = st.tabs(["Audit a screening tool", "Check one patient", "How it works"])

# ------------------------------------------------------------------ Function 1: audit a tool
with tab_audit:
    tool = st.radio("Screening tool to audit", list(TOOLS), horizontal=True, key="c2_tool",
                    format_func=lambda t: f"{t} ({TOOLS[t][1]})")
    source = TOOLS[tool][0]

    c1, c1_from = None, ""
    if source == "model":
        found_file = next((p for p in C1_FILES if p.exists()), None)
        with st.expander("Use a different predictions file (CSV with columns SEQN and probability)"):
            upload = st.file_uploader("Predictions file", type="csv", key="c2_upload",
                                      label_visibility="collapsed")
        if upload is not None:
            c1_from = f"uploaded file {upload.name}"
            try:
                c1 = pd.read_csv(upload)
            except Exception:
                c1 = "unreadable"
        elif found_file is not None:
            c1_from = f"Component 1's export, {found_file.relative_to(ROOT).as_posix()}"
            c1 = pd.read_csv(found_file)
        if c1_from:
            look.note(f"Reading {escape(c1_from)}. The model gets the same number of scans as FIB-4 Rule A, "
                      f"so the comparison is fair.")
        else:
            look.note("Component 1's predictions file was not found. Run Component 1's "
                      "<b>export_predictions.py all</b>, or upload a file above.")

    if st.button("Run audit", type="primary", key="c2_run"):
        if not TABLE.exists():
            result = {"status": "invalid",
                      "reason": "the patient table is missing; run component-2-failure-finder/src/build_table.py"}
        elif isinstance(c1, str):
            result = {"status": "invalid", "reason": "that file could not be read as a CSV"}
        else:
            with st.spinner("Testing every group against the referral bar..."):
                table = load_table(TABLE, TABLE.stat().st_mtime)
                result = run_audit(table, source, c1)
            if source == "model" and result["status"] == "ok":
                # FIB-4 Rule A on the very same patients, for a fair side-by-side
                same = table[table["id"].isin(pd.to_numeric(c1["SEQN"], errors="coerce"))]
                sent = same["cleared_A"] == 0
                result = dict(result, fib4_same={"scans": int(sent.sum()),
                                                 "found": int(same.loc[sent, "fibrosis"].sum())})
        st.session_state["c2_audit"] = {"tool": tool, "from": c1_from, "r": result}

    saved = st.session_state.get("c2_audit")
    if not saved or saved["tool"] != tool or saved["from"] != c1_from:
        look.empty(f"Press Run audit to audit {tool}. The audit compares every decision with the "
                   f"FibroScan result, then tests each kind of patient against the referral bar.")
    elif saved["r"]["status"] != "ok":
        look.verdict("error", "Cannot audit this input",
                     escape(saved["r"]["reason"].capitalize()) + ". Nothing was calculated, so no wrong "
                     "answer is shown.")
    else:
        r = saved["r"]
        bar, sent_home = r["bar"], r["patients"] - r["scans"]
        look.tiles([
            ("Patients audited", f"{r['patients']:,}", "US diabetic adults with a FibroScan (NHANES 2017-2020)", False),
            ("Sent for a scan", f"{r['scans']:,}", f"{r['found']} had fibrosis, so the referral bar is {bar:.1%}", False),
            ("Sent home", f"{sent_home:,}", f"{r['wrongly_cleared']} of them had fibrosis", True),
            ("Groups tested", f"{r['groups_scored']}", "1-3 traits each; at least 30 sent home and 10 with fibrosis", False),
        ])
        look.split(r["found"], r["wrongly_cleared"], tool)
        if "fib4_same" in r:
            f = r["fib4_same"]
            look.note(f"<b>Same patients, same budget:</b> on these {r['patients']:,} patients FIB-4 Rule A also uses "
                      f"{f['scans']} scans and finds <b>{f['found']}</b> of the {r['fibrosis']} with fibrosis. "
                      f"Component 1's model finds <b>{r['found']}</b>.")

        surv = r["survivors"]
        if len(surv) == 0:
            look.verdict("clear", f"No red-flag group for {tool}",
                         f"None of the {r['groups_scored']} groups beats the referral bar of <b>{bar:.1%}</b> "
                         f"once we correct for testing {r['groups_scored']} groups. That is a valid result: "
                         f"this audit finds no kind of patient worth flagging for this tool.")
        else:
            look.verdict("flag", f"{len(surv)} kinds of patients are worth a scan, even though {tool} sends them home",
                         f"In each group, the share with fibrosis among patients sent home is above the referral "
                         f"bar of <b>{bar:.1%}</b>, and the group passes the luck check across all "
                         f"{r['groups_scored']} groups tested.")
            look.group_rows([{"group": g.group, "cleared": int(g.cleared), "wrongly_cleared": int(g.wrongly_cleared),
                              "share": float(g.share_among_cleared), "p_adjusted": float(g.p_adjusted)}
                             for g in surv.itertuples()], bar)
            sets = [set(g.split(" AND ")) for g in surv["group"]]
            common = set.intersection(*sets) if len(sets) > 1 else set()
            if common:
                shared = " · ".join(look.plain(c) for c in look.by_trait(common))
                look.note(f"All {len(surv)} groups include <b>{escape(shared)}</b>: they are overlapping versions "
                          f"of one finding, not {len(surv)} separate findings.")
            look.note("<b>Retest on unseen patients</b> (run offline, repeat_splits.py): "
                      + escape(VALIDATION.get(source, "not run yet for this tool.")))
            out = surv[["group", "cleared", "wrongly_cleared", "share_among_cleared", "p_adjusted"]].copy()
            out.insert(1, "in_plain_words", out["group"].map(look.plain))
            st.download_button("Download the red-flag groups (CSV)", out.to_csv(index=False),
                               file_name=f"c2_red_flags_{source}.csv", mime="text/csv")

        with st.expander("Blind-spot map: where the tool misses most (background, not findings)"):
            look.note("Miss rate is the share of a group's fibrosis patients that the tool sends home. A high miss "
                      "rate alone is <b>not</b> a finding: if fibrosis is rare in that group, its rate among "
                      "patients sent home can still be below the bar. Only groups that pass the luck check count.")
            bs = r["blind_spots"]
            st.dataframe(pd.DataFrame({
                "Kind of patient": bs["group"].map(look.plain),
                "Sent home": bs["cleared"],
                "With fibrosis": bs["wrongly_cleared"],
                "Miss rate": bs["miss_rate"],
                "Fibrosis among sent home": bs["share_among_cleared"],
                "Luck chance (adjusted)": bs["p_adjusted"],
                "Finding?": ["Yes" if p <= 0.05 else "No" for p in bs["p_adjusted"]],
            }), hide_index=True, width="stretch", column_config={
                "Miss rate": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                "Fibrosis among sent home": st.column_config.NumberColumn(format="percent"),
                "Luck chance (adjusted)": st.column_config.NumberColumn(format="%.4f"),
            })
        st.caption("Prototype on US NHANES data. Groups are candidates until confirmed on local patients. "
                   "Research use only, not for clinical decisions.")

# --------------------------------------------------------- Function 2: check one patient
def load_example():
    ex = EXAMPLES[st.session_state["c2_example"]]
    if ex:
        for k, v in ex.items():
            st.session_state[f"c2_{k}"] = v


for k, v in dict(age=55, sex="male", bmi=30.0, ast=30.0, alt=40.0, platelets=250.0).items():
    st.session_state.setdefault(f"c2_{k}", v)

with tab_check:
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.selectbox("Load an example patient", list(EXAMPLES), key="c2_example", on_change=load_example)
        rule_tool = st.radio("Screening rule", ["FIB-4 Rule A", "FIB-4 Rule B"], horizontal=True, key="c2_rule",
                             format_func=lambda t: f"{t} ({TOOLS[t][1]})")
        with st.form("c2_patient", border=False):
            a, b = st.columns(2)
            a.number_input("Age, years", min_value=18, max_value=100, step=1, key="c2_age")
            b.selectbox("Sex", ["male", "female"], key="c2_sex", format_func=str.capitalize)
            a, b = st.columns(2)
            a.number_input("BMI, kg/m²", min_value=10.0, max_value=90.0, step=0.1, format="%.1f", key="c2_bmi")
            b.number_input("Platelets, ×10⁹/L", min_value=10.0, max_value=900.0, step=1.0, format="%.0f",
                           key="c2_platelets")
            a, b = st.columns(2)
            a.number_input("AST, U/L", min_value=1.0, max_value=500.0, step=1.0, format="%.0f", key="c2_ast")
            b.number_input("ALT, U/L", min_value=1.0, max_value=500.0, step=1.0, format="%.0f", key="c2_alt")
            if st.form_submit_button("Check patient", type="primary", width="stretch"):
                st.session_state["c2_checked"] = True

    with right:
        if not st.session_state.get("c2_checked"):
            look.empty("The result appears here. Enter the patient's routine results and choose Check patient, "
                       "or load one of the three example patients.")
        else:
            ss = st.session_state
            rule = TOOLS[rule_tool][0]
            try:
                p = check_patient(int(ss.c2_age), ss.c2_sex, float(ss.c2_bmi), float(ss.c2_ast),
                                  float(ss.c2_alt), float(ss.c2_platelets), rule=rule)
            except FileNotFoundError:
                p = {"status": "invalid", "reason": "the red-flag groups file is missing; run find_groups.py "
                                                    "and test_groups.py first"}
            if p["status"] != "ok":
                look.verdict("error", "Cannot check this patient", escape(p["reason"].capitalize()) + ".")
            else:
                flags = p["flags"]
                if not p["cleared"]:
                    look.verdict("scan", f"{rule_tool} already sends this patient for a scan",
                                 "No red flag is needed: the screening rule refers them.")
                elif not flags:
                    look.verdict("clear", "No red flag",
                                 f"{rule_tool} sends this patient home, and they belong to none of the groups "
                                 f"the audit found worth a scan.")
                else:
                    top = flags[0]
                    look.verdict("flag", "Red flag: consider a FibroScan",
                                 f"{rule_tool} sends this patient home, but they belong to a group worth a scan. "
                                 f"In the audit, <b>{top['wrongly_cleared']} of {top['cleared']}</b> patients like "
                                 f"this who were sent home had fibrosis ({top['share_among_cleared']:.0%}), above the "
                                 f"referral bar of {p['bar']:.1%}.")
                look.score(p["fib4"], p["cutoff"], p["cleared"], rule_tool)
                mine = patient_labels(ss.c2_age, ss.c2_sex, ss.c2_bmi, ss.c2_ast, ss.c2_alt, ss.c2_platelets)
                look.chips(mine, set(flags[0]["group"].split(" AND ")) if flags else set())
                if flags:
                    rows = [{"group": f["group"], "cleared": f["cleared"], "wrongly_cleared": f["wrongly_cleared"],
                             "share": f["share_among_cleared"], "p_adjusted": f["p_adjusted"]} for f in flags]
                    look.group_rows(rows[:1], p["bar"], compact=True)
                    if len(rows) > 1:
                        with st.expander(f"{len(rows) - 1} related groups this patient also belongs to "
                                         f"(nested versions of the same finding)"):
                            look.group_rows(rows[1:], p["bar"], compact=True)
                    look.note("<b>Retest on unseen patients:</b> " + escape(VALIDATION[rule]))
                look.note("A red flag is a group rule from the audit, not a diagnosis.")

# ------------------------------------------------------------------------------ explanation
with tab_how:
    look.heading("How one audit works")
    look.steps([
        ("Start from patients whose true result is known",
         "US diabetic adults from NHANES with a FibroScan. Fibrosis means liver stiffness of 8 kPa or more."),
        ("Let the screening tool decide",
         "FIB-4 Rule A, FIB-4 Rule B, or Component 1's model sends each patient home or for a scan."),
        ("Set the referral bar",
         "The fibrosis rate among patients the tool already sends for a scan (27.5% for Rule A): "
         "what a scan finds today."),
        ("Form the groups",
         "Every mix of 1 to 3 routine traits (age, sex, BMI, AST, ALT, platelets), in bins fixed before seeing any "
         "result. Groups need at least 30 patients sent home and 10 with fibrosis."),
        ("Ask whether each group is worth a scan",
         "An exact one-sided binomial test: is the group's fibrosis rate among patients sent home above the bar? "
         "Exact, because the groups are small."),
        ("Check that it is not luck",
         "Benjamini-Hochberg correction for testing every group. Offline, a second check finds groups on half the "
         "patients and retests them on the other half, 20 times."),
        ("Use the result",
         "The surviving groups become red flags. The patient checker flags someone the tool sends home who "
         "belongs to one."),
    ])
    look.heading("Evidence that the audit can be trusted")
    look.evidence([
        ("False-alarm test (null_test.py)", "4 of 200 runs",
         "On made-up data where no group is truly worse, the audit flagged a group in 2.0% of runs, "
         "within the 5% allowed."),
        ("Retest on unseen patients (repeat_splits.py)", "5-6 of 20 splits",
         "The BMI ≥ 35 groups held on patients the search never saw (Rule A). Rule B: 0 of 20."),
        ("Automated tests (pytest)", "23 pass",
         "Including 12 that feed the audit broken files. It refuses them with a reason instead of giving a "
         "wrong answer."),
        ("Connection to Component 1", "Same scan budget",
         "Audits Component 1's model with the same number of scans as FIB-4 Rule A, so the two are compared fairly."),
    ])
    look.note("<b>Limits.</b> These are US patients, so the groups are candidates. On local patients the full search "
              "is run again with locally suitable bins. A red flag is a group rule, not a diagnosis.")
