import sys
from pathlib import Path

# Repo root (for "shared") and Component 2's src folder (for the audit and the checker)
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "component-2-failure-finder" / "src"))

import pandas as pd
import streamlit as st
from red_flag_check import check_patient
from run_audit import run_audit

TABLE_PATH = ROOT / "data" / "processed" / "c2_patient_table.csv"

st.title("Component 2 - Failure-Finder")
st.caption("Which kinds of diabetic patients does a screening tool wrongly clear?")

tab_audit, tab_check = st.tabs(["Audit a screening tool", "Check one patient"])

# ---------------------------------------------------------------- Audit tab
with tab_audit:
    st.write("Pick a screening tool. The audit compares its decisions with the FibroScan result "
             "and lists the kinds of cleared patients who are worth a scan, after a check that "
             "it is not luck.")
    choice = st.radio("Screening tool to audit",
                      ["FIB-4 Rule A: flat cut-off 1.3",
                       "FIB-4 Rule B: age-adjusted (2.0 if 65+)",
                       "Component 1 model (upload its predictions)"], key="audit_choice")
    source = {"FIB-4 Rule A": "flat", "FIB-4 Rule B": "age_adjusted"}.get(choice[:12], "model")
    c1 = None
    if source == "model":
        up = st.file_uploader("Predictions CSV with columns SEQN, probability, referred", type="csv")
        if up is not None:
            try:
                c1 = pd.read_csv(up)
            except Exception:
                st.error("That file could not be read as a CSV.")
        st.caption("The model is given the same number of scans as FIB-4 Rule A, so the comparison is fair.")

    if st.button("Run audit"):
        if not TABLE_PATH.exists():
            st.error("The patient table is missing. Run build_table.py first.")
        elif source == "model" and c1 is None:
            st.warning("Upload the predictions file first.")
        else:
            table = pd.read_csv(TABLE_PATH)
            st.session_state["audit"] = (choice, run_audit(table, source, c1))

    if "audit" in st.session_state:
        shown, r = st.session_state["audit"]
        st.markdown(f"**Result for:** {shown}")
        if r["status"] != "ok":
            st.error(f"Cannot audit: {r['reason']}")
        else:
            a, b, c, d = st.columns(4)
            a.metric("Scans used", r["scans"])
            b.metric("Fibrosis found", r["found"])
            c.metric("Wrongly cleared", r["wrongly_cleared"])
            d.metric("Referral bar", f"{r['bar']:.1%}",
                     help="Share of already-referred patients who truly have fibrosis. "
                          "A cleared group must beat this to be worth a scan.")
            surv = r["survivors"]
            if len(surv) == 0:
                st.info(f"No group passes the luck check among {r['groups_scored']} groups tested. "
                        "For this tool there is no group we can call worth a scan. "
                        "That is a valid result.")
            else:
                st.error(f"{len(surv)} of {r['groups_scored']} groups are worth a scan "
                         "and pass the luck check.")
                show = surv[["group", "cleared", "wrongly_cleared", "share_among_cleared", "p_adjusted"]].copy()
                show["share_among_cleared"] = (show["share_among_cleared"] * 100).round(1)
                show.columns = ["Group", "Cleared patients", "Had fibrosis", "% with fibrosis", "Luck chance (adjusted)"]
                st.dataframe(show, hide_index=True)
                chart = surv.set_index("group")[["share_among_cleared"]] * 100
                chart.columns = ["% of cleared patients with fibrosis"]
                st.bar_chart(chart)
                st.caption(f"Compare each bar with the referral bar of {r['bar']:.1%}.")
            with st.expander("Blind-spot map: groups with the highest miss rate (background only)"):
                st.write("A high miss rate alone is not a finding. Look at the last column: "
                         "a small number means unlikely to be luck.")
                bs = r["blind_spots"][["group", "cleared", "wrongly_cleared", "miss_rate", "p_adjusted"]].copy()
                bs["miss_rate"] = (bs["miss_rate"] * 100).round(0)
                bs.columns = ["Group", "Cleared patients", "Had fibrosis", "Miss rate %", "Luck chance (adjusted)"]
                st.dataframe(bs, hide_index=True)
            st.caption("Prototype on US NHANES data. Groups are candidates until tested on local patients.")

# ------------------------------------------------------------ Checker tab
with tab_check:
    st.write("Enter routine results. If FIB-4 clears the patient, we check whether they belong to a "
             "candidate red-flag group found by the audit.")

    rule_name = st.radio("FIB-4 rule being audited",
                         ["Rule A: flat cut-off 1.3", "Rule B: age-adjusted (2.0 if 65+)"])
    rule = "flat" if rule_name.startswith("Rule A") else "age_adjusted"

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=55)
        sex = st.radio("Sex", ["male", "female"], horizontal=True)
        bmi = st.number_input("BMI (kg/m2)", min_value=10.0, max_value=90.0, value=30.0)
    with col2:
        ast = st.number_input("AST (U/L)", min_value=1.0, max_value=500.0, value=30.0)
        alt = st.number_input("ALT (U/L)", min_value=1.0, max_value=500.0, value=40.0)
        platelets = st.number_input("Platelets (10^9/L)", min_value=10.0, max_value=900.0, value=250.0)

    if st.button("Check patient"):
        try:
            r = check_patient(age, sex, bmi, ast, alt, platelets, rule=rule)
        except FileNotFoundError:
            st.error("The audit results file is missing. Run find_groups.py and test_groups.py first.")
            st.stop()

        if r["status"] != "ok":
            st.error(f"Cannot check this patient: {r['reason']}")
            st.stop()

        st.metric("FIB-4 score", f"{r['fib4']:.2f}", help=f"Cut-off for this patient: {r['cutoff']}")
        if not r["cleared"]:
            st.warning("FIB-4 sends this patient for a scan.")
        elif not r["flags"]:
            st.success("FIB-4 clears this patient. No candidate red flag matches.")
        else:
            top = r["flags"][0]
            st.error("FIB-4 clears this patient, but they match a candidate red flag.")
            st.write(f"**Strongest matching group:** {top['group']}")
            st.write(f"In the audit, **{top['wrongly_cleared']} of {top['cleared']}** cleared patients in this "
                     f"group had fibrosis ({top['share_among_cleared']:.0%}). "
                     f"The comparison bar for this rule is {r['bar']:.1%}.")
            st.caption(r["message"])
            if len(r["flags"]) > 1:
                with st.expander(f"{len(r['flags']) - 1} related groups (nested versions of the same finding)"):
                    for f in r["flags"][1:]:
                        st.write(f"- {f['group']}")