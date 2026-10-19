import sys
from pathlib import Path

# Repo root (for "shared") and Component 2's src folder (for check_patient / red_flag_check)
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "component-2-failure-finder" / "src"))

import streamlit as st
from red_flag_check import check_patient

st.title("Component 2 - Failure-Finder")
st.caption("Which kinds of diabetic patients does FIB-4 wrongly clear?")

st.subheader("Patient checker (prototype)")
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
        st.write(f"In the audit, {top['share_among_cleared']:.0%} of cleared patients in this group "
                 f"had fibrosis (the audit's comparison level was about 27-32%).")
        st.caption(r["message"])
        if len(r["flags"]) > 1:
            with st.expander(f"{len(r['flags']) - 1} related groups (nested versions of the same finding)"):
                for f in r["flags"][1:]:
                    st.write(f"- {f['group']}")