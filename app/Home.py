"""Home page of the shared screening tool. Run from the repo root:
    streamlit run app/Home.py
"""
import streamlit as st

st.set_page_config(page_title="Diabetic Fibrosis Screening Tool", layout="wide")

st.title("Diabetic Fibrosis Screening Tool")
st.write(
    "One tool, four research components. "
    "Pick a component from the menu on the left."
)

st.subheader("Components")
st.markdown(
    "- **Component 1** - fibrosis screening model\n"
    "- **Component 2** - Failure-Finder: which patients FIB-4 wrongly clears\n"
    "- **Component 3** - Scarce Resource Referral Prioritization: compares patient selection strategies under limited scan budgets\n"
    "- **Component 4** - (to be filled by its owner)"
)
st.info("Prototype for PP1. Research use only, not for clinical decisions.")