import sys
from pathlib import Path

# Add the repo root to Python's path so "shared" can be imported from a page
sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
from shared.fib4 import fib4, cutoff_for, is_cleared

st.title("Component 2 - Failure-Finder")
st.caption("Which kinds of diabetic patients does FIB-4 wrongly clear?")

st.subheader("Patient checker (prototype)")
st.write("Enter routine blood results. Red-flag checking will be added here later.")

age = st.number_input("Age (years)", min_value=18, max_value=100, value=50)
ast = st.number_input("AST (U/L)", min_value=1.0, max_value=500.0, value=30.0)
alt = st.number_input("ALT (U/L)", min_value=1.0, max_value=500.0, value=40.0)
platelets = st.number_input("Platelets (10^9/L)", min_value=10.0, max_value=900.0, value=250.0)

if st.button("Check patient"):
    score = fib4(age, ast, alt, platelets)
    bar = cutoff_for(age)
    st.metric("FIB-4 score", f"{score:.2f}", help=f"Clearing bar for this age: {bar}")
    if is_cleared(age, ast, alt, platelets):
        st.success("FIB-4 clears this patient (no scan).")
    else:
        st.warning("FIB-4 sends this patient for a scan.")