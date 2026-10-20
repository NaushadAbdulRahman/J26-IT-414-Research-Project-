import streamlit as st

st.title("Component 3 — Scarce Resource Referral Prioritization")

st.write(
    "Compare patient selection strategies when only a limited "
    "number of scan appointments are available."
)

st.info("PP1 prototype under development. Research use only.")

st.subheader("Scan budget")

scan_budget = st.number_input(
    "Available scan appointments per round",
    min_value=1,
    max_value=100,
    value=5,
    step=1,
)

st.write(f"Each strategy will select up to {scan_budget} patients per round.")

st.subheader("Planned selection strategies")

st.markdown(
    """
- **Risk-first:** highest predicted probability of a positive scan.
- **Uncertainty:** predictions closest to the decision boundary.
- **Diversity:** patients with different feature profiles.
- **Hybrid:** a combination of uncertainty and diversity.
- **Random:** a baseline for comparison.
"""
)

st.caption(
    "This page is the initial interface. Data loading, model training "
    "and patient selection will be added in subsequent commits."
)