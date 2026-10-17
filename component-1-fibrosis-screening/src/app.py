"""
app.py  --  the screening tool a doctor would see (demo).

    streamlit run src/app.py

Run  python src/run_all.py  first, so the trained model exists.
"""
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from explain import shap_values_for

HERE = os.path.dirname(os.path.abspath(__file__))
saved = joblib.load(os.path.join(HERE, "..", "outputs", "model.joblib"))
model, raw, q_pos, q_neg = saved["model"], saved["raw"], saved["q_pos"], saved["q_neg"]
FEATURES = saved["features"]

st.title("Liver Fibrosis Screening Tool")
st.caption(f"Research prototype - not for clinical use. Model: {saved['chosen_model']}, trained on {saved['data_source']} data.")

c1, c2, c3 = st.columns(3)
v = {
    "age": c1.number_input("Age (years)", 18, 90, 55),
    "sex": 1 if c1.selectbox("Sex", ["Male", "Female"]) == "Male" else 0,
    "bmi": c1.number_input("BMI", 15.0, 60.0, 29.0),
    "waist": c1.number_input("Waist (cm)", 50.0, 170.0, 98.0),
    "diabetes": 1 if c2.selectbox("Diabetes", ["No", "Yes"]) == "Yes" else 0,
    "ast": c2.number_input("AST (U/L)", 5.0, 500.0, 38.0),
    "alt": c2.number_input("ALT (U/L)", 5.0, 500.0, 42.0),
    "ggt": c2.number_input("GGT (U/L)", 5.0, 800.0, 45.0),
    "platelets": c3.number_input("Platelets (x10^9/L)", 30.0, 600.0, 210.0),
    "albumin": c3.number_input("Albumin (g/dL)", 2.0, 6.0, 4.2),
    "hba1c": c3.number_input("HbA1c (%)", 4.0, 14.0, 6.0),
    "triglycerides": c3.number_input("Triglycerides (mg/dL)", 30.0, 1000.0, 150.0),
}

if st.button("Screen patient"):
    x = pd.DataFrame([v])[FEATURES]
    p = float(model.predict_proba(x)[0, 1])
    could_be_sick, could_be_healthy = (1 - p) <= q_pos, p <= q_neg
    if could_be_healthy and not could_be_sick:
        st.success("NO SCAN - low risk, FibroScan not needed now")
    elif could_be_sick and not could_be_healthy:
        st.error("SCAN - refer for FibroScan")
    else:
        st.warning("UNSURE - refer for FibroScan to be safe")

    f4 = v["age"] * v["ast"] / (v["platelets"] * np.sqrt(v["alt"]))
    a, b = st.columns(2)
    a.metric("Estimated risk of significant fibrosis", f"{p:.0%}")
    b.metric("FIB-4 score (for comparison)", f"{f4:.2f}")

    sv = shap_values_for(raw, saved["background"], x)[0]
    st.subheader("Main reasons")
    for j in np.argsort(-np.abs(sv))[:4]:
        st.write(f"- **{FEATURES[j]}** = {x.iloc[0, j]:.1f} "
                 f"({'raises' if sv[j] > 0 else 'lowers'} risk)")
