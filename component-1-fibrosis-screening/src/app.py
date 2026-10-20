"""
app.py  --  Component 1 on its own (standalone demo).

    streamlit run src/app.py

Run  python src/run_all.py  first, so the trained model exists.
Inside the group's shared app the same screens appear as
app/pages/1_Component_1_Fibrosis_Referral.py (see views.py).
"""
import streamlit as st

st.set_page_config(page_title="Fibrosis referral", layout="wide", initial_sidebar_state="expanded")

import views  # noqa: E402  (page config must come first)

views.run("sidebar")
