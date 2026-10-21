"""Component 1 - Fibrosis referral: who needs a FibroScan? (IT23229884)

The screens and the model live in component-1-fibrosis-screening/src.
This page only places them inside the shared app, as tabs.
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))                                             # shared/fib4.py
sys.path.append(str(ROOT / "component-1-fibrosis-screening" / "src"))  # Component 1 code

st.set_page_config(page_title="Component 1 - Fibrosis referral", layout="wide")

import views  # noqa: E402  (page config must come first)

views.run("tabs")
