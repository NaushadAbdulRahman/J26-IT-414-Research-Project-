"""Shared Streamlit navigation entry for the Component 3 dashboard."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(
    str(ROOT / 'component-3-referral-prioritization' / 'src' / 'dashboard.py'),
    run_name='__main__',
)
