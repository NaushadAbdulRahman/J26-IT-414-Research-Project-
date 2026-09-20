"""Load NHANES XPT files and report what's in them."""

import pandas as pd
from pathlib import Path

RAW = Path("data/raw")

files = ["P_LUX", "P_DEMO", "P_BIOPRO", "P_CBC", "P_BMX"]

for name in files:
    df = pd.read_sas(RAW / f"{name}.xpt", format="xport")
    print(f"\n{name}: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  columns: {list(df.columns)}")