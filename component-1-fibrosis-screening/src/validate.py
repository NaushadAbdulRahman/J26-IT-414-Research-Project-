"""
validate.py  --  checks on what the user types, before anything is screened.

Three kinds of message:
  * error    the value is missing, impossible, or looks like the wrong unit -> not screened
  * warning  the value is possible but unusual, or outside what the model was trained on
             -> screened, but the user is asked to double-check
  * filled   an optional result was left blank -> the model fills in a typical value

The same rules are used for one patient (the form) and for many (the uploaded spreadsheet).
"""
import re
import math

# Results the FIB-4 formula and the decision cannot do without
REQUIRED = ["age", "sex", "diabetes", "ast", "alt", "platelets"]

# field: (label, unit, lowest possible, highest possible, usual low, usual high)
#   outside lowest/highest possible -> error (typing slip, or wrong unit)
#   outside usual low/high          -> warning (check it, or outside the training data)
RANGES = {
    "age":           ("Age", "years",          18,   100,  18,   80),
    "bmi":           ("BMI", "kg/m²",          12,    80,  16,   60),
    "waist":         ("Waist", "cm",           40,   200,  60,  155),
    "ast":           ("AST", "U/L",             3,  5000,   5,  400),
    "alt":           ("ALT", "U/L",             3,  5000,   5,  400),
    "ggt":           ("GGT", "U/L",             3,  5000,   5,  800),
    "platelets":     ("Platelets", "×10⁹/L",    5,  1500,  50,  600),
    "albumin":       ("Albumin", "g/dL",      1.0,   7.0, 2.5,  5.5),
    "hba1c":         ("HbA1c", "%",           3.0,  20.0, 4.0, 14.0),
    "triglycerides": ("Triglycerides", "mg/dL", 10, 5000,  30, 1000),
}

# Common unit mix-ups: (field, test on the value, message)
UNIT_HINTS = [
    ("albumin", lambda v: 10 <= v <= 70,
     "looks like g/L. This box takes g/dL: divide by 10 (for example 42 g/L = 4.2)."),
    ("hba1c", lambda v: 20 <= v <= 200,
     "looks like mmol/mol. This box takes %: for example 48 mmol/mol = 6.5%."),
    ("triglycerides", lambda v: 0 < v < 10,
     "looks like mmol/L. This box takes mg/dL: multiply by 88.6 (for example 1.7 mmol/L = 150)."),
    ("platelets", lambda v: 1500 < v,
     "is too high for ×10⁹/L. Enter the count in thousands per µL (for example 250)."),
]

LABEL = {"pid": "Patient ID", "sex": "Sex", "diabetes": "Diabetes", **{f: r[0] for f, r in RANGES.items()}}

PID_PATTERN = re.compile(r"^[A-Za-z0-9 _\-/.]{1,20}$")


def blank(v):
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip())


def fmt(v):
    return f"{v:g}"


def check(values, pid=""):
    """
    values: dict of the 12 results (sex and diabetes as "Male"/"Female", "Yes"/"No" or 1/0),
            blanks as None or NaN.
    Returns (errors, warnings, filled):
        errors   {field: message}   -> must be fixed before screening
        warnings [message, ...]     -> screened, but shown to the user
        filled   [label, ...]       -> optional results left blank
    """
    errors, warnings, filled = {}, [], []

    if pid and not PID_PATTERN.match(pid.strip()):
        errors["pid"] = "Use up to 20 letters, numbers, spaces or - _ / . (a clinic number, not a name)."

    for f in ("sex", "diabetes"):
        if blank(values.get(f)):
            errors[f] = "Required. Choose one."

    for f, (label, unit, lo, hi, ulo, uhi) in RANGES.items():
        v = values.get(f)
        if blank(v):
            if f in REQUIRED:
                errors[f] = "Required (needed for FIB-4 and the decision)."
            else:
                filled.append(label)
            continue
        hint = next((msg for g, test, msg in UNIT_HINTS if g == f and test(v)), None)
        if hint:
            errors[f] = f"{fmt(v)} {hint}"
        elif f == "age" and v < lo:
            errors[f] = "This tool is for adults: age must be 18 or over."
        elif v < lo or v > hi:
            errors[f] = f"Must be between {fmt(lo)} and {fmt(hi)} {unit}."
        elif v > uhi and f in ("age", "ast", "alt"):
            pass                                  # these have their own, clearer warning below
        elif v < ulo or v > uhi:
            warnings.append(f"{label} {fmt(v)} {unit} is unusual (usual range {fmt(ulo)} to {fmt(uhi)}). "
                            "Check it was typed correctly.")

    # A few checks across results (only when both values are usable)
    ok = lambda f: f not in errors and not blank(values.get(f))
    if ok("age") and values["age"] > 80:
        warnings.append("Age over 80: the training data records everyone over 80 as 80, "
                        "so the risk is less certain.")
    for f in ("ast", "alt"):
        if ok(f) and values[f] > 400:
            warnings.append(f"{RANGES[f][0]} above 400 U/L suggests acute liver injury. "
                            "This tool is meant for stable fatty liver disease, not acute illness.")
    if ok("hba1c") and values["hba1c"] >= 6.5 and str(values.get("diabetes")) in ("No", "0", "0.0"):
        warnings.append("HbA1c is 6.5% or more, which meets the diabetes threshold, "
                        "but Diabetes is set to No. Check the Diabetes box.")
    if ok("bmi") and ok("waist"):
        if values["bmi"] >= 35 and values["waist"] < 80:
            warnings.append("BMI is 35 or more but waist is under 80 cm. Check both values.")
        if values["bmi"] < 20 and values["waist"] > 120:
            warnings.append("BMI is under 20 but waist is over 120 cm. Check both values.")
    return errors, warnings, filled


def check_table(df):
    """
    Check every row of an uploaded spreadsheet (numbers already parsed, sex/diabetes as 1/0).
    Returns two lists, one entry per row: error text ("" if the row is fine) and warning text.
    """
    row_errors, row_warnings = [], []
    for _, r in df.iterrows():
        values = {f: (None if blank(r.get(f)) else float(r.get(f))) for f in RANGES}
        values["sex"] = r.get("sex")
        values["diabetes"] = r.get("diabetes")
        pid = str(r.get("patient_id", "")) if not blank(r.get("patient_id", "")) else ""
        errors, warnings, _ = check(values, pid)
        row_errors.append(" ".join(f"{LABEL.get(f, f)}: {m}" for f, m in errors.items()))
        row_warnings.append(" ".join(warnings))
    return row_errors, row_warnings
