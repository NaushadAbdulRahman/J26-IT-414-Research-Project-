"""Shared FIB-4 helper. All components import this, so everyone gets the same answer."""
import math

CUTOFF_DEFAULT = 1.3      # below this: cleared
CUTOFF_AGE_65_PLUS = 2.0  # age-adjusted bar for patients aged 65 and over


def fib4(age, ast, alt, platelets):
    """age in years, AST and ALT in U/L, platelets in 10^9/L."""
    if alt <= 0 or platelets <= 0:
        raise ValueError("ALT and platelets must be above zero")
    return (age * ast) / (platelets * math.sqrt(alt))


def cutoff_for(age):
    return CUTOFF_AGE_65_PLUS if age >= 65 else CUTOFF_DEFAULT


def is_cleared(age, ast, alt, platelets):
    """True = FIB-4 clears the patient (sent home without a scan)."""
    return fib4(age, ast, alt, platelets) < cutoff_for(age)