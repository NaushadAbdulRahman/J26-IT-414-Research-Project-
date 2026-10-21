"""Shared FIB-4 helper. All components import this, so everyone gets the same answer."""
import math

CUTOFF_DEFAULT = 1.3      # below this: cleared
CUTOFF_AGE_65_PLUS = 2.0  # age-adjusted bar for patients aged 65 and over
RULES = ("age_adjusted", "flat")  # the first one is the default


def fib4(age, ast, alt, platelets):
    """age in years, AST and ALT in U/L, platelets in 10^9/L."""
    if alt <= 0 or platelets <= 0:
        raise ValueError("ALT and platelets must be above zero")
    return (age * ast) / (platelets * math.sqrt(alt))


def cutoff_for(age, rule="age_adjusted"):
    """Clearing bar for one patient.
    rule="flat": 1.3 for everyone.
    rule="age_adjusted": 2.0 for age 65 and over, otherwise 1.3."""
    if rule == "flat":
        return CUTOFF_DEFAULT
    if rule == "age_adjusted":
        return CUTOFF_AGE_65_PLUS if age >= 65 else CUTOFF_DEFAULT
    raise ValueError(f"Unknown rule: {rule}. Use one of {RULES}")


def is_cleared(age, ast, alt, platelets, rule="age_adjusted"):
    """True = FIB-4 clears the patient (sent home without a scan)."""
    return fib4(age, ast, alt, platelets) < cutoff_for(age, rule)