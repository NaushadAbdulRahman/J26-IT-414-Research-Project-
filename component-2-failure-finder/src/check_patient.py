"""Function 2: check one patient. Piece 1: FIB-4 and the cleared / referred decision."""
import math

# Sensible ranges. Outside these we refuse to answer instead of guessing.
LIMITS = {
    "age": (18, 120),
    "ast": (1, 2000),
    "alt": (1, 2000),
    "platelets": (10, 1500),
}


def fib4_cutoff(age, rule):
    """[Standard] FIB-4 cut-off. Rule A: flat 1.3. Rule B: 2.0 if age 65+, else 1.3."""
    if rule == "age_adjusted" and age >= 65:
        return 2.0
    return 1.3


def fib4_decision(age, ast, alt, platelets, rule="flat"):
    """Input -> FIB-4 score -> cleared or referred. Bad input gives status 'invalid'."""
    values = {"age": age, "ast": ast, "alt": alt, "platelets": platelets}
    for name, v in values.items():
        if v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v):
            return {"status": "invalid", "reason": f"{name} is missing or not a number"}
        low, high = LIMITS[name]
        if not low <= v <= high:
            return {"status": "invalid", "reason": f"{name}={v} is outside {low}-{high}"}
    fib4 = age * ast / (platelets * math.sqrt(alt))   # [Standard] FIB-4 formula
    cutoff = fib4_cutoff(age, rule)
    return {"status": "ok", "fib4": round(fib4, 2), "cutoff": cutoff, "cleared": fib4 < cutoff}


if __name__ == "__main__":
    print("1 cleared  :", fib4_decision(55, 28, 45, 250))
    print("2 referred :", fib4_decision(70, 60, 30, 120))
    print("3 missing  :", fib4_decision(55, 28, None, 250))
    print("4 bad value:", fib4_decision(55, 28, 0, 250))
    print("5 age 66 A :", fib4_decision(66, 30, 30, 200, "flat"))
    print("6 age 66 B :", fib4_decision(66, 30, 30, 200, "age_adjusted"))