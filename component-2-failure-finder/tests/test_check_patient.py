"""Tests for Function 2: check_patient (piece 1) and red_flag_check (piece 2).
Run from the repo root:  python component-2-failure-finder/tests/test_check_patient.py
"""
import sys
sys.path.append("component-2-failure-finder/src")

from check_patient import fib4_decision
from red_flag_check import check_patient


# ---- happy paths ----
def test_cleared_value():
    r = fib4_decision(55, 28, 45, 250)
    assert r["status"] == "ok" and r["fib4"] == 0.92 and r["cleared"] is True

def test_referred_value():
    r = fib4_decision(70, 60, 30, 120)
    assert r["fib4"] == 6.39 and r["cleared"] is False

def test_rule_a_vs_rule_b_same_patient():
    flat = fib4_decision(66, 30, 30, 200, "flat")
    adj = fib4_decision(66, 30, 30, 200, "age_adjusted")
    assert flat["fib4"] == adj["fib4"] == 1.81
    assert flat["cleared"] is False and adj["cleared"] is True

def test_red_flag_found():
    r = check_patient(55, "male", 38, 28, 45, 250, rule="flat")
    assert r["cleared"] and len(r["flags"]) == 9
    assert r["flags"][0]["wrongly_cleared"] <= r["flags"][0]["cleared"]

def test_referred_patient_gets_no_flags():
    r = check_patient(70, "male", 40, 60, 30, 120, rule="flat")
    assert r["cleared"] is False and r["flags"] == [] and "bar" in r

# ---- boundary: bins must match the audit ----
def test_bmi_35_inside_34_9_outside():
    assert len(check_patient(55, "male", 35, 28, 30, 250)["flags"]) == 6
    assert len(check_patient(55, "male", 34.9, 28, 30, 250)["flags"]) == 0

# ---- failure paths ----
def test_missing_alt():
    assert fib4_decision(55, 28, None, 250)["status"] == "invalid"

def test_zero_alt():
    assert fib4_decision(55, 28, 0, 250)["status"] == "invalid"

def test_text_value():
    assert fib4_decision(55, "high", 45, 250)["status"] == "invalid"

def test_bad_sex():
    assert check_patient(55, "x", 38, 28, 45, 250)["status"] == "invalid"

def test_bad_bmi():
    assert check_patient(55, "male", 5, 28, 45, 250)["status"] == "invalid"


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("PASS", name)
            except AssertionError:
                failed += 1
                print("FAIL", name)
    print("\n", "ALL PASSED" if failed == 0 else f"{failed} FAILED")