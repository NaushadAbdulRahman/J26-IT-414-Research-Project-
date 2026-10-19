"""
Automated checks for Component 1. Run from component-1-fibrosis-screening/:

    python -m pytest -q

They need no patient data: they use made-up numbers and the practice
("synthetic") data, so they also run on a fresh copy of the repository.
"""
import os
import sys
import math

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))          # repo root, for the group's shared/ folder

import analyses as an          # noqa: E402
import load_data as ld         # noqa: E402
import validate                # noqa: E402

GOOD = dict(age=55, sex="Male", diabetes="No", bmi=29.0, waist=98.0, ast=38.0, alt=42.0,
            ggt=45.0, platelets=210.0, albumin=4.2, hba1c=6.0, triglycerides=150.0)


# ---------- FIB-4 ----------
def test_fib4_formula_matches_hand_calculation():
    # (55 x 38) / (210 x sqrt(42)) = 0.2412...
    df = pd.DataFrame([dict(age=55, ast=38, alt=42, platelets=210)])
    assert an.fib4(df).iloc[0] == pytest.approx(55 * 38 / (210 * math.sqrt(42)), rel=1e-9)


def test_fib4_agrees_with_group_shared_helper():
    shared = pytest.importorskip("shared.fib4", reason="shared/ folder not on the path")
    df = pd.DataFrame([dict(age=64, ast=58, alt=40, platelets=150)])
    assert an.fib4(df).iloc[0] == pytest.approx(shared.fib4(64, 58, 40, 150))


# ---------- input checks ----------
def test_valid_patient_has_no_errors():
    errors, warnings, filled = validate.check(GOOD)
    assert errors == {} and warnings == [] and filled == []


@pytest.mark.parametrize("field", ["age", "sex", "diabetes", "ast", "alt", "platelets"])
def test_required_fields_are_required(field):
    errors, _, _ = validate.check({**GOOD, field: None})
    assert field in errors


def test_optional_fields_can_be_blank():
    errors, _, filled = validate.check({**GOOD, "bmi": None, "ggt": None})
    assert errors == {} and set(filled) == {"BMI", "GGT"}


@pytest.mark.parametrize("field,value,unit", [("albumin", 42.0, "g/L"),
                                              ("hba1c", 48.0, "mmol/mol"),
                                              ("triglycerides", 1.7, "mmol/L")])
def test_wrong_units_are_caught(field, value, unit):
    errors, _, _ = validate.check({**GOOD, field: value})
    assert unit in errors[field]


def test_children_and_impossible_values_are_rejected():
    assert "age" in validate.check({**GOOD, "age": 15})[0]
    assert "alt" in validate.check({**GOOD, "alt": -3.0})[0]


def test_unusual_but_possible_values_only_warn():
    errors, warnings, _ = validate.check({**GOOD, "hba1c": 7.0})      # diabetes says No
    assert errors == {} and any("Diabetes" in w for w in warnings)


def test_patient_id_must_not_contain_odd_characters():
    assert "pid" in validate.check(GOOD, "Name <script>")[0]
    assert "pid" not in validate.check(GOOD, "CLN-0042")[0]


def test_spreadsheet_rows_checked_one_by_one():
    df = pd.DataFrame([{**GOOD, "sex": 1, "diabetes": 0},
                       {**GOOD, "sex": 1, "diabetes": 0, "age": 15}])
    errs, _ = validate.check_table(df)
    assert errs[0] == "" and "Age" in errs[1]


# ---------- decision layer ----------
def test_three_zones_follow_the_thresholds():
    p = np.array([0.01, 0.5, 0.99])
    cleared = an.cleared_by_model(p, q_pos=0.9, q_neg=0.05)
    assert cleared.tolist() == [True, False, False]


def test_conformal_bound_holds_on_average():
    """Made-up scores: sick patients tend to score higher. Over many random splits the
    share of sick patients cleared must be about alpha or less (the promise)."""
    rng = np.random.default_rng(0)
    alpha, misses = 0.10, []
    for _ in range(300):
        y = (rng.random(2000) < 0.2).astype(int)
        p = np.clip(rng.normal(0.25 + 0.3 * y, 0.15), 0.001, 0.999)
        cal, te = np.arange(1000), np.arange(1000, 2000)
        q_pos, q_neg = an.conformal_thresholds(p[cal], y[cal], alpha)
        _, miss = an.scans_and_misses(an.cleared_by_model(p[te], q_pos, q_neg), y[te])
        misses.append(miss)
    assert np.mean(misses) <= 100 * alpha + 1.0


# ---------- data ----------
def test_practice_data_has_the_expected_columns():
    df = ld._synthetic(n=200)
    assert list(df.columns) == ld.FEATURES + ["label"]
    assert set(df.label.unique()) <= {0, 1}


def test_external_data_is_optional():
    # returns None (not an error) when the 2021-2023 files have not been added
    if not os.path.isdir(os.path.join(ld.DATA_DIR, "nhanes_2021")):
        assert ld.load_external() is None
