"""Tests for the audit function run_audit (garbage input must be refused, never crash).
Run from the repo root:  python -m pytest component-2-failure-finder/tests/test_run_audit.py
These tests build their own small fake patient table, so they need no data files.
"""
import sys
sys.path.append("component-2-failure-finder/src")

import numpy as np
import pandas as pd

from run_audit import run_audit


def fake_table(n=600, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "id": range(n),
        "age": rng.integers(20, 80, n),
        "sex": rng.choice(["male", "female"], n),
        "bmi": rng.normal(32, 6, n),
        "ast": rng.normal(28, 10, n).clip(8),
        "alt": rng.normal(30, 12, n).clip(8),
        "platelets": rng.normal(250, 60, n).clip(60),
        "fibrosis": (rng.random(n) < 0.24).astype(int),
        "cleared_A": rng.random(n) < 0.6,
        "cleared_B": rng.random(n) < 0.6,
    })


def good_predictions(t, seed=2):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"SEQN": t["id"], "probability": rng.random(len(t))})


# ---- happy paths ----
def test_fib4_rule_a_runs():
    r = run_audit(fake_table(), "flat")
    assert r["status"] == "ok" and r["groups_scored"] > 0


def test_good_model_file_runs():
    t = fake_table()
    r = run_audit(t, "model", good_predictions(t))
    assert r["status"] == "ok"
    assert r["scans"] == int((t["cleared_A"] == 0).sum())   # same number of scans as FIB-4 Rule A


# ---- garbage input: refused with a reason ----
def test_no_file():
    assert run_audit(fake_table(), "model", None)["status"] == "invalid"


def test_missing_column():
    t = fake_table()
    assert run_audit(t, "model", pd.DataFrame({"SEQN": t["id"]}))["status"] == "invalid"


def test_empty_file():
    t = fake_table()
    assert run_audit(t, "model", pd.DataFrame({"SEQN": [], "probability": []}))["status"] == "invalid"


def test_text_probability():
    t = fake_table()
    c = pd.DataFrame({"SEQN": t["id"], "probability": "abc"})
    assert run_audit(t, "model", c)["status"] == "invalid"


def test_blank_probability():
    t = fake_table()
    c = good_predictions(t)
    c.loc[0, "probability"] = np.nan
    assert run_audit(t, "model", c)["status"] == "invalid"


def test_probability_out_of_range():
    t = fake_table()
    c = pd.DataFrame({"SEQN": t["id"], "probability": 5.0})
    assert run_audit(t, "model", c)["status"] == "invalid"


def test_all_probabilities_the_same():
    t = fake_table()
    c = pd.DataFrame({"SEQN": t["id"], "probability": 0.5})
    assert run_audit(t, "model", c)["status"] == "invalid"


def test_duplicate_patients():
    t = fake_table()
    c = pd.concat([good_predictions(t), good_predictions(t).head(5)])
    assert run_audit(t, "model", c)["status"] == "invalid"


def test_too_few_patients_match():
    t = fake_table()
    c = good_predictions(t).head(10)
    assert run_audit(t, "model", c)["status"] == "invalid"


def test_ids_that_do_not_match():
    t = fake_table()
    c = pd.DataFrame({"SEQN": t["id"] + 100000, "probability": np.random.default_rng(3).random(len(t))})
    assert run_audit(t, "model", c)["status"] == "invalid"