# Component 2: Failure-Finder

**Owner:** Abdul Rahman N. (IT23365896) · **Project:** J26-IT-414 · **Specialization:** Data Science

## The problem in one sentence
Which kinds of diabetic patients does a fibrosis screening tool (FIB-4, or any model) wrongly clear, and can we find those groups reliably enough to flag them?

**Why it matters:** FIB-4 uses a blood test to decide who needs a FibroScan. A cleared patient goes home and is never scanned, so nobody learns when FIB-4 was wrong. This component checks the decisions against FibroScan results to find the kinds of patients that get wrongly cleared.

## How it works
| Step | Question | Script |
|---|---|---|
| 0. Get patients | Where does the truth come from? | `src/build_cohort.py`, `component-2-failure-finder/src/build_table.py` |
| 1. Find groups | Which kinds of cleared patients exist? (1 to 3 traits, fixed bins) | `find_groups.py`, `group_tools.py` |
| 2. Worth a scan? | Is the group's fibrosis rate above the referral bar? (exact binomial test) | `test_groups.py` |
| 3. Not luck? | Benjamini-Hochberg correction, then a split-half retest on unseen patients | `retest_groups.py`, `repeat_splits.py` |
| 4. Use it | Does this cleared patient fall in a surviving group? | `check_patient.py`, `red_flag_check.py` |
| Any tool | Same audit for any screening decision, with equal scans | `run_audit.py`, `check_c1_connection.py`, `audit_c1_model.py` |

**Referral bar:** the share of fibrosis among patients the tool already refers (27.5% for FIB-4 Rule A).
**Truth:** FibroScan of 8 kPa or more. **Data:** US NHANES 2017 to March 2020, 1,249 adult diabetics with a complete FibroScan.

## Function specifications
### Function 1: Audit a screening tool (steps 1 to 3)
| | |
|---|---|
| Owner | Abdul Rahman N. |
| Input | Patient table (age, sex, BMI, AST, ALT, platelets, fibrosis truth) plus a screening decision: FIB-4 Rule A, Rule B, or Component 1 predictions (CSV with `SEQN`, `probability`, `referred`) |
| Processing | Score every group of 1 to 3 traits (min 30 patients, min 10 cases); exact one-sided binomial test of each group against the referral bar; Benjamini-Hochberg correction. For a model, it gets the same number of scans as FIB-4 Rule A |
| Output | Scans used, fibrosis found, wrongly cleared, the bar, the groups that pass the luck check, and a blind-spot map |
| Dependency | `shared/fib4.py`; Component 1 predictions (optional) |
| Test evidence | Null test: 4 of 200 runs flagged a group (2.0%); `check_group_tools.py`; Component 1 audit reproduces Sharuka's counts |

### Function 2: Red-flag check for one patient (step 4)
| | |
|---|---|
| Owner | Abdul Rahman N. |
| Input | One patient's age, sex, BMI, AST, ALT, platelets and the FIB-4 rule |
| Processing | Compute FIB-4; if cleared, match the patient to the groups that passed the luck check |
| Output | Cleared or referred; if cleared, the matching groups with their evidence, or a clear refusal for invalid input |
| Dependency | Tested groups from Function 1 (`c2_groups_<rule>_tested.csv`) |
| Test evidence | 11 unit tests in `tests/test_check_patient.py` (happy path, bin edges, missing, zero and text input) |

## Results so far (US data, prototype)
- **FIB-4 Rule A:** 1,249 patients, 299 with fibrosis. 751 cleared, 162 wrongly cleared. **9 of 275 groups** pass the luck check, all containing BMI 35 or more. They held up on unseen patients in 5 to 6 of 20 splits.
- **FIB-4 Rule B:** 4 groups pass the luck check, 0 of 20 splits hold.
- **Component 1 model, equal scans (1,011 patients):** FIB-4 finds 121 fibrosis patients with 425 scans, the model finds 177. **No group passes the luck check** for the model. A BMI and liver-enzyme pattern shows up as a candidate blind spot only.

## Run it
From the repo root, venv active:
```
python src\build_cohort.py
python component-2-failure-finder\src\build_table.py
python component-2-failure-finder\src\find_groups.py flat
python component-2-failure-finder\src\test_groups.py flat
python component-2-failure-finder\src\run_audit.py flat
python component-2-failure-finder\src\null_test.py
python -m pytest component-2-failure-finder\tests
streamlit run app\Home.py
```
Data files are not committed. Raw NHANES files go in `data/raw/`.

## Limitations (stated honestly)
- US data only. Groups are candidates until tested on local patients.
- The BMI finding is not proven: the retest held in only 5 to 6 of 20 splits.
- The audit's cohort definition differs slightly from Component 1's. We will agree one definition.
- Not yet done: the rescue test (do red flags beat simply lowering the cut-off at equal extra scans?) and a test with planted groups to measure what size of group can be detected.