# Component 2 – User and Functional Requirements

## Users
| Role | Who | What they need from Component 2 |
|---|---|---|
| Primary user | Doctor at a diabetes clinic | Know when a patient FIB-4 cleared still deserves a second look, with the evidence behind it |
| Secondary user | Screening programme lead / model developer (e.g. Component 1) | Know which kinds of patients a screening tool wrongly clears, before trusting it |
| Decision-maker | Hospital or clinic management | Evidence that extra scans go to groups that find more fibrosis than current referrals |

## User requirements
- UR1: A doctor can check one patient using routine results only (age, sex, BMI, AST, ALT, platelets).
- UR2: The result says plainly whether FIB-4 clears or refers the patient, and why.
- UR3: A flag shows its evidence (how many similar patients had fibrosis, compared with the referral bar), not just "high risk". 
- UR4: A programme lead can audit FIB-4 or another tool's decisions without writing code.
- UR5: The system says clearly when no group is worth a scan, instead of forcing a result.
- UR6: Results are labelled as candidates on US data until tested on local patients.

## Functional requirements
| ID | Requirement | Function | Evidence |
|---|---|---|---|
| FR1 | Load the patient table with FibroScan truth; stop with a message if it is missing | F1 | Page error message |
| FR2 | Audit FIB-4 Rule A (1.3) and Rule B (2.0 if 65+) | F1 | `run_audit.py flat` → 498 / 137 / 162 / 27.5% |
| FR3 | Audit any model from a CSV (SEQN, probability), giving it the same number of scans as FIB-4 Rule A; refuse a file without those columns | F1 | C1 audit → 425 / 177 / 70; refusal tested |
| FR4 | Score every group of 1–3 traits with fixed bins; skip groups under 30 patients or 10 cases | F1 | `group_tools.py`, 275 groups (Rule A) |
| FR5 | Test each group against the referral bar (exact binomial) and correct for many tests (Benjamini–Hochberg) | F1 | 9 groups pass (Rule A); null test 4/200 |
| FR6 | Retest surviving groups on unseen patients over 20 fixed splits | F1 | `repeat_splits.py`: 5–6 of 20 hold |
| FR7 | Show counts, the bar, the groups worth a scan and a blind-spot map on screen | F1 | Audit tab |
| FR8 | For one patient: compute FIB-4, decide cleared or referred, and if cleared list matching groups with evidence | F2 | Checker tab; unit tests |
| FR9 | Reject missing, zero, text or out-of-range input with a reason | F2 | 11 unit tests |

## Non-functional requirements
| ID | Requirement | Evidence so far |
|---|---|---|
| NFR1 Reliability | False alarms on data with no real group stay at or below 5% | Null test: 2.0% of 200 runs |
| NFR2 Performance | One audit runs in a few seconds on a laptop | About 0.4 s per audit in testing |
| NFR3 Privacy | No patient data in the repository; only de-identified public data | `.gitignore` blocks data files |
| NFR4 Maintainability | One shared audit function used by both the page and the command line; README with function specs | `run_audit.py`, `group_tools.py`, README |
| NFR5 Usability | A doctor completes a patient check without help |