# Fibrosis Screening and FibroScan Referral — Component 1

## Component details

- **Student:** Sharuka G
- **Student ID:** IT23229884
- **Specialization:** Information Technology
- **Project ID:** J26-IT-414
- **Institution:** Sri Lanka Institute of Information Technology (SLIIT)
- **Supervisor:** Dr. Prasanna Sumathipala
- **Stage:** Progress Presentation 1 (PP1), October 2026
- **Group project:** Machine-Learning Investigation of Non-Invasive Significant Liver Fibrosis Screening in MASLD

This README documents Component 1 within the shared four-member research project. All Component 1 code,
tests and documentation are in the `component-1-fibrosis-screening/` folder; its page in the group app is
`app/pages/1_Component_1_Fibrosis_Referral.py`.

**Repository:** https://github.com/NaushadAbdulRahman/J26-IT-414-Research-Project-

## Problem and objective

FibroScan machines are scarce in Sri Lanka, so doctors use FIB-4 (age, AST, ALT, platelets) to decide who
gets a scan. FIB-4 has a wide "unsure" zone (1.30–2.67) and gives no limit on how many sick patients it
sends home. In the PP1 test data it missed 70% of fatty-liver patients with significant fibrosis.

**Objective:** a machine-learning screening tool that uses 12 routine results to give each patient a
calibrated risk, a **NO SCAN / UNSURE / REFER** decision with a pre-set limit on missed patients, and the
reasons for the decision, evaluated head-to-head against FIB-4 on the same patients.

The Kelaniya University dataset remains pending. The PP1 prototype uses public US NHANES 2017–March 2020
data; these results do not establish performance in Sri Lankan clinics.

## Technologies and environment

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Matplotlib, joblib and Streamlit; pytest for automated
tests; Git/GitHub for version control.

Use Python 3.10 or later. Clone the shared repository and create an environment for Component 1:

```bash
git clone https://github.com/NaushadAbdulRahman/J26-IT-414-Research-Project-.git
cd J26-IT-414-Research-Project-/component-1-fibrosis-screening
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Component overview

**Inputs (12 routine results):** age, sex, BMI, waist, diabetes, AST, ALT, GGT, platelets, albumin,
HbA1c, triglycerides. FibroScan measurements are never used as inputs.

**Label:** significant fibrosis = liver stiffness **≥ 8 kPa** on FibroScan.

**Outputs for each patient:** calibrated risk %, decision (**No scan** below 7.6%, **Refer** above 34.9%,
**Unsure** in between), SHAP reasons, and FIB-4 shown alongside.

### Method

| Step | What is done |
|---|---|
| 1. Cohort | NHANES adults with a complete, reliable FibroScan (IQR/median ≤ 30%), CAP ≥ 275 dB/m and at least one cardiometabolic risk factor: **2,984 patients, 520 (17%) with fibrosis** |
| 2. Split | 60 / 20 / 20 stratified: training 1,790, calibration 597, test 597 (test used once, at the end) |
| 3. Preprocessing | Median imputation and standard scaling inside the model pipeline, fitted on training data only |
| 4. Model | Logistic regression, random forest and XGBoost compared by 5-fold cross-validated AUROC (0.804 / 0.806 / 0.792); the simplest within 0.01 is kept → **logistic regression** |
| 5. Calibration | None, sigmoid or isotonic chosen by cross-validated Brier score on training data → **sigmoid** |
| 6. Decision layer | Class-conditional split conformal prediction on the calibration patients, allowed miss rate **10%** (`ALPHA = 0.10`) |
| 7. Explanation | SHAP values per patient and overall |
| 8. Benchmarks | FIB-4 (1.30/2.67 and age-adjusted), the uncalibrated model, limited scan budgets, all-adults cohort |

### Prepare and run

Download the seven NHANES 2017–March 2020 files into `component-1-fibrosis-screening/data/nhanes/`:

```bash
mkdir -p data/nhanes
for f in P_DEMO P_BMX P_BIOPRO P_CBC P_GHB P_DIQ P_LUX; do
  curl -L -o data/nhanes/$f.xpt https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/$f.xpt
done
```

Then, from `component-1-fibrosis-screening/`:

```bash
python src/run_all.py                  # trains, evaluates, saves results and charts to outputs/ (~1 min)
python -m pytest -q                    # 21 automated tests
python src/trace_patient.py C          # recomputes one patient's result step by step
python -m streamlit run src/app.py     # the screening app on its own
```

Inside the group app, from the repository root: `python -m streamlit run app/Home.py`.

## Current results (597 unseen test patients)

| Measure | This tool | FIB-4 (1.30 / 2.67) |
|---|---:|---:|
| Patients with fibrosis missed | **2.9%** (95% CI 0–6.4%) | 70.2% |
| AUROC | **0.77** | 0.51 |
| Patients sent for a scan | 64% | 27% |
| Missed, with the same number of scans as FIB-4 | **47%** | 70% |
| Scans needed to miss at most 10% | **56%** | 83% |

AUROC difference 95% CI 0.19 to 0.33 (bootstrap). Safety bound check: over 500 new calibration/test
splits, the tool missed 8.4% of sick patients on average (allowed 10%). Top SHAP drivers: BMI, waist, age,
AST and GGT.

The trade-off is stated openly: the tool sends more patients for a scan than guideline FIB-4, in exchange
for missing far fewer.

## Verification and code organization

`tests/test_component1.py` holds **21 pytest tests** that need no patient data: the FIB-4 formula against a
hand calculation and the group's shared helper; input validation (required fields, ranges, unit mix-ups,
impossible values, patient-ID format, CSV rows); the three decision zones; the conformal miss bound on
repeated synthetic data; and data loading. `src/trace_patient.py` recomputes an example patient by hand
(scaling, weights, calibration, decision, SHAP) and matches the app.

| File | Responsibility |
|---|---|
| `src/load_data.py` | Data sources, cohort rules, label, external dataset |
| `src/run_all.py` | Full pipeline; writes `outputs/results.json`, charts and `model.joblib` |
| `src/analyses.py` | Bootstrap CIs, calibration, uncalibrated benchmark, bound stress test, scan budgets, SHAP alignment, external validation |
| `src/explain.py` | SHAP for the chosen model |
| `src/screening.py` | Decision logic used by the app (same rules as `run_all.py`) |
| `src/validate.py` | Checks every input before screening |
| `src/views.py`, `src/ui.py`, `src/app.py` | Streamlit screens and theme |
| `src/export_predictions.py` | Out-of-fold predictions shared with Component 2 |
| `src/trace_patient.py` | Step-by-step recomputation of one patient |
| `tests/test_component1.py` | 21 automated tests |

The app has five screens: **Screen a patient**, **Screen many patients** (CSV upload, sorted referral
list), **Session history**, **Model performance** and **About**.

## Implementation status at PP1

**Implemented:** cohort preparation, model selection, calibration, conformal three-zone decision, SHAP
explanations, FIB-4 and uncalibrated benchmarks, limited-scan analysis, bound stress test, input
validation, five-screen app, predictions shared with Component 2, 21 automated tests.

**Planned:** validation and recalibration on the Kelaniya University cohort, external check on NHANES
2021–2023 (code ready), clinician feedback on the app, and an adjustable miss limit in the app.

## Limitations

US survey data, not Sri Lankan clinic patients; South Asian patients may develop fibrosis at a lower BMI.
MASLD is approximated (alcohol and viral hepatitis not checked). FibroScan stiffness is used as the
reference, not biopsy.

## Data handling

Patient-level data, generated outputs, models and local environment files are never committed. Raw data
stays local under the repository's Git ignore rules.

**Research use only. This prototype is not intended for clinical decision-making.**
