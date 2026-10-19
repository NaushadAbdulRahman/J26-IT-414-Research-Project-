# [Group Project Title]

**Project ID:** J26-IT-414
**Institution:** Sri Lanka Institute of Information Technology (SLIIT), Faculty of Computing
**Supervisor:** [Name]  |  **Co-supervisor:** [Name]
**Clinical partner:** [Name, institution]

---

## 1. Project Overview

Metabolic dysfunction-associated steatotic liver disease (MASLD) is common and usually silent until
the liver is badly scarred (fibrosis). Fibrosis is confirmed with a FibroScan, but very few machines
are available in Sri Lanka, so doctors cannot scan everyone. Today the decision of who gets scanned
relies on FIB-4, a simple blood-test score with a wide "unsure" range that both wastes scans and
misses some patients.

This project builds a machine-learning screening system that uses routine clinical data to help
doctors decide which patients need a FibroScan, so the limited machines are used for the right people.

## 2. Research Problem

[One paragraph, agreed by the whole group.]

## 3. Main Objective

[One sentence, agreed by the whole group.]

## 4. System Architecture

[Insert the group's overall system architecture diagram, e.g. `docs/system_architecture.png`.]

## 5. Components

| # | Component | Member | Student ID | Folder |
|---|-----------|--------|------------|--------|
| 1 | Machine Learning Screening Tool for Fibrosis Risk Prediction and FibroScan Referral | [Your name] | IT23229884 | `component-1-fibrosis-screening/` |
| 2 | [Title] | [Name] | [ID] | `component-2-.../` |
| 3 | [Title] | [Name] | [ID] | `component-3-.../` |
| 4 | [Title] | [Name] | [ID] | `component-4-.../` |

### Component 1: Machine Learning Screening Tool for Fibrosis Risk Prediction and FibroScan Referral

**Problem.** FIB-4 decides who is sent for a FibroScan, but its indeterminate zone (1.30-2.67) is wide
and it misses some patients with significant fibrosis.

**Solution.** A screening tool that takes routine clinical data and returns, for each patient,
**SCAN / NO SCAN / UNSURE**, an estimated risk, and the main reasons for the decision. It is
evaluated head-to-head against FIB-4 on the same patients.

**How it is built** (see `docs/C1_Build_Pipeline.png`):

1. **Data**: age, sex, BMI, waist, diabetes, AST, ALT, GGT, platelets, albumin, HbA1c, triglycerides;
   label = significant fibrosis (>= F2) on FibroScan.
2. **Preprocessing**: missing-value handling and a locked train / calibration / test split.
3. **Model selection**: logistic regression, random forest and gradient boosting (XGBoost) compared
   by 5-fold cross-validation; the best is kept, preferring the simpler model when scores are tied.
4. **Calibration**: none, sigmoid or isotonic, chosen by cross-validated Brier score on the training data.
5. **Decision layer**: class-conditional conformal prediction turns risk into SCAN / NO SCAN / UNSURE,
   with the allowed miss rate set in advance.
6. **Explanation**: SHAP values give per-patient reasons.
7. **Output**: a demo screening app.
8. **Benchmark**: FIB-4 and the uncalibrated model on the same test patients: miss rate, scans needed,
   AUROC with bootstrap 95% CIs, calibration, decision-curve analysis, limited-scan budgets, and a
   500-split stress test of the safety bound.

**Data.** Public NHANES 2017-March 2020 data: 2,984 adults with fatty liver (CAP >= 275 dB/m plus a
metabolic risk factor) and a reliable FibroScan; 520 with significant fibrosis (>= 8 kPa). To be
recalibrated and validated on local patients; an external check on NHANES 2021-2023 is ready to run.

**Status (PP1).** All six proposal objectives implemented and evaluated on NHANES. On 597 unseen test
patients, the tool missed 2.9% of sick patients (allowed 10%) while FIB-4 missed 70%; AUROC 0.77 vs 0.51.
Includes input validation, a five-screen app, and 21 automated tests. Pending: local-cohort validation.

**Run it.**

```bash
cd component-1-fibrosis-screening
pip install -r requirements.txt
python src/run_all.py        # train, evaluate, save results and charts to outputs/
python -m pytest -q          # automated checks
streamlit run src/app.py     # open the demo screening tool
```

Details: [`component-1-fibrosis-screening/README.md`](component-1-fibrosis-screening/README.md)

## 6. Technologies

| Area | Tools |
|------|-------|
| Language | Python 3.10+ |
| Data | pandas, NumPy |
| Machine learning | scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualisation | Matplotlib |
| Demo interface | Streamlit |
| Version control / project management | Git, GitHub, Microsoft Planner |

[Teammates: add the tools your components use.]

## 7. Data Privacy

Patient data is **never** committed to this repository. All `data/` folders and data files are
excluded by `.gitignore`. The local dataset is de-identified and stored only in the locations
approved by the clinical partner.

## 8. Disclaimer

This is a research prototype. It is not a medical device and must not be used for clinical decisions.
