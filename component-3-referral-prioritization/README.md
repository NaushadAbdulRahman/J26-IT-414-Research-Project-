# Scarce Resource Referral Prioritization — Component 3

## Component details

- **Student:** Aaqib A.R
- **Student ID:** IT23228276
- **Specialization:** Information Technology
- **Project ID:** J26-IT-414
- **Institution:** Sri Lanka Institute of Information Technology (SLIIT)
- **Supervisor:** Dr. Prasanna Sumathipala
- **Stage:** Progress Presentation 1 (PP1), October 2026
- **Group project:** Machine-Learning Investigation of Non-Invasive Significant Liver Fibrosis Screening in MASLD

This README documents Component 3 within the shared four-member research project. This folder contains the C3 implementation, tests, dependency pins and documentation. A thin page in `app/pages/` connects its dashboard to the shared Streamlit navigation. Run all commands from the repository root unless stated otherwise.

**Repository:** https://github.com/NaushadAbdulRahman/J26-IT-414-Research-Project-

## Problem and objective

Clinics with limited scan capacity cannot scan every waiting patient immediately. This component investigates which patients to select within a fixed scan budget, comparing immediate detection of elevated liver stiffness with improvement in future model predictions.

The component has an independent workflow: prepare the cohort, train XGBoost, score waiting patients, select within the budget, reveal selected recorded outcomes, retrain independently and evaluate on held-out validation data.

The Kelaniya University dataset remains pending. The PP1 prototype uses public US NHANES 2017–March 2020 data to demonstrate the workflow; these results do not establish performance in Sri Lankan clinics.

## Technologies and environment

Python, Streamlit, XGBoost, pandas, NumPy and scikit-learn support data preparation, modelling and the dashboard. Python unittest supports output-verification tests, with Git/GitHub used for version control.

Use Python 3.12 or later. Clone the shared repository and create an environment:

```bash
git clone https://github.com/NaushadAbdulRahman/J26-IT-414-Research-Project-.git
cd J26-IT-414-Research-Project-
python -m venv .venv
```

Activate it on macOS/Linux with `source .venv/bin/activate`, or on Windows PowerShell with `.venv\Scripts\Activate.ps1`. Then install the shared dependencies:

```bash
python -m pip install -r requirements.txt
```

Continue with the C3 preparation commands below, from the repository root. A fresh clone does not contain the locally generated datasets and experiment outputs.

## Component overview


C3 compares how to allocate limited scan slots for **immediate case detection** and **future model improvement**. Its current cohort includes **all eligible adults aged 18+**, including people without diabetes.

XGBoost takes age, sex, BMI, AST, ALT, platelet count and diabetes indicators as pre-scan inputs. It outputs a score for the provisional target **observed liver stiffness ≥8 kPa**. Scan stiffness and other scan-derived measurements are excluded from predictors. This target does not establish a confirmed fibrosis diagnosis or clinical urgency, and the scores have not been calibrated.

### Selection policies

| Strategy | Patient-selection rule |
| --- | --- |
| Random | Seeded random selection as a reference |
| Risk-first | Highest predicted elevated-stiffness scores |
| Uncertainty | Scores closest to 0.5, where the model is least decisive |
| Diversity | Profiles farthest from the nearest labelled or already selected patient, with distance preprocessing fitted on the initial labelled group |
| Hybrid | Equal weighting of uncertainty and normalized diversity, updating diversity as patients are selected |

### Prepare and run

From the repository root:

```bash
python -m pip install -r component-3-referral-prioritization/requirements.txt
python component-3-referral-prioritization/src/download_nhanes.py
python component-3-referral-prioritization/src/prepare_nhanes.py
python component-3-referral-prioritization/src/train_c3_baseline.py --threshold-kpa 8
python component-3-referral-prioritization/src/compare_c3_strategies.py --budget 20
python -m unittest discover -s component-3-referral-prioritization/tests -p test_c3_verification.py -v
python -m streamlit run app/Home.py
```

`component-3-referral-prioritization/requirements.txt` records additional version pins for reproducing the C3 environment. Use a separate virtual environment if another component requires different versions.

The C3 downloader fetches six files into `data/raw/nhanes_2017_2020/`. It does not fetch `P_GHB.xpt`, which C1 and C2 require. Prepared C3 data and experiment outputs go under `data/interim/nhanes_2017_2020/`.

### Retrospective experiment

The current seed-42 cohort contains 7,768 eligible adults:

| Data group | Patients | Purpose |
| --- | ---: | --- |
| Initial labelled training | 400 | Train the starting XGBoost model |
| Simulated waiting pool | 4,260 | Candidates whose outcomes are hidden from the selection strategies |
| Validation | 1,554 | Evaluate models after retraining |
| Reserved final test | 1,554 | Excluded from the current comparison |

Each strategy starts from the same model and waiting pool, independently selects **10, 20 or 50** patients, then receives only its selected patients' recorded outcomes. A fresh XGBoost model is fitted on the initial group plus that strategy's selected batch and evaluated on the same validation group.

The current prototype runs **one batch per strategy**. Repeating a configuration reproduces the same experiment; it does not advance to a new round. The interface's **Show recorded outcomes** checkbox displays outcomes from the completed simulation and does not trigger retraining.

### Current 20-scan demonstration results

Starting validation AUROC: **0.7337**.

| Strategy | Positive scans / 20 | Precision at 20 | Validation AUROC after retraining |
| --- | ---: | ---: | ---: |
| Random | 1 / 20 | 5% | 0.7294 |
| Risk-first | 12 / 20 | 60% | 0.7286 |
| Uncertainty | 6 / 20 | 30% | 0.7232 |
| Diversity | 8 / 20 | 40% | 0.7529 |
| Hybrid | 6 / 20 | 30% | 0.7290 |

Risk-first finds the most positives in this run. Diversity produces the highest validation AUROC, improving it by **0.0192**. These are different objectives, and one split and batch do not establish a universally best strategy. Metrics are unweighted demonstration results, rather than population estimates.

### Verification and code organization

The four dashboard tabs cover the overview, selection demo, result explanations and verification. Explanations remain visible and information sections start expanded.

| File | Responsibility |
| --- | --- |
| `app/pages/3_Component_3_Referral_Prioritization.py` | Shared navigation entry that launches the C3 dashboard |
| `component-3-referral-prioritization/src/dashboard.py` | Streamlit controls and displayed results |
| `component-3-referral-prioritization/src/c3_verification.py` | Strategy descriptions and reusable output-verification functions |
| `component-3-referral-prioritization/src/train_c3_baseline.py` | Splits, initial training and baseline report |
| `component-3-referral-prioritization/src/compare_c3_strategies.py` | Five independent selection/retraining experiments and output exports |
| `component-3-referral-prioritization/tests/test_c3_verification.py` | 13 focused verification tests using synthetic fixtures |

The verification button checks partition IDs, scan budgets, unique selected patients, recorded outcomes, positive counts, precision and input-file SHA-256 fingerprints. It also recomputes AUROC, average precision and Brier score from `validation_predictions.csv`. Missing evidence is reported as unavailable, rather than counted as a pass. Rerun older comparison folders once to create this prediction export.

These checks establish numerical and file consistency. Statistical reliability still requires repeated experiments, and clinical transfer requires local evaluation.

**Planned research:** sequential acquisition rounds, repeated trials and learning curves, an XGBoost-informed acquisition extension, explicit equity constraints, budget sensitivity analysis and a statistical plateau rule. Their novelty and benefit still need literature support and experimental evaluation. A learning plateau concerns model improvement from additional labels, rather than when clinical scans should stop.

Details: [C3 demonstration and evidence guide](docs/c3-demo-guide.md)

## Implementation status at PP1

**Implemented:** all-adult cohort preparation, XGBoost baseline, five selection strategies, equal-budget comparisons, independent retraining, result explanations, output exports and numerical verification.

**Planned:** sequential rounds, repeated trials, learning curves, equity-aware constraints, an XGBoost-informed selection extension, statistical stopping rules and local-dataset evaluation when data becomes available.

## Data handling

Patient-level data, generated exports, credentials and local environment files must not be committed to the repository. Raw and prepared data remain local under the repository's Git ignore rules.

**Research use only. This prototype is not intended for clinical decision-making.**

## Component layout

| Path within this component | Purpose |
| --- | --- |
| `src/dashboard.py` | C3 Streamlit dashboard |
| `src/download_nhanes.py` | Download and validate source files |
| `src/prepare_nhanes.py` | Prepare predictors, outcomes and cohort audit |
| `src/train_c3_baseline.py` | Initial model and data partitions |
| `src/compare_c3_strategies.py` | Five selection and retraining experiments |
| `src/c3_verification.py` | Verify saved experiment outputs |
| `tests/test_c3_verification.py` | Output-verification tests |
| `docs/` | Demonstration guide and research plan |
| `requirements.txt` | C3 dependency pins, including shared requirements |

The shared navigation entry is `app/pages/3_Component_3_Referral_Prioritization.py`. Launch the application with `python -m streamlit run app/Home.py` from the repository root.

Existing data and generated results remain under the repository-level `data/raw/nhanes_2017_2020/` and `data/interim/nhanes_2017_2020/`. No data migration or retraining is required for this folder reorganization. Earlier PP1 submission copies describe the previous file locations; use the commands above for this version.
