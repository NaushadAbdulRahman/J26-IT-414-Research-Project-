# Component 1: Fibrosis Screening and FibroScan Referral

Decides which patients with fatty liver disease (MASLD) need a FibroScan, from 12 routine results,
and compares itself with FIB-4. Each patient gets **SCAN / NO SCAN / UNSURE**, an estimated risk,
and the main reasons (SHAP).

Owner: Sharuka G (IT23229884) · Project J26-IT-414

## Status (PP1)

All six specific objectives in the proposal have working code and results on public NHANES data:

| Objective (proposal 2.2) | Where | Result |
|---|---|---|
| 1. Variables and FibroScan label | `src/load_data.py` | 2,984 fatty-liver patients, 520 (17%) with stiffness ≥ 8 kPa |
| 2. Calibrated ML model, FIB-4 benchmark | `src/run_all.py` | Logistic regression chosen by 5-fold CV; calibration method also chosen on training data (sigmoid) |
| 3. Conformal three-zone decision with a miss bound | `src/run_all.py` | Allowed miss rate 10%; test set: 2.9% missed (95% CI 0 to 6.4%) |
| 4. Thresholds under limited scans | `src/analyses.py` | At every scan budget from 20% to 60%, fewer sick patients missed than FIB-4 |
| 5. Comparison with FIB-4 and an uncalibrated model, with uncertainty | `src/analyses.py` | AUROC 0.77 vs 0.51 (difference 95% CI 0.19 to 0.33); bootstrap CIs; DCA; calibration |
| 6. Global and per-patient explanations, prototype | `src/explain.py`, app | SHAP per patient and overall; Streamlit app with five screens |

Research questions (proposal 2.3), answered on the test set:
- **RQ1** bound kept? Yes on average: over 500 new calibration/test splits, 8.4% of sick patients missed (allowed 10%).
- **RQ2** fewer scans than FIB-4 at that safety level? Guideline FIB-4 scans fewer (27%) but misses 70% of sick patients.
  To miss at most 10%, FIB-4 needs 83% scans and the tool 56%. With the same number of scans as FIB-4, the tool misses 47% vs 70%.
- **RQ3** FIB-4's parts vs the score? No clear difference (test AUROC 0.77 vs 0.77); the gain comes from the other results.
- **RQ4** do explanations make sense? Top drivers: BMI, waist, age, AST, GGT, all recognised risk factors.

Still to do: validation on local (Sri Lankan) patients when data is available; external check on
NHANES 2021-2023 (code ready, add the files to `data/nhanes_2021/`).

## Run it

```bash
cd component-1-fibrosis-screening
source .venv/bin/activate          # or: pip install -r requirements.txt
python src/run_all.py              # about 1 minute: trains, evaluates, saves outputs/
python -m pytest -q                # 21 automated checks
python -m streamlit run src/app.py # the screening app on its own
```
Inside the group app: from the repo root, `python -m streamlit run app/Home.py`.

## Data

`src/load_data.py` is the only file to change for new data (`DATA_SOURCE`, `COHORT`).
- **nhanes** (default): NHANES 2017 to March 2020. Put P_DEMO, P_BMX, P_BIOPRO, P_CBC, P_GHB, P_DIQ, P_LUX (.xpt) in `data/nhanes/`.
  Kept: adults with a complete, reliable FibroScan (IQR/median ≤ 30%), CAP ≥ 275 dB/m and at least one
  cardiometabolic risk factor. Label: stiffness ≥ 8.0 kPa (significant fibrosis, F2 or worse).
  `COHORT = "all"` uses every adult instead (also reported as a sensitivity analysis).
- **External check**: NHANES August 2021 to August 2023. Put DEMO_L, BMX_L, BIOPRO_L, CBC_L, GHB_L, DIQ_L, LUX_L in
  `data/nhanes_2021/`; `run_all.py` then tests the frozen model on them automatically.
- **local**: the local cohort as `data/local.csv` with the columns listed in `load_data.py`.
- **synthetic**: invented practice data. Never present it as a result.

Patient data and outputs are never committed (see `.gitignore`).

## Method (matches `docs/C1_Build_Pipeline.png`)
1. Data and cohort selection. 2. Split 60/20/20 into training, calibration and test; the test set is
used once, at the end. 3. Model: logistic regression, random forest and XGBoost compared by 5-fold CV
AUROC on training data; the simplest within 0.01 is kept. 4. Calibration: none, sigmoid or isotonic,
chosen by 5-fold CV Brier score on training data. 5. Decision layer: class-conditional split conformal
prediction on the calibration patients (`ALPHA = 0.10`). 6. SHAP explanations. 7. App.
Benchmarks: FIB-4 (1.30/2.67 and age-adjusted), the uncalibrated model, scan budgets, all-adults cohort.

## Files
- `src/load_data.py` : data sources, cohort rules, external dataset.
- `src/run_all.py`   : the pipeline; writes `outputs/results.json`, charts, example card, `model.joblib`.
- `src/analyses.py`  : bootstrap CIs, calibration metrics, uncalibrated benchmark, bound stress test,
  scan budgets, composite vs decomposed FIB-4, SHAP alignment, external validation.
- `src/explain.py`   : SHAP for whichever model is chosen.
- `src/screening.py` : decision logic for the app (same rules as `run_all.py`); FIB-4 from the group's `shared/fib4.py` when available.
- `src/validate.py`  : checks every value before screening (required, ranges, unit mix-ups, cross-checks).
- `src/views.py`     : the app's five screens; `src/app.py` runs them alone, `app/pages/1_...py` inside the group app.
- `src/ui.py`        : look and feel; follows light or dark theme. `src/fonts/`: Atkinson Hyperlegible (OFL).
- `tests/`           : automated checks (no patient data needed).

## The app
- **Screen a patient**: decision, risk, where the patient falls on FIB-4 and on the tool, SHAP reasons, report download.
- **Screen many patients**: upload a CSV; rows are checked, then a sorted referral list with FIB-4 alongside.
- **Session history**, **Model performance** (results, research questions, benchmarks), **About**.

## Limits
US survey data, not Sri Lankan clinic patients. MASLD is approximated (alcohol and viral hepatitis
not checked). FibroScan stiffness is not biopsy. Research prototype, not for clinical use.
