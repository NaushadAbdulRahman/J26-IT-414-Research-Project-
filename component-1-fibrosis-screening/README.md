# Liver Fibrosis Screening Tool (Component 1)

Decides who needs a FibroScan from routine clinical data, and compares itself with FIB-4.

## Run it
    pip install -r requirements.txt
    python src/run_all.py          # trains, evaluates, saves charts in outputs/
    streamlit run src/app.py       # opens the doctor-facing demo

## Files
- src/load_data.py : where the data comes from. THE ONLY FILE TO CHANGE for new data.
- src/run_all.py   : the pipeline (model, calibration, decision layer, SHAP, FIB-4 comparison).
- src/explain.py   : SHAP reasons for whichever model is chosen.
- src/app.py       : runs the screening app on its own (`streamlit run src/app.py`).
- src/views.py     : the app's screens; also shown as a page in the group app
                     (app/pages/1_Component_1_Fibrosis_Referral.py, run `streamlit run app/Home.py` from the repo root).
- src/validate.py  : checks every value before screening (required, possible range, wrong units, cross-checks).
- src/screening.py : decision logic the app uses (same rules as run_all.py); FIB-4 comes from the group's shared/fib4.py when available.
- src/ui.py        : how the app looks (colours, type, the FIB-4 vs tool scales).
- src/fonts/       : Atkinson Hyperlegible typeface (SIL Open Font License), bundled so the app works offline.
- .streamlit/config.toml : app colours.

## Model choice
run_all.py tries logistic regression, random forest and gradient boosting, scores each by
5-fold cross-validation on the training data, and keeps the best. If models are within
0.01 AUROC (TIE_MARGIN), the simpler one is kept.
- outputs/         : results.json, charts, example patient card, saved model.

## Switching data
Open src/load_data.py and set DATA_SOURCE:
- "synthetic" : invented practice data. Results are NOT evidence. Never present them as findings.
- "nhanes"    : put P_DEMO, P_BMX, P_BIOPRO, P_CBC, P_GHB, P_DIQ, P_LUX (.XPT) in data/nhanes/.
                This loader has not been run on the real files yet - check column names if it errors.
- "local"     : save the cohort as data/local.csv with the columns listed in load_data.py.
Then run `python src/run_all.py` again.

## Setting
ALPHA in run_all.py = allowed miss rate (0.10 = at most about 10% of sick patients cleared).

## The screening app
`streamlit run src/app.py` opens five pages (left sidebar):
- **Screen a patient**: enter values or load an example patient; get SCAN / NO SCAN / UNSURE,
  the risk, FIB-4 for comparison, the main reasons, and a downloadable report.
- **Screen many patients**: upload a CSV (download the template first) or try 25 sample
  NHANES patients; get a sorted referral list, a summary against FIB-4, and a CSV download.
- **Session history**: everyone screened since the app was opened (kept in memory only).
- **Model performance**: test-set results, the fair comparison with FIB-4, and the charts.
- **About**: how the tool decides, the data, and its limitations.

