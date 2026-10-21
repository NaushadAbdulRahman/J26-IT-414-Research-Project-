# CardioLiver-AI: AI-Assisted Cardiovascular Risk Prediction and Explainable Clinical Decision Support in MASLD Patients

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-0f766e.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Modern_API-0d9488.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19_Vite-61dafb.svg)](https://react.dev/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-0284c7.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-7c3aed.svg)](https://shap.readthedocs.io/)
[![Cohort](https://img.shields.io/badge/Cohort-MASLD_Clinical_Registry-065f46.svg)]()

> **Research Goal:** To evaluate whether liver fibrosis severity improves 10-year cardiovascular disease (CVD) risk prediction beyond existing risk scores (SLCVD / WHO) in patients with Metabolic Dysfunction-Associated Steatotic Liver Disease (MASLD) and provide explainable, patient-specific clinical decision support.

---

## Architecture & Infographic Workflow

```
[1. Data Input] ──> [2. Data Preprocessing] ──> [3. AI Model Development] ──> [4. CVD Risk Prediction]
 Demographic          Median/Mode Imputation      Build Model A (Baseline)      Best Model Probability %
 Metabolic Factors    One-Hot Encoding            Build Model B (Enhanced)      Low / Moderate / High
 Lab Parameters       Robust Scaler               LR vs RF vs XGBoost           Interactive SVG Gauge
 Liver Fibrosis F0-F4 Train/Test Split (80/20)    Benchmark & ΔAUC
 Baseline CVD Scores
                                                         │
                                                         ▼
[6. Clinical Decision Support Report] <─────── [5. Explainable AI (SHAP)]
 Predicted Risk %                              Global Feature Importance (|SHAP|)
 Model A vs Model B Incremental Shift          Patient-Level Waterfall (+ & - forces)
 Fibrosis Contribution % (e.g., 34.2%)         Fibrosis Relative Attribution Share
 Actionable Clinical Guidance                  Interaction Analysis (Fibrosis × Diabetes)
```

### Core Modules:
1. **Data Input & Cohort Management**:
   - Calibrated to the **MASLD Clinical Patient Cohort — Sri Lanka Registry** (1,200 records with 22 clinical variables).
   - Dynamic cohort upload and validation via web interface or CSV file.
2. **AI Model Development & Incremental Value Analysis**:
   - **Model A (Baseline):** Traditional CVD risk factors.
   - **Model B (Fibrosis-Enhanced):** Traditional CVD risk factors + Liver Fibrosis Severity ($F_0 - F_4$) and transaminases.
   - Compares **Logistic Regression**, **Random Forest**, and **XGBoost**.
   - Computes $\Delta\text{AUC}$ gain, Net Reclassification Improvement (NRI), Integrated Discrimination Improvement (IDI), and Brier calibration score.
   - **Decision Curve Analysis (DCA):** Quantifies net clinical benefit across decision thresholds (5%–45%).
   - **Calibration Curves:** Decile observed vs. predicted reliability plots.
3. **Explainable AI (XAI)**:
   - **Global SHAP:** Population-level feature attribution rankings.
   - **Patient-Level SHAP Waterfall:** Decomposes individualized risk into base probability, risk-increasing factors (coral), and risk-reducing factors (teal).
   - **Fibrosis Contribution %:** Quantifies the relative share of elevated CVD risk attributed directly to liver fibrosis severity.
   - **Synergy & Interaction Analysis:** Identifies synergistic risk escalation when advanced fibrosis ($F_3 - F_4$) coexists with Type 2 Diabetes.
4. **Evidence-Based Clinical Decision Support Engine**:
   - Actionable guidelines across Cardiovascular Surveillance, Lipid & Statin Strategies, Hepatology/HCC Screening, and Dual Cardio-Hepatic Pharmacotherapy (GLP-1 RA / SGLT2i).
   - Printable patient clinical summary chart with download options.

---

## Quickstart Guide

### 1. Prerequisites & Installation

Ensure **Python 3.10+** and **Node.js 18+** are installed on your machine.

Install backend and frontend dependencies:

```powershell
pip install -r requirements.txt
cd frontend
npm install --legacy-peer-deps
cd ..
```

### 2. Run Application (Single Command)

Start both backend API (FastAPI) and frontend (React/Vite) simultaneously with one command:

```powershell
python run_app.py
```

Open your browser to:
- **Web Application (UI):** [http://localhost:5173/](http://localhost:5173/)
- **Interactive Backend API (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Dataset Schema & Preprocessing Specification

The cohort dataset is stored at [`data/masld_patient_cohort.csv`](file:///c:/Users/pirap/Desktop/Nethmi/data/masld_patient_cohort.csv). A template is available at [`data/dataset_template.csv`](file:///c:/Users/pirap/Desktop/Nethmi/data/dataset_template.csv).

| Feature | Type | Unit / Range | Clinical Description |
| :--- | :--- | :--- | :--- |
| `patient_id` | String | e.g. `UK-MASLD-0001` | De-identified patient study identifier |
| `age` | Integer | 30–80 years | Chronological age |
| `sex` | Categorical | `Male`, `Female` | Biological sex |
| `bmi` | Float | $\text{kg/m}^2$ | Body Mass Index (Asian cutoff: $\ge 23$ overweight, $\ge 27.5$ obese) |
| `diabetes` | Categorical | `Yes`, `No` | Type 2 Diabetes Mellitus diagnosis |
| `hypertension` | Categorical | `Yes`, `No` | Clinical hypertension diagnosis |
| `smoker` | Categorical | `Yes`, `No` | Current tobacco smoking status |
| `systolic_bp` | Integer | $\text{mmHg}$ | Systolic Blood Pressure |
| `diastolic_bp` | Integer | $\text{mmHg}$ | Diastolic Blood Pressure |
| `fasting_glucose` | Integer | $\text{mg/dL}$ | Fasting Blood Sugar |
| `total_cholesterol` | Integer | $\text{mg/dL}$ | Total Serum Cholesterol |
| `ldl` | Integer | $\text{mg/dL}$ | Low-Density Lipoprotein Cholesterol |
| `hdl` | Integer | $\text{mg/dL}$ | High-Density Lipoprotein Cholesterol |
| `triglycerides` | Integer | $\text{mg/dL}$ | Serum Triglycerides |
| `ast` | Integer | $\text{U/L}$ | Aspartate Aminotransferase |
| `alt` | Integer | $\text{U/L}$ | Alanine Aminotransferase |
| `platelet_count` | Integer | $\times 10^9/\text{L}$ | Thrombocyte / Platelet count |
| `fibrosis_stage` | Integer | 0 to 4 | Liver Fibrosis Stage ($F_0$: None, $F_1$: Mild, $F_2$: Moderate, $F_3$: Severe, $F_4$: Cirrhosis) |
| `fib_4_score` | Float | Calculated | Non-invasive FIB-4 index: $(\text{Age} \times \text{AST}) / (\text{Platelets} \times \sqrt{\text{ALT}})$ |
| `slcvd_score` | Float | % | Sri Lankan Cardiovascular Risk Score (10-year risk) |
| `who_cvd_risk` | Float | % | WHO Cardiovascular Disease Risk Chart estimate (%) |
| `cvd_event` | Categorical | `Yes`, `No` | Primary clinical outcome: 10-year incident cardiovascular event (Target variable) |

### Missing Value Preprocessing Protocol:
- **Continuous Parameters:** Median imputation (`SimpleImputer(strategy='median')`), robust against skewed distributions.
- **Categorical Parameters:** Most frequent imputation (`SimpleImputer(strategy='most_frequent')`).
- **Data Leakage Prevention:** Preprocessing transformers are fitted strictly on the training partition and transformed on test/inference data.

---

## Project Structure

```
Nethmi/
├── run_app.py                              # One-click dual launcher (Backend + Frontend)
├── MASLD_CVD_Risk_Prediction_and_XAI.ipynb # Complete research notebook with 34 cells
├── CardioLiver_AI_Documentation.docx       # Complete project documentation (Word format)
├── CardioLiver_AI_Documentation.pdf        # Complete project documentation (PDF format)
├── requirements.txt                        # Python dependencies
├── README.md                               # Unified project documentation
├── backend/
│   ├── main.py                             # FastAPI backend application & endpoints
│   └── models/                             # Serialized champion models and metadata
├── frontend/                               # React 19 + Vite + Tailwind/CSS frontend
│   ├── src/                                # React UI components and clinical views
│   └── package.json                        # Frontend dependencies
└── data/
    ├── masld_patient_cohort.csv            # 1,200 patient clinical registry dataset
    └── dataset_template.csv                # Reusable upload CSV template
```

---

## Scientific Novelty

> *First AI-assisted, explainable clinical decision support framework in a Sri Lankan MASLD cohort that evaluates whether liver fibrosis severity provides incremental value for cardiovascular risk prediction beyond existing risk scores and delivers patient-specific, interpretable risk assessment for better clinical decision-making.*
