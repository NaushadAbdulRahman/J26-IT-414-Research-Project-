# C3 — Scarce Resource Referral Prioritization

## Aim
Select patients for limited scan appointments and evaluate
selection strategies under the same scan budget.

## Current implementation
- C3 page integrated into the shared Streamlit application.
- Configurable scan budget.
- Descriptions of the five planned selection strategies.
- Application successfully launched locally.

Model training and patient selection are not yet connected
to this shared application.

## Planned workflow
1. Load and validate pre-scan patient features.
2. Separate initial labelled, waiting-pool and test data.
3. Fit preprocessing and XGBoost on labelled patients.
4. Select patients within the scan budget.
5. Reveal only the selected patients' scan outcomes.
6. Retrain and compare results over multiple rounds.

## Baseline strategies
- Risk-first
- Uncertainty
- Diversity
- Uncertainty–diversity hybrid
- Random selection

## Evaluation
Primary: positive scan outcomes detected at equal scan expenditure.
Secondary: precision per batch, held-out AUROC and average precision.
Repeat matched experiments to examine variability.

Hidden waiting-pool outcomes must not influence selection.
Test data must not influence preprocessing, training or tuning.

## Data plan
Use synthetic data to verify the workflow.
Proposed public dataset: NHANES 2017–March 2020.
Clinical dataset delivery is pending.

The final cohort definition, scan target and threshold remain
to be agreed. Scan-derived outcomes must not be predictor inputs.

## Later research work
Investigate an XGBoost-aware acquisition method, equity constraints,
budget sensitivity and statistical stopping.

These extensions are not implemented or established as novel yet.