# C3 — Research plan and PP1 status

## Aim
Evaluate patient selection under limited scan capacity, separating immediate detection of elevated stiffness from future model improvement.

## Implemented at PP1
- All-adult NHANES cohort preparation and separated pre-scan predictors/outcomes.
- Independent XGBoost baseline, initial labelled group, hidden-outcome waiting pool, validation group and reserved final test.
- Random, risk-first, uncertainty, diversity and hybrid selection at budgets 10, 20 and 50.
- One independent acquisition/retraining batch for each strategy, with held-out validation metrics.
- Shared Streamlit dashboard, evidence exports and numerical verification.

The provisional demonstration endpoint is recorded liver stiffness >=8 kPa. Scan measurements are excluded from predictors. NHANES demonstrates the workflow while the Kelaniya dataset is pending; this is not local clinical validation.

## Next research stages
1. Sequential acquisition rounds and repeated matched trials across seeds.
2. Learning curves and budget-sensitivity analysis with uncertainty estimates.
3. XGBoost-informed acquisition extension, supported by literature and ablation comparisons.
4. Equity constraints with explicit subgroup-coverage definitions and measured trade-offs.
5. Statistical learning-plateau stopping rule, validated over repeated runs.
6. Local data adaptation and evaluation after dataset receipt and schema agreement.

Hidden pool outcomes must not influence selection. Validation data must remain outside training, and final test data must remain reserved until research choices are fixed. The plateau concerns additional labels for model learning, not whether patients should receive clinically indicated scans.

These extensions are planned contributions; their novelty and benefit require literature support and experimental evidence.
