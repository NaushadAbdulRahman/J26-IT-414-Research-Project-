# C3 guided demo and evidence

Run the shared application from the repository root:

```bash
python -m streamlit run app/Home.py
```

Open **Component 3** in the sidebar. Explanations are always visible and information sections open expanded. The four tabs separate the workflow, selections, results and verification.

If a comparison already exists, a banner identifies the saved budget and seed. Changing the budget or strategy loads saved outputs without training. **Re-run comparison for this budget** recomputes selections and independently retrains all five models. A budget with no saved comparison instead shows **Run comparison for this budget**. Repeating the same configuration reproduces one batch from the same starting point.

## How the code is organized

These are supporting modules for one research component:

| File | Responsibility |
| --- | --- |
| `app/pages/3_Component_3_Referral_Prioritization.py` | The Streamlit page, controls and displayed results |
| `app/c3_verification.py` | Strategy descriptions and reusable functions that check saved outputs |
| `scripts/train_c3_baseline.py` | Data splits and initial XGBoost training |
| `scripts/compare_c3_strategies.py` | Patient selection, independent retraining and output exports |
| `tests/test_c3_verification.py` | Checks that verification accepts valid evidence and rejects incorrect evidence |

Viva explanation: "This is one component organized into modules. The page handles the interface, the scripts run the experiment, and the verification helper checks the saved results."

## A short demonstration

1. **Overview:** explain the clinic scenario and the two objectives: immediate detection and future model improvement. Point out the initial labelled, waiting and separate validation groups. Follow Train, Hide, Select, Reveal, Retrain, Evaluate.
2. Set **Scan budget per strategy** to **20** and **Strategy to inspect** to **Risk-first**.
3. **Selection demo:** run all five strategies. Every strategy gets its own simulation from the same initial model. The run selects, reveals and retrains. **Show recorded outcomes** only changes what is displayed afterwards.
4. **Results explained:** risk-first finds 12/20 positives, so precision at 20 is 60%. Inspect **Diversity** to show AUROC changing from 0.7337 to 0.7529 (+0.0192). These are the current seed-42 demonstration values; displayed cards always use the saved outputs for the chosen budget.
5. **Verification & evidence:** press **Verify the current saved outputs**. Show the checked files and calculations. Export the verification table or validation predictions if needed.
6. End with the distinction between completed work and planned contributions. Repeated trials, sequential rounds, equity constraints, an XGBoost-informed acquisition extension, a plateau rule and local evaluation remain planned.

## What verification establishes

The page does not assume a green verification state. Pressing the button checks:

- Unique partition IDs and recorded split counts.
- The same budget for all five strategies, unique selected IDs, waiting-pool membership, valid scores and selection ranks.
- Selected target labels against recorded scan stiffness, followed by recomputed positive counts and precision.
- Predictor and outcome SHA-256 fingerprints against those recorded at baseline training.
- AUROC, average precision, Brier score and AUROC changes recalculated from saved validation predictions, using the exact validation IDs and recorded targets. Reserved final test IDs must not appear.

Checks report **Passed**, **Failed** or **Not available**. A missing prediction export never counts as a pass. The result is a verification snapshot and automatically becomes stale when the relevant file timestamps/sizes or budget change. Press the button again after changing files.

This verifies file and numerical consistency. It does not establish clinical validity, absence of every possible modelling error, statistical superiority, or transfer to Sri Lankan patients.

## Existing results

Older result folders can still be displayed if their metadata and basic numeric checks pass. To enable validation-metric recomputation, rerun a budget once using the updated comparison script:

```bash
python scripts/compare_c3_strategies.py --budget 20
```

The script now writes `validation_predictions.csv` (initial model and five retrained models on validation patients). It records input hashes, validation size and initial label size in the comparison report. Selection policies, the XGBoost settings and evaluation split remain the same.

Generated CSVs and patient-level records stay ignored by Git. Only code and documentation belong in the commits.

## Checks performed during implementation

- 13 focused tests cover valid evidence and rejection of wrong precision, duplicate selections, changed inputs, missing predictions, mismatched metrics and validation/test mix-ups.
- Streamlit AppTest exercised all five strategy views at budgets 10, 20 and 50, the run button, recorded-outcome toggle and verification. The refined interface also checks permanently visible explanations and expanded information sections.
- The updated 20-scan experiment reproduced the current PP1 demonstration results.

Run the focused checks from the repository root:

```bash
python -m unittest discover -s tests -p test_c3_verification.py -v
```

## Suggested commits

First commit the evidence export, verification helper and meaningful checks:

```bash
git add scripts/compare_c3_strategies.py app/c3_verification.py tests/test_c3_verification.py
git commit -m "feat(c3): export validation predictions and verify experiment outputs"
```

Then commit the guided interface and demo instructions:

```bash
git add app/pages/3_Component_3_Referral_Prioritization.py docs/c3-demo-guide.md
git commit -m "feat(c3): add guided Streamlit demo and metric explanation cards"
```

Push your feature branch and open a pull request into `main` after inspecting the page locally.
