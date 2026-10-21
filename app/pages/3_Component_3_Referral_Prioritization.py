"""C3 PP1 dashboard: replay and run equal-budget retrospective comparisons."""
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/interim/nhanes_2017_2020'
BASE = DATA / 'baseline_kpa_8_seed_42_n_400'
DESCRIPTIONS = {
    'Random': 'Selects participants randomly as a reference strategy.',
    'Risk-first': 'Selects the highest predicted elevated-stiffness scores.',
    'Uncertainty': 'Selects scores closest to 0.5, where the model is least decisive.',
    'Diversity': 'Selects feature profiles farthest from labelled and already selected participants.',
    'Hybrid': 'Combines uncertainty and diversity equally, updating diversity after each selection.',
}

st.title('Scarce Resource Referral Prioritization')
st.caption('Component 3 · PP1 research prototype · NHANES 2017–March 2020')
st.info('Kelaniya University dataset pending. NHANES demonstrates the workflow; local validation is planned after receipt of the dataset.')
st.write('Compare how a limited scan budget affects case detection and model improvement among adults aged 18 and above.')

if not (BASE / 'baseline_report.json').exists():
    st.warning('Prepare the NHANES data and train the initial model first.')
    st.code('python scripts/prepare_nhanes.py\npython scripts/train_c3_baseline.py --threshold-kpa 8')
    st.stop()
try:
    baseline = json.loads((BASE / 'baseline_report.json').read_text())
except (OSError, ValueError) as exc:
    st.error(f'Cannot read the baseline report: {exc}')
    st.stop()

counts = baseline['split_counts']
a, b, c, d = st.columns(4)
a.metric('Initial labelled participants', f"{counts['initial']:,}")
b.metric('Simulated waiting pool', f"{counts['pool']:,}")
c.metric('Validation participants', f"{counts['validation']:,}")
d.metric('Initial validation AUROC', f"{baseline['validation_xgboost']['auroc']:.4f}")
st.caption('Demonstration target: observed stiffness ≥8 kPa. This provisional endpoint is not a confirmed fibrosis diagnosis. Scores are uncalibrated; this prototype is not for clinical decisions.')

left, right = st.columns(2)
with left:
    budget = st.selectbox('Available scans per strategy', [10, 20, 50], index=1)
with right:
    strategy = st.selectbox('Strategy to inspect', list(DESCRIPTIONS), index=1)
st.write(DESCRIPTIONS[strategy])
folder = BASE / f'comparison_budget_{budget}_alpha_0.5'
if st.button('Run comparison for this budget', type='primary'):
    with st.spinner('Selecting participants and retraining five independent models…'):
        try:
            process = subprocess.run(
                [sys.executable, str(ROOT / 'scripts/compare_c3_strategies.py'), '--budget', str(budget)],
                cwd=ROOT, capture_output=True, text=True, timeout=180,
            )
            if process.returncode:
                st.error('The comparison did not complete.')
                st.code(process.stderr or process.stdout)
                st.stop()
            st.success('Comparison completed. Each strategy used the same starting model and scan budget.')
        except (OSError, subprocess.TimeoutExpired) as exc:
            st.error(f'Could not run comparison: {exc}')
            st.stop()

if not (folder / 'comparison_report.json').exists():
    st.info('Click Run comparison for this budget to generate results.')
    st.stop()
try:
    report = json.loads((folder / 'comparison_report.json').read_text())
    results = pd.DataFrame(report['results'])
    selected = pd.read_csv(folder / 'selected_patients.csv')
    predictors = pd.read_csv(DATA / 'candidate_predictors.csv')
    if report['budget'] != budget or report['seed'] != baseline['seed']:
        raise ValueError('Saved results do not match the chosen configuration. Run the comparison again.')
except (OSError, ValueError, KeyError) as exc:
    st.error(f'Cannot load comparison outputs: {exc}')
    st.stop()

st.subheader('Selected participants')
st.caption('Retrospective replay: selections were frozen before outcomes were revealed to the simulation. These are NHANES participant IDs, not clinic appointments.')
queue = selected.loc[selected.strategy.eq(strategy)].sort_values('rank')
queue = queue.merge(predictors, on='SEQN', how='left', validate='one_to_one')
queue['Sex'] = queue.RIAGENDR.map({1: 'Male', 2: 'Female'}).fillna('Unknown')
queue = queue.rename(columns={'rank': 'Selection order', 'SEQN': 'Participant ID',
    'RIDAGEYR': 'Age', 'BMXBMI': 'BMI', 'diabetes_status': 'Diabetes status',
    'pre_scan_score': 'Pre-scan model score', 'revealed_target': 'Recorded outcome'})
visible = ['Selection order', 'Participant ID', 'Age', 'Sex', 'BMI', 'Diabetes status', 'Pre-scan model score']
reveal = st.checkbox('Reveal recorded scan outcomes', key=f'reveal_{budget}_{strategy}')
if reveal:
    queue['Recorded outcome'] = queue['Recorded outcome'].map({1: 'Elevated stiffness', 0: 'Below threshold'})
    visible.append('Recorded outcome')
st.dataframe(queue[visible], hide_index=True, width='stretch')
st.download_button('Download displayed selection', queue[visible].to_csv(index=False),
                   file_name=f'c3_{strategy.lower()}_{budget}_selections.csv', mime='text/csv')

st.subheader('Equal-budget comparison')
st.caption('All strategies start with the same 400 labelled participants and waiting pool. Each independently reveals its chosen outcomes and retrains from scratch.')
left, right = st.columns(2)
with left:
    st.write('**Elevated-stiffness cases found** — higher is better')
    st.bar_chart(results.set_index('strategy')[['positive_scans_found']])
with right:
    st.write('**Change in validation AUROC** — higher is better')
    st.bar_chart(results.set_index('strategy')[['auroc_change']])
labels = {'strategy': 'Strategy', 'scans': 'Scans', 'positive_scans_found': 'Positive scans',
          'precision_at_k': 'Precision at K', 'validation_auroc': 'Validation AUROC',
          'auroc_change': 'AUROC change', 'validation_average_precision': 'Average precision',
          'validation_brier': 'Brier score'}
st.dataframe(results.rename(columns=labels).round(4), hide_index=True, width='stretch')
st.download_button('Download comparison', results.to_csv(index=False),
                   file_name=f'c3_comparison_{budget}.csv', mime='text/csv')
most_cases = results.loc[results.positive_scans_found.eq(results.positive_scans_found.max()), 'strategy'].tolist()
best_auc = results.loc[results.validation_auroc.eq(results.validation_auroc.max()), 'strategy'].tolist()
st.write(f"Most positive scans in this run: **{', '.join(most_cases)}**. Highest validation AUROC after retraining: **{', '.join(best_auc)}**.")
st.warning('One batch and one data split: these results do not establish an overall winner. Repeated trials, sequential rounds and uncertainty estimates are still needed.')

with st.expander('What is implemented and what comes next?'):
    st.markdown('**Implemented:** all-adult cohort preparation, XGBoost baseline, five selection strategies, equal-budget comparison, independent retraining and this results dashboard.')
    st.markdown('**Next:** repeated trials and learning curves, equity-aware constraints, XGBoost-specific research extensions, and a statistical stopping rule. These are not implemented in this version.')
    st.write(f"Final test: {counts['test']:,} participants reserved; not evaluated by this comparison.")
    st.write('NHANES is a retrospective public survey sample, not a local referral queue. Results are unweighted demonstration metrics.')
