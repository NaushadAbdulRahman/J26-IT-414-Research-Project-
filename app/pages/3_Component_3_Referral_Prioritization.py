"""C3 guided research demo: explain, select, compare and verify one scan batch."""
from html import escape
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'app'))
sys.path.append(str(ROOT / 'component-1-fibrosis-screening' / 'src'))
from c3_verification import STRATEGIES, threshold_from_endpoint, validate_report, verify_outputs
import ui as look

DATA = ROOT / 'data/interim/nhanes_2017_2020'
BASE = DATA / 'baseline_kpa_8_seed_42_n_400'

st.set_page_config(page_title='C3 · Scan Referral Prioritization', layout='wide')
look.style(own_sidebar=False)
st.markdown('''<style>
.c3-card {border:1px solid var(--c1-rule);border-radius:10px;padding:20px 22px;
 background:var(--c1-field);margin:0 0 14px;min-height:150px}
.c3-card .label {font-size:14px;color:var(--c1-muted);margin-bottom:8px}
.c3-card .value {font-size:32px;font-weight:700;line-height:1.15;color:var(--c1-ink)}
.c3-card .detail {font-size:15px;color:var(--c1-muted);margin-top:10px;line-height:1.45}
.c3-explanation {border-left:3px solid var(--c1-accent);padding:10px 16px;margin:10px 0 20px;
 background:var(--c1-accent-soft);color:var(--c1-ink);font-size:16px;line-height:1.5}
.c3-step {border-top:2px solid var(--c1-accent);padding:12px 0;margin-bottom:12px}
.c3-step b {font-size:17px}.c3-step p {font-size:15px;color:var(--c1-muted);margin-top:5px}
@media(max-width:700px){.c3-card{min-height:0}.c3-card .value{font-size:26px}}
</style>''', unsafe_allow_html=True)


def card(label, value, detail):
    st.markdown(f'<div class="c3-card"><div class="label">{escape(label)}</div>'
                f'<div class="value">{escape(str(value))}</div>'
                f'<div class="detail">{escape(detail)}</div></div>', unsafe_allow_html=True)


def explanation(text):
    st.markdown(f'<div class="c3-explanation">{escape(text)}</div>', unsafe_allow_html=True)


look.title('Scarce Resource Referral Prioritization',
           'Compare who to scan within a limited budget, measuring immediate detection and model improvement.')
st.caption('Component 3 · Aaqib A.R · IT23228276 · All adults aged 18+')
st.info('Kelaniya dataset pending. NHANES (US, 2017–March 2020) supports this demonstration. '
        'Local evaluation is planned after receipt.')

if not (BASE / 'baseline_report.json').exists():
    st.warning('The initial model is not ready in this checkout.')
    with st.expander('Prepare the demonstration data and model', expanded=True):
        st.code('python scripts/download_nhanes.py\npython scripts/prepare_nhanes.py\n'
                'python scripts/train_c3_baseline.py --threshold-kpa 8')
    st.stop()
try:
    baseline = json.loads((BASE / 'baseline_report.json').read_text())
    counts = baseline['split_counts']
    threshold = threshold_from_endpoint(baseline['endpoint'])
except (OSError, ValueError, KeyError) as exc:
    st.error(f'Cannot read the initial model report: {exc}')
    st.stop()

for col, (label, value, detail) in zip(st.columns(4), [
    ('Initial labelled patients', f"{counts['initial']:,}", 'Known outcomes for the starting model'),
    ('Waiting pool', f"{counts['pool']:,}", 'Outcomes hidden from selection strategies'),
    ('Separate validation group', f"{counts['validation']:,}", 'Checks performance after retraining'),
    ('Starting validation AUROC', f"{baseline['validation_xgboost']['auroc']:.4f}", 'Ranking ability before selection'),
]):
    with col:
        card(label, value, detail)

controls = st.columns([1, 1, 2])
with controls[0]:
    budget = st.selectbox('Scan budget per strategy', [10, 20, 50], index=1, key='c3_budget',
                          help='Each strategy receives this same number of scans in its own simulation.')
with controls[1]:
    strategy = st.selectbox('Strategy to inspect', list(STRATEGIES), index=1, key='c3_strategy')
with controls[2]:
    st.caption('Selection rule')
    st.write(f'**{STRATEGIES[strategy][0]}** — {STRATEGIES[strategy][1]}')

folder = BASE / f'comparison_budget_{budget}_alpha_0.5'
report = results = selected = predictors = None
load_error = None
if (folder / 'comparison_report.json').exists():
    try:
        report = json.loads((folder / 'comparison_report.json').read_text())
        results = validate_report(report, baseline, budget)
        selected = pd.read_csv(folder / 'selected_patients.csv')
        predictors = pd.read_csv(DATA / 'candidate_predictors.csv')
        if predictors.SEQN.isna().any() or not predictors.SEQN.is_unique:
            raise ValueError('Predictor IDs are missing or duplicated.')
        if (set(selected.strategy) != set(STRATEGIES)
                or selected.groupby('strategy').size().ne(budget).any()
                or selected.duplicated(['strategy', 'SEQN']).any()
                or selected.SEQN.isna().any()):
            raise ValueError('Saved selections are incomplete or duplicated. Run the comparison again.')
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        load_error = str(exc)
        report = results = selected = predictors = None

overview, demo, comparison, evidence = st.tabs([
    '1 · Overview', '2 · Selection demo', '3 · Results explained', '4 · Verification & evidence'])

with overview:
    left, right = st.columns(2, gap='large')
    with left:
        st.subheader('The problem')
        st.write('A clinic has more eligible patients than available scan appointments. '
                 'For example, 100 waiting patients and only 20 scan slots.')
        st.write('Compare ways to allocate those slots for two purposes: **find elevated-stiffness '
                 'cases now** and **obtain useful labels for future predictions**.')
    with right:
        st.subheader('The prediction model')
        st.write('**XGBoost = Extreme Gradient Boosting.** It combines many decision trees. '
                 'Later trees help reduce errors in the combined prediction.')
        st.write('Inputs: **age, sex, BMI, AST, ALT, platelets and diabetes status**. '
                 'Output: a score for elevated stiffness, using pre-scan information.')
        st.caption(f'Provisional target: scan stiffness ≥{threshold:g} kPa. Scores have not been '
                   'calibrated. This target is not a confirmed fibrosis diagnosis.')
    st.subheader('One simulated scan batch')
    steps = [
        ('01 · Train', f"Learn from {counts['initial']:,} patients with known scan outcomes."),
        ('02 · Hide', f"Keep {counts['pool']:,} waiting-pool outcomes unavailable to selectors."),
        ('03 · Select', f'Each strategy independently chooses {budget} patients using pre-scan information.'),
        ('04 · Reveal', 'Read only the chosen patients’ existing scan outcomes from NHANES.'),
        ('05 · Retrain', f"Fit a fresh model using {counts['initial']:,} + {budget} labelled patients."),
        ('06 · Evaluate', f"Compare predictions on the same {counts['validation']:,} validation patients."),
    ]
    for offset in [0, 3]:
        for col, (title, detail) in zip(st.columns(3), steps[offset:offset + 3]):
            with col:
                st.markdown(f'<div class="c3-step"><b>{escape(title)}</b><p>{escape(detail)}</p></div>',
                            unsafe_allow_html=True)
    explanation('XGBoost predicts scores. The strategy chooses patients. Revealed outcomes provide new '
        'training labels. A separate validation group measures whether learning helped.')
    with st.expander('Data groups and terminology', expanded=True):
        st.dataframe(pd.DataFrame([
            ['Initial training', counts['initial'], 'Starting labelled examples'],
            ['Waiting pool', counts['pool'], 'Candidates for selection'],
            ['Validation', counts['validation'], 'Development evaluation'],
            ['Reserved final test', counts['test'], 'Excluded from this comparison'],
        ], columns=['Group', 'Patients', 'Purpose']), hide_index=True, width='stretch')
        st.write('**NHANES:** National Health and Nutrition Examination Survey. '
                 '**BMI:** Body Mass Index. **AST/ALT:** blood-test liver enzymes. '
                 '**kPa:** kilopascals, the stiffness measurement unit. '
                 '**Label:** the recorded target outcome used for training.')

with demo:
    st.subheader('Choose a policy, then inspect its patients')
    st.write(f'**{strategy}:** {STRATEGIES[strategy][1]} {STRATEGIES[strategy][2]}')
    with st.expander('All five selection rules', expanded=True):
        for name, (summary, rule, purpose) in STRATEGIES.items():
            st.markdown(f'**{name} · {summary}**  \n{rule} {purpose}')
        st.caption('Diversity considers feature profiles. Equity requires an explicit subgroup '
                   'coverage rule and remains planned. Hybrid combines only uncertainty and diversity.')
    if st.button('Run all five strategies at this budget', type='primary', key='c3_run'):
        with st.spinner('Selecting one batch and independently retraining five models…'):
            try:
                process = subprocess.run(
                    [sys.executable, str(ROOT / 'scripts/compare_c3_strategies.py'), '--budget', str(budget)],
                    cwd=ROOT, capture_output=True, text=True, timeout=180)
                if process.returncode:
                    st.error('The comparison did not complete.')
                    st.code(process.stderr or process.stdout)
                else:
                    st.session_state['c3_completed'] = budget
                    st.rerun()
            except (OSError, subprocess.TimeoutExpired) as exc:
                st.error(f'Could not run the comparison: {exc}')
    if st.session_state.pop('c3_completed', None) == budget:
        st.success('Comparison saved. Same initial model, waiting pool and budget for every strategy.')
    st.caption('One batch from the same starting point. Repeating a run reproduces the experiment, '
               'rather than advancing to another month. Retraining happens during the run.')
    if load_error:
        st.error(f'Cannot use saved results: {load_error}')
    elif report is None:
        st.info('Run the comparison to create selections and results for this budget.')
    else:
        queue = selected.loc[selected.strategy.eq(strategy)].sort_values('rank')
        queue = queue.merge(predictors, on='SEQN', how='left', validate='one_to_one')
        queue['Sex'] = queue.RIAGENDR.map({1: 'Male', 2: 'Female'}).fillna('Unknown')
        queue = queue.rename(columns={'rank': 'Selection order', 'SEQN': 'Participant ID',
            'RIDAGEYR': 'Age', 'BMXBMI': 'BMI', 'diabetes_status': 'Diabetes status',
            'pre_scan_score': 'Pre-scan score', 'revealed_target': 'Recorded outcome'})
        st.subheader(f'{budget} selected patients · {strategy}')
        show = st.checkbox('Show recorded outcomes', key=f'c3_show_{budget}_{strategy}',
                           help='Displays outcomes from the completed simulation. It does not trigger scans or retraining.')
        columns = ['Selection order', 'Participant ID', 'Age', 'Sex', 'BMI',
                   'Diabetes status', 'Pre-scan score']
        if show:
            queue['Recorded outcome'] = queue['Recorded outcome'].map(
                {1: 'Elevated stiffness', 0: 'Below threshold'})
            columns.append('Recorded outcome')
        st.dataframe(queue[columns], hide_index=True, width='stretch', column_config={
            'Pre-scan score': st.column_config.NumberColumn(format='%.3f',
                help='XGBoost score for the provisional target. Not a calibrated clinical probability.'),
            'Selection order': st.column_config.NumberColumn(
                help='Policy selection order, rather than a validated clinical urgency ranking.')})
        st.caption('NHANES participant IDs. Selectors never receive scan outcomes. '
                   'Showing outcomes here only reveals the saved simulation for inspection.')
        st.download_button('Download displayed selection', queue[columns].to_csv(index=False),
                           file_name=f'c3_{strategy.lower()}_{budget}_selections.csv', mime='text/csv')
        explanation(f'This policy chose {budget} patients from the same waiting pool. We can inspect '
            'their recorded outcomes and compare the choices with other policies.')

with comparison:
    if report is None:
        st.info('Create valid results in Selection demo to see explanations for this budget.')
    else:
        row = results.set_index('strategy').loc[strategy]
        before = report['baseline_validation']['auroc']
        positive = int(row.positive_scans_found)
        a, b, c = st.columns(3)
        with a:
            card(f'{strategy} · Cases detected', f'{positive} / {budget}',
                 f'Selected patients whose recorded stiffness meets the ≥{threshold:g} kPa target.')
        with b:
            card('Precision at K', f'{row.precision_at_k:.0%}',
                 f'{positive} positives ÷ {budget} selected = {row.precision_at_k:.2f}. K is the scan budget.')
        with c:
            card('Validation AUROC after retraining', f'{row.validation_auroc:.4f}',
                 f'Before {before:.4f}. Change {row.auroc_change:+.4f}. AUROC measures ranking ability.')
        most_cases = results.loc[results.positive_scans_found.eq(results.positive_scans_found.max())]
        best_auc = results.loc[results.validation_auroc.eq(results.validation_auroc.max())]
        st.subheader('Detection and learning can favour different policies')
        st.write(f"**Most positives in this run:** {', '.join(most_cases.strategy)} "
                 f"({int(most_cases.iloc[0].positive_scans_found)} / {budget}). "
                 f"**Highest validation AUROC:** {', '.join(best_auc.strategy)} "
                 f"({best_auc.iloc[0].validation_auroc:.4f}).")
        a, b = st.columns(2)
        with a:
            st.write('**Positive cases found**')
            st.caption('Immediate detection. More positives within the same budget is better.')
            st.bar_chart(results.set_index('strategy')[['positive_scans_found']], color='#268477')
        with b:
            st.write('**Change in validation AUROC**')
            st.caption('Above zero improves on the starting model; below zero declines.')
            st.bar_chart(results.set_index('strategy')[['auroc_change']], color='#6487A1')
        explanation(f'For {strategy}, precision is {positive} divided by {budget}, or {row.precision_at_k:.0%}. '
            f'AUROC changes from {before:.4f} to {row.validation_auroc:.4f}. '
            'The preferred policy depends on the objective. One run does not establish an overall winner.')
        with st.expander('Metric explanations', expanded=True):
            a, b = st.columns(2)
            with a:
                card('AUROC · Area Under the Receiver Operating Characteristic Curve',
                     'Ranking ability', 'How often a randomly chosen positive ranks above a randomly '
                     'chosen negative. Higher is better. It is not classification accuracy.')
                card('Average precision (AP)', f'{row.validation_average_precision:.4f}',
                     'Summarizes precision across recall levels. Higher is better. Covers validation '
                     'patients, rather than just the selected batch.')
            with b:
                card('Brier score', f'{row.validation_brier:.4f}',
                     'Average squared difference between scores and actual 0/1 outcomes. Lower is better. '
                     'It is not a misclassification percentage or proof of calibration.')
                card('AUROC change', f'{row.auroc_change:+.4f}',
                     f'{row.validation_auroc:.4f} after − {before:.4f} before. '
                     'A change on the AUROC scale, rather than a percentage increase in accuracy.')
        with st.expander('Full comparison table and export', expanded=True):
            labels = {'strategy': 'Strategy', 'scans': 'Scans', 'positive_scans_found': 'Positive scans',
                      'precision_at_k': 'Precision at K', 'validation_auroc': 'Validation AUROC',
                      'auroc_change': 'AUROC change', 'validation_average_precision': 'Average precision',
                      'validation_brier': 'Brier score'}
            st.dataframe(results.rename(columns=labels).round(4), hide_index=True, width='stretch')
            st.download_button('Download comparison', results.to_csv(index=False),
                               file_name=f'c3_comparison_{budget}.csv', mime='text/csv')
        st.caption('Exploratory results from one batch and one split. Repeated trials and sequential '
                   'learning curves are planned. NHANES metrics are unweighted demonstration results.')

with evidence:
    st.subheader('How the comparison is verified')
    st.write('**Equal conditions:** same initial training group, waiting pool, budget, XGBoost settings '
             'and validation group. Each policy retrains independently.')
    st.write('All five selections are frozen before reading waiting-pool outcomes. '
             f"The reserved final test group ({counts['test']:,} patients) is excluded from this experiment.")
    if report is not None:
        evidence_files = [BASE / 'baseline_report.json', BASE / 'split_assignments.csv',
                          DATA / 'candidate_predictors.csv', DATA / 'scan_outcomes.csv',
                          folder / 'comparison_report.json', folder / 'selected_patients.csv',
                          folder / 'validation_predictions.csv']
        signature = (budget, tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size)
                                   if path.exists() else (str(path), None, None)
                                   for path in evidence_files))
        if st.button('Verify the current saved outputs', key='c3_verify', type='primary'):
            try:
                checks = verify_outputs(BASE, baseline, report, selected)
                st.session_state['c3_checks'] = (signature, checks)
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                st.error(f'Cannot verify these outputs: {exc}')
        saved = st.session_state.get('c3_checks')
        if saved and saved[0] == signature:
            checks = saved[1]
            if checks.Status.eq('Failed').any():
                st.error('Some checks failed. Inspect the evidence and rerun preparation or comparison as indicated.')
            elif checks.Status.eq('Not available').any():
                st.warning('Available checks passed. Rerun the comparison to export missing validation predictions.')
            else:
                st.success('All five output checks passed for this saved run.')
            st.dataframe(checks, hide_index=True, width='stretch')
            st.caption('Verification snapshot from the button click. Verify again after changing files.')
            st.download_button('Download verification evidence', checks.to_csv(index=False),
                               file_name=f'c3_verification_{budget}.csv', mime='text/csv')
        else:
            st.info('Run verification to check files and recalculate metrics. No result is assumed.')
        if (folder / 'validation_predictions.csv').exists():
            st.download_button('Download validation predictions',
                               (folder / 'validation_predictions.csv').read_bytes(),
                               file_name=f'c3_validation_predictions_{budget}.csv', mime='text/csv')
    else:
        st.info('Run a comparison first to generate outputs that can be checked.')
    explanation('Verification checks selected IDs and budget, compares recorded scan outcomes, '
        'recalculates precision and recomputes validation metrics from saved predictions. '
        'This checks numerical consistency. '
        'Repeated experiments and local evaluation will assess reliability and transfer.')
    with st.expander('Completed work and planned research', expanded=True):
        left, right = st.columns(2)
        with left:
            st.markdown('**Implemented**\n- All-adult cohort preparation\n- XGBoost baseline\n'
                        '- Five selection strategies\n- Equal-budget independent retraining\n'
                        '- Results dashboard and output verification')
        with right:
            st.markdown('**Planned**\n- Repeated trials and sequential learning curves\n'
                        '- XGBoost-informed selection extension\n- Explicit equity constraints\n'
                        '- Statistical learning-plateau rule\n- Kelaniya dataset evaluation')
        st.caption('A learning plateau concerns diminishing model improvement from additional labels. '
                   'It does not decide when clinical scans should stop.')
    with st.expander('Run details', expanded=True):
        st.json({'endpoint': baseline['endpoint'], 'seed': baseline['seed'],
                 'scan_budget': budget, 'hybrid_uncertainty_weight': 0.5,
                 'training_versions': baseline.get('versions', {}),
                 'final_test_evaluated': False, 'equity_constraint_implemented': False})

st.divider()
st.caption('Research prototype using retrospective NHANES records. Current results do not establish '
           'clinical urgency, a confirmed fibrosis diagnosis, or performance in Sri Lankan clinics.')
