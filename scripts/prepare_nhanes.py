"""Join NHANES source files and audit candidate cohorts; no model training.

Run: python scripts/prepare_nhanes.py
Outputs stay in git-ignored data/interim/nhanes_2017_2020.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/raw/nhanes_2017_2020'
OUTPUT = ROOT / 'data/interim/nhanes_2017_2020'
COLUMNS = {
    'P_DEMO': ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'WTMECPRP', 'SDMVPSU', 'SDMVSTRA'],
    'P_BMX': ['SEQN', 'BMXBMI'],
    'P_BIOPRO': ['SEQN', 'LBXSASSI', 'LBXSATSI'],
    'P_CBC': ['SEQN', 'LBXPLTSI'],
    'P_DIQ': ['SEQN', 'DIQ010'],
    'P_LUX': ['SEQN', 'LUAXSTAT', 'LUANMVGP', 'LUXSMED', 'LUXSIQRM', 'LUXCAPM'],
}
FEATURES = ['RIDAGEYR', 'RIAGENDR', 'BMXBMI', 'LBXSASSI', 'LBXSATSI', 'LBXPLTSI']
SCAN_COLUMNS = COLUMNS['P_LUX'][1:]


def read_source(name):
    path = SOURCE / f'{name}.xpt'
    if not path.exists():
        raise FileNotFoundError(f'{path}: run scripts/download_nhanes.py first.')
    frame = pd.read_sas(path, format='xport')[COLUMNS[name]]
    if frame.empty or frame.SEQN.isna().any() or frame.SEQN.duplicated().any():
        raise ValueError(f'{name}: empty source or missing/duplicate SEQN.')
    return frame


def describe(frame):
    return {
        'participants': len(frame),
        'complete_core_predictors': int(frame[FEATURES].notna().all(axis=1).sum()),
        'missing_core_predictors': {c: int(frame[c].isna().sum()) for c in FEATURES},
        'diabetes_response_counts': frame.diabetes_status.value_counts().to_dict(),
    }


def main():
    merged = read_source('P_DEMO')
    source_counts = {'P_DEMO': len(merged)}
    for name in list(COLUMNS)[1:]:
        frame = read_source(name)
        source_counts[name] = len(frame)
        merged = merged.merge(frame, on='SEQN', how='left', validate='one_to_one')
    merged['SEQN'] = merged.SEQN.astype('int64')
    merged['diabetes_status'] = merged.DIQ010.map(
        {1: 'yes', 2: 'no', 3: 'borderline', 7: 'unknown', 9: 'unknown'}
    ).fillna('unknown')
    adults = merged.loc[merged.RIDAGEYR.ge(18)].copy()
    # CDC's complete-exam flag incorporates fasting and measurement quality.
    # https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_LUX.htm
    usable = adults.loc[adults.LUAXSTAT.eq(1) & adults.LUXSMED.gt(0)].copy()
    diabetic = usable.loc[usable.DIQ010.eq(1)].copy()
    report = {
        'cycle': 'NHANES 2017-March 2020 pre-pandemic',
        'source_rows': source_counts,
        'cohort_flow': {
            'demographic_participants': len(merged),
            'adults_age_18_plus': len(adults),
            'adults_complete_exam_with_stiffness': len(usable),
            'of_these_self_reported_diagnosed_diabetes': len(diabetic),
        },
        'all_adults_with_complete_scan': describe(usable),
        'diabetic_adults_with_complete_scan': describe(diabetic),
        'candidate_core_predictors': FEATURES,
        'excluded_from_predictors': ['SEQN', *SCAN_COLUMNS, 'WTMECPRP', 'SDMVPSU', 'SDMVSTRA'],
        'notes': [
            'Audit counts are unweighted sample counts, not population estimates.',
            'DIQ010=1 means self-reported doctor-diagnosed diabetes; it does not identify diabetes type.',
            'Borderline and unknown diabetes responses are preserved separately.',
            'No imputation, scaling, target threshold or train/test split has been applied.',
            'Choose the intended cohort and scan-derived endpoint before training.',
            'Complete-scan selection can introduce selection bias; this is a retrospective demo.',
            'Kelaniya dataset pending; NHANES does not establish local clinical validity.',
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    adults.to_csv(OUTPUT / 'adult_audit.csv', index=False)
    # Separate pre-scan inputs and outcomes to make leakage boundaries explicit.
    usable[['SEQN', *FEATURES, 'diabetes_status']].to_csv(OUTPUT / 'candidate_predictors.csv', index=False)
    usable[['SEQN', *SCAN_COLUMNS]].to_csv(OUTPUT / 'scan_outcomes.csv', index=False)
    usable[['SEQN', 'WTMECPRP', 'SDMVPSU', 'SDMVSTRA']].to_csv(OUTPUT / 'survey_design.csv', index=False)
    (OUTPUT / 'cohort_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for label, count in report['cohort_flow'].items():
        print(f'{label}: {count:,}')
    print(f"Complete core predictors (all adults): {report['all_adults_with_complete_scan']['complete_core_predictors']:,}")
    print(f"Complete core predictors (diabetic adults): {report['diabetic_adults_with_complete_scan']['complete_core_predictors']:,}")
    print(f'\nSaved audit and separated inputs/outcomes to: {OUTPUT}')
    print('No model trained yet. These are candidate cohorts, not a clinical referral queue.')


if __name__ == '__main__':
    main()
