"""Train a reproducible all-adult baseline for the C3 retrospective demo.

python component-3-referral-prioritization/src/train_c3_baseline.py --threshold-kpa 8
No pool outcomes or final test data are used for fitting or validation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/interim/nhanes_2017_2020'
CORE = ['RIDAGEYR', 'RIAGENDR', 'BMXBMI', 'LBXSASSI', 'LBXSATSI', 'LBXPLTSI']
DIABETES = ['diabetes_yes', 'diabetes_borderline', 'diabetes_unknown']
FEATURES = CORE + DIABETES


def make_features(frame):
    """Fixed encoding; learns nothing from validation/pool/test participants."""
    result = frame[CORE].astype(float).copy()
    result['RIAGENDR'] = result.RIAGENDR.map({1: 0, 2: 1})
    for status in ['yes', 'borderline', 'unknown']:
        result[f'diabetes_{status}'] = frame.diabetes_status.eq(status).astype(float)
    return result[FEATURES]


def metrics(y, probability):
    return {
        'auroc': float(roc_auc_score(y, probability)),
        'average_precision': float(average_precision_score(y, probability)),
        'brier_score': float(brier_score_loss(y, probability)),
        'positive_fraction': float(np.mean(y)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threshold-kpa', required=True, type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--initial-labels', type=int, default=400)
    args = parser.parse_args()
    if not np.isfinite(args.threshold_kpa) or args.threshold_kpa <= 0:
        parser.error('threshold-kpa must be finite and positive.')
    predictors = pd.read_csv(DATA / 'candidate_predictors.csv')
    outcomes = pd.read_csv(DATA / 'scan_outcomes.csv')
    for frame in [predictors, outcomes]:
        if frame.SEQN.isna().any() or frame.SEQN.duplicated().any():
            raise ValueError('Missing or duplicate participant IDs.')
    if set(predictors.SEQN) != set(outcomes.SEQN):
        raise ValueError('Predictor and outcome IDs do not match.')
    frame = predictors.merge(outcomes[['SEQN', 'LUXSMED']], on='SEQN', validate='one_to_one')
    frame = frame.sort_values('SEQN').reset_index(drop=True)
    if not frame.RIDAGEYR.ge(18).all() or not frame.LUXSMED.gt(0).all():
        raise ValueError('Expected adults with observed positive stiffness measurements.')
    y = frame.LUXSMED.ge(args.threshold_kpa).astype(int)
    if y.nunique() != 2 or y.value_counts().min() < 10:
        raise ValueError('This threshold leaves too few cases in one class for the demo splits.')
    # Labels are used only by this offline partitioning step for stratification.
    dev, test = train_test_split(frame.index, test_size=0.2, stratify=y, random_state=args.seed)
    available, validation = train_test_split(dev, test_size=0.25, stratify=y.loc[dev], random_state=args.seed)
    if not 2 <= args.initial_labels < len(available) - 2:
        parser.error('initial-labels must leave enough participants for both initial and pool sets.')
    initial, pool = train_test_split(available, train_size=args.initial_labels,
                                    stratify=y.loc[available], random_state=args.seed)
    splits = {'initial': initial, 'pool': pool, 'validation': validation, 'test': test}
    id_sets = [set(frame.loc[idx, 'SEQN']) for idx in splits.values()]
    if sum(map(len, id_sets)) != len(set.union(*id_sets)) or sum(map(len, id_sets)) != len(frame):
        raise AssertionError('Splits must be disjoint and cover every participant.')
    X = make_features(frame)
    # XGBoost handles missing numeric values natively; no full-dataset imputation.
    model = XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.05,
                          min_child_weight=3, reg_lambda=2, subsample=1,
                          colsample_bytree=1, tree_method='hist', n_jobs=1,
                          objective='binary:logistic', eval_metric='logloss', random_state=args.seed)
    model.fit(X.loc[initial], y.loc[initial])
    probability = model.predict_proba(X.loc[validation])[:, 1]
    baseline_probability = np.full(len(validation), y.loc[initial].mean())
    output = DATA / f'baseline_kpa_{args.threshold_kpa:g}_seed_{args.seed}_n_{args.initial_labels}'
    output.mkdir(parents=True, exist_ok=True)
    model.save_model(output / 'xgboost_baseline.json')
    assignments = frame[['SEQN']].copy()
    assignments['split'] = ''
    for name, idx in splits.items():
        assignments.loc[idx, 'split'] = name
    assignments.to_csv(output / 'split_assignments.csv', index=False)
    # The future selector reads pool_features only; the simulator owns pool_oracle.
    X.loc[pool].assign(SEQN=frame.loc[pool, 'SEQN']).to_csv(output / 'pool_features.csv', index=False)
    pd.DataFrame({'SEQN': frame.loc[pool, 'SEQN'], 'target': y.loc[pool]}).to_csv(output / 'pool_oracle.csv', index=False)
    report = {
        'cohort': 'Adults age >=18 with complete NHANES scan and observed stiffness',
        'endpoint': f'Observed liver stiffness >= {args.threshold_kpa:g} kPa',
        'endpoint_status': 'Provisional demonstration endpoint; not confirmed fibrosis or clinical urgency',
        'threshold_reference': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC10735173/',
        'reference_scope': 'NAFLD guidance motivates 8 kPa; does not validate this endpoint for all adults',
        'seed': args.seed, 'features': FEATURES,
        'split_counts': {name: len(idx) for name, idx in splits.items()},
        'initial_positive_count': int(y.loc[initial].sum()),
        'validation_xgboost': metrics(y.loc[validation], probability),
        'validation_constant_baseline': metrics(y.loc[validation], baseline_probability),
        'final_test_evaluated': False,
        'versions': {'xgboost': xgboost.__version__, 'sklearn': sklearn.__version__},
        'input_sha256': {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
                         for name in ['candidate_predictors.csv', 'scan_outcomes.csv']},
        'limitations': ['Single-seed internal validation; no strategy winner established.',
                       'Unweighted retrospective demo; not a population estimate or local clinical validation.',
                       'Final test reserved until strategy and hyperparameter choices are fixed.',
                       'Raw probability scores have not been calibrated.'],
    }
    (output / 'baseline_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('C3 baseline: all adults; provisional elevated-stiffness endpoint')
    for name, idx in splits.items():
        print(f'{name}: {len(idx):,}')
    for name, value in report['validation_xgboost'].items():
        print(f'Validation {name}: {value:.4f}')
    print(f'Saved: {output}')
    print('Final test remains unevaluated. Selection-strategy comparison is the next step.')


if __name__ == '__main__':
    main()
