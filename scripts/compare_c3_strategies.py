"""Compare one scan batch across five independent strategy simulations.

Run after train_c3_baseline.py:
python scripts/compare_c3_strategies.py --budget 20
This exploratory comparison does not establish a statistically reliable winner.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from train_c3_baseline import DATA, make_features, metrics

STRATEGIES = ['Random', 'Risk-first', 'Uncertainty', 'Diversity', 'Hybrid']


def select_batch(strategy, labelled_X, pool_X, probability, budget, seed=42, alpha=0.5):
    """Return pool positions using pre-scan inputs only; never accepts outcomes.

    Hybrid is alpha * uncertainty + (1-alpha) * diversity.
    Diversity is distance to the closest labelled or already-selected participant.
    """
    if strategy not in STRATEGIES:
        raise ValueError('Unknown strategy.')
    if not 1 <= budget <= len(pool_X) or not 0 <= alpha <= 1:
        raise ValueError('Invalid budget or hybrid weight.')
    probability = np.asarray(probability)
    if probability.shape != (len(pool_X),) or not np.isfinite(probability).all():
        raise ValueError('Expected one finite probability per pool participant.')
    uncertainty = 1 - 2 * np.abs(probability - 0.5)
    if strategy == 'Random':
        return np.random.default_rng(seed).permutation(len(pool_X))[:budget]
    if strategy in ['Risk-first', 'Uncertainty']:
        score = probability if strategy == 'Risk-first' else uncertainty
        return np.argsort(-score, kind='stable')[:budget]
    # Fit distance preprocessing only on the initially labelled group.
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    scaler = StandardScaler()
    labelled = scaler.fit_transform(imputer.fit_transform(labelled_X))
    pool = scaler.transform(imputer.transform(pool_X))
    _, distance = pairwise_distances_argmin_min(pool, labelled)
    denominator = max(float(distance.max()), 1e-12)
    selected = []
    available = np.ones(len(pool), dtype=bool)
    for _ in range(budget):
        diversity = distance / denominator
        score = diversity if strategy == 'Diversity' else alpha * uncertainty + (1-alpha) * diversity
        position = int(np.argmax(np.where(available, score, -np.inf)))
        selected.append(position)
        available[position] = False
        distance = np.minimum(distance, np.linalg.norm(pool - pool[position], axis=1))
    return np.asarray(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=int, default=20)
    parser.add_argument('--baseline-dir', type=Path, default=DATA / 'baseline_kpa_8_seed_42_n_400')
    parser.add_argument('--hybrid-alpha', type=float, default=0.5)
    args = parser.parse_args()
    base = args.baseline_dir
    report = json.loads((base / 'baseline_report.json').read_text())
    for name, expected in report['input_sha256'].items():
        if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Input data changed. Rerun baseline training before comparison.')
    assignment = pd.read_csv(base / 'split_assignments.csv')
    if assignment.SEQN.isna().any() or assignment.SEQN.duplicated().any():
        raise ValueError('Missing or duplicate split IDs.')
    # Final test rows are excluded before joining outcomes or making features.
    assignment = assignment.loc[assignment.split.ne('test')]
    predictors = pd.read_csv(DATA / 'candidate_predictors.csv')
    frame = assignment.merge(predictors, on='SEQN', validate='one_to_one').set_index('SEQN')
    if len(frame) != len(assignment):
        raise ValueError('Some split IDs are missing from predictors.')
    X = make_features(frame)
    initial_ids = frame.index[frame.split.eq('initial')]
    pool_ids = frame.index[frame.split.eq('pool')]
    validation_ids = frame.index[frame.split.eq('validation')]
    if not 1 <= args.budget <= len(pool_ids) or not 0 <= args.hybrid_alpha <= 1:
        parser.error('Budget must fit the pool; hybrid-alpha must be between 0 and 1.')
    initial_X, pool_X = X.loc[initial_ids], X.loc[pool_ids]
    baseline = XGBClassifier()
    baseline.load_model(base / 'xgboost_baseline.json')
    probability = baseline.predict_proba(pool_X)[:, 1]
    # Freeze every strategy's choice before loading the outcome oracle.
    choices = {strategy: select_batch(strategy, initial_X, pool_X, probability,
                                      args.budget, report['seed'], args.hybrid_alpha)
               for strategy in STRATEGIES}
    oracle = pd.read_csv(base / 'pool_oracle.csv').set_index('SEQN').target
    outcomes = pd.read_csv(DATA / 'scan_outcomes.csv').set_index('SEQN')
    # Obtain the exact threshold recorded by the training script's endpoint.
    threshold = float(report['endpoint'].split('>= ')[1].split(' kPa')[0])
    expected_oracle = outcomes.loc[pool_ids, 'LUXSMED'].ge(threshold).astype(int)
    if oracle.index.has_duplicates or set(oracle.index) != set(pool_ids):
        raise ValueError('Pool oracle IDs do not match the waiting pool.')
    if not np.array_equal(oracle.loc[pool_ids].to_numpy(), expected_oracle.to_numpy()):
        raise ValueError('Pool oracle outcomes do not match the baseline endpoint.')
    initial_y = outcomes.loc[initial_ids, 'LUXSMED'].ge(threshold).astype(int)
    validation_y = outcomes.loc[validation_ids, 'LUXSMED'].ge(threshold).astype(int)
    before = metrics(validation_y, baseline.predict_proba(X.loc[validation_ids])[:, 1])
    rows, selections = [], []
    for strategy, positions in choices.items():
        ids = pool_ids[positions]
        if len(set(ids)) != args.budget:
            raise AssertionError('Duplicate selections or incorrect budget.')
        revealed = oracle.loc[ids]
        training_X = pd.concat([initial_X, pool_X.loc[ids]])
        training_y = pd.concat([initial_y, revealed])
        # Same fixed hyperparameters as the initial baseline, fit afresh per strategy.
        model = XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.05,
                              min_child_weight=3, reg_lambda=2, subsample=1,
                              colsample_bytree=1, tree_method='hist', n_jobs=1,
                              objective='binary:logistic', eval_metric='logloss', random_state=report['seed'])
        model.fit(training_X, training_y)
        after = metrics(validation_y, model.predict_proba(X.loc[validation_ids])[:, 1])
        rows.append({'strategy': strategy, 'scans': len(ids),
                     'positive_scans_found': int(revealed.sum()),
                     'precision_at_k': float(revealed.mean()),
                     'validation_auroc': after['auroc'],
                     'auroc_change': after['auroc'] - before['auroc'],
                     'validation_average_precision': after['average_precision'],
                     'validation_brier': after['brier_score']})
        for rank, position in enumerate(positions, start=1):
            patient_id = pool_ids[position]
            selections.append({'strategy': strategy, 'rank': rank, 'SEQN': int(patient_id),
                               'pre_scan_score': float(probability[position]),
                               'revealed_target': int(oracle.loc[patient_id])})
    destination = base / f'comparison_budget_{args.budget}_alpha_{args.hybrid_alpha:g}'
    destination.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(destination / 'strategy_comparison.csv', index=False)
    pd.DataFrame(selections).to_csv(destination / 'selected_patients.csv', index=False)
    metadata = {'budget': args.budget, 'seed': report['seed'], 'hybrid_alpha': args.hybrid_alpha,
                'baseline_validation': before, 'final_test_evaluated': False,
                'scope': 'One batch and one split; exploratory, not evidence of a universal winner.',
                'primary_demo_metric': 'positive_scans_found at equal scan budget',
                'secondary_metrics': 'Validation discrimination and Brier score after retraining',
                'fairness_constraint_implemented': False,
                'endpoint': report['endpoint'], 'results': rows}
    (destination / 'comparison_report.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(result.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'\nSaved: {destination}')
    print('One-batch exploratory results. Final test remains unevaluated.')


if __name__ == '__main__':
    main()
