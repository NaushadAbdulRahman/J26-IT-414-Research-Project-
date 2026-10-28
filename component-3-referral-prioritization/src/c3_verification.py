"""C3 strategy descriptions and output-verification utilities.

This helper supports the C3 Streamlit page. It does not train a model or run a
second research component. The comparison script owns selection and retraining.
"""
import hashlib
import math
import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

STRATEGIES = {
    'Random': ('Pick by chance', 'Selects a seeded random sample as a reference.',
               'Provides a baseline for judging whether another policy helps.'),
    'Risk-first': ('Highest predicted scores', 'Selects the highest XGBoost scores for elevated stiffness.',
                   'Focuses on finding positive cases in the current batch.'),
    'Uncertainty': ('Scores closest to 0.5', 'Prioritizes predictions where the model is least decisive.',
                    'Revealed outcomes may provide useful examples for learning.'),
    'Diversity': ('Different patient profiles',
                  'Selects the profile farthest from its nearest labelled or already selected profile.',
                  'Distances use pre-scan features, with imputation and scaling fitted on labelled patients.'),
    'Hybrid': ('Uncertainty + diversity',
               'Combines 50% uncertainty with 50% normalized diversity.',
               'Balances both signals. Combining them does not guarantee better results.'),
}


def threshold_from_endpoint(endpoint):
    match = re.search(r'>=\s*([0-9]+(?:\.[0-9]+)?)\s*kPa', endpoint)
    if not match or float(match[1]) <= 0:
        raise ValueError('The recorded endpoint must contain a positive threshold in kPa.')
    return float(match[1])


def validate_report(report, baseline, budget):
    """Reject mismatched or incomplete saved results before rendering headlines."""
    if (report['budget'] != budget or report['seed'] != baseline['seed']
            or report['endpoint'] != baseline['endpoint'] or report['hybrid_alpha'] != 0.5):
        raise ValueError('Saved results do not match this budget, seed, endpoint or hybrid weight. Run again.')
    results = pd.DataFrame(report['results'])
    numeric = ['scans', 'positive_scans_found', 'precision_at_k', 'validation_auroc',
               'auroc_change', 'validation_average_precision', 'validation_brier']
    if len(results) != len(STRATEGIES) or set(results.strategy) != set(STRATEGIES):
        raise ValueError('Expected one result for each of the five strategies.')
    if not np.isfinite(results[numeric].to_numpy(dtype=float)).all():
        raise ValueError('Saved comparison contains missing or non-finite metrics.')
    if not results.scans.eq(budget).all():
        raise ValueError('A strategy used a different scan budget. Run again.')
    if not (results.positive_scans_found.between(0, budget).all()
            and results.positive_scans_found.mod(1).eq(0).all()
            and results[['precision_at_k', 'validation_auroc', 'validation_average_precision',
                         'validation_brier']].apply(lambda col: col.between(0, 1)).all().all()):
        raise ValueError('Saved counts or metrics are outside their valid ranges.')
    if not np.allclose(results.precision_at_k, results.positive_scans_found / budget):
        raise ValueError('Saved precision does not match positive scans divided by the budget.')
    before = float(report['baseline_validation']['auroc'])
    if (not math.isclose(before, baseline['validation_xgboost']['auroc'], abs_tol=1e-7)
            or not np.allclose(results.auroc_change, results.validation_auroc - before)):
        raise ValueError('Saved starting AUROC or AUROC changes are inconsistent. Run again.')
    return results


def verify_outputs(base, baseline, report, selected):
    """Recalculate evidence from files. Missing evidence is never counted as passed."""
    checks = []

    def check(name, operation):
        try:
            detail = operation()
            checks.append({'Check': name, 'Status': 'Passed', 'Evidence': detail})
        except FileNotFoundError as exc:
            checks.append({'Check': name, 'Status': 'Not available',
                           'Evidence': f'Missing {exc.filename or exc}. Rerun the comparison to create evidence.'})
        except (AssertionError, ValueError, KeyError, TypeError, OSError) as exc:
            checks.append({'Check': name, 'Status': 'Failed', 'Evidence': str(exc) or 'Output mismatch.'})

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    assignment = pd.read_csv(base / 'split_assignments.csv')
    outcomes = pd.read_csv(base.parent / 'scan_outcomes.csv')
    budget = report['budget']
    threshold = threshold_from_endpoint(report['endpoint'])
    results = pd.DataFrame(report['results']).set_index('strategy')
    pool = set(assignment.loc[assignment.split.eq('pool'), 'SEQN'])
    validation = set(assignment.loc[assignment.split.eq('validation'), 'SEQN'])

    def partitions():
        require(assignment.SEQN.notna().all() and assignment.SEQN.is_unique,
                'Partition IDs are missing or repeated across data groups.')
        require(set(assignment.split) == {'initial', 'pool', 'validation', 'test'},
                'Unexpected partition labels.')
        actual = assignment.groupby('split').size().to_dict()
        require(actual == baseline['split_counts'], 'Partition counts differ from the baseline report.')
        return 'Unique IDs across initial training, waiting pool, validation and reserved test groups.'

    def selections():
        require(set(selected.strategy) == set(STRATEGIES), 'Unexpected or missing selection strategies.')
        for name in STRATEGIES:
            batch = selected.loc[selected.strategy.eq(name)]
            require(len(batch) == budget and batch.SEQN.notna().all() and batch.SEQN.is_unique,
                    f'{name}: incorrect budget, missing IDs or duplicate selections.')
            require(set(batch.SEQN).issubset(pool), f'{name}: selected IDs outside the waiting pool.')
            require(sorted(batch['rank'].tolist()) == list(range(1, budget + 1)),
                    f'{name}: selection order is incomplete or repeated.')
            require(np.isfinite(batch.pre_scan_score).all()
                    and batch.pre_scan_score.between(0, 1).all(), f'{name}: invalid pre-scan scores.')
        return f'All five strategies selected {budget} distinct waiting-pool patients each.'

    def labels_and_precision():
        require(outcomes.SEQN.is_unique, 'Recorded outcome IDs are duplicated.')
        truth = outcomes.set_index('SEQN').LUXSMED
        for name in STRATEGIES:
            batch = selected.loc[selected.strategy.eq(name)]
            actual = truth.loc[batch.SEQN].ge(threshold).astype(int).to_numpy()
            require(batch.revealed_target.isin([0, 1]).all()
                    and np.array_equal(actual, batch.revealed_target.to_numpy()),
                    f'{name}: revealed labels differ from recorded scan outcomes.')
            row = results.loc[name]
            require(len(batch) == budget and row.scans == budget
                    and int(actual.sum()) == row.positive_scans_found
                    and math.isclose(float(actual.mean()), row.precision_at_k, abs_tol=1e-7),
                    f'{name}: positive count or precision differs from selected outcomes.')
        return f'Labels match recorded stiffness ≥{threshold:g} kPa. Counts and precision were recalculated.'

    def inputs():
        hashes = baseline['input_sha256']
        require(set(hashes) == {'candidate_predictors.csv', 'scan_outcomes.csv'},
                'Expected predictor and outcome fingerprints are missing.')
        for name, digest in hashes.items():
            require(hashlib.sha256((base.parent / name).read_bytes()).hexdigest() == digest,
                    f'{name} changed since baseline training. Retrain before comparing.')
        if 'input_sha256' in report:
            require(report['input_sha256'] == hashes, 'Comparison used different baseline inputs.')
        return 'Predictor and outcome SHA-256 fingerprints match the baseline training inputs.'

    def validation_metrics():
        predictions = pd.read_csv(base / f'comparison_budget_{budget}_alpha_0.5'
                                  / 'validation_predictions.csv')
        require(set(predictions.strategy) == {'Initial model', *STRATEGIES},
                'Expected initial and five retrained prediction sets.')
        truth = outcomes.set_index('SEQN').LUXSMED
        before = None
        for name in ['Initial model', *STRATEGIES]:
            batch = predictions.loc[predictions.strategy.eq(name)]
            require(len(batch) == len(validation) and batch.SEQN.is_unique
                    and set(batch.SEQN) == validation, f'{name}: prediction IDs differ from validation IDs.')
            require(batch.target.isin([0, 1]).all() and batch.target.nunique() == 2
                    and np.isfinite(batch.score).all() and batch.score.between(0, 1).all(),
                    f'{name}: invalid validation labels or scores.')
            require(np.array_equal(batch.target.to_numpy(),
                                   truth.loc[batch.SEQN].ge(threshold).astype(int).to_numpy()),
                    f'{name}: validation labels differ from recorded outcomes.')
            auc = roc_auc_score(batch.target, batch.score)
            ap = average_precision_score(batch.target, batch.score)
            brier = brier_score_loss(batch.target, batch.score)
            if name == 'Initial model':
                before = auc
                expected = report['baseline_validation']
                original = baseline['validation_xgboost']
                for key, value in [('auroc', auc), ('average_precision', ap), ('brier_score', brier)]:
                    require(math.isclose(value, expected[key], abs_tol=1e-7)
                            and math.isclose(value, original[key], abs_tol=1e-7),
                            f'Initial model: {key} differs from saved metrics.')
            else:
                row = results.loc[name]
                for key, value in [('validation_auroc', auc), ('validation_average_precision', ap),
                                   ('validation_brier', brier), ('auroc_change', auc - before)]:
                    require(math.isclose(value, row[key], abs_tol=1e-7), f'{name}: {key} mismatch.')
        return f'Recomputed AUROC, average precision and Brier from {len(validation):,} validation patients per model.'

    check('Separate data groups', partitions)
    check('Correct budget and unique selections', selections)
    check('Recorded outcomes and precision', labels_and_precision)
    check('Original input files', inputs)
    check('Validation metrics from saved predictions', validation_metrics)
    return pd.DataFrame(checks)
